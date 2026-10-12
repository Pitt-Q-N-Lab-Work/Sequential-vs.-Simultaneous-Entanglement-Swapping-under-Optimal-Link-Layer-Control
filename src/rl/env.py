"""WN2M2 link environment: one link, two memory slots, four actions.

Gymnasium wrapper over the analytic transition handler in transitions.py.
  reset()      -> (observation, info)
  step(action) -> (observation, reward, terminated, truncated, info)
Observation is {"obs": [F1, F2, p, t], "action_mask": 4 ints}.
terminated is set by a successful CONSUME, truncated when MAX_STEPS is hit.
The reward is a vector [R_fid, R_time], not a scalar, so a stock RL library
will need a wrapper around it.

Actions: 0 WAIT, 1 DISCARD, 2 PURIFY, 3 CONSUME.
"""
# Importing libraries
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional


class WN2M2Params:
    """Link parameters. tau is one tick (light travel time over L), p_gen the
    per-tick heralding probability, depol_factor the per-tick fidelity decay.
    Generalist mode samples F0 uniformly from [F0_min, F0_max] every episode."""

    def __init__(
            self,
            F0: float = 0.83,
            L: float = 10.0,
            L_att: float = 22.0,
            K: float = 0.9,
            Tc: float = 0.1,
            c_fiber: float = 200000,
            seed: int = 42,
            tau_override: Optional[float] = None,
            p_gen_override: Optional[float] = None,
            F0_min: Optional[float] = None,
            F0_max: Optional[float] = None,
            ):
        self.F0 = float(F0)
        self.L = float(L)
        self.L_att = float(L_att)
        self.K = float(K)
        self.Tc = float(Tc)
        self.c_fiber = float(c_fiber)
        self.seed = int(seed)
        self.F0_min = float(F0_min) if F0_min is not None else None
        self.F0_max = float(F0_max) if F0_max is not None else None
        self.generalist = (self.F0_min is not None and self.F0_max is not None)

        self.tau = float(tau_override) if tau_override is not None else self.L / self.c_fiber
        self.p_gen = float(p_gen_override) if p_gen_override is not None else float(self.K * np.exp(-self.L / self.L_att))
        self.depol_factor = np.exp(-2.0 * self.tau / self.Tc)

        self.rng = np.random.default_rng(self.seed)

        if self.generalist:
            print(f"[WN2M2Params] Generalist mode: F0 ~ U[{self.F0_min}, {self.F0_max}], "
                  f"L = {self.L} km, tau = {self.tau:.6e} s, p_gen = {self.p_gen:.4f}")
        else:
            print(f"[WN2M2Params] F0 = {self.F0}, L = {self.L} km, tau = {self.tau:.6e} s, "
                  f"p_gen = {self.p_gen:.4f}, Tc = {self.Tc} s, depol_factor = {self.depol_factor:.6f}")


class WN2M2NetworkEnvironment(gym.Env):
    """Gym wrapper. F_req, when given, gates delivery: a CONSUME below it
    ends the episode with zero fidelity reward, so the agent has to purify."""
    metadata = {"render_modes": []}
    MAX_STEPS = 1000

    def __init__(self, params: Optional[WN2M2Params] = None, F_req: Optional[float] = None):
        super().__init__()
        self.params = params if params is not None else WN2M2Params()
        self.F_req = float(F_req) if F_req is not None else None

        if self.F_req is not None:
            print(f"[Env] F_req = {self.F_req:.4f}, agent must purify to deliver above threshold")
            F_ceiling = self._dejmps_werner_ceiling(self.params.F0)

            # WN2M2 has only two memories, so purification rounds cannot be
            # stacked. One DEJMPS round from F0 has to clear F_req.
            if self.F_req > F_ceiling + 1e-6:
                min_F0 = self._min_F0_for_Freq(self.F_req)
                if np.isinf(min_F0):
                    fix_hint = "no F0 <= 1.0 can reach this F_req in one DEJMPS round, reduce F_req"
                else:
                    fix_hint = f"raise F0 to >= {min_F0:.4f}, or reduce F_req"
                raise ValueError(
                    f"[Env] infeasible configuration: F_req = {self.F_req:.4f} exceeds the "
                    f"one-round DEJMPS ceiling {F_ceiling:.4f} reachable from F0 = {self.params.F0:.4f}. "
                    f"{fix_hint}."
                )
            if self.params.F0 >= self.F_req - 1e-6 and not self.params.generalist:
                print(f"[Env] NOTE: F0 = {self.params.F0:.4f} already >= F_req = {self.F_req:.4f}; "
                      f"agent will likely learn consume-as-soon-as-possible (no purification needed).")

        from rl.transitions import QuantumEnvHandler
        self.handler = QuantumEnvHandler(
            F0=self.params.F0,
            p_gen=self.params.p_gen,
            Tc=self.params.Tc,
            tau=self.params.tau,
            L=self.params.L,
            L_att=self.params.L_att,
            K=self.params.K,
            seed=self.params.seed,
        )

        self.action_space = spaces.Discrete(4)
        self.state = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        self.step_count = 0

        self.max_t = float(self.MAX_STEPS * self.params.tau * 100)
        self.observation_space = spaces.Dict({
            "obs": spaces.Box(
                low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0, self.max_t], dtype=np.float32),
            ),
            "action_mask": spaces.MultiBinary(int(self.action_space.n))
        })

        self.eps = 1e-15

    @staticmethod
    def _dejmps_werner_ceiling(F0: float) -> float:
        """Max fidelity one DEJMPS round reaches from two Werner pairs at F0."""
        A = F0
        B = (1.0 - F0) / 3.0
        p_success = (A + B) ** 2 + (2.0 * B) ** 2
        if p_success < 1e-12:
            return F0
        return (A * A + B * B) / p_success

    @staticmethod
    def _min_F0_for_Freq(F_req: float) -> float:
        """Bisect for the smallest F0 whose one-round ceiling clears F_req.
        Assumes the ceiling is monotonic in F0. If that breaks (a non-twirled
        Bell-diagonal ceiling, say) bisection can land on the wrong point, so
        re-derive before reusing this helper."""
        if WN2M2NetworkEnvironment._dejmps_werner_ceiling(1.0) < F_req:
            return float('inf')

        lo, hi = 0.5, 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if WN2M2NetworkEnvironment._dejmps_werner_ceiling(mid) >= F_req:
                hi = mid
            else:
                lo = mid
        return hi

    def _allowed_actions_mask(self, state=None):
        """Which of WAIT, DISCARD, PURIFY, CONSUME are legal in this state.
        Keep this consistent with the branches in the handler."""
        if state is None:
            state = self.state

        F1, F2, p = float(state[0]), float(state[1]), float(state[2])
        has_F1 = F1 > self.eps
        has_F2 = F2 > self.eps
        certain = p >= 1.0 - self.eps
        mask = np.zeros(self.action_space.n, dtype=np.int8)
        if not has_F1 and not has_F2:
            mask[0] = 1
        elif has_F1 and has_F2 and certain:
            mask[1] = mask[2] = mask[3] = 1
        elif (has_F1 != has_F2) and certain:
            mask[0] = mask[1] = mask[3] = 1
        elif (has_F1 != has_F2) and not certain:
            mask[0] = mask[3] = 1
        elif has_F1 and has_F2 and not certain:
            # Both slots full with a purify outcome still pending (p < 1).
            # Only CONSUME is legal: WAIT needs a free slot to generate into,
            # and the transition table has no row for WAIT in this state.
            mask[3] = 1
        else:
            mask[0] = 1
        return mask

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        if seed is not None:
            self.params.rng = np.random.default_rng(seed)

        if self.params.generalist:
            self.params.F0 = float(self.params.rng.uniform(self.params.F0_min, self.params.F0_max))
            self.handler.F0 = self.params.F0

        self.state = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        self.step_count = 0

        if seed is not None:
            handler_seed = int(seed)
        else:
            handler_seed = int(self.params.rng.integers(0, 2 ** 31 - 1))

        self.handler.process_action({
            "action": "reset",
            "seed": handler_seed,
        })

        mask = self._allowed_actions_mask(self.state)
        return {"obs": self.state.copy(), "action_mask": mask}, {}

    def step(self, action: int):
        mask = self._allowed_actions_mask(self.state)
        if mask[action] == 0:
            return (
                {"obs": self.state.copy(), "action_mask": mask},
                np.array([0.0, 0.0], dtype=np.float32),
                False, False, {"error": "invalid_action"}
            )

        F1, F2, p, t = map(float, self.state)
        result = self.handler.process_action({"action": int(action), "F1": F1, "F2": F2, "p": p, "t": t})

        if "error" in result:
            return (
                {"obs": self.state.copy(), "action_mask": mask},
                np.array([0.0, 0.0], dtype=np.float32),
                False, False, result
            )

        self.state = np.array([
            float(result.get("F1", 0.0)),
            float(result.get("F2", 0.0)),
            float(result.get("p", 1.0)),
            float(result.get("t", t))
        ], dtype=np.float32)

        self.step_count += 1

        # R_time is the tick spent on WAIT. R_fid is the delivered fidelity on a
        # successful CONSUME, zero if it lands below F_req.
        reward = np.array([0.0, 0.0], dtype=np.float32)
        done = False

        if action == 0:
            reward[1] = self.params.tau
        elif action == 3:
            if result.get("consume_success", False):
                F_del = float(result.get("consumed_fidelity", 0.0))
                if self.F_req is None or F_del >= self.F_req:
                    reward[0] = F_del
                done = True

        truncated = self.step_count >= self.MAX_STEPS
        new_mask = self._allowed_actions_mask(self.state)
        return {"obs": self.state.copy(), "action_mask": new_mask}, reward, done, truncated, result

    def close(self):
        pass
