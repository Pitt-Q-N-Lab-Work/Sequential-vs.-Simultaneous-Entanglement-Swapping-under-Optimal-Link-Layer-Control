"""WN2M2 link-layer trainer: REINFORCE on the six-state SKR utility.

Trains one policy per (L, Tc) cell. The objective is the six-state secret key
rate u = (1 - H(J_F)) / J_T, where J_F is the mean delivered fidelity and J_T
the mean episode time over a batch of episodes. The gradient is assembled in
three phases of J_F:

  bootstrap     J_F < 0.25            push J_F up, ignore time (du/dJF has a saddle at 0.25)
  bootstrap_hi  0.25 <= J_F < 0.8114  push J_F up, plus the length penalty below
  skr           J_F >= 0.8114         the analytic partials of u

Length penalty (bootstrap_hi only):
    du_dJT = -len_penalty_coeff * (1 - H(F_req)) / J_T**2
A plain three-phase trainer sets du_dJT = 0 in bootstrap_hi, because the
unclamped SKR gradient has du_dJT > 0 below 0.8114 and would reward longer
episodes. But with du_dJT = 0 nothing penalizes episode length at long Tc
(there is no decoherence left to do it), and the policy drifts into an
over-purification trap near J_F = 0.887 instead of reaching the 0.9575 that
one DEJMPS round gives. The penalty supplies the missing pressure. It is
anchored to F_req, the delivery threshold the env enforces, so it depends only
on F_min and the chain length and assumes nothing about how many purification
rounds the agent runs. It is discontinuous with the skr phase at the boundary
on purpose: the skr-phase du_dJT vanishes at 0.8114, so matching it there
would give no backstop at all.
  coeff = 0.5  validated value, clears the trap at every Tc in the sweep
  coeff = 1.0  over-penalizes at n = 4 and never launches (stalls near 0.51)
  coeff = 0.0  the plain trainer (traps at long Tc)

Early stopping:
--iters is a cap, not a schedule. Training stops once the last conv_patience
iterations are all in the skr phase, J_F stays within conv_jf_tol across that
window, J_F >= F_req throughout, and the least-squares trend of J_F over the
window is below conv_jf_slope_tol. See REINFORCETrainer.__init__.

Usage:
  python src/rl/train.py --F0 0.94 --L 5 --n_links 4 --Tc 0.0125 \
      --len_penalty_coeff 0.5 --name six_n4_L5_Tc12500us --outputs_root runs/train
Bare runs default to runs/train_scratch so they never overwrite the bank.
"""
from __future__ import annotations
import json
import argparse
from pathlib import Path
from datetime import datetime
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rl.env import WN2M2NetworkEnvironment, WN2M2Params

# Six-state protocol, F_min = 0.8114. F_req per link from F_req_from_n:
#   n = 2: F_req = 0.8989 | F0 = 0.89
#   n = 3: F_req = 0.9310 | F0 = 0.92
#   n = 4: F_req = 0.9476 | F0 = 0.94
#   n = 5: F_req = 0.9578 | F0 = 0.95
# F0 is the generation fidelity of a fresh heralded pair and sits below F_req
# at every n, so a fresh pair scores zero and the agent has to purify once
# before it consumes. One DEJMPS round on F0 = 0.94 reaches the Werner
# ceiling 0.9575, which clears F_req(4) = 0.9476. That is the mean delivery
# fidelity seen at n = 4.
F_MIN = 0.8114  # 0.81 would put the end-to-end chain exactly at key zero
SIXSTATE_SKR_ZERO = 0.8114


# Required fidelity per link
def F_req_from_n(n: int, F_min: float = F_MIN) -> float:
    return 0.25 + 0.75 * ((4 * F_min - 1) / 3) ** (1.0 / n)


def _resolve_F_req(args) -> float | None:
    # Both given is a config error: pass one or the other.
    if args.n_links is not None and args.F_req is not None:
        raise ValueError(
            f"[Trainer] Both --n_links ({args.n_links}) and --F_req ({args.F_req}) "
            f"were given. Pass exactly one: --n_links to derive F_req from chain length, "
            f"or --F_req to set it directly."
        )
    if args.n_links is not None:
        F_req = F_req_from_n(args.n_links)
        print(f"[Trainer] n_links = {args.n_links} -> F_req = {F_req:.4f} (F_min = {F_MIN})")
        return F_req
    if args.F_req is not None:
        print(f"[Trainer] Explicit F_req = {args.F_req:.4f} (F_min = {F_MIN})")
        return args.F_req
    return None


# Output and logging
OUTPUTS_ROOT = Path("./outputs")
TRAIN_EVENTS_FILE = OUTPUTS_ROOT / "train_events.jsonl"


def _set_outputs_root(path: str):
    global OUTPUTS_ROOT, TRAIN_EVENTS_FILE
    OUTPUTS_ROOT = Path(path)
    OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
    TRAIN_EVENTS_FILE = OUTPUTS_ROOT / "train_events.jsonl"
    print(f"[Trainer] Output root set to: {OUTPUTS_ROOT.resolve()}")


def _append_event(obj: dict):
    OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
    with open(TRAIN_EVENTS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


class _TeeStream:
    """Write to the terminal and to the run's log file at the same time."""
    def __init__(self, stream, file_handle):
        self._stream = stream
        self._fh = file_handle

    def write(self, data):
        self._stream.write(data)
        self._stream.flush()
        self._fh.write(data)
        self._fh.flush()

    def flush(self):
        self._stream.flush()
        self._fh.flush()

    def isatty(self):
        return getattr(self._stream, "isatty", lambda: False)()


def _setup_train_logging(run_id: str):
    OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUTS_ROOT / f"train_stdout_{run_id}.log"
    orig_out = sys.stdout
    orig_err = sys.stderr
    fh = open(log_path, "a", encoding="utf-8")
    sys.stdout = _TeeStream(orig_out, fh)
    sys.stderr = _TeeStream(orig_err, fh)
    print(f"\n{'=' * 60}")
    print(f"[Trainer] Logging -> {log_path.resolve()}")
    print(f"{'=' * 60}\n")
    return fh, orig_out, orig_err, log_path


def _teardown_train_logging(fh, orig_out, orig_err):
    try:
        sys.stdout = orig_out
        sys.stderr = orig_err
    except Exception:
        pass
    try:
        fh.flush()
        fh.close()
    except Exception:
        pass


# _to_py and _jsonable differ only on ndarray. Collapse them if a non-scalar
# metric is ever added.
def _to_py(x):
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if torch.is_tensor(x):
        return float(x.detach().cpu().item())
    return x


def _jsonable(x):
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.ndarray,)):
        return x.tolist()
    if torch.is_tensor(x):
        return float(x.detach().cpu().item())
    return x


# Policy network
# Yau et al. use a linear softmax over a Fourier basis of the state. This
# trainer uses a two-hidden-layer MLP (4 -> 64 -> 64 -> 4) on the raw state
# [F1, F2, p, t] instead. Same MDP, same utility, different policy class.
class PolicyNetwork(nn.Module):
    """MLP policy over the raw WN2M2 observation [F1, F2, p, t].
    F1, F2, p are already in [0, 1]; t is a physical time in seconds, so it is
    divided by t_scale = env.max_t (MAX_STEPS * tau * 100) before layer 1."""
    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 64, t_scale: float = 1.0):
        super().__init__()
        self.t_scale = float(t_scale)
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_actions)

    def _normalize(self, state_t):
        s = state_t.clone()
        s[..., 3] = s[..., 3] / self.t_scale
        return s

    def forward(self, state, action_mask=None):
        x = F.relu(self.fc1(self._normalize(state)))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, -1e9)
        return F.softmax(logits, dim=-1)

    def get_action(self, state, action_mask):
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            mask_t = torch.FloatTensor(action_mask).unsqueeze(0)
            probs = self.forward(state_t, mask_t)
            action = int(torch.multinomial(probs, 1).item())
            log_prob = torch.log(probs[0, action] + 1e-10).item()
        return action, log_prob


# REINFORCE trainer
class REINFORCETrainer:
    def __init__(self, env, policy, learning_rate=1e-4, gamma=1.0,
                 output_dir="./outputs", run_id="run",
                 len_penalty_coeff=0.5,
                 early_stop=True, conv_patience=70, conv_jf_tol=1e-3,
                 conv_jf_slope_tol=3e-4):
        self.env = env
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = str(run_id)
        self.history = []

        # Early stopping. "Converged" means, over the last conv_patience iters:
        #   (a) every iter is in the skr phase,
        #   (b) J_F spread (max - min) < conv_jf_tol,
        #   (c) J_F >= F_req throughout,
        #   (d) the least-squares trend of J_F rises less than conv_jf_slope_tol.
        # (a) rules out stopping inside the bootstrap_hi trap, which lives below
        # the skr boundary. (c) rejects a flat line that drooped below the
        # delivery gate (it runs to the cap and prints a note instead). (d)
        # rejects a slow monotone climb whose spread happens to fit under
        # conv_jf_tol: the spread gate only sees the window extremes, so a
        # decelerating ramp can pass (b) while the mean is still creeping up
        # (seen at L = 10, mult 500: stopped at J_F = 0.9566 on a rising window,
        # ceiling 0.9572). H is not checked: it is a function of J_F.
        # conv_jf_tol = 1e-3 is about 3x the converged wobble (3e-4).
        # conv_jf_slope_tol = 3e-4 sits above plateau slope noise (1e-4) and
        # below an undertrained ramp (5e-4), so a noisy but flat plateau passes.
        self.early_stop = bool(early_stop)
        self.conv_patience = int(conv_patience)
        self.conv_jf_tol = float(conv_jf_tol)             # J_F spread (wobble)
        self.conv_jf_slope_tol = float(conv_jf_slope_tol)  # J_F drift (trend)
        self._conv_spread = None                           # last-window diagnostics
        self._conv_drift = None

        # Length penalty strength, computed once. update() applies
        # du_dJT = -len_penalty_strength / J_T**2 in bootstrap_hi.
        # Strength = coeff * (1 - H(F_req)), the SKR time sensitivity at the
        # delivery threshold. For n = 4: F_req = 0.9476, H = 0.380,
        # (1 - H) = 0.620, so coeff = 0.5 gives -0.310 / J_T**2.
        self.len_penalty_coeff = float(len_penalty_coeff)
        if env.F_req is None:
            raise ValueError(
                "[Trainer] The length penalty anchors to F_req, but env.F_req is None. "
                "Pass --n_links (or --F_req) so the delivery threshold is defined."
            )
        self.F_anchor = float(env.F_req)
        self.len_penalty_strength = self.len_penalty_coeff * (
            1.0 - self.H_werner(self.F_anchor)
        )

    @staticmethod
    def binary_entropy(p: float) -> float:
        """Binary entropy h(p), used by the BB84 rate."""
        if p < 1e-10 or p > 1 - 1e-10:
            return 0.0
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

    @staticmethod
    def H_werner(F: float) -> float:
        """H(F) = -F log2(F) - (1 - F) log2((1 - F)/3)"""
        if F <= 1e-10 or F >= 1 - 1e-10:
            return 0.0
        q = (1.0 - F) / 3.0
        if q <= 1e-10:
            return 0.0
        return -F * np.log2(F) - 3.0 * q * np.log2(q)

    def compute_sixstate_skr(self, J_F: float, J_T: float) -> float:
        if J_T < 1e-10:
            return 0.0
        H = self.H_werner(J_F)
        return max(0.0, (1.0 - H) / J_T)

    def compute_bb84_skr(self, J_F: float, J_T: float) -> float:
        if J_T < 1e-10:
            return 0.0
        beta = (2.0 / 3.0) * (1.0 - J_F)
        beta = float(np.clip(beta, 1e-10, 1 - 1e-10))
        h_beta = self.binary_entropy(beta)
        return max(0.0, (1.0 - 2.0 * h_beta) / J_T)

    def collect_episode(self):
        obs, _ = self.env.reset()
        states, actions, masks = [], [], []
        rewards_F, rewards_T = [], []
        done = False
        while not done:
            state = obs["obs"]
            mask = obs["action_mask"]
            action, _ = self.policy.get_action(state, mask)
            states.append(state.copy())
            actions.append(action)
            masks.append(mask.copy())
            obs, reward, terminated, truncated, info = self.env.step(action)
            rewards_F.append(float(reward[0]))
            rewards_T.append(float(reward[1]))
            done = terminated or truncated

        L = len(states)
        G_F_total = float(np.sum(rewards_F))
        rewards_T_arr = np.array(rewards_T, dtype=np.float64)
        # Per-step time to go, Yau Eq. 19. Reverse cumsum once, here only.
        G_T_per_step = np.cumsum(rewards_T_arr[::-1])[::-1].tolist()
        return {
            "states": states, "actions": actions, "masks": masks,
            "rewards_F": rewards_F, "rewards_T": rewards_T,
            "G_F": G_F_total,
            "G_T_per_step": G_T_per_step,
            "G_T": float(np.sum(rewards_T)),
            "length": L,
        }

    def update(self, episodes):
        N = len(episodes)
        J_F = float(np.mean([ep["G_F"] for ep in episodes]))
        J_T = float(np.mean([ep["G_T"] for ep in episodes]))
        H_JF = self.H_werner(J_F)
        eps = 1e-6

        # Phase selection, see the file header.
        if J_T < 1e-10 or J_F < 0.25 + eps or J_F > 1 - eps:
            du_dJF = 1.0 / max(J_T, eps)
            du_dJT = 0.0
            phase = "bootstrap"

        elif J_F < SIXSTATE_SKR_ZERO:
            du_dJF = 1.0 / max(J_T, eps)
            # Length backstop: negative, so it penalizes episode length.
            du_dJT = -self.len_penalty_strength / (J_T * J_T)
            phase = "bootstrap_hi"

        else:
            du_dJF = (1.0 / J_T) * np.log2((3.0 * J_F) / (1.0 - J_F))
            du_dJT = -(1.0 - H_JF) / (J_T * J_T)
            phase = "skr"

        ep_lengths = [len(ep["G_T_per_step"]) for ep in episodes]
        max_len = max(ep_lengths) if ep_lengths else 0
        togo = np.zeros((N, max_len), dtype=np.float64)
        alive = np.zeros((N, max_len), dtype=np.float64)
        for i, ep in enumerate(episodes):
            # G_T_per_step is already the time to go. Do not cumsum it again,
            # that double counts and over-weights the early steps.
            g = np.asarray(ep["G_T_per_step"], dtype=np.float64)
            L = len(g)
            if L > 0:
                togo[i, :L] = g
                alive[i, :L] = 1.0

        # Baselines: per-step mean time to go across the batch, and J_F.
        b_T_togo = togo.sum(axis=0) / np.maximum(alive.sum(axis=0), 1.0)
        b_F = J_F
        policy_loss = torch.tensor(0.0, requires_grad=False)
        for i, ep in enumerate(episodes):
            G_F_adv = float(ep["G_F"]) - b_F
            for t_idx, (state, action, mask) in enumerate(
                zip(ep["states"], ep["actions"], ep["masks"])
            ):
                state_t = torch.FloatTensor(state).unsqueeze(0)
                mask_t = torch.FloatTensor(mask).unsqueeze(0)
                probs = self.policy(state_t, mask_t)
                log_p = torch.log(probs[0, action] + 1e-10)
                gamma_t = self.gamma ** t_idx
                G_T_adv = float(togo[i, t_idx]) - float(b_T_togo[t_idx])
                advantage = du_dJF * G_F_adv + du_dJT * G_T_adv
                policy_loss = policy_loss - gamma_t * log_p * (advantage / N)
        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()
        skr_six = self.compute_sixstate_skr(J_F, J_T)
        skr_bb84 = self.compute_bb84_skr(J_F, J_T)
        # Diagnostics, not part of the gradient: purifications per episode and
        # the fraction of episodes that delivered above threshold. At n = 4
        # avg_purifies goes to about 1.0 and F_del to the ceiling of F0.
        avg_purifies = float(np.mean(
            [sum(1 for a in ep["actions"] if a == 2) for ep in episodes]
        ))
        above_thresh_frac = float(np.mean(
            [1.0 if ep["G_F"] > 1e-9 else 0.0 for ep in episodes]
        ))
        return {
            "skr": float(skr_six),
            "skr_sixstate": float(skr_six),
            "skr_bb84": float(skr_bb84),
            "J_F": float(J_F), "J_T": float(J_T),
            "du_dJF": float(du_dJF), "du_dJT": float(du_dJT),
            "H_JF": float(H_JF), "loss": float(policy_loss.item()),
            "avg_length": float(np.mean([ep["length"] for ep in episodes])),
            "avg_purifies": avg_purifies,
            "above_thresh_frac": above_thresh_frac,
            "phase": phase,
        }

    def _save_result(self):
        history_path = self.output_dir / f"training_history_{self.run_id}.json"
        model_path = self.output_dir / f"policy_{self.run_id}.pt"
        cleaned = [{k: _jsonable(v) for k, v in row.items()} for row in self.history]
        with open(history_path, "w") as f:
            json.dump(cleaned, f, indent=2)
        torch.save(self.policy.state_dict(), model_path)
        print(f"[Trainer] Saved history to {history_path.resolve()}")
        print(f"[Trainer] Saved model to {model_path.resolve()}")

    def _converged(self):
        """Convergence test over the last conv_patience iterations: all skr
        phase, J_F above F_req, flat (spread < conv_jf_tol) and stationary
        (trend < conv_jf_slope_tol). See __init__ for why each gate is there."""
        self._conv_spread = None
        self._conv_drift = None
        if not self.early_stop:
            return False
        if len(self.history) < self.conv_patience:
            return False
        window = self.history[-self.conv_patience:]
        if any(m["phase"] != "skr" for m in window):
            return False
        jfs = np.asarray([m["J_F"] for m in window], dtype=np.float64)
        if jfs.min() < self.F_anchor:          # must clear the delivery gate F_req
            return False
        spread = float(jfs.max() - jfs.min())
        # Iterations are evenly spaced, so regress J_F on 0..n-1. Slope times
        # (n - 1) is the total rise the fitted line predicts across the
        # window. abs() so a downward droop is rejected too.
        n = len(jfs)
        slope = float(np.polyfit(np.arange(n, dtype=np.float64), jfs, 1)[0])
        drift = abs(slope) * (n - 1)
        self._conv_spread = spread
        self._conv_drift = drift
        return (spread < self.conv_jf_tol) and (drift < self.conv_jf_slope_tol)

    def train(self, n_iterations=100, episodes_per_iter=100, verbose=True):
        print(f"[Trainer] Starting: {n_iterations} iters x {episodes_per_iter} episodes")
        print(f"[Trainer] F_min = {F_MIN}, tau = {self.env.params.tau:.6e}, "
              f"p_gen = {self.env.params.p_gen:.4f}, F0 = {self.env.params.F0}, "
              f"Tc = {self.env.params.Tc}, F_req = {self.env.F_req}")
        print(f"[Trainer] depol_factor per tick = exp(-2 tau / Tc) = "
              f"{np.exp(-2 * self.env.params.tau / self.env.params.Tc):.6f}")
        print(f"[Trainer] ticks per Tc = {self.env.params.Tc / self.env.params.tau:.2f}")
        print(f"[Trainer] Phase thresholds: "
              f"bootstrap below J_F = {0.25:.2f}, "
              f"bootstrap_hi in [{0.25:.2f}, {SIXSTATE_SKR_ZERO:.4f}], "
              f"skr above J_F = {SIXSTATE_SKR_ZERO:.4f}")
        print(f"[Trainer] Length penalty (bootstrap_hi only): "
              f"anchored to F_req = {self.F_anchor:.4f}, "
              f"H(F_req) = {self.H_werner(self.F_anchor):.4f}, "
              f"coeff = {self.len_penalty_coeff:.3f} -> "
              f"du_dJT = -{self.len_penalty_strength:.4f}/J_T^2")
        if self.early_stop:
            print(f"[Trainer] Early stop: cap = {n_iterations} iters; "
                  f"stop when, over {self.conv_patience} consecutive skr-phase iters, "
                  f"J_F spread < {self.conv_jf_tol} and trend drift < "
                  f"{self.conv_jf_slope_tol} and J_F >= F_req = {self.F_anchor:.4f}.")
        else:
            print(f"[Trainer] Early stop disabled: running all {n_iterations} iters.")
        stopped_early = False
        for iteration in range(1, n_iterations + 1):
            episodes = [self.collect_episode() for _ in range(episodes_per_iter)]
            metrics = self.update(episodes)
            metrics["iteration"] = int(iteration)
            metrics = {k: _to_py(v) for k, v in metrics.items()}
            self.history.append(metrics)
            _append_event({
                "ts": datetime.now().isoformat(timespec="seconds"),
                "service": "train_six_tc_lenpen", "event": "metrics",
                "run_id": self.run_id,
                "Tc": float(self.env.params.Tc),
                "L": float(self.env.params.L),
                "len_penalty_coeff": float(self.len_penalty_coeff),
                **metrics,
            })
            if verbose and iteration % 10 == 0:
                print(
                    f"[Iter {iteration:4d}] SKR = {metrics['skr']:.4f} "
                    f"J_F = {metrics['J_F']:.4f} J_T = {metrics['J_T']:.6f} "
                    f"H = {metrics['H_JF']:.4f} len = {metrics['avg_length']:.1f} "
                    f"purif = {metrics['avg_purifies']:.2f} phase = {metrics['phase']}"
                )
            if self._converged():
                stopped_early = True
                print(
                    f"[Trainer] Converged at iter {iteration}: "
                    f"J_F flat and stationary over {self.conv_patience} skr-phase iters "
                    f"(spread = {self._conv_spread:.2e} < {self.conv_jf_tol}, "
                    f"drift = {self._conv_drift:.2e} < {self.conv_jf_slope_tol}; "
                    f"J_F = {metrics['J_F']:.4f} >= F_req = {self.F_anchor:.4f}, "
                    f"H = {metrics['H_JF']:.4f}, SKR = {metrics['skr']:.2f}). "
                    f"Stopping early (cap was {n_iterations})."
                )
                _append_event({
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "service": "train_six_tc_lenpen", "event": "converged",
                    "run_id": self.run_id,
                    "stop_iteration": int(iteration), "cap": int(n_iterations),
                    "conv_patience": int(self.conv_patience),
                    "conv_jf_tol": float(self.conv_jf_tol),
                    "conv_jf_slope_tol": float(self.conv_jf_slope_tol),
                    "spread": float(self._conv_spread),
                    "drift": float(self._conv_drift),
                    "J_F": float(metrics["J_F"]), "H_JF": float(metrics["H_JF"]),
                    "skr": float(metrics["skr"]),
                })
                break
        if self.early_stop and not stopped_early:
            print(
                f"[Trainer] NOTE: hit the {n_iterations}-iter cap "
                f"without triggering early stop. Final J_F = "
                f"{self.history[-1]['J_F']:.4f}, phase = {self.history[-1]['phase']}. "
                f"If phase != skr or J_F < F_req this cell did not converge. Inspect it."
            )
        self._save_result()
        return self.history


def run_single(args):
    run_id = args.name if args.name else datetime.now().strftime("%Y%m%d_%H%M%S")
    log_fh, orig_out, orig_err, log_path = _setup_train_logging(run_id)
    print("=" * 60)
    print(f"Quantum network RL training (six-state, length-penalty trainer, F_min = {F_MIN})")
    print("=" * 60)
    print(f"  L                 = {args.L} km")
    print(f"  F0                = {args.F0}")
    print(f"  Tc                = {args.Tc} s  ({args.Tc * 1e6:.1f} us)")
    print(f"  n_links           = {args.n_links}")
    print(f"  len_penalty_coeff = {args.len_penalty_coeff}")
    print("=" * 60)
    params = WN2M2Params(
        F0=args.F0, L=args.L, L_att=22.0, K=0.9,
        Tc=args.Tc, seed=42,
        F0_min=args.F0_min, F0_max=args.F0_max,
    )
    # Seed torch (and legacy numpy) so same config + same seed -> same policy.
    # params.rng already seeds the env dynamics; torch.multinomial in
    # get_action draws from torch's global RNG, which is otherwise unseeded.
    torch.manual_seed(params.seed)
    np.random.seed(params.seed)
    F_req = _resolve_F_req(args)
    env = WN2M2NetworkEnvironment(params=params, F_req=F_req)
    t_scale = float(env.max_t)
    policy = PolicyNetwork(state_dim=4, n_actions=4, hidden_dim=64, t_scale=t_scale)
    # Optional warm start. Same L only: t_scale depends on L and the weights
    # live in normalized-t space. The sweep runs train from scratch.
    if getattr(args, "init_policy", None):
        state_dict = torch.load(args.init_policy, map_location="cpu", weights_only=True)
        policy.load_state_dict(state_dict)
        print(f"[Trainer] Warm-started weights from {args.init_policy}")
        print(f"[Trainer]   assumes source was trained at L = {args.L} km "
              f"(t_scale = {t_scale:.6e}); a different-L source would mis-scale t.")
    else:
        print("[Trainer] Fresh random init (no --init_policy).")
    trainer = REINFORCETrainer(env=env, policy=policy, learning_rate=args.lr,
                               gamma=1.0, output_dir=str(OUTPUTS_ROOT), run_id=run_id,
                               len_penalty_coeff=args.len_penalty_coeff,
                               early_stop=args.early_stop,
                               conv_patience=args.conv_patience,
                               conv_jf_tol=args.conv_jf_tol,
                               conv_jf_slope_tol=args.conv_jf_slope_tol)
    print(f"\nConfig: {args.iters} iters x {args.episodes} episodes, lr = {args.lr}")
    print(f"  t_scale = {t_scale:.6e} s  (policy time-input normalizer; "
          f"reported times stay in raw seconds)")
    print("-" * 60)
    try:
        trainer.train(n_iterations=args.iters, episodes_per_iter=args.episodes,
                      verbose=True)
    except KeyboardInterrupt:
        print("\nTraining interrupted")
    finally:
        try:
            env.close()
        except Exception:
            pass
        _teardown_train_logging(log_fh, orig_out, orig_err)
    if trainer.history:
        final = trainer.history[-1]
        print(f"\n{'=' * 60}")
        print(f"  Final SKR:   {final['skr']:.4f}")
        print(f"  Final J_F:   {final['J_F']:.4f}")
        print(f"  Final J_T:   {final['J_T']:.6f}   (raw seconds)")
        print(f"  Final H(F):  {final['H_JF']:.4f}")
        print(f"  Final phase: {final['phase']}")
        print(f"{'=' * 60}")
        return final
    return None


def main():
    parser = argparse.ArgumentParser(
        description="WN2M2 link-layer RL training (six-state SKR, length-penalty trainer)"
    )
    parser.add_argument("--iters", type=int, default=1500,
                        help="Cap on iterations. With early stopping (default on) "
                             "training halts at convergence; the cap only binds for a "
                             "cell that never triggers it. Most cells lock in by 500 to "
                             "700. The low-p_gen L=10 crossover band (mult 250 to 2500) "
                             "climbs slowly: give those 3000.")
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--F0", type=float, default=0.94)
    parser.add_argument("--L", type=float, default=10.0)
    parser.add_argument("--Tc", type=float, default=0.1,
                        help="Memory coherence time in seconds. Default 0.1 (Yau value, "
                             "trivial decoherence). For the Tc sweep pass values like "
                             "0.000125 (5 tau at L=5 km) or 0.005 (100 tau at L=10 km).")
    parser.add_argument("--F0_min", type=float, default=None,
                        help="Generalist mode: sample F0 ~ U[F0_min, F0_max] each episode")
    parser.add_argument("--F0_max", type=float, default=None)
    parser.add_argument("--F_req", type=float, default=None,
                        help="Minimum delivery fidelity for the chain (forces purification)")
    parser.add_argument("--n_links", type=int, default=None,
                        help="Derive F_req from the chain length using F_min = 0.8114")
    parser.add_argument("--len_penalty_coeff", type=float, default=0.5,
                        help="Coefficient on the bootstrap_hi length backstop "
                             "du_dJT = -coeff (1 - H(F_req)) / J_T^2. 0.5 is the validated "
                             "value. 1.0 over-penalizes at n=4 and never launches. 0.0 is "
                             "the plain trainer, which traps at long Tc.")
    parser.add_argument("--early_stop", dest="early_stop", action="store_true",
                        default=True,
                        help="Stop once converged (default on): J_F flat within "
                             "--conv_jf_tol for --conv_patience consecutive skr-phase "
                             "iters and J_F >= F_req.")
    parser.add_argument("--no_early_stop", dest="early_stop", action="store_false",
                        help="Disable early stopping; run the full --iters every time.")
    parser.add_argument("--conv_patience", type=int, default=70,
                        help="Consecutive skr-phase iters J_F must stay flat before "
                             "declaring convergence (default 70).")
    parser.add_argument("--conv_jf_tol", type=float, default=1e-3,
                        help="Max J_F spread (max - min) over the patience window "
                             "(default 1e-3; converged peak to peak is about 3e-4). "
                             "Wobble only; the trend is handled by --conv_jf_slope_tol.")
    parser.add_argument("--conv_jf_slope_tol", type=float, default=3e-4,
                        help="Max least-squares trend of J_F over the patience window, "
                             "as total rise across the window (default 3e-4). Rejects a "
                             "slow climb whose spread fits under --conv_jf_tol. For the "
                             "slow L=10 crossover band tighten to 2e-4 with "
                             "--conv_patience 100.")
    parser.add_argument("--init_policy", type=str, default=None,
                        help="Warm-start weights from an existing .pt. Must be the same L "
                             "as this run. Default None = fresh random init.")
    parser.add_argument("--name", type=str, default=None,
                        help="Run name (default: timestamp). For the sweep the name must "
                             "match the eval filename contract, e.g. six_n4_L5_Tc12500us.")
    parser.add_argument("--outputs_root", type=str,
                        default="./runs/train_scratch",
                        help="Root directory for all outputs. Defaults to a scratch dir so "
                             "manual runs never overwrite the bank in runs/train.")
    args = parser.parse_args()
    _set_outputs_root(args.outputs_root)
    run_single(args)


if __name__ == "__main__":
    main()
