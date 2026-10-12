"""WN2M2 transition handler: the analytic state update for each action.

State is (F1, F2, p, t). F1 and F2 are the slot fidelities (0 = empty), p is
the success probability of a pending purification (1 = nothing pending), t is
the elapsed time in seconds. Row numbers in the comments refer to Table 1 of
the WN2M2 transition table in the paper.
"""
# Importing libraries
import numpy as np
import warnings
EPS = 1e-15


class QuantumEnvHandler:
    """Analytic WN2M2 state transitions."""

    def __init__(self,
                 F0: float = 0.9,
                 p_gen: float = None,
                 Tc: float = 0.1,
                 tau: float = 5e-4,
                 L: float = 10.0,
                 L_att: float = 22.0,
                 K: float = 0.9,
                 seed: int = 42,):
        self.F0 = float(F0)
        self.Tc = float(Tc)
        self.tau = float(tau)
        self.L = float(L)
        self.L_att = float(L_att)
        self.K = float(K)
        self.rng = np.random.default_rng(seed)

        eta = np.exp(-self.L / self.L_att)
        self.p_gen = float(p_gen) if p_gen is not None else float(self.K * eta)

        self._depol_factor = np.exp(-2.0 * self.tau / self.Tc)

        self._F1 = 0.0
        self._F2 = 0.0
        self._p = 1.0
        self._t = 0.0

        print(f"[QuantumEnvHandler] F0 = {self.F0}, p_gen = {self.p_gen:.4f}, "
              f"Tc = {self.Tc} s, tau = {self.tau:.6e} s, L = {self.L} km, "
              f"depol_factor = {self._depol_factor:.6f}")

    # Physics
    def depolarize(self, F: float) -> float:
        if F < EPS:
            return 0.0
        return self._depol_factor * F + (1.0 - self._depol_factor) / 4.0

    def dejmps_werner(self, F1: float, F2: float) -> tuple:
        """One DEJMPS round on two Werner pairs: (F_purified, p_success)."""
        A1, B1 = F1, (1.0 - F1) / 3.0
        A2, B2 = F2, (1.0 - F2) / 3.0
        C1, D1 = B1, B1
        C2, D2 = B2, B2

        p_success = (A1 + B1) * (A2 + B2) + (C1 + D1) * (C2 + D2)
        if p_success < EPS:
            return 0.0, 0.0

        F_purified = (A1 * A2 + B1 * B2) / p_success
        return float(F_purified), float(p_success)

    # Dispatch
    def process_action(self, command_data: dict) -> dict:
        action = command_data.get('action')

        if action == 'reset':
            return self.reset(command_data.get('seed'))

        F1 = float(command_data.get('F1', self._F1))
        F2 = float(command_data.get('F2', self._F2))
        p = float(command_data.get('p', self._p))
        t = float(command_data.get('t', self._t))

        has_F1 = F1 > EPS
        has_F2 = F2 > EPS
        is_certain = p > 1.0 - EPS

        if action == 0:
            result = self._do_wait(F1, F2, p, t, has_F1, has_F2, is_certain)
        elif action == 1:
            result = self._do_discard(F1, F2, p, t, has_F1, has_F2)
        elif action == 2:
            result = self._do_purify(F1, F2, p, t, has_F1, has_F2)
        elif action == 3:
            result = self._do_consume(F1, F2, p, t, has_F1, has_F2, is_certain)
        else:
            return {'status': 'error', 'message': f'Unknown action: {action}'}

        self._F1 = result.get('F1', 0.0)
        self._F2 = result.get('F2', 0.0)
        self._p = result.get('p', 1.0)
        self._t = result.get('t', t)
        result['status'] = 'success'
        return result

    # WAIT, Table 1 rows 1-3, 6-7, 13-16
    def _do_wait(self, F1, F2, p, t, has_F1, has_F2, is_certain) -> dict:
        t_new = t + self.tau

        # Both empty: try to generate into each slot.
        if not has_F1 and not has_F2:
            gen1 = self.rng.random() < self.p_gen
            gen2 = self.rng.random() < self.p_gen
            return {
                'F1': self.F0 if gen1 else 0.0,
                'F2': self.F0 if gen2 else 0.0,
                'p': 1.0, 't': t_new,
                'new_entanglements': int(gen1) + int(gen2),
            }

        # One slot full, nothing pending: the stored pair decays, the empty
        # slot tries to generate.
        if is_certain and (has_F1 != has_F2):
            if has_F1:
                gen2 = self.rng.random() < self.p_gen
                return {'F1': self.depolarize(F1), 'F2': self.F0 if gen2 else 0.0,
                        'p': 1.0, 't': t_new,
                        'new_entanglements': int(gen2)}
            else:
                gen1 = self.rng.random() < self.p_gen
                return {'F1': self.F0 if gen1 else 0.0, 'F2': self.depolarize(F2),
                        'p': 1.0, 't': t_new, 'new_entanglements': int(gen1)}

        # One slot full with a purify pending: resolve it this tick. On
        # failure the purified pair is lost and the slot tries to generate.
        if not is_certain and (has_F1 != has_F2):
            purif_succeeded = self.rng.random() < p
            if not purif_succeeded:
                gen = self.rng.random() < self.p_gen
                return {
                    'F1': 0.0 if has_F1 else (self.F0 if gen else 0.0),
                    'F2': 0.0 if has_F2 else (self.F0 if gen else 0.0),
                    'p': 1.0, 't': t_new,
                    'new_entanglements': int(gen),
                    'purification_resolved': 'failed',
                }
            else:
                F_occ = F1 if has_F1 else F2
                gen = self.rng.random() < self.p_gen
                return {
                    'F1': self.depolarize(F_occ) if has_F1 else (self.F0 if gen else 0.0),
                    'F2': self.depolarize(F_occ) if has_F2 else (self.F0 if gen else 0.0),
                    'p': 1.0, 't': t_new,
                    'new_entanglements': int(gen),
                    'purification_resolved': 'succeeded',
                }

        # Both full: the mask should never let WAIT through here.
        warnings.warn('wait action invoked with full memory; check action masking', RuntimeWarning, stacklevel=2)
        return {
            'F1': self.depolarize(F1), 'F2': self.depolarize(F2),
            'p': 1.0, 't': t_new, 'new_entanglements': 0,
        }

    # PURIFY, Table 1 row 10
    def _do_purify(self, F1, F2, p, t, has_F1, has_F2) -> dict:
        if not (has_F1 and has_F2):
            return {
                'F1': F1, 'F2': F2, 'p': p, 't': t, 'warning': 'purify_needs_two_pairs'
            }
        F_purified, p_success = self.dejmps_werner(F1, F2)
        return {
            'F1': F_purified, 'F2': 0.0, 'p': p_success, 't': t,
            'F_purified': F_purified, 'p_success': p_success
        }

    # CONSUME, Table 1 rows 4, 8, 11-12
    def _do_consume(self, F1, F2, p, t, has_F1, has_F2, is_certain) -> dict:
        F_max = max(F1 if has_F1 else 0.0, F2 if has_F2 else 0.0)

        if not has_F1 and not has_F2:
            return {
                'F1': 0.0, 'F2': 0.0, 'p': 1.0, 't': t,
                'consumed_fidelity': 0.0, 'consume_success': False, 'warning': 'consume_empty'
            }

        if is_certain:
            return {
                'F1': 0.0, 'F2': 0.0, 'p': 1.0, 't': t,
                'consumed_fidelity': F_max, 'consume_success': True, 'terminal': True
            }

        # Purify pending: consuming resolves it on the spot.
        if self.rng.random() < p:
            return {'F1': 0.0, 'F2': 0.0, 'p': p, 't': t,
                    'consumed_fidelity': F_max, 'consume_success': True, 'terminal': True
                    }
        return {'F1': 0.0, 'F2': 0.0, 'p': 1.0, 't': t,
                'consumed_fidelity': 0.0, 'consume_success': False, 'terminal': False
                }

    # DISCARD, Table 1 rows 5, 9
    def _do_discard(self, F1, F2, p, t, has_F1, has_F2) -> dict:
        if p < 1.0 - EPS:
            return {
                'F1': F1, 'F2': F2, 'p': p, 't': t, 'warning': 'discard_with_purification_in_progress'
            }
        if has_F1 and has_F2:
            F_max = max(F1, F2)
            return {
                'F1': F_max, 'F2': 0.0, 'p': 1.0, 't': t
            }
        if not has_F1 and not has_F2:
            return {
                'F1': 0.0, 'F2': 0.0, 'p': 1.0, 't': t,
                'warning': 'discard_empty_memory'
            }
        return {'F1': 0.0, 'F2': 0.0, 'p': 1.0, 't': t}

    # Reset
    def reset(self, seed=None) -> dict:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._F1 = 0.0
        self._F2 = 0.0
        self._p = 1.0
        self._t = 0.0
        return {'status': 'reset', 'F1': 0.0, 'F2': 0.0, 'p': 1.0, 't': 0.0}

    def get_fidelities(self):
        return self._F1, self._F2
