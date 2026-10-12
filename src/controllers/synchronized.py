"""Synchronized swapping: wait until every link has a pair, then collapse the
whole binary swap tree in one tick.

The class is still called NestedController, the name it had while the code
was written. The paper calls the discipline "synchronized swapping".
"""
from constants import F_MIN, TC_EXT
from physics import decohere, nested_swap

#%% Synchronized controller
class NestedController:
    """Wait for a valid pair on every link, then swap all of them at once.
    kappa has no effect here: nothing is ever stored between swaps."""
    def __init__(self, n: int, Fmin: float = F_MIN, Tc_ext: float = TC_EXT,
                 kappa: float = 1.0):
        self.n = n
        self.Fmin = Fmin
        self.Tc_ext = Tc_ext
        self.kappa = kappa
        self._e2e_queue: list = []

    def step(self, t: float, link_buffers: list):
        if any(lb.is_empty() for lb in link_buffers):
            return
        fidelities = []
        t_dels = []
        F_at_deliveries = []
        link_storage_times = []
        for lb in link_buffers:
            pair = None
            while not lb.is_empty():
                candidate = lb.pop_freshest()
                if candidate is None:
                    break
                F_c, t_c, tc_c = candidate
                if (t - t_c) <= tc_c:
                    pair = (F_c, t_c)
                    break
            if pair is None:
                return
            F_l, t_l = pair
            F_dec = decohere(F_l, t - t_l, self.Tc_ext)
            fidelities.append(F_dec)
            t_dels.append(t_l)
            F_at_deliveries.append(F_l)
            link_storage_times.append(t - t_l)
        Fe2e = nested_swap(fidelities)
        t_oldest = min(t_dels)
        T_round = t - t_oldest
        oldest_idx = t_dels.index(t_oldest)
        F_at_oldest = F_at_deliveries[oldest_idx]
        self._e2e_queue.append((
            Fe2e, T_round, t,
            0.0, sum(link_storage_times), F_at_oldest,
        ))

    def collect_results(self):
        out = list(self._e2e_queue)
        self._e2e_queue.clear()
        return out
