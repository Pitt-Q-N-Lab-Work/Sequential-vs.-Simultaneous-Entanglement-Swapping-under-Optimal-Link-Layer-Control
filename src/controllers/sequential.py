"""Sequential swapping: pipelined left-to-right chain assembly.

As soon as a stored chain of l links and a fresh pair on link l + 1 coexist,
swap them and store the result one tier up. Nothing waits for the whole
chain to be ready at once.
"""
from constants import F_MIN, TC_EXT, BUFFER_SIZE
from physics import decohere, bsm_swap
from controllers.buffers import ChainBuffer

#%% Sequential controller
class SequentialController:
    """Pipelined chain assembly across n - 1 chain buffers."""
    def __init__(self, n: int, Fmin: float = F_MIN, Tc_ext: float = TC_EXT,
                 buffer_size: int = BUFFER_SIZE, kappa: float = 1.0):
        self.n = n
        self.Fmin = Fmin
        self.Tc_ext = Tc_ext
        self.kappa = kappa
        # chain_buffers[i] holds chains of i + 1 links: single links (drained
        # from link buffer 1 at the end of step) land in [0], and the tier-l
        # merge pushes its length-l result into [l - 1].
        self.chain_buffers = [
            ChainBuffer(capacity=buffer_size, n_chain=n, Fmin=Fmin, Tc_ext=Tc_ext,
                        chain_length=i + 1, kappa=kappa)
            for i in range(n - 1)
        ]
        self._e2e_queue: list = []

    def step(self, t: float, link_buffers: list):
        for cb in self.chain_buffers:
            cb.discard_expired(t)
        # Tier l: merge a stored chain of l - 1 links with a pair on link l.
        for ell in range(2, self.n + 1):
            cb_prev = self.chain_buffers[ell - 2]
            lb = link_buffers[ell - 1]
            while not cb_prev.is_empty() and not lb.is_empty():
                chain = cb_prev.pop_freshest()
                if chain is None:
                    break
                F_ch, t_sw, t_old, tc_ch, accum_chain, accum_link, F_at_old = chain
                if (t - t_sw) > tc_ch:
                    continue
                link = lb.pop_freshest()
                if link is None:
                    cb_prev.push(F_ch, t_sw, t_old, accum_chain, accum_link, F_at_old)
                    break
                F_l, t_l, tc_l = link
                if (t - t_l) > tc_l:
                    cb_prev.push(F_ch, t_sw, t_old, accum_chain, accum_link, F_at_old)
                    continue
                chain_dt = t - t_sw
                link_dt = t - t_l
                F_ch_now = decohere(F_ch, chain_dt, cb_prev.Tc_eff)
                F_l_now = decohere(F_l, link_dt, self.Tc_ext)
                F_new = bsm_swap(F_ch_now, F_l_now)
                t_oldest_new = min(t_old, t_l)
                new_accum_chain = accum_chain + chain_dt
                new_accum_link = accum_link + link_dt
                if t_old <= t_l:
                    new_F_at_old = F_at_old
                else:
                    new_F_at_old = F_l
                if ell == self.n:
                    T_round = t - t_oldest_new
                    self._e2e_queue.append((
                        F_new, T_round, t, new_accum_chain,
                        new_accum_link, new_F_at_old,
                    ))
                else:
                    cb_next = self.chain_buffers[ell - 1]
                    cb_next.push(F_new, t, t_oldest_new,
                                 new_accum_chain, new_accum_link, new_F_at_old)
        # Drain link 1 into the tier-1 chain buffer so it can be extended next tick.
        lb1 = link_buffers[0]
        while not lb1.is_empty():
            pair = lb1.pop_freshest()
            if pair is None:
                break
            F1, t1, tc1 = pair
            if (t - t1) > tc1:
                continue
            self.chain_buffers[0].push(F1, t1, t1, 0.0, 0.0, F1)

    def collect_results(self):
        out = list(self._e2e_queue)
        self._e2e_queue.clear()
        return out
