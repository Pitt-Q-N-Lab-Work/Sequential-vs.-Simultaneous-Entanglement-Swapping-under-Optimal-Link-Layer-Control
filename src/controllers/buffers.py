"""Storage layer. LinkBuffer holds delivered link pairs (tier 0). ChainBuffer
holds swapped intermediate chains and is used by the sequential controller
only.

Every entry carries its own t_cut: how long it can sit in memory before it
decays below the fidelity budget for its stage. Entries past t_cut are
dropped on the next discard_expired call.
"""
from __future__ import annotations
from collections import deque
from constants import F_MIN, TC_EXT, BUFFER_SIZE
from physics import compute_tcut_for_pair

#%% Link buffer
class LinkBuffer:
    """Tier-0 storage, always at the base Tc_ext (no kappa upgrade).
    Entries are (F_del, t_del, t_cut)."""
    def __init__(self, capacity: int = BUFFER_SIZE,
                 n_chain: int | None = None, Fmin: float = F_MIN, Tc_ext: float = TC_EXT):
        if n_chain is None:
            # n_chain sets F_req through compute_tcut_for_pair. A default
            # here would gate pairs against the wrong fidelity bar.
            raise ValueError("LinkBuffer needs an explicit n_chain (chain length).")
        self.capacity = capacity
        self.n_chain = n_chain
        self.Fmin = Fmin
        self.Tc_ext = Tc_ext
        self._pairs: deque = deque()
        self._n_rejected = 0

    def push(self, F_del: float, t_del: float):
        t_cut_i = compute_tcut_for_pair(F_del, self.n_chain, self.Fmin, self.Tc_ext)
        if t_cut_i <= 0:
            self._n_rejected += 1
            return
        if len(self._pairs) >= self.capacity:
            self._pairs.popleft()
        self._pairs.append((F_del, t_del, t_cut_i))

    def pop_freshest(self):
        if not self._pairs:
            return None
        return self._pairs.pop()

    def discard_expired(self, t_now: float):
        while self._pairs and (t_now - self._pairs[0][1]) > self._pairs[0][2]:
            self._pairs.popleft()

    def is_empty(self) -> bool:
        return len(self._pairs) == 0

    def __len__(self):
        return len(self._pairs)

    @property
    def n_rejected(self) -> int:
        return self._n_rejected

#%% Chain buffer
class ChainBuffer:
    """Storage for swapped intermediate chains of chain_length links.
    Entries are (F_chain, t_swap, t_oldest, t_cut, accum_chain_storage,
    accum_link_storage, F_at_oldest_delivery)."""
    def __init__(self, capacity: int = BUFFER_SIZE,
                 n_chain: int = 2, Fmin: float = F_MIN, Tc_ext: float = TC_EXT,
                 chain_length: int = 2, kappa: float = 1.0):
        self.capacity = capacity
        self.n_chain = n_chain
        self.Fmin = Fmin
        self.Tc_ext = Tc_ext
        self.chain_length = chain_length
        self.kappa = kappa
        self.Tc_eff = Tc_ext * (kappa ** (chain_length - 1))
        self._chains: deque = deque()
        self._n_rejected = 0

    def push(self, F_chain: float, t_swap: float, t_oldest: float,
             accum_chain_storage: float = 0.0,
             accum_link_storage: float = 0.0,
             F_at_oldest_delivery: float | None = None):
        # A chain covering chain_length of the n_chain links must clear the
        # residual budget F_req(n / l). Merging can only lower fidelity, so
        # anything below this bar can no longer reach F_MIN end to end.
        n_residual = self.n_chain / self.chain_length
        t_cut_i = compute_tcut_for_pair(F_chain, n_residual, self.Fmin, self.Tc_eff)
        if t_cut_i <= 0:
            self._n_rejected += 1
            return
        if len(self._chains) >= self.capacity:
            min_idx = min(range(len(self._chains)), key=lambda i: self._chains[i][2])
            del self._chains[min_idx]
        F_at_old = F_at_oldest_delivery if F_at_oldest_delivery is not None else F_chain
        self._chains.append((F_chain, t_swap, t_oldest, t_cut_i,
                             accum_chain_storage, accum_link_storage, F_at_old))

    def pop_freshest(self):
        if not self._chains:
            return None
        idx = max(range(len(self._chains)), key=lambda i: self._chains[i][2])
        entry = self._chains[idx]
        del self._chains[idx]
        return entry

    def discard_expired(self, t_now: float):
        # t_cut counts down from the swap instant (entry[1]). The decay before
        # the swap is already inside F_chain, so aging from t_oldest (entry[2])
        # would count it twice.
        self._chains = deque(
            entry for entry in self._chains
            if (t_now - entry[1]) <= entry[3]
        )

    def is_empty(self) -> bool:
        return len(self._chains) == 0

    def __len__(self):
        return len(self._chains)

    @property
    def n_rejected(self) -> int:
        return self._n_rejected
