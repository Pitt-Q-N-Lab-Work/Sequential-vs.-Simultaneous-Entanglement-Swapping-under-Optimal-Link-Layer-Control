"""Trained-policy side of the eval: the PolicyNetwork (same shape as the
trainer's), the LinkAgent that drives one link's env with a frozen policy,
and the Tc-aware policy bank lookup.

Policy filenames are a lookup contract: policy_six_n{n}_L{L}_Tc{us}us.pt.
Do not rename the files in the bank.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from constants import C_FIBER
from rl.env import WN2M2NetworkEnvironment, WN2M2Params
from controllers.buffers import LinkBuffer

#%% Policy network
class PolicyNetwork(nn.Module):
    """MLP over [F1, F2, p, t]. t is divided by t_scale before layer 1."""
    def __init__(self, state_dim=4, n_actions=4, hidden_dim=64, t_scale=1.0):
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

    def get_action_greedy(self, state, mask):
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0)
            m = torch.FloatTensor(mask).unsqueeze(0)
            probs = self.forward(s, m)
            return int(torch.argmax(probs).item())

#%% Link agent
class LinkAgent:
    """One link: a frozen policy stepping its own env, pushing every
    delivered pair into the link buffer the controller reads."""
    def __init__(self, policy: PolicyNetwork, params: WN2M2Params,
                 buffer: LinkBuffer, seed: int = 42):
        self.policy = policy
        self.buffer = buffer
        self.tau = params.tau
        self.env = WN2M2NetworkEnvironment(params=params, F_req=None)
        self.obs, _ = self.env.reset(seed=seed)
        self._delivery_fidelities = []

    def step(self, t_now: float):
        state = self.obs['obs']
        mask = self.obs['action_mask']
        action = self.policy.get_action_greedy(state, mask)
        self.obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated:
            F_del = float(info.get('consumed_fidelity', 0.0))
            if F_del > 0:
                self.buffer.push(F_del, t_now)
                self._delivery_fidelities.append(F_del)
            self.obs, _ = self.env.reset()
        if truncated:
            self.obs, _ = self.env.reset()

    @property
    def mean_delivery_fidelity(self) -> float:
        if not self._delivery_fidelities:
            return 0.0
        return float(np.mean(self._delivery_fidelities))

    @property
    def n_deliveries(self) -> int:
        return len(self._delivery_fidelities)

#%% Policy loading, Tc-aware lookup
def max_t_for_L(L: float) -> float:
    """The env's upper bound on t, used as the policy's t_scale. Depends on L."""
    tau = L / C_FIBER
    return float(WN2M2NetworkEnvironment.MAX_STEPS * tau * 100)

def load_policy(path: str, t_scale: float) -> PolicyNetwork:
    policy = PolicyNetwork(state_dim=4, n_actions=4, hidden_dim=64, t_scale=t_scale)
    policy.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    policy.eval()
    return policy

def Tc_us_from_seconds(Tc_seconds: float) -> int:
    return int(round(Tc_seconds * 1e6))

def resolve_policy_path(policy_dir: Path, n: int, L: float, Tc_seconds: float) -> Path:
    Tc_us = Tc_us_from_seconds(Tc_seconds)
    return policy_dir / f"policy_six_n{n}_L{int(L)}_Tc{Tc_us}us.pt"

_policy_cache: dict = {}

def get_policy(policy_dir: Path, n: int, L: float, Tc_seconds: float) -> PolicyNetwork:
    Tc_us = Tc_us_from_seconds(Tc_seconds)
    key = (str(policy_dir), n, int(L), Tc_us)
    if key not in _policy_cache:
        ppath = resolve_policy_path(policy_dir, n, L, Tc_seconds)
        if not ppath.exists():
            raise FileNotFoundError(
                f"Policy not found: {ppath}\n"
                f"  Looking for n={n}, L={L} km, Tc={Tc_us} us.\n"
                f"  Train that cell first, or download the bank (see README)."
            )
        _policy_cache[key] = load_policy(str(ppath), t_scale=max_t_for_L(L))
    return _policy_cache[key]
