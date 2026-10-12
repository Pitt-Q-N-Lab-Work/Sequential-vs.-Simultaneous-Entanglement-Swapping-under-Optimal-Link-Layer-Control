"""Core physics and the SKR metrics.

Everything is in the Werner picture: a pair with fidelity F has the other
three Bell components sharing (1 - F)/3 each.
"""
import math
import numpy as np
from constants import F_MIN

#%% Core physics
def decohere(F_val: float, dt: float, Tc: float) -> float:
    """Fidelity after dt seconds in a memory with coherence time Tc."""
    return 0.25 + (F_val - 0.25) * math.exp(-2.0 * dt / Tc)

def bsm_swap(F1: float, F2: float) -> float:
    """Fidelity after a Bell-state measurement joins two Werner pairs."""
    return F1 * F2 + (1.0 - F1) * (1.0 - F2) / 3.0

def _H(F_val: float) -> float:
    """Uncertainty in which Bell state you have. At F = 1, H = 0 (the state is
    certainly phi+). 1 - H(F) is the secret key yield per pair under the
    six-state protocol: nearly a full bit when H is small, and negative once
    F drops below F_MIN."""
    if F_val <= 0 or F_val >= 1:
        return 0.0
    return -F_val * math.log2(F_val) - (1.0 - F_val) * math.log2((1.0 - F_val) / 3.0)

def skr_sixstate(Fe2e: float, Tround: float) -> float:
    """Key rate when one pair of fidelity Fe2e lands every Tround seconds."""
    if Tround <= 0:
        return 0.0
    return max(0.0, (1.0 - _H(Fe2e)) / Tround)

def F_req_from_n(n: int, Fmin: float = F_MIN) -> float:
    """Per-link fidelity that n swapped links need so the chain still clears
    Fmin end to end, with the budget split equally across links."""
    return 0.25 + 0.75 * ((4.0 * Fmin - 1.0) / 3.0) ** (1.0 / n)

def F0_for_n(n: int) -> float:
    """Generation fidelity F0: floor(F_req(n), 2 dp) with F_MIN = 0.8114.
    F0 is the raw heralded-pair fidelity, one notch below the per-link
    delivery requirement F_req; the link agent purifies up to F_req at
    runtime. The floor (not round) guarantees F0 never overshoots F_req,
    leaving a learnable purification gap."""
    return math.floor(F_req_from_n(n, F_MIN) * 100) / 100

def compute_tcut_for_pair(F_del: float, n: int, Fmin: float, Tc_ext: float) -> float:
    """Seconds a pair delivered at F_del can sit in memory before it decays
    below F_req(n). Zero means it is already below the bar."""
    F_req_n = F_req_from_n(n, Fmin)
    if F_del <= F_req_n:
        return 0.0
    inner = (F_req_n - 0.25) / (F_del - 0.25)
    if inner <= 0 or inner >= 1:
        return 0.0
    return -(Tc_ext / 2.0) * math.log(inner)

def nested_swap(fidelities: list) -> float:
    """Binary swap tree over a list of link fidelities."""
    m = len(fidelities)
    if m == 1:
        return fidelities[0]
    mid = m // 2
    FL = nested_swap(fidelities[:mid])
    FR = nested_swap(fidelities[mid:])
    return bsm_swap(FL, FR)

#%% Three rate metrics
# Each emission is a tuple (F_e2e, T_round, t_emit, ...) from a controller.
def skr_rate(emissions: list, T_total: float) -> float:
    """u_throughput: sum over pairs of (1 - H(F_i)), divided by total wall
    clock. Rewards volume (secret bits per second of operation). Summing per
    pair over-credits a protocol with a wide fidelity spread (Jensen), and
    T_total includes the idle tail, so this is a literal rate but not a clean
    protocol comparison. Diagnostic."""
    if T_total <= 0 or not emissions:
        return 0.0
    bits = 0.0
    for em in emissions:
        F_e2e = em[0]
        bits += max(0.0, 1.0 - _H(F_e2e))
    return bits / T_total

def skr_u_yau(emissions: list, T_last: float) -> float:
    """u_skr_yau: max(0, 1 - H(<F>)) / <T>. The headline metric.
    The numerator pools fidelity first (mean over all pairs) and takes the
    entropy once, which removes the Jensen artifact. The denominator T_last/N
    is the mean spacing between deliveries, a property of the output stream
    only (how many pairs, when the last one landed), blind to how the chain
    was assembled. Both choices are symmetric across controllers."""
    if not emissions or T_last <= 0:
        return 0.0
    fe2es = [em[0] for em in emissions]
    mean_F = float(np.mean(fe2es))
    mean_T = T_last / len(emissions)
    if mean_T <= 0:
        return 0.0
    return max(0.0, (1.0 - _H(mean_F)) / mean_T)

def skr_u_tround(emissions: list) -> float:
    """u_skr_tround: diagnostic only, asymmetric across controllers.
    Same pooled numerator as u_yau, but the denominator is mean(T_round),
    each pair's own assembly latency (t_emit - t_oldest_link). T_round
    depends on how a pair was assembled, which differs between controllers
    (sequential accumulates pipeline dwell, synchronized is one shot), so this
    metric bakes in the assembly asymmetry."""
    if not emissions:
        return 0.0
    fe2es = [em[0] for em in emissions]
    trounds = [em[1] for em in emissions]
    mean_F = float(np.mean(fe2es))
    mean_T = float(np.mean(trounds))
    if mean_T <= 0:
        return 0.0
    return max(0.0, (1.0 - _H(mean_F)) / mean_T)
