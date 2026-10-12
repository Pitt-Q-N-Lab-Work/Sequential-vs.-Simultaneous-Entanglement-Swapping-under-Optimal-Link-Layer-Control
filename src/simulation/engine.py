"""Eval engine: run one chain config under each controller with trained link
agents, aggregate over trials, and a small CLI for one-off runs.

One simulation: n link agents each tick their own env at their own tau and
push delivered pairs into a LinkBuffer. Every tau_min the controller looks at
all the link buffers and emits end-to-end pairs. Emissions are tuples
(F_e2e, T_round, t_emit, chain_storage, link_storage, F_at_oldest_delivery).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import math
import numpy as np
import torch
from scipy import stats as scipy_stats
from constants import (F_MIN, TC_INT, TC_EXT, L_ATT, C_FIBER, K,
                       N_ROUNDS, N_TRIALS, BUFFER_SIZE, KAPPA)
from physics import (_H, F_req_from_n, F0_for_n,
                     skr_rate, skr_u_yau, skr_u_tround)
from controllers.buffers import LinkBuffer
from controllers.sequential import SequentialController
from controllers.synchronized import NestedController
from rl.env import WN2M2Params
from rl.policy import LinkAgent, get_policy

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY_DIR = _REPO_ROOT / 'runs' / 'train' / 'policies'

#%% One simulation
def run_simulation(
        link_Ls,
        policy_dir: Path,
        controller_type: str,
        Tc_int: float,
        Tc_ext: float = TC_EXT,
        kappa: float = KAPPA,
        buffer_size: int = BUFFER_SIZE,
        n_rounds: int = N_ROUNDS,
        T_sim: float = 5.0,
        seed: int = 42
) -> dict:
    n = len(link_Ls)
    F0 = F0_for_n(n)
    rng = np.random.default_rng(seed)
    taus = [L / C_FIBER for L in link_Ls]
    tau_min = min(taus)
    p_gens = [K * np.exp(-L / L_ATT) for L in link_Ls]
    F_req_chain = F_req_from_n(n, Fmin=F_MIN)
    # One DEJMPS round from F0 has to clear F_req, otherwise no policy can deliver.
    A = F0
    B_bell = (1.0 - F0) / 3.0
    p_succ = (A + B_bell) ** 2 + (2.0 * B_bell) ** 2
    F_ceiling = (A * A + B_bell * B_bell) / p_succ if p_succ > 1e-12 else F0
    if F_req_chain > F_ceiling + 1e-6:
        return {
            'error': f'chain infeasible: F_req={F_req_chain:.4f} > ceiling={F_ceiling:.4f} from F0={F0}',
            'n': n, 'link_Ls': link_Ls, 'F0': F0,
            'Tc_int': Tc_int, 'Tc_ext': Tc_ext, 'kappa': kappa, 'buffer_size': buffer_size,
        }
    try:
        policies = {L: get_policy(policy_dir, n, L, Tc_int) for L in set(link_Ls)}
    except FileNotFoundError as e:
        return {'error': str(e), 'n': n, 'link_Ls': link_Ls,
                'Tc_int': Tc_int, 'Tc_ext': Tc_ext, 'kappa': kappa, 'buffer_size': buffer_size}
    # Link buffers gate on the full chain length n, so a pair is only admitted
    # if it can still contribute to a chain above F_MIN.
    link_buffers = [
        LinkBuffer(capacity=buffer_size, n_chain=n, Fmin=F_MIN, Tc_ext=Tc_ext)
        for _ in range(n)
    ]
    agents = []
    for i, L in enumerate(link_Ls):
        params = WN2M2Params(
            F0=F0, L=L, L_att=L_ATT, K=K, Tc=Tc_int,
            seed=int(rng.integers(1e6))
        )
        agents.append(LinkAgent(
            policy=policies[L], params=params,
            buffer=link_buffers[i], seed=int(rng.integers(1e6)),
        ))
    if controller_type == 'sequential':
        controller = SequentialController(n=n, Fmin=F_MIN, Tc_ext=Tc_ext,
                                          buffer_size=buffer_size, kappa=kappa)
    else:
        controller = NestedController(n=n, Fmin=F_MIN, Tc_ext=Tc_ext, kappa=kappa)
    emissions = []
    first_e2e_time = None
    next_step_time = [0.0] * n
    # Occupancy instrumentation, diagnostic only, no effect on dynamics.
    link_occ_sum = 0.0   # running sum of mean pairs per link buffer, one sample per tick
    occ_count = 0
    t = 0.0
    while t < T_sim and len(emissions) < n_rounds:
        for i, agent in enumerate(agents):
            if t >= next_step_time[i] - 1e-15:
                agent.step(t)
                next_step_time[i] += taus[i]
        for lb in link_buffers:
            lb.discard_expired(t)
        # Sample after discard_expired and before controller.step: this is the
        # buffer state the controller sees this tick. Mean over the n buffers.
        link_occ_sum += sum(len(lb) for lb in link_buffers) / n
        occ_count += 1
        controller.step(t, link_buffers)
        for em in controller.collect_results():
            emissions.append(em)
            if first_e2e_time is None:
                first_e2e_time = em[2]
        t += tau_min
    T_sim_actual = t
    mean_link_occupancy = (link_occ_sum / occ_count) if occ_count else 0.0
    link_rejected = [lb.n_rejected for lb in link_buffers]
    link_deliveries = [a.n_deliveries for a in agents]
    link_mean_fdel = [a.mean_delivery_fidelity for a in agents]
    n_swap_rounds = (n - 1) if controller_type == 'sequential' else int(np.ceil(np.log2(n)))
    if not emissions:
        return {
            'n': n, 'link_Ls': link_Ls, 'F0': F0,
            'Tc_int': Tc_int, 'Tc_ext': Tc_ext, 'kappa': kappa, 'buffer_size': buffer_size,
            'controller': controller_type,
            'n_e2e': 0,
            'T_sim': T_sim, 'T_sim_actual': T_sim_actual, 'T_last': 0.0,
            'mean_inter_delivery_yau': float('nan'),
            'skr_throughput': 0.0,
            'skr_u_yau': float('nan'),
            'skr_u_tround': float('nan'),
            'skr_u_based': float('nan'),
            'mean_fe2e': float('nan'), 'std_fe2e': float('nan'),
            'p10_fe2e': float('nan'), 'p50_fe2e': float('nan'), 'p90_fe2e': float('nan'),
            'below_fmin_pct': float('nan'),
            'mean_tround': float('nan'), 'std_tround': float('nan'),
            'mean_chain_storage': float('nan'), 'std_chain_storage': float('nan'),
            'mean_link_storage': float('nan'), 'std_link_storage': float('nan'),
            'mean_total_storage': float('nan'),
            'mean_F_at_delivery': float('nan'),
            'mean_fidelity_loss': float('nan'),
            'mean_fidelity_loss_fraction': float('nan'),
            'n_swap_rounds': n_swap_rounds,
            'emission_rate': 0.0,
            'mean_link_occupancy': mean_link_occupancy,
            'first_e2e_time': None,
            'T_total': T_sim_actual,
            'zero_emit': True,
            'link_rejected': link_rejected,
            'link_deliveries': link_deliveries,
            'link_mean_fdel': link_mean_fdel,
            'link_p_gens': p_gens,
            'link_taus': taus,
            'chain_rejected': [],
        }
    fe2e = np.array([em[0] for em in emissions])
    tround = np.array([em[1] for em in emissions])
    emission_times = np.array([em[2] for em in emissions])
    chain_storage = np.array([em[3] for em in emissions])
    link_storage = np.array([em[4] for em in emissions])
    F_at_delivery = np.array([em[5] for em in emissions])
    T_last = float(emission_times[-1])
    total_storage = chain_storage + link_storage
    safe_F_at_delivery = np.maximum(F_at_delivery, 1e-9)
    fidelity_loss = F_at_delivery - fe2e
    fidelity_loss_fraction = fidelity_loss / safe_F_at_delivery
    chain_rejected = []
    if controller_type == 'sequential':
        chain_rejected = [cb.n_rejected for cb in controller.chain_buffers]
    u_thr = skr_rate(emissions, T_sim_actual)
    u_yau = skr_u_yau(emissions, T_last)
    u_tround = skr_u_tround(emissions)
    return {
        'n': n, 'link_Ls': link_Ls, 'F0': F0,
        'Tc_int': Tc_int, 'Tc_ext': Tc_ext, 'kappa': kappa, 'buffer_size': buffer_size,
        'controller': controller_type,
        'n_e2e': len(emissions),
        'T_sim': T_sim, 'T_sim_actual': T_sim_actual, 'T_last': T_last,
        'mean_inter_delivery_yau': T_last / len(emissions),
        'skr_throughput': u_thr,
        'skr_u_yau': u_yau,
        'skr_u_tround': u_tround,
        'skr_u_based': u_yau,
        'mean_fe2e': float(fe2e.mean()), 'std_fe2e': float(fe2e.std()),
        'p10_fe2e': float(np.percentile(fe2e, 10)),
        'p50_fe2e': float(np.percentile(fe2e, 50)),
        'p90_fe2e': float(np.percentile(fe2e, 90)),
        'below_fmin_pct': float((fe2e < F_MIN).mean() * 100),
        'mean_tround': float(tround.mean()), 'std_tround': float(tround.std()),
        'mean_chain_storage': float(chain_storage.mean()),
        'std_chain_storage': float(chain_storage.std()),
        'mean_link_storage': float(link_storage.mean()),
        'std_link_storage': float(link_storage.std()),
        'mean_total_storage': float(total_storage.mean()),
        'mean_F_at_delivery': float(F_at_delivery.mean()),
        'mean_fidelity_loss': float(fidelity_loss.mean()),
        'mean_fidelity_loss_fraction': float(fidelity_loss_fraction.mean()),
        'n_swap_rounds': n_swap_rounds,
        'emission_rate': len(emissions) / T_sim_actual,
        'mean_link_occupancy': mean_link_occupancy,
        'first_e2e_time': first_e2e_time,
        'T_total': T_sim_actual,
        'zero_emit': False,
        'link_rejected': link_rejected,
        'link_deliveries': link_deliveries,
        'link_mean_fdel': link_mean_fdel,
        'link_p_gens': p_gens,
        'link_taus': taus,
        'chain_rejected': chain_rejected,
    }

#%% Multi-trial aggregation
def run_trials(link_Ls, policy_dir, controller_type, Tc_int,
               Tc_ext=TC_EXT, kappa=KAPPA, buffer_size=BUFFER_SIZE,
               n_trials=N_TRIALS, **kwargs) -> dict:
    """Run n_trials seeds of one config. Rates are pooled across trials
    (fidelity first, entropy once); the CI columns come from per-trial values."""
    trial_skr = []
    trial_emit = []
    trial_occ = []
    trial_u_yau = []
    trial_u_tround = []
    trial_fe2e = []
    trial_chain_storage = []
    trial_link_storage = []
    trial_T_last = []
    trial_loss_fraction = []
    pool_n_emit = 0
    pool_F_sum = 0.0
    pool_T_last = 0.0
    pool_tround = 0.0
    n_zero_emit = 0
    skipped = 0
    last_error = None
    for i in range(n_trials):
        r = run_simulation(
            link_Ls, policy_dir, controller_type,
            Tc_int=Tc_int, Tc_ext=Tc_ext, kappa=kappa, buffer_size=buffer_size,
            seed=42 + i, **kwargs
        )
        if 'error' in r:
            skipped += 1
            last_error = r['error']
            continue
        trial_skr.append(r['skr_throughput'])
        trial_emit.append(r['emission_rate'])
        trial_occ.append(r.get('mean_link_occupancy', 0.0))
        if r.get('zero_emit', False):
            n_zero_emit += 1
            continue
        trial_u_yau.append(r['skr_u_yau'])
        trial_u_tround.append(r['skr_u_tround'])
        trial_fe2e.append(r['mean_fe2e'])
        trial_chain_storage.append(r['mean_chain_storage'])
        trial_link_storage.append(r['mean_link_storage'])
        trial_T_last.append(r['T_last'])
        trial_loss_fraction.append(r['mean_fidelity_loss_fraction'])
        n_i = r['n_e2e']
        F_i = r['mean_fe2e']
        T_last_i = r['T_last']
        tr_i = r['mean_tround']
        pool_n_emit += n_i
        pool_F_sum += n_i * F_i
        pool_T_last += T_last_i
        pool_tround += n_i * tr_i
    if not trial_skr:
        return {
            'skr_throughput': 0.0, 'ci_throughput': 0.0,
            'skr_u_yau': 0.0, 'ci_u_yau': 0.0,
            'skr_u_tround': 0.0, 'ci_u_tround': 0.0,
            'skr_u_based': 0.0, 'ci_u_based': 0.0,
            'mean_fe2e': 0.0, 'ci_fe2e': 0.0,
            'emission_rate': 0.0, 'ci_emission': 0.0,
            'mean_chain_storage': 0.0, 'mean_link_storage': 0.0,
            'mean_link_occupancy': 0.0,
            'mean_T_last': 0.0, 'mean_fidelity_loss_fraction': 0.0,
            'pool_n_emit': 0, 'pool_T_last': 0.0, 'pool_F_pool': float('nan'),
            'n_trials': 0, 'n_emitting_trials': 0, 'n_zero_emit_trials': 0,
            'skipped': skipped, 'error': last_error,
            'link_Ls': link_Ls, 'Tc_int': Tc_int, 'Tc_ext': Tc_ext, 'kappa': kappa,
            'buffer_size': buffer_size,
            'controller': controller_type,
        }

    def mean_ci(xs):
        if not xs:
            return float('nan'), float('nan')
        m = float(np.mean(xs))
        ci = (float(scipy_stats.sem(xs) * scipy_stats.t.ppf(0.975, len(xs) - 1))
              if len(xs) > 1 else 0.0)
        return m, ci

    m_skr, ci_skr = mean_ci(trial_skr)
    m_emit, ci_emit = mean_ci(trial_emit)
    m_fe2e, ci_fe2e = mean_ci(trial_fe2e)
    m_cs, _ = mean_ci(trial_chain_storage)
    m_ls, _ = mean_ci(trial_link_storage)
    m_tl, _ = mean_ci(trial_T_last)
    m_lf, _ = mean_ci(trial_loss_fraction)
    if pool_n_emit > 0 and pool_T_last > 0:
        F_pool = pool_F_sum / pool_n_emit
        mean_T_pool = pool_T_last / pool_n_emit
        mean_tr_pool = pool_tround / pool_n_emit
        m_u_yau_pool = max(0.0, (1.0 - _H(F_pool)) / mean_T_pool) if mean_T_pool > 0 else 0.0
        m_u_tr_pool = max(0.0, (1.0 - _H(F_pool)) / mean_tr_pool) if mean_tr_pool > 0 else 0.0
    else:
        F_pool = float('nan')
        mean_T_pool = float('nan')
        mean_tr_pool = float('nan')
        m_u_yau_pool = 0.0
        m_u_tr_pool = 0.0
    _, ci_u_yau = mean_ci(trial_u_yau)
    _, ci_u_tr = mean_ci(trial_u_tround)
    return {
        'skr_throughput': m_skr, 'ci_throughput': ci_skr,
        'skr_u_yau': m_u_yau_pool, 'ci_u_yau': ci_u_yau,
        'skr_u_tround': m_u_tr_pool, 'ci_u_tround': ci_u_tr,
        'skr_u_based': m_u_yau_pool, 'ci_u_based': ci_u_yau,
        'mean_fe2e': m_fe2e, 'ci_fe2e': ci_fe2e,
        'emission_rate': m_emit, 'ci_emission': ci_emit,
        'mean_link_occupancy': float(np.mean(trial_occ)) if trial_occ else 0.0,
        'mean_chain_storage': m_cs,
        'mean_link_storage': m_ls,
        'mean_T_last': m_tl,
        'mean_fidelity_loss_fraction': m_lf,
        'pool_n_emit': pool_n_emit,
        'pool_T_last': pool_T_last,
        'pool_F_pool': F_pool,
        'pool_mean_T': mean_T_pool,
        'n_trials': len(trial_skr),
        'n_emitting_trials': len(trial_u_yau),
        'n_zero_emit_trials': n_zero_emit,
        'skipped': skipped,
        'link_Ls': link_Ls, 'Tc_int': Tc_int, 'Tc_ext': Tc_ext, 'kappa': kappa,
        'buffer_size': buffer_size,
        'controller': controller_type,
    }

#%% CLI, one config
def main():
    parser = argparse.ArgumentParser(description='Run one chain config under both swap controllers')
    parser.add_argument('--link_Ls', type=float, nargs='+', required=True)
    parser.add_argument('--policy_dir', type=str,
                        default=str(DEFAULT_POLICY_DIR))
    parser.add_argument('--Tc_int', type=float, required=True)
    parser.add_argument('--Tc_ext', type=float, default=None)
    parser.add_argument('--kappa', type=float, default=KAPPA)
    parser.add_argument('--buffer_size', type=int, default=BUFFER_SIZE)
    parser.add_argument('--controller', type=str, default='both',
                        choices=['sequential', 'nested', 'both'])
    parser.add_argument('--n_trials', type=int, default=N_TRIALS)
    parser.add_argument('--n_rounds', type=int, default=N_ROUNDS)
    parser.add_argument('--T_sim', type=float, default=5.0)
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()
    Tc_ext = args.Tc_ext if args.Tc_ext is not None else args.Tc_int
    policy_dir = Path(args.policy_dir)
    controllers = ['sequential', 'nested'] if args.controller == 'both' else [args.controller]
    all_results = {}
    for ctrl in controllers:
        print(f"\n--- {ctrl}: links={args.link_Ls}, Tc_int={args.Tc_int}, "
              f"Tc_ext={Tc_ext}, kappa={args.kappa}, B={args.buffer_size} ---")
        r = run_trials(
            args.link_Ls, policy_dir, ctrl,
            Tc_int=args.Tc_int, Tc_ext=Tc_ext, kappa=args.kappa, buffer_size=args.buffer_size,
            n_trials=args.n_trials, n_rounds=args.n_rounds, T_sim=args.T_sim,
        )
        all_results[ctrl] = r
        if r['n_trials'] > 0:
            print(f"  u_throughput  : {r['skr_throughput']:.4f} +/- {r['ci_throughput']:.4f} bps")
            print(f"  u_skr_yau     : {r['skr_u_yau']:.4f} +/- {r['ci_u_yau']:.4f} bps  [headline]")
            print(f"  u_skr_tround  : {r['skr_u_tround']:.4f} +/- {r['ci_u_tround']:.4f} bps  [diagnostic]")
            print(f"  Fe2e          : {r['mean_fe2e']:.4f} +/- {r['ci_fe2e']:.4f}")
            print(f"  Emit          : {r['emission_rate']:.1f} +/- {r['ci_emission']:.1f} pairs/s")
            print(f"  Trials        : {r['n_trials']} total ({r['n_emitting_trials']} emitting, "
                  f"{r['n_zero_emit_trials']} zero-emit, {r['skipped']} skipped)")
        else:
            print(f"  ALL SKIPPED: {r.get('error')}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Saved: {args.out}")

if __name__ == '__main__':
    main()
