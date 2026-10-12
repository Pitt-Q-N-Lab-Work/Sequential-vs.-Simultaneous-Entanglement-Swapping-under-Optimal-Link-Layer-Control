"""Batch runner for the Tc sweep: sequential vs synchronized over a grid of
chain configs, each cell loading the link policy trained for its Tc.

Grid design (kappa = 1 throughout):

  The axis that matters is the dimensionless ratio Tc / tau, so the sweep is
  over the multiplier Tc_mult = Tc / tau rather than raw seconds. L = 5 and
  L = 10 then land on the same dimensionless points.
    Tc_seconds = Tc_mult * L_km / c_fiber,  c_fiber = 200000 km/s

  Tc_ext is a property of the storage hardware, not of fiber length: a memory
  decoheres at its own coherence time whatever link it terminates. It is one
  scalar per config, shared by the whole chain. A swapped composite decoheres
  at that same Tc_ext too, since after the swaps it lives in two end-node
  memories and the intermediate fiber is gone. Length enters only through
  p_gen (loss) and tau (latency).

  Multiplier floor is 5: below that the link agent is outside its training
  regime. Ceiling is 20000, where decoherence is negligible.
    stressed      {5, 10, 25, 50, 100}     collapse and partial recovery
    lower bridge  {250, 500, 1000}         sequential climbing toward parity
    upper bridge  {2500, 5000, 10000}      approach to equivalence
    relaxed       {20000}

  Group 1, symmetric [L, L, L, L]: Tc_int = Tc_ext per link, all links share
  L so all sit at the same Tc / tau. 2 L x 12 Tc_mult = 24 configs.

  Group 2, bottleneck: one 10 km link at position p in an otherwise 5 km
  chain, at a fixed absolute Tc_ext. Not matched: the 5 km links sit at twice
  the Tc / tau of the 10 km link, which is the point. The 10 km link is the
  bottleneck because its longer tau puts it at the worse ratio.
    A mixed chain needs an L = 5 and an L = 10 policy at the same absolute
    Tc_us, so only the intersection of the two per-L Tc_us sets is runnable:
    8 values, not all 12 multipliers.
      {250, 1250, 2500, 12500, 25000, 125000, 250000, 500000}
    4 positions x 8 Tc = 32 configs.

  Total 56 configs x 2 controllers. BUFFER_VALUES below is [20]; add more
  values there for a buffer-size ablation (it multiplies every subtotal).

Terminology: controller strings are 'sequential' and 'nested'. 'nested' is
the synchronized controller (wait for all links, then one swap tree); the
string is kept so existing output files and readers still match.

Output layout:
  <out_dir>/config_NNN_<label>.json   one per config, both controllers
  <out_dir>/summary.csv               written by --all and --group only

Modes:
  --task_id N    one config by index (SLURM array jobs)
  --all          every config, one after another
  --group NAME   every config in one group
  --list         print the grid and exit

Policy bank: 24 files named policy_six_n4_L{L}_Tc{Tc_us}us.pt in --policy_dir.
  L=5    Tc_us in {125, 250, 625, 1250, 2500, 6250, 12500, 25000,
                   62500, 125000, 250000, 500000}
  L=10   Tc_us in {250, 500, 1250, 2500, 5000, 12500, 25000, 50000,
                   125000, 250000, 500000, 1000000}
"""

#%% Importing libraries
import argparse
import csv
import json
import sys
import os
import time
from pathlib import Path

os.environ['PYTHONUNBUFFERED'] = '1'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.engine import run_trials
from physics import F0_for_n

_REPO_ROOT = Path(__file__).resolve().parents[2]

#%% Grid axes
# Per (L, Tc_mult): Tc_seconds = Tc_mult * L_km / 200_000.
# Tc_us is the integer-microsecond name used in policy filenames.
TC_MULTIPLIERS = [5, 10, 25, 50, 100,      # stressed
                  250, 500, 1000,          # lower bridge
                  2500, 5000, 10000,       # upper bridge
                  20000]                   # relaxed
L_BASES = [5, 10]
BUFFER_VALUES = [20]
KAPPA_FIXED = 1.0


def _Tc_seconds_from_mult_L(Tc_mult: int, L_km: int) -> float:
    return Tc_mult * L_km / 200_000.0


def _Tc_us_from_seconds(Tc_s: float) -> int:
    return int(round(Tc_s * 1e6))


def _Tc_seconds_from_us(Tc_us: int) -> float:
    return Tc_us / 1e6


def _bottleneck_tc_us():
    """Absolute Tc_us values at which a mixed [5, 5, 5, 10] chain can run:
    the engine uses one scalar Tc for the chain and looks up both the L = 5
    and the L = 10 policy at that Tc_us, so both files have to exist."""
    us5 = {_Tc_us_from_seconds(_Tc_seconds_from_mult_L(m, 5)) for m in TC_MULTIPLIERS}
    us10 = {_Tc_us_from_seconds(_Tc_seconds_from_mult_L(m, 10)) for m in TC_MULTIPLIERS}
    return sorted(us5 & us10)


BOTTLENECK_TC_US = _bottleneck_tc_us()


#%% Experiment grid
def build_experiment_grid():
    configs = []

    def add(label, Ls, Tc_us, L_base, group, B):
        Tc_seconds = _Tc_seconds_from_us(Tc_us)
        configs.append({
            'label': label,
            'Ls': Ls,
            'Tc_int': Tc_seconds,
            'Tc_ext': Tc_seconds,
            'kappa': KAPPA_FIXED,
            'buffer_size': B,
            'group': group,
            'Tc_us': Tc_us,
            'L_base': L_base,
        })

    # Group 1: symmetric, Tc_int = Tc_ext = Tc_mult * L / c
    for L_base in L_BASES:
        for Tc_mult in TC_MULTIPLIERS:
            Tc_seconds = _Tc_seconds_from_mult_L(Tc_mult, L_base)
            Tc_us = _Tc_us_from_seconds(Tc_seconds)
            Ls = [L_base] * 4
            for B in BUFFER_VALUES:
                label = f'sym_L{L_base}_Tc{Tc_us}us_B{B}'
                add(label, Ls, Tc_us, L_base, group='symmetric_tc', B=B)

    # Group 2: bottleneck, one 10 km link in a 5 km chain, over the Tc_us
    # values where both policies exist
    for Tc_us in BOTTLENECK_TC_US:
        for pos in [1, 2, 3, 4]:
            Ls = [5, 5, 5, 5]
            Ls[pos - 1] = 10
            for B in BUFFER_VALUES:
                label = f'bn_pos{pos}_Tc{Tc_us}us_B{B}'
                add(label, Ls, Tc_us, L_base='mixed', group='bottleneck_tc', B=B)

    for i, cfg in enumerate(configs):
        cfg['index'] = i
    return configs


ALL_CONFIGS = build_experiment_grid()


#%% Delta and winner for one metric
def _compute_delta_winner(seq_val, nest_val):
    """(delta, pct, winner). pct is None when sequential is zero, since a
    percentage of zero is undefined."""
    if seq_val is None or nest_val is None:
        return 0.0, None, 'NONE'

    if seq_val > 0 and nest_val > 0:
        delta = nest_val - seq_val
        pct = delta / seq_val * 100
        winner = 'NESTED' if delta > 0 else ('SEQ' if delta < 0 else 'TIE')
        return delta, pct, winner

    if seq_val == 0 and nest_val == 0:
        return 0.0, 0.0, 'NONE'

    delta = nest_val - seq_val
    winner = 'NESTED' if nest_val > seq_val else 'SEQ'
    return delta, None, winner


#%% Run one config
def run_config(cfg: dict, policy_dir: Path, out_dir: Path,
               n_trials: int, T_sim: float):
    idx = cfg['index']
    label = cfg['label']
    Ls = cfg['Ls']
    Tc_int = cfg['Tc_int']
    Tc_ext = cfg['Tc_ext']
    kappa = cfg.get('kappa', 1.0)
    buffer_size = cfg.get('buffer_size', 20)
    n = len(Ls)
    Tc_us = cfg.get('Tc_us', _Tc_us_from_seconds(Tc_int))
    out_file = out_dir / f"config_{idx:03d}_{label}.json"
    print(f"\n{'=' * 70}")
    print(f"  Config {idx:03d}: {label}")
    print(f"  Links:    {Ls}  (n={n}, F0={F0_for_n(n)})")
    print(f"  Tc_int:   {Tc_int} s  ({Tc_us} us)")
    print(f"  Tc_ext:   {Tc_ext} s  ({_Tc_us_from_seconds(Tc_ext)} us)")
    print(f"  kappa:    {kappa}")
    print(f"  Buffer:   B={buffer_size}")
    print(f"  Group:    {cfg['group']}")
    print(f"  Output:   {out_file}")
    print(f"{'=' * 70}")
    results = {'config': cfg}
    for ctrl in ['sequential', 'nested']:
        t0 = time.time()
        r = run_trials(
            Ls, policy_dir, ctrl,
            Tc_int=Tc_int, Tc_ext=Tc_ext, kappa=kappa, buffer_size=buffer_size,
            n_trials=n_trials, T_sim=T_sim,
        )
        elapsed = time.time() - t0
        r['elapsed_s'] = elapsed
        results[ctrl] = r
        if r['n_trials'] > 0:
            print(f"  {ctrl:<12} u_thr={r['skr_throughput']:>8.2f}+/-{r['ci_throughput']:.2f}  "
                  f"u_yau={r['skr_u_yau']:>8.2f}+/-{r['ci_u_yau']:.2f}  "
                  f"u_tr={r['skr_u_tround']:>8.2f}+/-{r['ci_u_tround']:.2f}  "
                  f"Fe2e={r['mean_fe2e']:.4f}  emit={r['emission_rate']:.1f}/s  "
                  f"({r['n_trials']} trials, {elapsed:.1f}s)")
            print(f"  {' ' * 12} chain_st={r['mean_chain_storage'] * 1e6:>7.2f}us  "
                  f"link_st={r['mean_link_storage'] * 1e6:>7.2f}us  "
                  f"loss_frac={r['mean_fidelity_loss_fraction'] * 100:.3f}%")
        else:
            print(f"  {ctrl:<12} ALL SKIPPED: {r.get('error', '?')}  ({elapsed:.1f}s)")

    s = results.get('sequential', {})
    ne = results.get('nested', {})
    s_thr = s.get('skr_throughput', 0.0)
    n_thr = ne.get('skr_throughput', 0.0)
    s_yau = s.get('skr_u_yau', 0.0)
    n_yau = ne.get('skr_u_yau', 0.0)
    s_tr = s.get('skr_u_tround', 0.0)
    n_tr = ne.get('skr_u_tround', 0.0)
    d_thr, p_thr, w_thr = _compute_delta_winner(s_thr, n_thr)
    d_yau, p_yau, w_yau = _compute_delta_winner(s_yau, n_yau)
    d_tr, p_tr, w_tr = _compute_delta_winner(s_tr, n_tr)
    results['delta_skr_thr'] = d_thr
    results['delta_pct_thr'] = p_thr
    results['winner_thr'] = w_thr
    results['delta_skr_yau'] = d_yau
    results['delta_pct_yau'] = p_yau
    results['winner_yau'] = w_yau
    results['delta_skr_tround'] = d_tr
    results['delta_pct_tround'] = p_tr
    results['winner_tround'] = w_tr
    # Older column names, kept so existing JSON readers still work. The
    # unsuffixed ones are the throughput metric, not the headline.
    results['delta_skr_u'] = d_yau
    results['delta_pct_u'] = p_yau
    results['winner_u'] = w_yau
    results['delta_skr'] = d_thr
    results['delta_pct'] = p_thr
    results['winner'] = w_thr

    p_thr_str = f"{abs(p_thr):.2f}%" if p_thr is not None else "n/a"
    p_yau_str = f"{abs(p_yau):.2f}%" if p_yau is not None else "n/a"
    p_tr_str = f"{abs(p_tr):.2f}%" if p_tr is not None else "n/a"
    print(f"  --> throughput : {w_thr} by {p_thr_str} (delta={d_thr:+.2f} bps)")
    print(f"      u_skr_yau  : {w_yau} by {p_yau_str} (delta={d_yau:+.2f} bps)  [headline]")
    print(f"      u_skr_tr   : {w_tr} by {p_tr_str} (delta={d_tr:+.2f} bps)  [diagnostic]")

    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)

    return results

#%% Summary CSV
def write_summary(all_results: list, out_dir: Path):
    """One row per config. simulation/aggregate.py rebuilds the same CSV from
    the per-config JSONs; keep its FIELDS in step with this list."""
    csv_path = out_dir / 'summary.csv'

    fields = [
        'index', 'label', 'group', 'Ls',
        'Tc_int', 'Tc_ext', 'Tc_us', 'L_base',
        'kappa', 'buffer_size',
        # throughput
        'seq_skr_thr', 'seq_ci_thr',
        'nest_skr_thr', 'nest_ci_thr',
        'delta_skr_thr', 'delta_pct_thr', 'winner_thr',
        # u_skr_yau (headline)
        'seq_skr_yau', 'seq_ci_yau',
        'nest_skr_yau', 'nest_ci_yau',
        'delta_skr_yau', 'delta_pct_yau', 'winner_yau',
        # u_skr_tround (diagnostic)
        'seq_skr_tround', 'seq_ci_tround',
        'nest_skr_tround', 'nest_ci_tround',
        'delta_skr_tround', 'delta_pct_tround', 'winner_tround',
        # fidelity and emission
        'seq_fe2e', 'seq_emit', 'seq_trials',
        'nest_fe2e', 'nest_emit', 'nest_trials',
        # decoherence breakdown
        'seq_chain_storage', 'seq_link_storage', 'seq_loss_frac',
        'nest_chain_storage', 'nest_link_storage', 'nest_loss_frac',
        # time accounting
        'seq_T_last', 'nest_T_last',
    ]

    def _fmt(v, fmt='.4f'):
        if v is None:
            return ''
        try:
            return f"{float(v):{fmt}}"
        except (TypeError, ValueError):
            return ''

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in all_results:
            cfg = r['config']
            s = r.get('sequential', {})
            ne = r.get('nested', {})
            writer.writerow({
                'index':       cfg['index'],
                'label':       cfg['label'],
                'group':       cfg['group'],
                'Ls':          str(cfg['Ls']),
                'Tc_int':      cfg['Tc_int'],
                'Tc_ext':      cfg['Tc_ext'],
                'Tc_us':       cfg.get('Tc_us', ''),
                'L_base':      cfg.get('L_base', ''),
                'kappa':       cfg.get('kappa', 1.0),
                'buffer_size': cfg.get('buffer_size', 20),
                'seq_skr_thr':   _fmt(s.get('skr_throughput', 0)),
                'seq_ci_thr':    _fmt(s.get('ci_throughput', 0)),
                'nest_skr_thr':  _fmt(ne.get('skr_throughput', 0)),
                'nest_ci_thr':   _fmt(ne.get('ci_throughput', 0)),
                'delta_skr_thr': _fmt(r.get('delta_skr_thr')),
                'delta_pct_thr': _fmt(r.get('delta_pct_thr'), '.2f'),
                'winner_thr':    r.get('winner_thr', ''),
                'seq_skr_yau':   _fmt(s.get('skr_u_yau', 0)),
                'seq_ci_yau':    _fmt(s.get('ci_u_yau', 0)),
                'nest_skr_yau':  _fmt(ne.get('skr_u_yau', 0)),
                'nest_ci_yau':   _fmt(ne.get('ci_u_yau', 0)),
                'delta_skr_yau': _fmt(r.get('delta_skr_yau')),
                'delta_pct_yau': _fmt(r.get('delta_pct_yau'), '.2f'),
                'winner_yau':    r.get('winner_yau', ''),
                'seq_skr_tround':   _fmt(s.get('skr_u_tround', 0)),
                'seq_ci_tround':    _fmt(s.get('ci_u_tround', 0)),
                'nest_skr_tround':  _fmt(ne.get('skr_u_tround', 0)),
                'nest_ci_tround':   _fmt(ne.get('ci_u_tround', 0)),
                'delta_skr_tround': _fmt(r.get('delta_skr_tround')),
                'delta_pct_tround': _fmt(r.get('delta_pct_tround'), '.2f'),
                'winner_tround':    r.get('winner_tround', ''),
                'seq_fe2e':    _fmt(s.get('mean_fe2e', 0)),
                'seq_emit':    _fmt(s.get('emission_rate', 0), '.2f'),
                'seq_trials':  s.get('n_trials', 0),
                'nest_fe2e':   _fmt(ne.get('mean_fe2e', 0)),
                'nest_emit':   _fmt(ne.get('emission_rate', 0), '.2f'),
                'nest_trials': ne.get('n_trials', 0),
                'seq_chain_storage':  _fmt(s.get('mean_chain_storage', 0), '.6e'),
                'seq_link_storage':   _fmt(s.get('mean_link_storage', 0), '.6e'),
                'seq_loss_frac':      _fmt(s.get('mean_fidelity_loss_fraction', 0), '.6e'),
                'nest_chain_storage': _fmt(ne.get('mean_chain_storage', 0), '.6e'),
                'nest_link_storage':  _fmt(ne.get('mean_link_storage', 0), '.6e'),
                'nest_loss_frac':     _fmt(ne.get('mean_fidelity_loss_fraction', 0), '.6e'),
                'seq_T_last':  _fmt(s.get('mean_T_last', 0), '.4f'),
                'nest_T_last': _fmt(ne.get('mean_T_last', 0), '.4f'),
            })

        print(f"\nSummary CSV: {csv_path}")

#%% Main
def main():
    parser = argparse.ArgumentParser(
        description='Tc sweep: sequential vs synchronized swapping over the config grid'
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--task_id', type=int, default=None,
                      help='Run one config by index (SLURM array jobs)')
    mode.add_argument('--all', action='store_true',
                      help='Run every config, one after another')
    mode.add_argument('--list', action='store_true',
                      help='Print the grid and exit')
    mode.add_argument('--group', type=str, default=None,
                      help='Run every config in one group')

    parser.add_argument('--policy_dir', type=str,
                        default=str(_REPO_ROOT / 'runs' / 'train' / 'policies'))
    parser.add_argument('--out_dir', type=str,
                        default=str(_REPO_ROOT / 'runs' / 'eval' / 'adhoc'))
    parser.add_argument('--n_trials', type=int, default=200)
    parser.add_argument('--T_sim', type=float, default=5.0,
                        help='Simulated wall clock per trial (seconds)')

    args = parser.parse_args()

    if args.list:
        print(f"Total configs: {len(ALL_CONFIGS)}\n")
        print(f"{'idx':>4} {'label':<40} {'group':<20} {'Ls':<22} "
              f"{'Tc_us':>8} {'L_base':>7} {'B':>4}")
        print("-" * 120)
        for cfg in ALL_CONFIGS:
            print(f"{cfg['index']:>4} {cfg['label']:<40} {cfg['group']:<20} "
                  f"{str(cfg['Ls']):<22} {cfg['Tc_us']:>7}us "
                  f"{str(cfg['L_base']):>7} {cfg.get('buffer_size', 20):>4}")
        return

    policy_dir = Path(args.policy_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.task_id is not None:
        if args.task_id < 0 or args.task_id >= len(ALL_CONFIGS):
            print(f"ERROR: task_id = {args.task_id} out of range [0, {len(ALL_CONFIGS) - 1}]")
            sys.exit(1)
        cfg = ALL_CONFIGS[args.task_id]
        run_config(cfg, policy_dir, out_dir, args.n_trials, args.T_sim)

    elif args.group:
        group_configs = [c for c in ALL_CONFIGS if c['group'] == args.group]
        if not group_configs:
            print(f"ERROR: no configs in group '{args.group}'")
            print(f"Available groups: {sorted(set(c['group'] for c in ALL_CONFIGS))}")
            sys.exit(1)
        print(f"Running group '{args.group}': {len(group_configs)} configs")
        all_results = []
        for cfg in group_configs:
            r = run_config(cfg, policy_dir, out_dir, args.n_trials, args.T_sim)
            all_results.append(r)
        write_summary(all_results, out_dir)

    elif args.all:
        print(f"Running all {len(ALL_CONFIGS)} configs")
        print(f" policy dir: {policy_dir}")
        print(f" out_dir: {out_dir}")
        print(f" n_trials: {args.n_trials}")
        print(f" T_sim: {args.T_sim} s")

        all_results = []
        t0_total = time.time()
        for cfg in ALL_CONFIGS:
            r = run_config(cfg, policy_dir, out_dir, args.n_trials, args.T_sim)
            all_results.append(r)
        elapsed_total = time.time() - t0_total

        print(f"\n{'=' * 70}")
        print(f" All done: {len(ALL_CONFIGS)} configs in {elapsed_total:.0f}s")
        print(f"{'=' * 70}")
        write_summary(all_results, out_dir)

if __name__ == '__main__':
    main()
