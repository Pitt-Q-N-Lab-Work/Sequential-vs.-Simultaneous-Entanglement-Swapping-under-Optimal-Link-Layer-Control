"""Rebuild summary.csv from per-config JSONs.

sweep.py in --task_id (SLURM array) mode writes one config_NNN_<label>.json
per config and no summary.csv (that is written in --all and --group mode
only). This walks one or more directories of those JSONs and writes the same
CSV write_summary() would have, so array-mode output feeds the same plots.

FIELDS mirrors write_summary() in simulation/sweep.py. Edit both together.

Usage:
  python src/simulation/aggregate.py runs/eval/sweep_sym_L5 --out runs/eval/summary_sym_L5.csv
  python src/simulation/aggregate.py runs/eval/sweep_sym_L5 runs/eval/sweep_sym_L10 \
         --out runs/eval/summary_all.csv

Rows are sorted by config index. A missing or short JSON is reported and
skipped, so a partial sweep still aggregates what landed.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

FIELDS = [
    'index', 'label', 'group', 'Ls',
    'Tc_int', 'Tc_ext', 'Tc_us', 'L_base',
    'kappa', 'buffer_size',
    # throughput
    'seq_skr_thr', 'seq_ci_thr', 'nest_skr_thr', 'nest_ci_thr',
    'delta_skr_thr', 'delta_pct_thr', 'winner_thr',
    # u_skr_yau (headline)
    'seq_skr_yau', 'seq_ci_yau', 'nest_skr_yau', 'nest_ci_yau',
    'delta_skr_yau', 'delta_pct_yau', 'winner_yau',
    # u_skr_tround (diagnostic)
    'seq_skr_tround', 'seq_ci_tround', 'nest_skr_tround', 'nest_ci_tround',
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


def _row_from_result(r: dict) -> dict:
    cfg = r['config']
    s = r.get('sequential', {})
    ne = r.get('nested', {})
    return {
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
    }


def main():
    ap = argparse.ArgumentParser(description='Aggregate per-config JSONs into summary.csv')
    ap.add_argument('dirs', nargs='+', help='One or more dirs of config_*.json')
    ap.add_argument('--out', required=True, help='Output CSV path')
    args = ap.parse_args()

    rows = []
    n_bad = 0
    for d in args.dirs:
        dpath = Path(d)
        files = sorted(dpath.glob('config_*.json'))
        if not files:
            print(f"WARN: no config_*.json in {dpath}", file=sys.stderr)
        for fp in files:
            try:
                with open(fp) as f:
                    r = json.load(f)
                if 'config' not in r:
                    print(f"WARN: {fp.name} has no 'config' key, skipping", file=sys.stderr)
                    n_bad += 1
                    continue
                rows.append(_row_from_result(r))
            except (json.JSONDecodeError, KeyError) as e:
                print(f"WARN: could not parse {fp.name} ({e}), skipping", file=sys.stderr)
                n_bad += 1

    rows.sort(key=lambda x: int(x['index']))

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows -> {outp}"
          + (f"  ({n_bad} skipped)" if n_bad else ""))


if __name__ == '__main__':
    main()
