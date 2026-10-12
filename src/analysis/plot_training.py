#!/usr/bin/env python3
"""
plot_training.py - plot the RL link-layer training summary.

Reads training_summary_collected.csv (one row per policy) and writes figures to
plot/training/. Run it from the folder that holds the CSV, or pass --csv.

Figures produced:
  1. u_skr_vs_Tc.png        - six-state and BB84 SKR utility vs Tc, L=5 and L=10 (headline)
  2. fidelity_vs_Tc.png     - converged J_F vs Tc, with F_req(4) reference line
  3. avg_length_vs_Tc.png   - mean episode length vs Tc (the pacing that drives u_skr)
  4. avg_purifies_vs_Tc.png - mean purifications per episode vs Tc
  5. above_thresh_vs_Tc.png - fraction of episodes above threshold vs Tc
  6. convergence_vs_Tc.png  - iterations to stop vs Tc, with the hard cap line
  7. dashboard.png          - all six panels in one 2x3 figure

Dependencies: matplotlib only (reads the CSV with the standard-library csv module).
    pip install matplotlib

X-axis is Tc/tau (the Tc multiplier), log scale. It is dimensionless and shared by
both link lengths, so L=5 and L=10 land on the same grid. Under the matched-diagonal
convention the per-tick fidelity decay depends on this multiplier and not on L, so J_F
should coincide for the two lengths; u_skr need not, because it carries an absolute
time that scales with tau=L/c.
"""
from __future__ import annotations
import argparse
import csv
import math
import os

import matplotlib
matplotlib.use("Agg")  # headless: we only save files
import matplotlib.pyplot as plt

# Known thresholds for annotation only (n=4 chain, six-state).
F_REQ_N4 = 0.9476   # per-link equitable budget, x_min^(1/4)
F_MIN = 0.8114      # six-state SKR-zero end-to-end floor

# Fixed styling per link length.
STYLE = {
    5:  {"color": "#1f77b4", "marker": "o", "label": "L=5 km"},
    10: {"color": "#d62728", "marker": "s", "label": "L=10 km"},
}


def to_float(s):
    if s is None:
        return math.nan
    s = str(s).strip()
    if s == "":
        return math.nan
    try:
        return float(s)
    except ValueError:
        return math.nan


def load(csv_path):
    """Return {L: [rows sorted by Tc_mult]} with numeric fields parsed."""
    groups = {}
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            r = {k: to_float(v) for k, v in row.items()
                 if k not in ("name", "source_file", "final_phase")}
            r["name"] = row.get("name", "")
            L = int(r["L"]) if not math.isnan(r["L"]) else None
            if L is None:
                continue
            groups.setdefault(L, []).append(r)
    for L in groups:
        groups[L].sort(key=lambda d: d["Tc_mult"])
    return groups


def series(rows, ycol):
    """(x, y) with NaN y dropped, x = Tc_mult."""
    xs, ys = [], []
    for d in rows:
        y = d.get(ycol, math.nan)
        if not math.isnan(y):
            xs.append(d["Tc_mult"])
            ys.append(y)
    return xs, ys


def line_metric(ax, groups, ycol, ylabel, title, logx=True):
    for L in sorted(groups):
        st = STYLE.get(L, {"color": None, "marker": "o", "label": f"L={L}"})
        x, y = series(groups[L], ycol)
        ax.plot(x, y, marker=st["marker"], color=st["color"],
                label=st["label"], linewidth=1.8, markersize=5)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(r"$T_c/\tau$  (coherence multiplier)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False, fontsize=9)


def save(fig, outdir, fname, dpi):
    path = os.path.join(outdir, fname)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")
    return path


# ---- individual figures ----

def fig_u_skr(groups, outdir, dpi):
    """Six-state (solid) and BB84 (dashed) SKR utility vs Tc."""
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for L in sorted(groups):
        st = STYLE.get(L)
        x6, y6 = series(groups[L], "final_skr_sixstate")
        xb, yb = series(groups[L], "final_skr_bb84")
        ax.plot(x6, y6, marker=st["marker"], color=st["color"], linewidth=1.9,
                markersize=5, label=f"{st['label']} six-state")
        ax.plot(xb, yb, marker=st["marker"], color=st["color"], linewidth=1.5,
                markersize=4, linestyle="--", alpha=0.75,
                label=f"{st['label']} BB84")
    ax.set_xscale("log")
    ax.set_xlabel(r"$T_c/\tau$  (coherence multiplier)")
    ax.set_ylabel(r"link SKR utility  $u_{skr}$  (bits/s)")
    ax.set_title("Link-layer SKR utility vs coherence time")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    return save(fig, outdir, "u_skr_vs_Tc.png", dpi)


def fig_fidelity(groups, outdir, dpi):
    fig, ax = plt.subplots(figsize=(7, 4.6))
    line_metric(ax, groups, "final_J_F", r"converged mean fidelity  $\bar{F}$",
                "Converged link fidelity vs coherence time")
    ax.axhline(F_REQ_N4, color="green", linestyle=":", linewidth=1.4,
               label=f"$F_{{req}}(4)$ = {F_REQ_N4}")
    ax.legend(frameon=False, fontsize=9)
    # focus near the data and the reference line
    ax.set_ylim(F_REQ_N4 - 0.003, 0.9585)
    ax.text(0.99, 0.02,
            f"all policies clear $F_{{req}}$; $F_{{min}}$={F_MIN} lies far below",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            color="0.35")
    return save(fig, outdir, "fidelity_vs_Tc.png", dpi)


def fig_length(groups, outdir, dpi):
    fig, ax = plt.subplots(figsize=(7, 4.6))
    line_metric(ax, groups, "final_avg_length", "mean episode length (ticks)",
                "Episode length vs coherence time")
    return save(fig, outdir, "avg_length_vs_Tc.png", dpi)


def fig_purifies(groups, outdir, dpi):
    fig, ax = plt.subplots(figsize=(7, 4.6))
    line_metric(ax, groups, "final_avg_purifies", "mean purifications / episode",
                "Purification count vs coherence time")
    return save(fig, outdir, "avg_purifies_vs_Tc.png", dpi)


def fig_above_thresh(groups, outdir, dpi):
    fig, ax = plt.subplots(figsize=(7, 4.6))
    line_metric(ax, groups, "final_above_thresh_frac",
                "fraction of episodes above threshold",
                "Above-threshold fraction vs coherence time")
    return save(fig, outdir, "above_thresh_vs_Tc.png", dpi)


def fig_convergence(groups, outdir, dpi, cap):
    fig, ax = plt.subplots(figsize=(7, 4.6))
    line_metric(ax, groups, "n_iters", "iterations to stop",
                "Training length vs coherence time")
    # mark policies that hit the cap
    for L in sorted(groups):
        st = STYLE.get(L)
        xs = [d["Tc_mult"] for d in groups[L] if d.get("hit_cap", 0) == 1]
        ys = [d["n_iters"] for d in groups[L] if d.get("hit_cap", 0) == 1]
        if xs:
            ax.scatter(xs, ys, marker="X", s=90, color=st["color"],
                       edgecolor="black", zorder=5)
    ax.axhline(cap, color="0.4", linestyle="--", linewidth=1.3,
               label=f"hard cap = {cap}")
    ax.legend(frameon=False, fontsize=9)
    ax.text(0.99, 0.02, "X = reached cap (did not early-stop)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            color="0.35")
    return save(fig, outdir, "convergence_vs_Tc.png", dpi)


def fig_dashboard(groups, outdir, dpi, cap):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    (a, b, c), (d, e, f) = axes

    # u_skr (six-state) panel
    for L in sorted(groups):
        st = STYLE.get(L)
        x, y = series(groups[L], "final_skr_sixstate")
        a.plot(x, y, marker=st["marker"], color=st["color"], label=st["label"],
               linewidth=1.8, markersize=5)
    a.set_xscale("log"); a.grid(True, which="both", alpha=0.3)
    a.set_ylabel(r"$u_{skr}$ six-state (bits/s)")
    a.set_title("SKR utility"); a.legend(frameon=False, fontsize=8)

    line_metric(b, groups, "final_J_F", r"$\bar{F}$", "Converged fidelity")
    b.axhline(F_REQ_N4, color="green", linestyle=":", linewidth=1.3)
    b.set_ylim(F_REQ_N4 - 0.003, 0.9585)

    line_metric(c, groups, "final_avg_length", "ticks", "Episode length")
    line_metric(d, groups, "final_avg_purifies", "purifies/ep", "Purification count")
    line_metric(e, groups, "final_above_thresh_frac", "fraction",
                "Above-threshold fraction")

    # convergence panel
    for L in sorted(groups):
        st = STYLE.get(L)
        x, y = series(groups[L], "n_iters")
        f.plot(x, y, marker=st["marker"], color=st["color"], label=st["label"],
               linewidth=1.8, markersize=5)
        xs = [dd["Tc_mult"] for dd in groups[L] if dd.get("hit_cap", 0) == 1]
        ys = [dd["n_iters"] for dd in groups[L] if dd.get("hit_cap", 0) == 1]
        if xs:
            f.scatter(xs, ys, marker="X", s=80, color=st["color"],
                      edgecolor="black", zorder=5)
    f.set_xscale("log"); f.grid(True, which="both", alpha=0.3)
    f.axhline(cap, color="0.4", linestyle="--", linewidth=1.2)
    f.set_ylabel("iterations"); f.set_title("Training length")
    f.legend(frameon=False, fontsize=8)

    for ax in (a, b, c, d, e, f):
        ax.set_xlabel(r"$T_c/\tau$")
    fig.suptitle("RL link-layer training summary (n=4, six-state)", fontsize=13)
    return save(fig, outdir, "dashboard.png", dpi)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="training_summary_collected.csv",
                    help="path to the summary CSV (default: ./training_summary_collected.csv)")
    ap.add_argument("--outdir", default=os.path.join("plot", "training"),
                    help="output directory (default: plot/training)")
    ap.add_argument("--cap", type=int, default=1000,
                    help="training hard cap for the reference line (default: 1000)")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    if not os.path.isfile(args.csv):
        raise SystemExit(f"ERROR: CSV not found: {args.csv}\n"
                         f"Run this from the folder with the CSV, or pass --csv.")
    os.makedirs(args.outdir, exist_ok=True)

    groups = load(args.csv)
    if not groups:
        raise SystemExit("ERROR: no rows parsed from the CSV.")
    npol = sum(len(v) for v in groups.values())
    print(f"Loaded {npol} policies: " +
          ", ".join(f"L={L}:{len(groups[L])}" for L in sorted(groups)))
    print(f"Writing figures to {os.path.abspath(args.outdir)}")

    fig_u_skr(groups, args.outdir, args.dpi)
    fig_fidelity(groups, args.outdir, args.dpi)
    fig_length(groups, args.outdir, args.dpi)
    fig_purifies(groups, args.outdir, args.dpi)
    fig_above_thresh(groups, args.outdir, args.dpi)
    fig_convergence(groups, args.outdir, args.dpi, args.cap)
    fig_dashboard(groups, args.outdir, args.dpi, args.cap)
    print("Done.")


if __name__ == "__main__":
    main()