#!/usr/bin/env python3
"""
plot_controller.py - plot the controller-layer sweep (Sequential vs Synchronized).

Reads controller_results.csv (one row per config) and writes TWO figure sets:
  plot/controller/         exploratory figures (grids, coverage in titles, dashboard)
  plot/controller/paper/   cleaned paper-ready figures (no grid, large fonts,
                           framed legends, B&W-safe linestyles)

Terminology: the CSV stores the wait-and-swap controller under nested_* columns.
Every plot LABELS it "Synchronized (wait-and-swap)". Never SWAP-ASAP or nested.

Paper figures (default = combined L=5 vs L=10 comparison, once >= 2 lengths have
>= --paper-min cells; otherwise per-length):
  skr_crossover_combined.png   2xN: absolute u_skr (top row) + relative diff (bottom)
  occupancy_combined.png       1xN: mean link occupancy, Sequential vs Synchronized
  fidelity_combined.png        1xN: end-to-end fidelity, zoomed y (F_min -> caption)
  bottleneck_occupancy.png     Sequential fans out by position, Synchronized does not
  bottleneck_skr.png           delivered rate, position-independent for both
Pass --per-length for the older one-figure-per-length layout instead.

Only matplotlib is required.  pip install matplotlib

X-axis is Tc in microseconds, log scale. u_skr is skr_u_yau (pooled-Yau). The
absolute panel keeps the collapsed Sequential zeros; the difference panel plots
only where both controllers emit (a percentage is undefined where Sequential=0).
"""
from __future__ import annotations
import argparse
import csv
import math
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

F_MIN = 0.8114
SEQ_LABEL = "Sequential (swap-and-wait)"
SYN_LABEL = "Synchronized (wait-and-swap)"

# exploratory styling
SEQ = {"color": "#E69F00", "marker": "o", "label": SEQ_LABEL}
SYN = {"color": "#0072B2", "marker": "s", "label": SYN_LABEL}
POS = {
    1: {"color": "#E69F00", "marker": "o", "label": "hop 1 (10,5,5,5)"},
    2: {"color": "#56B4E9", "marker": "s", "label": "hop 2 (5,10,5,5)"},
    3: {"color": "#009E73", "marker": "^", "label": "hop 3 (5,5,10,5)"},
    4: {"color": "#D55E00", "marker": "D", "label": "hop 4 (5,5,5,10)"},
}

# paper styling (generous sizes; normal weight; B&W-safe via linestyle+marker)
PAPER_RC = {
    "font.size": 14, "axes.labelsize": 16, "axes.titlesize": 16,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 12.5,
    "axes.grid": False, "axes.linewidth": 1.1, "lines.linewidth": 2.3,
    "font.family": "serif", "mathtext.fontset": "dejavuserif",
    "figure.titlesize": 17, "savefig.dpi": 300, "figure.dpi": 300,
}
SEQ_P = dict(color="#E69F00", marker="o", linestyle="-", label=SEQ_LABEL, markersize=7)
SYN_P = dict(color="#0072B2", marker="s", linestyle="--", label=SYN_LABEL,
             markersize=7, markerfacecolor="white", markeredgecolor="#0072B2",
             markeredgewidth=1.6)
POS_P = {
    1: dict(color="#E69F00", marker="o", linestyle="-",  label="hop 1 (10,5,5,5)"),
    2: dict(color="#56B4E9", marker="s", linestyle="--", label="hop 2 (5,10,5,5)"),
    3: dict(color="#009E73", marker="^", linestyle="-.", label="hop 3 (5,5,10,5)"),
    4: dict(color="#D55E00", marker="D", linestyle=":",  label="hop 4 (5,5,5,10)"),
}

STR_COLS = {"source_group", "source_file", "label", "group", "Ls",
            "seq_controller", "nested_controller", "seq_link_Ls",
            "nested_link_Ls", "L_base",
            "winner", "winner_yau", "winner_thr", "winner_tround", "winner_u"}


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
    rows = []
    with open(csv_path, newline="") as fh:
        for raw in csv.DictReader(fh):
            rows.append({k: (v if k in STR_COLS else to_float(v))
                         for k, v in raw.items()})
    return rows


def sym_by_L(rows):
    out = {}
    for r in rows:
        if r.get("group") != "symmetric_tc":
            continue
        try:
            L = int(float(r.get("L_base")))
        except (TypeError, ValueError):
            continue
        out.setdefault(L, []).append(r)
    for L in out:
        out[L].sort(key=lambda d: d["Tc_us"])
    return out


def bn_by_pos(rows):
    out = {}
    for r in rows:
        if r.get("group") != "bottleneck_tc":
            continue
        m = re.search(r"bn_pos(\d+)", str(r.get("label", "")))
        if not m:
            continue
        out.setdefault(int(m.group(1)), []).append(r)
    for p in out:
        out[p].sort(key=lambda d: d["Tc_us"])
    return out


def xy(rows, col, keep_zero=True):
    xs, ys = [], []
    for d in rows:
        y = d.get(col, math.nan)
        if math.isnan(y):
            continue
        if (not keep_zero) and y == 0:
            continue
        xs.append(d["Tc_us"])
        ys.append(y)
    return xs, ys


def collapse_boundary(rows, col="seq_skr_u_yau"):
    zeros = [d["Tc_us"] for d in rows if d.get(col) == 0]
    pos = [d["Tc_us"] for d in rows if (d.get(col) or 0) > 0]
    if not zeros or not pos:
        return None
    return math.sqrt(max(zeros) * min(pos))


def framed_legend(ax, **kw):
    return ax.legend(frameon=True, framealpha=0.92, edgecolor="0.7",
                     fancybox=False, handlelength=2.4, borderpad=0.6, **kw)


def save(fig, outdir, fname, use_tight=True):
    path = os.path.join(outdir, fname)
    if use_tight:
        fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")
    return path


#%% Exploratory figures
def _seq_syn_panel(ax, rows, seqcol, syncol, ylabel, keep_zero=True):
    x, y = xy(rows, seqcol, keep_zero)
    ax.plot(x, y, color=SEQ["color"], marker=SEQ["marker"], linewidth=1.8,
            markersize=5, label=SEQ["label"])
    x, y = xy(rows, syncol, keep_zero)
    ax.plot(x, y, color=SYN["color"], marker=SYN["marker"], linewidth=1.8,
            markersize=5, label=SYN["label"])
    ax.set_xscale("log")
    ax.set_xlabel(r"$T_c$ ($\mu$s)")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False, fontsize=9)


def explore_symmetric(symrows, seqcol, syncol, ylabel, title, fname, outdir,
                      refline=None, keep_zero=True):
    Ls = [L for L in (5, 10) if L in symrows]
    if not Ls:
        return
    fig, axes = plt.subplots(1, len(Ls), figsize=(6.6 * len(Ls), 4.6),
                             squeeze=False)
    for ax, L in zip(axes[0], Ls):
        _seq_syn_panel(ax, symrows[L], seqcol, syncol, ylabel, keep_zero)
        ax.set_title(f"L = {L} km   ({len(symrows[L])}/12 cells)")
        if refline is not None:
            ax.axhline(refline[0], color="green", linestyle=":", linewidth=1.3)
            ax.text(0.02, 0.03, refline[1], transform=ax.transAxes, fontsize=8,
                    color="0.35", ha="left", va="bottom")
    fig.suptitle(title, fontsize=13)
    save(fig, outdir, fname)


def explore_bottleneck(bnrows, seqcol, syncol, ylabel, title, fname, outdir,
                       keep_zero=True):
    if not bnrows:
        return
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(13.2, 4.8), sharey=True)
    for p in sorted(bnrows):
        st = POS[p]
        x, y = xy(bnrows[p], seqcol, keep_zero)
        axa.plot(x, y, color=st["color"], marker=st["marker"], linewidth=1.7,
                 markersize=5, label=st["label"])
        x, y = xy(bnrows[p], syncol, keep_zero)
        axb.plot(x, y, color=st["color"], marker=st["marker"], linewidth=1.7,
                 markersize=5, label=st["label"])
    for ax, name in ((axa, "Sequential"), (axb, "Synchronized")):
        ax.set_xscale("log"); ax.grid(True, which="both", alpha=0.3)
        ax.set_xlabel(r"$T_c$ ($\mu$s)"); ax.set_title(name)
        ax.legend(frameon=False, fontsize=8)
    axa.set_ylabel(ylabel)
    ncell = sum(len(v) for v in bnrows.values())
    fig.suptitle(f"{title}   ({ncell}/32 cells)", fontsize=13)
    save(fig, outdir, fname)


def explore_dashboard(symrows, bnrows, outdir):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    (a, b, c), (d, e, f) = axes
    if 5 in symrows:
        _seq_syn_panel(a, symrows[5], "seq_skr_u_yau", "nested_skr_u_yau",
                       r"$u_{skr}$ (bits/s)"); a.set_title("Symmetric L=5: SKR utility")
        _seq_syn_panel(b, symrows[5], "seq_mean_link_occupancy",
                       "nested_mean_link_occupancy", "occupancy")
        b.set_title("Symmetric L=5: occupancy")
        _seq_syn_panel(c, symrows[5], "seq_mean_fe2e", "nested_mean_fe2e",
                       r"$F_{e2e}$", keep_zero=False)
        c.axhline(F_MIN, color="green", linestyle=":", linewidth=1.2)
        c.set_title("Symmetric L=5: fidelity")
    if 10 in symrows:
        _seq_syn_panel(d, symrows[10], "seq_skr_u_yau", "nested_skr_u_yau",
                       r"$u_{skr}$ (bits/s)")
        d.set_title(f"Symmetric L=10: SKR ({len(symrows[10])}/12)")
    if bnrows:
        for p in sorted(bnrows):
            st = POS[p]; x, y = xy(bnrows[p], "nested_skr_u_yau")
            e.plot(x, y, color=st["color"], marker=st["marker"], linewidth=1.6,
                   markersize=4, label=f"hop {p}")
        e.set_xscale("log"); e.grid(True, which="both", alpha=0.3)
        e.set_xlabel(r"$T_c$ ($\mu$s)"); e.set_ylabel(r"$u_{skr}$ (bits/s)")
        e.set_title("Bottleneck: Synchronized SKR"); e.legend(frameon=False, fontsize=8)
    pal = {5: "#0072B2", 10: "#D55E00"}
    for L in sorted(symrows):
        x, y = xy(symrows[L], "delta_pct_yau")
        f.plot(x, y, marker="o", color=pal.get(L), linewidth=1.7, markersize=4,
               label=f"L={L}")
    f.axhline(0, color="0.4", linestyle="--", linewidth=1.1)
    f.set_xscale("log"); f.grid(True, which="both", alpha=0.3)
    f.set_xlabel(r"$T_c$ ($\mu$s)"); f.set_ylabel(r"$\Delta u_{skr}$ (%)")
    f.set_title("Winner (>0 Synchronized, <0 Sequential)"); f.legend(frameon=False, fontsize=8)
    fig.suptitle("Controller-layer sweep: Sequential vs Synchronized (n=4)", fontsize=14)
    save(fig, outdir, "dashboard.png")


#%% Paper figures, one per link length
def paper_skr_crossover(rows_L, L, outdir):
    with plt.rc_context(PAPER_RC):
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(5.2, 6.0), sharex=True, constrained_layout=True,
            gridspec_kw={"height_ratios": [1.7, 1.0]})
        x, y = xy(rows_L, "seq_skr_u_yau"); ax1.plot(x, y, **SEQ_P)
        x, y = xy(rows_L, "nested_skr_u_yau"); ax1.plot(x, y, **SYN_P)
        ax1.set_ylabel(r"$u_{\mathrm{skr}}$ (bits/s)")
        framed_legend(ax1, loc="upper left")
        xd, yd = xy(rows_L, "delta_pct_yau")
        ax2.plot(xd, yd, color="#222222", marker="D", linestyle="-", markersize=6)
        ax2.axhline(0, color="0.5", linestyle="--", linewidth=1.2)
        ax2.set_ylabel(r"$\Delta u_{\mathrm{skr}}$ (%)")
        ax2.set_xlabel(r"$T_c$ ($\mu$s)")
        ax2.set_xscale("log")
        if yd:
            lo, hi = min(yd + [0.0]), max(yd + [0.0])
            span = (hi - lo) or 1.0
            ax2.set_ylim(lo - 0.18 * span - 1.0, hi + 0.30 * span + 1.0)
        b = collapse_boundary(rows_L)
        if b is not None:
            ax2.axvline(b, color="0.6", linestyle=":", linewidth=1.4)
            ax2.annotate("Sequential = 0\n(collapsed)",
                         xy=(b, ax2.get_ylim()[0]),
                         xytext=(b / 1.25, ax2.get_ylim()[0]),
                         ha="right", va="bottom", fontsize=12, color="0.25")
        fig.suptitle(f"L = {L} km")
        save(fig, outdir, f"skr_crossover_L{L}.png", use_tight=False)


def paper_occupancy(rows_L, L, outdir):
    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(5.4, 4.2), constrained_layout=True)
        x, y = xy(rows_L, "seq_mean_link_occupancy"); ax.plot(x, y, **SEQ_P)
        x, y = xy(rows_L, "nested_mean_link_occupancy"); ax.plot(x, y, **SYN_P)
        ax.set_xscale("log")
        ax.set_xlabel(r"$T_c$ ($\mu$s)")
        ax.set_ylabel("mean link occupancy")
        ax.set_title(f"L = {L} km")
        framed_legend(ax, loc="upper left")
        save(fig, outdir, f"occupancy_L{L}.png", use_tight=False)


def paper_fidelity(rows_L, L, outdir):
    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(5.4, 4.4), constrained_layout=True)
        x, y = xy(rows_L, "seq_mean_fe2e", keep_zero=False); ax.plot(x, y, **SEQ_P)
        x, y = xy(rows_L, "nested_mean_fe2e", keep_zero=False); ax.plot(x, y, **SYN_P)
        ax.set_xscale("log")
        ax.set_xlabel(r"$T_c$ ($\mu$s)")
        ax.set_ylabel(r"end-to-end fidelity $F_{\mathrm{e2e}}$")
        allf = [d.get(c) for d in rows_L for c in ("seq_mean_fe2e", "nested_mean_fe2e")
                if isinstance(d.get(c), float) and math.isfinite(d.get(c)) and d.get(c) > 0]
        if allf:
            ax.set_ylim(min(allf) - 0.0025, max(allf) + 0.0085)
        ax.set_title(f"L = {L} km")
        framed_legend(ax, loc="upper center", fontsize=11)
        save(fig, outdir, f"fidelity_L{L}.png", use_tight=False)


#%% Paper figures, lengths side by side
def paper_skr_crossover_combined(symrows, Ls, outdir):
    """2 x N: absolute u_skr (top row) and relative difference (bottom row),
    one column per link length, shared x within each column."""
    ncol = len(Ls)
    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(
            2, ncol, figsize=(4.9 * ncol, 6.2), sharex="col",
            constrained_layout=True, squeeze=False,
            gridspec_kw={"height_ratios": [1.7, 1.0]})
        for j, L in enumerate(Ls):
            axt, axb = axes[0][j], axes[1][j]
            x, y = xy(symrows[L], "seq_skr_u_yau"); axt.plot(x, y, **SEQ_P)
            x, y = xy(symrows[L], "nested_skr_u_yau"); axt.plot(x, y, **SYN_P)
            axt.set_xscale("log"); axt.tick_params(labelbottom=False)
            axt.set_title(f"L = {L} km")

            xd, yd = xy(symrows[L], "delta_pct_yau")
            axb.plot(xd, yd, color="#222222", marker="D", linestyle="-", markersize=6)
            axb.axhline(0, color="0.5", linestyle="--", linewidth=1.2)
            axb.set_xscale("log"); axb.set_xlabel(r"$T_c$ ($\mu$s)")
            if yd:
                lo, hi = min(yd + [0.0]), max(yd + [0.0])
                span = (hi - lo) or 1.0
                axb.set_ylim(lo - 0.18 * span - 1.0, hi + 0.30 * span + 1.0)
            b = collapse_boundary(symrows[L])
            if b is not None:
                axb.axvline(b, color="0.6", linestyle=":", linewidth=1.4)
                axb.annotate("Seq = 0", xy=(b, axb.get_ylim()[0]),
                             xytext=(b / 1.3, axb.get_ylim()[0]),
                             ha="right", va="bottom", fontsize=10.5, color="0.3")
        axes[0][0].set_ylabel(r"$u_{\mathrm{skr}}$ (bits/s)")
        axes[1][0].set_ylabel(r"$\Delta u_{\mathrm{skr}}$ (%)")
        framed_legend(axes[0][0], loc="upper left", fontsize=11.5)
        save(fig, outdir, "skr_crossover_combined.png", use_tight=False)


def paper_occupancy_combined(symrows, Ls, outdir):
    ncol = len(Ls)
    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(1, ncol, figsize=(5.3 * ncol, 4.3),
                                 constrained_layout=True, squeeze=False)
        for j, L in enumerate(Ls):
            ax = axes[0][j]
            x, y = xy(symrows[L], "seq_mean_link_occupancy"); ax.plot(x, y, **SEQ_P)
            x, y = xy(symrows[L], "nested_mean_link_occupancy"); ax.plot(x, y, **SYN_P)
            ax.set_xscale("log"); ax.set_xlabel(r"$T_c$ ($\mu$s)")
            ax.set_title(f"L = {L} km")
        axes[0][0].set_ylabel("mean link occupancy")
        framed_legend(axes[0][0], loc="upper left", fontsize=11.5)
        save(fig, outdir, "occupancy_combined.png", use_tight=False)


def paper_fidelity_combined(symrows, Ls, outdir):
    ncol = len(Ls)
    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(1, ncol, figsize=(5.3 * ncol, 4.4),
                                 constrained_layout=True, squeeze=False)
        for j, L in enumerate(Ls):
            ax = axes[0][j]
            x, y = xy(symrows[L], "seq_mean_fe2e", keep_zero=False); ax.plot(x, y, **SEQ_P)
            x, y = xy(symrows[L], "nested_mean_fe2e", keep_zero=False); ax.plot(x, y, **SYN_P)
            ax.set_xscale("log"); ax.set_xlabel(r"$T_c$ ($\mu$s)")
            ax.set_title(f"L = {L} km")
            allf = [d.get(c) for d in symrows[L]
                    for c in ("seq_mean_fe2e", "nested_mean_fe2e")
                    if isinstance(d.get(c), float) and math.isfinite(d.get(c)) and d.get(c) > 0]
            if allf:
                # headroom so the top-center legend in panel 0 clears the flat line
                ax.set_ylim(min(allf) - 0.0025, max(allf) + 0.0085)
        axes[0][0].set_ylabel(r"end-to-end fidelity $F_{\mathrm{e2e}}$")
        framed_legend(axes[0][0], loc="upper center", fontsize=11)
        save(fig, outdir, "fidelity_combined.png", use_tight=False)


def paper_bottleneck(bnrows, seqcol, syncol, ylabel, fname, outdir, coincide=True):
    if not bnrows:
        return
    with plt.rc_context(PAPER_RC):
        fig, (axa, axb) = plt.subplots(1, 2, figsize=(10.4, 4.3), sharey=True,
                                       constrained_layout=True)
        for p in sorted(bnrows):
            st = POS_P[p]
            x, y = xy(bnrows[p], seqcol); axa.plot(x, y, markersize=6, **st)
            x, y = xy(bnrows[p], syncol); axb.plot(x, y, markersize=6, **st)
        for ax, name in ((axa, "Sequential"), (axb, "Synchronized")):
            ax.set_xscale("log"); ax.set_xlabel(r"$T_c$ ($\mu$s)"); ax.set_title(name)
        axa.set_ylabel(ylabel)
        framed_legend(axa, loc="upper left")
        if coincide:
            axb.annotate("all positions coincide",
                         xy=(0.04, 0.94), xycoords="axes fraction",
                         ha="left", va="top", fontsize=12.5, color="0.3")
        save(fig, outdir, fname, use_tight=False)


#%% Driver
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="controller_results.csv")
    ap.add_argument("--outdir", default=os.path.join("plot", "controller"))
    ap.add_argument("--paper-min", type=int, default=10,
                    help="min cells for a length to enter the paper figures (default 10)")
    ap.add_argument("--per-length", action="store_true",
                    help="emit one figure per length instead of the combined comparison")
    args = ap.parse_args()

    if not os.path.isfile(args.csv):
        raise SystemExit(f"ERROR: CSV not found: {args.csv}\n"
                         f"Run from the folder with the CSV, or pass --csv.")
    paper_dir = os.path.join(args.outdir, "paper")
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(paper_dir, exist_ok=True)

    rows = load(args.csv)
    symrows = sym_by_L(rows)
    bnrows = bn_by_pos(rows)
    print(f"Loaded {len(rows)} configs: "
          f"symmetric {sum(len(v) for v in symrows.values())} "
          f"({', '.join(f'L{L}:{len(symrows[L])}' for L in sorted(symrows))}), "
          f"bottleneck {sum(len(v) for v in bnrows.values())}")

    # ---- exploratory set ----
    print(f"Exploratory figures -> {os.path.abspath(args.outdir)}")
    explore_symmetric(symrows, "seq_skr_u_yau", "nested_skr_u_yau",
                      r"$u_{skr}$ (bits/s)", "SKR utility vs coherence time",
                      "symmetric_u_skr.png", args.outdir)
    explore_symmetric(symrows, "seq_mean_link_occupancy", "nested_mean_link_occupancy",
                      "mean link occupancy", "Link occupancy vs coherence time",
                      "symmetric_occupancy.png", args.outdir)
    explore_symmetric(symrows, "seq_mean_fe2e", "nested_mean_fe2e",
                      r"end-to-end fidelity $F_{e2e}$",
                      "Delivered fidelity vs coherence time",
                      "symmetric_fe2e.png", args.outdir,
                      refline=(F_MIN, f"$F_{{min}}$ = {F_MIN}"), keep_zero=False)
    explore_bottleneck(bnrows, "seq_skr_u_yau", "nested_skr_u_yau",
                       r"$u_{skr}$ (bits/s)", "Bottleneck: SKR utility",
                       "bottleneck_u_skr.png", args.outdir)
    explore_bottleneck(bnrows, "seq_mean_link_occupancy", "nested_mean_link_occupancy",
                       "mean link occupancy", "Bottleneck: link occupancy",
                       "bottleneck_occupancy.png", args.outdir)
    explore_dashboard(symrows, bnrows, args.outdir)

    # ---- paper set ----
    print(f"Paper figures -> {os.path.abspath(paper_dir)}")
    ok = [L for L in sorted(symrows) if len(symrows[L]) >= args.paper_min]
    for L in sorted(symrows):
        if len(symrows[L]) < args.paper_min:
            print(f"  L={L} incomplete ({len(symrows[L])}/12; need >= {args.paper_min}), excluded")

    if args.per_length or len(ok) < 2:
        for L in ok:
            paper_skr_crossover(symrows[L], L, paper_dir)
            paper_occupancy(symrows[L], L, paper_dir)
            paper_fidelity(symrows[L], L, paper_dir)
        if len(ok) < 2 and not args.per_length:
            print("  (fewer than 2 complete lengths: emitted per-length; the combined "
                  "L-comparison appears once >= 2 lengths are complete)")
    else:
        paper_skr_crossover_combined(symrows, ok, paper_dir)
        paper_occupancy_combined(symrows, ok, paper_dir)
        paper_fidelity_combined(symrows, ok, paper_dir)

    paper_bottleneck(bnrows, "seq_mean_link_occupancy", "nested_mean_link_occupancy",
                     "mean link occupancy", "bottleneck_occupancy.png", paper_dir)
    paper_bottleneck(bnrows, "seq_skr_u_yau", "nested_skr_u_yau",
                     r"$u_{\mathrm{skr}}$ (bits/s)", "bottleneck_skr.png", paper_dir)
    print("Done.")


if __name__ == "__main__":
    main()