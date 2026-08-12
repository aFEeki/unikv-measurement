#!/usr/bin/env python3
"""Figure 6 for the measurement paper — where the capacity wall actually is.

Optional figure for Finding 4. The finding has three parts that a table states
but does not show:

  1. Device cost per context token is not the KV alone. With flash attention off
     the attention scratch adds a term that scales with ubatch, so the same C
     costs a different amount of device memory at different micro-batches.
  2. The wall IS the Metal advisory budget: every configuration completes below
     it and is refused above it, so where a line crosses is where that ubatch
     runs out of context.
  3. Crossing is refused at EXECUTION, not allocation -- the arithmetic predicts
     WHERE, but nothing in the arithmetic says the failure arrives as a refused
     command buffer rather than a failed malloc.

Plotting measured device footprint against C, one line per ubatch, with the
budget drawn and each run marked by its outcome, puts (1) and (2) in one panel:
the fan of lines is the ubatch dependence, and every line's last filled marker
sits below the budget line while its first hollow marker sits above it.

NB the budget is printed by ggml in decimal MB and every footprint we record is
in MiB. Comparing the two directly understates every ratio by 4.9%; see the
conversion at BUDGET below.

Source: stress_results/f4_a1_ubatch_sweep.csv (Phase A, item A1).
Writes figures/fig6_capacity.{pdf,png} and copies the PDF into
paper/UNIKV-MEASUREMENT/ so it is available if the draft wants it. It is NOT
referenced by main.tex; adding the float is a prose decision.
"""

import csv
import shutil
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT      = Path(__file__).resolve().parents[1]
SRC_CSV   = ROOT / "stress_results" / "f4_a1_ubatch_sweep.csv"
FIG_DIR   = ROOT / "figures"
PAPER_DIR = ROOT / "paper" / "UNIKV-MEASUREMENT"

# ggml prints "recommendedMaxWorkingSetSize = 17179.89 MB" in DECIMAL megabytes.
# Every device figure we record is in MiB, so the budget must be converted:
# 17179.89e6 / 2^20 = 16384.0 MiB. Comparing MiB against the raw 17179.89
# understates every ratio by 4.9% and moves the wall line off its true place.
BUDGET_MB  = 17179.89
BUDGET     = BUDGET_MB * 1e6 / 2**20    # 16384.0 MiB
MODEL    = 4685.3       # MiB of weights, constant across the sweep
UBATCHES = [64, 128, 256, 512]

DARK   = "#222222"
GRAY   = "#888888"
COLOR  = {64: "#1f4e79", 128: "#2e7d32", 256: "#ef6c00", 512: "#c1272d"}


def die(msg):
    raise SystemExit(f"STOP (provenance mismatch): {msg}")


def load():
    if not SRC_CSV.exists():
        die(f"{SRC_CSV.name}: missing")
    rows = list(csv.DictReader(SRC_CSV.open()))

    fa = {r["flash_attn"] for r in rows}
    if fa != {"disabled"}:
        die(f"{SRC_CSV.name}: flash_attn = {fa}; the scratch term is an fa-off effect")
    bud = {float(r["budget_mib"]) for r in rows}
    if len(bud) != 1 or abs(bud.pop() - BUDGET_MB) > 0.01:
        die(f"{SRC_CSV.name}: budget disagrees with the locked {BUDGET_MB} MB")

    data = {}
    for r in rows:
        data.setdefault(int(r["n_ubatch"]), []).append(
            (int(r["ctx"]), float(r["device_total_mib"]), r["outcome"]))
    for ub in UBATCHES:
        if ub not in data:
            die(f"{SRC_CSV.name}: ubatch {ub} missing")
        data[ub].sort()

    # With the budget in the right units the simple claim holds: every run that
    # completed sits below it and every run that was refused sits above it.
    for ub, pts in data.items():
        for c, mib, out in pts:
            if out == "COMPLETES" and mib > BUDGET:
                die(f"ubatch {ub} C={c}: completed at {mib:.0f} MiB, above the "
                    f"{BUDGET:.0f} MiB budget")
            if out == "GPU_OOM" and mib < BUDGET:
                die(f"ubatch {ub} C={c}: refused at {mib:.0f} MiB, below the "
                    f"{BUDGET:.0f} MiB budget")
    hi_ok  = max(m for pts in data.values() for _, m, o in pts if o == "COMPLETES")
    lo_bad = min(m for pts in data.values() for _, m, o in pts if o == "GPU_OOM")
    print(f"  [ok] fig6 <- {SRC_CSV.name}: {len(rows)} runs, all fa off")
    print(f"       largest completion {hi_ok:.0f} MiB ({hi_ok/BUDGET:.3f}x), "
          f"smallest refusal {lo_bad:.0f} MiB ({lo_bad/BUDGET:.3f}x), "
          f"budget {BUDGET:.0f} MiB falls between them")
    return data, hi_ok, lo_bad


def render(data, hi_ok, lo_bad):
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "font.size": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": DARK, "axes.linewidth": 0.8,
        "grid.color": "#dddddd", "grid.linewidth": 0.5,
        "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True,
        "savefig.bbox": "tight", "savefig.dpi": 300,
    })
    fig, ax = plt.subplots(figsize=(3.15, 2.4))

    # The measured wall is a band, not a line: the largest working footprint and
    # the smallest refused one bracket it, and it sits just below the advisory
    # budget rather than exactly on it.
    ax.axhspan(BUDGET, 21000, color=GRAY, alpha=0.13, linewidth=0, zorder=0)
    ax.axhline(BUDGET, color=DARK, linestyle="--", linewidth=1.1, zorder=2)
    ax.text(41200, BUDGET - 500, "Metal working-set budget", fontsize=6.3,
            color=DARK, ha="left", va="top")

    for ub in UBATCHES:
        pts = data[ub]
        cs   = [c for c, _, _ in pts]
        mibs = [m for _, m, _ in pts]
        ax.plot(cs, mibs, color=COLOR[ub], linewidth=1.0, zorder=3)
        for c, m, out in pts:
            ok = out == "COMPLETES"
            ax.plot([c], [m], marker="o" if ok else "X", markersize=4.0 if ok else 4.6,
                    color=COLOR[ub],
                    markerfacecolor=COLOR[ub] if ok else "white",
                    markeredgecolor=COLOR[ub], markeredgewidth=1.0, zorder=4)
        ax.text(cs[-1] + 3500, mibs[-1], f"ub {ub}", fontsize=6.3,
                color=COLOR[ub], va="center")

    ax.set_xlabel("Context size $C$ (tokens)")
    ax.set_ylabel("Device working set (MiB)")
    ax.set_xlim(40000, 118000)
    ax.set_ylim(9000, 20200)
    ax.set_xticks([49152, 65536, 81920, 98304])
    ax.set_xticklabels(["48K", "64K", "80K", "96K"])

    handles = [
        Line2D([], [], marker="o", color=DARK, markerfacecolor=DARK,
               markersize=4.0, linestyle="none", label="completes"),
        Line2D([], [], marker="X", color=DARK, markerfacecolor="white",
               markeredgecolor=DARK, markersize=4.6, linestyle="none",
               label="refused (GPU OOM)"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False,
              handletextpad=0.4, borderaxespad=0.3, labelspacing=0.3)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig6_capacity.{ext}")
    plt.close(fig)
    print(f"  saved {FIG_DIR/'fig6_capacity.pdf'} (+ .png)")

    if PAPER_DIR.exists():
        shutil.copyfile(FIG_DIR / "fig6_capacity.pdf", PAPER_DIR / "fig6_capacity.pdf")
        print(f"  copied -> {PAPER_DIR/'fig6_capacity.pdf'}  (not referenced by main.tex)")


def main():
    print("Verifying fig6 against the A1 ubatch sweep ...")
    data, hi_ok, lo_bad = load()
    render(data, hi_ok, lo_bad)
    print("\n  per-ubatch device cost per context token (KV + scratch):")
    for ub in UBATCHES:
        pts = data[ub]
        (c0, m0, _), (c1, m1, _) = pts[0], pts[-1]
        kib = (m1 - m0) * 1024 / (c1 - c0)
        print(f"    ubatch {ub:3d}: {kib:6.1f} KiB/token   "
              f"-> budget crossed near C = {int((BUDGET - MODEL) * 1024 / kib):6d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
