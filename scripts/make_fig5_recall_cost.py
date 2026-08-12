#!/usr/bin/env python3
"""Figure 5 for the measurement paper — per-step recall cost, BOTH tier modes.

Supersedes the single-series version that analyze_recall_cost.py produced from
the old CPU-pinned-only sweep. Source is B2 Block 1
(stress_results/b2_isochronal_both_modes.csv), the cooled protocol block.

The figure has to carry two facts at once, and they are of different sizes:

  intercepts differ a lot     31.563 vs 25.423 ms  -> gamma 9.009 vs 2.879 ms
  slopes differ visibly less  3.0990 vs 2.1531 us/cell

Left to itself the intercept gap dominates and the slope difference reads as
"the lines are parallel", which is the wrong conclusion. Two devices stop that:
the region between the fits is shaded, so the widening wedge IS the slope
difference (6.1 ms at n=0 growing to 13.9 ms at n=8192), and each fit carries
its slope as a label. The y axis still starts at zero -- the project convention,
and the shared no-spill reference is drawn on the plot rather than hidden below
a truncated axis.

Every locked value is asserted against the CSV; the script dies naming the file
on any disagreement rather than silently drawing something else.

Writes figures/fig5_recall_cost.{pdf,png} and copies the PDF into
paper/UNIKV-MEASUREMENT/.
"""

import csv
import shutil
import statistics as st
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT     = Path(__file__).resolve().parents[1]
SRC_CSV  = ROOT / "stress_results" / "b2_isochronal_both_modes.csv"
FIG_DIR  = ROOT / "figures"
PAPER_DIR = ROOT / "paper" / "UNIKV-MEASUREMENT"

TOL_MS   = 0.005     # means are quoted to 3 dp
TOL_FIT  = 0.002     # fitted coefficients

# ---- LOCKED (B2 Block 1; figure conforms to these, not the reverse) --------
# mode -> (intercept ms, slope us/cell, gamma ms)
FITS = {
    "cpu": (31.563, 3.0990, 9.009),
    "dev": (25.423, 2.1531, 2.879),
}
# mode -> {n_spill: (mean ms, sd ms)}
POINTS = {
    "cpu": {0: (22.554, 0.034), 512: (33.958, 0.085), 1024: (34.702, 0.153),
            2048: (37.344, 0.200), 4096: (43.892, 0.696), 8192: (57.349, 1.456)},
    "dev": {0: (22.544, 0.052), 512: (26.552, 0.064), 1024: (27.639, 0.063),
            2048: (29.773, 0.077), 4096: (34.264, 0.021), 8192: (43.063, 0.028)},
}
NO_SPILL = 22.549    # the two references agree (t = -0.28); one shared line
NMAX     = 8192

LABEL = {"cpu": "CPU-pinned tier", "dev": "device-visible tier"}
DEVCOL = {"0": "cpu", "1": "dev"}

ACCENT = "#c1272d"
DARK   = "#222222"
GRAY   = "#888888"
COLOR  = {"cpu": ACCENT, "dev": "#1f4e79"}


def die(msg):
    raise SystemExit(f"STOP (provenance mismatch): {msg}")


def ols(xs, ys):
    n = len(xs)
    mx, my = st.mean(xs), st.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return b, my - b * mx


def verify():
    if not SRC_CSV.exists():
        die(f"{SRC_CSV.name}: missing")
    rows = list(csv.DictReader(SRC_CSV.open()))

    fa = {r["flash_attn"] for r in rows}
    if fa != {"disabled"}:
        die(f"{SRC_CSV.name}: flash_attn = {fa}, must be disabled on every run")

    for dev, mode in DEVCOL.items():
        got = {}
        for r in rows:
            if r["spill_dev"] == dev:
                got.setdefault(int(r["target"]), []).append(float(r["ms_mean"]))

        for n, (m_lock, sd_lock) in POINTS[mode].items():
            if n not in got:
                die(f"{SRC_CSV.name}: {mode} n_spill={n} missing")
            v = got[n]
            m = st.mean(v)
            sd = st.stdev(v) if len(v) > 1 else 0.0
            if abs(m - m_lock) > TOL_MS:
                die(f"{SRC_CSV.name}: {mode} n={n} mean {m:.3f}, locked {m_lock}")
            if abs(sd - sd_lock) > TOL_MS:
                die(f"{SRC_CSV.name}: {mode} n={n} sd {sd:.3f}, locked {sd_lock}")

        xs = [int(r["target"]) for r in rows
              if r["spill_dev"] == dev and int(r["target"]) > 0]
        ys = [float(r["ms_mean"]) for r in rows
              if r["spill_dev"] == dev and int(r["target"]) > 0]
        b, a = ols(xs, ys)
        a_lock, b_lock, g_lock = FITS[mode]
        if abs(a - a_lock) > TOL_FIT:
            die(f"{SRC_CSV.name}: {mode} intercept {a:.4f}, locked {a_lock}")
        if abs(b * 1000 - b_lock) > TOL_FIT:
            die(f"{SRC_CSV.name}: {mode} slope {b*1000:.4f}, locked {b_lock}")
        gamma = a_lock - POINTS[mode][0][0]
        if abs(gamma - g_lock) > TOL_FIT:
            die(f"{mode}: gamma {gamma:.4f} from locked values, locked {g_lock}")

    # the shared reference is only legitimate because the two agree
    d = abs(POINTS["cpu"][0][0] - POINTS["dev"][0][0])
    if d > 0.15:
        die(f"no-spill references differ by {d:.3f} ms — do not draw one shared line")

    print(f"  [ok] fig5 <- {SRC_CSV.name}: both series, 6 points each, "
          f"fits and gammas verified, flash attention off on all runs")


def render():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "font.size": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": DARK, "axes.linewidth": 0.8,
        "grid.color": "#dddddd", "grid.linewidth": 0.5,
        "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True,
        "savefig.bbox": "tight", "savefig.dpi": 300,
    })

    fig, ax = plt.subplots(figsize=(3.0, 2.35))
    xs = [0, NMAX]

    def line(mode):
        a, b, _ = FITS[mode]
        return [a + b * x / 1000.0 for x in xs]

    y_cpu, y_dev = line("cpu"), line("dev")

    # the widening gap IS the slope difference: 6.1 ms at n=0 -> 13.9 ms at NMAX
    ax.fill_between(xs, y_dev, y_cpu, color=GRAY, alpha=0.13, linewidth=0, zorder=1)

    # shared no-spill reference (the two modes agree, t = -0.28)
    ax.axhline(NO_SPILL, color=DARK, linestyle=":", linewidth=0.9, zorder=2)
    ax.text(NMAX * 0.985, NO_SPILL - 3.4, f"no spill ({NO_SPILL:.1f} ms)",
            ha="right", va="bottom", fontsize=6.5, color=DARK)

    for mode in ("cpu", "dev"):
        a, b, _ = FITS[mode]
        ns = sorted(POINTS[mode])
        ax.plot(xs, line(mode), color=COLOR[mode], linewidth=1.2,
                linestyle="-" if mode == "cpu" else "--", zorder=3)
        ax.errorbar(ns, [POINTS[mode][n][0] for n in ns],
                    yerr=[POINTS[mode][n][1] for n in ns],
                    marker="o" if mode == "cpu" else "s", markersize=3.4,
                    color=COLOR[mode], markerfacecolor=COLOR[mode],
                    markeredgecolor="white", markeredgewidth=0.6,
                    ecolor=DARK, elinewidth=0.8, capsize=2.0, capthick=0.7,
                    linestyle="none", zorder=4)

    # slope labels: the fact the wedge is there to make visible
    ax.text(4450, FITS["cpu"][0] + FITS["cpu"][1] * 4.45 + 3.0,
            r"$3.099\,\mu$s/cell", fontsize=6.5, color=COLOR["cpu"], ha="center")
    ax.text(4700, FITS["dev"][0] + FITS["dev"][1] * 4.70 - 5.2,
            r"$2.153\,\mu$s/cell", fontsize=6.5, color=COLOR["dev"], ha="center")

    ax.set_xlabel(r"Spilled cells $n_{\mathrm{spill}}$")
    ax.set_ylabel("Decode step time (ms)")
    ax.set_xlim(-250, NMAX + 250)
    ax.set_ylim(0, 60)
    ax.set_xticks([0, 2048, 4096, 6144, 8192])

    # Legend goes in the empty lower-left rather than over the data, and uses
    # proxy handles: the errorbar artists would otherwise draw their caps into
    # the legend, which reads as stray marks at this size.
    handles = [Line2D([], [], color=COLOR[m], marker=("o" if m == "cpu" else "s"),
                      markersize=3.4, markerfacecolor=COLOR[m],
                      markeredgecolor="white", markeredgewidth=0.6,
                      linestyle="-" if m == "cpu" else "--", linewidth=1.2,
                      label=LABEL[m]) for m in ("cpu", "dev")]
    ax.legend(handles=handles, loc="lower left", frameon=False,
              handletextpad=0.6, borderaxespad=0.4, labelspacing=0.35,
              handlelength=2.2)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig5_recall_cost.{ext}")
    plt.close(fig)
    print(f"  saved {FIG_DIR/'fig5_recall_cost.pdf'} (+ .png)")

    if PAPER_DIR.exists():
        shutil.copyfile(FIG_DIR / "fig5_recall_cost.pdf",
                        PAPER_DIR / "fig5_recall_cost.pdf")
        print(f"  copied -> {PAPER_DIR/'fig5_recall_cost.pdf'}")
    else:
        die(f"{PAPER_DIR} missing — figure not delivered to the paper")


def main():
    print("Verifying fig5 against B2 Block 1 ...")
    verify()
    render()
    a_c, b_c, g_c = FITS["cpu"]
    a_d, b_d, g_d = FITS["dev"]
    print(f"\n  gap at n=0    : {a_c - a_d:.2f} ms   (intercepts)")
    print(f"  gap at n={NMAX}: {(a_c + b_c*NMAX/1000) - (a_d + b_d*NMAX/1000):.2f} ms"
          f"   (the wedge; growth is the slope difference)")
    print(f"  gamma         : {g_c:.3f} -> {g_d:.3f} ms  ({100*(1-g_d/g_c):.0f}% lower)")
    print(f"  delta         : {b_c:.4f} -> {b_d:.4f} us/cell  ({100*(1-b_d/b_c):.1f}% lower)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
