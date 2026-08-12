#!/usr/bin/env python3
"""Figure 7 for the measurement paper — an injected delay is invisible undrained.

Finding 3's most surprising result is currently a five-row table the reader has
to do arithmetic on. The figure states it directly: with the pipeline drained,
the modelled transfer cost shows up; without the drain, the same injected sleep
is absorbed by asynchronous execution and barely registers.

PROVENANCE DECISION: this plots the drain-control block ALONE
(alpha_results/p3_drain_control_master.csv). The cooled alpha sweep has a fuller
drained curve (alpha 0 through 2) and the two blocks share an alpha=0 undrained
point agreeing to 0.30%, so overlaying was on the table. It was rejected:

  - the two blocks agree on LEVEL but disagree on SLOPE. The drain-control's
    drained segment falls at -1.991 +/- 0.300 tok/s per unit alpha; the cooled
    sweep's drained fit is -1.197 +/- 0.052 over alpha 0.25-2. That is t = -2.6.
  - this figure's whole claim is that two series differ in SLOPE, so agreement
    on level at one anchor does not license the splice. A combined drained
    series would carry a kink at alpha=0.25 that no reader could attribute to
    curvature rather than to the block change.
  - the within-block contrast is in any case the stronger statistic: at the one
    alpha where both conditions were measured, undrained minus drained is
    +0.713 +/- 0.068 tok/s, t = 10.5.

So the left-hand quarter of the x range carries the comparison and the drained
series stops where the data stops, which is marked on the plot.

HONESTY NOTE ON "FLAT": the undrained series is NOT flat. Over alpha 0 to 1 it
falls 0.541 +/- 0.183 tok/s (t = -3.0), a real 1.8% decline. What the figure
shows is that it is much SHALLOWER: -0.588 against -1.991 tok/s per unit alpha,
a factor of 3.4. The labels say so rather than claiming flatness.

AXIS: y is truncated. The claim is a difference in slope, not the size of a
ratio, and the whole effect spans 0.7 tok/s out of ~29.5, so a zero baseline
would render it invisible. The truncation is marked with a break glyph on the
axis.

Every plotted value is asserted against the CSV.
Writes figures/fig7_drain.{pdf,png} and copies the PDF to
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

ROOT      = Path(__file__).resolve().parents[1]
SRC_CSV   = ROOT / "alpha_results" / "p3_drain_control_master.csv"
FIG_DIR   = ROOT / "figures"
PAPER_DIR = ROOT / "paper" / "UNIKV-MEASUREMENT"

TOL = 0.0015

# ---- LOCKED (drain-control block; condition -> alpha, mean, sd) ------------
UNDRAINED = [(0.00, 29.762, 0.281, "a0_nodrain"),
             (0.25, 29.781, 0.110, "a025_nodrain"),
             (1.00, 29.221, 0.145, "a1_nodrain")]
DRAINED   = [(0.00, 29.566, 0.123, "a0_drain"),
             (0.25, 29.068, 0.042, "a025_drain")]

GAP_ALPHA  = 0.25       # the one alpha measured in both conditions
GAP_VALUE  = 0.713      # undrained - drained there
GAP_SE     = 0.068
GAP_T      = 10.5
SLOPE_UND  = -0.588     # tok/s per unit alpha, 3 points
SLOPE_DRN  = -1.991     # tok/s per unit alpha, 2 points

DARK   = "#222222"
GRAY   = "#888888"
C_UND  = "#1f4e79"      # undrained
C_DRN  = "#c1272d"      # drained


def die(msg):
    raise SystemExit(f"STOP (provenance mismatch): {msg}")


def verify():
    if not SRC_CSV.exists():
        die(f"{SRC_CSV.name}: missing")
    rows = list(csv.DictReader(SRC_CSV.open()))

    if {r["flash_attn"] for r in rows} != {"disabled"}:
        die(f"{SRC_CSV.name}: flash attention must be off on every run")
    if {r["rc"] for r in rows} != {"0"}:
        die(f"{SRC_CSV.name}: non-zero return codes present")
    if {r["decode_tokens"] for r in rows} != {"2048"}:
        die(f"{SRC_CSV.name}: not every run decoded the full budget")

    by = {}
    for r in rows:
        by.setdefault(r["condition"], []).append(float(r["tok_per_sec"]))

    for series, label in ((UNDRAINED, "undrained"), (DRAINED, "drained")):
        for alpha, mean_l, sd_l, cond in series:
            if cond not in by:
                die(f"{SRC_CSV.name}: condition {cond} missing")
            v = by[cond]
            if len(v) != 3:
                die(f"{SRC_CSV.name}: {cond} has {len(v)} trials, expected 3")
            m, s = st.mean(v), st.stdev(v)
            if abs(m - mean_l) > TOL:
                die(f"{SRC_CSV.name}: {cond} mean {m:.4f}, locked {mean_l}")
            if abs(s - sd_l) > TOL:
                die(f"{SRC_CSV.name}: {cond} sd {s:.4f}, locked {sd_l}")
            # the alpha recorded in the CSV must match where we plot the point
            got_alpha = {float(r["alpha"]) for r in rows if r["condition"] == cond}
            if got_alpha != {alpha}:
                die(f"{SRC_CSV.name}: {cond} alpha {got_alpha}, plotted at {alpha}")
            # and so must the drain setting
            got_drain = {r["drain"] for r in rows if r["condition"] == cond}
            drained_expected = label == "drained"
            drained_actual = got_drain in ({"1"}, {"default"}) and not (
                got_drain == {"default"} and alpha == 0.0)
            if drained_actual != drained_expected:
                die(f"{SRC_CSV.name}: {cond} drain={got_drain} at alpha={alpha} "
                    f"does not match the '{label}' series")

    gap = (st.mean(by["a025_nodrain"]) - st.mean(by["a025_drain"]))
    if abs(gap - GAP_VALUE) > TOL:
        die(f"{SRC_CSV.name}: alpha=0.25 gap {gap:.4f}, locked {GAP_VALUE}")

    print(f"  [ok] fig7 <- {SRC_CSV.name}: 5 conditions x 3 trials, flash "
          f"attention off, all rc=0, all 2048 tokens; means, SDs, alphas, drain "
          f"settings and the alpha=0.25 gap all verified")


def render():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "font.size": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": DARK, "axes.linewidth": 0.8,
        "grid.color": "#dddddd", "grid.linewidth": 0.5,
        "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True,
        "savefig.bbox": "tight", "savefig.dpi": 300,
    })
    fig, ax = plt.subplots(figsize=(3.15, 2.45))

    ylo, yhi = 28.85, 30.28

    for series, colour, marker, style, label in (
            (UNDRAINED, C_UND, "o", "-",
             f"no drain  (${SLOPE_UND:.2f}$/unit " + r"$\alpha$)"),
            (DRAINED,   C_DRN, "s", "--",
             f"drained  (${SLOPE_DRN:.2f}$/unit " + r"$\alpha$)")):
        xs = [a for a, _, _, _ in series]
        ys = [m for _, m, _, _ in series]
        es = [s for _, _, s, _ in series]
        ax.errorbar(xs, ys, yerr=es, marker=marker, markersize=4.0,
                    color=colour, markerfacecolor=colour,
                    markeredgecolor="white", markeredgewidth=0.6,
                    ecolor=DARK, elinewidth=0.8, capsize=2.2, capthick=0.7,
                    linestyle=style, linewidth=1.2, zorder=3, label=label)

    # the centrepiece: the one alpha where both conditions were measured
    yu = [m for a, m, _, _ in UNDRAINED if a == GAP_ALPHA][0]
    yd = [m for a, m, _, _ in DRAINED   if a == GAP_ALPHA][0]
    ax.annotate("", xy=(GAP_ALPHA, yu), xytext=(GAP_ALPHA, yd),
                arrowprops=dict(arrowstyle="<->", color=DARK, linewidth=0.9,
                                shrinkA=0, shrinkB=0), zorder=4)
    ax.text(GAP_ALPHA + 0.05, (yu + yd) / 2,
            f"{GAP_VALUE:.2f} tok/s\n$t={GAP_T:.1f}$",
            fontsize=6.5, color=DARK, va="center", ha="left", linespacing=1.3)

    # the drained series stops where the data stops
    ax.text(0.33, 29.00, "drained: not measured beyond\n" + r"$\alpha=0.25$"
            + " in this block", fontsize=6.0, color=GRAY, va="top", ha="left",
            linespacing=1.3)

    ax.set_xlabel(r"Transfer-cost coefficient $\alpha$")
    ax.set_ylabel("Decode throughput (tok/s)")
    ax.set_xlim(-0.06, 1.08)
    ax.set_ylim(ylo, yhi)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([29.0, 29.5, 30.0])

    # make the truncation unmistakable: break glyph on the y axis
    for dy in (0.0, 0.07):
        ax.plot([-0.088, -0.032], [ylo + 0.02 + dy, ylo + 0.13 + dy],
                color=DARK, linewidth=0.9, clip_on=False, zorder=6)

    ax.legend(loc="upper right", frameon=False, handletextpad=0.6,
              borderaxespad=0.2, labelspacing=0.35, handlelength=2.2)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig7_drain.{ext}")
    plt.close(fig)
    print(f"  saved {FIG_DIR/'fig7_drain.pdf'} (+ .png)")

    if PAPER_DIR.exists():
        shutil.copyfile(FIG_DIR / "fig7_drain.pdf", PAPER_DIR / "fig7_drain.pdf")
        print(f"  copied -> {PAPER_DIR/'fig7_drain.pdf'}  (not referenced by main.tex)")
    else:
        die(f"{PAPER_DIR} missing — figure not delivered")


def main():
    print("Verifying fig7 against the drain-control block ...")
    verify()
    render()
    print(f"\n  the figure's claim, within one block:")
    print(f"    alpha=0.25, undrained - drained = {GAP_VALUE:+.3f} +/- {GAP_SE:.3f} "
          f"tok/s (t = {GAP_T})")
    print(f"    undrained alpha 0 -> 0.25       =  +0.019 +/- 0.174 (t = 0.11) "
          f"— the injected delay, invisible")
    print(f"    slopes: undrained {SLOPE_UND:+.3f}, drained {SLOPE_DRN:+.3f} "
          f"tok/s per unit alpha (3.4x)")
    print(f"  NOT spliced with the cooled sweep: the blocks agree on level "
          f"(0.30%) but not on drained slope (t = -2.6).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
