#!/usr/bin/env python3
"""Figures from an earlier draft (fig1_baseline, fig3_tokens, fig4_alpha); the paper does not use them.

Regenerates the three figures from their locked values. Each value is asserted
against its source CSV, and the script stops, naming the CSV, on any
disagreement. Single-column (3.5 in), grayscale-safe, serif matplotlib style.

  fig1_baseline.pdf/.png   baseline decode throughput against context (e2e)
  fig3_tokens.pdf/.png     tokens generated per policy before halting or at the
                           decode budget (a count, not a rate)
  fig4_alpha.pdf/.png      policy-3 alpha sweep, e2e tok/s, error bars +/- 1 SD

fig3_tokens also needs an A100 baseline CSV that is not in this archive.
"""
from __future__ import annotations

import csv
import statistics as st
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT       = Path(__file__).resolve().parents[1]
FIG1_CSV   = ROOT / "artifacts" / "m4pro_baseline" / "e2e" / "baseline_m4pro_e2e.csv"
STRESS_DIR = ROOT / "stress_results"
ALPHA_DIR  = ROOT / "alpha_results"
FIG_DIR    = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TOL = 0.005  # values are printed to 2 dp; agree within half a hundredth

# ----------------------------------------------------------------------------
# locked values; the figures conform to these
# ----------------------------------------------------------------------------
FIG1_LOCK = {4096: 40.40, 8192: 36.50, 16384: 29.88, 32768: 20.72}  # ctx -> e2e tok/s

# Fig 3 shows the rolling window (policy 1) alongside the halting baseline. The
# window also completes the budget, so the difference in token count belongs to
# the baseline's error path, not to retention. The decode budget is drawn as a
# reference line, so the figure reads as completes against halts.
FIG3_CSV     = STRESS_DIR / "r3_policy_compare_cooled_master.csv"
FIG3_A100_CSV = ROOT / "a100_benchmark(server version)" / "baseline_a100.csv"
FIG3_BUDGET  = 2048
FIG3_LOCK = [
    ("M4 Pro\nbaseline",   512,  "p0_c1024"),   # stock: errors at cache-full
    ("A100\nbaseline",     511,  "a100"),       # same error path on CUDA
    ("M4 Pro\nrolling\nwindow", 2048, "p1_c1024"),  # lossy but completes
    ("M4 Pro\nUniKV",      2048, "p3_c1024"),   # lossless and completes
]

# Fig 4: the cooled policy-3 alpha sweep. Measured e2e wall-clock tok/s,
# +/- 1 SD (sample, ddof=1), from the cooled master CSV. alpha = 0 runs without
# the per-step drain and alpha > 0 runs with it, so the two differ by the drain
# as well as the injected delay (Appendix A of the paper).
FIG4_CSV    = ALPHA_DIR / "p3_alpha_sweep_cooled_master.csv"
FIG4_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
FIG4_MEAN   = {0.0: 29.67, 0.25: 29.19, 0.5: 28.67, 0.75: 28.50, 1.0: 28.25, 1.5: 27.59, 2.0: 27.01}
FIG4_STD    = {0.0: 0.11, 0.25: 0.24, 0.5: 0.14, 0.75: 0.09, 1.0: 0.03, 1.5: 0.04, 2.0: 0.07}

# ---- style -----------------------------------------------------------------
ACCENT = "#c1272d"   # deep red accent
DARK   = "#222222"
GRAY   = "#888888"

mpl.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["DejaVu Serif", "Times New Roman", "Times"],
    "font.size":        9,
    "axes.labelsize":   9,
    "axes.titlesize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.edgecolor":    DARK,
    "axes.linewidth":    0.8,
    "grid.color":        "#dddddd",
    "grid.linestyle":    "-",
    "grid.linewidth":    0.5,
    "axes.grid":         True,
    "axes.grid.axis":    "y",
    "axes.axisbelow":    True,
    "lines.linewidth":   1.6,
    "savefig.bbox":      "tight",
    "savefig.dpi":       300,
})
FIGSIZE = (3.5, 2.6)


def save(fig, name: str) -> None:
    fig.savefig(FIG_DIR / f"{name}.pdf")
    fig.savefig(FIG_DIR / f"{name}.png")
    print(f"  saved {name}.pdf + {name}.png")


def die(msg: str) -> None:
    raise SystemExit(f"STOP (provenance mismatch): {msg}")


# ----------------------------------------------------------------------------
# Provenance checks: assert each source CSV agrees with the locked values.
# ----------------------------------------------------------------------------
def verify_fig1() -> None:
    rows = list(csv.DictReader(FIG1_CSV.open()))
    got = {int(r["context_length_tokens"]): float(r["e2e_tok_per_sec"]) for r in rows}
    for ctx, lock in FIG1_LOCK.items():
        if ctx not in got:
            die(f"{FIG1_CSV.name}: ctx={ctx} missing")
        if abs(got[ctx] - lock) > TOL:
            die(f"{FIG1_CSV.name}: ctx={ctx} has {got[ctx]}, locked {lock}")
    print(f"  [ok] Fig1 <- {FIG1_CSV.name}  " +
          "  ".join(f"{c}={FIG1_LOCK[c]}" for c in sorted(FIG1_LOCK)))


def verify_fig3() -> None:
    """All three M4 bars come from the one cooled, flash-attention-off block, so
    the arms are protocol-matched to each other (the superseded provenance mixed
    a flash-attention-ON policy-1 run with flash-attention-off policy-0/3 runs)."""
    if not FIG3_CSV.exists():
        die(f"{FIG3_CSV.name}: missing — run scripts/run_policy_compare_cooled.py")
    rows = list(csv.DictReader(FIG3_CSV.open()))
    for label, lock, arm in FIG3_LOCK:
        if arm == "a100":
            a100 = list(csv.DictReader(FIG3_A100_CSV.open()))
            hit = [r for r in a100 if int(r["context_length_tokens"]) == 1024]
            if not hit:
                die(f"{FIG3_A100_CSV.name}: no ctx=1024 row")
            got = int(hit[0]["generated_tokens_reported"])
            if got != lock:
                die(f"{FIG3_A100_CSV.name}: ctx=1024 generated {got}, locked {lock}")
            continue
        got = {int(r["decode_tokens"]) for r in rows if r["arm"] == arm}
        if not got:
            die(f"{FIG3_CSV.name}: no rows for arm {arm}")
        if got != {lock}:
            die(f"{FIG3_CSV.name}: arm {arm} decoded {sorted(got)}, locked {lock}")
        fa = {r["flash_attn"] for r in rows if r["arm"] == arm}
        if fa != {"disabled"}:
            die(f"{FIG3_CSV.name}: arm {arm} flash_attn={fa}, must be disabled")
    print(f"  [ok] Fig3 <- {FIG3_CSV.name} (p0=512, p1=2048, p3=2048, all fa off) "
          f"+ {FIG3_A100_CSV.name} ctx=1024 -> 511")


def verify_fig4() -> None:
    if not FIG4_CSV.exists():
        die(f"{FIG4_CSV.name}: missing")
    rows = list(csv.DictReader(FIG4_CSV.open()))
    for a in FIG4_ALPHAS:
        vals = [float(r["tok_per_sec"]) for r in rows if float(r["alpha"]) == a]
        if len(vals) < 2:
            die(f"{FIG4_CSV.name}: alpha={a:g} has {len(vals)} runs")
        m, s = st.mean(vals), st.stdev(vals)
        if abs(m - FIG4_MEAN[a]) > TOL:
            die(f"{FIG4_CSV.name}: alpha={a:g} mean {m:.2f}, locked {FIG4_MEAN[a]}")
        if abs(s - FIG4_STD[a]) > TOL:
            die(f"{FIG4_CSV.name}: alpha={a:g} std {s:.2f}, locked {FIG4_STD[a]}")
    print(f"  [ok] Fig4 <- {FIG4_CSV.name} (policy-3 cooled; mean+/-1 STD, ddof=1)")


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def fig1_baseline() -> None:
    ctx = sorted(FIG1_LOCK)
    tps = [FIG1_LOCK[c] for c in ctx]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(ctx, tps, marker="o", color=ACCENT, markersize=5,
            markerfacecolor=ACCENT, markeredgecolor="white", markeredgewidth=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ctx)
    ax.set_xticklabels([f"{c // 1024}K" for c in ctx])
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Decode throughput (tok/s)")
    ax.set_ylim(bottom=0, top=max(tps) * 1.15)
    ax.margins(x=0.08)
    save(fig, "fig1_baseline")
    plt.close(fig)


def fig3_tokens() -> None:
    labels = [l for l, _, _ in FIG3_LOCK]
    values = [v for _, v, _ in FIG3_LOCK]
    colors = [DARK, GRAY, "#6f6f6f", ACCENT]
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    bars = ax.bar(labels, values, color=colors, width=0.62, edgecolor="none")
    ax.set_ylabel("Tokens generated")
    ax.set_ylim(0, FIG3_BUDGET * 1.32)
    ax.tick_params(axis="x", length=0)
    ax.axhline(FIG3_BUDGET, color=DARK, linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(-0.42, FIG3_BUDGET * 1.03, "decode budget",
            ha="left", va="bottom", fontsize=7, color=DARK)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + FIG3_BUDGET * 0.02,
                f"{v:,}", ha="center", va="bottom", fontsize=8, color=DARK)
    # the split the figure is actually about: halts vs completes
    ax.text(0.5, FIG3_BUDGET * 0.42, "halts at\ncache-full", ha="center",
            va="center", fontsize=7, color=GRAY)
    save(fig, "fig3_tokens")
    plt.close(fig)


def fig4_alpha() -> None:
    pos = [x for x in FIG4_ALPHAS if x > 0]          # blocking transfer-cost trend
    mp  = [FIG4_MEAN[x] for x in pos]
    sp  = [FIG4_STD[x]  for x in pos]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    # alpha=1.0 PCIe-equivalent guide (a marker only, not a throughput comparison)
    ax.axvline(1.0, color=DARK, linestyle="--", linewidth=0.8, alpha=0.5)
    ax.text(1.06, 0.6, "PCIe 4.0 x16 equiv", rotation=90, ha="left", va="bottom",
            fontsize=7, color=DARK)

    # alpha>0: connected trend with +/-1 STD bars
    ax.errorbar(pos, mp, yerr=sp, marker="o", markersize=4.5, color=ACCENT,
                markerfacecolor=ACCENT, markeredgecolor="white", markeredgewidth=0.7,
                ecolor=GRAY, elinewidth=1.0, capsize=2.5, capthick=0.9,
                linewidth=1.5, zorder=3, label=r"$\alpha{>}0$ (transfer trend)")

    # alpha=0: plotted separately and not joined to the trend line, because it
    # has no pipeline drain and the alpha > 0 points do (Appendix A)
    ax.errorbar([0.0], [FIG4_MEAN[0.0]], yerr=[FIG4_STD[0.0]], marker="D",
                markersize=5.5, color=DARK, markerfacecolor="white",
                markeredgecolor=DARK, markeredgewidth=1.2, ecolor=GRAY,
                elinewidth=1.0, capsize=2.5, capthick=0.9, linestyle="none",
                zorder=4, label=r"$\alpha{=}0$ (unified anchor)")

    ax.set_xlabel(r"Transfer cost coefficient $\alpha$")
    ax.set_ylabel("UniKV decode throughput (tok/s)")
    ax.set_ylim(0, 32)          # y from 0; do not clip to exaggerate slope
    ax.set_xlim(-0.12, 2.12)
    ax.set_xticks(FIG4_ALPHAS)
    ax.legend(loc="lower left", bbox_to_anchor=(0.01, 0.03), frameon=False, fontsize=7)
    save(fig, "fig4_alpha")
    plt.close(fig)


def main() -> None:
    print("Verifying provenance against locked values...")
    verify_fig1()
    verify_fig3()
    verify_fig4()
    print(f"\nRendering -> {FIG_DIR}")
    fig1_baseline()
    fig3_tokens()
    fig4_alpha()
    print("\nData-to-figure mapping:")
    print(f"  fig1_baseline : {FIG1_CSV.name} -> ctx{{4096,8192,16384,32768}} e2e "
          f"= {[FIG1_LOCK[c] for c in sorted(FIG1_LOCK)]} tok/s (no error bars)")
    print(f"  fig3_tokens   : {FIG3_CSV.name} p0_c1024=512, p1_c1024=2048, "
          f"p3_c1024=2048 (all flash-attn off) + A100 ctx=1024=511 -> bars "
          f"{[v for _, v, _ in FIG3_LOCK]} tokens, budget line at {FIG3_BUDGET}")
    print(f"  fig4_alpha    : {FIG4_CSV.name} -> mean +/- 1 STD per alpha={FIG4_ALPHAS} "
          f"(policy-3 cooled; alpha=0 anchor, alpha>0 trend)")
    print("Done.")


if __name__ == "__main__":
    main()
