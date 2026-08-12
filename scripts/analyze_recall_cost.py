#!/usr/bin/env python3
"""Fit the recall-cost curve produced by run_recall_cost_sweep.py.

Model. A decode step under policy 3 attends over the resident window plus the
whole spilled tier, so the natural form is

    t_step(n_spill) = t_gpu + f * [n_spill > 0] + b * n_spill

  t_gpu  cost of the resident-only step (the GPU path, no spilled tier)
  f      fixed cost of entering the two-tier path at all (the CPU-pinned
         spilled matmuls, their graph nodes and the concat/softmax over the
         union) -- paid as soon as anything has spilled
  b      marginal cost per spilled cell

Separating f from b matters for the paper's cost model: a large f with a small
b says the recall cost is dominated by a per-step constant of the
implementation, not by the size of the retained set, which is what decides
whether the policy survives 10^4-10^5-token contexts.

Thermal correction. In the policy-3 run n_spill grows monotonically with time,
so on a passively cooled device thermal droop aliases straight onto b. The
policy-1 control run does constant work per step at the same resident size, so
its ms/step against elapsed time measures the droop alone. That drift is fitted
against WALL-CLOCK seconds (not step index, since the two runs advance through
time at different rates) and subtracted before b is refitted.
"""

import csv
import os
import statistics as st
import sys
from pathlib import Path

ROOT     = Path(__file__).resolve().parents[1]
STRESS   = ROOT / "stress_results"
FIG_DIR  = ROOT / "figures"
ART_DIR  = ROOT / "artifacts" / "recall_cost"

A_CSV   = Path(os.environ.get("UNIKV_RC_A", STRESS / "recall_cost_steps_A_p3_long.csv"))
B_CSV   = Path(os.environ.get("UNIKV_RC_B", STRESS / "recall_cost_steps_B_p1_ctrl.csv"))
E2E_CSV = Path(os.environ.get("UNIKV_RC_E2E", STRESS / "recall_cost_e2e.csv"))
ISO_CSV = Path(os.environ.get("UNIKV_RC_ISO", STRESS / "recall_cost_isochronal.csv"))
OUT_TXT = Path(os.environ.get("UNIKV_RC_OUT", ART_DIR / "recall_cost_analysis.txt"))
MAKE_FIG = os.environ.get("UNIKV_RC_FIG", "1") == "1"


def load_steps(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = [r for r in csv.DictReader(path.open()) if int(r["step"]) > 1]
    out, cum = [], 0.0
    for r in rows:
        ms = float(r["ttft_ms"])          # per-call wall clock, synchronize()d both ends
        cum += ms
        out.append({
            "step": int(r["step"]),
            "ms": ms,
            "kv_used": int(r["kv_used"]),
            "n_spill": int(r.get("n_spilled", 0) or 0),
            "elapsed_s": cum / 1000.0,
        })
    return out


def ols(xs, ys):
    """slope, intercept, se(slope), residual sd."""
    n = len(xs)
    if n < 3:
        return None
    mx, my = st.mean(xs), st.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    s = (sum(r * r for r in resid) / (n - 2)) ** 0.5
    return b, a, s / sxx ** 0.5, s


def main() -> int:
    A = load_steps(A_CSV)
    B = load_steps(B_CSV)
    if not A:
        raise SystemExit(f"missing or empty: {A_CSV}")

    L = []
    def say(s=""):
        print(s)
        L.append(s)

    say("RECALL COST vs SPILLED-SET SIZE")
    say("=" * 72)
    say(f"policy-3 run : {A_CSV.name}  ({len(A)} decode steps)")
    say(f"control run  : {B_CSV.name}  ({len(B)} decode steps)" if B else
        "control run  : ABSENT (no thermal correction)")
    say("NB the per-call CSV's `ttft_ms` column holds the per-decode-call wall")
    say("clock (llama_context::decode is bracketed by synchronize() on both ends")
    say("when UNIKV_LOG is set), not a time-to-first-token; the header name is a")
    say("legacy misnomer. Step 1 (prefill) is excluded throughout.")
    say()

    pre  = [r for r in A if r["n_spill"] == 0]
    post = [r for r in A if r["n_spill"] > 0]
    if not post:
        raise SystemExit("no spilling steps in the policy-3 run")

    say("-- raw per-step cost --")
    if pre:
        say(f"  pre-spill  (n_spill=0) : {len(pre):6d} steps, mean {st.mean([r['ms'] for r in pre]):7.2f} ms "
            f"(resident window filling, GPU-only path)")
    say(f"  spilling   (n_spill>0) : {len(post):6d} steps, "
        f"n_spill {min(r['n_spill'] for r in post)}..{max(r['n_spill'] for r in post)}")
    say()

    # binned view: honest look at the shape before any fit is imposed on it
    say("  n_spill bin        steps   mean ms   tok/s")
    nmax = max(r["n_spill"] for r in post)
    edges = [1, 256, 512, 1024, 2048, 4096, 6144, 8192, 10240, 12288, 16384]
    edges = [e for e in edges if e <= nmax + 1] + [nmax + 1]
    for lo, hi in zip(edges, edges[1:]):
        v = [r["ms"] for r in post if lo <= r["n_spill"] < hi]
        if v:
            m = st.mean(v)
            say(f"  [{lo:6d},{hi:6d})   {len(v):6d}   {m:7.2f}   {1000 / m:5.2f}")
    say()

    fit = ols([r["n_spill"] for r in post], [r["ms"] for r in post])
    b, a, seb, s = fit
    b_raw_us = b * 1000
    say("-- raw fit over spilling steps:  t_step = a + b * n_spill --")
    say(f"  a = {a:8.3f} ms      (two-tier step with an empty spilled tier)")
    say(f"  b = {b * 1000:8.4f} us/cell  (+/- {seb * 1000:.4f}, residual sd {s:.2f} ms)")
    if pre:
        say(f"  fixed cost of entering the two-tier path: "
            f"{a - st.mean([r['ms'] for r in pre]):.2f} ms/step "
            f"(vs the GPU-only pre-spill steps; part of this is the larger "
            f"resident window, see the control below)")
    say()

    drift = None
    if B:
        dfit = ols([r["elapsed_s"] for r in B], [r["ms"] for r in B])
        if dfit:
            drift, d0, dse, dsd = dfit
            span = (max(r["elapsed_s"] for r in B) - min(r["elapsed_s"] for r in B))
            say("-- thermal control (policy 1, constant work per step) --")
            say(f"  mean {st.mean([r['ms'] for r in B]):.2f} ms/step, "
                f"kv_used pinned at {st.mode([r['kv_used'] for r in B])}")
            say(f"  drift = {drift * 1000:+.4f} us per step per second of elapsed time "
                f"(+/- {dse * 1000:.4f})")
            say(f"  over the {span:.0f} s control run that is "
                f"{drift * span:+.2f} ms/step end to end "
                f"({100 * drift * span / st.mean([r['ms'] for r in B]):+.1f}%)")
            say()

            corrected = [(r["n_spill"], r["ms"] - drift * r["elapsed_s"]) for r in post]
            cfit = ols([x for x, _ in corrected], [y for _, y in corrected])
            if cfit:
                cb, ca, cseb, cs = cfit
                say("-- drift-corrected fit (control's drift-per-second removed) --")
                say(f"  a = {ca:8.3f} ms")
                say(f"  b = {cb * 1000:8.4f} us/cell  (+/- {cseb * 1000:.4f}, "
                    f"residual sd {cs:.2f} ms)")
                say(f"  thermal droop accounts for "
                    f"{100 * (b - cb) / b:.1f}% of the raw slope")
                say()
                b, a = cb, ca

    iso_fit = None
    if ISO_CSV.exists():
        irows = list(csv.DictReader(ISO_CSV.open()))
        iby: dict[int, list[float]] = {}
        for r in irows:
            iby.setdefault(int(r["target"]), []).append(float(r["ms_mean"]))
        say("-- CONTROLLED measurement (isochronal): spilled set fixed by prefill,")
        say("   short decode burst timed, randomized block with cooldowns --")
        say("   This is the fit to quote: n_spill is set before the burst instead of")
        say("   growing through it, so it is not aliased with elapsed time.")
        say(f"   {'n_spill':>8s} {'n':>2s} {'ms/step':>9s} {'sd':>6s} {'tok/s':>7s}")
        ipts = []
        for n in sorted(iby):
            v = iby[n]
            m = st.mean(v)
            sd = st.stdev(v) if len(v) > 1 else 0.0
            if n > 0:
                ipts.append((n, m))
            say(f"   {n:8d} {len(v):2d} {m:9.2f} {sd:6.2f} {1000 / m:7.2f}")
        base_iso = st.mean(iby[0]) if 0 in iby else None
        if len(ipts) > 2:
            f_all = ols([n for n, _ in ipts], [m for _, m in ipts])
            clean = [(n, m) for n, m in ipts if n != 4096]
            f_cln = ols([n for n, _ in clean], [m for _, m in clean])
            iso_fit = f_cln or f_all
            say()
            say(f"   fit, all points      : {f_all[1]:.2f} ms + {f_all[0] * 1000:.3f} us/cell "
                f"(+/- {f_all[2] * 1000:.3f}, resid sd {f_all[3]:.2f} ms)")
            if f_cln:
                say(f"   fit, excl. n=4096    : {f_cln[1]:.2f} ms + {f_cln[0] * 1000:.3f} us/cell "
                    f"(+/- {f_cln[2] * 1000:.3f}, resid sd {f_cln[3]:.2f} ms)")
                say("   (n=4096 is the one irreproducible point: 5 trials spread")
                say("    44.0-60.9 ms where every other point repeats to <0.5 ms.)")
            if base_iso:
                say(f"   policy-3 no-spill reference: {base_iso:.2f} ms/step")
                say(f"   fixed cost of entering the two-tier path: "
                    f"{iso_fit[1] - base_iso:.2f} ms/step")
                say(f"   marginal term equals the fixed term at n_spill = "
                    f"{(iso_fit[1] - base_iso) / iso_fit[0]:.0f} cells")
            say(f"   vs the single-run raw slope ({b_raw_us:.2f} us/cell): heating inflated")
            say(f"   the uncontrolled slope by ~{100 * (b_raw_us - iso_fit[0] * 1000) / b_raw_us:.0f}%.")
            say()

        iso_prefill: dict[int, list[float]] = {}
        for r in irows:
            if r["prefill_ms"]:
                iso_prefill.setdefault(int(r["prompt_tokens"]), []).append(
                    float(r["prefill_ms"]) / 1000)
        if iso_prefill:
            say("-- prefill cost when the tier is built during prompt ingestion --")
            say("   (bonus from the same runs; sizes any long-context experiment)")
            for p in sorted(iso_prefill):
                v = st.mean(iso_prefill[p])
                say(f"   prompt {p:6d} tok : {v:7.2f} s ({p / v:6.1f} tok/s)")
            say("   Growth is ~quadratic in prompt length: every ubatch attends over")
            say("   the whole spilled tier, on the CPU.")
            say()

    say("-- what the fit implies (controlled fit where available) --")
    if iso_fit:
        b, a = iso_fit[0], iso_fit[1]
    for n in (1535, 4096, 10240, 100000):
        ms = a + b * n
        marker = "  [measured range]" if n <= nmax else "  [extrapolated]"
        say(f"  n_spill = {n:7d} -> {ms:9.2f} ms/step = {1000 / ms:6.2f} tok/s{marker}")
    half = (a / b) if b > 0 else float("inf")
    say(f"  the marginal term equals the WHOLE two-tier intercept (GPU base +")
    say(f"  two-tier entry) at n_spill = {half:.0f} cells; it equals the two-tier")
    say(f"  ENTRY cost alone much earlier -- see the controlled section above.")
    say()

    if E2E_CSV.exists():
        say("-- e2e cross-check (uninstrumented wall clock, randomized block) --")
        rows = list(csv.DictReader(E2E_CSV.open()))
        by: dict[int, list[float]] = {}
        for r in rows:
            if r["tok_per_sec"]:
                by.setdefault(int(r["n_spill_final"]), []).append(float(r["tok_per_sec"]))
        say(f"  {'n_spill_final':>13s} {'n':>2s} {'mean tok/s':>10s} {'sd':>6s}")
        for n in sorted(by):
            v = by[n]
            sd = st.stdev(v) if len(v) > 1 else 0.0
            say(f"  {n:13d} {len(v):2d} {st.mean(v):10.2f} {sd:6.2f}")
        say("  (an e2e run averages over n_spill = 0..final, so these sit above the")
        say("   per-step cost at their final n_spill by construction -- they test the")
        say("   shape, not the level.)")
        say()

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT_TXT}")

    if MAKE_FIG:
        try:
            import matplotlib as mpl
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib unavailable; skipped figure")
            return 0
        mpl.rcParams.update({
            "font.family": "serif", "font.serif": ["DejaVu Serif", "Times New Roman"],
            "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8,
            "ytick.labelsize": 8, "legend.fontsize": 7,
            "axes.spines.top": False, "axes.spines.right": False,
            "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.5,
            "axes.axisbelow": True, "savefig.bbox": "tight", "savefig.dpi": 300,
        })
        ACCENT, DARK, GRAY = "#c1272d", "#222222", "#888888"
        # bin to keep the PDF small and the trend legible
        step = max(1, nmax // 220)
        bins: dict[int, list[float]] = {}
        for r in post:
            bins.setdefault(r["n_spill"] // step, []).append(r["ms"])
        xs = [k * step for k in sorted(bins)]
        ys = [st.mean(bins[k]) for k in sorted(bins)]

        fig, ax = plt.subplots(figsize=(3.5, 2.6))
        ax.plot(xs, ys, color=GRAY, linewidth=0.9, alpha=0.85,
                label="single long run (heat-confounded)")
        if ISO_CSV.exists():
            iby2: dict[int, list[float]] = {}
            for r in csv.DictReader(ISO_CSV.open()):
                iby2.setdefault(int(r["target"]), []).append(float(r["ms_mean"]))
            ix = sorted(n for n in iby2 if n > 0)
            iy = [st.mean(iby2[n]) for n in ix]
            ie = [st.stdev(iby2[n]) if len(iby2[n]) > 1 else 0.0 for n in ix]
            ax.errorbar(ix, iy, yerr=ie, marker="o", markersize=4.5, color=ACCENT,
                        markerfacecolor=ACCENT, markeredgecolor="white",
                        markeredgewidth=0.7, ecolor=DARK, elinewidth=0.9,
                        capsize=2.5, capthick=0.8, linestyle="none", zorder=4,
                        label="controlled (fixed $n_{\\mathrm{spill}}$)")
            if 0 in iby2:
                ax.axhline(st.mean(iby2[0]), color=DARK, linewidth=0.9, linestyle=":",
                           label=f"no spill ({st.mean(iby2[0]):.1f} ms)")
        ax.plot([0, nmax], [a, a + b * nmax], color=DARK, linestyle="--",
                linewidth=0.9, zorder=3,
                label=f"fit: {a:.1f} + {b * 1000:.2f}$\\,\\mu$s$\\,\\cdot n$")
        ax.set_xlabel("Spilled cells $n_{\\mathrm{spill}}$")
        ax.set_ylabel("Decode step time (ms)")
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper left", frameon=False)
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        # NOT fig5_recall_cost: that name now belongs to the two-series figure
        # built by scripts/make_fig5_recall_cost.py from the B2 Block-1 data,
        # which is the one the measurement paper includes. This single-series
        # plot is retained for the older CPU-pinned-only analysis only.
        fig.savefig(FIG_DIR / "fig5_recall_cost_cpuonly_superseded.pdf")
        fig.savefig(FIG_DIR / "fig5_recall_cost_cpuonly_superseded.png")
        plt.close(fig)
        print(f"wrote {FIG_DIR / 'fig5_recall_cost_cpuonly_superseded.pdf'} (+ .png)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
