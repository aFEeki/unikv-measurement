#!/usr/bin/env python3
"""The demotion copy's share of the fixed charge, from the copy-gate blocks.

Reads, per model:
  stress_results/copy_gate_<m>.csv           run_copy_gate.py: each tier's
                                              no-spill reference and target 128
                                              at UNIKV_COPY_REPS 0, 1, 2
  stress_results/resident_baseline_<m>.csv   the resident slope, to price the
                                              568 cells a plateau run holds
                                              beyond its reference (Table 2)

Per tier:
  copy by difference   reps 1 - reps 0 (the copy's full cost) and reps 2 - reps 1
                       (one more, identical copy); standard errors from the
                       three runs per cell
  copy directly        copy_us, the timed copy loop, mean over runs
  fixed charge         reps 1 - reference - 568 x resident slope, the Table 2
                       definition, re-measured inside this block
  without the copy     reps 0 - reference - 568 x resident slope
  share                copy by difference / fixed charge, and by the timer. On
                       the host tier the arm's heavy right tail makes the
                       difference too noisy to resolve; the timer is not affected

No new runs.
"""

import csv, math, statistics as st, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "stress_results"
MODELS = [  # tag, name, layers, n_embd_v_gqa
    ("llama", "Llama 3.1 8B", 32, 1024),
    ("qwen",  "Qwen2.5 7B",   28, 512),
    ("l3b",   "Llama 3.2 3B", 28, 1024),
    ("l1b",   "Llama 3.2 1B", 16, 512),
]
EXTRA_CELLS = 568
TARGET = "128"


def resident_slope(tag):
    r = [x for x in csv.DictReader((ROOT / f"resident_baseline_{tag}.csv").open())
         if x["rc"] == "0" and x["arm"] == "p0_resident" and int(x["target"]) >= 512]
    xs = [float(x["context_mean"]) for x in r]; ys = [float(x["ms_mean"]) for x in r]
    n = len(xs); mx, my = st.mean(xs), st.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    s2 = sum((y - (my + b * (x - mx))) ** 2 for x, y in zip(xs, ys)) / (n - 2)
    return b, math.sqrt(s2 / sxx)


def mse(v):
    return st.mean(v), (st.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else float("nan"))


def diff(a, b):
    (ma, sa), (mb, sb) = mse(a), mse(b)
    return ma - mb, math.hypot(sa, sb)


def main():
    for tag, name, L, nv in MODELS:
        p = ROOT / f"copy_gate_{tag}.csv"
        if not p.exists():
            print(f"\n{name}: no {p.name}"); continue
        rows = [x for x in csv.DictReader(p.open()) if x["rc"] == "0" and x["ms_mean"]]
        if any(x["flash_attn"] != "disabled" for x in rows):
            raise SystemExit(f"STOP: flash attention on in {p.name}")
        b, sb = resident_slope(tag)
        corr, se_corr = EXTRA_CELLS * b, EXTRA_CELLS * sb
        calls = 2 * nv * L                     # V element calls per step, get + set
        print(f"\n{name}  (L = {L}, {calls:,} V-element calls per demotion)")
        res = {}
        for dev, tier in (("0", "host"), ("1", "device")):
            g = lambda t, r, k="ms_mean": [float(x[k]) for x in rows if x["spill_dev"] == dev
                                           and x["target"] == t and x["reps"] == r]
            ref, r0, r1, r2 = g("0", "1"), g(TARGET, "0"), g(TARGET, "1"), g(TARGET, "2")
            if not (len(ref) and len(r0) and len(r1) and len(r2)):
                print(f"  {tier}: incomplete"); continue
            c10, s10 = diff(r1, r0)
            c21, s21 = diff(r2, r1)
            cu1, cu2 = st.mean(g(TARGET, "1", "copy_us_mean")), st.mean(g(TARGET, "2", "copy_us_mean"))
            j, sj = diff(r1, ref)
            f, sf = j - corr, math.hypot(sj, se_corr)
            j0, sj0 = diff(r0, ref)
            f0, sf0 = j0 - corr, math.hypot(sj0, se_corr)
            res[tier] = (f, f0)
            print(f"  {tier:6s} ref {st.mean(ref):.3f}  reps 0/1/2 {st.mean(r0):.3f} / "
                  f"{st.mean(r1):.3f} / {st.mean(r2):.3f} ms  (n = {len(ref)}/{len(r0)}/{len(r1)}/{len(r2)})")
            print(f"         copy by difference {c10:.3f} +/- {s10:.3f} ms (1-0), "
                  f"{c21:.3f} +/- {s21:.3f} (2-1); timed {cu1/1000:.3f} ms at reps 1, "
                  f"{cu2/1000:.3f} at reps 2; {1e6*c10/calls:.1f} ns per call")
            print(f"         fixed charge {f:.2f} +/- {sf:.2f} ms, without the copy "
                  f"{f0:.2f} +/- {sf0:.2f}; copy share {100*c10/f:.0f}%; "
                  f"per layer {f/L:.3f} -> {f0/L:.3f} ms")
            print(f"         by the timer: share {100*cu1/1000/f:.1f}%, without the copy "
                  f"{f - cu1/1000:.2f} ms, {(f - cu1/1000)/L:.3f} per layer; "
                  f"timer minus difference {cu1/1000 - c10:+.3f} ms")
        if "host" in res and "device" in res:
            (fh, fh0), (fd, fd0) = res["host"], res["device"]
            print(f"  device tier removes {100*(1-fd/fh):.1f}% of the host fixed charge, "
                  f"{100*(1-fd0/fh0):.1f}% once the copy is taken out of both")
    return 0


if __name__ == "__main__":
    sys.exit(main())
