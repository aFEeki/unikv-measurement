#!/usr/bin/env python3
"""M2 follow-up — flip margins SPLIT BY TIER, and whether eps grows with step.

Two questions the pooled numbers could not answer:

1. Is device-visible actually the SAFER tier, or merely the less-perturbed one?
   Pooling min-margin across tiers cannot tell those apart. If device-visible's
   worst margin is also ~0.005 then both tiers sit equally close to a flip and
   the honest move is to downgrade both, not to recommend one.

   Reported per tier: min(m - 2*eps) and the count of steps with m <= 2*eps,
   where m is the REFERENCE's top1-top2 margin at that step and eps is the arm's
   max |delta logit| at that step. m <= 2*eps is the conservative flip condition:
   a perturbation of size eps applied adversarially to both the winner and the
   runner-up closes a margin of 2*eps.

2. Does eps grow with step index — i.e. with how long the spilled tier has been
   accumulating? Flat points at a fixed reassociation difference. Growing points
   at accumulation and predicts failure at longer horizons than 512.

3. Did the perturbation actually move the reference's top-two margin? max
   |delta logit| over the vocabulary bounds the shift; the shift itself is
   (arm[i] - arm[j]) - (ref[i] - ref[j]) for the reference's top two tokens i
   and j. Reported up to the first step where the arm's argmax leaves the
   reference's (all steps if it never does), with the margin and shift at that
   step. After it the two runs condition on different tokens and stop being
   comparable.

No new runs: this reads the dumps already in artifacts/logit_bound/.
"""

import csv, os, sys
import numpy as np
from pathlib import Path

# UNIKV_LOGIT_TAG reads another model's dumps (see run_logit_bound.py).
_T  = os.environ.get("UNIKV_LOGIT_TAG", "")
ART = Path(__file__).resolve().parents[1] / "artifacts" / (f"logit_bound_{_T}" if _T else "logit_bound")
OUT = Path(__file__).resolve().parents[1] / "quality_results" / (f"logit_margins_{_T}.csv" if _T else "logit_margins.csv")
NV  = 128256
ARMS = ["p3_cpu_c1024", "p3_dev_c1024", "p4_h2o_c1024"]


def load(p):
    a = np.fromfile(p, dtype=np.float32)
    return a.reshape(-1, NV)


def main():
    rows = []
    for pname in ("passkey", "prose"):
        ref = load(ART / f"logits_{pname}_ref_c8192.bin")
        # reference top1/top2 margin per step
        part = np.partition(ref, -2, axis=1)
        m = (part[:, -1] - part[:, -2]).astype(np.float64)
        top1 = ref.argmax(axis=1)
        top2 = np.argsort(ref, axis=1)[:, -2]
        print(f"\n=== {pname} ===  {ref.shape[0]} steps")
        print(f"  reference top1-top2 margin: min {m.min():.6g}  median {np.median(m):.4g}")
        print(f"  {'arm':14} {'max eps':>10} {'min(m-2eps)':>13} {'steps m<=2eps':>14} "
              f"{'actual flips':>13}")
        for tag in ARMS:
            arm = load(ART / f"logits_{pname}_{tag}.bin")
            n = min(ref.shape[0], arm.shape[0])
            d = np.abs(ref[:n] - arm[:n])
            eps = d.max(axis=1).astype(np.float64)
            slack = m[:n] - 2.0 * eps
            at_risk = int((slack <= 0).sum())
            # actual flips: did the arm's argmax move off the reference's?
            flips = int((arm[:n].argmax(axis=1) != top1[:n]).sum())
            print(f"  {tag:14} {eps.max():10.5g} {slack.min():13.6g} {at_risk:14d} "
                  f"{flips:13d}")

            # eps vs step index, in quarters
            q = np.array_split(eps, 4)
            qs = "  ".join(f"{x.mean():.4g}" for x in q)
            r = np.corrcoef(np.arange(n), eps)[0, 1]
            print(f"                 eps by quarter: {qs}   pearson r vs step = {r:+.3f}")
            # direct shift of the reference's top-two margin, up to the first flip
            idx = np.arange(n)
            arm_m = (arm[idx, top1[:n]] - arm[idx, top2[:n]]).astype(np.float64)
            shift = arm_m - m[:n]
            off = np.flatnonzero(arm[:n].argmax(axis=1) != top1[:n])
            ff = int(off[0]) if off.size else n
            pre = slice(0, ff)
            qp = np.array_split(eps[pre], 4) if ff >= 4 else [np.array([np.nan])] * 4
            print(f"                 first flip {ff if off.size else 'none'}; before it: "
                  f"max eps {eps[pre].max() if ff else float('nan'):.4g}, "
                  f"max |margin shift| {np.abs(shift[pre]).max() if ff else float('nan'):.4g}, "
                  f"min own margin {arm_m[pre].min() if ff else float('nan'):.4g}")
            rows.append({"prompt": pname, "arm": tag, "steps": n,
                         "max_eps": eps.max(), "min_slack": slack.min(),
                         "steps_at_risk": at_risk, "actual_flips": flips,
                         "eps_q1": q[0].mean(), "eps_q2": q[1].mean(),
                         "eps_q3": q[2].mean(), "eps_q4": q[3].mean(),
                         "eps_r_vs_step": r,
                         "ref_min_margin": m[:n].min(),
                         "first_flip": ff if off.size else "",
                         "max_eps_pre_flip": eps[pre].max() if ff else "",
                         "max_margin_shift": np.abs(shift[pre]).max() if ff else "",
                         "min_arm_margin": arm_m[pre].min() if ff else "",
                         "eps_pre_q1": qp[0].mean(), "eps_pre_q4": qp[3].mean(),
                         "flip_ref_margin": m[ff] if off.size else "",
                         "flip_margin_shift": shift[ff] if off.size else ""})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
