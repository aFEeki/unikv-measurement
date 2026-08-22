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

No new runs: this reads the dumps already in artifacts/logit_bound/.
"""

import csv, sys
import numpy as np
from pathlib import Path

ART = Path(__file__).resolve().parents[1] / "artifacts" / "logit_bound"
OUT = Path(__file__).resolve().parents[1] / "quality_results" / "logit_margins.csv"
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
            rows.append({"prompt": pname, "arm": tag, "steps": n,
                         "max_eps": eps.max(), "min_slack": slack.min(),
                         "steps_at_risk": at_risk, "actual_flips": flips,
                         "eps_q1": q[0].mean(), "eps_q2": q[1].mean(),
                         "eps_q3": q[2].mean(), "eps_q4": q[3].mean(),
                         "eps_r_vs_step": r,
                         "ref_min_margin": m[:n].min()})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
