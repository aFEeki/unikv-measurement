#!/usr/bin/env python3
"""Count the ggml scheduler's graph splits in the two-tier attention path.

This is a STRUCTURAL census, not a timing measurement. The split count is a
property of the graph the scheduler builds for a given (model, tier placement),
so it is deterministic and does not need the cooled randomized-block protocol
of Section 3: no cooldowns, no repeats, no randomization, and a warm machine is
fine. Nothing here may be quoted as a rate.

What it answers: Section 4 says the two-tier attention path is built once per
layer, and that the CPU-pinned configuration additionally pins two of its nodes
per layer to the host backend. That predicts a split count linear in layer
count on the CPU-pinned arm and a model-independent one on the device-visible
arm. This counts them.

Method: GGML_SCHED_DEBUG=1 makes ggml_backend_sched_print_assignments emit one
"## SPLIT #<n>: <backend>" line per split on every graph build. A build is
delimited by the counter resetting to 0. Each run produces builds of two kinds:
before the first demotion the graph is single-tier, after it the graph spans
both tiers. We report both.

  python3 scripts/run_split_census.py
"""

import csv
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT      = Path(__file__).resolve().parents[1]
LLAMA_DIR = ROOT / "llama.cpp"
BIN       = LLAMA_DIR / "build-m4pro-metal" / "bin" / "llama-completion"
PROMPT    = ROOT / "artifacts" / "b2_cooled" / "prompts" / "prompt_1536tok.txt"
OUT_CSV   = ROOT / "stress_results" / "split_census.csv"
LOGS_DIR  = ROOT / "artifacts" / "split_census" / "logs"

# Same four models and the same kernel setting as the Section 4 blocks.
MODELS = [
    ("Llama 3.1 8B",  "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"),
    ("Qwen2.5 7B",    "Qwen2.5-7B-Instruct-Q4_K_M.gguf"),
    ("Llama 3.2 3B",  "Llama-3.2-3B-Instruct-Q4_K_M.gguf"),
    ("Llama 3.2 1B",  "Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
]

CTX, BATCH, UBATCH, GEN, SEED, THREADS = 512, 256, 256, 2, 1234, 8


def parse_builds(text):
    """[(n_splits, {backend: count}), ...] in graph-build order."""
    builds, cur = [], []
    for ln in text.splitlines():
        m = re.match(r"## SPLIT #(\d+): (\S+)", ln)
        if not m:
            continue
        if int(m.group(1)) == 0 and cur:
            builds.append(cur)
            cur = []
        cur.append(m.group(2))
    if cur:
        builds.append(cur)
    return [(len(b), dict(Counter(b))) for b in builds]


def run(label, gguf, dev):
    model = LLAMA_DIR / "models" / gguf
    if not model.exists():
        sys.exit(f"missing model: {model}")

    env = os.environ.copy()
    env.update({
        "GGML_SCHED_DEBUG": "1", "UNIKV_POLICY": "3", "UNIKV_ALPHA": "0",
        "UNIKV_SPILL_DEV": str(dev), "UNIKV_SPILL_CAP": "4096",
    })
    for k in ("UNIKV_LOG", "UNIKV_E2E_LOG", "UNIKV_H2O_TRACE"):
        env.pop(k, None)

    args = [
        str(BIN), "-m", str(model), "-f", str(PROMPT),
        "-n", str(GEN), "-c", str(CTX), "-b", str(BATCH), "-ub", str(UBATCH),
        "-ngl", "99", "-t", str(THREADS), "-fa", "off", "-fit", "off",
        "--temp", "0", "--seed", str(SEED), "--ignore-eos", "--no-warmup",
        "--simple-io", "--no-display-prompt", "-no-cnv", "-v",
    ]
    d = subprocess.run(args, cwd=LLAMA_DIR, env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True, errors="replace")

    tag = f"{gguf.split('-Q4')[0]}_dev{dev}"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    (LOGS_DIR / f"splits_{tag}.txt").write_text(d.stdout)

    builds = parse_builds(d.stdout)
    n_layer = int(re.search(r"n_layer\s*=\s*(\d+)", d.stdout).group(1))
    spills  = len(re.findall(r"decode: unikv: spilled", d.stdout))
    fa      = re.search(r"flash_attn\s*=\s*(\w+)", d.stdout)
    on_dev  = bool(re.search(r"spilled tier -> .*device-visible", d.stdout))

    # The first build predates the first demotion and is single-tier; the last
    # follows it and is two-tier. Comparing first to last rather than counting
    # distinct sizes keeps the case where the two are EQUAL, which is the
    # device-visible result and would otherwise read as missing data. It is
    # only meaningful if the run actually spilled, so that is asserted.
    if not spills:
        sys.exit(f"{label} dev={dev}: no demotion occurred, so no two-tier "
                 f"graph was ever built; the census would be vacuous")

    pre, post = builds[0], builds[-1]

    return {
        "model": label, "n_layer": n_layer, "spill_dev": dev,
        "tier_device_visible": on_dev, "flash_attn": fa.group(1) if fa else "UNPARSED",
        "rc": d.returncode, "spill_events": spills, "graph_builds": len(builds),
        "splits_single_tier": pre[0], "splits_two_tier": post[0],
        "splits_two_tier_cpu": post[1].get("CPU", 0),
        "splits_two_tier_metal": post[1].get("MTL0", 0),
        "extra_splits": post[0] - pre[0],
        "extra_per_layer": round((post[0] - pre[0]) / n_layer, 4),

    }


def main():
    print(f"split census: C={CTX}, ubatch={UBATCH}, flash attention off, "
          f"policy 3, {GEN} decode tokens")
    print("structural count -- no cooled protocol, no repeats\n")

    rows = []
    for label, gguf in MODELS:
        for dev in (0, 1):
            arm = "device-visible" if dev else "CPU-pinned"
            print(f"  {label:14s} {arm:15s} ...", end="", flush=True)
            r = run(label, gguf, dev)
            rows.append(r)
            print(f" rc={r['rc']} L={r['n_layer']:2d} "
                  f"single-tier={r['splits_single_tier']} "
                  f"two-tier={r['splits_two_tier']} "
                  f"(+{r['extra_splits']}, {r['extra_per_layer']}/layer)")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("\n" + "=" * 78)
    print(f"{'model':14s} {'L':>3s} {'arm':15s} {'1-tier':>7s} {'2-tier':>7s} "
          f"{'extra':>6s} {'/layer':>7s}")
    print("-" * 78)
    for r in rows:
        arm = "device-visible" if r["spill_dev"] else "CPU-pinned"
        print(f"{r['model']:14s} {r['n_layer']:3d} {arm:15s} "
              f"{r['splits_single_tier']:7d} {r['splits_two_tier']:7d} "
              f"{r['extra_splits']:6d} {r['extra_per_layer']:7.2f}")
    print("=" * 78)
    print(f"\nwrote {OUT_CSV.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
