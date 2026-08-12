#!/usr/bin/env python3
"""Recall cost at CONTROLLED spilled-set size (the clean version of the curve).

Why this exists. In a single long decode run the spilled set grows
monotonically with elapsed time, so on a passively cooled device n_spill and
thermal state are perfectly confounded: the marginal cost per spilled cell
measured across one run is part recall and part heating, and no within-run
regression can separate them (the policy-1 control in
run_recall_cost_sweep.py measures the droop of a DIFFERENT, GPU-only workload,
and its droop saturates on a different timescale than the CPU-heavy policy-3
run, so subtracting it is an approximation at best).

This harness decouples the two. For each target spilled-set size it builds a
prompt of (C + target) tokens, so the spilled tier is established during
PREFILL, then decodes a short burst of 128 tokens and times only those. Across
the burst n_spill moves by 128 out of the target, so each point is a per-step
cost at an essentially fixed spilled-set size, measured inside a few seconds
rather than across ten minutes. Points are run in randomized-block order with
cooldowns, so thermal state is decorrelated from n_spill instead of aliased
onto it.

Protocol: policy 3, C=1024, -fa off, -fit off, greedy (temp 0), seed 123,
--ignore-eos, -b/-ub 512, alpha 0, UNIKV_LOG on (per-call wall clock is
bracketed by synchronize() on both ends). flash_attn is read back from each
run's own log.
"""

import csv
import datetime
import os
import random
import re
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

ROOT           = Path(__file__).resolve().parents[1]
LLAMA_DIR      = ROOT / "llama.cpp"
BIN_DIR        = LLAMA_DIR / "build-m4pro-metal" / "bin"
COMPLETION_BIN = BIN_DIR / "llama-completion"
TOKENIZE_BIN   = BIN_DIR / "llama-tokenize"
MODEL_PATH     = LLAMA_DIR / "models" / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

RESULTS_DIR = ROOT / "stress_results"
ART_DIR     = ROOT / "artifacts" / "recall_cost"
PROMPTS_DIR = ART_DIR / "prompts"
LOGS_DIR    = ART_DIR / "iso_logs"

CTX         = 1024
BURST       = int(os.environ.get("UNIKV_ISO_BURST", "128"))
WARMUP      = int(os.environ.get("UNIKV_ISO_WARMUP", "32"))
THREADS     = 10
GPU_LAYERS  = 999
BATCH       = 512
UBATCH      = 512
SEED        = 123
FIXED_TOKEN = " token"
CEILING     = 58.0
SHUFFLE_SEED = 20260804

# target spilled-set sizes; prompt = CTX + target
TARGETS = [int(x) for x in os.environ.get(
    "UNIKV_ISO_TARGETS", "0,512,1024,2048,4096,8192").split(",")]
TRIALS     = int(os.environ.get("UNIKV_ISO_TRIALS", "2"))
COOLDOWN_S = int(os.environ.get("UNIKV_ISO_COOLDOWN", "150"))


def count_tokens(path: Path) -> int:
    out = subprocess.run(
        [str(TOKENIZE_BIN), "-m", str(MODEL_PATH), "-f", str(path),
         "--show-count", "--log-disable"],
        cwd=LLAMA_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, errors="replace", check=True).stdout
    m = re.search(r"Total number of tokens:\s*(\d+)", out)
    if not m:
        raise RuntimeError(f"token count parse failed:\n{out}")
    return int(m.group(1))


def ensure_prompt(n_tokens: int) -> Path:
    path = PROMPTS_DIR / f"iso_prompt_{n_tokens}tok.txt"
    if not path.exists():
        path.write_text(FIXED_TOKEN * max(n_tokens - 1, 0))
        got = count_tokens(path)
        if got != n_tokens:
            raise RuntimeError(f"prompt token mismatch: want {n_tokens} got {got}")
    return path


def iso(ts: float) -> str:
    return datetime.datetime.fromtimestamp(ts).isoformat(timespec="seconds")


def run_point(target: int, trial: int) -> dict:
    prompt_tokens = CTX + target if target > 0 else CTX // 2
    prompt = ensure_prompt(prompt_tokens)
    tag = f"iso_n{target}_t{trial}"
    step_csv = LOGS_DIR / f"step_{tag}.csv"
    log_txt  = LOGS_DIR / f"gen_{tag}.txt"
    step_csv.unlink(missing_ok=True)

    args = [
        str(COMPLETION_BIN), "-m", str(MODEL_PATH), "-f", str(prompt),
        "-n", str(BURST), "-c", str(CTX), "-b", str(BATCH), "-ub", str(UBATCH),
        "-ngl", str(GPU_LAYERS), "-t", str(THREADS), "-fa", "off", "-fit", "off",
        "--temp", "0", "--seed", str(SEED), "--ignore-eos", "--no-warmup",
        "--simple-io", "--no-display-prompt", "-no-cnv",
    ]
    env = os.environ.copy()
    env.update({
        "UNIKV_POLICY": "3", "UNIKV_ALPHA": "0",
        "UNIKV_LOG": str(step_csv),
        "UNIKV_SPILL_CAP": str(max(target + 2048, 4096)),
    })

    t0 = time.time()
    done = subprocess.run(args, cwd=LLAMA_DIR, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True, errors="replace")
    dur = time.time() - t0
    log_txt.write_text("=== STDOUT ===\n" + done.stdout +
                       "\n=== STDERR ===\n" + done.stderr[-8000:])

    rows = list(csv.DictReader(step_csv.open())) if step_csv.exists() else []
    burst = rows[-BURST:] if len(rows) >= BURST else rows
    burst_ms = [float(r["ttft_ms"]) for r in burst]

    # The first decode step after a batched prefill pays a one-off graph
    # reallocation (the ubatch shape changes from 512 to 1 with a large spilled
    # tier attached): ~8 s at n_spill=8192, against a ~68 ms steady state. That
    # is a real cost but a per-shape-change one, not part of the per-step recall
    # cost being measured, so the leading WARMUP steps are excluded and reported
    # separately rather than averaged in.
    warm = burst[:WARMUP]
    meas = burst[WARMUP:]
    ms = [float(r["ttft_ms"]) for r in meas]
    nsp = [int(r.get("n_spilled", 0) or 0) for r in meas]
    transient_ms = round(max(float(r["ttft_ms"]) for r in warm), 1) if warm else None

    mfa = re.search(r"flash_attn\s*=\s*(\w+)", done.stderr)
    mpe = re.search(r"prompt eval time =\s+([\d.]+) ms\s*/\s*(\d+) tokens", done.stderr)

    return {
        "target": target, "trial": trial, "prompt_tokens": prompt_tokens,
        "rc": done.returncode, "burst_steps": len(meas),
        "transient_ms": transient_ms,
        "n_spill_mean": round(st.mean(nsp), 1) if nsp else None,
        "n_spill_min": min(nsp) if nsp else None,
        "n_spill_max": max(nsp) if nsp else None,
        "ms_mean": round(st.mean(ms), 3) if ms else None,
        "ms_median": round(st.median(ms), 3) if ms else None,
        "ms_sd": round(st.stdev(ms), 3) if len(ms) > 1 else 0.0,
        "tok_per_sec": round(1000 / st.mean(ms), 3) if ms else None,
        "flash_attn": mfa.group(1) if mfa else "UNPARSED",
        "prefill_ms": float(mpe.group(1)) if mpe else None,
        "prefill_tokens": int(mpe.group(2)) if mpe else None,
        "duration_s": round(dur, 1),
    }


def main() -> int:
    for d in (RESULTS_DIR, PROMPTS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    for p in (COMPLETION_BIN, TOKENIZE_BIN, MODEL_PATH):
        if not p.exists():
            raise FileNotFoundError(p)

    plan = [(n, t) for n in TARGETS for t in range(1, TRIALS + 1)]
    random.Random(SHUFFLE_SEED).shuffle(plan)
    print(f"isochronal recall cost: {len(TARGETS)} targets x {TRIALS} trials "
          f"= {len(plan)} runs, C={CTX}, burst={BURST}, -fa off")
    print(f"  randomized block (seed {SHUFFLE_SEED}), cooldown {COOLDOWN_S}s")
    print("  order:", " ".join(f"n{n}t{t}" for n, t in plan))
    print()

    out = RESULTS_DIR / "recall_cost_isochronal.csv"
    out.unlink(missing_ok=True)
    fields = ["order_idx", "target", "trial", "prompt_tokens", "rc", "burst_steps",
              "transient_ms", "n_spill_mean", "n_spill_min", "n_spill_max",
              "ms_mean", "ms_median", "ms_sd", "tok_per_sec", "flash_attn",
              "prefill_ms", "prefill_tokens", "duration_s", "t_start_iso"]
    with out.open("w", newline="") as fh:
        csv.writer(fh).writerow(fields)

    by: dict[int, list[float]] = {n: [] for n in TARGETS}
    flags = []
    for idx, (target, trial) in enumerate(plan, 1):
        if COOLDOWN_S > 0:
            print(f"  [{idx:2d}/{len(plan)}] cooldown {COOLDOWN_S}s ...", flush=True)
            time.sleep(COOLDOWN_S)
        t0 = time.time()
        print(f"  [{idx:2d}/{len(plan)}] target n_spill={target:5d} trial={trial} ...",
              end="", flush=True)
        r = run_point(target, trial)
        print(f" n_spill={r['n_spill_mean']} {r['ms_mean']} ms/step "
              f"= {r['tok_per_sec']} tok/s  prefill {r['prefill_ms']} ms "
              f"transient {r['transient_ms']} ms ({r['duration_s']}s)")

        if r["flash_attn"] != "disabled":
            flags.append(f"n{target}t{trial}: flash_attn={r['flash_attn']}")
        if r["burst_steps"] != BURST - WARMUP:
            flags.append(f"n{target}t{trial}: {r['burst_steps']} measured steps != {BURST - WARMUP}")
        if r["tok_per_sec"] and r["tok_per_sec"] >= CEILING:
            flags.append(f"n{target}t{trial}: {r['tok_per_sec']} >= {CEILING} ceiling")
        elif r["ms_mean"]:
            by[target].append(r["ms_mean"])

        with out.open("a", newline="") as fh:
            csv.writer(fh).writerow([idx] + [r[k] for k in fields[1:-1]] + [iso(t0)])

    print("\n" + "=" * 68)
    print(f"{'target':>7s} {'n':>2s} {'ms/step':>9s} {'sd':>6s} {'tok/s':>7s}")
    print("-" * 68)
    pts = []
    for n in TARGETS:
        v = by[n]
        if v:
            m = st.mean(v)
            sd = st.stdev(v) if len(v) > 1 else 0.0
            pts.append((n, m))
            print(f"{n:7d} {len(v):2d} {m:9.3f} {sd:6.3f} {1000 / m:7.2f}")
    print("=" * 68)

    spill_pts = [(n, m) for n, m in pts if n > 0]
    if len(spill_pts) > 2:
        xs = [n for n, _ in spill_pts]
        ys = [m for _, m in spill_pts]
        mx, my = st.mean(xs), st.mean(ys)
        sxx = sum((x - mx) ** 2 for x in xs)
        b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
        a = my - b * mx
        resid = [y - (a + b * x) for x, y in zip(xs, ys)]
        s = (sum(r * r for r in resid) / (len(xs) - 2)) ** 0.5
        print(f"\nfit over controlled points: t_step = {a:.2f} ms + "
              f"{b * 1000:.4f} us/cell * n_spill   (+/- {s / sxx ** 0.5 * 1000:.4f}, "
              f"residual sd {s:.2f} ms)")
        base = next((m for n, m in pts if n == 0), None)
        if base:
            print(f"no-spill reference (policy 3, nothing spilled): {base:.2f} ms/step")
            print(f"fixed cost of entering the two-tier path: {a - base:.2f} ms/step")

    if flags:
        print("\n!! FLAGS:")
        for f in flags:
            print(f"   {f}")
    print(f"\nwrote {out}")
    return 1 if flags else 0


if __name__ == "__main__":
    sys.exit(main())
