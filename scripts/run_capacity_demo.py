#!/usr/bin/env python3
"""A workload where enlarging -c is genuinely unavailable (review M2, P1, C4).

Established by scripts/run_capacity_probe.py and the bisection in
artifacts/capacity_probe/bisect.log:

  * Metal reports recommendedMaxWorkingSetSize = 16383 MiB on this 24 GB M4 Pro.
  * With flash attention off the device working set grows ~193 KiB per token of
    context: 128 KiB of KV plus ~65 KiB of attention scratch (the explicit
    n_kv x n_ubatch KQ matrix), on top of 4685 MiB of weights.
  * Allocation is NOT the binding step -- macOS lazily backs the buffers, so
    llama.cpp happily "loads" at C=131072 (29537 MiB, 1.72x the budget) and only
    dies when real work touches the memory.
  * Processing a 16384-token prompt: C=49152 (14017 MiB, 0.86x) succeeds;
    C=65536 (17121 MiB, 1.05x) and above fail with
    kIOGPUCommandBufferCallbackErrorOutOfMemory -- the Metal driver refusing to
    execute the command buffer, not llama.cpp's cache-full guard.

So stock's maximum WORKABLE context here is between 49152 and 65536 tokens, and
it is set by hardware, not by the -c flag. This script runs a workload above
that ceiling, where every stock configuration fails for a different reason and
neither failure can be fixed by choosing a different -c:

  A  stock at C=49152   the largest context that actually runs -> the prompt no
                        longer fits, rejected by the length guard
  B  stock at C=73728   a context that WOULD fit the prompt -> GPU out of memory
  C  UniKV at C=4096    bounded device window plus a host tier -> completes

This is the configuration the review says the paper lacks: "one overflow that
hardware, not -c, causes".

Protocol: -fa off, -fit off, greedy (temp 0), seed 123, -b/-ub 512.
The UniKV arm is slow by design -- the spilled tier is attended on the CPU and
prefill through it grows superlinearly (~N^1.76 measured). Budget ~1 hour.
"""

import csv
import os
import re
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
ART_DIR     = ROOT / "artifacts" / "capacity_probe"
PROMPTS_DIR = ART_DIR / "prompts"
LOGS_DIR    = ART_DIR / "demo_logs"

PROMPT_TOKENS = int(os.environ.get("UNIKV_CD_PROMPT", "65536"))
GEN_TOKENS    = int(os.environ.get("UNIKV_CD_GEN", "32"))
THREADS       = 10
GPU_LAYERS    = 999
BATCH         = 512
UBATCH        = 512
SEED          = 123
FIXED_TOKEN   = " token"
COOLDOWN_S    = int(os.environ.get("UNIKV_CD_COOLDOWN", "90"))
TIMEOUT_S     = int(os.environ.get("UNIKV_CD_TIMEOUT", "7200"))

ARMS = [
    ("A_stock_c49152",  49152, 0, None,  "stock at its largest workable C"),
    ("B_stock_c73728",  73728, 0, None,  "stock at a C large enough to hold the prompt"),
    ("C_unikv_c4096",    4096, 3, PROMPT_TOKENS + 4096,
                                         "UniKV, bounded device window + host tier"),
]


def build_prompt(n: int) -> Path:
    path = PROMPTS_DIR / f"prompt_{n}tok.txt"
    if not path.exists():
        path.write_text(FIXED_TOKEN * max(n - 1, 0))
        out = subprocess.run(
            [str(TOKENIZE_BIN), "-m", str(MODEL_PATH), "-f", str(path),
             "--show-count", "--log-disable"],
            cwd=LLAMA_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, errors="replace", check=True).stdout
        m = re.search(r"Total number of tokens:\s*(\d+)", out)
        if not m or int(m.group(1)) != n:
            raise RuntimeError(f"prompt token mismatch: want {n}, got "
                               f"{m.group(1) if m else '?'}")
    return path


def run_arm(prompt: Path, tag: str, ctx: int, policy: int, cap, role: str) -> dict:
    e2e_csv = LOGS_DIR / f"e2e_{tag}.csv"
    log_txt = LOGS_DIR / f"gen_{tag}.txt"
    e2e_csv.unlink(missing_ok=True)

    args = [
        str(COMPLETION_BIN), "-m", str(MODEL_PATH), "-f", str(prompt),
        "-n", str(GEN_TOKENS), "-c", str(ctx), "-b", str(BATCH), "-ub", str(UBATCH),
        "-ngl", str(GPU_LAYERS), "-t", str(THREADS), "-fa", "off", "-fit", "off",
        "--temp", "0", "--seed", str(SEED), "--ignore-eos", "--no-warmup",
        "--simple-io", "--no-display-prompt", "-no-cnv",
    ]
    env = os.environ.copy()
    env.update({"UNIKV_POLICY": str(policy), "UNIKV_ALPHA": "0",
                "UNIKV_E2E_LOG": str(e2e_csv), "UNIKV_MEM_BREAKDOWN": "1"})
    if cap is not None:
        env["UNIKV_SPILL_CAP"] = str(cap)

    t0 = time.time()
    try:
        d = subprocess.run(args, cwd=LLAMA_DIR, env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, text=True, errors="replace",
                           timeout=TIMEOUT_S)
        rc, out, err, timed_out = d.returncode, d.stdout, d.stderr, False
    except subprocess.TimeoutExpired as e:
        rc, timed_out = None, True
        out = e.stdout.decode("utf-8", "replace") if e.stdout else ""
        err = e.stderr.decode("utf-8", "replace") if e.stderr else ""
    dur = time.time() - t0
    log_txt.write_text(f"=== ARGS ===\n{' '.join(args)}\n=== STDOUT ===\n{out}"
                       f"\n=== STDERR ===\n{err}")

    def g(pat, cast=str):
        m = re.search(pat, err)
        return cast(m.group(1)) if m else None

    dec, tps = None, None
    if e2e_csv.exists():
        rows = list(csv.DictReader(e2e_csv.open()))
        if rows:
            dec = int(rows[-1]["decode_tokens"])
            tps = float(rows[-1]["tok_per_sec"])

    kv    = g(r"llama_kv_cache:\s+size\s+=\s+([\d.]+)\s+MiB", float)
    model = g(r"MTL0_Mapped model buffer size\s*=\s*([\d.]+)", float)
    comp  = g(r"MTL0 compute buffer size\s*=\s*([\d.]+)", float)
    budget_mb = g(r"recommendedMaxWorkingSetSize\s+=\s+([\d.]+)", float)
    # ggml prints this in DECIMAL MB; our footprints are MiB. Convert, or every
    # ratio comes out 4.9% low.
    budget = budget_mb * 1e6 / 2**20 if budget_mb else None
    device_total = sum(x for x in (kv, model, comp) if x)

    gpu_oom = len(re.findall(r"kIOGPUCommandBufferCallbackErrorOutOfMemory", err))
    too_long = bool(re.search(r"prompt is too long|exceeds.*context|n_tokens.*>.*n_ctx",
                              err, re.IGNORECASE))

    if gpu_oom:
        outcome = "GPU OUT OF MEMORY (Metal command buffer refused)"
    elif too_long:
        outcome = "REJECTED: prompt longer than the context"
    elif dec == GEN_TOKENS:
        outcome = "COMPLETED"
    elif rc != 0:
        outcome = f"FAILED rc={rc}"
    else:
        outcome = f"partial ({dec} tokens)"

    return {
        "arm": tag, "role": role, "ctx": ctx, "policy": policy, "spill_cap": cap,
        "prompt_tokens": PROMPT_TOKENS, "rc": rc, "timed_out": timed_out,
        "outcome": outcome, "gpu_oom_lines": gpu_oom, "prompt_rejected": too_long,
        "flash_attn": g(r"flash_attn\s*=\s*(\w+)"),
        "kv_device_mib": kv, "model_device_mib": model, "compute_device_mib": comp,
        "device_total_mib": round(device_total, 1),
        "metal_budget_mib": budget,
        "device_over_budget": round(device_total / budget, 3) if budget else None,
        "host_store_mib": g(r"host spill store\s*=\s*([\d.]+)\s+MiB", float),
        "prefill_ms": g(r"prompt eval time =\s+([\d.]+) ms", float),
        "decode_tokens": dec, "decode_tok_per_sec": tps,
        "duration_s": round(dur, 1),
    }


def main() -> int:
    for d in (RESULTS_DIR, PROMPTS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    print(f"capacity demo: {PROMPT_TOKENS}-token prompt + {GEN_TOKENS} decode, -fa off")
    print(f"  building prompt ...", flush=True)
    prompt = build_prompt(PROMPT_TOKENS)
    print()

    rows = []
    for i, (tag, ctx, policy, cap, role) in enumerate(ARMS):
        if COOLDOWN_S > 0 and i > 0:
            print(f"  cooldown {COOLDOWN_S}s ...", flush=True)
            time.sleep(COOLDOWN_S)
        print(f"  {tag:16s} C={ctx:6d} policy {policy} ...", end="", flush=True)
        r = run_arm(prompt, tag, ctx, policy, cap, role)
        rows.append(r)
        print(f" {r['outcome']}  device={r['device_total_mib']} MiB "
              f"({r['device_over_budget']}x) ({r['duration_s']}s)")

    out = RESULTS_DIR / "capacity_demo.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("\n" + "=" * 104)
    print(f"workload: {PROMPT_TOKENS}-token prompt, {GEN_TOKENS}-token decode, "
          f"Metal budget {rows[0]['metal_budget_mib']} MiB")
    print("-" * 104)
    print(f"{'arm':16s} {'C':>7s} {'devMiB':>8s} {'x':>6s} {'hostMiB':>8s} "
          f"{'tok/s':>7s}  outcome")
    for r in rows:
        print(f"{r['arm']:16s} {r['ctx']:7d} {r['device_total_mib']:8.0f} "
              f"{r['device_over_budget'] or 0:6.2f} {r['host_store_mib'] or 0:8.0f} "
              f"{r['decode_tok_per_sec'] or 0:7.2f}  {r['outcome']}")
    print("=" * 104)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
