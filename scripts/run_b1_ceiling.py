#!/usr/bin/env python3
"""Where a device-visible spilled tier meets the Finding 4 working-set wall.

Device-visibility is not free: the tier is allocated from the Metal shared
buffer type, so it counts against the same advisory working set whose crossing
Finding 4 shows is refused at execution. The CPU-pinned tier does not, which is
the whole reason the pin exists.

That makes the ceiling the other half of the trade, not a caveat on it. The
statement the paper wants is "device-visibility buys throughput until spill
capacity pushes the working set into the wall, which happens at X cells", and X
has to be measured.

Method. Fix C=1024 (128 MiB of resident KV) and walk UNIKV_SPILL_CAP upward in
both modes, with a workload that actually spills, recording where it stops
working and how. The store is allocated at CAPACITY rather than occupancy, so
the charge lands up front and the ceiling is a property of the cap, not of how
much has spilled.

Predicted from the measured accounting: model 4685 + KV 128 + compute 258 leaves
about 11312 MiB under the 16383 MiB advisory budget, so at 128 KiB per cell the
device-visible tier should meet the wall near 90000 cells. Finding 4's lesson is
that predictions like that name the wrong wall, which is exactly why this is a
measurement.

Both modes are run at every capacity so the contrast is paired: the CPU-pinned
arm is expected to sail past the point where the device-visible arm stops.
Outcomes are categorical, so no cooldowns.
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
MODEL_PATH     = LLAMA_DIR / "models" / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

RESULTS_DIR = ROOT / "stress_results"
ART_DIR     = ROOT / "artifacts" / "b1_gpu_tier"
PROMPTS_DIR = ART_DIR / "prompts"
LOGS_DIR    = ART_DIR / "ceiling_logs"

CTX           = 1024
PROMPT_TOKENS = 512
GEN_TOKENS    = int(os.environ.get("UNIKV_CEIL_GEN", "1536"))
THREADS       = 10
GPU_LAYERS    = 999
BATCH         = 512
UBATCH        = 512
SEED          = 123
FIXED_TOKEN   = " token"
TIMEOUT_S     = 1800

CAPS = [int(x) for x in os.environ.get(
    "UNIKV_CEIL_CAPS",
    "16384,32768,49152,65536,81920,90112,98304,131072").split(",")]
MODES = [x for x in os.environ.get("UNIKV_CEIL_MODES", "1,0").split(",")]


def ensure_prompt() -> Path:
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PROMPTS_DIR / f"prompt_{PROMPT_TOKENS}tok.txt"
    if not path.exists():
        path.write_text(FIXED_TOKEN * max(PROMPT_TOKENS - 1, 0))
    return path


def run_one(prompt: Path, cap: int, dev: str) -> dict:
    tag = f"cap{cap}_dev{dev}"
    e2e_csv = LOGS_DIR / f"e2e_{tag}.csv"
    log_txt = LOGS_DIR / f"gen_{tag}.txt"
    e2e_csv.unlink(missing_ok=True)

    args = [
        str(COMPLETION_BIN), "-m", str(MODEL_PATH), "-f", str(prompt),
        "-n", str(GEN_TOKENS), "-c", str(CTX), "-b", str(BATCH), "-ub", str(UBATCH),
        "-ngl", str(GPU_LAYERS), "-t", str(THREADS), "-fa", "off", "-fit", "off",
        "--temp", "0", "--seed", str(SEED), "--ignore-eos", "--no-warmup",
        "--simple-io", "--no-display-prompt", "-no-cnv",
    ]
    env = os.environ.copy()
    env.update({"UNIKV_POLICY": "3", "UNIKV_ALPHA": "0", "UNIKV_SPILL_DEV": dev,
                "UNIKV_SPILL_CAP": str(cap), "UNIKV_E2E_LOG": str(e2e_csv),
                "UNIKV_MEM_BREAKDOWN": "1"})
    env.pop("UNIKV_LOG", None)

    t0 = time.time()
    try:
        d = subprocess.run(args, cwd=LLAMA_DIR, env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, text=True, errors="replace",
                           timeout=TIMEOUT_S)
        rc, out, err, to = d.returncode, d.stdout, d.stderr, False
    except subprocess.TimeoutExpired as e:
        rc, to = None, True
        out = e.stdout.decode("utf-8", "replace") if e.stdout else ""
        err = e.stderr.decode("utf-8", "replace") if e.stderr else ""
    dur = time.time() - t0
    log_txt.write_text(f"=== STDOUT ===\n{out}\n=== STDERR ===\n{err}")

    def g(pat, cast=str):
        m = re.search(pat, err)
        return cast(m.group(1)) if m else None

    dec, tps = None, None
    if e2e_csv.exists():
        rows = list(csv.DictReader(e2e_csv.open()))
        if rows:
            dec = int(rows[-1]["decode_tokens"])
            tps = float(rows[-1]["tok_per_sec"])

    gpu_oom = len(re.findall(r"kIOGPUCommandBufferCallbackErrorOutOfMemory", err))
    alloc_fail = bool(re.search(r"failed to allocate spill buffer|failed to initialize the host spill store"
                                r"|ggml_backend_alloc_ctx_tensors", err))
    if alloc_fail:
        outcome = "ALLOC_FAILED"
    elif gpu_oom:
        outcome = "GPU_OOM"
    elif to:
        outcome = "TIMEOUT"
    elif rc == 0 and dec == GEN_TOKENS:
        outcome = "COMPLETES"
    else:
        outcome = f"FAILED_rc{rc}"

    # device-side accounting: `unaccounted` is where a Metal-allocated tier lands
    mb = re.search(r"MTL0[^|]*\|\s*(\d+)\s*=\s*(\d+)\s*\+\s*\((\d+)\s*=\s*(\d+)\s*\+\s*(\d+)\s*\+\s*(\d+)\)\s*\+\s*(\d+)", err)
    total = free = self_ = unacc = None
    if mb:
        total, free, self_, unacc = (int(mb.group(1)), int(mb.group(2)),
                                     int(mb.group(3)), int(mb.group(7)))

    return {
        "spill_cap_cells": cap, "spill_dev": dev,
        "cap_mib": round(cap * 128 / 1024, 1),
        "outcome": outcome, "rc": rc,
        "decode_tokens": dec, "tok_per_sec": tps,
        "flash_attn": g(r"flash_attn\s*=\s*(\w+)"),
        "tier_device_visible": bool(re.search(r"spilled tier -> .*device-visible", err)),
        "mtl_total_mib": total, "mtl_free_mib": free,
        "mtl_self_mib": self_, "mtl_unaccounted_mib": unacc,
        "host_store_mib": g(r"host spill store\s*=\s*([\d.]+)\s+MiB", float),
        "gpu_oom_lines": gpu_oom, "duration_s": round(dur, 1),
    }


def main() -> int:
    for d in (RESULTS_DIR, LOGS_DIR, PROMPTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    prompt = ensure_prompt()

    print(f"device-visible ceiling probe: C={CTX}, {GEN_TOKENS} decode, -fa off")
    print(f"  caps (cells): {CAPS}")
    print(f"  modes (UNIKV_SPILL_DEV): {MODES}\n")

    rows = []
    for dev in MODES:
        label = "device-visible" if dev == "1" else "CPU-pinned"
        print(f"-- {label} (UNIKV_SPILL_DEV={dev}) --")
        for cap in CAPS:
            print(f"   cap {cap:7d} ({cap*128/1024:7.0f} MiB) ...", end="", flush=True)
            r = run_one(prompt, cap, dev)
            rows.append(r)
            print(f" {r['outcome']:13s} free={r['mtl_free_mib']} unacc={r['mtl_unaccounted_mib']} "
                  f"tok/s={r['tok_per_sec']} ({r['duration_s']}s)")
            if r["outcome"] in ("GPU_OOM", "ALLOC_FAILED"):
                print(f"   -> {label} ceiling is below {cap} cells "
                      f"({cap*128/1024:.0f} MiB of tier)")
                break
        print()

    out = RESULTS_DIR / "b1_device_tier_ceiling.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("=" * 92)
    print(f"{'mode':15s} {'cap cells':>10s} {'cap MiB':>8s} {'free MiB':>9s} "
          f"{'unacc MiB':>10s} {'tok/s':>7s}  outcome")
    print("-" * 92)
    for r in rows:
        m = "device-visible" if r["spill_dev"] == "1" else "CPU-pinned"
        print(f"{m:15s} {r['spill_cap_cells']:10d} {r['cap_mib']:8.0f} "
              f"{r['mtl_free_mib'] or 0:9d} {r['mtl_unaccounted_mib'] or 0:10d} "
              f"{r['tok_per_sec'] or 0:7.2f}  {r['outcome']}")
    print("=" * 92)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
