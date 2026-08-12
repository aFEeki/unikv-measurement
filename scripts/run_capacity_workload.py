#!/usr/bin/env python3
"""Does an over-large context actually cost anything? (review M2, P1, C4;
instantiates the paper's "Why not simply enlarge the cache?" argument.)

Phase 1 of run_capacity_probe.py returned a negative result: on this machine
stock llama.cpp ALLOCATES a 131072-token context without failing, 29537 MiB of
device buffers against a 16383 MiB recommendedMaxWorkingSetSize. Metal's budget
is advisory and macOS over-commits unified memory, so the allocation failure the
review hypothesised does not occur here.

That leaves the interesting question. The paper's Discussion answers "why not
just raise -c" with three arguments and instantiates none of them. Raising -c
costs device memory in two ways that both scale with C: the KV cache (128
KiB/token) and, with flash attention off, the attention scratch (the explicit
n_kv x n_ubatch KQ matrix). This measures whether paying that costs throughput
on an IDENTICAL workload.

Three arms, same 16384-token prompt and same short decode:
  A  stock at C=32768    -- comfortably provisioned (~11 GiB device)
  B  stock at C=131072   -- same workload, over-provisioned (~29.5 GiB device,
                            1.8x the advisory budget)
  C  UniKV at C=4096 with a host tier sized for 131072 cells (~5.5 GiB device)

A vs B isolates the cost of an over-large C at fixed work. B vs C is the
capacity comparison at matched capability: both can hold 131072 tokens, but only
one keeps the device working set inside the budget.
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
LOGS_DIR    = ART_DIR / "workload_logs"

PROMPT_TOKENS = int(os.environ.get("UNIKV_CW_PROMPT", "16384"))
GEN_TOKENS    = int(os.environ.get("UNIKV_CW_GEN", "64"))
THREADS       = 10
GPU_LAYERS    = 999
BATCH         = 512
UBATCH        = 512
SEED          = 123
FIXED_TOKEN   = " token"
COOLDOWN_S    = int(os.environ.get("UNIKV_CW_COOLDOWN", "120"))
TIMEOUT_S     = int(os.environ.get("UNIKV_CW_TIMEOUT", "3600"))

# tag, ctx, policy, spill_cap, role
ARMS = [
    ("A_stock_c32768",   32768, 0, None,   "stock, comfortably provisioned"),
    ("B_stock_c131072", 131072, 0, None,   "stock, over-provisioned (1.8x advisory budget)"),
    ("C_unikv_c4096",     4096, 3, 131072, "UniKV, bounded device window + host tier"),
]


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


def ensure_prompt(n: int) -> Path:
    path = PROMPTS_DIR / f"prompt_{n}tok.txt"
    if not path.exists():
        path.write_text(FIXED_TOKEN * max(n - 1, 0))
        got = count_tokens(path)
        if got != n:
            raise RuntimeError(f"prompt token mismatch: want {n} got {got}")
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

    def g(pat, cast=str, src=None):
        m = re.search(pat, src if src is not None else err)
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
    host  = g(r"host spill store\s*=\s*([\d.]+)\s+MiB", float)
    budget_mb = g(r"recommendedMaxWorkingSetSize\s+=\s+([\d.]+)", float)
    # ggml prints this in DECIMAL MB; our footprints are MiB. Convert, or every
    # ratio comes out 4.9% low.
    budget = budget_mb * 1e6 / 2**20 if budget_mb else None
    pe_ms = g(r"prompt eval time =\s+([\d.]+) ms", float)
    pe_tok = g(r"prompt eval time =.*?/\s+(\d+) tokens", int)
    device_total = sum(x for x in (kv, model, comp) if x)

    return {
        "arm": tag, "role": role, "ctx": ctx, "policy": policy,
        "spill_cap": cap, "rc": rc, "timed_out": timed_out,
        "duration_s": round(dur, 1),
        "flash_attn": g(r"flash_attn\s*=\s*(\w+)"),
        "kv_device_mib": kv, "model_device_mib": model,
        "compute_device_mib": comp, "device_total_mib": round(device_total, 1),
        "host_store_mib": host,
        "metal_budget_mib": budget,
        "device_over_budget": round(device_total / budget, 3) if budget else None,
        "prefill_ms": pe_ms, "prefill_tokens": pe_tok,
        "prefill_tok_per_sec": round(pe_tok / (pe_ms / 1000), 1) if pe_ms and pe_tok else None,
        "decode_tokens": dec, "decode_tok_per_sec": tps,
    }


def main() -> int:
    for d in (RESULTS_DIR, PROMPTS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    prompt = ensure_prompt(PROMPT_TOKENS)
    print(f"capacity workload: {PROMPT_TOKENS}-token prompt + {GEN_TOKENS} decode, "
          f"-fa off, seed {SEED}\n")

    rows = []
    for i, (tag, ctx, policy, cap, role) in enumerate(ARMS):
        if COOLDOWN_S > 0 and i > 0:
            print(f"  cooldown {COOLDOWN_S}s ...", flush=True)
            time.sleep(COOLDOWN_S)
        print(f"  {tag:18s} C={ctx:6d} policy {policy} ...", end="", flush=True)
        r = run_arm(prompt, tag, ctx, policy, cap, role)
        rows.append(r)
        print(f" rc={r['rc']} device={r['device_total_mib']} MiB "
              f"({r['device_over_budget']}x budget) prefill "
              f"{r['prefill_tok_per_sec']} tok/s decode {r['decode_tok_per_sec']} "
              f"tok/s ({r['duration_s']}s)")

    out = RESULTS_DIR / "capacity_workload.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("\n" + "=" * 100)
    print(f"{'arm':18s} {'C':>7s} {'devMiB':>8s} {'xbudget':>8s} {'hostMiB':>8s} "
          f"{'prefill t/s':>11s} {'decode t/s':>10s}")
    print("-" * 100)
    for r in rows:
        print(f"{r['arm']:18s} {r['ctx']:7d} {r['device_total_mib']:8.0f} "
              f"{r['device_over_budget'] or 0:8.2f} "
              f"{r['host_store_mib'] or 0:8.0f} "
              f"{r['prefill_tok_per_sec'] or 0:11.1f} "
              f"{r['decode_tok_per_sec'] or 0:10.2f}")
    print("=" * 100)

    a = next((r for r in rows if r["arm"].startswith("A")), None)
    b = next((r for r in rows if r["arm"].startswith("B")), None)
    if a and b and a["prefill_tok_per_sec"] and b["prefill_tok_per_sec"]:
        print(f"\ncost of the over-large context at IDENTICAL work (A -> B):")
        print(f"  prefill {a['prefill_tok_per_sec']:.1f} -> {b['prefill_tok_per_sec']:.1f} tok/s "
              f"({100 * (b['prefill_tok_per_sec'] / a['prefill_tok_per_sec'] - 1):+.1f}%)")
        if a["decode_tok_per_sec"] and b["decode_tok_per_sec"]:
            print(f"  decode  {a['decode_tok_per_sec']:.2f} -> {b['decode_tok_per_sec']:.2f} tok/s "
                  f"({100 * (b['decode_tok_per_sec'] / a['decode_tok_per_sec'] - 1):+.1f}%)")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
