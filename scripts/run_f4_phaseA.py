#!/usr/bin/env python3
"""Phase A — closing the asserted-but-unmeasured gaps in Finding 4 (capacity).

Three sub-runs, selected with UNIKV_F4=A1|A2|A3 (comma-separated, default all).
Outcomes here are binary or coarse (completes / out of memory / a buffer size in
MiB), so thermal state is irrelevant and no cooldowns are used.  Memory pressure
is kept low because an out-of-memory outcome is decided by working-set size.

A3  Flash-attention-on capacity re-run.  Finding 4's ceiling is measured with
    flash attention off, to match the spill path.  The draft then ASSERTS that a
    fused build reaches a higher C, and separately that the 65536-token workload
    has no upstream setting that runs it.  Those two statements are in tension:
    if the fused kernel removes the scratch term, upstream may well run the
    workload after all.  Run first because it is cheap and because a positive
    result changes the finding's headline.

A1  ubatch sweep.  The scratch term is n_kv x n_ubatch x n_head, so it scales
    with ubatch and ubatch therefore moves the ceiling and the 193 KiB/token
    accounting.  For each ubatch, walk C upward until the driver refuses, giving
    a per-ubatch ceiling and a per-ubatch scratch coefficient.

A2  Continuation-policy arms for the 65536-token workload.  The draft states
    that the rolling window and H2O would also complete it at a small C by
    discarding context.  Asserted, not measured.  Measuring it turns the claim
    into "what is unavailable at any upstream setting is completing it
    losslessly", which is the claim the paper actually wants.

Protocol: seed 123, temp 0, greedy, --ignore-eos, -fit off, and the flash_attn
setting parsed back out of each run's own log rather than trusted from here.
Nothing in A1/A3 produces a tok/s headline; A2 does, so A2 runs uninstrumented.
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
# Model/tag overridable so a second model runs THIS code path; unset = Llama.
MODEL_PATH  = Path(os.environ.get(
    "UNIKV_MODEL", LLAMA_DIR / "models" / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"))
TAG         = os.environ.get("UNIKV_TAG", "")
SUF         = f"_{TAG}" if TAG else ""

RESULTS_DIR = ROOT / "stress_results"
ART_DIR     = ROOT / "artifacts" / f"f4_phaseA{SUF}"
# per-tag: prompt token counts are tokenizer-specific
PROMPTS_DIR = (ROOT / "artifacts" / "capacity_probe" / "prompts" if not TAG
               else ART_DIR / "prompts")
LOGS_DIR    = ART_DIR / "logs"

THREADS     = 10
GPU_LAYERS  = 999
SEED        = 123
FIXED_TOKEN = " token"

WHICH = [x.strip() for x in os.environ.get("UNIKV_F4", "A3,A1,A2").split(",") if x.strip()]

# A1: contexts walked upward per ubatch until refusal
A1_UBATCH = [int(x) for x in os.environ.get("UNIKV_F4_UBATCH", "64,128,256,512").split(",")]
A1_CTX    = [int(x) for x in os.environ.get(
                 "UNIKV_F4_CTX", "49152,65536,81920,98304,131072").split(",")]
A1_PROMPT = int(os.environ.get("UNIKV_F4_A1_PROMPT", "16384"))

# A2/A3: the workload Finding 4 says upstream cannot run
BIG_PROMPT = int(os.environ.get("UNIKV_F4_BIG_PROMPT", "65536"))
BIG_GEN    = int(os.environ.get("UNIKV_F4_BIG_GEN", "32"))

TIMEOUT_S = int(os.environ.get("UNIKV_F4_TIMEOUT", "7200"))


def count_tokens(path: Path) -> int:
    out = subprocess.run(
        [str(TOKENIZE_BIN), "-m", str(MODEL_PATH), "-f", str(path),
         "--show-count", "--log-disable"],
        cwd=LLAMA_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, errors="replace", check=True).stdout
    m = re.search(r"Total number of tokens:\s*(\d+)", out)
    if not m:
        raise RuntimeError(f"token count parse failed for {path.name}:\n{out}")
    return int(m.group(1))


def ensure_prompt(n: int) -> Path:
    """Exactly n tokens under whichever tokenizer MODEL_PATH carries.

    The original assumed FIXED_TOKEN is one token AND that the tokenizer prepends
    exactly one BOS (so n-1 repeats give n). Llama 3.1 does; Qwen2.5 does not.
    Solve for the repeat count instead of assuming it."""
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PROMPTS_DIR / f"prompt_{n}tok.txt"
    if path.exists() and count_tokens(path) == n:
        return path
    reps, seen = max(n - 1, 0), {}
    for _ in range(16):
        path.write_text(FIXED_TOKEN * reps)
        got = seen[reps] = count_tokens(path)
        if got == n:
            return path
        nxt = reps + (n - got)
        if nxt in seen or nxt <= 0:
            break
        reps = nxt
    raise RuntimeError(f"prompt token mismatch for {n}: repeat count -> {seen}")


def run(tag, prompt, ctx, policy, fa, ubatch, batch, gen,
        spill_cap=None, uninstrumented=True):
    e2e_csv = LOGS_DIR / f"e2e_{tag}.csv"
    log_txt = LOGS_DIR / f"gen_{tag}.txt"
    e2e_csv.unlink(missing_ok=True)

    args = [
        str(COMPLETION_BIN), "-m", str(MODEL_PATH), "-f", str(prompt),
        "-n", str(gen), "-c", str(ctx), "-b", str(batch), "-ub", str(ubatch),
        "-ngl", str(GPU_LAYERS), "-t", str(THREADS), "-fa", fa, "-fit", "off",
        "--temp", "0", "--seed", str(SEED), "--ignore-eos", "--no-warmup",
        "--simple-io", "--no-display-prompt", "-no-cnv",
    ]
    env = os.environ.copy()
    env.update({"UNIKV_POLICY": str(policy), "UNIKV_ALPHA": "0",
                "UNIKV_E2E_LOG": str(e2e_csv), "UNIKV_MEM_BREAKDOWN": "1"})
    env.pop("UNIKV_LOG", None)          # never instrument a tok/s number
    env.pop("UNIKV_H2O_TRACE", None)
    if spill_cap is not None:
        env["UNIKV_SPILL_CAP"] = str(spill_cap)

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

    kv     = g(r"llama_kv_cache:\s+size\s+=\s+([\d.]+)\s+MiB", float)
    model  = g(r"MTL0_Mapped model buffer size\s*=\s*([\d.]+)", float)
    comp   = g(r"MTL0 compute buffer size\s*=\s*([\d.]+)", float)
    budget_mb = g(r"recommendedMaxWorkingSetSize\s+=\s+([\d.]+)", float)
    # ggml prints this in DECIMAL MB; our footprints are MiB. Convert, or every
    # ratio comes out 4.9% low.
    budget = budget_mb * 1e6 / 2**20 if budget_mb else None
    device = sum(x for x in (kv, model, comp) if x)

    gpu_oom  = len(re.findall(r"kIOGPUCommandBufferCallbackErrorOutOfMemory", err))
    too_long = bool(re.search(r"prompt is too long", err, re.IGNORECASE))

    if gpu_oom:
        outcome = "GPU_OOM"
    elif too_long:
        outcome = "PROMPT_TOO_LONG"
    elif timed_out:
        outcome = "TIMEOUT"
    elif rc == 0 and (dec == gen or dec):
        outcome = "COMPLETES"
    else:
        outcome = f"FAILED_rc{rc}"

    return {
        "tag": tag, "ctx": ctx, "policy": policy, "fa_requested": fa,
        "flash_attn": g(r"flash_attn\s*=\s*(\w+)"),
        "n_batch": g(r"n_batch\s*=\s*(\d+)", int),
        "n_ubatch": g(r"n_ubatch\s*=\s*(\d+)", int),
        "prompt_tokens": None, "rc": rc, "outcome": outcome,
        "kv_mib": kv, "model_mib": model, "scratch_mib": comp,
        "device_total_mib": round(device, 1), "budget_mib": budget,
        "x_budget": round(device / budget, 3) if budget else None,
        "scratch_kib_per_ctx_token": round(comp * 1024 / ctx, 2) if comp else None,
        "device_kib_per_ctx_token": round((kv + comp) * 1024 / ctx, 2) if kv and comp else None,
        "decode_tokens": dec, "tok_per_sec": tps,
        "prefill_ms": g(r"prompt eval time =\s+([\d.]+) ms", float),
        "gpu_oom_lines": gpu_oom, "duration_s": round(dur, 1),
    }


def write(rows, name):
    out = RESULTS_DIR / name
    cols = sorted({k for r in rows for k in r})
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {out}")


def main() -> int:
    for d in (RESULTS_DIR, LOGS_DIR, PROMPTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # ---------------- A3: flash attention ON at the big workload -------------
    if "A3" in WHICH:
        print("== A3: does a fused (flash-attention-on) upstream build run the "
              f"{BIG_PROMPT}-token workload? ==")
        big = ensure_prompt(BIG_PROMPT)
        rows = []
        # C values that can hold the prompt; the FA-off build refused all of these
        for ctx in [int(x) for x in os.environ.get(
                "UNIKV_F4_A3_CTX", "65536,73728,98304,131072").split(",")]:
            tag = f"A3_p0_fa_on_c{ctx}"
            print(f"  stock FA-ON  C={ctx:6d} ...", end="", flush=True)
            r = run(tag, big, ctx, 0, "on", 512, 512, BIG_GEN)
            r["prompt_tokens"] = BIG_PROMPT
            rows.append(r)
            print(f" {r['outcome']:16s} fa={r['flash_attn']} scratch="
                  f"{r['scratch_mib']} MiB device={r['device_total_mib']} MiB "
                  f"({r['x_budget']}x) {r['duration_s']}s")
            if r["outcome"] == "COMPLETES":
                print()
                print("  *** FLAG: a fused upstream build RUNS this workload. ***")
                print("  *** Finding 4's 'no upstream setting runs it' claim is FALSE")
                print("      as stated and must be scoped to the non-fused path.  ***")
                break
        write(rows, "f4_a3_flash_attn_capacity.csv")
        print()

    # ---------------- A1: ubatch sweep ---------------------------------------
    if "A1" in WHICH:
        print("== A1: ubatch sweep — how ubatch moves the ceiling and the "
              "per-token accounting ==")
        p = ensure_prompt(A1_PROMPT)
        rows = []
        for ub in A1_UBATCH:
            batch = max(512, ub)
            print(f"  -- ubatch {ub} --")
            for ctx in A1_CTX:
                tag = f"A1_ub{ub}_c{ctx}"
                print(f"     C={ctx:6d} ...", end="", flush=True)
                r = run(tag, p, ctx, 0, "off", ub, batch, 8)
                r["prompt_tokens"] = A1_PROMPT
                rows.append(r)
                print(f" {r['outcome']:16s} scratch={r['scratch_mib']} MiB "
                      f"({r['scratch_kib_per_ctx_token']} KiB/tok) device="
                      f"{r['device_total_mib']} MiB ({r['x_budget']}x) "
                      f"{r['duration_s']}s")
                if r["outcome"] == "GPU_OOM":
                    print(f"     -> ceiling for ubatch {ub} is below C={ctx}")
                    break
        write(rows, f"f4_a1_ubatch_sweep{SUF}.csv")
        print()

    # ---------------- A2: continuation policies at the big workload ----------
    if "A2" in WHICH:
        print(f"== A2: do the continuation policies complete the {BIG_PROMPT}-token "
              "workload by discarding? ==")
        big = ensure_prompt(BIG_PROMPT)
        rows = []
        for tag, policy, ctx, ub, cap, role in (
                ("A2_p1_c4096", 1, 4096, 512, None, "rolling window (context shift)"),
                ("A2_p4_c4096", 4, 4096, 512, None, "H2O fixed-budget eviction"),
                # A1 showed the scratch term scales with ubatch, so ubatch moves
                # the ceiling as much as the kernel choice does. If a small-ubatch
                # upstream config holds this prompt with flash attention OFF, then
                # "no upstream setting runs it" is false even on the non-fused
                # path, independently of the A3 result.
                ("A2_p0_c81920_ub64", 0, 81920, 64, None,
                 "stock, fa OFF, ubatch 64 (lossless, whole prompt resident)"),
        ):
            print(f"  {role:52s} C={ctx} ub={ub} ...", end="", flush=True)
            r = run(tag, big, ctx, policy, "off", ub, max(512, ub), BIG_GEN,
                    spill_cap=cap)
            r["prompt_tokens"] = BIG_PROMPT
            r["role"] = role
            rows.append(r)
            print(f" {r['outcome']:16s} decode={r['tok_per_sec']} tok/s "
                  f"prefill={r['prefill_ms']} ms device={r['device_total_mib']} MiB "
                  f"({r['duration_s']}s)")
        write(rows, "f4_a2_continuation_arms.csv")
        print()

    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
