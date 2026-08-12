#!/usr/bin/env python3
"""Capacity-bounded overflow: find the C that hardware, not -c, forbids
(review M2, P1, C4).

Every overflow experiment in the paper overflows because -c was set below the
workload on a 24 GB machine that completes the same workload at C=4096. The
honest reader's summary is "set -c higher". This probe locates the point where
that answer runs out: Metal reports a recommendedMaxWorkingSetSize (~17.2 GiB
of 24 GB on this machine) and the KV cache costs 128 KiB/token for this model
(32 layers x 8 KV heads x 128 dim x 2 (K+V) x 2 bytes), so a large enough C
cannot be allocated at all.

Phase 1  bisect the stock (policy 0) allocation boundary: for each candidate C,
         load the model with a 1-token generation and record whether allocation
         succeeded, the KV buffer size, and the Metal working-set line. Cheap --
         a failing C fails at load.

Phase 2  at the same C, confirm UniKV (policy 3) loads with device KV bounded at
         a small resident window while the host tier carries the rest, i.e. the
         configuration stock cannot express.

Phase 3  prefill-rate scaling under policy 3, used to size (and cost out) an
         end-to-end run at the boundary rather than guessing at it.

Nothing here is timed against the thermal protocol -- these are allocation
outcomes and order-of-magnitude rates, not throughput claims.
"""

import csv
import json
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
LOGS_DIR    = ART_DIR / "logs"

THREADS     = 10
GPU_LAYERS  = 999
BATCH       = 512
UBATCH      = 512
SEED        = 123
FIXED_TOKEN = " token"

PHASES = os.environ.get("UNIKV_CAP_PHASES", "1,2,3")
# Candidate contexts (tokens). KV is 128 KiB/token, so 65536 -> 8 GiB and
# 131072 -> 16 GiB of KV alone. With flash attention off the attention scratch
# also scales with C (the explicit KQ matrix is n_kv x n_ubatch x n_head), so
# the device working set grows faster than the KV line suggests and the
# boundary is expected below the pure-KV arithmetic. Sampled finely across the
# region where that is likely to bite.
CANDIDATES = [int(x) for x in os.environ.get(
    "UNIKV_CAP_CTX",
    "16384,32768,49152,65536,73728,81920,98304,131072").split(",")]
PREFILL_SIZES = [int(x) for x in os.environ.get(
    "UNIKV_CAP_PREFILL", "4096,8192,16384").split(",")]
PREFILL_CTX = int(os.environ.get("UNIKV_CAP_PREFILL_CTX", "2048"))


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


def make_prompt(n_tokens: int) -> Path:
    path = PROMPTS_DIR / f"prompt_{n_tokens}tok.txt"
    if not path.exists():
        path.write_text(FIXED_TOKEN * max(n_tokens - 1, 0))
        got = count_tokens(path)
        if got != n_tokens:
            raise RuntimeError(f"prompt token mismatch: want {n_tokens} got {got}")
    return path


def probe(ctx: int, policy: int, prompt: Path, gen: int, tag: str,
          spill_cap: int | None = None, timeout: int = 1800) -> dict:
    log_txt = LOGS_DIR / f"probe_{tag}.txt"
    e2e_csv = LOGS_DIR / f"e2e_{tag}.csv"
    e2e_csv.unlink(missing_ok=True)

    args = [
        str(COMPLETION_BIN), "-m", str(MODEL_PATH), "-f", str(prompt),
        "-n", str(gen), "-c", str(ctx), "-b", str(BATCH), "-ub", str(UBATCH),
        "-ngl", str(GPU_LAYERS), "-t", str(THREADS), "-fa", "off", "-fit", "off",
        "--temp", "0", "--seed", str(SEED), "--ignore-eos", "--no-warmup",
        "--simple-io", "--no-display-prompt", "-no-cnv",
    ]
    env = os.environ.copy()
    env.update({"UNIKV_POLICY": str(policy), "UNIKV_ALPHA": "0",
                "UNIKV_E2E_LOG": str(e2e_csv), "UNIKV_MEM_BREAKDOWN": "1"})
    if spill_cap is not None:
        env["UNIKV_SPILL_CAP"] = str(spill_cap)

    t0 = time.time()
    try:
        done = subprocess.run(args, cwd=LLAMA_DIR, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True, errors="replace",
                              timeout=timeout)
        rc, out, err = done.returncode, done.stdout, done.stderr
        timed_out = False
    except subprocess.TimeoutExpired as e:
        rc, timed_out = None, True
        out = e.stdout.decode("utf-8", "replace") if e.stdout else ""
        err = e.stderr.decode("utf-8", "replace") if e.stderr else ""
    dur = time.time() - t0
    log_txt.write_text(f"=== ARGS ===\n{' '.join(args)}\n"
                       f"=== STDOUT ===\n{out}\n=== STDERR ===\n{err}")

    def grab(pat, cast=str):
        m = re.search(pat, err)
        return cast(m.group(1)) if m else None

    # llama.cpp surfaces Metal's budget once per context; a failed KV allocation
    # shows up as a ggml/metal buffer-allocation error, not as a UniKV message.
    alloc_errors = re.findall(
        r"^.*(?:failed to allocate|ggml_backend_metal_buffer|"
        r"unable to allocate|out of memory|failed to allocate buffer).*$",
        err, re.MULTILINE | re.IGNORECASE)

    prefill_ms = grab(r"prompt eval time =\s+([\d.]+) ms", float)
    prefill_tok = grab(r"prompt eval time =.*?/\s+(\d+) tokens", int)

    dec_tokens, tok_s = None, None
    if e2e_csv.exists():
        with e2e_csv.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        if rows:
            dec_tokens = int(rows[-1]["decode_tokens"])
            tok_s = float(rows[-1]["tok_per_sec"])

    return {
        "tag": tag, "ctx": ctx, "policy": policy, "rc": rc, "timed_out": timed_out,
        "duration_s": round(dur, 1),
        "loaded": grab(r"llama_kv_cache:\s+size\s+=\s+([\d.]+)\s+MiB", float) is not None,
        "kv_mib": grab(r"llama_kv_cache:\s+size\s+=\s+([\d.]+)\s+MiB", float),
        "metal_working_set_mib": grab(r"recommendedMaxWorkingSetSize\s+=\s+([\d.]+)", float),
        "host_store_mib": grab(r"host spill store\s+=\s+([\d.]+)\s+MiB", float),
        "flash_attn": grab(r"flash_attn\s*=\s*(\w+)"),
        "decode_tokens": dec_tokens, "tok_per_sec": tok_s,
        "prefill_ms": prefill_ms, "prefill_tokens": prefill_tok,
        "alloc_error": alloc_errors[0].strip()[:200] if alloc_errors else "",
        "n_alloc_errors": len(alloc_errors),
    }


def main() -> int:
    for d in (RESULTS_DIR, PROMPTS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    for p in (COMPLETION_BIN, TOKENIZE_BIN, MODEL_PATH):
        if not p.exists():
            raise FileNotFoundError(p)

    rows = []
    tiny = make_prompt(512)

    if "1" in PHASES:
        print("== Phase 1: stock (policy 0) allocation boundary ==")
        print(f"  KV per token = 128 KiB for this model; candidates: {CANDIDATES}\n")
        for ctx in CANDIDATES:
            print(f"  C={ctx:7d} ({ctx * 128 / 1024 / 1024:6.2f} GiB KV) ...",
                  end="", flush=True)
            r = probe(ctx, 0, tiny, 1, f"p0_c{ctx}", timeout=900)
            r["phase"] = 1
            rows.append(r)
            status = "LOADED" if r["loaded"] else "FAILED"
            print(f" {status:6s} rc={r['rc']} kv={r['kv_mib']} MiB "
                  f"errs={r['n_alloc_errors']} ({r['duration_s']}s)")
            if r["alloc_error"]:
                print(f"      {r['alloc_error']}")
        print()

    if "2" in PHASES:
        print("== Phase 2: UniKV (policy 3) at the same contexts, bounded device KV ==")
        failed = [r["ctx"] for r in rows if r["phase"] == 1 and not r["loaded"]]
        target = failed[0] if failed else CANDIDATES[-1]
        print(f"  stock's first failing C = {failed[0] if failed else 'none in range'}; "
              f"UniKV resident window C=4096, host tier sized for {target} cells\n")
        for resident in (4096,):
            print(f"  UniKV C={resident} spill_cap={target} ...", end="", flush=True)
            r = probe(resident, 3, tiny, 1, f"p3_c{resident}_cap{target}",
                      spill_cap=target, timeout=900)
            r["phase"] = 2
            rows.append(r)
            print(f" {'LOADED' if r['loaded'] else 'FAILED'} rc={r['rc']} "
                  f"kv={r['kv_mib']} MiB host_store={r['host_store_mib']} MiB "
                  f"({r['duration_s']}s)")
        print()

    if "3" in PHASES:
        print("== Phase 3: policy-3 prefill scaling (sizes an end-to-end run) ==")
        for n in PREFILL_SIZES:
            p = make_prompt(n)
            print(f"  prompt {n:6d} tok @ C={PREFILL_CTX} ...", end="", flush=True)
            r = probe(PREFILL_CTX, 3, p, 1, f"p3_prefill{n}", spill_cap=n + 2048,
                      timeout=3600)
            r["phase"] = 3
            r["prompt_tokens"] = n
            rows.append(r)
            rate = (r["prefill_tokens"] / (r["prefill_ms"] / 1000)
                    if r["prefill_ms"] and r["prefill_tokens"] else None)
            print(f" rc={r['rc']} prefill {r['prefill_ms']} ms "
                  f"({rate:.1f} tok/s)" if rate else f" rc={r['rc']} (no prefill timing)")
        print()

    out = RESULTS_DIR / "capacity_probe.csv"
    cols = sorted({k for r in rows for k in r})
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")
    print(json.dumps([{k: r.get(k) for k in
                       ("tag", "ctx", "policy", "loaded", "kv_mib", "rc",
                        "prefill_ms", "prefill_tokens")} for r in rows], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
