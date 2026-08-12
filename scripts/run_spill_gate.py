#!/usr/bin/env python3
"""UniKV Phase 1 (policy 3) acceptance gate — lossless spill-and-recall.

Four recorded arms, identical prompt / seed / greedy decode / -fa off:

  1  p0 @ C=4096  stock, UniKV off        -> reference truth
  2  p3 @ C=4096  UniKV, nothing spills   -> p3 is inert when idle (no regression)
  3  p3 @ C=1024  heavy spill             -> losslessness under real pressure
  4  p1 @ C=1024  naive rolling eviction  -> loss control

Gates:
  arms 1, 2, 3 emit IDENTICAL sampled-token-ID sequences and all retrieve the
  passkey; arm 4 loses it. Every arm's decode rate must sit under the ~58 tok/s
  unified-memory ceiling.

Exactness is compared on the token IDs sampled at each step (UNIKV_TOKEN_LOG),
not on re-encoded output text -- re-tokenizing the strings would only compare
strings again.

A mismatch among arms 1/2/3 is NOT a number to average away: seed + greedy
decoding is deterministic, so a mismatch means nondeterminism in the spill
path, which is itself a finding. The script fails loudly.

Also records, per arm, the device-side KV buffer (memory_breakdown), which is
what distinguishes real spill (device KV = O(C)) from retention (O(N)).

Ingest uses the batched long-prompt path validated byte-identical in Path B Gate-1.
"""

import csv
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT           = Path(__file__).resolve().parents[1]
LLAMA_DIR      = ROOT / "llama.cpp"
BIN_DIR        = LLAMA_DIR / "build-m4pro-metal" / "bin"
COMPLETION_BIN = BIN_DIR / "llama-completion"
TOKENIZE_BIN   = BIN_DIR / "llama-tokenize"
MODEL_PATH     = LLAMA_DIR / "models" / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

RESULTS_DIR  = ROOT / "quality_results"
ARTIFACT_DIR = ROOT / "artifacts" / "spill_gate"
PROMPTS_DIR  = ARTIFACT_DIR / "prompts"
LOGS_DIR     = ARTIFACT_DIR / "logs"

# identical to run_quality_probe_A.py so the passkey position is unchanged
PASSKEY      = "48291"
FILLER       = ("The grass is green. The sky is blue. The sun is yellow. "
                "Here we go. There and back again. ")
INTRO        = ("There is an important piece of information hidden inside the "
                "text below. Find it and remember it, because you will be asked "
                "about it at the end.\n\n")
PASSKEY_LINE = f"\nThe pass key is {PASSKEY}. Remember it. {PASSKEY} is the pass key.\n\n"
QUESTION     = "\nQuestion: What is the pass key?\nAnswer: The pass key is"

GEN_TOKENS = 32
THREADS    = 10
GPU_LAYERS = 999
SEED       = 123
BATCH      = 256
UBATCH     = 256
SPILL_CAP  = 8192

TOK_S_CEILING = 58.0

# (tag, ctx, policy, role)
ARMS = [
    ("arm1_p0_c4096", 4096, 0, "reference truth (stock)"),
    ("arm2_p3_c4096", 4096, 3, "UniKV idle (0 spill)"),
    ("arm3_p3_c1024", 1024, 3, "UniKV heavy spill"),
    ("arm4_p1_c1024", 1024, 1, "naive eviction (loss control)"),
]

EXACT_ARMS = ["arm1_p0_c4096", "arm2_p3_c4096", "arm3_p3_c1024"]
LOSS_ARM   = "arm4_p1_c1024"


def count_tokens(text: str, tmp_name: str) -> int:
    tmp = PROMPTS_DIR / tmp_name
    tmp.write_text(text)
    out = subprocess.run(
        [str(TOKENIZE_BIN), "-m", str(MODEL_PATH), "-f", str(tmp),
         "--show-count", "--log-disable"],
        cwd=LLAMA_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, errors="replace", check=True,
    ).stdout
    m = re.search(r"Total number of tokens:\s*(\d+)", out)
    if not m:
        raise RuntimeError(f"token count parse failed:\n{out}")
    return int(m.group(1))


def build_prompt() -> tuple[Path, int]:
    prefix = INTRO + (FILLER * 7) + PASSKEY_LINE
    full   = prefix + (FILLER * 150) + QUESTION
    path   = PROMPTS_DIR / "spill_gate_prompt.txt"
    path.write_text(full)
    return path, count_tokens(full, "_c_spill_gate.txt")


def strip_marker(stdout: str) -> str:
    return stdout.split("UNIKV_E2E")[0]


def run_arm(prompt_path: Path, ctx: int, policy: int, tag: str, role: str) -> dict:
    step_csv = LOGS_DIR / f"unikv_log_{tag}.csv"
    tok_log  = LOGS_DIR / f"tokens_{tag}.txt"
    step_csv.unlink(missing_ok=True)
    tok_log.unlink(missing_ok=True)

    args = [
        str(COMPLETION_BIN), "-m", str(MODEL_PATH), "-f", str(prompt_path),
        "-n", str(GEN_TOKENS), "-c", str(ctx), "-b", str(BATCH), "-ub", str(UBATCH),
        "-ngl", str(GPU_LAYERS), "-t", str(THREADS), "-fa", "off", "-fit", "off",
        "--temp", "0", "--seed", str(SEED), "--no-warmup", "--simple-io",
        "--no-display-prompt", "-no-cnv",
    ]
    env = os.environ.copy()
    env.update({
        "UNIKV_POLICY": str(policy),
        "UNIKV_LOG": str(step_csv),
        "UNIKV_TOKEN_LOG": str(tok_log),
        "UNIKV_SPILL_CAP": str(SPILL_CAP),
        "UNIKV_MEM_BREAKDOWN": "1",
    })

    done = subprocess.run(args, cwd=LLAMA_DIR, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True, errors="replace")

    (LOGS_DIR / f"gen_{tag}.txt").write_text(
        "=== STDOUT ===\n" + done.stdout + "\n=== STDERR ===\n" + done.stderr)

    text = strip_marker(done.stdout).strip()

    token_ids = []
    if tok_log.exists():
        token_ids = [int(l) for l in tok_log.read_text().split() if l.strip()]

    spill_events = 0
    if step_csv.exists():
        with step_csv.open(newline="") as fh:
            spill_events = sum(1 for r in csv.DictReader(fh)
                               if int(r.get("shift_events", 0) or 0) > 0)

    retained = 0
    m = re.findall(r"retained=(\d+)", done.stderr)
    if m:
        retained = int(m[-1])

    kv_mib = None
    mk = re.search(r"llama_kv_cache:\s+size\s+=\s+([\d.]+)\s+MiB", done.stderr)
    if mk:
        kv_mib = float(mk.group(1))

    host_mib = None
    mh = re.search(r"host spill store\s+=\s+([\d.]+)\s+MiB", done.stderr)
    if mh:
        host_mib = float(mh.group(1))

    tok_s = None
    mt = re.search(r"UNIKV_E2E .*tok_per_sec=([\d.]+)", done.stdout)
    if mt:
        tok_s = float(mt.group(1))

    # device-side context memory, straight from the breakdown table
    dev_ctx_mib = None
    md = re.search(r"MTL0.*?\(\s*\d+\s*=\s*\d+\s*\+\s*(\d+)\s*\+", done.stderr)
    if md:
        dev_ctx_mib = int(md.group(1))

    return {
        "tag": tag, "role": role, "ctx": ctx, "policy": policy, "rc": done.returncode,
        "text": text, "token_ids": token_ids, "n_tokens": len(token_ids),
        "spill_events": spill_events, "retained": retained,
        "kv_device_mib": kv_mib, "device_ctx_mib": dev_ctx_mib,
        "host_store_mib": host_mib, "tok_per_sec": tok_s,
        "passkey": PASSKEY in text,
    }


def first_id_divergence(a: list, b: list) -> int:
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return i
    return -1 if len(a) == len(b) else min(len(a), len(b))


def main() -> int:
    for d in (RESULTS_DIR, PROMPTS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    for p in (COMPLETION_BIN, TOKENIZE_BIN, MODEL_PATH):
        if not p.exists():
            raise FileNotFoundError(p)

    prompt, ntok = build_prompt()
    print(f"prompt tokens = {ntok}   gen = {GEN_TOKENS}   seed = {SEED}   greedy, -fa off\n")

    results = {}
    for tag, ctx, policy, role in ARMS:
        print(f"running {tag:16s} (policy {policy} @ C={ctx})  {role} ...", flush=True)
        results[tag] = run_arm(prompt, ctx, policy, tag, role)

    print("\n" + "=" * 96)
    hdr = f"{'arm':16s} {'C':>5s} {'pol':>3s} {'spills':>6s} {'retain':>6s} {'kvMiB':>7s} {'tok/s':>6s} {'ids':>4s} {'passkey':>7s}"
    print(hdr)
    print("-" * 96)
    for tag, *_ in ARMS:
        a = results[tag]
        rate = f"{a['tok_per_sec']:.2f}" if a["tok_per_sec"] is not None else "n/a"
        kv   = f"{a['kv_device_mib']:.0f}" if a["kv_device_mib"] is not None else "n/a"
        print(f"{tag:16s} {a['ctx']:5d} {a['policy']:3d} {a['spill_events']:6d} "
              f"{a['retained']:6d} {kv:>7s} {rate:>6s} "
              f"{a['n_tokens']:4d} {str(a['passkey']):>7s}")
    print("=" * 96)

    problems = []

    for tag, *_ in ARMS:
        a = results[tag]
        if a["rc"] != 0:
            problems.append(f"{tag}: nonzero return code {a['rc']}")
        if not a["token_ids"]:
            problems.append(f"{tag}: no sampled token ids captured")
        if a["tok_per_sec"] is None:
            problems.append(f"{tag}: no throughput recorded")
        elif a["tok_per_sec"] >= TOK_S_CEILING:
            problems.append(f"{tag}: {a['tok_per_sec']:.2f} tok/s >= {TOK_S_CEILING} ceiling — STOP")

    # arm-specific structural invariants
    if results["arm2_p3_c4096"]["spill_events"] != 0:
        problems.append("arm2 spilled — p3 is not inert at C=4096")
    if results["arm3_p3_c1024"]["spill_events"] == 0:
        problems.append("arm3 never spilled — no memory pressure, gate is vacuous")
    if results["arm4_p1_c1024"]["spill_events"] == 0:
        problems.append("arm4 never evicted — loss control is vacuous")

    # exactness on sampled token IDs
    ref_ids = results[EXACT_ARMS[0]]["token_ids"]
    exact_ok = True
    for tag in EXACT_ARMS[1:]:
        ids = results[tag]["token_ids"]
        idx = first_id_divergence(ref_ids, ids)
        if idx != -1:
            exact_ok = False
            problems.append(
                f"{tag}: token-ID divergence vs {EXACT_ARMS[0]} at step {idx} "
                f"(ref={ref_ids[idx] if idx < len(ref_ids) else 'EOS'}, "
                f"got={ids[idx] if idx < len(ids) else 'EOS'}) — greedy+seed is "
                f"deterministic, so this indicates nondeterminism in the spill path")

    passkey_ok = all(results[t]["passkey"] for t in EXACT_ARMS)
    if not passkey_ok:
        problems.append("one of arms 1/2/3 failed to retrieve the passkey")

    loss_ok = not results[LOSS_ARM]["passkey"]
    if not loss_ok:
        problems.append(f"{LOSS_ARM} retrieved the passkey — loss control is not a control")

    verdict = "PASS" if (exact_ok and passkey_ok and loss_ok and not problems) else "FAIL"

    print(f"\ntoken-ID exactness (arms 1/2/3): {'IDENTICAL' if exact_ok else 'DIVERGED'}")
    print(f"passkey  arms 1/2/3: {[results[t]['passkey'] for t in EXACT_ARMS]}   arm4: {results[LOSS_ARM]['passkey']}")
    print(f"device KV: C=4096 -> {results['arm1_p0_c4096']['kv_device_mib']} MiB   "
          f"C=1024 -> {results['arm3_p3_c1024']['kv_device_mib']} MiB")
    print(f"recall-cost anchor: p3@4096 {results['arm2_p3_c4096']['tok_per_sec']} tok/s "
          f"-> p3@1024 {results['arm3_p3_c1024']['tok_per_sec']} tok/s (single pair, thermal-noisy)")

    print("\n" + "=" * 96)
    print(f"VERDICT: {verdict}")
    for p in problems:
        print(f"  ! {p}")
    print("=" * 96)

    out = RESULTS_DIR / "spill_gate.csv"
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "role", "ctx", "policy", "rc", "spill_events", "retained",
                    "kv_device_mib", "device_ctx_mib", "host_store_mib", "tok_per_sec",
                    "n_tokens", "passkey", "verdict", "token_ids", "text"])
        for tag, *_ in ARMS:
            a = results[tag]
            w.writerow([a["tag"], a["role"], a["ctx"], a["policy"], a["rc"],
                        a["spill_events"], a["retained"], a["kv_device_mib"],
                        a["device_ctx_mib"], a["host_store_mib"], a["tok_per_sec"],
                        a["n_tokens"], a["passkey"], verdict,
                        " ".join(str(t) for t in a["token_ids"]), a["text"]])
    print(f"wrote {out}")

    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
