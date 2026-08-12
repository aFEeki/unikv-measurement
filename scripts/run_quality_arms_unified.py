#!/usr/bin/env python3
"""All passkey arms in ONE harness at ONE batch configuration (review M5).

The published Table 2 was assembled across two harnesses: run_spill_gate.py
(-b 256 -ub 256, flash attention off) supplied the UniKV, no-eviction-reference
and naive-window rows, while run_quality_probe_A.py ran flash attention ON and
recorded shift_events under a different definition (a per-call flag, not a
count), and the flash-attention-off StreamingLLM re-run used a third batch
tiling (512). So whichever provenance the naive row had, the sentence "all four
arms run with flash attention disabled, so the sink is the sole difference
between the eviction runs" could not be true as written.

This harness runs every arm in one process sequence, one batch configuration,
flash attention off everywhere, and records a per-arm demotion count under a
single definition: the number of decode calls whose shift_events flag fired
(= total spill events for policy 3, total shift events for policy 1). The two
eviction arms must therefore be comparable on that column, which is what makes
"the sink is the sole difference" checkable rather than asserted.

Arms (identical prompt, seed, greedy decode, -fa off, -b/-ub 256):

  ref_p0_c4096          policy 0, C=4096   no-eviction reference (stock)
  idle_p3_c4096         policy 3, C=4096   UniKV with nothing spilled
  unikv_p3_c1024        policy 3, C=1024   UniKV lossless spill-and-recall
  streamingllm_p1_c1024 policy 1, C=1024, UNIKV_SINK=4   sink-preserving control
  naive_p1_c1024        policy 1, C=1024, UNIKV_SINK=0   naive rolling window
  h2o_p4_c1024          policy 4, C=1024   H2O fixed-budget eviction (prior art)

Gates: ref / idle / unikv emit identical sampled token IDs and all retrieve the
passkey; both eviction arms lose it. Exactness is compared on sampled token IDs
(UNIKV_TOKEN_LOG), not on re-encoded output text.
"""

import csv
import hashlib
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

RESULTS_DIR  = ROOT / "quality_results"
ARTIFACT_DIR = ROOT / "artifacts" / "quality_unified"
PROMPTS_DIR  = ARTIFACT_DIR / "prompts"
LOGS_DIR     = ARTIFACT_DIR / "logs"

# identical to run_spill_gate.py / run_quality_probe_A.py so the passkey sits at
# the same position (~219) and prompt length (3834 tokens) as the published table
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
BATCH      = 256          # ONE batch configuration for every arm
UBATCH     = 256
SPILL_CAP  = 8192
COOLDOWN_S = int(os.environ.get("UNIKV_QA_COOLDOWN", "0"))
CEILING    = 58.0

# (tag, ctx, policy, sink, role)
ARMS = [
    ("ref_p0_c4096",          4096, 0, 0, "no-eviction reference (stock)"),
    ("idle_p3_c4096",         4096, 3, 0, "UniKV idle, nothing spills"),
    ("unikv_p3_c1024",        1024, 3, 0, "UniKV lossless spill-and-recall"),
    ("streamingllm_p1_c1024", 1024, 1, 4, "StreamingLLM k=4 (sink control)"),
    ("naive_p1_c1024",        1024, 1, 0, "naive rolling window"),
    ("h2o_p4_c1024",          1024, 4, 0, "H2O fixed-budget eviction (prior art)"),
]

EXACT_ARMS = ["ref_p0_c4096", "idle_p3_c4096", "unikv_p3_c1024"]
# Window-eviction arms, gated to lose the passkey: a size-C window whose span
# excludes position ~219 cannot contain it. This is a structural guarantee.
LOSS_ARMS  = ["streamingllm_p1_c1024", "naive_p1_c1024"]
# The two arms that differ ONLY in the sink, for the controlled comparison.
EVICT_ARMS = LOSS_ARMS
# H2O is deliberately NOT gated either way. It selects by accumulated attention
# rather than by recency, so it CAN retain a token at position 219, and whether
# it does is the result this arm exists to measure -- not a precondition of the
# run. Gating it to lose would be exactly the strawman this comparator is meant
# to avoid.
COMPARATOR_ARMS = ["h2o_p4_c1024"]


def count_tokens(text: str, tmp_name: str) -> int:
    tmp = PROMPTS_DIR / tmp_name
    tmp.write_text(text)
    out = subprocess.run(
        [str(TOKENIZE_BIN), "-m", str(MODEL_PATH), "-f", str(tmp),
         "--show-count", "--log-disable"],
        cwd=LLAMA_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, errors="replace", check=True).stdout
    m = re.search(r"Total number of tokens:\s*(\d+)", out)
    if not m:
        raise RuntimeError(f"token count parse failed:\n{out}")
    return int(m.group(1))


def build_prompt() -> tuple[Path, int]:
    full = INTRO + (FILLER * 7) + PASSKEY_LINE + (FILLER * 150) + QUESTION
    path = PROMPTS_DIR / "quality_unified_prompt.txt"
    path.write_text(full)
    return path, count_tokens(full, "_count_tmp.txt")


def run_arm(prompt: Path, tag: str, ctx: int, policy: int, sink: int, role: str) -> dict:
    step_csv = LOGS_DIR / f"step_{tag}.csv"
    tok_log  = LOGS_DIR / f"tokens_{tag}.txt"
    step_csv.unlink(missing_ok=True)
    tok_log.unlink(missing_ok=True)

    args = [
        str(COMPLETION_BIN), "-m", str(MODEL_PATH), "-f", str(prompt),
        "-n", str(GEN_TOKENS), "-c", str(ctx), "-b", str(BATCH), "-ub", str(UBATCH),
        "-ngl", str(GPU_LAYERS), "-t", str(THREADS), "-fa", "off", "-fit", "off",
        "--temp", "0", "--seed", str(SEED), "--no-warmup", "--simple-io",
        "--no-display-prompt", "-no-cnv",
    ]
    env = os.environ.copy()
    env.update({
        "UNIKV_POLICY": str(policy),
        "UNIKV_SINK": str(sink),
        "UNIKV_LOG": str(step_csv),
        "UNIKV_TOKEN_LOG": str(tok_log),
        "UNIKV_SPILL_CAP": str(SPILL_CAP),
        "UNIKV_MEM_BREAKDOWN": "1",
    })
    if policy == 4:
        # survivor dump: (position, cell, accumulated mass) after the last
        # eviction, so a retrieval result can be attributed to the ranking
        # rather than taken on trust
        env["UNIKV_H2O_TRACE"] = str(LOGS_DIR / f"h2o_trace_{tag}.csv")

    done = subprocess.run(args, cwd=LLAMA_DIR, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True, errors="replace")
    (LOGS_DIR / f"gen_{tag}.txt").write_text(
        "=== STDOUT ===\n" + done.stdout + "\n=== STDERR ===\n" + done.stderr)

    text = done.stdout.split("UNIKV_E2E")[0].strip()

    token_ids = []
    if tok_log.exists():
        token_ids = [int(l) for l in tok_log.read_text().split() if l.strip()]

    # ONE definition for every arm: decode calls whose demotion flag fired.
    demotion_calls, first_demotion = 0, None
    if step_csv.exists():
        with step_csv.open(newline="") as fh:
            for r in csv.DictReader(fh):
                if int(r.get("shift_events", 0) or 0) > 0:
                    demotion_calls += 1
                    if first_demotion is None:
                        first_demotion = int(r["step"])

    retained = 0
    rm = re.findall(r"retained=(\d+)", done.stderr)
    if rm:
        retained = int(rm[-1])

    mfa = re.search(r"flash_attn\s*=\s*(\w+)", done.stderr)
    mk  = re.search(r"llama_kv_cache:\s+size\s+=\s+([\d.]+)\s+MiB", done.stderr)
    mh  = re.search(r"host spill store\s+=\s+([\d.]+)\s+MiB", done.stderr)
    mb  = re.search(r"n_batch\s*=\s*(\d+)", done.stderr)
    mu  = re.search(r"n_ubatch\s*=\s*(\d+)", done.stderr)
    mt  = re.search(r"UNIKV_E2E .*tok_per_sec=([\d.]+)", done.stdout)

    # Two definitions kept side by side: `distinct_word_ratio` is
    # case-SENSITIVE, matching run_quality_probe_A.py so the numbers are
    # directly comparable to the published Table 2; the case-folded variant is
    # the slightly stricter loop indicator (it merges "The"/"the").
    words = text.split()
    dwr = round(len(set(words)) / len(words), 3) if words else 0.0
    wl = text.lower().split()
    dwr_cf = round(len(set(wl)) / len(wl), 3) if wl else 0.0

    return {
        "arm": tag, "role": role, "ctx": ctx, "policy": policy, "sink": sink,
        "rc": done.returncode, "text": text, "token_ids": token_ids,
        "token_md5": hashlib.md5(
            " ".join(str(t) for t in token_ids).encode()).hexdigest(),
        "demotion_calls": demotion_calls, "first_demotion_step": first_demotion,
        "retained": retained,
        "flash_attn": mfa.group(1) if mfa else "UNPARSED",
        "kv_device_mib": float(mk.group(1)) if mk else None,
        "host_store_mib": float(mh.group(1)) if mh else None,
        "n_batch": int(mb.group(1)) if mb else None,
        "n_ubatch": int(mu.group(1)) if mu else None,
        "tok_per_sec": float(mt.group(1)) if mt else None,
        "distinct_word_ratio": dwr, "distinct_word_ratio_casefold": dwr_cf,
        "passkey_found": PASSKEY in text,
    }


def first_divergence(a: list, b: list) -> int:
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
    print(f"prompt {ntok} tokens, gen {GEN_TOKENS}, seed {SEED}, greedy, -fa off, "
          f"-b {BATCH} -ub {UBATCH} on every arm\n")

    res = {}
    for i, (tag, ctx, policy, sink, role) in enumerate(ARMS):
        if COOLDOWN_S > 0 and i > 0:
            print(f"  cooldown {COOLDOWN_S}s ...", flush=True)
            time.sleep(COOLDOWN_S)
        print(f"running {tag:22s} (policy {policy} @ C={ctx}, sink={sink}) ...",
              end="", flush=True)
        t0 = time.time()
        res[tag] = run_arm(prompt, tag, ctx, policy, sink, role)
        print(f" {time.time() - t0:.0f}s  fa={res[tag]['flash_attn']}  "
              f"demotions={res[tag]['demotion_calls']}  "
              f"passkey={res[tag]['passkey_found']}")

    print("\n" + "=" * 104)
    print(f"{'arm':22s} {'C':>5s} {'pol':>3s} {'sink':>4s} {'demot':>6s} "
          f"{'retain':>6s} {'kvMiB':>6s} {'tok/s':>6s} {'dwr':>5s} {'fa':>9s} {'passkey':>7s}")
    print("-" * 104)
    for tag, *_ in ARMS:
        a = res[tag]
        kv = f"{a['kv_device_mib']:.0f}" if a["kv_device_mib"] is not None else "n/a"
        ts = f"{a['tok_per_sec']:.2f}" if a["tok_per_sec"] is not None else "n/a"
        print(f"{tag:22s} {a['ctx']:5d} {a['policy']:3d} {a['sink']:4d} "
              f"{a['demotion_calls']:6d} {a['retained']:6d} {kv:>6s} {ts:>6s} "
              f"{a['distinct_word_ratio']:5.3f} {a['flash_attn']:>9s} "
              f"{str(a['passkey_found']):>7s}")
    print("=" * 104)

    problems = []
    for tag, *_ in ARMS:
        a = res[tag]
        if a["rc"] != 0:
            problems.append(f"{tag}: rc={a['rc']}")
        if a["flash_attn"] != "disabled":
            problems.append(f"{tag}: flash_attn={a['flash_attn']} — must be off on every arm")
        if not a["token_ids"]:
            problems.append(f"{tag}: no sampled token IDs captured")
        if a["tok_per_sec"] is not None and a["tok_per_sec"] >= CEILING:
            problems.append(f"{tag}: {a['tok_per_sec']:.2f} tok/s >= {CEILING} ceiling — STOP")

    batches = {(res[t]["n_batch"], res[t]["n_ubatch"]) for t, *_ in ARMS}
    if len(batches) != 1:
        problems.append(f"batch tiling differs across arms: {batches}")

    if res["idle_p3_c4096"]["demotion_calls"] != 0:
        problems.append("idle_p3_c4096 spilled — policy 3 is not inert at C=4096")
    for t in EVICT_ARMS + ["unikv_p3_c1024"] + COMPARATOR_ARMS:
        if res[t]["demotion_calls"] == 0:
            problems.append(f"{t}: never demoted — arm is vacuous")

    ev = {t: res[t]["demotion_calls"] for t in EVICT_ARMS}
    sink_sole_difference = len(set(ev.values())) == 1

    ref = res[EXACT_ARMS[0]]["token_ids"]
    exact_ok = True
    for tag in EXACT_ARMS[1:]:
        idx = first_divergence(ref, res[tag]["token_ids"])
        if idx != -1:
            exact_ok = False
            problems.append(f"{tag}: token-ID divergence vs {EXACT_ARMS[0]} at step {idx}")

    passkey_ok = all(res[t]["passkey_found"] for t in EXACT_ARMS)
    if not passkey_ok:
        problems.append("a lossless arm failed to retrieve the passkey")
    loss_ok = all(not res[t]["passkey_found"] for t in LOSS_ARMS)
    if not loss_ok:
        problems.append("an eviction arm retrieved the passkey — control is not a control")

    verdict = "PASS" if (exact_ok and passkey_ok and loss_ok and not problems) else "FAIL"

    print(f"\ntoken-ID exactness {EXACT_ARMS}: "
          f"{'IDENTICAL' if exact_ok else 'DIVERGED'}  (md5 "
          f"{res[EXACT_ARMS[0]]['token_md5'][:12]}…)")
    print(f"eviction-arm demotion counts: {ev}  -> sink is the sole difference: "
          f"{sink_sole_difference}")
    print(f"no-overhead paired datum @ C=4096 (same harness, same block): stock "
          f"{res['ref_p0_c4096']['tok_per_sec']} vs UniKV-idle "
          f"{res['idle_p3_c4096']['tok_per_sec']} tok/s")

    for t in COMPARATOR_ARMS:
        a = res[t]
        print(f"\ncomparator {t}: passkey {'RETRIEVED' if a['passkey_found'] else 'LOST'}, "
              f"{a['demotion_calls']} demotions, dwr {a['distinct_word_ratio']}")
        naive = res["naive_p1_c1024"]
        if a["passkey_found"] and not naive["passkey_found"]:
            print("  -> beats the naive rolling window at the same budget "
                  "(sanity check on the implementation: PASS)")
        elif not a["passkey_found"]:
            print("  -> does NOT beat the naive window. Per the pre-registered rule, "
                  "suspect the implementation before the baseline.")

    print("\n" + "=" * 104)
    print(f"VERDICT: {verdict}")
    for p in problems:
        print(f"  ! {p}")
    print("=" * 104)

    out = RESULTS_DIR / "quality_arms_unified.csv"
    cols = ["arm", "role", "ctx", "policy", "sink", "rc", "flash_attn", "n_batch",
            "n_ubatch", "prompt_tokens", "gen_tokens", "demotion_calls",
            "first_demotion_step", "retained", "kv_device_mib", "host_store_mib",
            "tok_per_sec", "distinct_word_ratio", "distinct_word_ratio_casefold",
            "passkey_found", "token_md5",
            "sink_sole_difference", "verdict", "token_ids", "output_verbatim"]
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for tag, *_ in ARMS:
            a = res[tag]
            w.writerow([a["arm"], a["role"], a["ctx"], a["policy"], a["sink"],
                        a["rc"], a["flash_attn"], a["n_batch"], a["n_ubatch"],
                        ntok, GEN_TOKENS, a["demotion_calls"],
                        a["first_demotion_step"], a["retained"],
                        a["kv_device_mib"], a["host_store_mib"], a["tok_per_sec"],
                        a["distinct_word_ratio"], a["distinct_word_ratio_casefold"],
                        a["passkey_found"],
                        a["token_md5"], sink_sole_difference, verdict,
                        " ".join(str(t) for t in a["token_ids"]), a["text"]])
    print(f"wrote {out}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
