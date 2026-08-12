#!/usr/bin/env python3
"""
Path B Item 2, Option A — HONEST quality contrast via a completion.cpp patch that
lets UNIKV_POLICY!=0 ingest a prompt longer than the KV cache (evicting during
prefill through the in-library UniKV policy, NOT the stock front-preserving shift).

GATE-1 (run FIRST; contrast is untrusted unless this passes):
  A prompt that FITS the cache is decoded two ways that differ ONLY in how the
  prefill is split into llama_decode() calls:
    - batched   : -b 256   (several decode calls)
    - single    : -b 1024  (one decode call)
  Both pin -ub 256 (identical ubatch tiling => identical matmul reduction order =>
  no batch-size floating-point drift). Greedy (--temp 0). Therefore any output
  difference is position/n_past corruption from multi-call chunking — which is the
  exact machinery the long-prompt contrast relies on. Outputs must be IDENTICAL.
  If not identical -> STOP, do not run the contrast, fall back to Option D.

CONTRAST (only if Gate-1 identical):
  Same long prompt (passkey early + long filler + trailing question), L >> C.
    - policy-2 arm: UNIKV_POLICY=2, -c 4096 (>= L)  -> retains -> passkey RETRIEVED
    - policy-1 arm: UNIKV_POLICY=1, -c 1024 (< L)   -> evicts  -> passkey LOST,
                    and shift_events > 0 (proof the in-library eviction fired).

Greedy/deterministic; reports NO tok/s. Strips the UNIKV_E2E stdout marker before
any comparison/matching. Results -> quality_results/ (mirrors alpha_results/).
"""

from __future__ import annotations

import csv
import os
import re
import subprocess
from pathlib import Path

ROOT           = Path(__file__).resolve().parents[1]
LLAMA_DIR      = ROOT / "llama.cpp"
BIN_DIR        = LLAMA_DIR / "build-m4pro-metal" / "bin"
COMPLETION_BIN = BIN_DIR / "llama-completion"
TOKENIZE_BIN   = BIN_DIR / "llama-tokenize"
MODEL_PATH     = LLAMA_DIR / "models" / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

RESULTS_DIR    = ROOT / "quality_results"
ARTIFACT_DIR   = ROOT / "artifacts" / "quality_probe"
PROMPTS_DIR    = ARTIFACT_DIR / "prompts"
LOGS_DIR       = ARTIFACT_DIR / "logs"

PASSKEY      = "48291"
FILLER       = ("The grass is green. The sky is blue. The sun is yellow. "
                "Here we go. There and back again. ")
INTRO        = ("There is an important piece of information hidden inside the "
                "text below. Find it and remember it, because you will be asked "
                "about it at the end.\n\n")
PASSKEY_LINE = f"\nThe pass key is {PASSKEY}. Remember it. {PASSKEY} is the pass key.\n\n"
QUESTION     = "\nQuestion: What is the pass key?\nAnswer: The pass key is"

GEN_TOKENS   = 32
THREADS      = 10
GPU_LAYERS   = 999
SEED         = 123
SMALL_C      = 1024   # resident cache for the policy-1 (eviction) arm
BIG_C        = 4096   # cache for the policy-2 (retention) arm


def ensure_paths() -> None:
    for d in (RESULTS_DIR, PROMPTS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def must_exist(p: Path) -> None:
    if not p.exists():
        raise FileNotFoundError(f"Required path not found: {p}")


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


def build_prompt(pre_repeat: int, post_repeat: int, name: str) -> tuple[Path, int, int]:
    prefix = INTRO + (FILLER * pre_repeat) + PASSKEY_LINE
    full   = prefix + (FILLER * post_repeat) + QUESTION
    path   = PROMPTS_DIR / name
    path.write_text(full)
    return path, count_tokens(full, f"_c_{name}"), count_tokens(prefix, f"_p_{name}")


def strip_marker(stdout: str) -> str:
    """Drop the UNIKV_E2E stdout line (its decode_ms varies run-to-run)."""
    return stdout.split("UNIKV_E2E")[0]


def run_completion(prompt_path: Path, ctx: int, policy: int, batch: int,
                   ubatch: int, tag: str, sink: int = 0) -> tuple[str, int, Path]:
    step_csv = LOGS_DIR / f"unikv_log_{tag}.csv"
    step_csv.unlink(missing_ok=True)
    args = [
        str(COMPLETION_BIN), "-m", str(MODEL_PATH), "-f", str(prompt_path),
        "-n", str(GEN_TOKENS), "-c", str(ctx), "-b", str(batch), "-ub", str(ubatch),
        "-ngl", str(GPU_LAYERS), "-t", str(THREADS), "-fa", "on", "-fit", "off",
        "--temp", "0", "--seed", str(SEED), "--no-warmup", "--simple-io",
        "--no-display-prompt", "-no-cnv",
    ]
    env = os.environ.copy()
    env.update({"UNIKV_POLICY": str(policy), "UNIKV_LOG": str(step_csv),
                "UNIKV_SINK": str(sink)})
    completed = subprocess.run(
        args, cwd=LLAMA_DIR, env=env, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, errors="replace",
    )
    (LOGS_DIR / f"gen_{tag}.txt").write_text(
        "=== STDOUT ===\n" + completed.stdout +
        "\n=== STDERR (tail) ===\n" + completed.stderr[-3000:])
    return completed.stdout, completed.returncode, step_csv


def max_shift_events(csv_path: Path) -> tuple[int, str]:
    if not csv_path.exists():
        return 0, ""
    with csv_path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    mx = max((int(r.get("shift_events", 0) or 0) for r in rows), default=0)
    first = next((r["step"] for r in rows if int(r.get("shift_events", 0) or 0) > 0), "")
    return mx, first


def gate1() -> bool:
    print("=" * 68)
    print("GATE-1: batched (-b 256) vs single-shot (-b 1024), -ub 256 both, fits cache")
    print("=" * 68)
    path, ntok, pk_end = build_prompt(5, 28, "gate_prompt.txt")
    print(f"  gate prompt tokens = {ntok} (passkey ends ~{pk_end}); must fit c={SMALL_C}")
    assert ntok + GEN_TOKENS < SMALL_C, "gate prompt must fit the cache (no overflow)"

    # (a) primary: policy-2, no sink. (b) regression: policy-1 sink=4 (sink code must
    # be inert without overflow -> still byte-identical). Both must be identical.
    ok = True
    for label, policy, sink in [("policy2_sink0", 2, 0), ("policy1_sink4", 1, 4)]:
        out_b, rc_b, _ = run_completion(path, SMALL_C, policy, 256, 256,
                                        f"gate_batched_{label}", sink=sink)
        out_s, rc_s, _ = run_completion(path, SMALL_C, policy, 1024, 256,
                                        f"gate_single_{label}", sink=sink)
        tb, ts = strip_marker(out_b), strip_marker(out_s)
        identical = (rc_b == 0 and rc_s == 0 and tb == ts)
        print(f"  [{label}] batched rc={rc_b} single rc={rc_s} -> IDENTICAL: "
              f"{'YES' if identical else 'NO'}")
        print(f"      out: {' '.join(tb.split())[:100]!r}")
        if not identical:
            (RESULTS_DIR / f"gate1_batched_{label}.txt").write_text(tb)
            (RESULTS_DIR / f"gate1_single_{label}.txt").write_text(ts)
        ok = ok and identical
    return ok


def contrast() -> None:
    print("\n" + "=" * 68)
    print("CONTRAST: policy-2 retains (c=4096) vs policy-1 evicts (c=1024)")
    print("=" * 68)
    path, ntok, pk_end = build_prompt(7, 150, "contrast_prompt.txt")
    print(f"  contrast prompt tokens = {ntok} (passkey ends ~{pk_end})")
    checks = {
        "policy-1 arm overflows (L > small c)":       ntok > SMALL_C,
        "policy-2 arm fits (L+gen < big c)":          ntok + GEN_TOKENS < BIG_C,
        "passkey outside policy-1 retained tail":     pk_end < (ntok - SMALL_C),
    }
    for k, v in checks.items():
        print(f"    [{'ok' if v else 'FAIL'}] {k}")
    assert all(checks.values()), "contrast preconditions not met"

    rows = []
    # Four arms: UniKV retention; StreamingLLM (sink-preserving, k=4); naive rolling
    # (sink=0); and an honesty CONTROL (policy-1 big cache, no eviction) that must
    # also retrieve. distinct_word_ratio is an objective loop signal (a degenerate
    # loop -> few distinct words -> low ratio; coherent text -> high ratio).
    arms = [
        ("UniKV_retention", 2, BIG_C,   0, "contrast_policy2_c4096"),
        ("StreamingLLM_k4", 1, SMALL_C, 4, "contrast_streamingllm_c1024_sink4"),
        ("naive_rolling",   1, SMALL_C, 0, "contrast_naive_c1024_sink0"),
        ("control_p1_bigC", 1, BIG_C,   0, "control_policy1_c4096"),
    ]
    for label, policy, ctx, sink, tag in arms:
        out, rc, step_csv = run_completion(path, ctx, policy, 512, 512, tag, sink=sink)
        text = strip_marker(out).strip()
        found = PASSKEY in text
        se, first = max_shift_events(step_csv)
        words = text.split()
        dwr = round(len(set(words)) / len(words), 3) if words else 0.0
        rows.append({
            "arm": label, "policy": policy, "ctx": ctx, "sink": sink,
            "prompt_tokens": ntok, "passkey_end_pos": pk_end,
            "gen_tokens": GEN_TOKENS, "passkey": PASSKEY,
            "passkey_found": int(found), "shift_events": se,
            "first_shift_step": first, "distinct_word_ratio": dwr, "rc": rc,
            "output_verbatim": text.replace("\n", " "),
        })

    fields = ["arm", "policy", "ctx", "sink", "prompt_tokens", "passkey_end_pos",
              "gen_tokens", "passkey", "passkey_found", "shift_events",
              "first_shift_step", "distinct_word_ratio", "rc", "output_verbatim"]
    out_csv = RESULTS_DIR / "quality_probe_A.csv"
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print("\n  RESULT:")
    for r in rows:
        verdict = "RETRIEVED" if r["passkey_found"] else "LOST"
        print(f"    {r['arm']:16} policy={r['policy']} c={r['ctx']:>4} sink={r['sink']}  "
              f"passkey={verdict:9} shift={r['shift_events']:>3} "
              f"distinct_word_ratio={r['distinct_word_ratio']:.3f} rc={r['rc']}")
        print(f"        out: {r['output_verbatim'][:200]!r}")
    print(f"\n  CSV: {out_csv}")


def main() -> None:
    ensure_paths()
    for p in (COMPLETION_BIN, TOKENIZE_BIN, MODEL_PATH):
        must_exist(p)
    if not gate1():
        print("\nGATE-1 FAILED — batched ingest is not position-faithful. "
              "STOP. Do NOT trust a contrast. Fall back to Option D.")
        raise SystemExit(2)
    print("\nGATE-1 PASSED — batched ingest is position-faithful. Proceeding to contrast.")
    contrast()
    print("Done.")


if __name__ == "__main__":
    main()
