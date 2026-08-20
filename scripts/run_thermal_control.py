#!/usr/bin/env python3
"""Item 3 — the constant-work thermal control. Converts a bound into a measurement.

Section 4 reports both per-cell terms as UPPER BOUNDS because part of the
measured step-time rise with n_spill could be heat rather than recall: a run at a
large spill target spends far longer under load before its measurement window
opens, so the machine is hotter when the window is measured.

THE CONTROL: policy 1 at a pinned C=1024 window does CONSTANT work per step and
spills nothing. Run it for the same elapsed time and read its step time over the
same wall-clock band; whatever it has drifted is thermal, because nothing else
about its work has changed.

MATCHED ON ELAPSED WALL-CLOCK, NOT ON STEP COUNT. The brief says burst duration;
the timeline says it has to be more than that. The bursts themselves are 2.2-5.5 s
and differ little across targets, but the PROCESS durations are 4.7 s to 94.3 s
(CPU-pinned) and 4.7 s to 45.5 s (device-visible) because the prefill that
establishes the spilled tier dominates. Matching burst duration alone would leave
almost all of the heat uncontrolled. So each control is matched to the elapsed
time at which the paired isochronal run's measurement window opens and closes:

   mode            target   T_start_s   T_end_s   window_s
   CPU-pinned           0        2.5       4.7      2.17
   CPU-pinned         512        6.5       9.8      3.26
   CPU-pinned        1024        9.0      12.3      3.33
   CPU-pinned        2048       15.4      19.0      3.59
   CPU-pinned        4096       33.4      37.6      4.24
   CPU-pinned        8192       88.8      94.3      5.51
   device-visible       0        2.5       4.7      2.16
   device-visible     512        5.2       7.7      2.55
   device-visible    1024        6.9       9.6      2.65
   device-visible    2048       10.4      13.3      2.86
   device-visible    4096       18.7      22.0      3.29
   device-visible    8192       41.4      45.5      4.13

These bands are DESIGN CONSTANTS taken from the published isochronal block
(stress_results/b2_isochronal_both_modes.csv, mean over its three trials). They
choose how long each control runs; they are not used in any arithmetic. Every
number in the analysis comes from THIS block.

NO SPLICING. The isochronal arms are re-run here, interleaved with the controls
in the same randomised complete block, so the correction subtracts a control
measured in the same block from an isochronal value measured in the same block.
Correcting the OLD block with a NEW control would have been a cross-block
operation of exactly the kind the rules forbid.

Both kinds of run are INSTRUMENTED (UNIKV_LOG), as the published isochronal block
was: per-step wall clock is the measurement on both sides, so the instrumentation
is common-mode and cancels in the subtraction.

CONTROL PROMPT: 1000 tokens at C=1024, so the window is full after 24 generated
tokens (~0.7 s) and every measured band — the earliest opens at 2.5 s — sits in
the constant-work regime. A 512-token prompt would have left the earliest band
inside the still-growing-cache regime and broken the control.

DESIGN: randomised complete block. Each round is a random permutation of all 24
cells (12 configurations x {isochronal, control}), so kind and target are
orthogonal to position by construction. Results are written after every run and
an interim fit is printed after each round, so stopping after round 1 or 2 still
leaves a balanced design with fewer trials.
"""

import csv, datetime, math, os, random, re, statistics as st, subprocess, sys, time
from pathlib import Path

ROOT           = Path(__file__).resolve().parents[1]
LLAMA_DIR      = ROOT / "llama.cpp"
BIN            = LLAMA_DIR / "build-m4pro-metal" / "bin"
COMPLETION_BIN = BIN / "llama-completion"
TOKENIZE_BIN   = BIN / "llama-tokenize"
MODEL_PATH     = Path(os.environ.get(
    "UNIKV_MODEL", LLAMA_DIR / "models" / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"))

RESULTS_DIR = ROOT / "stress_results"
ART_DIR     = ROOT / "artifacts" / "thermal_control"
PROMPTS_DIR = ART_DIR / "prompts"
LOGS_DIR    = ART_DIR / "logs"

ISO_CTX, ISO_BURST, ISO_WARMUP = 1024, 128, 32
TARGETS   = [0, 512, 1024, 2048, 4096, 8192]
CTRL_PROMPT = 1000
TRIALS     = int(os.environ.get("UNIKV_TC_TRIALS", "3"))
COOLDOWN_S = int(os.environ.get("UNIKV_TC_COOLDOWN", "200"))
THREADS, GPU_LAYERS, BATCH, UBATCH, SEED = 10, 999, 512, 512, 123
FIXED_TOKEN, SHUFFLE = " token", 20260820

# (spill_dev, target) -> (T_end_s, window_s); T_start = T_end - window
BANDS = {
    ("0", 0): (4.7, 2.17), ("0", 512): (9.8, 3.26), ("0", 1024): (12.3, 3.33),
    ("0", 2048): (19.0, 3.59), ("0", 4096): (37.6, 4.24), ("0", 8192): (94.3, 5.51),
    ("1", 0): (4.7, 2.16), ("1", 512): (7.7, 2.55), ("1", 1024): (9.6, 2.65),
    ("1", 2048): (13.3, 2.86), ("1", 4096): (22.0, 3.29), ("1", 8192): (45.5, 4.13),
}


def count_tokens(path):
    out = subprocess.run([str(TOKENIZE_BIN), "-m", str(MODEL_PATH), "-f", str(path),
                          "--show-count", "--log-disable"], cwd=LLAMA_DIR,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, errors="replace", check=True).stdout
    m = re.search(r"Total number of tokens:\s*(\d+)", out)
    if not m: raise RuntimeError(f"token count parse failed:\n{out}")
    return int(m.group(1))


def ensure_prompt(n):
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    p = PROMPTS_DIR / f"prompt_{n}tok.txt"
    if p.exists() and count_tokens(p) == n: return p
    reps, seen = max(n - 1, 0), {}
    for _ in range(16):
        p.write_text(FIXED_TOKEN * reps)
        got = seen[reps] = count_tokens(p)
        if got == n: return p
        nxt = reps + (n - got)
        if nxt in seen or nxt <= 0: break
        reps = nxt
    raise RuntimeError(f"could not build {n}-token prompt: {seen}")


def launch(prompt, ctx, policy, dev, gen, tag, spill_cap):
    step_csv = LOGS_DIR / f"step_{tag}.csv"; step_csv.unlink(missing_ok=True)
    args = [str(COMPLETION_BIN), "-m", str(MODEL_PATH), "-f", str(prompt),
            "-n", str(gen), "-c", str(ctx), "-b", str(BATCH), "-ub", str(UBATCH),
            "-ngl", str(GPU_LAYERS), "-t", str(THREADS), "-fa", "off", "-fit", "off",
            "--temp", "0", "--seed", str(SEED), "--ignore-eos", "--no-warmup",
            "--simple-io", "--no-display-prompt", "-no-cnv"]
    env = os.environ.copy()
    env.update({"UNIKV_POLICY": str(policy), "UNIKV_ALPHA": "0",
                "UNIKV_SPILL_DEV": dev, "UNIKV_SPILL_CAP": str(spill_cap),
                "UNIKV_LOG": str(step_csv)})
    for k in ("UNIKV_H2O_TRACE", "UNIKV_NO_REENCODE", "UNIKV_E2E_LOG"):
        env.pop(k, None)
    t0 = time.time()
    d = subprocess.run(args, cwd=LLAMA_DIR, env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, text=True, errors="replace")
    dur = time.time() - t0
    (LOGS_DIR / f"gen_{tag}.txt").write_text("=== STDERR ===\n" + d.stderr[-6000:])
    mfa = re.search(r"flash_attn\s*=\s*(\w+)", d.stderr)
    rows = list(csv.DictReader(step_csv.open())) if step_csv.exists() else []
    return {"rc": d.returncode, "flash_attn": mfa.group(1) if mfa else "UNPARSED",
            "duration_s": round(dur, 1), "steps": rows}


def measure_iso(rows):
    burst = rows[-ISO_BURST:] if len(rows) >= ISO_BURST else rows
    meas  = burst[ISO_WARMUP:]
    ms  = [float(x["ttft_ms"]) for x in meas]
    nsp = [int(x.get("n_spilled", 0) or 0) for x in meas]
    return (len(meas), round(st.mean(ms), 3), round(st.median(ms), 3),
            round(st.stdev(ms), 3) if len(ms) > 1 else 0.0,
            round(st.mean(nsp), 1) if nsp else None)


def measure_band(rows, t_start, t_end):
    """Steps whose cumulative elapsed time falls in [t_start, t_end] seconds."""
    ms = [float(x["ttft_ms"]) for x in rows]
    cum, acc = [], 0.0
    for v in ms:
        acc += v / 1000.0
        cum.append(acc)
    sel = [v for c, v in zip(cum, ms) if t_start <= c <= t_end]
    if len(sel) < 5:
        return None
    return (len(sel), round(st.mean(sel), 3), round(st.median(sel), 3),
            round(st.stdev(sel), 3), round(cum[-1], 2))


def main():
    for p in (RESULTS_DIR, PROMPTS_DIR, LOGS_DIR): p.mkdir(parents=True, exist_ok=True)

    # calibrate the control's instrumented step time so each run is long enough
    print("Calibrating the control's step time (instrumented) ...", end="", flush=True)
    cp = ensure_prompt(CTRL_PROMPT)
    cal = launch(cp, ISO_CTX, 1, "0", 300, "calib", 4096)
    cal_ms = st.median([float(r["ttft_ms"]) for r in cal["steps"][50:]])
    print(f" {cal_ms:.2f} ms/step (fa={cal['flash_attn']})")
    if cal["flash_attn"] != "disabled":
        raise SystemExit("STOP: calibration ran with flash attention on")

    cells = [(k, dev, t) for k in ("iso", "ctrl") for dev in ("0", "1") for t in TARGETS]
    rng = random.Random(SHUFFLE)
    plan = []
    for r in range(1, TRIALS + 1):
        rnd = cells[:]; rng.shuffle(rnd)
        plan += [(k, d, t, r) for k, d, t in rnd]

    print(f"\nthermal control: {len(plan)} runs "
          f"({len(cells)} cells x {TRIALS} rounds), {COOLDOWN_S}s cooldowns")
    print(f"  randomised complete block (seed {SHUFFLE}); each round is a full "
          f"balanced replicate,\n  so stopping after any round leaves a valid design\n")

    out = RESULTS_DIR / "thermal_control_block.csv"
    cols = ["order_idx", "round", "kind", "spill_dev", "target", "rc", "flash_attn",
            "n_steps", "ms_mean", "ms_median", "ms_sd", "n_spill_mean",
            "band_start_s", "band_end_s", "run_total_s", "duration_s", "t_start"]
    with out.open("w", newline="") as fh: csv.writer(fh).writerow(cols)

    flags = []
    for idx, (kind, dev, target, rnd) in enumerate(plan, 1):
        t_end, win = BANDS[(dev, target)]
        t_start = t_end - win
        if COOLDOWN_S:
            print(f"  [{idx:2d}/{len(plan)}] cooldown {COOLDOWN_S}s ...", flush=True)
            time.sleep(COOLDOWN_S)
        t0 = time.time()
        tag = f"{kind}_dev{dev}_n{target}_r{rnd}"
        print(f"  [{idx:2d}/{len(plan)}] r{rnd} {kind:4s} dev={dev} n={target:5d} ...",
              end="", flush=True)

        if kind == "iso":
            ptok = ISO_CTX + target if target > 0 else ISO_CTX // 2
            r = launch(ensure_prompt(ptok), ISO_CTX, 3, dev, ISO_BURST, tag,
                       max(target + 2048, 4096))
            n_steps, mm, md, sd, nsp = measure_iso(r["steps"])
            band = (None, None, None)
        else:
            gen = int(t_end * 1000 / cal_ms * 1.12) + 150
            r = launch(cp, ISO_CTX, 1, "0", gen, tag, 4096)
            b = measure_band(r["steps"], t_start, t_end)
            if b is None:
                flags.append(f"{tag}: control too short for band [{t_start},{t_end}]")
                n_steps = mm = md = sd = nsp = None; band = (t_start, t_end, None)
            else:
                n_steps, mm, md, sd, total = b
                band = (t_start, t_end, total)
                if total < t_end:
                    flags.append(f"{tag}: control ran {total}s < band end {t_end}s")

        dur = round(time.time() - t0, 1)
        print(f" {mm} ms/step over {n_steps} steps ({dur}s)")
        if r["flash_attn"] != "disabled":
            flags.append(f"{tag}: flash_attn={r['flash_attn']}")
        if r["rc"] != 0:
            flags.append(f"{tag}: rc={r['rc']}")

        with out.open("a", newline="") as fh:
            csv.writer(fh).writerow([idx, rnd, kind, dev, target, r["rc"],
                                     r["flash_attn"], n_steps, mm, md, sd, nsp,
                                     band[0], band[1], band[2], dur,
                                     datetime.datetime.fromtimestamp(t0).isoformat(timespec="seconds")])

        if idx % len(cells) == 0:
            print(f"\n  --- round {rnd} complete ({idx}/{len(plan)}) ---\n")

    print(f"\nwrote {out}")
    if flags:
        print("\n!! FLAGS:")
        for f in flags: print(f"   {f}")
    print("\nRun scripts/analyze_thermal_control.py for the corrected fit.")
    return 1 if flags else 0


if __name__ == "__main__":
    sys.exit(main())
