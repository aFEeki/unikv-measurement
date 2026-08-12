#!/usr/bin/env python3

from __future__ import annotations

import csv
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LLAMA_DIR = ROOT / "llama.cpp"
BUILD_DIR = LLAMA_DIR / "build-m4pro-metal"
BIN_DIR = BUILD_DIR / "bin"
COMPLETION_BIN = BIN_DIR / "llama-completion"
TOKENIZE_BIN = BIN_DIR / "llama-tokenize"
MODEL_PATH = LLAMA_DIR / "models" / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
OUTPUT_CSV = ROOT / "baseline_m4pro.csv"
ARTIFACT_DIR = ROOT / "artifacts" / "m4pro_baseline"
PROMPTS_DIR = ARTIFACT_DIR / "prompts"
LOGS_DIR = ARTIFACT_DIR / "logs"

MODEL_REPO = "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
MODEL_FILE = "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
MODEL_SHA256 = "7b064f5842bf9532c91456deda288a1b672397a54fa729aa665952863033557c"

CONTEXT_SIZES = [4096, 8192, 16384, 32768]
GEN_TOKENS = 256
THREADS = 10
THREADS_BATCH = 10
GPU_LAYERS = 999
BATCH_SIZE = 2048
UBATCH_SIZE = 512
SEED = 123
FIXED_PROMPT_TOKEN = " token"
WARMUP_CONTEXT = 4096
PROBE_MAX_CONTEXT = 197632

CONFIGURE_CMD = "cmake -S . -B build-m4pro-metal -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release"
BUILD_CMD = "cmake --build build-m4pro-metal --config Release -j 10"
DOWNLOAD_CMD = (
    "curl -L --fail --progress-bar -o "
    "models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf "
    "'https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/"
    "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf?download=true'"
)


@dataclass
class RunMetrics:
    ctx_size: int
    prompt_tokens_requested: int
    prompt_tokens_reported: int
    generated_tokens_reported: int
    sampling_ms: float
    prompt_eval_ms: float
    eval_ms: float
    decode_tokens_per_second: float
    ttft_ms: float
    gpu_name: str
    unified_memory: str
    offloaded_layers: str
    kv_buffer_mib: float
    full_command: str
    log_path: Path


def ensure_paths() -> None:
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def must_exist(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required path not found: {path}")


def run_capture(args: list[str], log_path: Path) -> tuple[int, str]:
    completed = subprocess.run(
        args,
        cwd=LLAMA_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    )
    log_path.write_text(completed.stdout)
    return completed.returncode, completed.stdout


def command_string(args: list[str]) -> str:
    return shlex.join(str(arg) for arg in args)


def count_tokens(prompt_path: Path) -> int:
    args = [
        str(TOKENIZE_BIN),
        "-m",
        str(MODEL_PATH),
        "-f",
        str(prompt_path),
        "--show-count",
        "--log-disable",
    ]
    completed = subprocess.run(
        args,
        cwd=LLAMA_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        check=True,
    )
    match = re.search(r"Total number of tokens:\s*(\d+)", completed.stdout)
    if not match:
        raise RuntimeError(f"Could not parse token count from tokenizer output:\n{completed.stdout}")
    return int(match.group(1))


def prompt_path_for_ctx(ctx_size: int, kind: str = "benchmark") -> Path:
    return PROMPTS_DIR / f"{kind}_prompt_ctx_{ctx_size}.txt"


def ensure_exact_prompt(ctx_size: int, prompt_tokens: int, kind: str = "benchmark") -> Path:
    prompt_path = prompt_path_for_ctx(ctx_size, kind)
    text = FIXED_PROMPT_TOKEN * max(prompt_tokens - 1, 0)
    prompt_path.write_text(text)
    actual_tokens = count_tokens(prompt_path)
    if actual_tokens != prompt_tokens:
        raise RuntimeError(
            f"Prompt token mismatch for ctx={ctx_size}: requested {prompt_tokens}, got {actual_tokens}"
        )
    return prompt_path


def common_completion_args(ctx_size: int, prompt_path: Path, n_predict: int) -> list[str]:
    return [
        str(COMPLETION_BIN),
        "-m",
        str(MODEL_PATH),
        "-f",
        str(prompt_path),
        "-n",
        str(n_predict),
        "-c",
        str(ctx_size),
        "-ngl",
        str(GPU_LAYERS),
        "-t",
        str(THREADS),
        "-tb",
        str(THREADS_BATCH),
        "-b",
        str(BATCH_SIZE),
        "-ub",
        str(UBATCH_SIZE),
        "-fa",
        "on",
        "-fit",
        "off",
        "--temp",
        "0",
        "--seed",
        str(SEED),
        "--ignore-eos",
        "--no-warmup",
        "--perf",
        "--simple-io",
        "--no-display-prompt",
        "-no-cnv",
    ]


def parse_float(pattern: str, text: str, field: str) -> float:
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        raise RuntimeError(f"Could not parse {field} from output")
    return float(match.group(1))


def parse_int(pattern: str, text: str, field: str) -> int:
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        raise RuntimeError(f"Could not parse {field} from output")
    return int(match.group(1))


def parse_text(pattern: str, text: str, field: str) -> str:
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        raise RuntimeError(f"Could not parse {field} from output")
    return match.group(1).strip()


def run_benchmark(ctx_size: int) -> RunMetrics:
    prompt_tokens = ctx_size - GEN_TOKENS
    prompt_path = ensure_exact_prompt(ctx_size, prompt_tokens, "benchmark")
    args = common_completion_args(ctx_size, prompt_path, GEN_TOKENS)
    log_path = LOGS_DIR / f"benchmark_ctx_{ctx_size}.log"
    return_code, output = run_capture(args, log_path)
    if return_code != 0:
        raise RuntimeError(f"Benchmark failed for ctx={ctx_size}. See {log_path}")

    sampling_ms = parse_float(
        r"^common_perf_print:\s+sampling time =\s+([0-9.]+) ms",
        output,
        "sampling time",
    )
    prompt_eval_ms = parse_float(
        r"^common_perf_print:\s+prompt eval time =\s+([0-9.]+) ms",
        output,
        "prompt eval time",
    )
    eval_ms = parse_float(
        r"^common_perf_print:\s+eval time =\s+([0-9.]+) ms",
        output,
        "eval time",
    )
    prompt_tokens_reported = parse_int(
        r"^common_perf_print:\s+prompt eval time =\s+[0-9.]+ ms /\s+(\d+) tokens",
        output,
        "prompt token count",
    )
    generated_tokens_reported = parse_int(
        r"^common_perf_print:\s+eval time =\s+[0-9.]+ ms /\s+(\d+) runs",
        output,
        "generated token count",
    )
    gpu_name = parse_text(r"ggml_metal_device_init: GPU name:\s+(.*)", output, "GPU name")
    unified_memory = parse_text(
        r"ggml_metal_device_init: has unified memory\s+=\s+(true|false)",
        output,
        "unified memory flag",
    )
    offloaded_layers = parse_text(
        r"load_tensors: offloaded (\d+/\d+ layers to GPU)",
        output,
        "offloaded layers",
    )
    kv_buffer_mib = parse_float(
        r"llama_kv_cache:\s+.* KV buffer size =\s+([0-9.]+) MiB",
        output,
        "KV buffer size",
    )

    if eval_ms <= 0:
        raise RuntimeError(f"Eval time was not positive for ctx={ctx_size}. See {log_path}")

    decode_tokens_per_second = generated_tokens_reported / (eval_ms / 1000.0)
    ttft_ms = prompt_eval_ms + sampling_ms

    return RunMetrics(
        ctx_size=ctx_size,
        prompt_tokens_requested=prompt_tokens,
        prompt_tokens_reported=prompt_tokens_reported,
        generated_tokens_reported=generated_tokens_reported,
        sampling_ms=sampling_ms,
        prompt_eval_ms=prompt_eval_ms,
        eval_ms=eval_ms,
        decode_tokens_per_second=decode_tokens_per_second,
        ttft_ms=ttft_ms,
        gpu_name=gpu_name,
        unified_memory=unified_memory,
        offloaded_layers=offloaded_layers,
        kv_buffer_mib=kv_buffer_mib,
        full_command=command_string(args),
        log_path=log_path,
    )


def run_warmup() -> None:
    warmup_prompt = ensure_exact_prompt(WARMUP_CONTEXT, 128, "warmup")
    args = common_completion_args(WARMUP_CONTEXT, warmup_prompt, 8)
    log_path = LOGS_DIR / "warmup.log"
    return_code, _ = run_capture(args, log_path)
    if return_code != 0:
        raise RuntimeError(f"Warmup failed. See {log_path}")


def probe_context(ctx_size: int) -> bool:
    prompt_path = ensure_exact_prompt(ctx_size, 2, "probe")
    args = common_completion_args(ctx_size, prompt_path, 1)
    log_path = LOGS_DIR / f"probe_ctx_{ctx_size}.log"
    return_code, output = run_capture(args, log_path)
    if return_code != 0:
        return False
    if "failed to initialize" in output.lower():
        return False
    if "out of memory" in output.lower():
        return False
    return True


def find_max_stable_context() -> tuple[int, int | None]:
    low = max(CONTEXT_SIZES)
    if not probe_context(low):
        raise RuntimeError(f"Baseline probe failed at ctx={low}")

    high = PROBE_MAX_CONTEXT
    if probe_context(high):
        return high, None

    first_failed = high
    while low + 1 < high:
        mid = (low + high) // 2
        if probe_context(mid):
            low = mid
        else:
            high = mid
            first_failed = mid

    if probe_context(high):
        return high, None
    return low, first_failed


def write_csv(
    metrics: list[RunMetrics],
    max_stable_context: int,
    first_failed_context: int | None,
) -> None:
    benchmark_cmd_template = (
        f"{COMPLETION_BIN.relative_to(LLAMA_DIR)} -m {MODEL_PATH.relative_to(LLAMA_DIR)} "
        "-f <prompt_file> -n 256 -c <ctx_size> -ngl 999 -t 10 -tb 10 -b 2048 -ub 512 "
        "-fa on -fit off --temp 0 --seed 123 --ignore-eos --no-warmup --perf "
        "--simple-io --no-display-prompt -no-cnv"
    )
    probe_cmd_template = (
        f"{COMPLETION_BIN.relative_to(LLAMA_DIR)} -m {MODEL_PATH.relative_to(LLAMA_DIR)} "
        "-f <probe_prompt_file> -n 1 -c <ctx_size> -ngl 999 -t 10 -tb 10 -b 2048 -ub 512 "
        "-fa on -fit off --temp 0 --seed 123 --ignore-eos --no-warmup --perf "
        "--simple-io --no-display-prompt -no-cnv"
    )

    fieldnames = [
        "context_length_tokens",
        "prompt_tokens_requested",
        "prompt_tokens_reported",
        "generated_tokens_reported",
        "decode_tokens_per_second",
        "ttft_ms",
        "sampling_ms",
        "prompt_eval_ms",
        "eval_ms",
        "kv_buffer_mib",
        "gpu_name",
        "metal_unified_memory",
        "offloaded_layers",
        "max_stable_context_tokens",
        "first_failed_context_tokens",
        "fixed_prompt_description",
        "prompt_file",
        "benchmark_log",
        "build_configure_cmd",
        "build_compile_cmd",
        "download_cmd",
        "benchmark_cmd_template",
        "benchmark_cmd_exact",
        "max_ctx_probe_cmd_template",
        "model_repo",
        "model_file",
        "model_sha256",
    ]

    with OUTPUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in metrics:
            writer.writerow(
                {
                    "context_length_tokens": item.ctx_size,
                    "prompt_tokens_requested": item.prompt_tokens_requested,
                    "prompt_tokens_reported": item.prompt_tokens_reported,
                    "generated_tokens_reported": item.generated_tokens_reported,
                    "decode_tokens_per_second": f"{item.decode_tokens_per_second:.4f}",
                    "ttft_ms": f"{item.ttft_ms:.2f}",
                    "sampling_ms": f"{item.sampling_ms:.2f}",
                    "prompt_eval_ms": f"{item.prompt_eval_ms:.2f}",
                    "eval_ms": f"{item.eval_ms:.2f}",
                    "kv_buffer_mib": f"{item.kv_buffer_mib:.2f}",
                    "gpu_name": item.gpu_name,
                    "metal_unified_memory": item.unified_memory,
                    "offloaded_layers": item.offloaded_layers,
                    "max_stable_context_tokens": max_stable_context,
                    "first_failed_context_tokens": "" if first_failed_context is None else first_failed_context,
                    "fixed_prompt_description": (
                        'Exact-token prompt with BOS plus repeated " token" filler tokens; '
                        "prompt_tokens = context_length - 256"
                    ),
                    "prompt_file": str(prompt_path_for_ctx(item.ctx_size, "benchmark").relative_to(ROOT)),
                    "benchmark_log": str(item.log_path.relative_to(ROOT)),
                    "build_configure_cmd": CONFIGURE_CMD,
                    "build_compile_cmd": BUILD_CMD,
                    "download_cmd": DOWNLOAD_CMD,
                    "benchmark_cmd_template": benchmark_cmd_template,
                    "benchmark_cmd_exact": item.full_command,
                    "max_ctx_probe_cmd_template": probe_cmd_template,
                    "model_repo": MODEL_REPO,
                    "model_file": MODEL_FILE,
                    "model_sha256": MODEL_SHA256,
                }
            )


def main() -> None:
    ensure_paths()
    must_exist(LLAMA_DIR)
    must_exist(COMPLETION_BIN)
    must_exist(TOKENIZE_BIN)
    must_exist(MODEL_PATH)

    print("Running warmup...")
    run_warmup()

    measurements: list[RunMetrics] = []
    for ctx_size in CONTEXT_SIZES:
        print(f"Benchmarking ctx={ctx_size}...")
        measurements.append(run_benchmark(ctx_size))

    print("Probing maximum stable context...")
    max_stable_context, first_failed_context = find_max_stable_context()

    write_csv(measurements, max_stable_context, first_failed_context)
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Maximum stable context: {max_stable_context}")
    if first_failed_context is not None:
        print(f"First failed context: {first_failed_context}")


if __name__ == "__main__":
    main()
