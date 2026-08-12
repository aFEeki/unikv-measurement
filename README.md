# What Vanishing Transfer Cost Does Not Buy

Measuring exact KV-cache retention on unified memory.

Alp Demir Ekinci — Sabancı University

This repository is the artifact for a measurement paper about what happens to
KV-cache policy when the interconnect disappears. It contains the measurement
harnesses, the raw results, and the per-experiment analyses behind every
number in the paper.

## The short version

Offloading systems for KV cache are built around a transfer cost. Unified memory
appears to remove it: on Apple Silicon the CPU and GPU address one physical DRAM
pool, so demoting a cache entry moves no bytes. The apparent implication is that
such a cache should never discard anything — spill it and recall it exactly.

We implemented that policy in a `llama.cpp` fork and measured it on an M4 Pro.
The implication does not survive. Five findings:

1. **The cost of exact retention is affine and dominated by a fixed term.**
   Entering the two-tier attention path costs ~9 ms per decode step regardless of
   how much is retained, against ~3.1 µs per retained cell. The engineering
   target is the handoff, not the data.
2. **Retrieval does not separate lossless retention from eviction.** An H2O
   heavy-hitter evictor recovers a mid-context passkey at the same budget and
   runs faster. What retention uniquely provides is token-identical reproduction
   of the no-eviction computation — identical over 512 generated tokens on two
   prompts, where H2O diverges within the first 16.
3. **Simulated transfer cost is inseparable from the instrument that exposes
   it.** Asynchronous dispatch absorbs an undrained delay; draining exposes it.
   The drain has no cost of its own — it costs what it stops the pipeline
   absorbing, so it cannot be calibrated away.
4. **Capacity limits appear at execution, not allocation.** Contexts allocate and
   generate at 1.80× the advisory working-set budget and are refused at 1.85×,
   with an attention scratch term set by the micro-batch.
5. **Context shift costs more than eviction.** The ostensibly cheap baseline
   re-encodes positions across the whole cache and is the more expensive
   mechanism.

The paper's one positive recommendation is narrow: exact retention buys a
reproducible decode, which matters for evaluation harnesses and differential
testing, and costs 11.4% throughput against H2O where three-fifths of the
context is non-resident.

## Layout

| | |
|---|---|
| `scripts/` | Measurement harnesses and figure generators |
| `stress_results/`, `alpha_results/`, `quality_results/` | Raw result CSVs |
| `artifacts/` | Per-experiment analyses, run logs, prompts, token traces |
| `figures/` | Figures as published, regenerable from `scripts/make_fig*.py` |
| `llama.cpp` | Submodule: the fork the measurements run against |

`ARTIFACT.md` maps every quantitative claim in the paper to the harness, the data
file and the analysis that produced it.

## Building the fork

```
git clone --recurse-submodules <this repo>
cd llama.cpp && cmake -B build -DGGML_METAL=ON && cmake --build build -j
```

Policies are selected at runtime through `UNIKV_POLICY`: 0 upstream (errors when
the cache fills), 1 rolling window, 3 exact spill-and-recall, 4 H2O. Related
variables — `UNIKV_ALPHA`, `UNIKV_DRAIN`, `UNIKV_SPILL_DEV`, `UNIKV_SPILL_CAP`,
`UNIKV_LOG` — are documented in the harnesses that use them.

## Reproducing

Claims differ in what they need:

- **Deterministic** (token sequences, spill counts, passkey outcomes) — greedy
  decoding at a fixed seed, reproduces anywhere the fork runs.
- **Binary** (capacity outcomes) — a configuration either executes or is refused.
- **Rates** (tok/s, ms/step) — require the cooled protocol: randomised block
  order, 200 s cooldowns, an otherwise idle machine, and runs left
  uninstrumented, because the per-step log itself drains the GPU pipeline.

Two experiments in `artifacts/` document how far off you land without the
protocol: a counterbalanced A/B was wrong in **both** directions, overstating one
effect by 20 points and understating another.

Flash attention state is parsed back from each run's own stderr rather than
trusted from the harness source; several harnesses fail loudly on a mismatch.

The model is not redistributed. Every run uses Llama 3.1 8B Instruct at Q4_K_M.

## Citation

Cite the paper and, for exactness, the pinned version DOI of this artifact rather
than the concept DOI.

## License

Harnesses: MIT (`LICENSE`). Result files and analyses:
CC-BY-4.0 (`LICENSE-DATA`). The `llama.cpp` fork inherits upstream's MIT license
and retains its attribution.
