# What Vanishing Transfer Cost Does Not Buy

Measuring exact KV-cache retention on unified memory.

Alp Demir Ekinci, Sabancı University

This repository is the artifact for a measurement paper about what happens to
KV-cache policy when the interconnect disappears. It contains the measurement
harnesses and the raw results behind every number in the paper.

## The short version

Offloading systems for KV cache are built around a transfer cost. Unified memory
appears to remove it: on Apple Silicon the CPU and GPU address one physical DRAM
pool, so demoting a cache entry crosses no interconnect. The apparent implication
is that such a cache should never discard anything: spill it and recall it
exactly.

We implemented that policy in a `llama.cpp` fork and measured it on an M4 Pro
across four models. The implication fails. Five findings:

1. **The cost of exact retention is affine, with a fixed and a per-cell term.**
   Entering the two-tier attention path costs 9.7 ms per decode step with the
   retained tier read by the host and 2.7 ms with it read by the accelerator,
   against 3.1 and 2.2 µs per retained cell. Neither term is a byte count; both
   scale with layer count to within ±10% across the four models.
2. **Retrieval does not separate lossless retention from eviction.** An H2O
   heavy-hitter evictor recovers a passkey far outside its resident window at
   the same capacity and runs faster. Only token identity with the no-eviction
   reference separates them, and it held on 15 of 16 arm-and-prompt pairs over
   four models and failed once.
3. **Simulated transfer cost is inseparable from the instrument that exposes
   it.** The pipeline drain has no cost of its own; it costs whatever delay it
   stops the pipeline absorbing, so it cannot be calibrated away.
4. **Capacity limits appear at execution, not allocation.** Contexts allocate at
   1.80x the advisory working-set budget and are refused once the memory is
   touched past it. A scratch term set by the micro-batch predicts the ceiling on
   three unseen architectures. Counted across both tiers, exact retention's
   footprint at 65k tokens is within 3% of upstream's.
5. **What a rolling window costs is set by how much it discards per event.**
   Evicting one cell per overflow, it is slower than H2O; discarding half the
   cache per event, as upstream ships, it is faster.

What exact retention buys is narrow: reproduction of the no-eviction computation
token for token, as an observed outcome rather than a guarantee.

## Layout

| | |
|---|---|
| `scripts/` | Measurement harnesses and figure generators |
| `stress_results/`, `alpha_results/`, `quality_results/` | Raw result CSVs |
| `quality_results/token_sequences/` | Full sampled-token-ID sequences |
| `figures/` | Figures as published, regenerable from `scripts/make_fig*.py` |
| `llama.cpp` | Submodule: the fork the measurements run against |

`ARTIFACT.md` maps every quantitative claim in the paper to the harness and the
data file that produced it.

## Building the fork

```
git clone --recurse-submodules <this repo>
cd llama.cpp && cmake -B build -DGGML_METAL=ON && cmake --build build -j
```

Policies are selected at runtime through `UNIKV_POLICY`: 0 upstream (errors when
the cache fills), 1 rolling window, 3 exact spill-and-recall, 4 H2O. Related
variables (`UNIKV_ALPHA`, `UNIKV_DRAIN`, `UNIKV_SPILL_DEV`, `UNIKV_SPILL_CAP`,
`UNIKV_WINDOW_DISCARD`, `UNIKV_LOG`) are documented in the harnesses that use
them.

## Reproducing

Claims differ in what they need:

- **Deterministic** (token sequences, spill counts, passkey outcomes, graph
  split counts): greedy decoding at a fixed seed, reproduces anywhere the fork
  runs.
- **Binary** (capacity outcomes): a configuration either executes or is refused.
- **Rates** (tok/s, ms/step): require the cooled protocol, meaning randomised
  block order, 200 s cooldowns, an otherwise idle machine, and runs left
  uninstrumented, because the per-step log itself drains the GPU pipeline.

Without the protocol the error is not small: a counterbalanced A/B
(`stress_results/b1_interleaved_ab.csv`) was wrong in **both** directions,
overstating one effect by 20 points and understating another.

Flash attention state is parsed back from each run's own stderr rather than
trusted from the harness source; several harnesses fail loudly on a mismatch.

Model weights are not redistributed. Runs use Llama 3.1 8B, Qwen2.5 7B,
Llama 3.2 3B and Llama 3.2 1B, all Instruct at Q4_K_M.

## Citation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21907114.svg)](https://doi.org/10.5281/zenodo.21907114)

- This artifact: [10.5281/zenodo.21907114](https://doi.org/10.5281/zenodo.21907114)
- The `llama.cpp` fork: [10.5281/zenodo.21907123](https://doi.org/10.5281/zenodo.21907123)

Both are concept DOIs and resolve to the latest release. Each Zenodo record also
carries a version-specific DOI if you need to pin an exact snapshot.

## License

Harnesses: MIT (`LICENSE`). Result files: CC-BY-4.0 (`LICENSE-DATA`). The
`llama.cpp` fork inherits upstream's MIT license and retains its attribution.
