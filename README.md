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
across four models, against the unmodified runtime holding the same context
resident. The implication fails. Three findings:

1. **What retention costs.** A fixed charge on every decode step, 0.58-1.51 ms
   with the retained tier read by the accelerator and 3.5-8.5 ms with it read
   by the host, growing with layer count; on the host tier the scheduler adds
   exactly two graph splits per layer. Per retained cell, 92-95% of the
   device-tier cost is ordinary unfused attention the runtime pays anyway;
   retention adds a 5.5-8.4% surcharge.
2. **What it buys.** Not retrieval: an H2O heavy-hitter evictor recovers a
   passkey far outside its resident window at the same capacity and runs
   faster. Only token identity with the no-eviction reference separates them,
   and it held on 15 of 16 arm-and-prompt pairs. Where it failed, a near-tie
   in the reference's logits met the host tier's larger perturbation.
3. **What limits capacity.** Contexts allocate at 1.80x the advisory
   working-set budget and are refused when the memory is touched. A larger
   resident cache reaches as far as the device tier and runs faster; the host
   tier reaches further at 3.9-10.7x the decode cost, on the same total
   physical memory to within 3%.

Two results concern measurement rather than retention: delay injection needs
a pipeline drain, and the drain has no fixed cost that could be subtracted;
and a rolling window's cost is set by how much it discards per shift event.

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

- **Deterministic** (token sequences, spill counts, passkey outcomes, logit
  dumps, graph split counts): greedy decoding at a fixed seed, reproduces anywhere the fork
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
