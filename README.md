# What Vanishing Transfer Cost Does Not Buy

Measuring exact KV-cache retention on unified memory.

Alp Demir Ekinci, Sabancı University

This repository is the artifact for the paper: the measurement harnesses and
the raw results behind every number in it.

## Summary

Offloading systems for the KV cache are designed around the cost of moving
evicted state back from host memory. On Apple Silicon the CPU and GPU address
one physical DRAM pool, so demoting a cache entry crosses no interconnect, and
the offloading literature's own cost model then says a cache should never
discard anything: spill to the shared pool and recall exactly. We built that
policy into a `llama.cpp` fork and measured it on an M4 Pro, on four models,
against the unmodified runtime holding the same context resident.

Retention is not free. It adds a fixed charge on every decode step,
0.58–1.51 ms when the accelerator reads the retained tier and 3.5–8.5 ms when
the host does, and the charge grows with layer count; on the host tier the
scheduler adds two graph splits per layer. Per retained cell, 92–95% of the
device-tier cost is attention the runtime pays anyway, and retention adds
5.5–8.4% on top.

It buys little. A heavy-hitter evictor recovers a passkey far outside its
resident window at the same capacity and runs faster. A larger resident cache
reaches as far as the device tier and runs faster; the host tier reaches
further, at 3.9–10.7× the decode cost and on the same total physical memory to
within 3%. What retention does buy is token-for-token reproduction of the
no-eviction computation. That held on 15 of 16 arm-and-prompt pairs and failed
where a near-tie in the reference's logits met the host tier's larger
perturbation.

Contexts allocate at 1.80× the advisory working-set budget and are refused only
when the memory is touched. Two further results concern measurement: delay
injection needs a pipeline drain, and the drain has no fixed cost that could be
subtracted afterwards; and a rolling window's cost is set by how much it
discards per shift event.

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
variables (`UNIKV_SPILL_DEV`, `UNIKV_SPILL_CAP`, `UNIKV_ALPHA`, `UNIKV_DRAIN`,
`UNIKV_WINDOW_DISCARD`, `UNIKV_NO_REENCODE`, `UNIKV_LOGIT_LOG`, `UNIKV_LOG`) are
documented in the harnesses that use them.

## Reproducing

Token sequences, spill counts, passkey outcomes, logit dumps and graph split
counts are deterministic under greedy decoding at a fixed seed and reproduce on
any machine that runs the fork. Capacity outcomes are binary: a configuration
either runs or is refused. Rates, in tok/s or ms/step, need the cooled
protocol: randomized block order, 200 s cooldowns, an otherwise idle machine,
and runs left uninstrumented, because the per-step log itself drains the GPU
pipeline. Without it the error is large. A counterbalanced A/B
(`stress_results/b1_interleaved_ab.csv`) was wrong in both directions,
overstating one effect by 20 points and understating another.

Flash attention state is read back from each run's own log rather than trusted
from the harness, and several harnesses stop on a mismatch.

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
