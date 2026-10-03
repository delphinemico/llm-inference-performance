# Experiment 03: GPU Runtime Profiling

This experiment uses NVIDIA Nsight Systems to connect high-level LLM inference behavior with GPU runtime execution in vLLM.

The goal is not to benchmark another serving engine. Experiments 01 and 02 already characterize serving behavior and engine workload fit. Here, the objective is narrower: compare a prefill-heavy workload with a decode-heavy workload and inspect how those two phases appear at the GPU-kernel level.

## Question

How does GPU execution differ between prefill-heavy and decode-heavy inference?

## Environment

- GPU: NVIDIA L40S 48 GB
- Model: `Qwen/Qwen2.5-7B-Instruct`
- vLLM: `0.30.0`
- Nsight Systems: `2025.3.2`
- Precision: BF16
- Single GPU
- Prefix caching disabled for the profiling runs

## Workloads

| Workload | Prompt tokens | Completion tokens | Purpose |
| --- | ---: | ---: | --- |
| Prefill-heavy | 2,562 | 64 | Emphasize prompt processing |
| Decode-heavy | 641 | 512 | Emphasize autoregressive token generation |

Each workload is first executed once as an unprofiled warmup. The identical workload is then executed again inside a CUDA profiler capture range. This keeps model loading, graph capture, and one-time JIT compilation outside the measured region.

vLLM V1 multiprocessing is disabled during profiling so that the CUDA profiler range and GPU execution occur in the same process.

## Method

The profiling command uses:

```bash
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
nsys profile \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --output=<trace_name> \
  python scripts/offline_profile.py --workload <prefill|decode>
```

The Python workload explicitly calls `cudaProfilerStart()` immediately before the measured `llm.generate(...)` call and `cudaProfilerStop()` after GPU synchronization.

## Results

The binary Nsight `.nsys-rep` reports are intentionally excluded from version control because they are generated profiling artifacts. They can be regenerated using the profiling command above.

Committed text summaries:

- [Prefill-heavy Nsight statistics](traces/prefill_heavy_final_stats.txt)
- [Decode-heavy Nsight statistics](traces/decode_heavy_final_stats.txt)

See [analysis/findings.md](analysis/findings.md) for interpretation.

## Scope

This is a controlled GPU runtime profiling study, not an HTTP serving benchmark. Application-level serving behavior is covered in the earlier experiments.
