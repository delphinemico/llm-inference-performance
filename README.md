# LLM Inference Performance

Controlled characterization and optimization of LLM inference systems.

This repository explores how workload shape, serving configuration, caching, decoding behavior, serving-engine design, and GPU execution affect inference latency, throughput, memory usage, and scalability.

The repository currently contains three completed experiments on an NVIDIA L40S:

1. single-GPU vLLM serving characterization
2. controlled serving-engine comparison across vLLM, SGLang, and TensorRT-LLM
3. GPU runtime profiling with NVIDIA Nsight Systems

## Experiment Summary

| Experiment | Focus | Main takeaway |
| --- | --- | --- |
| **01. Single-GPU vLLM Characterization** | Concurrency, prefill, decode, and prefix caching | Shows how workload shape and concurrency change TTFT, throughput, and end-to-end latency on a single L40S. |
| **02. Serving Engine Comparison** | vLLM, SGLang, and TensorRT-LLM | Shows that baseline performance was broadly similar in the tested configuration, while SGLang demonstrated effective shared-prefix reuse with RadixAttention. |
| **03. GPU Runtime Profiling** | Nsight Systems analysis of prefill and decode | Connects application-level inference behavior to GPU execution, including the repeated per-token kernel pattern of autoregressive decoding. |


## Completed Experiments

### Experiment 01: Single-GPU vLLM Serving Characterization

Characterizes vLLM inference behavior under controlled changes in concurrency, prompt length, generation length, and automatic prefix caching using Qwen2.5-7B-Instruct on a single NVIDIA L40S.

Key observations:

- increasing concurrency from 1 to 16 increased output-token throughput from 49.0 to 567.6 tokens/s while median TTFT increased from 53.4 ms to 365.4 ms
- increasing prompt length from 512 to 4096 tokens increased median TTFT from 135.2 ms to 791.4 ms
- increasing generation length from 64 to 512 tokens increased median end-to-end latency from approximately 1.48 s to 10.95 s while TTFT and TPOT remained comparatively stable
- prewarmed automatic prefix caching reduced median TTFT from 464.0 ms to 164.6 ms for the tested shared-prefix workload


* [Experiment 01 README](experiments/01_vllm_single_gpu/README.md)
* [Experiment 01 findings](experiments/01_vllm_single_gpu/analysis/findings.md)

<img src="experiments/01_vllm_single_gpu/figures/concurrency_throughput.png" alt="Output throughput vs concurrency" width="600">

<img src="experiments/01_vllm_single_gpu/figures/prefix_cache_ttft.png" alt="Prefix caching median TTFT" width="600">

### Experiment 02: Serving Engine Comparison

Compares vLLM, SGLang, and TensorRT-LLM using the same model, GPU class, benchmark client, token lengths, request counts, and concurrency across four controlled workload shapes.

The three engines were broadly similar in the controlled baseline. At concurrency 16, vLLM was numerically highest in the observed run at 568.97 output tokens/s, compared with 547.44 for SGLang and 550.46 for TensorRT-LLM. Because each condition represents a single benchmark run and several differences are small, these measurements should not be interpreted as a stable engine ranking.

A targeted SGLang shared-prefix test demonstrated effective RadixAttention reuse. After the shared prefix was cached, approximately 4,843 prompt tokens were reused per request and steady-state client-observed latency decreased from approximately 1.069 s with RadixAttention disabled to approximately 0.718 s with it enabled.

* [Experiment 02 README](experiments/02_engine_comparison/README.md)
* [Experiment 02 findings](experiments/02_engine_comparison/analysis/findings.md)
* [Experiment 02 summary CSV](experiments/02_engine_comparison/results/summary.csv)

### Experiment 03: GPU Runtime Profiling

Uses NVIDIA Nsight Systems to connect high-level prefill and decode behavior with GPU runtime execution in vLLM.

The prefill-heavy trace concentrated most GPU kernel time in a relatively small number of larger matrix operations. In the decode-heavy trace, the dominant GEMV-style kernel accounted for approximately 90.5% of recorded GPU kernel time and executed 512 times, matching the 512 generated tokens. Several supporting kernels also appeared 512 times, directly exposing the repeated per-token execution pattern of autoregressive decoding.

* [Experiment 03 README](experiments/03_gpu_runtime_profiling/README.md)
* [Experiment 03 findings](experiments/03_gpu_runtime_profiling/analysis/findings.md)
* [Prefill Nsight statistics](experiments/03_gpu_runtime_profiling/traces/prefill_heavy_final_stats.txt)
* [Decode Nsight statistics](experiments/03_gpu_runtime_profiling/traces/decode_heavy_final_stats.txt)

## Repository Structure

```text
llm-inference-performance/
├── README.md
├── requirements.txt
├── LICENSE
│
├── docs/
│   └── methodology.md
│
└── experiments/
    ├── 01_vllm_single_gpu/
    │   ├── README.md
    │   ├── analysis/
    │   ├── figures/
    │   ├── results/
    │   └── scripts/
    │
    ├── 02_engine_comparison/
    │   ├── README.md
    │   ├── analysis/
    │   ├── results/
    │   └── scripts/
    │
    └── 03_gpu_runtime_profiling/
        ├── README.md
        ├── analysis/
        ├── scripts/
        └── traces/
```

## Methodology

The repository uses controlled synthetic workloads to isolate specific inference behaviors.

Experiment 01 focuses on application-level latency and throughput under controlled changes in concurrency, prompt length, generation length, and prefix reuse.

Experiment 02 compares serving engines under common workload shapes, then evaluates one targeted engine-specific behavior: SGLang RadixAttention under substantial shared-prefix reuse.

Experiment 03 moves below the application layer and uses Nsight Systems to inspect CUDA execution for prefill-heavy and decode-heavy workloads.

See [docs/methodology.md](docs/methodology.md) for additional benchmark methodology.

## Reproducibility

Each experiment includes its own methodology, scripts, results, and interpretation.

- Experiment 01 publishes raw benchmark outputs, summaries, figures, and reproduction scripts.
- Experiment 02 publishes the controlled baseline results, targeted SGLang shared-prefix results, and engine-specific benchmark scripts.
- Experiment 03 publishes the profiling workload, Nsight text summaries, and findings. Binary `.nsys-rep` traces are intentionally excluded from version control and can be regenerated using the documented profiling command.

Repository-level `requirements.txt` contains analysis dependencies only. Serving engines and profiling tools use separate runtime environments documented within the corresponding experiments.

## Potential Future Work

Possible future extensions include:

- quantization
- speculative decoding
- mixed and enterprise-style workload shapes
- additional hardware configurations
- deeper kernel-level profiling where a specific performance question justifies it

These are potential extensions rather than committed milestones.

## License

This project is licensed under the [MIT License](LICENSE).
