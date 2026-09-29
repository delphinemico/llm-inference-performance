# LLM Inference Performance

Controlled characterization and optimization of LLM inference systems.

This repository explores how workload shape, serving configuration, caching, decoding behavior, model precision, and GPU execution affect inference latency, throughput, memory usage, and scalability.

The current implementation begins with a single-GPU vLLM benchmark on NVIDIA L40S. Additional serving engines and inference optimizations are planned as the repository evolves.

## Current Benchmark

### Benchmark 01: Single-GPU vLLM Serving Characterization

The first benchmark characterizes vLLM inference behavior on a single NVIDIA L40S using Qwen2.5-7B-Instruct.

The experiments cover:

- concurrency scaling
- prefill-heavy workloads
- decode-heavy workloads
- automatic prefix caching
- latency and throughput tradeoffs

### Selected Results

In a closed-loop saturation benchmark, increasing concurrency from 1 to 16 increased output-token throughput from 49.0 to 567.6 tokens/s, while median TTFT increased from 53.4 ms to 365.4 ms.

<img src="experiments/01_vllm_single_gpu/figures/concurrency_throughput.png" alt="Output throughput vs concurrency" width="600">

For a workload with a prewarmed 1536-token shared prefix, the cache-enabled condition reduced median TTFT from 464.0 ms to 164.6 ms.

<img src="experiments/01_vllm_single_gpu/figures/prefix_cache_ttft.png" alt="Prefix caching median TTFT" width="600">

These results illustrate why inference latency should not be treated as a single metric. Concurrency, prompt length, generation length, and prefix reuse affect different parts of the serving path and can produce very different latency and throughput profiles.

## Benchmark Details

- [Benchmark 01 README](experiments/01_vllm_single_gpu/README.md)
- [Detailed findings](experiments/01_vllm_single_gpu/analysis/findings.md)
- [Raw benchmark results](experiments/01_vllm_single_gpu/results/L40S_2026-09-26/)
- [Summary CSV](experiments/01_vllm_single_gpu/results/L40S_2026-09-26/summary.csv)
- [Figures](experiments/01_vllm_single_gpu/figures/)
- [Benchmark scripts](experiments/01_vllm_single_gpu/scripts/)

## Repository Structure

```text
llm-inference-performance/
├── README.md
├── roadmap.md
├── requirements.txt
├── LICENSE
│
├── docs/
│   └── methodology.md
│
└── experiments/
    └── 01_vllm_single_gpu/
        ├── README.md
        ├── analysis/
        │   └── findings.md
        ├── figures/
        ├── results/
        │   └── L40S_2026-09-26/
        └── scripts/
```

## Methodology

The benchmarks use controlled synthetic workloads with fixed input and output lengths so that individual serving variables can be isolated.

Benchmark 01 uses closed-loop saturation workloads. All requests are made available immediately with `--request-rate inf`, while `--max-concurrency` limits the number of in-flight requests. Reported TTFT and end-to-end latency therefore represent service-level latency after request admission and exclude load-generator queue time.

See [docs/methodology.md](docs/methodology.md) for the full benchmark methodology.

## Repository Scope

The repository is intended to expand beyond the initial vLLM characterization to cover additional inference-performance topics, including:

- serving-engine comparison
- quantization
- speculative decoding
- realistic enterprise workload shapes
- GPU profiling and bottleneck analysis

See [roadmap.md](roadmap.md) for the current plan.

## Reproducibility

Benchmark 01 includes:

- raw vLLM benchmark outputs
- summarized results
- benchmark execution scripts
- figure-generation scripts
- methodology documentation
- engineering interpretation and limitations

Detailed reproduction instructions are provided in the [Benchmark 01 README](experiments/01_vllm_single_gpu/README.md).

## License

This project is licensed under the [MIT License](LICENSE).
