# LLM Inference Performance

Controlled performance characterization and optimization of modern LLM inference systems.

This repository investigates how serving architecture, workload shape, caching, decoding strategies, model precision, and GPU behavior affect LLM inference latency, throughput, memory utilization, and scalability.

## Study 01: Single-GPU vLLM Serving Characterization

The first study characterizes vLLM inference behavior on a single NVIDIA L40S using Qwen2.5-7B-Instruct.

The experiments examine:

- concurrency scaling
- prefill-heavy workloads
- decode-heavy workloads
- automatic prefix caching
- latency and throughput tradeoffs

### Selected Results

Increasing concurrency from 1 to 16 increased output-token throughput from 49.0 to 567.6 tokens/s, while median TTFT increased from 53.4 ms to 365.4 ms.

<img src="experiments/01_vllm_single_gpu/figures/concurrency_throughput.png" alt="Output throughput vs concurrency" width="600">

For a workload with a 1536-token shared prefix, enabling automatic prefix caching reduced median TTFT from 464.0 ms to 164.6 ms.

<img src="experiments/01_vllm_single_gpu/figures/prefix_cache_ttft.png" alt="Prefix caching median TTFT" width="600">

### Study Details

- [Study README](experiments/01_vllm_single_gpu/README.md)
- [Detailed findings](experiments/01_vllm_single_gpu/analysis/findings.md)
- [Raw benchmark results](experiments/01_vllm_single_gpu/results/L40S_2026-09-26/)
- [Benchmark scripts](experiments/01_vllm_single_gpu/scripts/)

## Repository Structure

```text
docs/
    Benchmark methodology

experiments/
    Controlled inference performance studies
```

### Methodology

See [methodology.md](docs/methodology.md).

### Roadmap

See [roadmap.md](roadmap.md).