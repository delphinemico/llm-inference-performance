# Experiment 02: Serving Engine Comparison

## Question

How do vLLM, SGLang, and TensorRT-LLM differ in serving architecture and workload fit on a single NVIDIA L40S?

The goal is not to identify a universal winner. The goal is to understand what each engine is designed to optimize and demonstrate those differences with a small number of targeted measurements.

## Engines

### vLLM

vLLM is used as the general-purpose serving baseline. Its PagedAttention-based KV-cache management and continuous batching are designed to support efficient memory use and strong throughput under concurrent inference.

### SGLang

SGLang is particularly interesting for workloads with reusable prompt structure. RadixAttention can reuse KV-cache state across requests that share prefixes, which is relevant to multi-turn and agentic workloads with repeated system prompts, tool definitions, or conversation context.

### TensorRT-LLM

TensorRT-LLM represents the NVIDIA-optimized serving path. It provides a deeper hardware-specific optimization stack for extracting inference performance from NVIDIA GPUs.

## Environment

- GPU: NVIDIA L40S
- Model: Qwen/Qwen2.5-7B-Instruct
- Single GPU
- Temperature: 0
- Benchmark client: NVIDIA AIPerf
- Streaming responses

## Controlled Baseline

A common four-workload baseline was collected for all three engines using the same model, GPU class, benchmark client, token lengths, request counts, and concurrency.

| Workload | Input | Output | Concurrency |
|---|---:|---:|---:|
| Interactive baseline | 512 | 128 | 4 |
| High concurrency | 512 | 128 | 16 |
| Prefill-heavy | 4096 | 64 | 4 |
| Decode-heavy | 512 | 512 | 4 |

Cross-request prefix reuse was disabled for the controlled baseline so that differences in reused prefill computation would not dominate the comparison.

The baseline is supporting evidence, not the entire purpose of the experiment.

## Engine-Specific Analysis

The comparison focuses on the architectural strengths that matter for actual workload selection:

- **vLLM:** throughput and concurrency behavior for general-purpose inference.
- **SGLang:** the effect of RadixAttention when requests share substantial prefixes, especially in agentic or multi-turn workloads.
- **TensorRT-LLM:** latency, throughput, and hardware efficiency from the NVIDIA-optimized serving stack.

Engine-native optimizations are evaluated when they answer a specific workload question rather than being disabled simply to make every engine behave identically.

## Metrics

Primary metrics include:

- TTFT
- inter-token latency
- request latency
- request throughput
- output-token throughput
- GPU utilization
- GPU memory
- GPU power when available

## Principle

Only experiments that answer a concrete serving question are included. Additional workloads or tuning are added only when they materially change the interpretation of engine behavior.

## Results

- Controlled baseline summary: `results/summary.csv`
- SGLang RadixAttention shared-prefix demonstration: `results/sglang_radix_shared_prefix/results.txt`
- Full interpretation: `analysis/findings.md`

## Status

Complete.

The controlled three-engine baseline is finished, and the targeted SGLang shared-prefix experiment demonstrated substantial prefix reuse with RadixAttention. No additional Experiment 02 benchmarking is planned.
