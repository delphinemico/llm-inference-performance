# Experiment 02 — Serving Engine Comparison

## Question

How do vLLM, SGLang, and TensorRT-LLM differ in latency and throughput
under identical single-GPU workloads?

## Controlled Variables

- GPU: NVIDIA L40S
- Model: Qwen/Qwen2.5-7B-Instruct
- Single GPU
- Temperature: 0
- Benchmark client: NVIDIA AIPerf
- Fixed synthetic input/output lengths
- Streaming responses
- Same request counts and concurrency per workload

## Workload Matrix

| Workload | Input | Output | Concurrency |
|---|---:|---:|---:|
| Interactive baseline | 512 | 128 | 4 |
| High concurrency | 512 | 128 | 16 |
| Prefill-heavy | 4096 | 64 | 4 |
| Decode-heavy | 512 | 512 | 4 |

## Metrics

- Request throughput
- Output-token throughput
- Median and P99 TTFT
- ITL
- Request/E2E latency
- GPU utilization
- GPU memory
- Power when available

## Engines

- vLLM
- SGLang
- TensorRT-LLM

## Status

In progress.