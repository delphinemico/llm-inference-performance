# 01 - Single-GPU vLLM Serving Characterization

## Objective

Characterize single-GPU LLM inference behavior under controlled changes
in concurrency, prompt length, generation length, and prefix caching.

## Environment

- GPU: NVIDIA L40S 48 GB
- Model: Qwen/Qwen2.5-7B-Instruct
- Serving engine: vLLM
- vLLM version: 0.30.0
- Maximum model length: 8192
- GPU memory utilization target: 0.90
- Deterministic decoding: `temperature = 0`

## Experiments

### E1 - Concurrency Scaling

- Input length: 512 tokens
- Output length: 128 tokens
- Concurrency: 1, 4, 8, 16
- Requests per condition: 100

Purpose: characterize latency and throughput behavior as concurrent load
increases.

### E2 - Prefill Scaling

Compare:

- 512 input / 64 output
- 4096 input / 64 output

Concurrency: 4

Purpose: isolate the impact of increased prompt processing on inference
performance.

### E3 - Decode Scaling

Compare:

- 512 input / 64 output
- 512 input / 512 output

Concurrency: 4

Purpose: characterize the effect of substantially longer autoregressive
generation.

### E4 - Prefix Caching

Compare automatic prefix caching:

- OFF
- ON

Workload:

- 1536-token shared prefix
- 512 variable input tokens
- 64 output tokens
- concurrency 4

Purpose: measure the impact of reusable prompt prefixes on prefill-related
latency and throughput.

## Metrics

The study tracks:

- request throughput
- output-token throughput
- Time to First Token (TTFT)
- Time per Output Token (TPOT)
- Inter-Token Latency (ITL)
- end-to-end latency
- GPU utilization
- GPU memory utilization

## Results

Raw and summarized benchmark results are available under:

`results/L40S_2026-09-26/`

Consolidated benchmark results:

`results/L40S_2026-09-26/summary.csv`

Detailed analysis:

`analysis/findings.md`

## Reproduction

Benchmark and telemetry scripts are available under:

`scripts/`