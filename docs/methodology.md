# Benchmark Methodology

This repository uses controlled experiments to characterize the
performance behavior of LLM inference systems.

## Experimental Control

Whenever practical, experiments vary one primary serving or workload
variable while holding unrelated configuration constant.

Examples include:

- concurrency
- prompt length
- generation length
- caching configuration
- serving engine
- model precision
- decoding strategy

## Environment Recording

Each study records relevant execution context, including:

- GPU
- model
- serving engine and version
- model precision
- input length
- output length
- concurrency
- decoding configuration
- cache configuration
- relevant engine parameters

## Workload Control

Controlled comparisons use reproducible workload definitions.

Deterministic decoding (`temperature = 0`) is used for the controlled
benchmark comparisons so that workload and serving-system behavior,
rather than sampling variation, dominate differences between runs.

## Metrics

Depending on the study, measurements may include:

- Time to First Token (TTFT)
- Time per Output Token (TPOT)
- Inter-Token Latency (ITL)
- end-to-end latency
- request throughput
- input-token throughput
- output-token throughput
- GPU utilization
- GPU memory utilization

## Raw Results

Raw benchmark outputs are preserved alongside summarized results.

Derived tables, figures, and findings should remain traceable to the
underlying measurements.

## Interpretation

Analysis distinguishes between:

1. directly measured observations;
2. architectural interpretations supported by those observations;
3. hypotheses requiring additional experiments or profiling.

### Load Model

These benchmarks use closed-loop saturation workloads. All requests are made available immediately with `--request-rate inf`, while `--max-concurrency` limits the number of in-flight requests. Reported TTFT and end-to-end latency measure service-level latency after a request is admitted and exclude load-generator queue time. These measurements should not be interpreted as open-loop production arrival latency.