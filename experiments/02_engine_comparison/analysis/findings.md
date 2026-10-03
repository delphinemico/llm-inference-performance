# Experiment 02: Findings

## Question

How do vLLM, SGLang, and TensorRT-LLM differ in serving architecture and workload fit on a single NVIDIA L40S?

The goal is not to identify a universal winner. The controlled baseline provides a common reference, while the targeted SGLang experiment demonstrates an engine-specific optimization that matters for reusable-prefix workloads.

## Controlled Baseline

The same four workload shapes were measured for vLLM, SGLang, and TensorRT-LLM using Qwen2.5-7B-Instruct on a single NVIDIA L40S. Cross-request prefix reuse was disabled where applicable so reused prefill computation would not dominate the baseline.

| Engine | Workload | Request throughput (req/s) | Output throughput (tok/s) | Median TTFT (ms) | Median ITL (ms) | Median request latency (ms) |
|---|---|---:|---:|---:|---:|---:|
| vLLM | Interactive c4 | 1.41 | 179.92 | 133.04 | 21.28 | 2837.15 |
| SGLang | Interactive c4 | 1.40 | 178.99 | 168.33 | 21.12 | 2855.74 |
| TensorRT-LLM | Interactive c4 | 1.40 | 179.44 | 159.73 | 21.16 | 2847.86 |
| vLLM | High concurrency c16 | 4.45 | 568.97 | 297.03 | 23.27 | 3267.55 |
| SGLang | High concurrency c16 | 4.28 | 547.44 | 530.16 | 22.71 | 3414.47 |
| TensorRT-LLM | High concurrency c16 | 4.30 | 550.46 | 608.98 | 21.97 | 3395.12 |
| vLLM | Prefill-heavy | 1.55 | 99.22 | 776.13 | 28.50 | 2572.97 |
| SGLang | Prefill-heavy | 1.48 | 94.94 | 816.44 | 30.02 | 2692.04 |
| TensorRT-LLM | Prefill-heavy | 1.46 | 93.30 | 843.92 | 30.12 | 2737.46 |
| vLLM | Decode-heavy | 0.36 | 186.55 | 145.60 | 21.19 | 10963.68 |
| SGLang | Decode-heavy | 0.36 | 186.07 | 162.32 | 21.19 | 10989.34 |
| TensorRT-LLM | Decode-heavy | 0.36 | 185.35 | 147.03 | 21.30 | 11022.08 |

## Baseline Interpretation

The three engines were close on the interactive and decode-heavy workloads when cross-request prefix reuse was removed from the comparison.

vLLM was numerically highest in the observed concurrency-16 run, reaching 568.97 output tokens/s compared with 547.44 for SGLang and 550.46 for TensorRT-LLM. Its median TTFT was also numerically lower in that run. Because each baseline condition was measured with one benchmark run, these relatively small differences should not be interpreted as a stable engine ranking without repeated trials.

vLLM was also numerically highest in the observed prefill-heavy run, with 776.13 ms median TTFT and 99.22 output tokens/s. As with the concurrency comparison, the differences should be treated as observed measurements rather than evidence of a stable ranking.

The decode-heavy workload produced almost identical sustained decode behavior across the three engines. Median ITL was approximately 21 ms for all three, and median request latency was approximately 11 seconds.

These results do not establish a universal engine ranking. They show that under this controlled single-GPU baseline, the engines were much more similar than their architectural differences might suggest.

## SGLang RadixAttention Shared-Prefix Demonstration

A separate targeted experiment tested the architectural behavior that is particularly relevant to SGLang: reuse of large shared prompt prefixes through RadixAttention.

Five sequential requests used approximately 4,850 prompt tokens each. Most of the prompt was identical across requests, while each request had a small unique task suffix. Each request generated 32 output tokens.

### RadixAttention Disabled

Server logs reported zero cached tokens.

Steady-state requests 2 through 5 took:

```text
1.066 s
1.070 s
1.071 s
1.069 s
```

Average steady-state client-observed latency was approximately 1.069 s.

### RadixAttention Enabled

For requests 2 through 5, SGLang reported approximately 4,843 cached prompt tokens per request.

The server logs showed only approximately 6 to 7 new prompt tokens requiring prefill after the shared prefix had been cached.

Steady-state requests 2 through 5 took:

```text
0.769 s
0.716 s
0.698 s
0.687 s
```

Average steady-state client-observed latency was approximately 0.718 s.

This corresponds to an approximately 33% reduction in client-observed steady-state latency for this targeted shared-prefix workload.

## What the RadixAttention Test Demonstrates

The main result is not the exact latency percentage. The important architectural observation is that SGLang reused nearly the entire shared prompt prefix rather than recomputing it for each subsequent request.

That behavior is relevant to workloads such as:

- repeated system prompts;
- shared tool definitions;
- multi-turn conversations;
- planner and tool loops;
- agent branches that inherit substantial prior context.

The test therefore supports SGLang as an interesting serving option when repeated prompt structure is an important workload characteristic.

## Engine Workload Fit

### vLLM

vLLM remains a strong general-purpose serving baseline. Its PagedAttention-based KV-cache management and continuous batching support efficient memory use and strong concurrency behavior. In this experiment, vLLM produced the strongest high-concurrency and prefill-heavy baseline results.

### SGLang

SGLang's main differentiator in this study was not the uncached baseline. It was RadixAttention under deliberate shared-prefix reuse. When the workload reused approximately 4,843 prompt tokens across requests, SGLang avoided almost all repeated prefill computation and reduced observed request latency.

This makes SGLang particularly relevant when application structure naturally creates reusable context.

### TensorRT-LLM

TensorRT-LLM represents the NVIDIA-optimized serving path and exposes deeper NVIDIA-specific optimization opportunities.

In the configuration tested here, TensorRT-LLM did not outperform vLLM in the controlled baseline. That result should not be interpreted as a general limitation of TensorRT-LLM. TensorRT-LLM has a broader optimization surface, and this experiment intentionally avoided extensive engine-specific tuning.

Its value is therefore better understood as access to a deeper NVIDIA-specific optimization stack rather than an expectation that a minimally tuned configuration must win every baseline workload.

## Limitations

This experiment used one model, one GPU class, one benchmark client, and a small number of workload shapes.

The NVIDIA L40S was held constant as the hardware target, but the complete software environments were not identical across every engine run. Engine versions, PyTorch/CUDA runtimes, server implementations, and engine-native behavior can affect results.

The SGLang shared-prefix test was a small architectural demonstration rather than a full statistical benchmark. Its client-observed latency values should therefore be treated as supporting evidence. The stronger result is the direct server evidence that approximately 4,843 tokens were reused while only a handful of new prompt tokens required prefill.

No extensive TensorRT-LLM tuning, quantization, speculative decoding, multi-GPU scaling, or distributed serving was evaluated.

## Conclusion

The controlled baseline showed that vLLM, SGLang, and TensorRT-LLM can exhibit broadly similar behavior when engine-specific prefix reuse is removed from the comparison.

The more useful distinction appears when workload structure is considered.

vLLM provided strong general-purpose concurrency and prefill performance in the tested baseline. SGLang demonstrated effective shared-prefix reuse when requests reused a large common prefix through RadixAttention. TensorRT-LLM provides the NVIDIA-specific optimization path for workloads where deeper hardware-aware tuning is justified.

Serving-engine selection should therefore be based on workload characteristics and operational requirements rather than a single headline throughput number.

Experiment 02 is complete.
