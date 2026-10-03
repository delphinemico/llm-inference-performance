# Findings

## Summary

The Nsight Systems traces show a clear execution difference between prefill-heavy and decode-heavy inference in vLLM.

The prefill-heavy workload concentrates most GPU time into a relatively small number of large matrix operations over the prompt. The decode-heavy workload instead shows repeated per-token execution across the autoregressive loop.

## Workloads

| Workload | Prompt tokens | Completion tokens |
| --- | ---: | ---: |
| Prefill-heavy | 2,562 | 64 |
| Decode-heavy | 641 | 512 |

Both workloads used the same model, GPU, vLLM version, and profiling method. Prefix caching was disabled. Each workload was warmed once before profiling so that one-time JIT compilation and initialization effects were excluded from the measured capture.

## Prefill-heavy behavior

The prefill trace is dominated by a small number of comparatively large GEMM operations:

- The largest CUTLASS BF16 GEMM family accounted for about 39.9% of recorded GPU kernel time across 84 instances.
- A GEMV-style kernel accounted for about 31.8% across 64 instances.
- Another BF16 GEMM family accounted for about 17.9% across 28 instances.
- FlashAttention was visible at about 3.7% across 28 instances.

This is consistent with prompt processing concentrating work into larger matrix operations over many input tokens.

## Decode-heavy behavior

The decode trace is dominated by repeated execution:

- The dominant GEMV-style kernel accounted for about 90.5% of recorded GPU kernel time across exactly 512 instances.
- The sampling kernel also appeared 512 times.
- Several supporting kernels, including block-table gathering, indexing, and token-selection work, also appeared 512 times.

The 512-instance repetition aligns directly with the 512 generated tokens in the workload.

## Interpretation

The traces connect the high-level prefill/decode distinction to GPU runtime behavior:

- Prefill processes many prompt tokens together and therefore concentrates work into fewer, larger matrix operations.
- Decode generates tokens autoregressively and therefore repeats a smaller set of per-token GPU operations across the generation loop.

For this experiment, the clearest runtime signature of decode is not simply longer wall-clock duration. It is the repeated kernel execution pattern that tracks the number of generated tokens.

## Caveats

This experiment is a controlled GPU runtime profiling study rather than an HTTP serving benchmark.

The prompt and completion lengths are intentionally different because the goal is to emphasize the two execution regimes, not to compare end-to-end latency under identical token counts.

The traces should therefore be interpreted structurally rather than as a throughput or latency ranking between prefill and decode.

## Conclusion

Nsight Systems provides a direct bridge between application-level inference concepts and GPU execution.

In this vLLM study, prefill appears as concentrated matrix-heavy work over the prompt, while decode appears as repeated per-token execution. The decode trace is especially clear: the dominant GEMV-style kernel and several supporting kernels execute 512 times, matching the 512-token generation length.
