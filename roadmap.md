# Roadmap

## 01 - Single-GPU vLLM Serving Characterization

**Status:** Complete

Characterize concurrency, prefill, decode, prefix caching, latency,
throughput, and GPU behavior using vLLM on a single GPU.

---

## 02 - Serving Engine Comparison

**Status:** Planned

Compare controlled workloads across:

- vLLM
- SGLang
- TensorRT-LLM

---

## 03 - Quantization

**Status:** Planned

Evaluate performance and memory tradeoffs across supported lower-precision
inference configurations.

---

## 04 - Speculative Decoding

**Status:** Planned

Characterize when speculative decoding improves generation performance
and when its overhead provides limited benefit.

---

## 05 - Enterprise Workload Characterization

**Status:** Planned

Evaluate realistic inference patterns including:

- RAG-style workloads
- agentic/tool-using workloads
- repeated-prefix workloads
- long-context workloads
- mixed request lengths

---

## 06 - Nsight Profiling and Bottleneck Analysis

**Status:** Planned

Connect application-level benchmark behavior with GPU-level execution,
memory behavior, kernel activity, and system bottlenecks.