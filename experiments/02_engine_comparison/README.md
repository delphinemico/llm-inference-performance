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
- Model: `Qwen/Qwen2.5-7B-Instruct`
- Single GPU
- Temperature: 0
- Benchmark client: NVIDIA AIPerf 0.13.0
- Streaming responses
- vLLM: 0.30.0
- SGLang: 0.5.9
- TensorRT-LLM: 1.2.1

The engine runs used separate serving environments rather than a single universal Python environment. The recorded SGLang environment used PyTorch 2.9.1+cu128. The recorded TensorRT-LLM environment used PyTorch 2.9.1+cu130 with CUDA toolkit 13.1. The complete resolved Python and PyTorch package set for every original baseline run was not retained, so the versions above document the engine versions and known runtime details without implying a fully frozen environment.

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

## Reproduction

The original comparison was run on a Linux host with one NVIDIA L40S. The benchmark client and serving engine ran on the same host, with AIPerf sending requests to `http://127.0.0.1:8000`.

The repository-level `requirements.txt` contains analysis dependencies only. Each serving engine should be installed in its own compatible environment.

### 1. Clone the repository

```bash
git clone https://github.com/delphinemico/llm-inference-performance.git
cd llm-inference-performance/experiments/02_engine_comparison
```

### 2. Benchmark client

Use NVIDIA AIPerf 0.13.0 in a separate environment.

The baseline scripts use the OpenAI-compatible completions endpoint with streaming enabled, 100 measured requests, 16 warmup requests, deterministic decoding, and seed 42.

### 3. Start one serving engine

Run only one serving engine at a time on port 8000.

#### vLLM baseline

The recorded baseline used vLLM 0.30.0 with automatic prefix caching disabled.

A reproduction command matching the recorded configuration is:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct   --host 0.0.0.0   --port 8000   --dtype auto   --max-model-len 8192   --gpu-memory-utilization 0.90   --no-enable-prefix-caching
```

#### SGLang baseline

The recorded baseline used SGLang 0.5.9 with RadixAttention caching disabled:

```bash
python -m sglang.launch_server   --model-path Qwen/Qwen2.5-7B-Instruct   --host 0.0.0.0   --port 8000   --dtype auto   --context-length 8192   --disable-radix-cache
```

#### TensorRT-LLM baseline

The recorded TensorRT-LLM 1.2.1 run used the PyTorch backend with direct Hugging Face checkpoint loading:

```bash
trtllm-serve serve Qwen/Qwen2.5-7B-Instruct   --backend pytorch   --host 0.0.0.0   --port 8000   --config configs/trtllm.yaml
```

The configuration used:

```yaml
max_input_len: 4096
max_seq_len: 8192
max_num_tokens: 8192
kv_cache_config:
  enable_block_reuse: false
```

Disabling block reuse keeps cross-request prefix reuse out of the controlled baseline.

### 4. Run the four baseline workloads

With the selected server running, execute the matching script:

```bash
./scripts/run_vllm.sh
./scripts/run_sglang.sh
./scripts/run_trtllm.sh
```

Run only the script corresponding to the active server.

Each script executes the same four conditions:

1. interactive baseline: 512 input / 128 output / concurrency 4
2. high concurrency: 512 input / 128 output / concurrency 16
3. prefill-heavy: 4096 input / 64 output / concurrency 4
4. decode-heavy: 512 input / 512 output / concurrency 4

Reproduced outputs are written under `results/local/` so the committed baseline results are not overwritten.

### 5. Reproduce the SGLang shared-prefix test

The targeted SGLang experiment compares RadixAttention disabled with RadixAttention enabled.

For the disabled condition, start SGLang with:

```bash
--disable-radix-cache
```

For the enabled condition, restart the same SGLang server without `--disable-radix-cache`.

Then run:

```bash
python scripts/run_sglang_radix_shared_prefix.py
```

The published result is available at [results/sglang_radix_shared_prefix/results.txt](results/sglang_radix_shared_prefix/results.txt).

## Principle

Only experiments that answer a concrete serving question are included. Additional workloads or tuning are added only when they materially change the interpretation of engine behavior.

## Results

- [Controlled baseline summary](results/summary.csv)
- [SGLang RadixAttention shared-prefix demonstration](results/sglang_radix_shared_prefix/results.txt)
- [Full interpretation](analysis/findings.md)

## Status

Complete.

The controlled three-engine baseline is finished, and the targeted SGLang shared-prefix experiment demonstrated effective shared-prefix reuse with RadixAttention. No additional Experiment 02 benchmarking is planned.
