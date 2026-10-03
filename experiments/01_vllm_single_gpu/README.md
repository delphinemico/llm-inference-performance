# Benchmark 01: Single-GPU vLLM Serving Characterization

## Objective

Characterize single-GPU LLM inference behavior under controlled changes in concurrency, prompt length, generation length, and automatic prefix caching.

The benchmark focuses on service-level latency and throughput behavior for admitted requests under closed-loop saturation workloads.

## Environment

- Platform: RunPod Secure Cloud
- GPU: NVIDIA L40S 48 GB
- Model: Qwen/Qwen2.5-7B-Instruct
- Serving engine: vLLM 0.30.0
- NVIDIA driver: 580.159.04
- CUDA reported by `nvidia-smi`: 13.0
- Configured dtype: `auto`
- Maximum model length: 8192
- GPU memory utilization target: 0.90
- Deterministic decoding: `temperature = 0`
- Benchmark client and vLLM server ran on the same host
- Benchmark traffic used the loopback interface at `127.0.0.1:8000`

The exact Python, PyTorch, resolved model dtype, and Hugging Face model revision were not separately pinned in the original benchmark run.

## Workload Model

These benchmarks use closed-loop saturation workloads. All requests are made available immediately with `--request-rate inf`, while `--max-concurrency` limits the number of in-flight requests.

Reported TTFT and end-to-end latency represent service-level latency after request admission and exclude load-generator queue time. The results should therefore not be interpreted as open-loop production arrival latency.

Synthetic random workloads with fixed input and output lengths are used to isolate the serving variable being tested.

## Experiments

### E1: Concurrency Scaling

The input and output lengths are held constant while maximum concurrency increases.

- Input length: 512 tokens
- Output length: 128 tokens
- Requests per condition: 100
- Concurrency: 1, 4, 8, 16
- Prefix caching: disabled

Purpose: characterize the tradeoff between aggregate throughput and service-level latency as concurrent load increases.

### E2: Prefill Scaling

Output length and concurrency are held constant while prompt length increases.

Conditions:

- 512 input / 64 output
- 4096 input / 64 output

Configuration:

- Requests per condition: 80
- Concurrency: 4
- Prefix caching: disabled

Purpose: isolate the effect of increased prompt processing on TTFT, throughput, and end-to-end latency.

### E3: Decode Scaling

Input length and concurrency are held constant while generation length increases.

Conditions:

- 512 input / 64 output
- 512 input / 512 output

Configuration:

- Requests per condition: 60
- Concurrency: 4
- Prefix caching: disabled

Purpose: characterize the effect of longer autoregressive generation on completion latency and request throughput.

### E4: Automatic Prefix Caching

The workload uses a 2048-token input composed of a shared prefix and variable suffix.

Configuration:

- Shared prefix: 1536 tokens
- Variable input: 512 tokens
- Output length: 64 tokens
- Requests per measured condition: 100
- Concurrency: 4
- Seed: 42

Conditions:

- Prefix caching disabled
- Prefix caching enabled with a prewarmed shared prefix

For the cache-enabled condition, the shared prefix is warmed with 10 requests before the measured benchmark run. The reported cache-enabled result therefore represents a warmed, steady-state exact-prefix cache-hit condition rather than a cold-start comparison.

Purpose: measure the impact of prefix reuse on prefill-related latency and throughput.

## Metrics

The benchmark reports:

- request throughput
- output-token throughput
- Time to First Token (TTFT)
- Time per Output Token (TPOT)
- Inter-Token Latency (ITL)
- end-to-end latency

P99 values are observed within individual benchmark runs and should not be interpreted as robust production tail-latency estimates.

## Selected Results

### Concurrency Scaling

Increasing concurrency from 1 to 16 increased output-token throughput from 49.0 to 567.6 tokens/s. Over the same range, median TTFT increased from 53.4 ms to 365.4 ms.

### Prefill Scaling

Increasing input length from 512 to 4096 tokens increased median TTFT from 135.2 ms to 791.4 ms while reducing request throughput from 2.687 to 1.547 requests/s.

### Decode Scaling

Increasing output length from 64 to 512 tokens left median TTFT and TPOT nearly unchanged, while median end-to-end latency increased from approximately 1.48 s to 10.95 s.

### Prefix Caching

For the prewarmed cache-enabled condition, median TTFT was 164.6 ms compared with 464.0 ms with caching disabled. Output-token throughput increased from 130.5 to 168.8 tokens/s.

See the full analysis in [analysis/findings.md](analysis/findings.md).

## Results and Analysis

- [Raw benchmark results](results/L40S_2026-09-26/)
- [Summary CSV](results/L40S_2026-09-26/summary.csv)
- [Detailed findings](analysis/findings.md)
- [Figures](figures/)
- [Benchmark scripts](scripts/)

The published `L40S_2026-09-26` directory is the baseline result set used by the committed findings and figures.

## Reproduction

The original benchmark was executed on a Linux GPU host. The vLLM server and benchmark client ran on the same host, with benchmark traffic sent over the loopback interface.

The benchmark workflow uses separate terminals so that the serving process remains active while experiments are executed.

### Terminal Roles

- **Terminal 1:** vLLM server
- **Terminal 2:** benchmark execution
- **Terminal 3:** optional GPU telemetry collection

Terminal 3 is not required to reproduce the reported latency and throughput results. The published telemetry is retained as auxiliary data and is not used in the benchmark conclusions.

### 1. Clone the Repository

```bash
git clone https://github.com/delphinemico/llm-inference-performance.git
cd llm-inference-performance
```

### 2. Prepare the Environment

Create and activate a Python virtual environment.

On Linux or macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, for analysis and figure-generation workflows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the repository-level analysis dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The benchmark serving environment additionally requires vLLM 0.30.0 on a compatible GPU system:

```bash
python -m pip install vllm==0.30.0
```

The benchmark scripts under `experiments/01_vllm_single_gpu/scripts/` are Bash scripts and are intended to run in a Linux-compatible shell.

Change to the benchmark directory:

```bash
cd experiments/01_vllm_single_gpu
```

### 3. Terminal 1: Start vLLM

Start the server with automatic prefix caching disabled:

```bash
./scripts/start_vllm.sh off
```

The script defaults to:

- host: `127.0.0.1`
- port: `8000`
- model: `Qwen/Qwen2.5-7B-Instruct`

These can be overridden with environment variables such as `HOST`, `PORT`, and `MODEL`.

Leave this terminal running while the benchmark client executes in Terminal 2.

### 4. Terminal 2: Verify Server Readiness

Open a second terminal on the same host and change to the benchmark directory.

Verify that the server is ready:

```bash
curl "http://${HOST:-127.0.0.1}:${PORT:-8000}/v1/models"
```

Proceed after the endpoint responds successfully.

### 5. Terminal 3: Optional GPU Telemetry

If GPU telemetry is being collected, open a third terminal and run:

```bash
./scripts/collect_gpu_telemetry.sh
```

Telemetry collection is optional for reproducing the reported benchmark results.

### 6. Terminal 2: Run E1, E2, and E3

With the cache-disabled server still running in Terminal 1:

```bash
./scripts/run_e1_concurrency.sh
./scripts/run_e2_prefill.sh
./scripts/run_e3_decode.sh
```

### 7. Terminal 2: Run E4 with Prefix Caching Disabled

```bash
./scripts/run_e4_prefix_cache.sh off
```

### 8. Terminal 1: Restart with Prefix Caching Enabled

Stop the cache-disabled vLLM server in Terminal 1.

Restart it with automatic prefix caching enabled:

```bash
./scripts/start_vllm.sh on
```

Wait for the server to become ready.

### 9. Terminal 2: Run E4 with Prefix Caching Enabled

```bash
./scripts/run_e4_prefix_cache.sh on
```

The cache-enabled script performs a 10-request shared-prefix warmup before the measured benchmark run. The published cache-enabled result therefore represents a warmed cache-hit condition.

### 10. Generate the Summary

After the benchmark runs are complete:

```bash
python scripts/generate_summary.py
```

### 11. Regenerate the Figures

```bash
python scripts/generate_figures.py
```

## Output Locations

Published baseline results:

```text
results/L40S_2026-09-26/
```

Generated local benchmark results and summary:

```
results/local/
```

Generated local figures:

```text
figures/local/
```

Detailed interpretation:

```text
analysis/findings.md
```

## Limitations

This benchmark characterizes one controlled serving configuration rather than provide universal performance claims.

The current results are limited to:

- one NVIDIA L40S GPU
- one model
- one serving engine
- synthetic fixed-length random workloads
- a limited set of concurrency and sequence-length configurations
- one primary benchmark run per reported condition
- a prewarmed cache-enabled condition for the prefix-caching comparison

Production traffic can include heterogeneous request lengths, bursty arrivals, application-level preprocessing, retrieval or tool latency, different cache-hit distributions, and multi-tenant interference.

This experiment is followed by [Experiment 02: Serving Engine Comparison](../02_engine_comparison/README.md) and [Experiment 03: GPU Runtime Profiling](../03_gpu_runtime_profiling/README.md).
