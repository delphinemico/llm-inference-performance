#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/llm-inference-performance"
OUT="$ROOT/experiments/02_engine_comparison/results/trtllm"
URL="http://127.0.0.1:8000"
MODEL="Qwen/Qwen2.5-7B-Instruct"

run_workload() {
  local name="$1"
  local input_tokens="$2"
  local output_tokens="$3"
  local concurrency="$4"

  mkdir -p "$OUT/$name"

  aiperf profile \
    --model "$MODEL" \
    --endpoint-type completions \
    --endpoint /v1/completions \
    --streaming \
    --url "$URL" \
    --synthetic-input-tokens-mean "$input_tokens" \
    --synthetic-input-tokens-stddev 0 \
    --output-tokens-mean "$output_tokens" \
    --output-tokens-stddev 0 \
    --extra-inputs temperature:0 \
    --extra-inputs min_tokens:"$output_tokens" \
    --extra-inputs ignore_eos:true \
    --concurrency "$concurrency" \
    --request-count 100 \
    --warmup-request-count 16 \
    --random-seed 42 \
    --gpu-telemetry pynvml \
    --artifact-dir "$OUT/$name"
}

run_workload interactive_c4 512 128 4
run_workload high_concurrency_c16 512 128 16
run_workload prefill_heavy 4096 64 4
run_workload decode_heavy 512 512 4
