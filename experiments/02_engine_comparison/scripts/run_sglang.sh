#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

OUT="${OUT:-$EXP_DIR/results/local/sglang}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
URL="http://${HOST}:${PORT}"

run_case () {
    local name="$1"
    local input_tokens="$2"
    local output_tokens="$3"
    local concurrency="$4"

    echo
    echo "=================================================="
    echo "Running: $name"
    echo "Input=$input_tokens Output=$output_tokens Concurrency=$concurrency"
    echo "=================================================="

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

run_case interactive_c4       512  128 4
run_case high_concurrency_c16 512  128 16
run_case prefill_heavy        4096 64  4
run_case decode_heavy         512  512 4