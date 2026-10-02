#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/llm-inference-performance"
OUT="$ROOT/experiments/02_engine_comparison/results/sglang"
URL="http://127.0.0.1:8000"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"

run_case () {
    NAME="$1"
    INPUT="$2"
    OUTPUT="$3"
    CONCURRENCY="$4"

    echo
    echo "=================================================="
    echo "Running: $NAME"
    echo "Input=$INPUT Output=$OUTPUT Concurrency=$CONCURRENCY"
    echo "=================================================="

    aiperf profile \
      --model "$MODEL" \
      --endpoint-type completions \
      --endpoint /v1/completions \
      --streaming \
      --url "$URL" \
      --synthetic-input-tokens-mean "$INPUT" \
      --synthetic-input-tokens-stddev 0 \
      --output-tokens-mean "$OUTPUT" \
      --output-tokens-stddev 0 \
      --extra-inputs temperature:0 \
      --extra-inputs min_tokens:"$OUTPUT" \
      --extra-inputs ignore_eos:true \
      --concurrency "$CONCURRENCY" \
      --request-count 100 \
      --warmup-request-count 16 \
      --random-seed 42 \
      --gpu-telemetry pynvml \
      --artifact-dir "$OUT/$NAME"
}

run_case interactive_c4       512  128 4
run_case high_concurrency_c16 512  128 16
run_case prefill_heavy        4096 64  4
run_case decode_heavy         512  512 4
