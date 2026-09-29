#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

RESULT_DIR="${RESULT_DIR:-$BENCHMARK_DIR/results/local}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

mkdir -p "$RESULT_DIR"

echo
echo "===== E3: 512 input / 64 output ====="

vllm bench serve \
  --backend vllm \
  --host "$HOST" \
  --port "$PORT" \
  --model "$MODEL" \
  --endpoint /v1/completions \
  --dataset-name random \
  --seed 42 \
  --num-prompts 60 \
  --random-input-len 512 \
  --random-output-len 64 \
  --random-range-ratio 0 \
  --max-concurrency 4 \
  --request-rate inf \
  --ignore-eos \
  --temperature 0 \
  --percentile-metrics ttft,tpot,itl,e2el \
  --save-result \
  --result-dir "$RESULT_DIR" \
  --result-filename "e3_decode_512x64.json"

echo
echo "===== E3: 512 input / 512 output ====="

vllm bench serve \
  --backend vllm \
  --host "$HOST" \
  --port "$PORT" \
  --model "$MODEL" \
  --endpoint /v1/completions \
  --dataset-name random \
  --seed 42 \
  --num-prompts 60 \
  --random-input-len 512 \
  --random-output-len 512 \
  --random-range-ratio 0 \
  --max-concurrency 4 \
  --request-rate inf \
  --ignore-eos \
  --temperature 0 \
  --percentile-metrics ttft,tpot,itl,e2el \
  --save-result \
  --result-dir "$RESULT_DIR" \
  --result-filename "e3_decode_512x512.json"