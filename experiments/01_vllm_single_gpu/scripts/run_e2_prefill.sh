#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

RESULT_DIR="${RESULT_DIR:-$STUDY_DIR/results/L40S_2026-09-26}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"

mkdir -p "$RESULT_DIR"

echo
echo "===== E2: 512 input / 64 output ====="

vllm bench serve \
  --backend vllm \
  --host 127.0.0.1 \
  --port 8000 \
  --model "$MODEL" \
  --endpoint /v1/completions \
  --dataset-name random \
  --seed 42 \
  --num-prompts 80 \
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
  --result-filename "e2_prefill_512x64.json"

echo
echo "===== E2: 4096 input / 64 output ====="

vllm bench serve \
  --backend vllm \
  --host 127.0.0.1 \
  --port 8000 \
  --model "$MODEL" \
  --endpoint /v1/completions \
  --dataset-name random \
  --seed 42 \
  --num-prompts 80 \
  --random-input-len 4096 \
  --random-output-len 64 \
  --random-range-ratio 0 \
  --max-concurrency 4 \
  --request-rate inf \
  --ignore-eos \
  --temperature 0 \
  --percentile-metrics ttft,tpot,itl,e2el \
  --save-result \
  --result-dir "$RESULT_DIR" \
  --result-filename "e2_prefill_4096x64.json"