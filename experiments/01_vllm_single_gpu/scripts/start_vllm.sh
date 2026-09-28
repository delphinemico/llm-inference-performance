#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./start_vllm.sh off
#   ./start_vllm.sh on
#
# "off" = prefix caching disabled
# "on"  = prefix caching enabled

CACHE_MODE="${1:-off}"

export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"

case "$CACHE_MODE" in
  off)
    CACHE_FLAG="--no-enable-prefix-caching"
    ;;
  on)
    CACHE_FLAG="--enable-prefix-caching"
    ;;
  *)
    echo "Usage: $0 {off|on}"
    exit 1
    ;;
esac

echo "Starting vLLM"
echo "Model: $MODEL"
echo "Prefix caching: $CACHE_MODE"

vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  "$CACHE_FLAG"