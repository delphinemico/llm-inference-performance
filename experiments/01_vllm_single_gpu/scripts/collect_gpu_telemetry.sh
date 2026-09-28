#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

RESULT_DIR="${RESULT_DIR:-$STUDY_DIR/results/L40S_2026-09-26}"

mkdir -p "$RESULT_DIR"

echo "Collecting GPU telemetry every 1 second"
echo "Output: $RESULT_DIR/gpu_telemetry.csv"
echo "Press Ctrl+C to stop."

nvidia-smi \
  --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,power.draw \
  --format=csv \
  -l 1 | tee "$RESULT_DIR/gpu_telemetry.csv"