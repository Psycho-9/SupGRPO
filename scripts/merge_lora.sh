#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL_PATH="${MODEL_PATH:-models/Qwen3-VL-8B-Instruct}"
ADAPTER_PATH="${ADAPTER_PATH:-outputs/Qwen3-VL-8B-SupGRPO}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/Qwen3-VL-8B-SupGRPO-merged}"

python scripts/merge_lora.py \
  --model-path "$MODEL_PATH" \
  --adapter-path "$ADAPTER_PATH" \
  --output-dir "$OUTPUT_DIR"

