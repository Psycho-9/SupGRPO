#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL_PATH="${MODEL_PATH:-models/Qwen3-VL-8B-Instruct}"
ADAPTER_PATH="${ADAPTER_PATH:-outputs/Qwen3-VL-8B-SupGRPO}"
DATA_JSON="${DATA_JSON:-data/test.json}"
IMAGE_ROOT="${IMAGE_ROOT:-data/images}"
GT_FILE="${GT_FILE:-data/test.json}"
OUTPUT_DIR="${OUTPUT_DIR:-evaluation/results/Qwen3-VL-8B-SupGRPO}"

mkdir -p "$OUTPUT_DIR"
python evaluation/infer_qwen3_vl.py \
  --model-path "$MODEL_PATH" \
  --adapter-path "$ADAPTER_PATH" \
  --data-json "$DATA_JSON" \
  --image-root "$IMAGE_ROOT" \
  --output "$OUTPUT_DIR/predictions_normalized.jsonl" \
  --num-gpus "${NUM_GPUS:-4}"
python evaluation/convert_pred_coords.py \
  --pred "$OUTPUT_DIR/predictions_normalized.jsonl" \
  --out "$OUTPUT_DIR/predictions.jsonl" \
  --image-root "$IMAGE_ROOT" \
  --mode normalized_to_raw

cd "$ROOT/evaluation"
GT_FILE_PATH="$ROOT/$GT_FILE" PRED_FILE_PATH="$ROOT/$OUTPUT_DIR/predictions.jsonl" \
  python evaluation_rec.py | tee "$ROOT/$OUTPUT_DIR/recognition.txt"
GT_FILE_PATH="$ROOT/$GT_FILE" PRED_FILE_PATH="$ROOT/$OUTPUT_DIR/predictions.jsonl" \
  python evaluation_det.py | tee "$ROOT/$OUTPUT_DIR/detection.txt"
GT_FILE_PATH="$ROOT/$GT_FILE" PRED_FILE_PATH="$ROOT/$OUTPUT_DIR/predictions.jsonl" \
  python evaluation_e2e.py | tee "$ROOT/$OUTPUT_DIR/end_to_end.txt"

