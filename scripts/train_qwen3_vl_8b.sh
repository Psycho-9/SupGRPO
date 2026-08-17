#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL_PATH="${MODEL_PATH:-models/Qwen3-VL-8B-Instruct}"
DATA_CONFIG="${DATA_CONFIG:-configs/text_spotting.yaml}"
IMAGE_ROOT="${IMAGE_ROOT:-data/images}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/Qwen3-VL-8B-SupGRPO}"
NPROC="${NPROC:-4}"
MASTER_PORT="${MASTER_PORT:-29501}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export SFT_COORD_LAMBDA="${SFT_COORD_LAMBDA:-1e-4}"
# This only slices the forward computation; the effective per-device batch remains 2.
export SUPGRPO_FORWARD_BATCH_SIZE="${SUPGRPO_FORWARD_BATCH_SIZE:-4}"

python -m torch.distributed.run \
  --nproc_per_node="$NPROC" \
  --master_port="$MASTER_PORT" \
  -m open_r1.train \
  --deepspeed configs/zero3.json \
  --output_dir "$OUTPUT_DIR" \
  --model_name_or_path "$MODEL_PATH" \
  --dataset_name "$DATA_CONFIG" \
  --image_root "$IMAGE_ROOT" \
  --max_completion_length 1024 \
  --max_pixels 12845056 \
  --min_pixels 3136 \
  --num_generations 8 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 1 \
  --logging_steps 1 \
  --bf16 \
  --torch_dtype bfloat16 \
  --data_seed 42 \
  --seed 42 \
  --report_to none \
  --gradient_checkpointing true \
  --gradient_checkpointing_kwargs '{"use_reentrant":false}' \
  --attn_implementation sdpa \
  --num_train_epochs 1 \
  --save_steps 500 \
  --save_total_limit 3 \
  --learning_rate 1e-6 \
  --beta 0.04 \
  --reward_funcs precision recall content format \
  --use_peft true \
  --lora_r 64 \
  --lora_alpha 128 \
  --lora_dropout 0.05 \
  --lora_task_type CAUSAL_LM \
  --freeze_vision_modules true \
  "$@"
