#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${ROOT_DIR}/scripts/train_original_zip.py"
OUTPUT_DIR="${ROOT_DIR}/results"
WINDOW_SIZE=4000
NUM_EPOCHS=50
BATCH_SIZE=256
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

run_job() {
  local model_path="$1"
  local compression_ratio="$2"
  local decoder_stride="$3"
  local out_dir="$4"

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python "${TRAIN_SCRIPT}" \
    --model vq_model \
    --vqvae_model_path "${model_path}" \
    --cnn_compression_ratio "${compression_ratio}" \
    --batch-size "${BATCH_SIZE}" \
    --window-size "${WINDOW_SIZE}" \
    --num-epochs "${NUM_EPOCHS}" \
    --decoder_stride "${decoder_stride}" \
    --output-dir "${out_dir}"
}

# 16
run_job \
  "/owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain/outputs/wave_vqmodel_small_cnn_16/checkpoints/best_model.pth" \
  1 \
  "[2]" \
  "${OUTPUT_DIR}/wave_16"

# 32
run_job \
  "/owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain/outputs/wave_vqmodel_small_cnn_32/checkpoints/best_model.pth" \
  2 \
  "[2,2]" \
  "${OUTPUT_DIR}/wave_32"
