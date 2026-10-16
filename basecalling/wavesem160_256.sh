#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${ROOT_DIR}/scripts/train_original_zip.py"
OUTPUT_DIR="${ROOT_DIR}/results"
WINDOW_SIZE=4000
NUM_EPOCHS=60
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}

run_job() {
  local model_path="$1"
  local batch_size="$2"
  local decoder_stride="$3"
  local strides="$4"
  local out_dir="$5"

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python "${TRAIN_SCRIPT}" \
    --model vq_model_wavesem_stride \
    --vqvae_model_path "${model_path}" \
    --window-size "${WINDOW_SIZE}" \
    --num-epochs "${NUM_EPOCHS}" \
    --batch-size "${batch_size}" \
    --decoder_stride "${decoder_stride}" \
    --strides "${strides}" \
    --output-dir "${out_dir}"
}

# 96
run_job \
  "/owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain/outputs/wavesem_pretrain_128_stride/checkpoints/best_model.pth" \
  256 \
  "[2,5,2]" \
  "[2,5]" \
  "${OUTPUT_DIR}/wave_160"

# 128
run_job \
  "/owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain/outputs/wavesem_pretrain_160_stride/checkpoints/best_model.pth" \
  256 \
  "[2,4,4]" \
  "[2,4,2]" \
  "${OUTPUT_DIR}/wave_256"
