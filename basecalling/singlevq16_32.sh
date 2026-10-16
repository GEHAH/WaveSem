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
  local decoder_stride="$2"
  local strides="$3"
  local compression_ratio="$4"
  local out_dir="$5"

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python "${TRAIN_SCRIPT}" \
    --model single_codebook_vqvae_stride_basecalling \
    --vqvae_model_path "${model_path}" \
    --batch-size "${BATCH_SIZE}" \
    --window-size "${WINDOW_SIZE}" \
    --num-epochs "${NUM_EPOCHS}" \
    --strides "${strides}" \
    --decoder_stride "${decoder_stride}" \
    --cnn_compression_ratio "${compression_ratio}" \
    --output-dir "${out_dir}"
}

# 16
run_job \
  "/owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain/outputs/single_codebook_vqvae_cr16/checkpoints/best_model.pth" \
  "[2]" \
  "[2,2,2,2]" \
  16 \
  "${OUTPUT_DIR}/singlevq_16"

# 32
run_job \
  "/owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain/outputs/single_codebook_vqvae_cr32/checkpoints/best_model.pth" \
  "[2,2]" \
  "[2,2,2,2,2]" \
  32 \
  "${OUTPUT_DIR}/singlevq_32"
