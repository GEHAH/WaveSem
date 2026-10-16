#!/usr/bin/env bash

set -euo pipefail
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export HDF5_USE_FILE_LOCKING=${HDF5_USE_FILE_LOCKING:-"FALSE"}

ROOT_DIR="/owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain"
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Data
DATA_PATH=${DATA_PATH:-"/owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain/raw_data/temp_signal_segments_10240"}
SEGMENT_LENGTH=${SEGMENT_LENGTH:-10240}
MAX_FILES=${MAX_FILES:-5}      # 0 = all
NUM_WORKERS=${NUM_WORKERS:-16}
BATCH_SIZE=${BATCH_SIZE:-128}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-2}
NORMALIZE=${NORMALIZE:-"none"}  # none|zscore|scale

# Model
HIDDEN_DIM=${HIDDEN_DIM:-256}
COMPRESSION_RATIO=${COMPRESSION_RATIO:-64}
CODEBOOK_SIZE=${CODEBOOK_SIZE:-512}
COMMITMENT_COST=${COMMITMENT_COST:-0.25}

# Loss
LAMBDA_TIME=${LAMBDA_TIME:-5.0}
LAMBDA_FREQ=${LAMBDA_FREQ:-1.0}
LAMBDA_VQ=${LAMBDA_VQ:-1.0}

# Train
EPOCHS=${EPOCHS:-5}
BASE_LR=${BASE_LR:-1e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.05}
GRAD_ACCUM=${GRAD_ACCUM:-4}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}
MIXED_PRECISION=${MIXED_PRECISION:-"bf16"}   # no|fp16|bf16
USE_SCHEDULER=${USE_SCHEDULER:-1}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-1}
LOG_EVERY=${LOG_EVERY:-100}
SEED=${SEED:-42}

# Accelerate launch
NUM_PROC=${NUM_PROC:-8}
MAIN_PORT=${MAIN_PORT:-29521}

# Output
RUN_NAME=${RUN_NAME:-"single_codebook_vqvae_cr${COMPRESSION_RATIO}"}
OUT_DIR=${OUT_DIR:-"${ROOT_DIR}/outputs/${RUN_NAME}"}
MODEL_DIR="${OUT_DIR}/checkpoints"
LOG_DIR="${OUT_DIR}/logs"
PROJECT_NAME=${PROJECT_NAME:-"single_codebook_vqvae_pretrain"}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-1}
RESUME_FROM=${RESUME_FROM:-""}

mkdir -p "${MODEL_DIR}" "${LOG_DIR}"

ARGS=(
  "${CODE_DIR}/train_single_codebook_vqvae_accelerate.py"
  --data_path "${DATA_PATH}"
  --segment_length "${SEGMENT_LENGTH}"
  --max_files "${MAX_FILES}"
  --num_workers "${NUM_WORKERS}"
  --prefetch_factor "${PREFETCH_FACTOR}"
  --batch_size "${BATCH_SIZE}"
  --normalize "${NORMALIZE}"
  --hidden_dim "${HIDDEN_DIM}"
  --cnn_compression_ratio "${COMPRESSION_RATIO}"
  --codebook_size "${CODEBOOK_SIZE}"
  --commitment_cost "${COMMITMENT_COST}"
  --lambda_time "${LAMBDA_TIME}"
  --lambda_freq "${LAMBDA_FREQ}"
  --lambda_vq "${LAMBDA_VQ}"
  --num_epochs "${EPOCHS}"
  --base_learning_rate "${BASE_LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --gradient_accumulation_steps "${GRAD_ACCUM}"
  --max_grad_norm "${MAX_GRAD_NORM}"
  --mixed_precision "${MIXED_PRECISION}"
  --warmup_epochs "${WARMUP_EPOCHS}"
  --log_every "${LOG_EVERY}"
  --seed "${SEED}"
  --model_save_path "${MODEL_DIR}"
  --log_dir "${LOG_DIR}"
  --project_name "${PROJECT_NAME}"
  --checkpoint_interval "${CHECKPOINT_INTERVAL}"
  --warmup_h5_in_workers
)

if [[ "${USE_SCHEDULER}" == "1" ]]; then
  ARGS+=( --use_scheduler )
fi

if [[ -n "${RESUME_FROM}" ]]; then
  ARGS+=( --resume_from "${RESUME_FROM}" )
fi

CMD=(
  accelerate launch
  --num_processes "${NUM_PROC}"
  --main_process_port "${MAIN_PORT}"
  "${ARGS[@]}"
)

TOTAL_WORKERS=$((NUM_WORKERS * NUM_PROC))
EFFECTIVE_GLOBAL_BATCH=$((BATCH_SIZE * NUM_PROC * GRAD_ACCUM))
echo "Info:"
echo "  NUM_PROC=${NUM_PROC}"
echo "  NUM_WORKERS(per-proc)=${NUM_WORKERS}  => total_workers=${TOTAL_WORKERS}"
echo "  BATCH_SIZE(per-proc)=${BATCH_SIZE} * GRAD_ACCUM=${GRAD_ACCUM} * NUM_PROC=${NUM_PROC} => effective_global_batch≈${EFFECTIVE_GLOBAL_BATCH}"
echo "  OMP_NUM_THREADS=${OMP_NUM_THREADS} MKL_NUM_THREADS=${MKL_NUM_THREADS} HDF5_USE_FILE_LOCKING=${HDF5_USE_FILE_LOCKING}"

echo "Running:"
printf '  %q' "${CMD[@]}"
echo

"${CMD[@]}"


