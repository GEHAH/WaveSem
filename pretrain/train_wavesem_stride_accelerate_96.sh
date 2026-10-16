#!/usr/bin/env bash

set -euo pipefail

# Reduce CPU oversubscription (DataLoader workers * OMP threads can cause stalls/jitter)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
# HDF5 file locking can cause stalls on some network/distributed filesystems (read-only access is safe to disable)
export HDF5_USE_FILE_LOCKING=${HDF5_USE_FILE_LOCKING:-"FALSE"}

# Usage:
#   bash ONT/ont_paper/ont_pretrain/train_wave_vqmodel_accelerate.sh
#
# Override via env vars, e.g.:
#   export DATA_PATH=/path/to/h5_dir
#   export BATCH_SIZE=64
#   export NUM_PROC=4
#   export BASE_LR=3e-4
#   export WEIGHT_DECAY=0.05
#   export WARMUP_EPOCHS=1
#   bash ONT/ont_paper/ont_pretrain/train_wave_vqmodel_accelerate.sh

ROOT_DIR="/owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain"
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Data
DATA_PATH=${DATA_PATH:-"/owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain/raw_data/temp_signal_segments_10240"}
SEGMENT_LENGTH=${SEGMENT_LENGTH:-10240}
MAX_FILES=${MAX_FILES:-10}      # 0 = all
NUM_WORKERS=${NUM_WORKERS:-16}
BATCH_SIZE=${BATCH_SIZE:-512}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-2}
NORMALIZE=${NORMALIZE:-"none"} # none|zscore|scale

# Model
HIDDEN_DIM=${HIDDEN_DIM:-256}
STRIDES=${STRIDES:-"[2,3]"}
WAVELET=${WAVELET:-"db4"}
LEARNABLE_WAVELET=${LEARNABLE_WAVELET:-1} # 1 to enable

# Loss weights
LAMBDA_TIME=${LAMBDA_TIME:-10.0}
LAMBDA_FREQ=${LAMBDA_FREQ:-1.0}
LAMBDA_SUB_LOW=${LAMBDA_SUB_LOW:-5.0}
LAMBDA_SUB_HIGH=${LAMBDA_SUB_HIGH:-1.0}
LAMBDA_VQ=${LAMBDA_VQ:-1.0}

# Train
EPOCHS=${EPOCHS:-50}
BASE_LR=${BASE_LR:-1e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.05}
GRAD_ACCUM=${GRAD_ACCUM:-8}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}
MIXED_PRECISION=${MIXED_PRECISION:-"bf16"}   # no|fp16|bf16
USE_SCHEDULER=${USE_SCHEDULER:-1}            # 1 to enable cosine
WARMUP_EPOCHS=${WARMUP_EPOCHS:-1}
LOG_EVERY=${LOG_EVERY:-100}
SEED=${SEED:-42}

# Accelerate launch
NUM_PROC=${NUM_PROC:-4}
MAIN_PORT=${MAIN_PORT:-29501}

# Output
RUN_NAME=${RUN_NAME:-"wavesem_pretrain_96"}
OUT_DIR=${OUT_DIR:-"/owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain/outputs/${RUN_NAME}_stride"}
MODEL_DIR="${OUT_DIR}/checkpoints"
LOG_DIR="${OUT_DIR}/logs"
PROJECT_NAME=${PROJECT_NAME:-"wavesem_stride_pretrain"}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-1}
RESUME_FROM=${RESUME_FROM:-'/owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain/outputs/wavesem_pretrain_96_stride/checkpoints/epoch_12.pth'}  # path to *.pth

mkdir -p "${MODEL_DIR}" "${LOG_DIR}"

ARGS=(
  "${CODE_DIR}/train_wavesem_stride_accelerate.py"
  --data_path "${DATA_PATH}"
  --segment_length "${SEGMENT_LENGTH}"
  --max_files "${MAX_FILES}"
  --num_workers "${NUM_WORKERS}"
  --prefetch_factor "${PREFETCH_FACTOR}"
  --batch_size "${BATCH_SIZE}"
  --normalize "${NORMALIZE}"
  --hidden_dim "${HIDDEN_DIM}"
  --strides "${STRIDES}"
  --wavelet "${WAVELET}"
  --lambda_time "${LAMBDA_TIME}"
  --lambda_freq "${LAMBDA_FREQ}"
  --lambda_sub_low "${LAMBDA_SUB_LOW}"
  --lambda_sub_high "${LAMBDA_SUB_HIGH}"
  --lambda_vq "${LAMBDA_VQ}"
  --num_epochs "${EPOCHS}"
  --optimizer "adamw"
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

if [[ "${LEARNABLE_WAVELET}" == "1" ]]; then
  ARGS+=( --learnable_wavelet )
fi

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


