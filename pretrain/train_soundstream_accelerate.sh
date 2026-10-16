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
MAX_FILES=${MAX_FILES:-10}      # 0 = all
NUM_WORKERS=${NUM_WORKERS:-16}
BATCH_SIZE=${BATCH_SIZE:-256}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-2}
NORMALIZE=${NORMALIZE:-"none"}  # none|zscore|scale

# Model (SoundStream)
CHANNELS=${CHANNELS:-32}
STRIDES=${STRIDES:-"2 4 4 2"}
CHANNEL_MULTS=${CHANNEL_MULTS:-"2 4 8 16"}
CODEBOOK_DIM=${CODEBOOK_DIM:-512}
CODEBOOK_SIZE=${CODEBOOK_SIZE:-256}
RQ_NUM=${RQ_NUM:-8}
RQ_COMMIT=${RQ_COMMIT:-1.0}
RQ_EMA_DECAY=${RQ_EMA_DECAY:-0.95}
RQ_GROUPS=${RQ_GROUPS:-1}
RQ_STOCHASTIC=${RQ_STOCHASTIC:-0}
QUANTIZE_DROPOUT_CUTOFF=${QUANTIZE_DROPOUT_CUTOFF:-1}
TARGET_SAMPLE_HZ=${TARGET_SAMPLE_HZ:-4000}
RECON_WEIGHT=${RECON_WEIGHT:-1.0}
MULTI_SPEC_WEIGHT=${MULTI_SPEC_WEIGHT:-1e-5}
ADV_WEIGHT=${ADV_WEIGHT:-1.0}
FEATURE_WEIGHT=${FEATURE_WEIGHT:-100.0}
STFT_NORMALIZED=${STFT_NORMALIZED:-0}
DISABLE_LOCAL_ATTN=${DISABLE_LOCAL_ATTN:-0}
ATTN_WINDOW_SIZE=${ATTN_WINDOW_SIZE:-128}
ATTN_DIM_HEAD=${ATTN_DIM_HEAD:-64}
ATTN_HEADS=${ATTN_HEADS:-8}
ATTN_DEPTH=${ATTN_DEPTH:-1}
USE_FSQ=${USE_FSQ:-0}
FSQ_LEVELS=${FSQ_LEVELS:-""}      # e.g. "8,5,5"

# Train
EPOCHS=${EPOCHS:-50}
BASE_LR=${BASE_LR:-3e-4}
DISC_LR=${DISC_LR:-""}            # optional
WEIGHT_DECAY=${WEIGHT_DECAY:-0.05}
GRAD_ACCUM=${GRAD_ACCUM:-4}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}
DISC_MAX_GRAD_NORM=${DISC_MAX_GRAD_NORM:-1.0}
MIXED_PRECISION=${MIXED_PRECISION:-"bf16"}   # no|fp16|bf16
USE_SCHEDULER=${USE_SCHEDULER:-1}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-1}
LOG_EVERY=${LOG_EVERY:-100}
SEED=${SEED:-42}
APPLY_GRAD_PENALTY=${APPLY_GRAD_PENALTY:-0}

# Accelerate launch
NUM_PROC=${NUM_PROC:-4}
MAIN_PORT=${MAIN_PORT:-29519}

# Output
RUN_NAME=${RUN_NAME:-"soundstream"}
OUT_DIR=${OUT_DIR:-"${ROOT_DIR}/outputs/${RUN_NAME}"}
MODEL_DIR="${OUT_DIR}/checkpoints"
LOG_DIR="${OUT_DIR}/logs"
PROJECT_NAME=${PROJECT_NAME:-"soundstream_pretrain"}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-1}
RESUME_FROM=${RESUME_FROM:-""}

mkdir -p "${MODEL_DIR}" "${LOG_DIR}"

ARGS=(
  "${CODE_DIR}/train_soundstream_accelerate.py"
  --data_path "${DATA_PATH}"
  --segment_length "${SEGMENT_LENGTH}"
  --max_files "${MAX_FILES}"
  --num_workers "${NUM_WORKERS}"
  --prefetch_factor "${PREFETCH_FACTOR}"
  --batch_size "${BATCH_SIZE}"
  --normalize "${NORMALIZE}"
  --channels "${CHANNELS}"
  --strides ${STRIDES}
  --channel_mults ${CHANNEL_MULTS}
  --codebook_dim "${CODEBOOK_DIM}"
  --codebook_size "${CODEBOOK_SIZE}"
  --rq_num_quantizers "${RQ_NUM}"
  --rq_commitment_weight "${RQ_COMMIT}"
  --rq_ema_decay "${RQ_EMA_DECAY}"
  --rq_groups "${RQ_GROUPS}"
  --quantize_dropout_cutoff_index "${QUANTIZE_DROPOUT_CUTOFF}"
  --target_sample_hz "${TARGET_SAMPLE_HZ}"
  --recon_loss_weight "${RECON_WEIGHT}"
  --multi_spectral_recon_loss_weight "${MULTI_SPEC_WEIGHT}"
  --adversarial_loss_weight "${ADV_WEIGHT}"
  --feature_loss_weight "${FEATURE_WEIGHT}"
  --attn_window_size "${ATTN_WINDOW_SIZE}"
  --attn_dim_head "${ATTN_DIM_HEAD}"
  --attn_heads "${ATTN_HEADS}"
  --attn_depth "${ATTN_DEPTH}"
  --num_epochs "${EPOCHS}"
  --base_learning_rate "${BASE_LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --gradient_accumulation_steps "${GRAD_ACCUM}"
  --max_grad_norm "${MAX_GRAD_NORM}"
  --disc_max_grad_norm "${DISC_MAX_GRAD_NORM}"
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

if [[ -n "${DISC_LR}" ]]; then
  ARGS+=( --disc_learning_rate "${DISC_LR}" )
fi

if [[ "${RQ_STOCHASTIC}" == "1" ]]; then
  ARGS+=( --rq_stochastic_sample_codes )
fi

if [[ "${STFT_NORMALIZED}" == "1" ]]; then
  ARGS+=( --stft_normalized )
fi

if [[ "${DISABLE_LOCAL_ATTN}" == "1" ]]; then
  ARGS+=( --disable_local_attn )
fi

if [[ "${APPLY_GRAD_PENALTY}" == "1" ]]; then
  ARGS+=( --apply_grad_penalty )
fi

if [[ "${USE_FSQ}" == "1" ]]; then
  if [[ -z "${FSQ_LEVELS}" ]]; then
    echo "FSQ is enabled but FSQ_LEVELS is empty (e.g. 8,5,5)"; exit 1;
  fi
  ARGS+=( --use_finite_scalar_quantizer --finite_scalar_quantizer_levels "${FSQ_LEVELS}" )
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


