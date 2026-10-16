#!/usr/bin/env bash
#
# Texture Stream 消融实验启动脚本
#
# 审稿人 Review 1 要求: 移除 Texture Stream 后仅依赖 Semantic Stream 进行重构
# 对比三种模式:
#   1. full      — 完整模型 (Semantic + Texture)
#   2. low_only  — 仅语义流 (高频置零)
#   3. high_only — 仅纹理流 (低频置零)
#
# Usage:
#   bash eval_texture_stream_ablation.sh
#   # 或指定参数:
#   MAX_FILES=5 NUM_SAMPLES=10000 bash eval_texture_stream_ablation.sh

set -euo pipefail

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export HDF5_USE_FILE_LOCKING=${HDF5_USE_FILE_LOCKING:-"FALSE"}

ROOT_DIR="/owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain"
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Paths ──
CHECKPOINT=${CHECKPOINT:-"${ROOT_DIR}/outputs/wave_vqmodel/checkpoints/best_model.pth"}
DATA_PATH=${DATA_PATH:-"${ROOT_DIR}/raw_data/temp_signal_segments_10240"}
OUTPUT_DIR=${OUTPUT_DIR:-"${ROOT_DIR}/outputs/texture_stream_ablation"}

# ── Data ──
MAX_FILES=${MAX_FILES:-2}           # 用几个H5文件 (0=全部)
NUM_SAMPLES=${NUM_SAMPLES:-5000}    # 验证集样本数 (0=全部)
BATCH_SIZE=${BATCH_SIZE:-64}
NUM_WORKERS=${NUM_WORKERS:-4}
SEGMENT_LENGTH=${SEGMENT_LENGTH:-10240}
NORMALIZE=${NORMALIZE:-"none"}
SEED=${SEED:-42}

# ── Model ──
HIDDEN_DIM=${HIDDEN_DIM:-256}
CNN_COMPRESSION_RATIO=${CNN_COMPRESSION_RATIO:-4}
WAVELET=${WAVELET:-"db4"}

# ── Eval ──
MODES=${MODES:-"full,low_only,high_only"}    # 要运行的模式
SAVE_EXAMPLES=${SAVE_EXAMPLES:-10}            # 保存的示例数
DEVICE=${DEVICE:-"cuda"}

mkdir -p "${OUTPUT_DIR}"

echo "═══════════════════════════════════════════════════════════"
echo "  Texture Stream Ablation Experiment"
echo "═══════════════════════════════════════════════════════════"
echo "  Checkpoint : ${CHECKPOINT}"
echo "  Data       : ${DATA_PATH}"
echo "  Max files  : ${MAX_FILES}"
echo "  Num samples: ${NUM_SAMPLES}"
echo "  Batch size : ${BATCH_SIZE}"
echo "  Modes      : ${MODES}"
echo "  Output     : ${OUTPUT_DIR}"
echo "  Device     : ${DEVICE}"
echo "═══════════════════════════════════════════════════════════"

cd "${CODE_DIR}"

python eval_texture_stream_ablation.py \
    --checkpoint "${CHECKPOINT}" \
    --data_path "${DATA_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --hidden_dim "${HIDDEN_DIM}" \
    --cnn_compression_ratio "${CNN_COMPRESSION_RATIO}" \
    --wavelet "${WAVELET}" \
    --learnable_wavelet \
    --segment_length "${SEGMENT_LENGTH}" \
    --max_files "${MAX_FILES}" \
    --num_samples "${NUM_SAMPLES}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --normalize "${NORMALIZE}" \
    --seed "${SEED}" \
    --save_examples "${SAVE_EXAMPLES}" \
    --modes "${MODES}" \
    --device "${DEVICE}"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  ✅ Done! Results saved to: ${OUTPUT_DIR}"
echo "═══════════════════════════════════════════════════════════"
