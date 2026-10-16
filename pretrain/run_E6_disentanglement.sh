#!/usr/bin/env bash
# ============================================================================
# E6: 定量解耦指标 (回应 Reviewer3 R3.4)  —— 在 GPU 机器上运行
#   CKA / cross-stream R² / band-energy / spectral-flatness / autocorr / MI
# 用法:
#   bash run_E6_disentanglement.sh
#   CKPT=outputs/wave_vqmodel/checkpoints/best_model.pth bash run_E6_disentanglement.sh
# ============================================================================
set -euo pipefail

# PY=${PY:-/data/home/lqszchen/miniconda3/envs/llm3.10/bin/python}
DATA_ROOT="/owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKPT=${CKPT:-$DATA_ROOT/outputs/wave_vqmodel/checkpoints/best_model.pth}
DATA=${DATA:-$DATA_ROOT/raw_data/temp_signal_segments_10240}
OUT=${OUT:-$DATA_ROOT/outputs/disentanglement_quant}
NUM_SAMPLES=${NUM_SAMPLES:-5000}
BATCH_SIZE=${BATCH_SIZE:-64}
MAX_FILES=${MAX_FILES:-2}
export HDF5_USE_FILE_LOCKING=${HDF5_USE_FILE_LOCKING:-FALSE}

cd "$ROOT"
echo "══════ 环境自检 ══════"
python -c "import torch;assert torch.cuda.is_available(),'no GPU visible';print('GPU:',torch.cuda.get_device_name(0))"

echo "══════ E6: 定量解耦 (ckpt=$CKPT) ══════"
python eval_disentanglement_quant.py \
    --checkpoint "$CKPT" \
    --data_path "$DATA" \
    --output_dir "$OUT" \
    --num_samples "$NUM_SAMPLES" \
    --batch_size "$BATCH_SIZE" \
    --max_files "$MAX_FILES"

echo ""
echo "结果: $OUT/disentanglement_metrics.json"
