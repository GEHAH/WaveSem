#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# WaveSem 5mC E. coli 端到端微调（不冻结编码器）
# ═══════════════════════════════════════════════════════════════════
#
# 与 Linear Probing（冻结编码器）不同，此脚本解冻整个 WaveSem encoder，
# 以端到端方式微调，编码器使用 0.1x 学习率，分类头使用标准学习率。
#
# 预期效果：AUROC / F1 显著提升，因为编码器可以学习适配甲基化检测任务。
#

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python run_ecoli_5mC_experiment.py \
    --no_freeze_encoder \
    --epochs 50 \
    --batch_size 32 \
    --lr 5e-4 \
    --warmup_epochs 5 \
    --weight_decay 0.01 \
    --head_type attention \
    --output_dir outputs/methylation_5mC_ecoli_finetune \
    --log_every 50

echo ""
echo "=========================================="
echo " Fine-tuning experiment completed!"
echo " Results saved to: outputs/methylation_5mC_ecoli_finetune/"
echo "=========================================="
