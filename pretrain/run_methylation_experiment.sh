#!/usr/bin/env bash
# ==============================================================================
# run_methylation_experiment.sh
# ==============================================================================
# ONT 甲基化检测实验完整工作流
#
# 用法:
#   bash ont_pretrain/run_methylation_experiment.sh
#
# 前提:
#   1. 已安装: dorado, samtools, modkit, pod5, pysam
#   2. 已有预训练的 WaveSem 模型 checkpoint
#   3. 已有 ONT POD5 原始数据
# ==============================================================================

set -euo pipefail

# ──── 配置 ────────────────────────────────────────────────────────────────────

ROOT_DIR="/owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain"
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 输入数据路径 (请根据实际情况修改)
POD5_DIR="/apdcephfs_qy3/share_301997302/chenchenqin/buddy1/jayjlzhou/giab_2025.01/flowcells/HG001/PAW79146/pod5"
REFERENCE="/path/to/hg38.fa"  # 参考基因组

# 预训练模型 checkpoint
PRETRAINED_CKPT="${ROOT_DIR}/outputs/wave_vqmodel_Equal/checkpoints/best_model.pth"

# 修饰类型 (5mC 或 6mA)
MOD_TYPE="5mC"
MOD_CODE="m"   # m=5mC, a=6mA

# 输出目录
METH_DATA_DIR="${ROOT_DIR}/methylation_data"
OUTPUT_DIR="${ROOT_DIR}/outputs/methylation_${MOD_TYPE}"

# 训练参数
EPOCHS=50
BATCH_SIZE=64
LR=1e-3
FREEZE_ENCODER=1  # 1 = linear probing, 0 = fine-tuning

mkdir -p "${METH_DATA_DIR}" "${OUTPUT_DIR}"

echo "============================================================"
echo " ONT 甲基化检测实验 (${MOD_TYPE})"
echo "============================================================"
echo " POD5 数据:      ${POD5_DIR}"
echo " 预训练模型:     ${PRETRAINED_CKPT}"
echo " 输出目录:       ${OUTPUT_DIR}"
echo "============================================================"

# ──── Step 1: Dorado Basecalling (带修饰检测) ─────────────────────────────────

echo ""
echo ">>> Step 1: Dorado Basecalling with modification detection"
echo "    (如果已有带 MM/ML 标签的 BAM 文件，可跳过此步)"
echo ""

BAM_FILE="${METH_DATA_DIR}/${MOD_TYPE}_calls.sorted.bam"

if [[ ! -f "${BAM_FILE}" ]]; then
    echo "  运行 Dorado basecaller..."

    if [[ "${MOD_TYPE}" == "5mC" ]]; then
        MOD_BASE_FLAG="5mCG"
    elif [[ "${MOD_TYPE}" == "6mA" ]]; then
        MOD_BASE_FLAG="6mA"
    fi

    # 请确保 dorado 已安装且在 PATH 中
    # 如果没有安装，请参考: https://github.com/nanoporetech/dorado
    #
    # dorado basecaller sup "${POD5_DIR}" \
    #   --modified-bases "${MOD_BASE_FLAG}" \
    #   --reference "${REFERENCE}" \
    #   | samtools sort -o "${BAM_FILE}"
    # samtools index "${BAM_FILE}"

    echo "  [注意] 请手动运行以下命令 (取消注释上方代码):"
    echo "    dorado basecaller sup ${POD5_DIR} --modified-bases ${MOD_BASE_FLAG} --reference ${REFERENCE} | samtools sort -o ${BAM_FILE}"
    echo "    samtools index ${BAM_FILE}"
    echo ""
    echo "  或者，如果你已有 BAM 文件，请将其复制/链接到: ${BAM_FILE}"
    echo "  跳过后续步骤..."
    # exit 0  # 取消注释此行如果需要等待 BAM 生成
else
    echo "  找到已有 BAM 文件: ${BAM_FILE}"
fi

# ──── Step 2: 准备训练数据 ────────────────────────────────────────────────────

echo ""
echo ">>> Step 2: 准备甲基化训练数据"

H5_FILE="${METH_DATA_DIR}/${MOD_TYPE}_train.h5"

if [[ ! -f "${H5_FILE}" ]] && [[ -f "${BAM_FILE}" ]]; then
    python "${CODE_DIR}/prepare_methylation_data.py" \
        --pod5_path "${POD5_DIR}" \
        --bam_path "${BAM_FILE}" \
        --mod_code "${MOD_CODE}" \
        --output "${H5_FILE}" \
        --segment_length 10240 \
        --context_bases 10 \
        --min_prob 0.8

    echo "  训练数据已保存到: ${H5_FILE}"
else
    if [[ -f "${H5_FILE}" ]]; then
        echo "  找到已有训练数据: ${H5_FILE}"
    else
        echo "  [跳过] BAM 文件不存在，无法生成训练数据"
        echo "  请先完成 Step 1"
    fi
fi

# ──── Step 3: 训练甲基化检测模型 ──────────────────────────────────────────────

echo ""
echo ">>> Step 3: 训练甲基化检测模型"

TRAIN_ARGS=(
    "${CODE_DIR}/train_methylation.py"
    --data_path "${H5_FILE}"
    --pretrained_checkpoint "${PRETRAINED_CKPT}"
    --output_dir "${OUTPUT_DIR}"
    --mod_type "${MOD_TYPE}"
    --hidden_dim 256
    --wavelet "db4"
    --learnable_wavelet
    --cnn_compression_ratio 4
    --head_type "attention"
    --epochs "${EPOCHS}"
    --batch_size "${BATCH_SIZE}"
    --lr "${LR}"
    --weight_decay 0.01
    --warmup_epochs 5
    --num_workers 4
    --seed 42
)

if [[ "${FREEZE_ENCODER}" == "1" ]]; then
    TRAIN_ARGS+=( --freeze_encoder )
    echo "  微调策略: Linear Probing (冻结 encoder)"
else
    echo "  微调策略: Full Fine-tuning"
fi

if [[ -f "${H5_FILE}" ]]; then
    python "${TRAIN_ARGS[@]}"
    echo "  训练完成！模型保存在: ${OUTPUT_DIR}/checkpoints/"
else
    echo "  [跳过] 训练数据不存在"
fi

# ──── Step 4: 预测 ────────────────────────────────────────────────────────────

echo ""
echo ">>> Step 4: 进行甲基化预测"

MODEL_CKPT="${OUTPUT_DIR}/checkpoints/best_model.pth"
PRED_OUTPUT="${OUTPUT_DIR}/predictions.tsv"

if [[ -f "${MODEL_CKPT}" ]]; then
    # 对训练数据进行验证预测
    python "${CODE_DIR}/predict_methylation.py" \
        --model_checkpoint "${MODEL_CKPT}" \
        --pretrained_checkpoint "${PRETRAINED_CKPT}" \
        --input_h5 "${H5_FILE}" \
        --output "${PRED_OUTPUT}" \
        --hidden_dim 256 \
        --wavelet "db4" \
        --learnable_wavelet \
        --cnn_compression_ratio 4 \
        --head_type "attention" \
        --batch_size 128

    echo "  预测结果已保存到: ${PRED_OUTPUT}"

    # 也可以直接对 POD5 进行全序列扫描
    # python "${CODE_DIR}/predict_methylation.py" \
    #     --model_checkpoint "${MODEL_CKPT}" \
    #     --pretrained_checkpoint "${PRETRAINED_CKPT}" \
    #     --input_pod5 "${POD5_DIR}" \
    #     --output "${OUTPUT_DIR}/pod5_scan_predictions.tsv" \
    #     --max_reads 100 \
    #     --batch_size 128
else
    echo "  [跳过] 模型 checkpoint 不存在"
fi

echo ""
echo "============================================================"
echo " 实验完成！"
echo "============================================================"
echo " 输出文件:"
echo "   训练数据:  ${H5_FILE}"
echo "   模型:      ${OUTPUT_DIR}/checkpoints/"
echo "   预测结果:  ${PRED_OUTPUT}"
echo "   训练历史:  ${OUTPUT_DIR}/training_history.json"
echo "============================================================"
