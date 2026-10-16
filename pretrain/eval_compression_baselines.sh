#!/bin/bash
# eval_compression_baselines.sh
# =============================
# 运行统一压缩基线对比实验
# 对比: WaveSem vs GZIP vs Zstd vs VBZ vs SLOW5 vs ex-zd vs ENANO
#
# 使用方法:
#   chmod +x eval_compression_baselines.sh
#   bash eval_compression_baselines.sh              # 完整模式 (包含 VBZ proxy 实测)
#   bash eval_compression_baselines.sh --fast        # 快速模式 (仅文献数据)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ===== 配置 =====
COMPRESS_STATS="/owenbhe/buddy1/lqszchen/ONT/ont_paper/wave_result/compress_stats.json"
EVAL_RESULTS="/owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_results/results_20251102_181341/compression_results.json"
DATA_PATH="/owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain/raw_data/temp_signal_segments_10240"
OUTPUT_DIR="/owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain/outputs/compression_baselines"

# 解析参数
SKIP_VBZ_PROXY=""
SKIP_PLOTS=""
for arg in "$@"; do
    case $arg in
        --fast)
            SKIP_VBZ_PROXY="--skip_vbz_proxy"
            echo "[MODE] Fast mode: skip VBZ proxy measurement"
            ;;
        --no-plots)
            SKIP_PLOTS="--skip_plots"
            echo "[MODE] Skip plot generation"
            ;;
    esac
done

echo "============================================================"
echo "  Compression Baselines Evaluation"
echo "  Output: ${OUTPUT_DIR}"
echo "============================================================"
echo ""

# 运行对比脚本
python eval_compression_baselines.py \
    --compress_stats "${COMPRESS_STATS}" \
    --eval_results "${EVAL_RESULTS}" \
    --data_path "${DATA_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    ${SKIP_VBZ_PROXY} \
    ${SKIP_PLOTS}

echo ""
echo "=========================================="
echo " 完成! 输出文件列表:"
echo "=========================================="
ls -lh "${OUTPUT_DIR}/"

echo ""
echo "下一步:"
echo "  1. 将 ${OUTPUT_DIR}/compression_baselines_table.tex 插入论文"
echo "  2. 将 ${OUTPUT_DIR}/exzd_bib_entry.bib 内容添加到 example_paper.bib"
echo "  3. 使用 ${OUTPUT_DIR}/compression_baselines_summary.md 作为审稿人回复素材"
echo "  4. 将图片复制到论文 figures 目录"
