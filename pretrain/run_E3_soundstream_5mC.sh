#!/usr/bin/env bash
# ============================================================================
# E3: SoundStream 5mC frozen-encoder probing (回应 Reviewer1 R1.3)
# 在 **GPU 机器** 上运行。代码/数据均在共享路径, 无需额外拷贝。
#
# 用法:
#   bash run_E3_soundstream_5mC.sh                 # 只跑 soundstream
#   MODELS="soundstream,wavesem_dual" bash run_E3_soundstream_5mC.sh   # 顺带复现对照
#   CUDA_VISIBLE_DEVICES=0 bash run_E3_soundstream_5mC.sh              # 指定卡
# ============================================================================
set -euo pipefail

# ── 配置 ─────────────────────────────────────────────────────────────────
# PY=${PY:-/data/home/lqszchen/miniconda3/envs/llm3.10/bin/python}   # 若 GPU 机器路径不同, 用 PY=... 覆盖
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS=${MODELS:-soundstream}
EPOCHS=${EPOCHS:-30}
BATCH_SIZE=${BATCH_SIZE:-64}
CR=${CR:-64}
export HDF5_USE_FILE_LOCKING=${HDF5_USE_FILE_LOCKING:-FALSE}

cd "$ROOT"

echo "══════════════ 环境自检 ══════════════"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
else:
    raise SystemExit("[FATAL] 该环境的 torch 看不到 GPU。请在此环境 pip 安装 CUDA 版 torch, 或换一个带 CUDA torch 的环境 (用 PY=... 指定)。")
PY

echo ""
echo "══════════════ 运行 E3: models=${MODELS} ══════════════"
LOG="outputs/methylation_5mC_comparison/E3_run_$(date +%Y%m%d_%H%M%S).log"
mkdir -p outputs/methylation_5mC_comparison
python run_5mC_baseline_comparison.py \
    --models "${MODELS}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --compression_ratio "${CR}" 2>&1 | tee "$LOG"

echo ""
echo "══════════════ 完成 ══════════════"
echo "结果: outputs/methylation_5mC_comparison/soundstream/results.json"
echo "汇总: outputs/methylation_5mC_comparison/comparison_summary.json"
echo "日志: $LOG"
