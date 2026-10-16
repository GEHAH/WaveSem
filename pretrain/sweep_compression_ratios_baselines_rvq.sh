#!/usr/bin/env bash

set -euo pipefail
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# 顺序跑四个压缩比（两模型各一遍）：
#   - Strided CNN AE
#   - Single-codebook VQ-VAE (no dual-stream)
#
# 用法：
#   bash /owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain/sweep_compression_ratios_baselines.sh
#
# 可通过环境变量覆盖（与各自 train_*.sh 一致）：
#   export DATA_PATH=...
#   export NUM_PROC=...
#   export BATCH_SIZE=...
#   export CUDA_VISIBLE_DEVICES=...

ROOT_DIR="/owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain"
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RATIOS_STR="${RATIOS:-"16 64 128 32"}"
# 容错：允许用户写成 "[16 32 64 128]" 或 "16,32,64,128"
RATIOS_STR="${RATIOS_STR//[/}"
RATIOS_STR="${RATIOS_STR//]/}"
RATIOS_STR="${RATIOS_STR//,/ }"
read -r -a RATIOS_ARR <<< "${RATIOS_STR}"
if [[ ${#RATIOS_ARR[@]} -eq 0 ]]; then
  echo "Error: empty RATIOS. Example: export RATIOS='16 32 64 128'"
  exit 1
fi

echo "Sweep compression ratios: ${RATIOS_ARR[*]}"
echo "Start time: $(date)"

for CR in "${RATIOS_ARR[@]}"; do
  echo
  echo "=============================="
  echo "Compression ratio = ${CR}"
  echo "=============================="

  # AE
  (
    export COMPRESSION_RATIO="${CR}"
    # Optional: separate ports to be extra safe even if user background-runs
    export MAIN_PORT=${MAIN_PORT_AE_BASE:-29511}
    bash "${CODE_DIR}/train_R_codebook_vqvae_accelerate.sh"
  )

done

echo
echo "Done. End time: $(date)"


