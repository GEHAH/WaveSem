#!/usr/bin/env bash
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_VISIBLE_DEVICES=0 python scripts/train_original_zip.py --model 'vq_model_wavesem_oDWT' --vqvae_model_path '/owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain/outputs/wavesem_oDWT64/checkpoints/best_model.pth' --window-size 4000 --num-epochs 60 --decoder_stride [2,2,2] --output-dir "$(pwd)/results/wave_64_odwt"
