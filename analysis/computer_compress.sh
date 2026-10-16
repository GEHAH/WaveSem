#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1,2,3}

accelerate launch "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/computer_compress.py" \
  --data_path /owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain/raw_data/temp_signal_segments_10240 \
  --checkpoint_path /owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain/outputs/wave_vqmodel_small_cnn/checkpoints/best_model.pth \
  --segment_length 10240 \
  --batch_size 128 \
  --wavelet db4 \
  --normalize none \
  --output_json compress_stats.json