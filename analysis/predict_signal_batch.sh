#!/usr/bin/env bash
python "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/predict_signal_batch.py" \
  --data /owenbhe/buddy1/lqszchen/ONT/ont_basecalling/data/example_data_dna_r10.4.1_v0/validation/chunks.npy \
  --checkpoint /owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain/outputs/wave_vqmodel_small_cnn/checkpoints/best_model.pth \
  --batch-size 64 \
  --device cuda \
  --compression-ratio 4 \
  --wavelet db4 \
  --learnable \
  --out /owenbhe/buddy1/lqszchen/ONT/ont_paper/wave_result/data/val/recon_results_val.npz