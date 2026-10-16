#!/usr/bin/env bash
python "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/compute_gflops_basecalling.py" \
  --model wavesem \
  --vqvae_model_path /owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain/outputs/wave_vqmodel_small_cnn_32/checkpoints/best_model.pth \
  --basecalling_checkpoint /owenbhe/buddy1/lqszchen/ONT/basecalling_architectures/results/wave_32/checkpoints/best_checkpoint_184000.pt \
  --segment_length 10240 \
  --batch_size 1 \
  --decoder_stride "[2,2]" \
  --prefer_thop


python "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/compute_gflops_basecalling.py" \
  --model basecalling \
  --vqvae_model_path dummy.pth \
  --basecalling_checkpoint /owenbhe/buddy1/lqszchen/ONT/basecalling_architectures/results/basecalling/checkpoints/best_checkpoint_141000.pt \
  --segment_length 10240 \
  --batch_size 1 \
  --decoder_stride "[2,2,2]" \
  --prefer_thop


python "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/compute_gflops_basecalling.py" \
  --model singlevq \
  --vqvae_model_path /owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain/outputs/single_codebook_vqvae_cr32/checkpoints/best_model.pth \
  --basecalling_checkpoint /owenbhe/buddy1/lqszchen/ONT/basecalling_architectures/results/singlevq_32/checkpoints/best_checkpoint_78500.pt \
  --strides "[2,2,2,2,2]" \
  --decoder_stride "[2,2]" \
  --prefer_thop

  python "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/compute_gflops_basecalling.py" \
  --model singlevq \
  --vqvae_model_path /owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain/outputs/single_codebook_vqvae_cr64/checkpoints/best_model.pth \
  --basecalling_checkpoint /owenbhe/buddy1/lqszchen/ONT/basecalling_architectures/results/singlevq_64/checkpoints/best_checkpoint_113500.pt \
  --strides "[2,2,2,2,2,2]" \
  --decoder_stride "[2,2,2]" \
  --prefer_thop

python "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/compute_gflops_basecalling.py" \
  --model soundstream \
  --vqvae_model_path /owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain/outputs/soundstream/checkpoints/best_model.pth \
  --basecalling_checkpoint /owenbhe/buddy1/lqszchen/ONT/basecalling_architectures/results/checkpoints/best_checkpoint_141500.pt \
  --ss_strides "2,4,4,2" \
  --decoder_stride "[2,2,2]" \
  --prefer_thop