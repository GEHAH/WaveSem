# Ceph mount command omitted: it contained a storage credential.
cd /owenbhe/buddy1/lqszchen/ONT/basecalling_architectures

conda activate env-3.10
pip install h5py vector_quantize_pytorch PyWavelets
pip install ont-koi
pip install scipy pandas einops
pip install fast_ctc_decode
pip install tqdm ont_fast5_api parasail
pip install vector_quantize_pytorch

pip -vvv install --upgrade --force-reinstall cffi


CUDA_VISIBLE_DEVICES=0 python scripts/train_original_zip.py --model 'vq_model' --window-size 4000 --num-epochs 60 --vqvae_stride [2,2,2,2,2] --decoder_stride [2,2] --output-dir '/owenbhe/buddy1/lqszchen/ONT/basecalling_architectures/results64_lstm'
CUDA_VISIBLE_DEVICES=0 python scripts/train_original_zip.py --model 'vq_model' --window-size 4000 --num-epochs 60 --vqvae_stride [2,3,2,3,2] --decoder_stride [3,3] --output-dir '/owenbhe/buddy1/lqszchen/ONT/basecalling_architectures/results144_lstm'

CUDA_VISIBLE_DEVICES=1 python scripts/train_original_zip.py --model 'vq_model' --window-size 4000 --num-epochs 60 --vqvae_stride [2,2,2,5,2] --decoder_stride [2,5] --output-dir '/owenbhe/buddy1/lqszchen/ONT/basecalling_architectures/results160_lstm'

CUDA_VISIBLE_DEVICES=2 python scripts/train_original_zip.py --model 'vq_model' --window-size 4000 --num-epochs 60 --vqvae_stride [2,2,1,11,2] --decoder_stride [1,11] --output-dir '/owenbhe/buddy1/lqszchen/ONT/basecalling_architectures/results176_lstm'

CUDA_VISIBLE_DEVICES=3 python scripts/train_original_zip.py --model 'vq_model' --window-size 4000 --num-epochs 60 --vqvae_stride [2,2,3,4,2] --decoder_stride [3,2,2] --output-dir '/owenbhe/buddy1/lqszchen/ONT/basecalling_architectures/results192_lstm'

CUDA_VISIBLE_DEVICES=4 python scripts/train_original_zip.py --model 'vq_model' --window-size 4000 --num-epochs 60 --vqvae_stride [2,2,1,13,2] --decoder_stride [1,13] --output-dir '/owenbhe/buddy1/lqszchen/ONT/basecalling_architectures/results208_lstm'

CUDA_VISIBLE_DEVICES=5 python scripts/train_original_zip.py --model 'vq_model' --window-size 4000 --num-epochs 60 --vqvae_stride [2,2,2,7,2] --decoder_stride [2,7] --output-dir '/owenbhe/buddy1/lqszchen/ONT/basecalling_architectures/results224_lstm'

CUDA_VISIBLE_DEVICES=6 python scripts/train_original_zip.py --model 'vq_model' --window-size 4000 --num-epochs 60 --vqvae_stride [2,2,3,5,2] --decoder_stride [3,5] --output-dir '/owenbhe/buddy1/lqszchen/ONT/basecalling_architectures/results240_lstm'

CUDA_VISIBLE_DEVICES=7 python scripts/train_original_zip.py --model 'vq_model' --window-size 4000 --num-epochs 60 --vqvae_stride [2,2,4,4,2] --decoder_stride [2,2,4] --output-dir '/owenbhe/buddy1/lqszchen/ONT/basecalling_architectures/results256_lstm'

CUDA_VISIBLE_DEVICES=1 python scripts/train_original_zip.py --model 'vq_model' --window-size 4000 --num-epochs 60 --decoder_stride [2,2,2] --output-dir '/owenbhe/buddy1/lqszchen/ONT/basecalling_architectures/results/wave64_rec'

CUDA_VISIBLE_DEVICES=0 python scripts/train_original_zip.py --model 'basecalling' --window-size 4000 --num-epochs 60 --decoder_stride [2,2,2] --batch-size 256  --output-dir '/owenbhe/buddy1/lqszchen/ONT/basecalling_architectures/results/basecalling'

CUDA_VISIBLE_DEVICES=1 python scripts/train_original_zip.py --model 'vq_model_rec_basecalling' --vqvae_model_path '/owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain/outputs/wave_vqmodel_small_cnn_16/checkpoints/best_model.pth' --window-size 4000 --num-epochs 60 --decoder_stride [2,2,2] --output-dir '/owenbhe/buddy1/lqszchen/ONT/basecalling_architectures/results/rec_16'

CUDA_VISIBLE_DEVICES=2 python scripts/train_original_zip.py --model 'vq_model_rec_basecalling' --vqvae_model_path '/owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain/outputs/wave_vqmodel_small_cnn_32/checkpoints/best_model.pth' --window-size 4000 --num-epochs 60 --decoder_stride [2,2,2] --output-dir '/owenbhe/buddy1/lqszchen/ONT/basecalling_architectures/results/rec_32'

CUDA_VISIBLE_DEVICES=3 python scripts/train_original_zip.py --model 'VQVAEModel_wavesem_all_basecalling' --vqvae_model_path '/owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain/outputs/wave_vqmodel_small_cnn/checkpoints/best_model.pth' --window-size 4000 --num-epochs 60 --decoder_stride [2,2,2] --output-dir '/owenbhe/buddy1/lqszchen/ONT/basecalling_architectures/results/rec_64_low'

CUDA_VISIBLE_DEVICES=4 python scripts/train_original_zip.py --model 'vq_model' --vqvae_model_path '/owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain/outputs/wave_vqmodel_small_cnn_32/checkpoints/best_model.pth' --window-size 4000 --num-epochs 60 --decoder_stride [2,2] --output-dir '/owenbhe/buddy1/lqszchen/ONT/basecalling_architectures/results/wave_32_finetuning'

CUDA_VISIBLE_DEVICES=3 python scripts/train_original_zip.py --model 'vq_model' --vqvae_model_path '/owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain/outputs/wave_vqmodel_Equal/checkpoints/best_model.pth' --window-size 4000 --num-epochs 60 --decoder_stride [2,2,2] --output-dir '/owenbhe/buddy1/lqszchen/ONT/basecalling_architectures/results/wave_64_vq2'


CUDA_VISIBLE_DEVICES=1 python scripts/train_original_zip.py --model 'vq_model_wavesem_ogate' --vqvae_model_path '/owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain/outputs/wavesem_ogate_ogate/checkpoints/best_model.pth' --window-size 4000 --num-epochs 60 --decoder_stride [2,2,2] --output-dir '/owenbhe/buddy1/lqszchen/ONT/basecalling_architectures/results/wave_64_ogate'

CUDA_VISIBLE_DEVICES=6 python scripts/train_original_zip.py --model 'vq_model_wavesem_oDWT' --vqvae_model_path '/owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain/outputs/wavesem_oDWT64/checkpoints/best_model.pth' --window-size 4000 --num-epochs 60 --decoder_stride [2,2,2] --output-dir '/owenbhe/buddy1/lqszchen/ONT/basecalling_architectures/results/wave_64_odwt'

CUDA_VISIBLE_DEVICES=3 python scripts/train_original_zip.py --model 'vq_model_wavesem_all' --vqvae_model_path '/owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain/outputs/wave_vqmodel_small_cnn/checkpoints/best_model.pth' --window-size 4000 --num-epochs 60 --decoder_stride [2,2,2] --output-dir '/owenbhe/buddy1/lqszchen/ONT/basecalling_architectures/results/rec_64_all'

CUDA_VISIBLE_DEVICES=2 python scripts/train_original_zip.py --model 'VQVAEModel_wavesem_high_basecalling' --vqvae_model_path '/owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain/outputs/wave_vqmodel_small_cnn/checkpoints/best_model.pth' --window-size 4000 --num-epochs 60 --decoder_stride [2,2,2] --output-dir '/owenbhe/buddy1/lqszchen/ONT/basecalling_architectures/results/rec_64_high/'


CUDA_VISIBLE_DEVICES=0 python scripts/train_original_zip.py --model 'vq_model_wavesem_stride' --vqvae_model_path '/owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain/outputs/wavesem_pretrain_96_stride/checkpoints/best_model.pth' --window-size 4000 --num-epochs 60 --batch-size 256 --decoder_stride [2,3,2] --strides "[2,3]" --output-dir '/owenbhe/buddy1/lqszchen/ONT/basecalling_architectures/results/wave_96'

CUDA_VISIBLE_DEVICES=1 python scripts/train_original_zip.py --model 'vq_model_wavesem_stride' --vqvae_model_path '/owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain/outputs/wavesem_pretrain_128_stride/checkpoints/best_model.pth' --window-size 4000 --num-epochs 60 --batch-size 128 --decoder_stride [2,4,2] --strides "[2,4]" --output-dir '/owenbhe/buddy1/lqszchen/ONT/basecalling_architectures/results/wave_128'

CUDA_VISIBLE_DEVICES=2 python scripts/train_original_zip.py --model 'vq_model_wavesem_stride' --vqvae_model_path '/owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain/outputs/wavesem_pretrain_256_stride/checkpoints/best_model.pth' --window-size 4000 --num-epochs 60 --batch-size 256 --decoder_stride [2,4,4] --strides "[2,4,2]" --output-dir '/owenbhe/buddy1/lqszchen/ONT/basecalling_architectures/results/wave_256'

CUDA_VISIBLE_DEVICES=0 python scripts/train_original_zip.py --model 'single_codebook_vqvae_stride_basecalling' --vqvae_model_path '//owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain/outputs/single_codebook_vqvae_cr64/checkpoints/best_model.pth' --window-size 4000 --num-epochs 60 --batch-size 256 --decoder_stride [2,2,2] --strides [2,2,2,2,2,2] --output-dir '/owenbhe/buddy1/lqszchen/ONT/basecalling_architectures/results/singlevq_64'

CUDA_VISIBLE_DEVICES=1 python scripts/train_original_zip.py --model 'single_codebook_vqvae_stride_basecalling' --vqvae_model_path '/owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain/outputs/single_codebook_vqvae_cr32/checkpoints/best_model.pth' --window-size 4000 --num-epochs 60 --batch-size 256 --decoder_stride [2,2,2] --strides [2,2,2,4,2] --cnn_compression_ratio 64 --output-dir '/owenbhe/buddy1/lqszchen/ONT/basecalling_architectures/results/singlevq_64'

CUDA_VISIBLE_DEVICES=5 python scripts/train_original_zip.py --model 'single_codebook_vqvae_stride_basecalling' --vqvae_model_path '/owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain/outputs/single_codebook_vqvae_cr16/checkpoints/best_model.pth' --window-size 4000 --num-epochs 60 --batch-size 256 --decoder_stride [2] --strides [2,2,2,2] --output-dir '/owenbhe/buddy1/lqszchen/ONT/basecalling_architectures/results/singlevq_16'

CUDA_VISIBLE_DEVICES=3 python scripts/train_original_zip.py --model 'single_codebook_vqvae_stride_basecalling' --vqvae_model_path '/owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain/outputs/single_codebook_vqvae_cr32/checkpoints/best_model.pth' --window-size 4000 --num-epochs 60 --batch-size 256 --decoder_stride [2,4,2] --strides [2,2,4,4,2] --output-dir '/owenbhe/buddy1/lqszchen/ONT/basecalling_architectures/results/singlevq_128'

CUDA_VISIBLE_DEVICES=3 python scripts/train_original_zip.py --model 'single_codebook_vqvae_stride_basecalling' --vqvae_model_path '/owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain/outputs/single_codebook_vqvae_cr32/checkpoints/best_model.pth' --window-size 4000 --num-epochs 60 --batch-size 256 --decoder_stride [2,4,4] --strides [2,4,4,4,2] --output-dir '/owenbhe/buddy1/lqszchen/ONT/basecalling_architectures/results/singlevq_256'

CUDA_VISIBLE_DEVICES=2 python scripts/train_original_zip.py --model 'single_codebook_vqvae_stride_basecalling' --vqvae_model_path '/owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain/outputs/single_codebook_vqvae_cr32/checkpoints/best_model.pth' --window-size 4000 --num-epochs 60 --batch-size 256 --decoder_stride [2,2,5] --strides [2,2,4,5,2] --output-dir '/owenbhe/buddy1/lqszchen/ONT/basecalling_architectures/results/singlevq_160'

CUDA_VISIBLE_DEVICES=4 python scripts/train_original_zip.py --model 'single_codebook_vqvae_stride_basecalling' --vqvae_model_path '/owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain/outputs/single_codebook_vqvae_cr32/checkpoints/best_model.pth' --window-size 4000 --num-epochs 60 --batch-size 256 --decoder_stride [2,2] --strides [2,2,2,2,2] --output-dir '/owenbhe/buddy1/lqszchen/ONT/basecalling_architectures/results/singlevq_32'



#rvqvae
CUDA_VISIBLE_DEVICES=1 python scripts/train_original_zip.py --model 'r_codebook_vqvae_stride_basecalling' --vqvae_model_path '/owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain/outputs/r_codebook_vqvae_cr32/checkpoints/best_model.pth' --window-size 4000 --num-epochs 60 --batch-size 256 --decoder_stride [2,2,2] --strides [2,2,2,4,2] --cnn_compression_ratio 32 --output-dir '/owenbhe/buddy1/lqszchen/ONT/basecalling_architectures/results/rvq_64'

CUDA_VISIBLE_DEVICES=7 python scripts/train_original_zip.py --model 'r_codebook_vqvae_stride_basecalling' --vqvae_model_path '/owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain/outputs/r_codebook_vqvae_cr16/checkpoints/best_model.pth' --window-size 4000 --num-epochs 60 --batch-size 256 --decoder_stride [2] --strides [2,2,2,2] --output-dir '/owenbhe/buddy1/lqszchen/ONT/basecalling_architectures/results/rvq_16'

CUDA_VISIBLE_DEVICES=0 python scripts/train_original_zip.py --model 'r_codebook_vqvae_stride_basecalling' --vqvae_model_path '/owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain/outputs/r_codebook_vqvae_cr32/checkpoints/best_model.pth' --window-size 4000 --num-epochs 60 --batch-size 256 --decoder_stride [2,2] --strides [2,2,2,2,2] --output-dir '/owenbhe/buddy1/lqszchen/ONT/basecalling_architectures/results/rvq_32'

conda activate env-3.11.9
CUDA_VISIBLE_DEVICES=0 python scripts/train_original_zip.py --model 'r_codebook_vqvae_stride_basecalling' --vqvae_model_path '/owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain/outputs/r_codebook_vqvae_cr32/checkpoints/best_model.pth' --window-size 4000 --num-epochs 60 --batch-size 256 --decoder_stride [2,2,3] --strides [2,2,4,3,2] --output-dir '/owenbhe/buddy1/lqszchen/ONT/basecalling_architectures/results/rvq_96'

conda activate env-3.11.9
CUDA_VISIBLE_DEVICES=1 python scripts/train_original_zip.py --model 'r_codebook_vqvae_stride_basecalling' --vqvae_model_path '/owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain/outputs/r_codebook_vqvae_cr32/checkpoints/best_model.pth' --window-size 4000 --num-epochs 60 --batch-size 256 --decoder_stride [2,2,4] --strides [2,2,4,4,2] --output-dir '/owenbhe/buddy1/lqszchen/ONT/basecalling_architectures/results/rvq_128'

conda activate env-3.11.9
CUDA_VISIBLE_DEVICES=3 python scripts/train_original_zip.py --model 'r_codebook_vqvae_stride_basecalling' --vqvae_model_path '/owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain/outputs/r_codebook_vqvae_cr32/checkpoints/best_model.pth' --window-size 4000 --num-epochs 60 --batch-size 256 --decoder_stride [2,4,4] --strides [2,4,4,4,2] --output-dir '/owenbhe/buddy1/lqszchen/ONT/basecalling_architectures/results/rvq_256'

CUDA_VISIBLE_DEVICES=1,3 python /owenbhe/buddy1/lqszchen/gpu/calass/mult_gpu.py
CUDA_VISIBLE_DEVICES=1 python /owenbhe/buddy1/lqszchen/gpu/calass/class.py
yum install screen
screen -R gpu
conda activate env-3.11.9

export http_proxy="http://star-proxy.oa.com:3128"
export https_proxy="http://star-proxy.oa.com:3128"
conda create -n env-3.10 python=3.10

conda activate env-3.10
cd /owenbhe/buddy1/lqszchen/ONT/basecalling_architectures


CUDA_VISIBLE_DEVICES=1 python scripts/train_original_zip.py --model 'vq_model' --vqvae_model_path '/owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain/outputs/wave_vqmodel_small_cnn/checkpoints/best_model.pth' --cnn_compression_ratio 4 --window-size 4000 --num-epochs 60 --decoder_stride [2,2,2] --batch_size 128 --output-dir '/owenbhe/buddy1/lqszchen/ONT/basecalling_architectures/results/wave64'

python /owenbhe/buddy1/lqszchen/ONT/basecalling_architectures/scripts/train_original_zip.py \
  --model soundstream_basecalling \
  --vqvae_model_path /owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain/outputs/soundstream/checkpoints/best_model.pth \
  --decoder_stride "[2,2,2]" \
  --ss_channels 32 \
  --ss_strides "[2,4,4,2]" \
  --ss_channel_mults "[2,4,8,16]" \
  --ss_codebook_dim 512 \
  --ss_codebook_size 256 \
  --ss_rq_num_quantizers 8 \
  --ss_rq_groups 1 \
  --ss_target_sample_hz 4000 \
  --batch-size 128


  python /owenbhe/buddy1/lqszchen/ONT/basecalling_architectures/scripts/eval_wavesem_snr.py   --data-dir-val /owenbhe/buddy1/lqszchen/ONT/ont_basecalling/data/example_data_dna_r10.4.1_v0/validation   --vqvae_model_path /owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain/outputs/wave_vqmodel_small_cnn_32/checkpoints/best_model.pth   --basecalling_checkpoint /owenbhe/buddy1/lqszchen/ONT/basecalling_architectures/results/wave_32/checkpoints/best_checkpoint_184000.pt   --batch-size 128   --output-json snr_eval_32.json

python /owenbhe/buddy1/lqszchen/ONT/basecalling_architectures/scripts/eval_wavesem_snr.py   --data-dir-val /owenbhe/buddy1/lqszchen/ONT/ont_basecalling/data/example_data_dna_r10.4.1_v0/validation   --vqvae_model_path /owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain/outputs/wave_vqmodel_small_cnn/checkpoints/best_model.pth   --basecalling_checkpoint /owenbhe/buddy1/lqszchen/ONT/basecalling_architectures/results/wave64/checkpoints/best_checkpoint_233500.pt   --batch-size 128   --output-json snr_eval_64.json


python scripts/eval_wavesem_snr.py --model singlevq --strides "[2,2,2,2]" --decoder_stride "[2]" --data-dir-val /owenbhe/buddy1/lqszchen/ONT/ont_basecalling/data/example_data_dna_r10.4.1_v0/validation   --vqvae_model_path /owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain/outputs/single_codebook_vqvae_cr16/checkpoints/best_model.pth   --basecalling_checkpoint /owenbhe/buddy1/lqszchen/ONT/basecalling_architectures/results/singlevq_16/checkpoints/best_checkpoint_78500.pt   --batch-size 128   --output-json single_eval_16.json

python scripts/eval_wavesem_snr.py --model singlevq --strides "[2,2,2,2,2]" --decoder_stride "[2,2]" --data-dir-val /owenbhe/buddy1/lqszchen/ONT/ont_basecalling/data/example_data_dna_r10.4.1_v0/validation   --vqvae_model_path /owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain/outputs/single_codebook_vqvae_cr32/checkpoints/best_model.pth   --basecalling_checkpoint /owenbhe/buddy1/lqszchen/ONT/basecalling_architectures/results/singlevq_32/checkpoints/best_checkpoint_78500.pt   --batch-size 128   --output-json single_eval_32.json

python scripts/eval_wavesem_snr.py --model basecalling  --data-dir-val /owenbhe/buddy1/lqszchen/ONT/ont_basecalling/data/example_data_dna_r10.4.1_v0/validation   --vqvae_model_path /owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain/outputs/single_codebook_vqvae_cr32/checkpoints/best_model.pth   --basecalling_checkpoint /owenbhe/buddy1/lqszchen/ONT/basecalling_architectures/results/basecalling/checkpoints/best_checkpoint_94000.pt   --batch-size 128   --output-json basecalling.json
