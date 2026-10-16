# WaveSem: Frequency-Adaptive Tokenization for Disentangling Semantics and Noise in Genomics

Official code for training WaveSem and reproducing the main experiments: signal reconstruction, downstream basecalling, 5mC methylation probing, ablations, and compression analysis.

WaveSem is a frequency-adaptive neural codec for nanopore sequencing signals. A discrete wavelet transform splits each latent sequence into a low-frequency semantic core and high-frequency texture residuals. The two streams are quantized by separate residual vector quantizers, so codebook capacity is spent on biological structure rather than sensor noise.

## Abstract

High-throughput scientific sensing, exemplified by genomics, faces a representation bottleneck: discrete biological events are often obscured by continuous, high-entropy stochastic noise. Standard neural codecs allocate their information budget to those fluctuations instead of to low-entropy structural motifs. WaveSem imposes a wavelet inductive bias on discrete representation learning and explicitly disentangles each signal into a structured semantic core and a residual texture component. On large-scale nanopore data, WaveSem achieves a 14× storage reduction while retaining competitive basecalling accuracy. Linear probing of the frozen encoder reaches an AUROC of 0.928 for 5mC methylation detection, showing that the learned tokens preserve information beyond nucleotide identity.

## Highlights

| Setting | Result |
| --- | --- |
| Storage vs. int16 raw signal | **14.2×** reduction (13.4 GB for 10M segments of length 10,240) |
| Reconstruction at 64× | MSE **0.0757**, STFT loss **0.8697** |
| Basecalling from discrete tokens | **96.88%** at 32×, **95.61%** at 64× (Bonito raw-signal baseline: 98.12%) |
| 5mC CpG, frozen encoder, linear probe | ACC 0.852, F1 0.851, AUROC **0.928**, AUPRC **0.936** |

All neural codecs in the comparison are trained on the same nanopore corpus. The 5mC probe freezes the encoder and trains only the classification head.

## Repository layout

```
pretrain/     WaveSem, ablations, and codec baselines; 5mC probing
basecalling/  CRF decoder trained on frozen discrete tokens
analysis/     reconstruction, compression ratio, and GFLOPs
```

`pretrain/soundstream/audiolm-pytorch-main/` vendors the SoundStream baseline. `basecalling/` adapts the public basecalling-architectures codebase.

This release contains source code only. Raw signals, pretrained checkpoints, and result files are not included.

## Setup

```bash
pip install -r requirements.txt
```

Basecalling uses an additional stack listed in `basecalling/requirements.txt` (`fast_ctc_decode`, `parasail`, and related packages). SoundStream is imported from the vendored tree; the training and evaluation scripts add that path automatically.

Training uses Hugging Face Accelerate on one or more GPUs. The paper models were trained with AdamW, a global batch size of 2,048, and a cosine schedule that warms up to `1e-4`.

## Data

Pretraining expects a directory of HDF5 files. Each file stores `signal_segments` with shape `[N, 10240]`. Windows are cut from HG002 nanopore current with a 1,024-sample overlap.

Downstream basecalling uses paired signal chunks and reference sequences in the Bonito / basecalling-architectures format. 5mC probing uses the ONT *E. coli* CpG benchmark (M.SssI-treated positive reads and whole-genome-amplified negative reads), MAD-normalized and saved as HDF5.

Point the launch scripts at those directories before training:

```bash
export DATA_PATH=/path/to/signal_segments_10240
export OUT_DIR=/path/to/outputs/wavesem
```

Checkpoint paths inside `basecalling/*.sh` and the analysis scripts should be edited to the WaveSem checkpoints produced above.

## Training

From `pretrain/`:

```bash
bash train_wave_vqmodel_accelerate.sh
```

| Script | Model |
| --- | --- |
| `train_wave_vqmodel_accelerate.sh` | WaveSem |
| `train_wavesem_stride_accelerate_{96,128,160,256}.sh` | stride variants |
| `train_wavesem_oDWT_accelerate.sh` | no wavelet decomposition |
| `train_wavesem_ogate_accelerate.sh` | no contextual gating |
| `train_single_codebook_vqvae_accelerate.sh` | single-codebook VQ-VAE |
| `train_R_codebook_vqvae_accelerate.sh` | residual VQ-VAE |
| `train_strided_cnn_ae_accelerate.sh` | strided CNN autoencoder |
| `train_soundstream_accelerate.sh` | SoundStream, reconstruction only |
| `sweep_compression_ratios_baselines_*.sh` | baseline compression-ratio sweep |

`DATA_PATH`, `BATCH_SIZE`, `NUM_PROC`, and `OUT_DIR` override the defaults in each script.

## Experiments

**Reconstruction and compression.** `analysis/predict_signal_batch.sh` reconstructs held-out windows. `analysis/computer_compress.sh` reports the token bitrate. `pretrain/eval_compression_baselines.sh` compares WaveSem with generic and nanopore codecs.

**Basecalling.** From `basecalling/`, train a CRF decoder on frozen tokens:

```bash
bash wavesem16_32.sh     # WaveSem, 16× and 32×
bash singlevq16_32.sh    # single-codebook VQ
bash rvq16_32.sh         # residual VQ
bash odwt.sh             # WaveSem without DWT
```

`wavesem96_128.sh`, `wavesem160_256.sh`, and the matching `singlevq*` / `rvq*` scripts cover the remaining ratios. The continuous-waveform Bonito baseline is `scripts/train_original_zip.py --model bonito`.

**Compute.** `analysis/compute_gflops.py` measures the codecs. `analysis/basecalling.sh` measures the basecalling stack.

**5mC probing.** From `pretrain/`:

```bash
python run_ecoli_5mC_experiment.py     # frozen WaveSem encoder
python run_5mC_baseline_comparison.py  # RVQ, single-VQ, no-DWT, texture, SoundStream
bash run_5mC_finetune.sh               # encoder unfrozen
```

`train_methylation.py` trains the probe. `analyze_5mC_results.py` and `visualize_5mC_comparison.py` summarize the metrics.

**Ablation and codebook structure.** `eval_texture_stream_ablation.sh` compares the full model, the semantic stream, and the texture stream. `run_E6_disentanglement.sh` reports sub-band energy, spectral flatness, and cross-stream dependence.

## Citation

```bibtex
@misc{wavesem2026,
  title = {WaveSem: Frequency-Adaptive Tokenization for Disentangling Semantics and Noise in Genomics},
  year  = {2026},
  url   = {https://github.com/GEHAH/WaveSem}
}
```

## Acknowledgements

The SoundStream baseline is taken from [audiolm-pytorch](https://github.com/lucidrains/audiolm-pytorch). The basecalling decoder builds on [basecalling_architectures](https://github.com/marcpaga/basecalling_architectures).
