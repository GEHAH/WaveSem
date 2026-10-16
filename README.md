# WaveSem 训练与主实验代码

论文目录 `NeurIPS_WaveSem 2` 对应的模型训练和主要实验分析程序。这里只收源码，不含原始信号、checkpoint 和已有实验结果。

数据和权重仍使用原来的路径：

- 预训练语料：`/owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain/raw_data/temp_signal_segments_10240`
- 模型与实验输出：`/owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain/outputs`
- 5mC 数据：`/owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain/methylation_data`
- basecalling 标注数据：`/owenbhe/buddy1/lqszchen/ONT/ont_basecalling/data/example_data_dna_r10.4.1_v0`

脚本里的代码入口已经改到本目录。上述数据路径保持不变，所以在这台机器上可以直接跑。

## 目录

| 目录 | 内容 |
| --- | --- |
| `pretrain/` | WaveSem、消融变体和对比基线的预训练；5mC 探测；重建消融与解耦指标 |
| `pretrain/soundstream/audiolm-pytorch-main/` | SoundStream 基线所用的 audiolm-pytorch |
| `basecalling/` | 冻结 token 上的 CRF basecalling（Bonito 风格解码器） |
| `analysis/` | 重建、压缩率、GFLOPs |

未纳入：`fairseq`、审稿补充实验 E1/E2/E5/E7–E14、原始数据和 checkpoint。

## 预训练

在 `pretrain/` 下执行。默认读上面的 H5 语料，checkpoint 写回原来的 `outputs/`。

```bash
cd pretrain
bash train_wave_vqmodel_accelerate.sh
```

同一套启动方式：

| 脚本 | 模型 |
| --- | --- |
| `train_wave_vqmodel_accelerate.sh` | WaveSem |
| `train_wavesem_stride_accelerate_{96,128,160,256}.sh` | stride 变体 |
| `train_wavesem_oDWT_accelerate.sh` | 去掉 DWT |
| `train_wavesem_ogate_accelerate.sh` | 去掉 contextual gating |
| `train_single_codebook_vqvae_accelerate.sh` | Single-codebook VQ-VAE |
| `train_R_codebook_vqvae_accelerate.sh` | RVQ-VAE |
| `train_strided_cnn_ae_accelerate.sh` | Strided CNN AE |
| `train_soundstream_accelerate.sh` | SoundStream |
| `sweep_compression_ratios_baselines_*.sh` | 基线的压缩比扫描 |

`DATA_PATH`、`BATCH_SIZE`、`NUM_PROC`、`OUT_DIR` 可以用环境变量覆盖。

## 主要实验

**重建与压缩。** `analysis/predict_signal_batch.sh` 批量重构；`analysis/computer_compress.sh` 统计压缩率；`pretrain/eval_compression_baselines.sh` 对比通用与领域压缩工具。

**Basecalling。** 在 `basecalling/` 下用已训好的 codec checkpoint 训练解码器：

```bash
cd basecalling
bash wavesem16_32.sh          # WaveSem 16x / 32x
bash singlevq16_32.sh         # Standard VQ
bash rvq16_32.sh              # RVQ
bash odwt.sh                  # No-DWT 消融
```

`wavesem96_128.sh`、`wavesem160_256.sh` 以及对应的 singlevq/rvq 脚本覆盖其余下采样率。连续波形 Bonito 基线走 `scripts/train_original_zip.py --model bonito`。

**计算量。** `analysis/compute_gflops.py` 统计 codec；`analysis/basecalling.sh` 统计 basecalling 链路。

**5mC 线性探测。**

```bash
cd pretrain
python run_ecoli_5mC_experiment.py          # WaveSem frozen encoder
python run_5mC_baseline_comparison.py       # RVQ / Single-VQ / No-DWT / Texture / SoundStream
bash run_5mC_finetune.sh                    # 解冻编码器的上界
```

`train_methylation.py` 和 `run_methylation_experiment.sh` 是探测头训练入口。`analyze_5mC_results.py`、`visualize_5mC_comparison.py`、`compare_with_existing_methods.py` 整理结果。

**消融与码本。** `eval_texture_stream_ablation.sh` 比较 full / semantic-only / texture-only。`run_E6_disentanglement.sh` 计算子带能量、谱平坦度和跨流相关。

## 依赖

预训练与分析：

```bash
pip install -r requirements.txt
```

SoundStream 通过各脚本把 `pretrain/soundstream/audiolm-pytorch-main` 放进 `PYTHONPATH`，不需要再安装一份。Basecalling 额外依赖见 `basecalling/requirements.txt`。
