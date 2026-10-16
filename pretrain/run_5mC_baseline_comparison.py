#!/usr/bin/env python3
"""
run_5mC_baseline_comparison.py
=================================
多模型 5mC CpG 甲基化检测对比实验。

对比方案：
  1. WaveSem (Ours)         - WaveSem 语义流 (DWT + Transformer + RVQ)
  2. WaveSem-Texture         - WaveSem 纹理流 only (DWT + 级联门控 + RVQ)
  3. WaveSem-Dual            - WaveSem 双流 (语义流 + 纹理流拼接)
  4. StridedCNNAE            - 纯 CNN AutoEncoder (无 VQ)
  5. SingleCodebookVQVAE     - 单码本 VQ-VAE
  6. RCodebookVQVAE          - 残差码本 VQ-VAE (8 quantizers)
  7. WaveSem-oDWT            - WaveSem 消融：无 DWT (channel split 替代)
  8. SoundStream             - 音频压缩模型 baseline

所有模型均使用 **预训练编码器 + 相同的注意力分类头** (Fine-tuning, 全部参数可训练)，
在同一份训练/验证数据上训练，公平对比不同预训练编码器的特征质量。

用法:
  python run_5mC_baseline_comparison.py [--epochs 30] [--batch_size 64]
"""

import os
import sys
import json
import time
import copy
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score, confusion_matrix,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_DIR = Path("/owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain")
sys.path.insert(0, str(Path(__file__).resolve().parent))


# ─── 数据集 ──────────────────────────────────────────────────────────────────

class MethylationDataset(Dataset):
    """从 HDF5 加载甲基化训练数据"""

    def __init__(self, h5_path: str):
        super().__init__()
        with h5py.File(h5_path, "r") as f:
            self.signals = f["signal_segments"][:]
            self.labels = f["labels"][:]
        self.num_positive = int(np.sum(self.labels == 1))
        self.num_negative = int(np.sum(self.labels == 0))
        logger.info(f"Loaded {len(self.signals)} samples: pos={self.num_positive}, neg={self.num_negative}")

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = torch.from_numpy(self.signals[idx].astype(np.float32)).unsqueeze(0)
        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return signal, label


# ─── 通用分类头 ──────────────────────────────────────────────────────────────

class AttentionPooling(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.Tanh(),
            nn.Linear(dim // 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        x_t = x.transpose(1, 2)  # [B, T, C]
        weights = F.softmax(self.attn(x_t), dim=1)  # [B, T, 1]
        pooled = (x_t * weights).sum(dim=1)  # [B, C]
        return pooled


class ClassificationHead(nn.Module):
    """通用的注意力池化 + MLP 分类头"""

    def __init__(self, in_channels: int = 256, hidden_dim: int = 128, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.attn_pool = AttentionPooling(in_channels)
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        pooled = self.attn_pool(feat)
        return self.classifier(pooled)


# ─── 各模型的编码器包装器 ─────────────────────────────────────────────────────

class WaveSemEncoder(nn.Module):
    """WaveSem: CNN → DWT → LowFreqStream (Transformer + RVQ) → feat_low_2 (语义流 only)"""

    def __init__(self, pretrained_model):
        super().__init__()
        self.encoder = pretrained_model.encoder
        self.multi_level_dwt = pretrained_model.multi_level_dwt
        self.low_stream = pretrained_model.low_stream
        self.out_channels = pretrained_model.hidden_dim  # 256

    def forward(self, x):
        feat_cnn = self.encoder(x)
        feat_low_1, feat_high = self.multi_level_dwt(feat_cnn)
        feat_low_2, indices, vq_loss = self.low_stream(feat_low_1)
        return feat_low_2  # [B, 256, 160]


class WaveSemTextureEncoder(nn.Module):
    """WaveSem 纹理流 only: CNN → DWT → HighFreqStream (级联门控 + RVQ) → feat_high_1"""

    def __init__(self, pretrained_model):
        super().__init__()
        self.encoder = pretrained_model.encoder
        self.multi_level_dwt = pretrained_model.multi_level_dwt
        self.high_stream = pretrained_model.high_stream
        self.out_channels = pretrained_model.hidden_dim  # 256

    def forward(self, x):
        feat_cnn = self.encoder(x)
        feat_low_1, feat_high = self.multi_level_dwt(feat_cnn)
        # feat_high: [H1, H2, H3, H4]，high_stream 输入顺序为 (h4, h3, h2, h1)
        feat_high_1, indices, vq_loss = self.high_stream(
            feat_high[3], feat_high[2], feat_high[1], feat_high[0]
        )
        return feat_high_1  # [B, 256, 160]


class WaveSemDualStreamEncoder(nn.Module):
    """WaveSem 双流: CNN → DWT → LowFreqStream + HighFreqStream → concat → [B, 512, 160]"""

    def __init__(self, pretrained_model):
        super().__init__()
        self.encoder = pretrained_model.encoder
        self.multi_level_dwt = pretrained_model.multi_level_dwt
        self.low_stream = pretrained_model.low_stream
        self.high_stream = pretrained_model.high_stream
        self.out_channels = pretrained_model.hidden_dim * 2  # 512

    def forward(self, x):
        feat_cnn = self.encoder(x)
        feat_low_1, feat_high = self.multi_level_dwt(feat_cnn)
        feat_low_2, _, _ = self.low_stream(feat_low_1)
        feat_high_1, _, _ = self.high_stream(
            feat_high[3], feat_high[2], feat_high[1], feat_high[0]
        )
        return torch.cat([feat_low_2, feat_high_1], dim=1)  # [B, 512, 160]


class CNNOnlyEncoder(nn.Module):
    """纯 CNN encoder (来自 StridedCNNAE)"""

    def __init__(self, pretrained_model):
        super().__init__()
        self.encoder = pretrained_model.encoder
        self.out_channels = pretrained_model.hidden_dim

    def forward(self, x):
        return self.encoder(x)  # [B, hidden_dim, T/cr]


class VQVAEEncoder(nn.Module):
    """VQ-VAE 系列的 encoder (Single/Residual codebook)"""

    def __init__(self, pretrained_model):
        super().__init__()
        self.encoder = pretrained_model.encoder
        self.vq = pretrained_model.vq
        self.out_channels = pretrained_model.hidden_dim

    def forward(self, x):
        z = self.encoder(x)  # [B, D, T]
        z_t = z.transpose(1, 2)  # [B, T, D]
        quant, _, _ = self.vq(z_t)
        return quant.transpose(1, 2)  # [B, D, T]


class WaveSemODWTEncoder(nn.Module):
    """WaveSem-oDWT: CNN → channel split → 两个CNN → LowFreqStream → feat"""

    def __init__(self, pretrained_model):
        super().__init__()
        self.encoder = pretrained_model.encoder
        self.encoder_1 = pretrained_model.encoder_1
        self.low_stream = pretrained_model.low_stream
        self.hidden_dim = pretrained_model.hidden_dim
        self.out_channels = pretrained_model.hidden_dim

    def forward(self, x):
        feat = self.encoder(x)
        x_low_1, x_high_1 = feat.split(self.hidden_dim // 2, dim=1)
        x_1 = self.encoder_1(x_low_1)
        feat_low, _, _ = self.low_stream(x_1)
        return feat_low  # [B, 256, T']


class SoundStreamEncoder(nn.Module):
    """SoundStream frozen 编码器包装。

    复现 audiolm_pytorch.SoundStream 的 encode 路径直到 **RVQ 量化之后**，
    返回量化后的离散 token 表示 [B, C, N]，与其它 VQ 编码器（VQVAEEncoder 返回 quant）
    保持一致，从而公平地探测"离散 token"的信息保留能力，而非量化前的连续特征。
    """

    def __init__(self, pretrained_model):
        super().__init__()
        from einops import rearrange as _rearrange
        self._rearrange = _rearrange
        self.encoder = pretrained_model.encoder
        self.encoder_attn = getattr(pretrained_model, "encoder_attn", None)
        self.rq = pretrained_model.rq
        self.out_channels = None  # 首次 forward 时确定 (= codebook_dim)

    def forward(self, x):
        # x: [B, 1, T]
        z = self.encoder(x)                          # [B, codebook_dim, N]
        z = self._rearrange(z, "b c n -> b n c")     # [B, N, C]
        if self.encoder_attn is not None:
            z = self.encoder_attn(z)
        quant, _, _ = self.rq(z)                      # 量化后 [B, N, C]
        feat = self._rearrange(quant, "b n c -> b c n")  # [B, C, N]
        if self.out_channels is None:
            self.out_channels = feat.shape[1]
        return feat


# ─── 完整的对比模型 ──────────────────────────────────────────────────────────

class BaselineDetector(nn.Module):
    """通用的 baseline 甲基化检测模型: encoder + 分类头 (全部可训练, fine-tuning)"""

    def __init__(self, encoder: nn.Module, in_channels: int = 256, freeze: bool = True):
        super().__init__()
        self.encoder = encoder
        self.head = ClassificationHead(in_channels=in_channels)

        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        feat = self.encoder(x)
        return self.head(feat)


# ─── 模型加载器 ──────────────────────────────────────────────────────────────

def load_checkpoint(model, ckpt_path: str, device: str = "cpu"):
    """通用 checkpoint 加载 (对 shape 不匹配鲁棒)。

    不同 vector_quantize_pytorch 版本会把 buffer (如 `_codebook.initted`) 存成
    标量 [] 或 [1]; strict=False 仍会因 shape 不一致报错。这里对 numel 相同的键做
    reshape, numel 不同的键跳过, 并打印告警。
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    model_sd = model.state_dict()
    filtered = {}
    skipped = []
    for k, v in state_dict.items():
        if k not in model_sd:
            continue
        tgt = model_sd[k]
        if not hasattr(v, "shape"):
            filtered[k] = v
        elif v.shape == tgt.shape:
            filtered[k] = v
        elif v.numel() == tgt.numel():
            filtered[k] = v.reshape(tgt.shape)
        else:
            skipped.append((k, tuple(v.shape), tuple(tgt.shape)))
    if skipped:
        logger.warning(
            f"load_checkpoint: skipped {len(skipped)} shape-mismatched key(s); e.g. {skipped[:3]}"
        )
    model.load_state_dict(filtered, strict=False)
    return model


def _load_wavesem_model(ckpt_path: str, device: str):
    """加载 WaveSem 预训练模型（共享）"""
    from wave_vqmodel import WaveletMambaDualStreamVQVAE
    model = WaveletMambaDualStreamVQVAE(
        input_dim=1, hidden_dim=256, cnn_compression_ratio=4,
        wavelet="db4", learnable=True,
    )
    model = load_checkpoint(model, ckpt_path, device)
    return model


def build_wavesem_detector(ckpt_path: str, device: str) -> BaselineDetector:
    """构建 WaveSem 语义流 甲基化检测器"""
    model = _load_wavesem_model(ckpt_path, device)
    encoder = WaveSemEncoder(model)
    return BaselineDetector(encoder, in_channels=256).to(device)


def build_wavesem_texture_detector(ckpt_path: str, device: str) -> BaselineDetector:
    """构建 WaveSem 纹理流 only 甲基化检测器"""
    model = _load_wavesem_model(ckpt_path, device)
    encoder = WaveSemTextureEncoder(model)
    return BaselineDetector(encoder, in_channels=256).to(device)


def build_wavesem_dual_detector(ckpt_path: str, device: str) -> BaselineDetector:
    """构建 WaveSem 双流 (语义+纹理) 甲基化检测器"""
    model = _load_wavesem_model(ckpt_path, device)
    encoder = WaveSemDualStreamEncoder(model)
    return BaselineDetector(encoder, in_channels=512).to(device)


def build_cnnae_detector(ckpt_path: str, device: str, cr: int = 64) -> BaselineDetector:
    """构建 StridedCNNAE 甲基化检测器"""
    from baseline_models import StridedCNNAE
    model = StridedCNNAE(input_dim=1, hidden_dim=256, cnn_compression_ratio=cr)
    model = load_checkpoint(model, ckpt_path, device)
    encoder = CNNOnlyEncoder(model)
    return BaselineDetector(encoder, in_channels=256).to(device)


def build_random_frozen_detector(device: str, cr: int = 64) -> BaselineDetector:
    """构建 **冻结随机初始化** CNN 编码器 + 线性头 (批次效应/泄漏控制实验)。

    与 'cnnae' 的区别: 不加载任何预训练权重, 编码器保持随机初始化并冻结,
    只训练分类头。这是回应 Reviewer2 "批次效应" 疑问的关键对照:
      - 若 random_frozen AUROC 很高 (接近学习型编码器) => 高 AUROC 主要来自
        DM_MSssI/DM 之间的低层批次/run 差异, 而非学到的甲基化表示 (泄漏警告)。
      - 若 random_frozen 明显低于 WaveSem => 高 AUROC 来自学到的表示, 结论可靠。
    """
    from baseline_models import StridedCNNAE
    model = StridedCNNAE(input_dim=1, hidden_dim=256, cnn_compression_ratio=cr)
    # 不加载 checkpoint: 保持随机初始化
    encoder = CNNOnlyEncoder(model)
    return BaselineDetector(encoder, in_channels=256, freeze=True).to(device)


def build_svqvae_detector(ckpt_path: str, device: str, cr: int = 64) -> BaselineDetector:
    """构建 SingleCodebookVQVAE 甲基化检测器"""
    from baseline_models import SingleCodebookVQVAE
    model = SingleCodebookVQVAE(input_dim=1, hidden_dim=256, compression_ratio=cr)
    model = load_checkpoint(model, ckpt_path, device)
    encoder = VQVAEEncoder(model)
    return BaselineDetector(encoder, in_channels=256).to(device)


def build_rvqvae_detector(ckpt_path: str, device: str, cr: int = 64) -> BaselineDetector:
    """构建 RCodebookVQVAE 甲基化检测器"""
    from baseline_models import RCodebookVQVAE
    model = RCodebookVQVAE(input_dim=1, hidden_dim=256, compression_ratio=cr)
    model = load_checkpoint(model, ckpt_path, device)
    encoder = VQVAEEncoder(model)
    return BaselineDetector(encoder, in_channels=256).to(device)


def build_odwt_detector(ckpt_path: str, device: str) -> BaselineDetector:
    """构建 WaveSem-oDWT (无 DWT 消融) 甲基化检测器"""
    from baseline_models import wavesem_oDWT
    model = wavesem_oDWT(input_dim=1, hidden_dim=256)
    model = load_checkpoint(model, ckpt_path, device)
    encoder = WaveSemODWTEncoder(model)
    return BaselineDetector(encoder, in_channels=256).to(device)


def build_soundstream_detector(ckpt_path: str, device: str) -> Optional[BaselineDetector]:
    """构建 SoundStream 甲基化检测器。

    修复要点:
      1. 从 audiolm-pytorch 仓库正确导入 SoundStream (原路径 soundstream.soundstream_pytorch 不存在)。
      2. 架构超参严格读取 checkpoint 同目录 train_args.json, 与训练时一致 (否则 state_dict 无法对齐):
         channels=32, strides=[2,4,4,2], channel_mults=[2,4,8,16], codebook_dim=512, codebook_size=256, ...
    """
    try:
        repo_root = str(Path(__file__).resolve().parent / "soundstream" / "audiolm-pytorch-main")
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from audiolm_pytorch import SoundStream

        # 读取训练超参以对齐架构
        ta_path = os.path.join(os.path.dirname(ckpt_path), "train_args.json")
        ta: Dict[str, Any] = {}
        if os.path.exists(ta_path):
            with open(ta_path) as f:
                ta = json.load(f)

        model = SoundStream(
            channels=ta.get("channels", 32),
            strides=tuple(ta.get("strides", [2, 4, 4, 2])),
            channel_mults=tuple(ta.get("channel_mults", [2, 4, 8, 16])),
            codebook_dim=ta.get("codebook_dim", 512),
            codebook_size=ta.get("codebook_size", 256),
            rq_num_quantizers=ta.get("rq_num_quantizers", 8),
            rq_commitment_weight=ta.get("rq_commitment_weight", 1.0),
            rq_ema_decay=ta.get("rq_ema_decay", 0.95),
            rq_groups=ta.get("rq_groups", 1),
            quantize_dropout_cutoff_index=ta.get("quantize_dropout_cutoff_index", 1),
            target_sample_hz=ta.get("target_sample_hz", 4000),
            use_local_attn=not ta.get("disable_local_attn", False),
            attn_window_size=ta.get("attn_window_size", 128),
            attn_dim_head=ta.get("attn_dim_head", 64),
            attn_heads=ta.get("attn_heads", 8),
            attn_depth=ta.get("attn_depth", 1),
            input_channels=1,
        )
        model = load_checkpoint(model, ckpt_path, device)
        model.to(device)
        encoder = SoundStreamEncoder(model)

        # 先做一次 forward 确定输出通道数 (= codebook_dim)
        with torch.no_grad():
            dummy = torch.randn(1, 1, 10240).to(device)
            feat = encoder(dummy)
            out_channels = feat.shape[1]
        logger.info(f"  SoundStream frozen feature shape: {tuple(feat.shape)} (in_channels={out_channels})")

        return BaselineDetector(encoder, in_channels=out_channels).to(device)
    except Exception as e:
        logger.warning(f"Failed to build SoundStream detector: {e}")
        import traceback
        traceback.print_exc()
        return None


# ─── 评估函数 ──────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, dataloader, device, criterion=None):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    total_loss, n_batches = 0.0, 0

    for signals, labels in dataloader:
        signals, labels = signals.to(device), labels.to(device)
        logits = model(signals)
        probs = F.softmax(logits, dim=-1)[:, 1]
        preds = logits.argmax(dim=-1)

        if criterion is not None:
            total_loss += criterion(logits, labels).item()
            n_batches += 1

        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    metrics = {"loss": total_loss / max(n_batches, 1)}

    if HAS_SKLEARN and len(np.unique(all_labels)) > 1:
        metrics["accuracy"] = accuracy_score(all_labels, all_preds)
        metrics["precision"] = precision_score(all_labels, all_preds, zero_division=0)
        metrics["recall"] = recall_score(all_labels, all_preds, zero_division=0)
        metrics["f1"] = f1_score(all_labels, all_preds, zero_division=0)
        metrics["auroc"] = roc_auc_score(all_labels, all_probs)
        metrics["auprc"] = average_precision_score(all_labels, all_probs)
        metrics["confusion_matrix"] = confusion_matrix(all_labels, all_preds).tolist()

    return metrics


@torch.no_grad()
def predict_wt(model, h5_path, device, batch_size=64):
    """WT 测试集预测"""
    model.eval()
    with h5py.File(h5_path, "r") as f:
        signals = f["signal_segments"][:]

    all_probs = []
    for i in range(0, len(signals), batch_size):
        batch = torch.from_numpy(signals[i:i + batch_size].astype(np.float32)).unsqueeze(1).to(device)
        logits = model(batch)
        probs = F.softmax(logits, dim=-1)[:, 1]
        all_probs.extend(probs.cpu().numpy())

    return np.array(all_probs)


# ─── 训练单个模型 ─────────────────────────────────────────────────────────────

def train_single_model(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    args,
    class_counts: np.ndarray,
) -> Dict[str, Any]:
    """训练并评估单个模型，返回结果字典"""
    logger.info(f"\n{'=' * 70}")
    logger.info(f"  Training: {model_name}")
    logger.info(f"{'=' * 70}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Params: {trainable:,} trainable / {total_params:,} total ({100 * trainable / max(total_params, 1):.1f}%)")

    # 损失
    pos_weight = class_counts[0] / max(class_counts[1], 1)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, pos_weight], dtype=torch.float32, device=device)
    )

    # 优化器 — 只优化可训练参数; 若 encoder 被解冻则给它更小的 LR
    encoder_params = [p for n, p in model.named_parameters()
                      if p.requires_grad and n.startswith("encoder.")]
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and not n.startswith("encoder.")]
    if not (encoder_params or head_params):
        logger.warning(f"  {model_name}: No trainable parameters! Skipping.")
        return {}

    enc_lr = getattr(args, "encoder_lr", args.lr)
    param_groups = []
    if head_params:
        param_groups.append({"params": head_params, "lr": args.lr})
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": enc_lr})
        logger.info(f"  Optimizer: encoder group lr={enc_lr}, head group lr={args.lr}")
    optimizer = AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=max(args.epochs - args.warmup_epochs, 1))
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[args.warmup_epochs])

    # 训练
    best_f1, best_epoch = 0.0, 0
    best_state = None
    history = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, n_steps = 0.0, 0

        for signals, labels in train_loader:
            signals, labels = signals.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(signals)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_steps += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_steps, 1)

        # 验证
        val_metrics = evaluate(model, val_loader, device, criterion)
        val_f1 = val_metrics.get("f1", 0.0)

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                f"  [{model_name}] Epoch {epoch}/{args.epochs} | "
                f"Loss={avg_loss:.4f} | Val Acc={val_metrics.get('accuracy', 0):.4f} "
                f"F1={val_f1:.4f} AUROC={val_metrics.get('auroc', 0):.4f}"
            )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        history.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            **{f"val_{k}": v for k, v in val_metrics.items() if k != "confusion_matrix"},
        })

    train_time = time.time() - start_time

    # 加载最佳模型并最终评估
    if best_state is not None:
        model.load_state_dict(best_state)

    final_metrics = evaluate(model, val_loader, device, criterion)

    logger.info(f"  [{model_name}] Best epoch: {best_epoch}")
    logger.info(f"  [{model_name}] Acc={final_metrics.get('accuracy', 0):.4f} "
                f"F1={final_metrics.get('f1', 0):.4f} "
                f"AUROC={final_metrics.get('auroc', 0):.4f} "
                f"AUPRC={final_metrics.get('auprc', 0):.4f}")

    # WT 预测
    wt_results = None
    if args.test_data and os.path.exists(args.test_data):
        wt_probs = predict_wt(model, args.test_data, device, args.batch_size)
        threshold = 0.5
        n_meth = int(np.sum(wt_probs >= threshold))
        wt_results = {
            "total_segments": len(wt_probs),
            "n_methylated": n_meth,
            "pct_methylated": round(100 * n_meth / len(wt_probs), 2),
            "mean_prob": round(float(wt_probs.mean()), 4),
            "median_prob": round(float(np.median(wt_probs)), 4),
            "std_prob": round(float(wt_probs.std()), 4),
        }

    return {
        "model_name": model_name,
        "trainable_params": trainable,
        "total_params": total_params,
        "best_epoch": best_epoch,
        "train_time_seconds": round(train_time, 1),
        "validation_metrics": {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in final_metrics.items()
        },
        "wt_test_results": wt_results,
        "training_history": history,
    }


# ─── 主函数 ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="5mC Methylation Detection: Baseline Comparison")

    parser.add_argument("--train_data", type=str,
                        default=str(PROJECT_DIR / "methylation_data/ecoli_5mC_train.h5"))
    parser.add_argument("--test_data", type=str,
                        default=str(PROJECT_DIR / "methylation_data/ecoli_5mC_test_wt.h5"))
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str,
                        default=str(PROJECT_DIR / "outputs/methylation_5mC_comparison"))
    parser.add_argument("--compression_ratio", type=int, default=64,
                        help="Compression ratio for baseline models (16/32/64/128)")
    parser.add_argument("--models", type=str, default="all",
                        help="Comma-separated model names or 'all'. "
                             "Options: wavesem,wavesem_texture,wavesem_dual,cnnae,svqvae,rvqvae,odwt,soundstream,random_frozen")
    parser.add_argument("--finetune", action="store_true",
                        help="Unfreeze the pretrained encoder and train it end-to-end "
                             "(diagnostic UPPER-BOUND; this is NO LONGER a frozen-probing experiment).")
    parser.add_argument("--encoder_lr", type=float, default=1e-4,
                        help="LR for the (unfrozen) encoder when --finetune is set; "
                             "head keeps --lr. Ignored in the default frozen-probing mode.")

    args = parser.parse_args()
    cr = args.compression_ratio
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ─── 输出目录 ─────────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ─── 数据加载（共享） ──────────────────────────────────────────────────
    dataset = MethylationDataset(args.train_data)
    n_total = len(dataset)
    n_val = int(n_total * args.val_ratio)
    n_train = n_total - n_val

    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_labels = dataset.labels[train_ds.indices]
    class_counts = np.bincount(train_labels.astype(int), minlength=2)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[train_labels.astype(int)]
    sampler = WeightedRandomSampler(sample_weights, len(train_ds), replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    logger.info(f"Data: {n_train} train, {n_val} val")
    logger.info(f"Class counts: neg={class_counts[0]}, pos={class_counts[1]}")

    # ─── 模型定义 ─────────────────────────────────────────────────────────
    # checkpoint 路径
    ckpt_dir = PROJECT_DIR / "outputs"

    model_configs = {
        "wavesem": {
            "builder": lambda: build_wavesem_detector(
                str(ckpt_dir / "wave_vqmodel/checkpoints/best_model.pth"), str(device)),
            "description": "WaveSem (Ours): DWT + Transformer + RVQ (语义流 only)",
        },
        "wavesem_texture": {
            "builder": lambda: build_wavesem_texture_detector(
                str(ckpt_dir / "wave_vqmodel/checkpoints/best_model.pth"), str(device)),
            "description": "WaveSem-Texture: DWT + 级联门控 + RVQ (纹理流 only)",
        },
        "wavesem_dual": {
            "builder": lambda: build_wavesem_dual_detector(
                str(ckpt_dir / "wave_vqmodel/checkpoints/best_model.pth"), str(device)),
            "description": "WaveSem-Dual: 语义流 + 纹理流 (双流拼接)",
        },
        "cnnae": {
            "builder": lambda: build_cnnae_detector(
                str(ckpt_dir / f"strided_cnn_ae_cr{cr}/checkpoints/best_model.pth"), str(device), cr=cr),
            "description": f"StridedCNNAE (cr={cr})",
        },
        "svqvae": {
            "builder": lambda: build_svqvae_detector(
                str(ckpt_dir / f"single_codebook_vqvae_cr{cr}/checkpoints/best_model.pth"), str(device), cr=cr),
            "description": f"SingleCodebookVQVAE (cr={cr})",
        },
        "rvqvae": {
            "builder": lambda: build_rvqvae_detector(
                str(ckpt_dir / f"r_codebook_vqvae_cr{cr}/checkpoints/best_model.pth"), str(device), cr=cr),
            "description": f"RCodebookVQVAE (cr={cr})",
        },
        "odwt": {
            "builder": lambda: build_odwt_detector(
                str(ckpt_dir / "wavesem_oDWT64/checkpoints/best_model.pth"), str(device)),
            "description": "WaveSem-oDWT (no DWT ablation)",
        },
        "soundstream": {
            "builder": lambda: build_soundstream_detector(
                str(ckpt_dir / "soundstream/checkpoints/best_model.pth"), str(device)),
            "description": "SoundStream (audio codec baseline)",
        },
        "random_frozen": {
            "builder": lambda: build_random_frozen_detector(str(device), cr=cr),
            "description": "Frozen random-init CNN (batch-effect/leakage control)",
        },
    }

    # 解析要运行的模型
    if args.models == "all":
        selected_models = list(model_configs.keys())
    else:
        selected_models = [m.strip() for m in args.models.split(",")]

    # ─── 运行对比实验 ─────────────────────────────────────────────────────
    all_results = {}
    comparison_table = []

    for model_key in selected_models:
        if model_key not in model_configs:
            logger.warning(f"Unknown model: {model_key}, skipping")
            continue

        config = model_configs[model_key]
        logger.info(f"\n{'#' * 70}")
        logger.info(f"  Building: {config['description']}")
        logger.info(f"{'#' * 70}")

        try:
            detector = config["builder"]()
            if detector is None:
                logger.warning(f"  Failed to build {model_key}, skipping")
                continue

            # ── 可选: 解冻编码器做端到端微调 (诊断用上界, 非 frozen probing) ──
            if getattr(args, "finetune", False):
                if model_key == "random_frozen":
                    logger.info("  [finetune] random_frozen 保持冻结 (对照实验语义), 跳过解冻")
                elif hasattr(detector, "encoder"):
                    n_unfrozen = 0
                    for p in detector.encoder.parameters():
                        p.requires_grad = True
                        n_unfrozen += p.numel()
                    logger.info(f"  [finetune] 已解冻 encoder: +{n_unfrozen:,} 参数可训练 "
                                f"(encoder_lr={args.encoder_lr}, head_lr={args.lr})")

            result = train_single_model(
                model=detector,
                model_name=model_key,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                args=args,
                class_counts=class_counts,
            )

            if result:
                result["description"] = config["description"]
                all_results[model_key] = result

                # 保存单模型结果
                model_dir = output_dir / model_key
                model_dir.mkdir(exist_ok=True)
                with open(model_dir / "results.json", "w") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)

                # 保存最佳模型权重
                ckpt_save_path = model_dir / "best_model.pth"
                torch.save({
                    "model_state_dict": detector.state_dict(),
                    "model_name": model_key,
                    "description": config["description"],
                    "best_epoch": result.get("best_epoch", 0),
                    "validation_metrics": result.get("validation_metrics", {}),
                }, ckpt_save_path)
                logger.info(f"  [{model_key}] Saved best checkpoint to {ckpt_save_path}")

                # 汇总
                vm = result.get("validation_metrics", {})
                comparison_table.append({
                    "model": model_key,
                    "description": config["description"],
                    "accuracy": vm.get("accuracy", 0),
                    "precision": vm.get("precision", 0),
                    "recall": vm.get("recall", 0),
                    "f1": vm.get("f1", 0),
                    "auroc": vm.get("auroc", 0),
                    "auprc": vm.get("auprc", 0),
                    "trainable_params": result.get("trainable_params", 0),
                    "train_time": result.get("train_time_seconds", 0),
                    "best_epoch": result.get("best_epoch", 0),
                    "wt_pct_methylated": result.get("wt_test_results", {}).get("pct_methylated", None)
                        if result.get("wt_test_results") else None,
                })

        except Exception as e:
            logger.error(f"  Failed to train {model_key}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # ─── 保存汇总结果 ─────────────────────────────────────────────────────
    summary = {
        "experiment": "5mC CpG Methylation Detection: Baseline Comparison",
        "compression_ratio": cr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "train_samples": n_train,
        "val_samples": n_val,
        "comparison_table": comparison_table,
    }

    with open(output_dir / "comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 打印汇总表格
    logger.info(f"\n\n{'=' * 90}")
    logger.info("  5mC CpG Methylation Detection: Comparison Summary")
    logger.info(f"{'=' * 90}")
    logger.info(f"{'Model':<25} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'AUROC':>8} {'AUPRC':>8}")
    logger.info("-" * 90)

    # 按 AUROC 排序
    comparison_table.sort(key=lambda x: x.get("auroc", 0), reverse=True)

    for row in comparison_table:
        logger.info(
            f"{row['model']:<25} "
            f"{row.get('accuracy', 0):>8.4f} "
            f"{row.get('precision', 0):>8.4f} "
            f"{row.get('recall', 0):>8.4f} "
            f"{row.get('f1', 0):>8.4f} "
            f"{row.get('auroc', 0):>8.4f} "
            f"{row.get('auprc', 0):>8.4f}"
        )

    logger.info(f"\nAll results saved to {output_dir}")
    return summary


if __name__ == "__main__":
    main()
