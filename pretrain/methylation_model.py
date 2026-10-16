#!/usr/bin/env python3
"""
methylation_model.py
=====================
基于 WaveSem 预训练 encoder 的甲基化检测模型。

架构设计：
  ┌─────────────────────────────────────────────────┐
  │  Pretrained WaveSem Encoder (frozen/fine-tuned)  │
  │  Input: [B, 1, 10240] ONT raw signal            │
  │  Output: feat_low [B, 256, 160] (低频语义特征)    │
  └────────────────────┬────────────────────────────┘
                       │
  ┌────────────────────▼────────────────────────────┐
  │  Methylation Classification Head                 │
  │  1. Global Avg/Max Pooling                       │
  │  2. MLP: 256 → 128 → 2 (modified / unmodified) │
  └─────────────────────────────────────────────────┘

支持三种微调策略：
  1. Linear probing: 冻结 encoder，只训练分类头
  2. Fine-tuning: 解冻 encoder 最后几层
  3. Full fine-tuning: 全量微调
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MethylationHead(nn.Module):
    """
    甲基化检测分类头。

    接收 WaveSem encoder 的低频语义特征 feat_low [B, C, T]，
    通过池化 + MLP 输出二分类结果。
    """

    def __init__(
        self,
        in_channels: int = 256,
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.3,
        pool_mode: str = "both",  # "avg", "max", or "both"
    ):
        super().__init__()
        self.pool_mode = pool_mode

        pool_out = in_channels * 2 if pool_mode == "both" else in_channels

        self.classifier = nn.Sequential(
            nn.Linear(pool_out, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: [B, C, T] from encoder
        Returns:
            logits: [B, num_classes]
        """
        if self.pool_mode == "avg":
            pooled = feat.mean(dim=-1)
        elif self.pool_mode == "max":
            pooled = feat.max(dim=-1).values
        else:  # both
            avg_pool = feat.mean(dim=-1)
            max_pool = feat.max(dim=-1).values
            pooled = torch.cat([avg_pool, max_pool], dim=-1)

        return self.classifier(pooled)


class AttentionPooling(nn.Module):
    """
    Attention-based pooling：学习每个时间步的注意力权重。
    比简单的全局平均池化能更好地聚焦于修饰位点附近的信号特征。
    """

    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.Tanh(),
            nn.Linear(dim // 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T]
        Returns:
            pooled: [B, C]
        """
        x_t = x.transpose(1, 2)  # [B, T, C]
        weights = self.attn(x_t)  # [B, T, 1]
        weights = F.softmax(weights, dim=1)
        pooled = (x_t * weights).sum(dim=1)  # [B, C]
        return pooled


class MethylationHeadWithAttention(nn.Module):
    """带注意力池化的甲基化检测头"""

    def __init__(
        self,
        in_channels: int = 256,
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
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
        pooled = self.attn_pool(feat)  # [B, C]
        return self.classifier(pooled)


class MethylationDetector(nn.Module):
    """
    完整的甲基化检测模型。

    集成预训练 WaveSem encoder + 分类头。
    支持不同的微调策略。
    """

    def __init__(
        self,
        pretrained_model: nn.Module,
        head_type: str = "attention",  # "simple" or "attention"
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.3,
        freeze_encoder: bool = True,
        use_low_freq_only: bool = True,  # 是否只使用低频特征
    ):
        super().__init__()
        self.use_low_freq_only = use_low_freq_only
        self.freeze_encoder = freeze_encoder

        # 预训练 encoder 部分
        self.encoder = pretrained_model.encoder
        self.multi_level_dwt = pretrained_model.multi_level_dwt
        self.low_stream = pretrained_model.low_stream

        if not use_low_freq_only:
            self.high_stream = pretrained_model.high_stream

        # 冻结 encoder
        if freeze_encoder:
            self._freeze_encoder()

        # 分类头
        in_channels = pretrained_model.hidden_dim  # 256
        if head_type == "attention":
            self.head = MethylationHeadWithAttention(
                in_channels=in_channels,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                dropout=dropout,
            )
        else:
            self.head = MethylationHead(
                in_channels=in_channels,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                dropout=dropout,
            )

    def _freeze_encoder(self):
        """冻结预训练 encoder 的参数"""
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.multi_level_dwt.parameters():
            param.requires_grad = False
        for param in self.low_stream.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self, unfreeze_layers: Optional[int] = None):
        """
        解冻 encoder 的参数用于 fine-tuning。

        Args:
            unfreeze_layers: None = 全部解冻, int = 只解冻最后 N 层
        """
        if unfreeze_layers is None:
            # 全部解冻
            for param in self.encoder.parameters():
                param.requires_grad = True
            for param in self.multi_level_dwt.parameters():
                param.requires_grad = True
            for param in self.low_stream.parameters():
                param.requires_grad = True
        else:
            # 只解冻 low_stream 的最后几个 Transformer 层
            layers = list(self.low_stream.transformer.layers)
            for layer in layers[-unfreeze_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1, L] ONT raw signal
        Returns:
            logits: [B, num_classes]
        """
        # Encoder
        feat_cnn = self.encoder(x)
        feat_low_1, feat_high = self.multi_level_dwt(feat_cnn)

        # 使用低频流提取语义特征
        feat_low_2, indices_low, vq_loss_low = self.low_stream(feat_low_1)

        # 分类
        logits = self.head(feat_low_2)

        return logits

    def forward_with_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        同时返回 logits 和中间特征，用于可视化分析。
        """
        feat_cnn = self.encoder(x)
        feat_low_1, feat_high = self.multi_level_dwt(feat_cnn)
        feat_low_2, indices_low, vq_loss_low = self.low_stream(feat_low_1)
        logits = self.head(feat_low_2)
        return logits, feat_low_2


# ─── 用于加载模型的辅助函数 ────────────────────────────────────────────────────

def build_methylation_detector(
    pretrained_checkpoint: str,
    device: str = "cuda",
    head_type: str = "attention",
    freeze_encoder: bool = True,
    hidden_dim: int = 256,
    wavelet: str = "db4",
    learnable: bool = True,
    cnn_compression_ratio: int = 4,
) -> MethylationDetector:
    """
    从预训练 checkpoint 构建甲基化检测模型。

    Args:
        pretrained_checkpoint: 预训练模型 .pth 路径
        device: 设备
        head_type: 分类头类型
        freeze_encoder: 是否冻结 encoder
        hidden_dim: 分类头隐藏维度

    Returns:
        MethylationDetector 模型
    """
    # 导入你的预训练模型
    import sys
    sys.path.insert(0, "/owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain")
    from wave_vqmodel import WaveletMambaDualStreamVQVAE

    # 创建预训练模型
    pretrained = WaveletMambaDualStreamVQVAE(
        input_dim=1,
        hidden_dim=hidden_dim,
        cnn_compression_ratio=cnn_compression_ratio,
        wavelet=wavelet,
        learnable=learnable,
    )

    # 加载权重
    ckpt = torch.load(pretrained_checkpoint, map_location="cpu")
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    pretrained.load_state_dict(state_dict, strict=False)
    print(f"Loaded pretrained weights from {pretrained_checkpoint}")

    # 构建检测模型
    model = MethylationDetector(
        pretrained_model=pretrained,
        head_type=head_type,
        hidden_dim=128,
        num_classes=2,
        dropout=0.3,
        freeze_encoder=freeze_encoder,
    )

    return model.to(device)
