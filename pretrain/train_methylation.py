#!/usr/bin/env python3
"""
train_methylation.py
=====================
基于预训练 WaveSem encoder 的甲基化检测训练脚本。

用法:
  python train_methylation.py \
    --data_path methylation_data/5mC_train.h5 \
    --pretrained_checkpoint outputs/wave_vqmodel_Equal/checkpoints/best_model.pth \
    --output_dir outputs/methylation_5mC \
    --mod_type 5mC \
    --freeze_encoder \
    --epochs 50

支持：
  - 自动处理类别不平衡（加权损失）
  - 多种微调策略
  - 详细评估指标 (Accuracy, Precision, Recall, F1, AUROC, AUPRC)
"""

import os
import sys
import json
import argparse
import time
import logging
from pathlib import Path

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score, confusion_matrix,
        classification_report,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── 添加项目路径 ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))


# ─── 数据集 ──────────────────────────────────────────────────────────────────

class MethylationDataset(Dataset):
    """
    从 HDF5 加载甲基化训练数据。

    HDF5 结构:
      - signal_segments: [N, segment_length] float32
      - labels: [N] int8
    """

    def __init__(self, h5_path: str, normalize: str = "zscore"):
        super().__init__()
        self.h5_path = h5_path
        self.normalize = normalize

        # 加载到内存
        with h5py.File(h5_path, "r") as f:
            self.signals = f["signal_segments"][:]  # [N, L]
            self.labels = f["labels"][:]  # [N]

        self.num_positive = int(np.sum(self.labels == 1))
        self.num_negative = int(np.sum(self.labels == 0))

        logger.info(
            f"Loaded {len(self.signals)} samples from {h5_path}: "
            f"positive={self.num_positive}, negative={self.num_negative}"
        )

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx].astype(np.float32)

        if self.normalize == "zscore":
            mu = signal.mean()
            std = max(signal.std(), 1e-6)
            signal = (signal - mu) / std

        signal = torch.from_numpy(signal).unsqueeze(0)  # [1, L]
        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return signal, label


def create_balanced_sampler(dataset: MethylationDataset) -> WeightedRandomSampler:
    """创建平衡采样器处理类别不平衡"""
    labels = dataset.labels
    class_counts = np.bincount(labels.astype(int))
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels.astype(int)]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True,
    )


# ─── 评估 ────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, dataloader, device, criterion=None):
    """评估模型性能"""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    total_loss = 0.0
    n_batches = 0

    for signals, labels in dataloader:
        signals = signals.to(device)
        labels = labels.to(device)

        logits = model(signals)
        probs = F.softmax(logits, dim=-1)[:, 1]
        preds = logits.argmax(dim=-1)

        if criterion is not None:
            loss = criterion(logits, labels)
            total_loss += loss.item()
            n_batches += 1

        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    metrics = {"loss": total_loss / max(n_batches, 1)}

    if HAS_SKLEARN:
        metrics["accuracy"] = accuracy_score(all_labels, all_preds)
        metrics["precision"] = precision_score(all_labels, all_preds, zero_division=0)
        metrics["recall"] = recall_score(all_labels, all_preds, zero_division=0)
        metrics["f1"] = f1_score(all_labels, all_preds, zero_division=0)

        if len(np.unique(all_labels)) > 1:
            metrics["auroc"] = roc_auc_score(all_labels, all_probs)
            metrics["auprc"] = average_precision_score(all_labels, all_probs)
        else:
            metrics["auroc"] = 0.0
            metrics["auprc"] = 0.0

        metrics["confusion_matrix"] = confusion_matrix(all_labels, all_preds).tolist()
    else:
        correct = (all_preds == all_labels).sum()
        metrics["accuracy"] = float(correct) / len(all_labels)

    return metrics


# ─── 训练循环 ─────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 数据
    dataset = MethylationDataset(args.data_path, normalize=args.normalize)

    # Train/Val 划分
    n_total = len(dataset)
    n_val = int(n_total * args.val_ratio)
    n_train = n_total - n_val

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    # 平衡采样 (只对训练集)
    train_labels = dataset.labels[train_dataset.indices]

    class_counts = np.bincount(train_labels.astype(int), minlength=2)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[train_labels.astype(int)]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=sampler, num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=True,
    )

    # 模型
    from methylation_model import build_methylation_detector

    model = build_methylation_detector(
        pretrained_checkpoint=args.pretrained_checkpoint,
        device=str(device),
        head_type=args.head_type,
        freeze_encoder=args.freeze_encoder,
        hidden_dim=args.hidden_dim,
        wavelet=args.wavelet,
        learnable=args.learnable_wavelet,
        cnn_compression_ratio=args.cnn_compression_ratio,
    )

    # 加权交叉熵损失
    pos_weight = torch.tensor(
        [class_counts[0] / max(class_counts[1], 1)],
        device=device,
    )
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            [1.0, pos_weight.item()],
            device=device,
        )
    )

    # 优化器
    param_groups = [
        {"params": [p for p in model.head.parameters() if p.requires_grad], "lr": args.lr},
    ]
    if not args.freeze_encoder:
        encoder_params = [p for n, p in model.named_parameters()
                          if p.requires_grad and "head" not in n]
        param_groups.append({"params": encoder_params, "lr": args.lr * 0.1})

    optimizer = AdamW(param_groups, weight_decay=args.weight_decay)

    # 学习率调度
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[args.warmup_epochs])

    # 输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    # 保存配置
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # 训练
    best_f1 = 0.0
    best_epoch = 0
    history = []

    logger.info(f"Starting training for {args.epochs} epochs...")
    logger.info(f"  Train samples: {n_train}, Val samples: {n_val}")
    logger.info(f"  Class distribution: negative={class_counts[0]}, positive={class_counts[1]}")
    logger.info(f"  Freeze encoder: {args.freeze_encoder}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_steps = 0

        for batch_idx, (signals, labels) in enumerate(train_loader):
            signals = signals.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(signals)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_steps += 1

            if batch_idx % args.log_every == 0:
                logger.info(
                    f"  Epoch {epoch}/{args.epochs} | Step {batch_idx}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f}"
                )

        scheduler.step()
        avg_train_loss = epoch_loss / max(n_steps, 1)

        # 验证
        val_metrics = evaluate(model, val_loader, device, criterion)
        val_f1 = val_metrics.get("f1", 0.0)

        logger.info(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Acc: {val_metrics['accuracy']:.4f} | "
            f"F1: {val_f1:.4f} | "
            f"AUROC: {val_metrics.get('auroc', 0):.4f} | "
            f"AUPRC: {val_metrics.get('auprc', 0):.4f}"
        )

        # 保存最佳模型
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                },
                os.path.join(args.output_dir, "checkpoints", "best_model.pth"),
            )
            logger.info(f"  → Saved best model (F1={best_f1:.4f})")

        # 定期保存
        if epoch % args.save_every == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                },
                os.path.join(args.output_dir, "checkpoints", f"epoch_{epoch}.pth"),
            )

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items() if k != "confusion_matrix"},
        })

    # 保存训练历史
    with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"\nTraining complete! Best F1: {best_f1:.4f} at epoch {best_epoch}")

    # 最终评估
    if HAS_SKLEARN:
        logger.info("\n=== Final Evaluation on Validation Set ===")
        final_metrics = evaluate(model, val_loader, device, criterion)
        report = classification_report(
            [0] * final_metrics.get("confusion_matrix", [[0, 0], [0, 0]])[0][0] +
            [0] * final_metrics.get("confusion_matrix", [[0, 0], [0, 0]])[0][1] +
            [1] * final_metrics.get("confusion_matrix", [[0, 0], [0, 0]])[1][0] +
            [1] * final_metrics.get("confusion_matrix", [[0, 0], [0, 0]])[1][1],
            target_names=["Unmodified", "Modified"],
            zero_division=0,
        )
        logger.info(f"\n{report}")


def main():
    parser = argparse.ArgumentParser(description="Train methylation detector")

    # 数据
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to methylation HDF5 data")
    parser.add_argument("--normalize", type=str, default="zscore",
                        choices=["none", "zscore"], help="Signal normalization")
    parser.add_argument("--val_ratio", type=float, default=0.15)

    # 预训练模型
    parser.add_argument("--pretrained_checkpoint", type=str, required=True,
                        help="Path to pretrained WaveSem .pth")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--cnn_compression_ratio", type=int, default=4)
    parser.add_argument("--wavelet", type=str, default="db4")
    parser.add_argument("--learnable_wavelet", action="store_true")

    # 微调策略
    parser.add_argument("--freeze_encoder", action="store_true",
                        help="Freeze pretrained encoder (linear probing)")
    parser.add_argument("--head_type", type=str, default="attention",
                        choices=["simple", "attention"])

    # 训练
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=10)

    # 修饰类型
    parser.add_argument("--mod_type", type=str, default="5mC",
                        choices=["5mC", "6mA"])

    # 输出
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
