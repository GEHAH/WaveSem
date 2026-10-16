#!/usr/bin/env python3
"""
run_ecoli_5mC_experiment.py
==============================
一键运行 E. coli 5mC CpG 甲基化检测实验。

实验流程:
  1. 加载预训练 WaveSem encoder
  2. 在 DM_MSssI (阳性) + DM (阴性) 数据上训练 MethylationDetector
  3. 验证集评估 (Accuracy, F1, AUROC, AUPRC)
  4. 在 WT 测试集上进行预测
  5. 输出完整结果报告

基于 ONT 官方 E. coli 5mC CpG 基准数据集。
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path

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
        classification_report,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── 项目路径 ─────────────────────────────────────────────────────────────────
PROJECT_DIR = Path("/owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain")
sys.path.insert(0, str(Path(__file__).resolve().parent))


# ─── 数据集 ──────────────────────────────────────────────────────────────────

class MethylationDataset(Dataset):
    """从 HDF5 加载甲基化训练数据"""

    def __init__(self, h5_path: str, normalize: str = "none"):
        """
        Args:
            h5_path: HDF5 文件路径
            normalize: "none" (数据已预归一化), "zscore"
        """
        super().__init__()
        self.normalize = normalize

        with h5py.File(h5_path, "r") as f:
            self.signals = f["signal_segments"][:]
            self.labels = f["labels"][:]

        self.num_positive = int(np.sum(self.labels == 1))
        self.num_negative = int(np.sum(self.labels == 0))
        logger.info(
            f"Loaded {len(self.signals)} samples: "
            f"pos={self.num_positive}, neg={self.num_negative}"
        )

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx].astype(np.float32)

        if self.normalize == "zscore":
            mu, std = signal.mean(), max(signal.std(), 1e-6)
            signal = (signal - mu) / std

        signal = torch.from_numpy(signal).unsqueeze(0)  # [1, L]
        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return signal, label


# ─── 评估 ────────────────────────────────────────────────────────────────────

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
    else:
        correct = (all_preds == all_labels).sum()
        metrics["accuracy"] = float(correct) / len(all_labels)

    return metrics


# ─── WT 预测 ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_wt(model, h5_path, device, batch_size=64, normalize="none"):
    """对 WT 测试集进行预测"""
    model.eval()

    with h5py.File(h5_path, "r") as f:
        signals = f["signal_segments"][:]

    all_probs = []
    n = len(signals)

    for i in range(0, n, batch_size):
        batch = signals[i:i + batch_size].astype(np.float32)
        if normalize == "zscore":
            mu = batch.mean(axis=-1, keepdims=True)
            std = np.maximum(batch.std(axis=-1, keepdims=True), 1e-6)
            batch = (batch - mu) / std

        batch_tensor = torch.from_numpy(batch).unsqueeze(1).to(device)  # [B, 1, L]
        logits = model(batch_tensor)
        probs = F.softmax(logits, dim=-1)[:, 1]
        all_probs.extend(probs.cpu().numpy())

    all_probs = np.array(all_probs)
    return all_probs


# ─── 主训练函数 ────────────────────────────────────────────────────────────────

def run_experiment(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    start_time = time.time()

    # ─── 数据加载 ──────────────────────────────────────────────────────────
    # 数据已经做过MAD归一化，这里不需要再归一化
    dataset = MethylationDataset(args.train_data, normalize="none")

    n_total = len(dataset)
    n_val = int(n_total * args.val_ratio)
    n_train = n_total - n_val

    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    # 平衡采样
    train_labels = dataset.labels[train_ds.indices]
    class_counts = np.bincount(train_labels.astype(int), minlength=2)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[train_labels.astype(int)]
    sampler = WeightedRandomSampler(sample_weights, len(train_ds), replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    # ─── 模型 ──────────────────────────────────────────────────────────────
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

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model params: {trainable:,} trainable / {total_params:,} total "
                f"({100 * trainable / total_params:.1f}%)")

    # ─── 损失与优化器 ──────────────────────────────────────────────────────
    pos_weight_val = class_counts[0] / max(class_counts[1], 1)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, pos_weight_val], dtype=torch.float32, device=device)
    )

    param_groups = [
        {"params": [p for p in model.head.parameters() if p.requires_grad], "lr": args.lr},
    ]
    if not args.freeze_encoder:
        encoder_params = [p for n, p in model.named_parameters()
                          if p.requires_grad and "head" not in n]
        if encoder_params:
            param_groups.append({"params": encoder_params, "lr": args.lr * 0.1})

    optimizer = AdamW(param_groups, weight_decay=args.weight_decay)

    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=max(args.epochs - args.warmup_epochs, 1))
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[args.warmup_epochs])

    # ─── 输出目录 ──────────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "experiment_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # ─── 训练循环 ──────────────────────────────────────────────────────────
    best_f1, best_epoch = 0.0, 0
    history = []

    logger.info(f"Training: {args.epochs} epochs, batch_size={args.batch_size}")
    logger.info(f"  Train: {n_train}, Val: {n_val}")
    logger.info(f"  Class: neg={class_counts[0]}, pos={class_counts[1]}")
    logger.info(f"  Freeze encoder: {args.freeze_encoder}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, n_steps = 0.0, 0

        for batch_idx, (signals, labels) in enumerate(train_loader):
            signals, labels = signals.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(signals)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_steps += 1

            if batch_idx % args.log_every == 0 and batch_idx > 0:
                logger.info(f"  Epoch {epoch}/{args.epochs} Step {batch_idx}/{len(train_loader)} "
                            f"Loss={loss.item():.4f}")

        scheduler.step()
        avg_train_loss = epoch_loss / max(n_steps, 1)

        # 验证
        val_metrics = evaluate(model, val_loader, device, criterion)
        val_f1 = val_metrics.get("f1", 0.0)

        logger.info(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss={avg_train_loss:.4f} | "
            f"Val: Loss={val_metrics['loss']:.4f} "
            f"Acc={val_metrics.get('accuracy', 0):.4f} "
            f"F1={val_f1:.4f} "
            f"AUROC={val_metrics.get('auroc', 0):.4f} "
            f"AUPRC={val_metrics.get('auprc', 0):.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "val_metrics": val_metrics, "config": vars(args)},
                ckpt_dir / "best_model.pth",
            )
            logger.info(f"  → Best model saved (F1={best_f1:.4f})")

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items() if k != "confusion_matrix"},
        })

    # 保存训练历史
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # ─── 最终验证集评估 ────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("FINAL EVALUATION ON VALIDATION SET")
    logger.info("=" * 60)

    # 加载最佳模型
    best_ckpt = torch.load(ckpt_dir / "best_model.pth", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    final_metrics = evaluate(model, val_loader, device, criterion)
    logger.info(f"Best epoch: {best_epoch}")
    logger.info(f"  Accuracy:  {final_metrics.get('accuracy', 0):.4f}")
    logger.info(f"  Precision: {final_metrics.get('precision', 0):.4f}")
    logger.info(f"  Recall:    {final_metrics.get('recall', 0):.4f}")
    logger.info(f"  F1 Score:  {final_metrics.get('f1', 0):.4f}")
    logger.info(f"  AUROC:     {final_metrics.get('auroc', 0):.4f}")
    logger.info(f"  AUPRC:     {final_metrics.get('auprc', 0):.4f}")

    if "confusion_matrix" in final_metrics:
        cm = final_metrics["confusion_matrix"]
        logger.info(f"  Confusion Matrix:")
        logger.info(f"    TN={cm[0][0]}, FP={cm[0][1]}")
        logger.info(f"    FN={cm[1][0]}, TP={cm[1][1]}")

    # ─── WT 测试集预测 ─────────────────────────────────────────────────────
    if args.test_data and os.path.exists(args.test_data):
        logger.info("\n" + "=" * 60)
        logger.info("WT TEST SET PREDICTION")
        logger.info("=" * 60)

        wt_probs = predict_wt(model, args.test_data, device, args.batch_size)

        # 统计
        threshold = 0.5
        n_methylated = int(np.sum(wt_probs >= threshold))
        n_unmethylated = int(np.sum(wt_probs < threshold))
        total_wt = len(wt_probs)

        logger.info(f"WT predictions (threshold={threshold}):")
        logger.info(f"  Methylated:   {n_methylated} ({100 * n_methylated / total_wt:.1f}%)")
        logger.info(f"  Unmethylated: {n_unmethylated} ({100 * n_unmethylated / total_wt:.1f}%)")
        logger.info(f"  Mean prob:    {wt_probs.mean():.4f}")
        logger.info(f"  Median prob:  {np.median(wt_probs):.4f}")
        logger.info(f"  Std prob:     {wt_probs.std():.4f}")

        # 保存预测结果
        np.save(output_dir / "wt_predictions.npy", wt_probs)

        # 概率分布统计
        bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        hist, _ = np.histogram(wt_probs, bins=bins)
        logger.info("  Probability distribution:")
        for i in range(len(bins) - 1):
            logger.info(f"    [{bins[i]:.1f}, {bins[i + 1]:.1f}): {hist[i]} "
                        f"({100 * hist[i] / total_wt:.1f}%)")

        # WT 预测结果汇总
        wt_results = {
            "total_segments": total_wt,
            "n_methylated": n_methylated,
            "n_unmethylated": n_unmethylated,
            "pct_methylated": round(100 * n_methylated / total_wt, 2),
            "mean_prob": round(float(wt_probs.mean()), 4),
            "median_prob": round(float(np.median(wt_probs)), 4),
            "std_prob": round(float(wt_probs.std()), 4),
        }
    else:
        wt_results = None

    # ─── 完整结果报告 ──────────────────────────────────────────────────────
    total_time = time.time() - start_time
    results = {
        "experiment": "E. coli 5mC CpG Methylation Detection",
        "model": "WaveSem + MethylationDetector",
        "pretrained_checkpoint": args.pretrained_checkpoint,
        "device": str(device),
        "total_time_seconds": round(total_time, 1),
        "train_samples": n_train,
        "val_samples": n_val,
        "best_epoch": best_epoch,
        "validation_metrics": {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in final_metrics.items()
        },
        "wt_test_results": wt_results,
        "config": vars(args),
    }

    with open(output_dir / "experiment_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"\nExperiment completed in {total_time:.1f}s")
    logger.info(f"Results saved to {output_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(description="E. coli 5mC CpG Methylation Detection Experiment")

    # 数据
    parser.add_argument("--train_data", type=str,
                        default=str(PROJECT_DIR / "methylation_data/ecoli_5mC_train.h5"))
    parser.add_argument("--test_data", type=str,
                        default=str(PROJECT_DIR / "methylation_data/ecoli_5mC_test_wt.h5"))
    parser.add_argument("--val_ratio", type=float, default=0.15)

    # 预训练模型
    parser.add_argument("--pretrained_checkpoint", type=str,
                        default=str(PROJECT_DIR / "outputs/wave_vqmodel_Equal/checkpoints/best_model.pth"))
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--cnn_compression_ratio", type=int, default=4)
    parser.add_argument("--wavelet", type=str, default="db4")
    parser.add_argument("--learnable_wavelet", action="store_true", default=True)

    # 微调策略
    parser.add_argument("--freeze_encoder", action="store_true", default=False,
                        help="Freeze pretrained encoder (linear probing)")
    parser.add_argument("--no_freeze_encoder", dest="freeze_encoder", action="store_false",
                        help="Unfreeze encoder for end-to-end fine-tuning")
    parser.add_argument("--head_type", type=str, default="attention",
                        choices=["simple", "attention"])

    # 训练超参
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for classification head (encoder uses lr*0.1)")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=100)

    # 输出
    parser.add_argument("--output_dir", type=str, default=None)

    args = parser.parse_args()

    # 根据 freeze_encoder 自动设置输出目录
    if args.output_dir is None:
        if args.freeze_encoder:
            args.output_dir = str(PROJECT_DIR / "outputs/methylation_5mC_ecoli")
        else:
            args.output_dir = str(PROJECT_DIR / "outputs/methylation_5mC_ecoli_finetune")

    run_experiment(args)


if __name__ == "__main__":
    main()
