#!/usr/bin/env python3
"""
eval_finetune_model.py
========================
加载端到端微调（不冻结编码器）的 WaveSem checkpoint，在验证集和 WT 测试集上评估性能。
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
from torch.utils.data import Dataset, DataLoader, random_split

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

PROJECT_DIR = Path("/owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain")
sys.path.insert(0, str(Path(__file__).resolve().parent))


class MethylationDataset(Dataset):
    """从 HDF5 加载甲基化训练数据"""

    def __init__(self, h5_path: str, normalize: str = "none"):
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

    return metrics, all_probs, all_labels


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
        batch_tensor = torch.from_numpy(batch).unsqueeze(1).to(device)
        logits = model(batch_tensor)
        probs = F.softmax(logits, dim=-1)[:, 1]
        all_probs.extend(probs.cpu().numpy())

    return np.array(all_probs)


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned WaveSem 5mC model")
    parser.add_argument("--finetune_dir", type=str,
                        default=str(PROJECT_DIR / "outputs/methylation_5mC_ecoli_finetune"))
    parser.add_argument("--train_data", type=str,
                        default=str(PROJECT_DIR / "methylation_data/ecoli_5mC_train.h5"))
    parser.add_argument("--test_data", type=str,
                        default=str(PROJECT_DIR / "methylation_data/ecoli_5mC_test_wt.h5"))
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    finetune_dir = Path(args.finetune_dir)
    ckpt_path = finetune_dir / "checkpoints" / "best_model.pth"

    # ─── 加载配置 ──────────────────────────────────────────────────────────
    config_path = finetune_dir / "experiment_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        logger.info(f"Loaded config: freeze_encoder={config.get('freeze_encoder')}, "
                    f"epochs={config.get('epochs')}, lr={config.get('lr')}, "
                    f"head_type={config.get('head_type')}")
    else:
        config = {}

    # ─── 加载模型 ──────────────────────────────────────────────────────────
    from methylation_model import build_methylation_detector

    pretrained_ckpt = config.get("pretrained_checkpoint",
                                  str(PROJECT_DIR / "outputs/wave_vqmodel_Equal/checkpoints/best_model.pth"))
    hidden_dim = config.get("hidden_dim", 256)
    head_type = config.get("head_type", "attention")
    wavelet = config.get("wavelet", "db4")
    learnable = config.get("learnable_wavelet", True)
    cnn_ratio = config.get("cnn_compression_ratio", 4)

    model = build_methylation_detector(
        pretrained_checkpoint=pretrained_ckpt,
        device=str(device),
        head_type=head_type,
        freeze_encoder=False,  # finetune 模型不冻结
        hidden_dim=hidden_dim,
        wavelet=wavelet,
        learnable=learnable,
        cnn_compression_ratio=cnn_ratio,
    )

    # 加载 finetune checkpoint
    logger.info(f"Loading fine-tuned checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        ckpt_epoch = ckpt.get("epoch", "unknown")
        ckpt_val_metrics = ckpt.get("val_metrics", {})
        logger.info(f"Checkpoint from epoch {ckpt_epoch}")
        if ckpt_val_metrics:
            logger.info(f"  Checkpoint val metrics: F1={ckpt_val_metrics.get('f1', 'N/A'):.4f}, "
                       f"AUROC={ckpt_val_metrics.get('auroc', 'N/A'):.4f}")
    else:
        model.load_state_dict(ckpt)

    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model total params: {total_params:,}")

    # ─── 数据加载 ──────────────────────────────────────────────────────────
    dataset = MethylationDataset(args.train_data, normalize="none")

    n_total = len(dataset)
    n_val = int(n_total * args.val_ratio)
    n_train = n_total - n_val

    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    # ─── 验证集评估 ────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION SET EVALUATION (Fine-tuned Model)")
    logger.info("=" * 60)

    criterion = nn.CrossEntropyLoss()
    val_metrics, val_probs, val_labels = evaluate(model, val_loader, device, criterion)

    logger.info(f"  Accuracy:  {val_metrics.get('accuracy', 0):.4f}")
    logger.info(f"  Precision: {val_metrics.get('precision', 0):.4f}")
    logger.info(f"  Recall:    {val_metrics.get('recall', 0):.4f}")
    logger.info(f"  F1 Score:  {val_metrics.get('f1', 0):.4f}")
    logger.info(f"  AUROC:     {val_metrics.get('auroc', 0):.4f}")
    logger.info(f"  AUPRC:     {val_metrics.get('auprc', 0):.4f}")

    if "confusion_matrix" in val_metrics:
        cm = val_metrics["confusion_matrix"]
        logger.info(f"  Confusion Matrix:")
        logger.info(f"    TN={cm[0][0]:>6}, FP={cm[0][1]:>6}")
        logger.info(f"    FN={cm[1][0]:>6}, TP={cm[1][1]:>6}")

    # ─── WT 测试集预测 ─────────────────────────────────────────────────────
    wt_results = None
    if args.test_data and os.path.exists(args.test_data):
        logger.info("\n" + "=" * 60)
        logger.info("WT TEST SET PREDICTION (Fine-tuned Model)")
        logger.info("=" * 60)

        wt_probs = predict_wt(model, args.test_data, device, args.batch_size)

        threshold = 0.5
        n_methylated = int(np.sum(wt_probs >= threshold))
        n_unmethylated = int(np.sum(wt_probs < threshold))
        total_wt = len(wt_probs)

        logger.info(f"  WT predictions (threshold={threshold}):")
        logger.info(f"    Methylated:   {n_methylated} ({100 * n_methylated / total_wt:.1f}%)")
        logger.info(f"    Unmethylated: {n_unmethylated} ({100 * n_unmethylated / total_wt:.1f}%)")
        logger.info(f"    Mean prob:    {wt_probs.mean():.4f}")
        logger.info(f"    Median prob:  {np.median(wt_probs):.4f}")
        logger.info(f"    Std prob:     {wt_probs.std():.4f}")

        # 概率分布
        bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        hist, _ = np.histogram(wt_probs, bins=bins)
        logger.info("  Probability distribution:")
        for i in range(len(bins) - 1):
            logger.info(f"    [{bins[i]:.1f}, {bins[i+1]:.1f}): {hist[i]:>6} "
                       f"({100 * hist[i] / total_wt:.1f}%)")

        np.save(finetune_dir / "wt_predictions.npy", wt_probs)
        logger.info(f"  Saved WT predictions to {finetune_dir / 'wt_predictions.npy'}")

        wt_results = {
            "total_segments": total_wt,
            "n_methylated": n_methylated,
            "n_unmethylated": n_unmethylated,
            "pct_methylated": round(100 * n_methylated / total_wt, 2),
            "mean_prob": round(float(wt_probs.mean()), 4),
            "median_prob": round(float(np.median(wt_probs)), 4),
            "std_prob": round(float(wt_probs.std()), 4),
        }

    # ─── 与冻结编码器结果对比 ──────────────────────────────────────────────
    frozen_results_path = PROJECT_DIR / "outputs/methylation_5mC_ecoli/experiment_results.json"
    if frozen_results_path.exists():
        logger.info("\n" + "=" * 60)
        logger.info("COMPARISON: Fine-tuned vs Frozen Encoder (Linear Probe)")
        logger.info("=" * 60)

        with open(frozen_results_path) as f:
            frozen_results = json.load(f)

        frozen_metrics = frozen_results.get("validation_metrics", {})
        finetune_metrics = val_metrics

        comparison_keys = ["accuracy", "precision", "recall", "f1", "auroc", "auprc"]
        logger.info(f"  {'Metric':<12} {'Frozen':<10} {'Fine-tuned':<12} {'Delta':<10} {'Improve%':<10}")
        logger.info(f"  {'-'*54}")

        for key in comparison_keys:
            frozen_val = frozen_metrics.get(key, 0)
            finetune_val = finetune_metrics.get(key, 0)
            delta = finetune_val - frozen_val
            improve_pct = (delta / max(frozen_val, 1e-8)) * 100
            direction = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
            logger.info(f"  {key:<12} {frozen_val:<10.4f} {finetune_val:<12.4f} "
                       f"{direction}{abs(delta):<9.4f} {improve_pct:>+8.1f}%")
    else:
        frozen_results = None

    # ─── 保存完整结果 ──────────────────────────────────────────────────────
    results = {
        "experiment": "E. coli 5mC CpG Methylation Detection (Fine-tuned, Unfrozen Encoder)",
        "model": "WaveSem + MethylationDetector (End-to-End Fine-tuning)",
        "freeze_encoder": False,
        "pretrained_checkpoint": pretrained_ckpt,
        "finetune_checkpoint": str(ckpt_path),
        "device": str(device),
        "val_samples": n_val,
        "validation_metrics": {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in val_metrics.items()
        },
        "wt_test_results": wt_results,
        "config": config,
    }

    if frozen_results:
        results["comparison_with_frozen"] = {
            "frozen_metrics": frozen_results.get("validation_metrics", {}),
            "finetune_metrics": {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in val_metrics.items()
            },
        }

    results_path = finetune_dir / "experiment_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"\nResults saved to {results_path}")

    return results


if __name__ == "__main__":
    main()
