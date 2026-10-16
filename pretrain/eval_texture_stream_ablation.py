#!/usr/bin/env python
"""
Texture Stream 消融实验：对比 WaveSem 三种推理模式的重构质量。

审稿人 Review 1 & 3 要求：
  "provide an ablation where the Texture Stream is completely removed during
   inference—i.e., where reconstruction relies solely on the Semantic Stream."

本脚本加载训练好的 WaveSem checkpoint，在验证集上分别运行:
  1. full        — model.forward()          (Semantic + Texture)
  2. low_only    — model.forward_low_only()  (Semantic only, Texture 全部置零)
  3. high_only   — model.forward_high_only() (Texture only, Semantic 全部置零)

报告指标:
  - MSE, MAE, RMSE, PSNR, SNR, Pearson Correlation
  - Multi-Scale STFT Error (Spectral Convergence + Log-Mag)
  - VQ Loss

输出:
  - JSON 结果文件
  - 对比可视化 (可选)
  - 控制台汇总表格

Usage:
  python eval_texture_stream_ablation.py \
      --checkpoint outputs/wave_vqmodel_Equal/checkpoints/best_model.pth \
      --data_path raw_data/temp_signal_segments_10240 \
      --output_dir outputs/texture_stream_ablation \
      --max_files 2 \
      --batch_size 64 \
      --num_samples 5000
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# ── project imports ──────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from h5_data_process import HDF5SignalDataset  # noqa: E402
from wave_vqmodel import (  # noqa: E402
    MultiScaleSpectralLoss,
    WaveletMambaDualStreamVQVAE,
)

import warnings
warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════════
#  Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_mse(orig: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    """Per-sample MSE.  orig/recon: [B, 1, L] -> [B]"""
    return F.mse_loss(recon, orig, reduction="none").mean(dim=(1, 2))


def compute_mae(orig: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(recon, orig, reduction="none").mean(dim=(1, 2))


def compute_snr(orig: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    """Signal-to-Noise Ratio in dB.  orig/recon: [B, 1, L] -> [B]"""
    noise = orig - recon
    sig_power = (orig ** 2).mean(dim=(1, 2))
    noise_power = (noise ** 2).mean(dim=(1, 2)).clamp_min(1e-10)
    return 10 * torch.log10(sig_power / noise_power)


def compute_psnr(orig: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    """Peak SNR.  orig/recon: [B, 1, L] -> [B]"""
    mse = F.mse_loss(recon, orig, reduction="none").mean(dim=(1, 2)).clamp_min(1e-10)
    peak = orig.abs().amax(dim=(1, 2)) ** 2
    return 10 * torch.log10(peak / mse)


def compute_pearson(orig: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    """Pearson correlation per sample.  [B, 1, L] -> [B]"""
    o = orig.squeeze(1)  # [B, L]
    r = recon.squeeze(1)
    o_centered = o - o.mean(dim=1, keepdim=True)
    r_centered = r - r.mean(dim=1, keepdim=True)
    num = (o_centered * r_centered).sum(dim=1)
    den = (o_centered.norm(dim=1) * r_centered.norm(dim=1)).clamp_min(1e-8)
    return num / den


def compute_stft_error(orig: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    """Multi-scale STFT error (scalar per batch, replicated to [B])."""
    spec_loss_fn = MultiScaleSpectralLoss()
    spec_loss_fn = spec_loss_fn.to(orig.device)
    val = spec_loss_fn(orig, recon)
    return val.expand(orig.size(0))


@dataclass
class AblationMetrics:
    """Accumulates per-sample metrics across batches."""
    mse: List[float] = field(default_factory=list)
    mae: List[float] = field(default_factory=list)
    snr: List[float] = field(default_factory=list)
    psnr: List[float] = field(default_factory=list)
    pearson: List[float] = field(default_factory=list)
    stft_error: List[float] = field(default_factory=list)
    vq_loss: List[float] = field(default_factory=list)

    def update(self, batch_metrics: Dict[str, torch.Tensor]):
        for key in ["mse", "mae", "snr", "psnr", "pearson", "stft_error", "vq_loss"]:
            values = batch_metrics.get(key)
            if values is not None:
                getattr(self, key).extend(values.detach().cpu().tolist())

    def summarize(self) -> Dict[str, float]:
        result = {}
        for key in ["mse", "mae", "snr", "psnr", "pearson", "stft_error", "vq_loss"]:
            arr = getattr(self, key)
            if arr:
                result[f"{key}_mean"] = float(np.mean(arr))
                result[f"{key}_std"] = float(np.std(arr))
                result[f"{key}_median"] = float(np.median(arr))
            else:
                result[f"{key}_mean"] = 0.0
                result[f"{key}_std"] = 0.0
                result[f"{key}_median"] = 0.0
        result["num_samples"] = len(self.mse)
        return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Model loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(
    checkpoint_path: str,
    device: torch.device,
    hidden_dim: int = 256,
    cnn_compression_ratio: int = 4,
    wavelet: str = "db4",
    learnable: bool = True,
) -> WaveletMambaDualStreamVQVAE:
    """Load a pretrained WaveSem model from a checkpoint."""
    model = WaveletMambaDualStreamVQVAE(
        input_dim=1,
        hidden_dim=hidden_dim,
        cnn_compression_ratio=cnn_compression_ratio,
        wavelet=wavelet,
        learnable=learnable,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 从 checkpoint 中提取 args (如果有的话)
    ckpt_args = ckpt.get("args", {})
    print(f"[Model] Loaded checkpoint from: {checkpoint_path}")
    print(f"[Model] hidden_dim={hidden_dim}, cnn_compression_ratio={cnn_compression_ratio}, "
          f"wavelet={wavelet}, learnable={learnable}")
    if ckpt_args:
        print(f"[Model] Training args from ckpt: epochs={ckpt_args.get('num_epochs')}, "
              f"lr={ckpt_args.get('base_learning_rate')}")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
#  Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def build_val_loader(
    data_path: str,
    max_files: int = 0,
    batch_size: int = 64,
    num_workers: int = 4,
    segment_length: int = 10240,
    num_samples: int = 0,
    seed: int = 42,
) -> DataLoader:
    """Build a validation DataLoader from H5 files."""
    h5_files = sorted(glob.glob(os.path.join(data_path, "*.h5")))
    if max_files > 0:
        h5_files = h5_files[:max_files]
    if len(h5_files) == 0:
        raise ValueError(f"No *.h5 files found in {data_path}")

    dataset = HDF5SignalDataset(h5_files, target_length=segment_length, lazy_load=True)
    if len(dataset) == 0:
        raise ValueError("Dataset is empty")

    # 使用与训练相同的 seed 划分 train/val，然后只取 val 部分
    train_ratio = 0.9
    total = len(dataset)
    train_size = int(train_ratio * total)
    val_size = total - train_size

    generator = torch.Generator().manual_seed(seed)
    _, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)
    print(f"[Data] Total: {total}, Train: {train_size}, Val: {val_size}")

    # 如果指定了 num_samples, 则只取前 N 个样本
    if 0 < num_samples < len(val_dataset):
        val_dataset = Subset(val_dataset, list(range(num_samples)))
        print(f"[Data] Subsampled to {num_samples} validation samples")

    loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=num_workers,
    )
    return loader


# ═══════════════════════════════════════════════════════════════════════════════
#  Normalization (same as training)
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_batch(x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "none":
        return x
    if mode == "zscore":
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
        return (x - mean) / std
    if mode == "scale":
        denom = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
        return x / denom
    raise ValueError(f"Unknown normalize mode: {mode}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Evaluation routines
# ═══════════════════════════════════════════════════════════════════════════════

FORWARD_MODES = {
    "full": "forward",
    "low_only": "forward_low_only",
    "high_only": "forward_high_only",
}


@torch.no_grad()
def evaluate_mode(
    model: WaveletMambaDualStreamVQVAE,
    val_loader: DataLoader,
    mode: str,
    normalize: str = "none",
    device: torch.device = torch.device("cpu"),
    save_examples: int = 0,
) -> Tuple[Dict[str, float], Optional[Dict[str, np.ndarray]]]:
    """
    Run a single forward mode and collect reconstruction metrics.

    Args:
        model:         Pretrained model (already on device).
        val_loader:    Validation DataLoader.
        mode:          One of 'full', 'low_only', 'high_only'.
        normalize:     Normalization mode matching training ('none', 'zscore', 'scale').
        device:        Torch device.
        save_examples: Number of example reconstructions to save for visualization.

    Returns:
        (summary_dict, examples_dict_or_None)
    """
    forward_fn = getattr(model, FORWARD_MODES[mode])
    metrics = AblationMetrics()

    # For saving example signals
    saved_orig, saved_recon = [], []
    n_saved = 0

    pbar = tqdm(val_loader, desc=f"  [{mode:>10s}]", ncols=120)
    for batch in pbar:
        x = batch.float().to(device)  # [B, 1, L]
        x = normalize_batch(x, normalize)

        outputs = forward_fn(x)
        x_recon = outputs[0]        # reconstructed signal
        vq_loss_low = outputs[3]    # VQ loss (low stream)
        vq_loss_high = outputs[4]   # VQ loss (high stream)

        # compute metrics
        batch_mse = compute_mse(x, x_recon)
        batch_mae = compute_mae(x, x_recon)
        batch_snr = compute_snr(x, x_recon)
        batch_psnr = compute_psnr(x, x_recon)
        batch_pearson = compute_pearson(x, x_recon)
        batch_stft = compute_stft_error(x, x_recon)

        # VQ loss (scalar -> expand to batch)
        vq_total = vq_loss_low + vq_loss_high
        batch_vq = vq_total.expand(x.size(0))

        metrics.update({
            "mse": batch_mse,
            "mae": batch_mae,
            "snr": batch_snr,
            "psnr": batch_psnr,
            "pearson": batch_pearson,
            "stft_error": batch_stft,
            "vq_loss": batch_vq,
        })

        # Save a few examples for visualization
        if save_examples > 0 and n_saved < save_examples:
            n_to_save = min(save_examples - n_saved, x.size(0))
            saved_orig.append(x[:n_to_save].cpu().numpy())
            saved_recon.append(x_recon[:n_to_save].cpu().numpy())
            n_saved += n_to_save

        pbar.set_postfix(mse=f"{batch_mse.mean():.6f}", snr=f"{batch_snr.mean():.1f}dB")

    summary = metrics.summarize()
    summary["mode"] = mode

    examples = None
    if saved_orig:
        examples = {
            "original": np.concatenate(saved_orig, axis=0),
            "reconstructed": np.concatenate(saved_recon, axis=0),
        }

    return summary, examples


# ═══════════════════════════════════════════════════════════════════════════════
#  Visualization
# ═══════════════════════════════════════════════════════════════════════════════

def plot_comparison(
    examples: Dict[str, Dict[str, np.ndarray]],
    output_dir: str,
    n_plots: int = 5,
):
    """Generate comparison plots for full / low_only / high_only."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Warning] matplotlib not available, skipping plots.")
        return

    os.makedirs(output_dir, exist_ok=True)
    modes = list(examples.keys())
    n_avail = min(n_plots, min(ex["original"].shape[0] for ex in examples.values()))

    for i in range(n_avail):
        fig, axes = plt.subplots(len(modes) + 1, 1, figsize=(16, 3 * (len(modes) + 1)), sharex=True)

        # Original (from any mode, they're the same)
        orig = examples[modes[0]]["original"][i, 0, :]
        axes[0].plot(orig, color="black", linewidth=0.5)
        axes[0].set_title("Original Signal", fontsize=12)
        axes[0].set_ylabel("Amplitude")

        for j, mode in enumerate(modes):
            recon = examples[mode]["reconstructed"][i, 0, :]
            color = {"full": "blue", "low_only": "green", "high_only": "red"}.get(mode, "gray")
            label = {"full": "Full (Semantic+Texture)", "low_only": "Semantic Only (No Texture)",
                     "high_only": "Texture Only (No Semantic)"}.get(mode, mode)

            axes[j + 1].plot(orig, color="black", linewidth=0.3, alpha=0.4, label="Original")
            axes[j + 1].plot(recon, color=color, linewidth=0.5, alpha=0.8, label=label)
            axes[j + 1].set_title(label, fontsize=12)
            axes[j + 1].set_ylabel("Amplitude")
            axes[j + 1].legend(loc="upper right", fontsize=8)

        axes[-1].set_xlabel("Time (samples)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"ablation_comparison_sample_{i}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"[Plot] Saved {n_avail} comparison plots to {output_dir}")


def print_summary_table(results: Dict[str, Dict[str, float]]):
    """Print a formatted comparison table to console."""
    modes = list(results.keys())
    metrics_keys = ["mse_mean", "mae_mean", "snr_mean", "psnr_mean", "pearson_mean", "stft_error_mean", "vq_loss_mean"]
    display_names = ["MSE ↓", "MAE ↓", "SNR (dB) ↑", "PSNR (dB) ↑", "Pearson ↑", "STFT Error ↓", "VQ Loss ↓"]

    header_labels = {
        "full": "Full Model",
        "low_only": "Semantic Only\n(No Texture)",
        "high_only": "Texture Only\n(No Semantic)",
    }

    print("\n" + "=" * 90)
    print("  Texture Stream Ablation — Reconstruction Quality Comparison")
    print("=" * 90)

    # Header
    header = f"{'Metric':<20s}"
    for m in modes:
        header += f" | {header_labels.get(m, m):>20s}"
    print(header)
    print("-" * 90)

    for key, name in zip(metrics_keys, display_names):
        row = f"{name:<20s}"
        for m in modes:
            val = results[m].get(key, 0.0)
            if "snr" in key or "psnr" in key:
                row += f" | {val:>20.2f}"
            elif "pearson" in key:
                row += f" | {val:>20.4f}"
            else:
                row += f" | {val:>20.6f}"
        print(row)

    print("-" * 90)
    # Number of samples
    row = f"{'Num Samples':<20s}"
    for m in modes:
        row += f" | {results[m].get('num_samples', 0):>20d}"
    print(row)
    print("=" * 90)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Texture Stream Ablation Evaluation")

    # Model & checkpoint
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to WaveSem checkpoint (.pth)")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--cnn_compression_ratio", type=int, default=4)
    parser.add_argument("--wavelet", type=str, default="db4")
    parser.add_argument("--learnable_wavelet", action="store_true", default=True)

    # Data
    parser.add_argument("--data_path", type=str, required=True,
                        help="Directory containing *.h5 files")
    parser.add_argument("--segment_length", type=int, default=10240)
    parser.add_argument("--max_files", type=int, default=2,
                        help="Max H5 files to load (0=all)")
    parser.add_argument("--num_samples", type=int, default=5000,
                        help="Max validation samples to evaluate (0=all)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--normalize", type=str, default="none",
                        choices=["none", "zscore", "scale"])
    parser.add_argument("--seed", type=int, default=42)

    # Output
    parser.add_argument("--output_dir", type=str, default="outputs/texture_stream_ablation")
    parser.add_argument("--save_examples", type=int, default=10,
                        help="Number of example signals to save for visualization")
    parser.add_argument("--no_plot", action="store_true",
                        help="Skip generating comparison plots")

    # Device
    parser.add_argument("--device", type=str, default="cuda")

    # Modes to run
    parser.add_argument("--modes", type=str, default="full,low_only,high_only",
                        help="Comma-separated list of modes to evaluate")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load model ──
    model = load_model(
        checkpoint_path=args.checkpoint,
        device=device,
        hidden_dim=args.hidden_dim,
        cnn_compression_ratio=args.cnn_compression_ratio,
        wavelet=args.wavelet,
        learnable=args.learnable_wavelet,
    )

    # ── Load data ──
    val_loader = build_val_loader(
        data_path=args.data_path,
        max_files=args.max_files,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        segment_length=args.segment_length,
        num_samples=args.num_samples,
        seed=args.seed,
    )

    # ── Run ablation modes ──
    selected_modes = [m.strip() for m in args.modes.split(",")]
    all_results = {}
    all_examples = {}

    print(f"\n{'#' * 70}")
    print(f"  Texture Stream Ablation Experiment")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Data: {args.data_path} (max_files={args.max_files})")
    print(f"  Modes: {selected_modes}")
    print(f"{'#' * 70}\n")

    for mode in selected_modes:
        if mode not in FORWARD_MODES:
            print(f"[Warning] Unknown mode '{mode}', skipping.")
            continue

        print(f"\n▶ Evaluating mode: {mode}")
        t0 = time.time()

        summary, examples = evaluate_mode(
            model=model,
            val_loader=val_loader,
            mode=mode,
            normalize=args.normalize,
            device=device,
            save_examples=args.save_examples,
        )

        elapsed = time.time() - t0
        summary["eval_time_sec"] = elapsed
        all_results[mode] = summary

        if examples is not None:
            all_examples[mode] = examples

        print(f"  ✓ {mode}: MSE={summary['mse_mean']:.6f}, "
              f"SNR={summary['snr_mean']:.2f} dB, "
              f"Pearson={summary['pearson_mean']:.4f}, "
              f"STFT={summary['stft_error_mean']:.6f} "
              f"({elapsed:.1f}s)")

    # ── Print summary table ──
    print_summary_table(all_results)

    # ── Save results ──
    results_path = os.path.join(args.output_dir, "ablation_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[Saved] Results → {results_path}")

    # ── Save args ──
    args_path = os.path.join(args.output_dir, "eval_args.json")
    with open(args_path, "w") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    # ── Visualization ──
    if not args.no_plot and all_examples:
        plot_dir = os.path.join(args.output_dir, "plots")
        plot_comparison(all_examples, plot_dir, n_plots=min(5, args.save_examples))

    # ── Save example signals as .npz ──
    if all_examples:
        for mode, exs in all_examples.items():
            npz_path = os.path.join(args.output_dir, f"examples_{mode}.npz")
            np.savez_compressed(npz_path, **exs)
        print(f"[Saved] Example signals → {args.output_dir}/examples_*.npz")

    # ── Generate LaTeX-friendly table ──
    print("\n── LaTeX Table (for paper) ──")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Texture Stream Ablation: Reconstruction quality with and without the Texture Stream.}")
    print(r"\label{tab:texture_ablation}")
    print(r"\begin{tabular}{l c c c c c}")
    print(r"\toprule")
    print(r"Variant & MSE $\downarrow$ & SNR (dB) $\uparrow$ & Pearson $\uparrow$ & STFT Error $\downarrow$ & VQ Loss $\downarrow$ \\")
    print(r"\midrule")

    latex_names = {
        "full": r"WaveSem (Full)",
        "low_only": r"Semantic Only (No Texture)",
        "high_only": r"Texture Only (No Semantic)",
    }
    for mode in selected_modes:
        if mode not in all_results:
            continue
        r = all_results[mode]
        name = latex_names.get(mode, mode)
        print(f"{name} & {r['mse_mean']:.4f} & {r['snr_mean']:.2f} & "
              f"{r['pearson_mean']:.4f} & {r['stft_error_mean']:.4f} & {r['vq_loss_mean']:.4f} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    print("\n✅ Texture Stream ablation evaluation complete!")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
