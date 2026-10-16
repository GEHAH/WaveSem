"""
批量 GPU 推理与重构保存脚本。
基于 notebook predict_signal.ipynb 重写，适合大批量数据离线处理。

使用示例：
python predict_signal_batch.py \
    --data /path/to/chunks.npy \
    --checkpoint /owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain/outputs/wave_vqmodel_small_cnn_32/checkpoints/best_model.pth \
    --batch-size 64 \
    --device cuda \
    --out /tmp/recon_results.npz
"""

import argparse
import os
import sys
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# 确保可以找到模型定义
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pretrain"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from wave_vqmodel import WaveletMambaDualStreamVQVAE  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch reconstruct signals with WaveletMambaDualStreamVQVAE")
    parser.add_argument("--data", required=True, help="输入 .npy 文件，形状 [N, L] 或 [N, 1, L]")
    parser.add_argument("--checkpoint", required=True, help="模型 checkpoint 路径 (包含 model_state_dict)")
    parser.add_argument("--batch-size", type=int, default=32, help="推理 batch 大小")
    parser.add_argument("--device", default="cuda", help="'cuda' 或 'cpu'")
    parser.add_argument("--out", default="recon_results.npz", help="输出 .npz 路径，包含 orig/recon/mse")
    parser.add_argument("--hidden-dim", type=int, default=256, help="与训练一致的 hidden_dim")
    parser.add_argument("--compression-ratio", type=int, default=2, help="与训练一致的 cnn_compression_ratio")
    parser.add_argument("--wavelet", default="db4", help="与训练一致的 wavelet 名称")
    parser.add_argument("--learnable", action="store_true", help="若训练时开启了 learnable 滤波器则加上此项")
    return parser.parse_args()


def load_model(args: argparse.Namespace, device: torch.device) -> WaveletMambaDualStreamVQVAE:
    model = WaveletMambaDualStreamVQVAE(
        input_dim=1,
        hidden_dim=args.hidden_dim,
        cnn_compression_ratio=args.compression_ratio,
        wavelet=args.wavelet,
        learnable=args.learnable,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model


def prepare_data(path: str, device: torch.device) -> torch.Tensor:
    arr = np.load(path)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim == 2:
        arr = arr[:, None, :]  # -> [N, 1, L]
    if arr.ndim != 3:
        raise ValueError(f"Expected data shape [N, 1, L] or [N, L], got {arr.shape}")
    tensor = torch.from_numpy(arr).float().to(device)
    # 末尾补 56 个 0，保持与 notebook 流程一致
    tensor = F.pad(tensor, (0, 56))
    return tensor


@torch.no_grad()
def run_inference(
    model: WaveletMambaDualStreamVQVAE,
    data: torch.Tensor,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Args:
        model: 已加载权重的模型
        data: [N, 1, L] tensor on target device
        batch_size: 推理 batch 大小
    Returns:
        orig_np: 原始信号 (N, L)
        recon_np: 重构信号 (N, L)
        mse_np: 每条样本的 MSE (N,)
    """
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)
    orig_all, recon_all, mse_all = [], [], []

    for batch in tqdm(loader, desc="Reconstructing", ncols=100):
        batch = batch  # already on device
        recon, *_ = model(batch)
        recon = recon.squeeze(1)  # [B, L]
        orig = batch.squeeze(1)   # [B, L]

        mse = F.mse_loss(recon, orig, reduction="none").mean(dim=1)  # [B]

        orig_all.append(orig.detach().cpu())
        recon_all.append(recon.detach().cpu())
        mse_all.append(mse.detach().cpu())

    orig_np = torch.cat(orig_all, dim=0).numpy()
    recon_np = torch.cat(recon_all, dim=0).numpy()
    mse_np = torch.cat(mse_all, dim=0).numpy()
    return orig_np, recon_np, mse_np


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    print(f"Loading data from {args.data}")
    data = prepare_data(args.data, device)
    print(f"Data shape: {tuple(data.shape)}")

    print(f"Loading model from {args.checkpoint} to {device}")
    model = load_model(args, device)

    print("Running inference...")
    orig_np, recon_np, mse_np = run_inference(model, data, args.batch_size)

    # 仅保存前 5000 长度
    orig_np = orig_np[..., :5000]
    recon_np = recon_np[..., :5000]

    np.savez_compressed(
        args.out,
        original=orig_np,
        reconstructed=recon_np
    )

    print(f"Saved results to {args.out}")
    print(f"Mean MSE: {mse_np.mean():.6f}")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()

