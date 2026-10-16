#!/usr/bin/env python
"""
E6: 定量解耦分析 (回应 Reviewer3 R3.4 disentanglement)

Reviewer3 指出 "physically disentangled / zero-mean white noise" 仅有 t-SNE 与两个
手工 segment 的定性证据。本脚本给出**定量**指标, 在冻结的 WaveSem 上计算:

  1. Linear CKA(z_sem, z_tex)                 —— 两流表示的相似度 (低=更分离)
  2. Cross-stream linear predictability R²     —— z_sem->z_tex 与 z_tex->z_sem 的可预测性 (低=更独立)
  3. Band energy ratio (Parseval split)        —— 低频/高频 DWT 系数能量占比
  4. Texture residual spectral flatness (SFM)  —— 纹理流重构信号的谱平坦度 (接近1=更白)
  5. Full-recon residual autocorrelation       —— 残差 (x - x_hat) 的低阶自相关 (接近0=更像噪声)
  6. MI estimate (histogram, PC1)              —— 两流主成分的互信息估计 (仅作参考)

输出: JSON + 控制台表 + LaTeX 片段。

Usage:
  python eval_disentanglement_quant.py \
      --checkpoint outputs/wave_vqmodel/checkpoints/best_model.pth \
      --data_path  raw_data/temp_signal_segments_10240 \
      --output_dir outputs/disentanglement_quant \
      --num_samples 5000 --batch_size 64
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from h5_data_process import HDF5SignalDataset  # noqa: E402
from wave_vqmodel import WaveletMambaDualStreamVQVAE  # noqa: E402

import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────── 指标实现 ───────────────────────────

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two [n, d] feature matrices. Returns scalar in [0, 1]."""
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    hsic = np.linalg.norm(Y.T @ X) ** 2
    nx = np.linalg.norm(X.T @ X)
    ny = np.linalg.norm(Y.T @ Y)
    return float(hsic / (nx * ny + 1e-12))


def cross_stream_r2(src: np.ndarray, dst: np.ndarray, alpha: float = 1.0, seed: int = 42) -> float:
    """Ridge 回归 src->dst 的 held-out R² (多输出取平均)。低 R² 支持两流线性不可预测。"""
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score

    n = src.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_tr = int(0.8 * n)
    tr, te = idx[:n_tr], idx[n_tr:]
    reg = Ridge(alpha=alpha)
    reg.fit(src[tr], dst[tr])
    pred = reg.predict(src[te])
    return float(r2_score(dst[te], pred, multioutput="variance_weighted"))


def per_sample_normalize(M: np.ndarray) -> np.ndarray:
    """逐样本 (逐行) 去均值+除标准差, 移除每段共享的全局幅度/水平分量。

    诊断用: 若归一化后 CKA/R² 显著下降, 说明池化特征的高纠缠主要来自共享的
    per-segment 信号水平, 而非细结构纠缠。
    """
    m = M - M.mean(axis=1, keepdims=True)
    s = M.std(axis=1, keepdims=True) + 1e-8
    return m / s


def token_mutual_info(codes_a: np.ndarray, codes_b: np.ndarray) -> Dict[str, float]:
    """两流 RVQ 首层 token 索引之间的互信息 (在所有 (sample,timestep) 位置上)。

    这是**离散 token 层面**的依赖度量, 比连续池化特征更贴近论文关心的
    "两套码本是否编码相同信息"。返回原始 MI(nats) 与 归一化 MI(NMI∈[0,1])。
    """
    from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
    a = codes_a.reshape(-1)
    b = codes_b.reshape(-1)
    return {
        "token_mi_nats": float(mutual_info_score(a, b)),
        "token_nmi": float(normalized_mutual_info_score(a, b)),
    }


def _first_layer_codes(idx: torch.Tensor, seq_len: int) -> np.ndarray:
    """从 RVQ indices 提取首层码 [B, seq_len] (兼容 [B,N,Q]/[B,Q,N]/[B,N])。"""
    idx = idx.detach().cpu()
    if idx.dim() == 2:
        out = idx
    elif idx.dim() == 3:
        b, a1, a2 = idx.shape
        if a1 == seq_len:      # [B, N, Q]
            out = idx[:, :, 0]
        elif a2 == seq_len:    # [B, Q, N]
            out = idx[:, 0, :]
        else:
            out = idx.reshape(b, -1)[:, :seq_len]
    else:
        out = idx.reshape(idx.shape[0], -1)[:, :seq_len]
    return out.numpy()


def spectral_flatness(signal: np.ndarray, eps: float = 1e-10) -> float:
    """谱平坦度 (geo-mean / arith-mean of PSD). signal: [N, L] -> 标量 (样本平均)。"""
    spec = np.abs(np.fft.rfft(signal, axis=-1)) ** 2 + eps
    geo = np.exp(np.mean(np.log(spec), axis=-1))
    ari = np.mean(spec, axis=-1)
    return float(np.mean(geo / ari))


def residual_autocorr(residual: np.ndarray, max_lag: int = 10) -> Dict[str, float]:
    """残差自相关: 返回 lag1..max_lag 的平均 |autocorr|。residual: [N, L]。"""
    r = residual - residual.mean(axis=-1, keepdims=True)
    var = (r ** 2).mean(axis=-1) + 1e-10
    acs = []
    for lag in range(1, max_lag + 1):
        ac = (r[:, :-lag] * r[:, lag:]).mean(axis=-1) / var
        acs.append(np.mean(np.abs(ac)))
    return {
        "mean_abs_autocorr_lag1_10": float(np.mean(acs)),
        "abs_autocorr_lag1": float(acs[0]),
    }


def mi_pc1_hist(X: np.ndarray, Y: np.ndarray, bins: int = 32) -> float:
    """两流 PC1 之间的直方图互信息估计 (nats)。仅作参考, 非严格 MI。"""
    def pc1(M):
        Mc = M - M.mean(0, keepdims=True)
        # 幂迭代取第一主成分投影
        u, s, vt = np.linalg.svd(Mc, full_matrices=False)
        return Mc @ vt[0]
    x, y = pc1(X), pc1(Y)
    c_xy, _, _ = np.histogram2d(x, y, bins=bins)
    p_xy = c_xy / c_xy.sum()
    p_x = p_xy.sum(1, keepdims=True)
    p_y = p_xy.sum(0, keepdims=True)
    nz = p_xy > 0
    mi = np.sum(p_xy[nz] * np.log(p_xy[nz] / (p_x @ p_y)[nz] + 1e-12))
    return float(max(mi, 0.0))


# ─────────────────────────── 特征提取 ───────────────────────────

@torch.no_grad()
def extract_streams(model, loader, device, max_samples: int, max_positions: int = 200000):
    """返回池化特征 z_sem/z_tex [N,C]、能量/残差数组、RVQ 首层 token 索引、
    以及 (上限 max_positions 的) 逐时间步原始特征 (用于 per-timestep CKA)。"""
    zs, zt = [], []
    band_low_e, band_high_e = [], []
    tex_only_recon, full_residual = [], []
    codes_low, codes_high = [], []
    pos_sem, pos_tex = [], []
    n_pos = 0
    n = 0
    for batch in tqdm(loader, desc="  extract", ncols=100):
        x = batch.float().to(device)                 # [B,1,L]
        feat_cnn = model.encoder(x)
        feat_low_1, feat_high = model.multi_level_dwt(feat_cnn)
        feat_low_2, idx_low, _ = model.low_stream(feat_low_1)
        feat_high_1, idx_high, _ = model.high_stream(
            feat_high[3], feat_high[2], feat_high[1], feat_high[0]
        )
        # 时间维平均池化 -> [B, C]
        zs.append(feat_low_2.mean(dim=-1).cpu().numpy())
        zt.append(feat_high_1.mean(dim=-1).cpu().numpy())
        # RVQ 首层 token 索引 (同一时间分辨率 N)
        n_low = feat_low_2.shape[-1]
        n_high = feat_high_1.shape[-1]
        cl = _first_layer_codes(idx_low, n_low)
        ch = _first_layer_codes(idx_high, n_high)
        m = min(cl.shape[1], ch.shape[1])            # 对齐时间步
        codes_low.append(cl[:, :m])
        codes_high.append(ch[:, :m])
        # 逐时间步原始特征 (capped): [B,C,N] -> [B*N, C]
        if n_pos < max_positions:
            ps = feat_low_2.permute(0, 2, 1).reshape(-1, feat_low_2.shape[1]).cpu().numpy()
            pt = feat_high_1.permute(0, 2, 1).reshape(-1, feat_high_1.shape[1]).cpu().numpy()
            take = min(len(ps), max_positions - n_pos)
            pos_sem.append(ps[:take]); pos_tex.append(pt[:take])
            n_pos += take
        # Parseval 能量: 低频系数 vs 所有高频系数
        band_low_e.append((feat_low_1 ** 2).sum(dim=(1, 2)).cpu().numpy())
        band_high_e.append(sum((h ** 2).sum(dim=(1, 2)) for h in feat_high).cpu().numpy())
        # 纹理流重构 + 全重构残差
        x_full = model.forward(x)[0]
        x_tex = model.forward_high_only(x)[0]
        full_residual.append((x - x_full).squeeze(1).cpu().numpy())
        tex_only_recon.append(x_tex.squeeze(1).cpu().numpy())

        n += x.size(0)
        if max_samples and n >= max_samples:
            break

    return {
        "z_sem": np.concatenate(zs, 0),
        "z_tex": np.concatenate(zt, 0),
        "band_low_e": np.concatenate(band_low_e, 0),
        "band_high_e": np.concatenate(band_high_e, 0),
        "tex_only_recon": np.concatenate(tex_only_recon, 0),
        "full_residual": np.concatenate(full_residual, 0),
        "codes_low": np.concatenate(codes_low, 0),
        "codes_high": np.concatenate(codes_high, 0),
        "pos_sem": np.concatenate(pos_sem, 0),
        "pos_tex": np.concatenate(pos_tex, 0),
    }


def build_loader(data_path, max_files, batch_size, num_workers, seg_len, num_samples, seed):
    files = sorted(glob.glob(os.path.join(data_path, "*.h5")))
    if max_files > 0:
        files = files[:max_files]
    if not files:
        raise ValueError(f"No *.h5 in {data_path}")
    ds = HDF5SignalDataset(files, target_length=seg_len, lazy_load=True)
    total = len(ds)
    tr = int(0.9 * total)
    g = torch.Generator().manual_seed(seed)
    _, val = torch.utils.data.random_split(ds, [tr, total - tr], generator=g)
    if 0 < num_samples < len(val):
        val = Subset(val, list(range(num_samples)))
    return DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


def main():
    p = argparse.ArgumentParser(description="E6: quantitative disentanglement metrics")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_path", required=True)
    p.add_argument("--output_dir", default="outputs/disentanglement_quant")
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--cnn_compression_ratio", type=int, default=4)
    p.add_argument("--wavelet", default="db4")
    p.add_argument("--segment_length", type=int, default=10240)
    p.add_argument("--max_files", type=int, default=2)
    p.add_argument("--num_samples", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    model = WaveletMambaDualStreamVQVAE(
        input_dim=1, hidden_dim=args.hidden_dim,
        cnn_compression_ratio=args.cnn_compression_ratio,
        wavelet=args.wavelet, learnable=True,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[Model] loaded {args.checkpoint}")

    loader = build_loader(args.data_path, args.max_files, args.batch_size,
                          args.num_workers, args.segment_length, args.num_samples, args.seed)

    feats = extract_streams(model, loader, device, args.num_samples)
    z_sem, z_tex = feats["z_sem"], feats["z_tex"]
    # 逐样本归一化版本 (去共享全局幅度)
    zs_n, zt_n = per_sample_normalize(z_sem), per_sample_normalize(z_tex)
    # 逐时间步原始特征
    ps, pt = feats["pos_sem"], feats["pos_tex"]

    results = {
        "num_samples": int(z_sem.shape[0]),
        "feature_dim": int(z_sem.shape[1]),
        # ── 池化特征 (原始, 可能被共享全局幅度抬高) ──
        "cka_sem_tex": linear_cka(z_sem, z_tex),
        "r2_sem_to_tex": cross_stream_r2(z_sem, z_tex, seed=args.seed),
        "r2_tex_to_sem": cross_stream_r2(z_tex, z_sem, seed=args.seed),
        # ── 变体1: 逐样本归一化 (移除共享全局幅度分量) ──
        "cka_sem_tex_norm": linear_cka(zs_n, zt_n),
        "r2_sem_to_tex_norm": cross_stream_r2(zs_n, zt_n, seed=args.seed),
        "r2_tex_to_sem_norm": cross_stream_r2(zt_n, zs_n, seed=args.seed),
        # ── 变体2: 逐时间步 (不池化) CKA ──
        "cka_sem_tex_pertimestep": linear_cka(ps, pt),
        "num_positions": int(ps.shape[0]),
        # ── 变体3: RVQ 首层 token 索引级 MI ──
        **token_mutual_info(feats["codes_low"], feats["codes_high"]),
        # ── 能量 / 谱 / 残差 ──
        "band_low_energy_frac": float(
            feats["band_low_e"].mean() / (feats["band_low_e"].mean() + feats["band_high_e"].mean())
        ),
        "texture_recon_spectral_flatness": spectral_flatness(feats["tex_only_recon"]),
        **{f"residual_{k}": v for k, v in residual_autocorr(feats["full_residual"]).items()},
        "residual_spectral_flatness": spectral_flatness(feats["full_residual"]),
        "mi_pc1_hist_nats": mi_pc1_hist(z_sem, z_tex),
    }

    out = os.path.join(args.output_dir, "disentanglement_metrics.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 控制台表
    print("\n" + "=" * 60)
    print("  E6: Quantitative Disentanglement Metrics")
    print("=" * 60)
    for k, v in results.items():
        print(f"  {k:<38s} {v:.6f}" if isinstance(v, float) else f"  {k:<38s} {v}")
    print("=" * 60)
    print(f"[Saved] {out}")

    # LaTeX 片段
    print("\n── LaTeX (Appendix) ──")
    print(r"\begin{tabular}{lc}\toprule Metric & Value \\\midrule")
    print(rf"Linear CKA($z_{{sem}}, z_{{tex}}$), pooled $\downarrow$ & {results['cka_sem_tex']:.3f} \\")
    print(rf"Linear CKA, per-sample normalized $\downarrow$ & {results['cka_sem_tex_norm']:.3f} \\")
    print(rf"Linear CKA, per-timestep $\downarrow$ & {results['cka_sem_tex_pertimestep']:.3f} \\")
    print(rf"Cross-stream $R^2$ (sem$\to$tex) $\downarrow$ & {results['r2_sem_to_tex']:.3f} \\")
    print(rf"Cross-stream $R^2$ (tex$\to$sem) $\downarrow$ & {results['r2_tex_to_sem']:.3f} \\")
    print(rf"Cross-stream $R^2$ norm (sem$\to$tex) $\downarrow$ & {results['r2_sem_to_tex_norm']:.3f} \\")
    print(rf"Cross-stream $R^2$ norm (tex$\to$sem) $\downarrow$ & {results['r2_tex_to_sem_norm']:.3f} \\")
    print(rf"Token-index NMI (RVQ $Q_1$) $\downarrow$ & {results['token_nmi']:.3f} \\")
    print(rf"Low-band energy fraction & {results['band_low_energy_frac']:.3f} \\")
    print(rf"Texture recon spectral flatness $\uparrow$ & {results['texture_recon_spectral_flatness']:.3f} \\")
    print(rf"Residual autocorr (lag1--10) $\downarrow$ & {results['residual_mean_abs_autocorr_lag1_10']:.3f} \\")
    print(rf"Residual spectral flatness $\uparrow$ & {results['residual_spectral_flatness']:.3f} \\")
    print(r"\bottomrule\end{tabular}")


if __name__ == "__main__":
    main()
