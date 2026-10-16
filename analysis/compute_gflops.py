#!/usr/bin/env python
"""
Compute GFLOPs for SoundStream / WaveSem / SingleCodebookVQVAE / RCodebookVQVAE.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Iterable, List

import torch
import torch.nn as nn
_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(_PKG, "basecalling", "src"))
_AUDIO_LM_ROOT = os.path.join(_PKG, "pretrain", "soundstream", "audiolm-pytorch-main")
if _AUDIO_LM_ROOT not in sys.path:
    sys.path.append(_AUDIO_LM_ROOT)

from wave_vqmodel import WaveletMambaDualStreamVQVAE  # noqa: E402
from baseline_models import SingleCodebookVQVAE_stride, RCodebookVQVAE_stride  # noqa: E402
from audiolm_pytorch import SoundStream  # noqa: E402



def parse_int_list(value: str) -> List[int]:
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    if not value:
        return []
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def prod(values: Iterable[int]) -> int:
    out = 1
    for v in values:
        out *= int(v)
    return out


class SoundStreamWrapper(nn.Module):
    def __init__(self, model: SoundStream):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x, return_recons_only=True)


class SoundStreamEncoderWrapper(nn.Module):
    def __init__(self, model: SoundStream):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model.encoder(x)
        x = x.transpose(1, 2)
        if self.model.encoder_attn is not None:
            x = self.model.encoder_attn(x)
        if not self.model.use_finite_scalar_quantizer:
            x, _indices, _commit_loss = self.model.rq(x)
        else:
            x, _indices = self.model.rq(x)
        return x


class WaveSemEncoderWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat_cnn = self.model.encoder(x)
        feat_low_1, feat_high = self.model.multi_level_dwt(feat_cnn)
        feat_low_2, _indices_low, _vq_loss_low = self.model.low_stream(feat_low_1)
        # feat_high_1, _indices_high, _vq_loss_high = self.model.high_stream(
        #     feat_high[3], feat_high[2], feat_high[1], feat_high[0]
        # )
        return feat_low_2


class SingleVQEncoderWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_b_d_t = self.model.encoder(x)
        z_b_t_d = z_b_d_t.transpose(1, 2)
        quant_b_t_d, _indices_b_t_nq, _vq_loss = self.model.vq(z_b_t_d)
        return quant_b_t_d


class RVQEncoderWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_b_d_t = self.model.encoder(x)
        z_b_t_d = z_b_d_t.transpose(1, 2)
        quant_b_t_d, _indices_b_t_nq, _vq_loss = self.model.vq(z_b_t_d)
        return quant_b_t_d


def compute_gflops(model: nn.Module, inputs: tuple[torch.Tensor, ...], prefer_thop: bool = True) -> tuple[float, str, float | None]:
    """
    Returns (gflops, method, gmacs_if_available).
    thop reports MACs -> we convert to FLOPs as 2 * MACs.
    """
    model.eval()

    if prefer_thop:
        try:
            from thop import profile as thop_profile  # type: ignore
        except Exception:
            thop_profile = None
        if thop_profile is not None:
            with torch.no_grad():
                macs, _params = thop_profile(model, inputs=inputs, verbose=False)
            gmacs = float(macs) / 1e9
            gflops = gmacs * 2.0
            return gflops, "thop", gmacs

    # fallback to fvcore
    # try:
    from fvcore.nn import FlopCountAnalysis
    # except Exception as e:
    #     raise SystemExit(
    #         "Missing dependency: thop or fvcore.\n"
    #         "Install one of: pip install thop  OR  pip install fvcore\n"
    #     ) from e

    with torch.no_grad():
        flops = FlopCountAnalysis(model, inputs)
        total_flops = flops.total()
    return float(total_flops) / 1e9, "fvcore", None


def build_soundstream(args, wrap: bool = True) -> nn.Module:
    if SoundStream is None:
        raise SystemExit("SoundStream import failed. Check audiolm-pytorch path.")
    model = SoundStream(
        channels=args.ss_channels,
        strides=tuple(args.ss_strides),
        channel_mults=tuple(args.ss_channel_mults),
        codebook_dim=args.ss_codebook_dim,
        codebook_size=args.ss_codebook_size,
        rq_num_quantizers=args.ss_rq_num_quantizers,
        rq_groups=args.ss_rq_groups,
        rq_commitment_weight=args.ss_rq_commitment_weight,
        rq_ema_decay=args.ss_rq_ema_decay,
        target_sample_hz=args.ss_target_sample_hz,
        recon_loss_weight=1.0,
        multi_spectral_recon_loss_weight=0.0,
        adversarial_loss_weight=0.0,
        feature_loss_weight=0.0,
        use_local_attn=not args.ss_disable_local_attn,
        attn_window_size=args.ss_attn_window_size,
        attn_dim_head=args.ss_attn_dim_head,
        attn_heads=args.ss_attn_heads,
        attn_depth=args.ss_attn_depth,
        input_channels=1,
    )
    return SoundStreamWrapper(model) if wrap else model


def build_wavesem(args) -> nn.Module:
    return WaveletMambaDualStreamVQVAE(
        input_dim=args.ws_input_dim,
        hidden_dim=args.ws_hidden_dim,
        cnn_compression_ratio=args.ws_cnn_compression_ratio,
        wavelet=args.ws_wavelet,
        learnable=args.ws_learnable_wavelet,
    )


def build_singlevq(args) -> nn.Module:
    strides = args.sv_strides
    compression_ratio = args.sv_compression_ratio
    if compression_ratio is None:
        compression_ratio = prod(strides) if strides else 1
    return SingleCodebookVQVAE_stride(
        input_dim=args.sv_input_dim,
        hidden_dim=args.sv_hidden_dim,
        compression_ratio=compression_ratio,
        codebook_size=args.sv_codebook_size,
        strides=strides,
    )


def build_rvq(args) -> nn.Module:
    strides = args.rv_strides
    compression_ratio = args.rv_compression_ratio
    if compression_ratio is None:
        compression_ratio = prod(strides) if strides else 1
    return RCodebookVQVAE_stride(
        input_dim=args.rv_input_dim,
        hidden_dim=args.rv_hidden_dim,
        compression_ratio=compression_ratio,
        codebook_size=args.rv_codebook_size,
        strides=strides,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Compute GFLOPs for multiple model components")
    p.add_argument("--models", type=str, default="soundstream,wavesem,singlevq,rvq",
                   help="Comma-separated: soundstream,wavesem,singlevq,rvq")
    p.add_argument(
        "--component",
        type=str,
        default="encoder",
        choices=["encoder", "full"],
        help="Compute GFLOPs for encoder or full model",
    )
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--segment_length", type=int, default=10240)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--output_json", type=str, default=None)
    p.add_argument("--prefer_thop", action="store_true", help="Prefer thop (no torchscript) over fvcore.")

    # SoundStream
    p.add_argument("--ss_channels", type=int, default=32)
    p.add_argument("--ss_strides", type=parse_int_list, default=parse_int_list("2,2,2,2,2,2"))
    p.add_argument("--ss_channel_mults", type=parse_int_list, default=parse_int_list("2,4,8,8,16,16"))
    p.add_argument("--ss_codebook_dim", type=int, default=512)
    p.add_argument("--ss_codebook_size", type=int, default=1024)
    p.add_argument("--ss_rq_num_quantizers", type=int, default=8)
    p.add_argument("--ss_rq_groups", type=int, default=1)
    p.add_argument("--ss_rq_commitment_weight", type=float, default=1.0)
    p.add_argument("--ss_rq_ema_decay", type=float, default=0.95)
    p.add_argument("--ss_target_sample_hz", type=int, default=16000)
    p.add_argument("--ss_disable_local_attn", action="store_true")
    p.add_argument("--ss_attn_window_size", type=int, default=128)
    p.add_argument("--ss_attn_dim_head", type=int, default=64)
    p.add_argument("--ss_attn_heads", type=int, default=8)
    p.add_argument("--ss_attn_depth", type=int, default=1)

    # WaveSem
    p.add_argument("--ws_input_dim", type=int, default=1)
    p.add_argument("--ws_hidden_dim", type=int, default=256)
    p.add_argument("--ws_cnn_compression_ratio", type=int, default=2)
    p.add_argument("--ws_wavelet", type=str, default="db4")
    p.add_argument("--ws_learnable_wavelet", action="store_true")

    # Single codebook VQ-VAE
    p.add_argument("--sv_input_dim", type=int, default=1)
    p.add_argument("--sv_hidden_dim", type=int, default=256)
    p.add_argument("--sv_codebook_size", type=int, default=512)
    p.add_argument("--sv_strides", type=parse_int_list, default=parse_int_list("2,2,2,2"))
    p.add_argument("--sv_compression_ratio", type=int, default=64)

    # R-codebook VQ-VAE
    p.add_argument("--rv_input_dim", type=int, default=1)
    p.add_argument("--rv_hidden_dim", type=int, default=256)
    p.add_argument("--rv_codebook_size", type=int, default=512)
    p.add_argument("--rv_strides", type=parse_int_list, default=parse_int_list("2,2,2,2,2"))
    p.add_argument("--rv_compression_ratio", type=int, default=64)

    args = p.parse_args()
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    x = torch.randn(args.batch_size, 1, args.segment_length, device=device)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    results = {}

    for name in models:
        if name == "soundstream":
            if args.component == "encoder":
                model = SoundStreamEncoderWrapper(build_soundstream(args, wrap=False)).to(device)
            else:
                model = build_soundstream(args, wrap=True).to(device)
        elif name == "wavesem":
            base_model = build_wavesem(args).to(device)
            model = WaveSemEncoderWrapper(base_model) if args.component == "encoder" else base_model
        elif name == "singlevq":
            base_model = build_singlevq(args).to(device)
            model = SingleVQEncoderWrapper(base_model) if args.component == "encoder" else base_model
        elif name == "rvq":
            base_model = build_rvq(args).to(device)
            model = RVQEncoderWrapper(base_model) if args.component == "encoder" else base_model
        else:
            raise SystemExit(f"Unknown model name: {name}")

        gflops, method, gmacs = compute_gflops(model, (x,), prefer_thop=args.prefer_thop)

        results[name] = {
            "gflops": gflops,
            "gmacs": gmacs,
            "method": method,
            "batch_size": args.batch_size,
            "segment_length": args.segment_length,
            "component": args.component,
        }

    print(json.dumps(results, indent=2, ensure_ascii=False))
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

