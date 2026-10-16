#!/usr/bin/env python
"""
Compute GFLOPs for basecalling models.
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
from torch.utils.data import DataLoader, Dataset

BASECALLING_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "basecalling"))
sys.path.append(os.path.join(BASECALLING_ROOT, "src"))
sys.path.append(os.path.join(BASECALLING_ROOT, "models"))

from bonito.vq_model import (  # noqa: E402
    VQVAEModel_wavesem_basecalling,
    SingleCodebookVQVAEModel_stride_basecalling,
    BasecallingModel,
    SoundStream_basecalling,
)



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


class DummyDataset(Dataset):
    def __init__(self, length: int = 8, signal_length: int = 10240):
        self.length = length
        self.signal_length = signal_length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = torch.randn(self.signal_length, dtype=torch.float32)
        y = torch.zeros(self.signal_length, dtype=torch.long)
        return {"x": x, "y": y}


class BasecallingHeadWrapper(nn.Module):
    def __init__(self, basecalling_model: nn.Module):
        super().__init__()
        self.decoder_basecalling = basecalling_model.decoder_basecalling
        self.dropout = basecalling_model.dropout
        self.bonito_encoder = basecalling_model.bonito_encoder
        self.decoder = basecalling_model.decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.decoder_basecalling(x)
        x = x.permute(2, 0, 1).contiguous()
        x = self.dropout(x)
        x = self.bonito_encoder(x)
        x = self.decoder(x)
        return x

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


def build_basecalling_model(args, device, loader):
    if args.model == "wavesem":
        model = VQVAEModel_wavesem_basecalling(
            vqvae_model_path=args.vqvae_model_path,
            cnn_compression_ratio=args.cnn_compression_ratio,
            decoder_stride=args.decoder_stride,
            device=device,
            dataloader_train=loader,
            dataloader_validation=loader,
            use_amp=False,
        )
    elif args.model == "singlevq":
        model = SingleCodebookVQVAEModel_stride_basecalling(
            vqvae_model_path=args.vqvae_model_path,
            strides=args.strides,
            decoder_stride=args.decoder_stride,
            device=device,
            dataloader_train=loader,
            dataloader_validation=loader,
            use_amp=False,
        )
    elif args.model == "soundstream":
        model = SoundStream_basecalling(
            vqvae_model_path=args.vqvae_model_path,
            decoder_stride=args.decoder_stride,
            ss_channels=args.ss_channels,
            ss_strides=args.ss_strides,
            ss_channel_mults=args.ss_channel_mults,
            ss_codebook_dim=args.ss_codebook_dim,
            ss_codebook_size=args.ss_codebook_size,
            ss_rq_num_quantizers=args.ss_rq_num_quantizers,
            ss_rq_groups=args.ss_rq_groups,
            ss_use_local_attn=not args.ss_disable_local_attn,
            ss_attn_window_size=args.ss_attn_window_size,
            ss_attn_dim_head=args.ss_attn_dim_head,
            ss_attn_heads=args.ss_attn_heads,
            ss_attn_depth=args.ss_attn_depth,
            ss_target_sample_hz=args.ss_target_sample_hz,
            device=device,
            dataloader_train=loader,
            dataloader_validation=loader,
            use_amp=False,
        )
    elif args.model == "basecalling":
        model = BasecallingModel(
            vqvae_model_path=args.vqvae_model_path,
            decoder_stride=args.decoder_stride,
            device=device,
            dataloader_train=loader,
            dataloader_validation=loader,
            use_amp=False,
        )
    else:
        raise SystemExit(f"Unknown model: {args.model}")

    model = model.to(device)
    if args.basecalling_checkpoint:
        ckpt = torch.load(args.basecalling_checkpoint, map_location="cpu")
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)
    model.eval()
    return model


def main() -> None:
    p = argparse.ArgumentParser(description="Compute GFLOPs for basecalling models")
    p.add_argument("--model", type=str, default="wavesem",
                   choices=["wavesem", "singlevq", "soundstream", "basecalling"])
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--segment_length", type=int, default=10240)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--output_json", type=str, default=None)
    p.add_argument("--prefer_thop", action="store_true", help="Prefer thop (no torchscript) over fvcore.")
    p.add_argument("--vqvae_model_path", type=str, required=True, help="Pretrained encoder checkpoint")
    p.add_argument("--basecalling_checkpoint", type=str, default=None, help="Fine-tuned basecalling checkpoint")
    p.add_argument("--decoder_stride", type=str, default="[2,2,2]")
    p.add_argument("--cnn_compression_ratio", type=int, default=2)
    p.add_argument("--strides", type=parse_int_list, default=parse_int_list("2,2,2,2,2,2"))

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

    args = p.parse_args()
    if isinstance(args.decoder_stride, str):
        try:
            import ast
            args.decoder_stride = ast.literal_eval(args.decoder_stride)
        except Exception:
            args.decoder_stride = [int(x.strip()) for x in args.decoder_stride.strip("[]").split(",") if x.strip()]
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    x = torch.randn(args.batch_size, 1, args.segment_length, device=device)

    dummy_loader = DataLoader(DummyDataset(length=4, signal_length=args.segment_length), batch_size=1)
    model = build_basecalling_model(args, device, dummy_loader)

    if args.model == "basecalling":
        gflops, method, gmacs = compute_gflops(model, (x,), prefer_thop=args.prefer_thop)
        results = {
            "model": args.model,
            "gflops": gflops,
            "gmacs": gmacs,
            "method": method,
            "batch_size": args.batch_size,
            "segment_length": args.segment_length,
            "component": "full",
        }
    else:
        # compute GFLOPs after codebook encoding (basecalling head only)
        with torch.no_grad():
            if args.model == "wavesem":
                feat, _vq = model.vqvae_model(x)
            elif args.model == "singlevq":
                feat, _vq = model.vqvae_model(x)
            elif args.model == "soundstream":
                feat, _vq = model.soundstream_encoder(x)
            else:
                raise SystemExit(f"Unknown model: {args.model}")
        head = BasecallingHeadWrapper(model).to(device)
        gflops, method, gmacs = compute_gflops(head, (feat,), prefer_thop=args.prefer_thop)
        results = {
            "model": args.model,
            "gflops": gflops,
            "gmacs": gmacs,
            "method": method,
            "batch_size": args.batch_size,
            "segment_length": args.segment_length,
            "component": "post_codebook_head",
        }

    print(json.dumps(results, indent=2, ensure_ascii=False))
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

