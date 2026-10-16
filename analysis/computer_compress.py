#!/usr/bin/env python
import sys
import os
import math
import json
import argparse
import glob
import gzip
from typing import Iterable, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import zstandard as zstd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pretrain")))
from h5_data_process import HDF5SignalDataset  # noqa: E402
from wave_vqmodel import WaveletMambaDualStreamVQVAE, str_to_list  # noqa: E402

try:
    from accelerate import Accelerator
except ImportError as e:
    raise SystemExit(
        "Missing dependency: accelerate.\n"
        "Install with: pip install accelerate\n"
        "Then launch with: accelerate launch computer_compress.py ..."
    ) from e


def _normalize_batch(x: torch.Tensor, mode: str) -> torch.Tensor:
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


def _toggle_grad(modules: Iterable[torch.nn.Module], requires_grad: bool) -> None:
    if modules is None:
        return
    if isinstance(modules, torch.nn.Module):
        modules = [modules]
    for module in modules:
        if module is None:
            continue
        for p in module.parameters():
            p.requires_grad = requires_grad


def build_dataloader(args) -> DataLoader:
    h5_files = sorted(glob.glob(os.path.join(args.data_path, "*.h5")))
    if args.max_files > 0:
        h5_files = h5_files[: args.max_files]
    if len(h5_files) == 0:
        raise ValueError(f"No *.h5 files found in {args.data_path}")

    dataset = HDF5SignalDataset(h5_files, target_length=args.segment_length, lazy_load=True)
    if len(dataset) == 0:
        raise ValueError("Dataset is empty")

    def _worker_init_fn(_worker_id: int) -> None:
        if not args.warmup_h5_in_workers:
            return
        try:
            if hasattr(dataset, "warmup_open_files"):
                dataset.warmup_open_files()
        except Exception:
            pass

    common_kwargs = dict(
        pin_memory=True,
        num_workers=args.num_workers,
        worker_init_fn=_worker_init_fn if args.num_workers > 0 else None,
    )
    if args.num_workers > 0:
        common_kwargs.update(dict(prefetch_factor=args.prefetch_factor, persistent_workers=True))

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        **common_kwargs,
    )


def build_model(args) -> WaveletMambaDualStreamVQVAE:
    return WaveletMambaDualStreamVQVAE(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        cnn_compression_ratio=args.cnn_compression_ratio,
        wavelet=args.wavelet,
        learnable=args.learnable_wavelet,
    )


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, accelerator: Accelerator) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    unwrapped = accelerator.unwrap_model(model)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        unwrapped.load_state_dict(ckpt["model_state_dict"], strict=False)
        return
    if isinstance(ckpt, dict):
        unwrapped.load_state_dict(ckpt, strict=False)
        return
    raise ValueError(f"Unrecognized checkpoint format: {ckpt_path}")


def get_codebook_size(vq_layer: torch.nn.Module) -> Optional[int]:
    if hasattr(vq_layer, "codebook_size"):
        try:
            return int(vq_layer.codebook_size)
        except Exception:
            pass
    for name, tensor in vq_layer.state_dict().items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if "codebook" in name or "embed" in name:
            if tensor.ndim >= 2:
                return int(tensor.shape[-2])
    return None


def codebook_bytes(vq_layer: torch.nn.Module) -> int:
    total = 0
    for name, tensor in vq_layer.state_dict().items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if "codebook" in name or "embed" in name:
            total += tensor.numel() * tensor.element_size()
    return total


def bitpack_bytes(numel: int, codebook_size: int) -> int:
    if codebook_size <= 1 or numel <= 0:
        return 0
    bits = int(math.ceil(math.log2(codebook_size)))
    return (numel * bits + 7) // 8


def infer_and_compress(args) -> dict:
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    dataloader = build_dataloader(args)
    model = build_model(args)
    model.eval()

    dataloader, model = accelerator.prepare(dataloader, model)
    load_checkpoint(model, args.checkpoint_path, accelerator)

    unwrapped = accelerator.unwrap_model(model)
    _toggle_grad([unwrapped.low_stream, unwrapped.high_stream], False)

    codebook_size_low = get_codebook_size(unwrapped.low_stream.vq_layer)
    codebook_size_high = get_codebook_size(unwrapped.high_stream.vq_layer)

    total_raw_bytes = 0
    total_zstd_bytes = 0
    total_gzip_bytes = 0
    total_low_elems = 0
    total_high_elems = 0
    total_low_packed_bytes = 0
    total_high_packed_bytes = 0

    zstd_comp = zstd.ZstdCompressor(level=args.zstd_level)
    pbar = tqdm(total=len(dataloader), disable=not accelerator.is_main_process)
    pbar.set_description("compress")

    with torch.inference_mode():
        for batch in dataloader:
            x_raw = batch.float()
            x = _normalize_batch(x_raw, args.normalize)

            _recon, indices_low, indices_high, *_ = model(x)

            raw_bytes = x_raw.numel() * x_raw.element_size()
            total_raw_bytes += raw_bytes

            raw_bytes_blob = x_raw.detach().cpu().numpy().astype(np.float32, copy=False).tobytes()
            total_zstd_bytes += len(zstd_comp.compress(raw_bytes_blob))
            total_gzip_bytes += len(gzip.compress(raw_bytes_blob, compresslevel=args.gzip_level))

            if indices_low is not None:
                low_numel = int(indices_low.numel())
                total_low_elems += low_numel
                if codebook_size_low is not None:
                    total_low_packed_bytes += bitpack_bytes(low_numel, codebook_size_low)

            if indices_high is not None:
                high_numel = int(indices_high.numel())
                total_high_elems += high_numel
                if codebook_size_high is not None:
                    total_high_packed_bytes += bitpack_bytes(high_numel, codebook_size_high)

            if accelerator.is_main_process:
                pbar.update(1)

    if accelerator.is_main_process:
        pbar.close()

    stats = torch.tensor(
        [
            total_raw_bytes,
            total_zstd_bytes,
            total_gzip_bytes,
            total_low_elems,
            total_high_elems,
            total_low_packed_bytes,
            total_high_packed_bytes,
        ],
        device=accelerator.device,
        dtype=torch.float64,
    )
    stats = accelerator.reduce(stats, "sum")
    (
        total_raw_bytes,
        total_zstd_bytes,
        total_gzip_bytes,
        total_low_elems,
        total_high_elems,
        total_low_packed_bytes,
        total_high_packed_bytes,
    ) = [int(x.item()) for x in stats]

    result = {
        "data_path": args.data_path,
        "checkpoint_path": args.checkpoint_path,
        "segment_length": args.segment_length,
        "normalize": args.normalize,
        "codebook_size_low": codebook_size_low,
        "codebook_size_high": codebook_size_high,
        "raw_bytes": total_raw_bytes,
        "raw_gzip_bytes": total_gzip_bytes,
        "raw_zstd_bytes": total_zstd_bytes,
        "indices_low_elems": total_low_elems,
        "indices_high_elems": total_high_elems,
        "indices_low_bitpacked_bytes": total_low_packed_bytes,
        "indices_high_bitpacked_bytes": total_high_packed_bytes,
    }

    if accelerator.is_main_process:
        low_codebook_bytes = codebook_bytes(unwrapped.low_stream.vq_layer)
        high_codebook_bytes = codebook_bytes(unwrapped.high_stream.vq_layer)
        result["codebook_low_bytes"] = low_codebook_bytes
        result["codebook_high_bytes"] = high_codebook_bytes
        result["codebook_total_bytes"] = low_codebook_bytes + high_codebook_bytes
        result["bitpacked_total_bytes"] = total_low_packed_bytes + total_high_packed_bytes
        result["bitpacked_plus_codebook_bytes"] = (
            total_low_packed_bytes + total_high_packed_bytes + low_codebook_bytes + high_codebook_bytes
        )

    accelerator.wait_for_everyone()
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute compression stats for WaveSem stride model.")

    # Data
    p.add_argument("--data_path", type=str, required=True, help="Directory containing *.h5")
    p.add_argument("--segment_length", type=int, default=10240)
    p.add_argument("--max_files", type=int, default=25, help="0 = all")
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--normalize", type=str, default="none", choices=["none", "zscore", "scale"])
    p.add_argument("--warmup_h5_in_workers", action="store_true")

    # Model (match training)
    p.add_argument("--input_dim", type=int, default=1)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--cnn_compression_ratio", type=int, default=4)
    p.add_argument("--wavelet", type=str, default="db4")
    p.add_argument("--learnable_wavelet", action="store_true")

    # Checkpoint & compression
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--zstd_level", type=int, default=3)
    p.add_argument("--gzip_level", type=int, default=6)
    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--output_json", type=str, default=None)

    args = p.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    result = infer_and_compress(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output_json:
        with open(args.output_json, "w") as f:
            f.write(json.dumps(result, indent=2, ensure_ascii=False))

