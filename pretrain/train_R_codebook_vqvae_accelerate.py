#!/usr/bin/env python
"""
Train WaveletMambaDualStreamVQVAE (defined in wave_vqmodel.py) with H5 signal segments using 🤗 Accelerate.

Dataset:
  - Directory containing *.h5 files, each with dataset key "signal_segments" of shape [N, segment_length]
  - Loader implementation reused from h5_data_process.py (HDF5SignalDataset)

This script supports:
  - multi-GPU / DDP via accelerate launch
  - mixed precision (bf16/fp16)
  - gradient accumulation
  - checkpoint save/resume
  - tensorboard logging via accelerator.log
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from h5_data_process import HDF5SignalDataset
from baseline_models import ReconSpecVQLoss,RCodebookVQVAE

try:
    from accelerate import Accelerator
    from accelerate.utils import set_seed as accelerate_set_seed
except ImportError as e:
    raise SystemExit(
        "Missing dependency: accelerate.\n"
        "Install with: pip install accelerate\n"
        "Then launch with: accelerate launch train_wave_vqmodel_accelerate.py ...\n"
    ) from e
import warnings
warnings.filterwarnings("ignore")

def set_seed(seed: int) -> None:
    # Keep same behavior as other scripts in this repo + make accelerate deterministic across ranks.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    accelerate_set_seed(seed, device_specific=True)


def _maybe_set_matmul_precision(precision: str) -> None:
    # Optional perf tweak (safe no-op on older torch).
    try:
        if precision in {"high", "medium"}:
            torch.set_float32_matmul_precision(precision)
    except Exception:
        pass


@dataclass
class TrainState:
    epoch: int = 0
    best_val_loss: float = float("inf")


def build_dataloaders(args) -> Tuple[DataLoader, DataLoader]:
    h5_files = sorted(glob.glob(os.path.join(args.data_path, "*.h5")))
    if args.max_files > 0:
        h5_files = h5_files[: args.max_files]
    if len(h5_files) == 0:
        raise ValueError(f"No *.h5 files found in {args.data_path}")

    dataset = HDF5SignalDataset(
        h5_files,
        target_length=args.segment_length,
        lazy_load=True,
    )
    if len(dataset) == 0:
        raise ValueError("Dataset is empty")

    train_size = int(args.train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    def _worker_init_fn(_worker_id: int) -> None:
        # In each worker process, warmup-open H5 files to avoid mid-epoch stalls on first access.
        if not args.warmup_h5_in_workers:
            return
        try:
            # random_split returns Subset; underlying dataset is at .dataset
            ds = getattr(train_dataset, "dataset", train_dataset)
            if hasattr(ds, "warmup_open_files"):
                ds.warmup_open_files()
        except Exception:
            # Don't crash training for warmup; worst case it falls back to lazy open-on-demand.
            pass

    # Mirror the dataloader strategy used by train_with_h5.py, but keep it safe when num_workers=0.
    common_kwargs = dict(
        pin_memory=True,
        num_workers=args.num_workers,
        worker_init_fn=_worker_init_fn if args.num_workers > 0 else None,
    )
    if args.num_workers > 0:
        common_kwargs.update(
            dict(
                prefetch_factor=args.prefetch_factor,
                persistent_workers=True,
            )
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        **common_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        **common_kwargs,
    )
    return train_loader, val_loader


def build_model_and_loss(args) -> Tuple[nn.Module, nn.Module]:
    model = RCodebookVQVAE(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        compression_ratio=args.cnn_compression_ratio,
    )

    loss_fn = ReconSpecVQLoss(
        lambda_time=args.lambda_time,
        lambda_freq=args.lambda_freq,
        lambda_vq=args.lambda_vq,
    )
    return model, loss_fn


def save_checkpoint(
    accelerator: Accelerator,
    args,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    state: TrainState,
    ckpt_path: str,
) -> None:
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return

    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    payload = {
        "train_state": asdict(state),
        "model_state_dict": accelerator.get_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    accelerator.save(payload, ckpt_path)


def load_checkpoint(
    accelerator: Accelerator,
    args,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
) -> TrainState:
    if not args.resume_from:
        return TrainState(epoch=0, best_val_loss=float("inf"))

    ckpt_path = args.resume_from
    if accelerator.is_main_process:
        accelerator.print(f"Resuming from checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    train_state = ckpt.get("train_state", {})
    epoch = int(train_state.get("epoch", 0))
    best_val_loss = float(train_state.get("best_val_loss", float("inf")))
    if accelerator.is_main_process:
        accelerator.print(f"Resumed: start_epoch={epoch+1}, best_val_loss={best_val_loss:.6f}")
    return TrainState(epoch=epoch + 1, best_val_loss=best_val_loss)


def _normalize_batch(x: torch.Tensor, mode: str) -> torch.Tensor:
    """
    x: [B, 1, L]
    """
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


@torch.no_grad()
def evaluate(
    accelerator: Accelerator,
    model: nn.Module,
    loss_fn: nn.Module,
    val_loader: DataLoader,
    args,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_time = 0.0
    total_freq = 0.0
    total_vq = 0.0
    n = 0

    for batch in val_loader:
        x = batch.float()  # [B,1,L]
        x = _normalize_batch(x, args.normalize)

        x_recon, _, vq_loss = model(x)

        loss, parts = loss_fn(x=x, x_recon=x_recon, vq_loss=vq_loss)

        loss = accelerator.reduce(loss, "mean")
        total_loss += loss.item()
        total_time += float(parts["loss_time"])
        total_freq += float(parts["loss_spec"])
        total_vq += float(parts["loss_vq"])
        n += 1

    if n == 0:
        return {"loss": 0.0}
    return {
        "loss": total_loss / n,
        "loss_time": total_time / n,
        "loss_freq": total_freq / n,
        "loss_vq": total_vq / n,
    }


def train(args) -> None:
    _maybe_set_matmul_precision(args.matmul_precision)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=args.log_dir,
    )
    set_seed(args.seed)
    accelerator.init_trackers(project_name=args.project_name, config=vars(args))

    train_loader, val_loader = build_dataloaders(args)
    model, loss_fn = build_model_and_loss(args)

    if args.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.base_learning_rate,
            weight_decay=args.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    scheduler = None
    if args.use_scheduler:
        # Warmup + cosine by optimizer steps.
        # Note: scheduler is stepped once per optimizer step in the training loop.
        total_steps = math.ceil(len(train_loader) / args.gradient_accumulation_steps) * args.num_epochs
        warmup_steps = int(math.ceil(len(train_loader) / args.gradient_accumulation_steps) * args.warmup_epochs)
        warmup_steps = max(0, min(warmup_steps, max(0, total_steps - 1)))

        if warmup_steps <= 0:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps))
        else:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-6,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, total_steps - warmup_steps),
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_steps],
            )

    if scheduler is not None:
        train_loader, val_loader, model, loss_fn, optimizer, scheduler = accelerator.prepare(
            train_loader, val_loader, model, loss_fn, optimizer, scheduler
        )
    else:
        train_loader, val_loader, model, loss_fn, optimizer = accelerator.prepare(
            train_loader, val_loader, model, loss_fn, optimizer
        )

    os.makedirs(args.model_save_path, exist_ok=True)
    if accelerator.is_main_process:
        with open(os.path.join(args.model_save_path, "train_args.json"), "w") as f:
            json.dump(vars(args), f, indent=2, ensure_ascii=False)

    state = load_checkpoint(accelerator, args, model, optimizer, scheduler)

    global_step = 0
    for epoch in range(state.epoch, args.num_epochs):
        model.train()

        pbar = tqdm(total=len(train_loader), disable=not accelerator.is_main_process)
        pbar.set_description(f"epoch {epoch+1}/{args.num_epochs}")

        # Local sums; reduce once per epoch to avoid per-step all-reduce stalls.
        running = {"loss": 0.0, "loss_time": 0.0, "loss_freq": 0.0, "loss_vq": 0.0}
        n_steps = 0

        for batch in train_loader:
            with accelerator.accumulate(model):
                x = batch.float()
                x = _normalize_batch(x, args.normalize)

                x_recon, _, vq_loss = model(x)

                loss, parts = loss_fn(
                    x=x,
                    x_recon=x_recon,
                    vq_loss=vq_loss,
                )

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    if args.max_grad_norm > 0:
                        accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            # Local logging (avoid per-step all-reduce / barrier)
            loss_det = float(loss.detach().item())
            running["loss"] += loss_det
            running["loss_time"] += float(parts["loss_time"])
            running["loss_freq"] += float(parts["loss_spec"])
            running["loss_vq"] += float(parts["loss_vq"])
            n_steps += 1

            if accelerator.is_main_process:
                pbar.update(1)
                pbar.set_postfix(loss=f"{loss_det:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

            global_step += 1
            if accelerator.is_main_process and args.log_every > 0 and (global_step % args.log_every == 0):
                accelerator.log(
                    {
                        "train/step_loss": loss_det,
                        "train/lr": optimizer.param_groups[0]["lr"],
                    },
                    step=global_step,
                )

        if accelerator.is_main_process:
            pbar.close()

        # Epoch metrics
        if n_steps > 0:
            # Reduce epoch sums across processes once per epoch
            sums = torch.tensor(
                [
                    running["loss"],
                    running["loss_time"],
                    running["loss_freq"],
                    running["loss_vq"],
                    float(n_steps),
                ],
                device=accelerator.device,
                dtype=torch.float32,
            )
            sums = accelerator.reduce(sums, "sum")
            denom = max(1.0, float(sums[-1].item()))
            train_metrics = {
                "loss": float(sums[0].item()) / denom,
                "loss_time": float(sums[1].item()) / denom,
                "loss_freq": float(sums[2].item()) / denom,
                "loss_vq": float(sums[3].item()) / denom,
            }
        else:
            train_metrics = {"loss": 0.0}

        val_metrics = evaluate(accelerator, model, loss_fn, val_loader, args)

        if accelerator.is_main_process:
            accelerator.print(
                f"\nEpoch {epoch+1} | "
                f"train_loss={train_metrics['loss']:.6f} | val_loss={val_metrics['loss']:.6f} | "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )
            accelerator.log(
                {
                    "train/loss": train_metrics["loss"],
                    "train/loss_time": train_metrics.get("loss_time", 0.0),
                    "train/loss_freq": train_metrics.get("loss_freq", 0.0),
                    "train/loss_vq": train_metrics.get("loss_vq", 0.0),
                    "val/loss": val_metrics["loss"],
                    "val/loss_time": val_metrics.get("loss_time", 0.0),
                    "val/loss_freq": val_metrics.get("loss_freq", 0.0),
                    "val/loss_vq": val_metrics.get("loss_vq", 0.0),
                    "train/epoch": epoch + 1,
                },
                step=global_step,
            )

        # Save best + periodic
        state.epoch = epoch
        if val_metrics["loss"] < state.best_val_loss:
            state.best_val_loss = val_metrics["loss"]
            save_checkpoint(
                accelerator,
                args,
                model,
                optimizer,
                scheduler,
                state,
                ckpt_path=os.path.join(args.model_save_path, "best_model.pth"),
            )
            if accelerator.is_main_process:
                accelerator.print(f"Saved best checkpoint: val_loss={state.best_val_loss:.6f}")

        if args.checkpoint_interval > 0 and ((epoch + 1) % args.checkpoint_interval == 0):
            save_checkpoint(
                accelerator,
                args,
                model,
                optimizer,
                scheduler,
                state,
                ckpt_path=os.path.join(args.model_save_path, f"epoch_{epoch+1}.pth"),
            )

    accelerator.end_training()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train WaveletMambaDualStreamVQVAE with accelerate")

    # Data
    p.add_argument("--data_path", type=str, required=True, help="Directory containing *.h5")
    p.add_argument("--segment_length", type=int, default=10240)
    p.add_argument("--train_ratio", type=float, default=0.9)
    p.add_argument("--max_files", type=int, default=0, help="0 = all")
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--normalize", type=str, default="none", choices=["none", "zscore", "scale"])
    p.add_argument(
        "--warmup_h5_in_workers",
        action="store_true",
        help="Warmup-open H5 files in each DataLoader worker to avoid mid-epoch stalls on first file access.",
    )

    # Model
    p.add_argument("--input_dim", type=int, default=1)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--cnn_compression_ratio", type=int, default=4)
    p.add_argument("--codebook_size", type=int, default=512)
    p.add_argument("--commitment_cost", type=float, default=0.25)

    # Loss weights (WaveTransformerLoss)
    p.add_argument("--lambda_time", type=float, default=5.0)
    p.add_argument("--lambda_freq", type=float, default=1.0)
    p.add_argument("--lambda_vq", type=float, default=1.0)

    # Train
    p.add_argument("--num_epochs", type=int, default=100)
    p.add_argument("--optimizer", type=str, default="adamw", choices=["adamw"])
    p.add_argument("--base_learning_rate", type=float, default=3e-4)
    p.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Deprecated alias of --base_learning_rate (kept for backwards compatibility).",
    )
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--matmul_precision", type=str, default="high", choices=["high", "medium", "default"])
    p.add_argument("--use_scheduler", action="store_true")
    p.add_argument("--warmup_epochs", type=float, default=1.0, help="Warmup epochs for cosine scheduler (optimizer steps).")
    p.add_argument("--log_every", type=int, default=100, help="log step metrics every N steps (main process)")

    # I/O
    p.add_argument("--model_save_path", type=str, default="outputs/wave_vqmodel")
    p.add_argument("--log_dir", type=str, default="outputs/logs_wave_vqmodel")
    p.add_argument("--project_name", type=str, default="wave_vqmodel_pretrain")
    p.add_argument("--checkpoint_interval", type=int, default=10)
    p.add_argument("--resume_from", type=str, default=None)

    # Repro
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    # Backwards compatibility: if the old flag is used, it takes precedence.
    if args.learning_rate is not None:
        args.base_learning_rate = float(args.learning_rate)
    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)