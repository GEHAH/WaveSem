#!/usr/bin/env python
"""
Train SoundStream (from audiolm-pytorch) on H5 signal segments using 🤗 Accelerate.

Dataset:
  - Directory containing *.h5 files, each with dataset key "signal_segments" of shape [N, segment_length]
  - Loader implementation reused from h5_data_process.py (HDF5SignalDataset)

Features:
  - multi-GPU / DDP via accelerate launch
  - mixed precision (bf16/fp16)
  - gradient accumulation
  - separate generator / discriminator optimizers
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
import sys
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Local imports
from h5_data_process import HDF5SignalDataset

# Make sure audiolm-pytorch is importable when run from this folder
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "soundstream", "audiolm-pytorch-main"))
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

try:
    from audiolm_pytorch import SoundStream
except ImportError as e:
    raise SystemExit(
        "Missing dependency: audiolm-pytorch.\n"
        "Install with: pip install -e ./soundstream/audiolm-pytorch-main\n"
        "or adjust PYTHONPATH to include audiolm_pytorch."
    ) from e

try:
    from accelerate import Accelerator
    from accelerate.utils import set_seed as accelerate_set_seed
    from accelerate import DistributedDataParallelKwargs
except ImportError as e:
    raise SystemExit(
        "Missing dependency: accelerate.\n"
        "Install with: pip install accelerate\n"
        "Then launch with: accelerate launch train_soundstream_accelerate.py ..."
    ) from e


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    accelerate_set_seed(seed, device_specific=True)


def _maybe_set_matmul_precision(precision: str) -> None:
    try:
        if precision in {"high", "medium"}:
            torch.set_float32_matmul_precision(precision)
    except Exception:
        pass


@dataclass
class TrainState:
    epoch: int = 0
    best_val_loss: float = float("inf")


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


def parse_comma_ints(arg: str) -> Tuple[int, ...]:
    vals = [v.strip() for v in arg.split(",") if v.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("Expected non-empty comma-separated integers.")
    return tuple(int(v) for v in vals)


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
        if not args.warmup_h5_in_workers:
            return
        try:
            ds = getattr(train_dataset, "dataset", train_dataset)
            if hasattr(ds, "warmup_open_files"):
                ds.warmup_open_files()
        except Exception:
            pass

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


def build_model(args) -> SoundStream:
    return SoundStream(
        channels=args.channels,
        strides=tuple(args.strides),
        channel_mults=tuple(args.channel_mults),
        codebook_dim=args.codebook_dim,
        codebook_size=args.codebook_size,
        rq_num_quantizers=args.rq_num_quantizers,
        rq_commitment_weight=args.rq_commitment_weight,
        rq_ema_decay=args.rq_ema_decay,
        rq_groups=args.rq_groups,
        rq_stochastic_sample_codes=args.rq_stochastic_sample_codes,
        quantize_dropout_cutoff_index=args.quantize_dropout_cutoff_index,
        target_sample_hz=args.target_sample_hz,
        recon_loss_weight=args.recon_loss_weight,
        multi_spectral_recon_loss_weight=args.multi_spectral_recon_loss_weight,
        adversarial_loss_weight=args.adversarial_loss_weight,
        feature_loss_weight=args.feature_loss_weight,
        stft_normalized=args.stft_normalized,
        use_local_attn=not args.disable_local_attn,
        attn_window_size=args.attn_window_size,
        attn_dim_head=args.attn_dim_head,
        attn_heads=args.attn_heads,
        attn_depth=args.attn_depth,
        input_channels=1,
    )


def save_checkpoint(
    accelerator: Accelerator,
    args,
    model,
    gen_optimizer: torch.optim.Optimizer,
    disc_optimizer: torch.optim.Optimizer,
    gen_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    disc_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
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
        "gen_optimizer_state_dict": gen_optimizer.state_dict(),
        "disc_optimizer_state_dict": disc_optimizer.state_dict(),
        "args": vars(args),
    }
    if gen_scheduler is not None:
        payload["gen_scheduler_state_dict"] = gen_scheduler.state_dict()
    if disc_scheduler is not None:
        payload["disc_scheduler_state_dict"] = disc_scheduler.state_dict()
    accelerator.save(payload, ckpt_path)


def load_checkpoint(
    accelerator: Accelerator,
    args,
    model,
    gen_optimizer: torch.optim.Optimizer,
    disc_optimizer: torch.optim.Optimizer,
    gen_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    disc_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
) -> TrainState:
    if not args.resume_from:
        return TrainState(epoch=0, best_val_loss=float("inf"))

    ckpt_path = args.resume_from
    if accelerator.is_main_process:
        accelerator.print(f"Resuming from checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.load_state_dict(ckpt["model_state_dict"])
    gen_optimizer.load_state_dict(ckpt["gen_optimizer_state_dict"])
    disc_optimizer.load_state_dict(ckpt["disc_optimizer_state_dict"])
    if gen_scheduler is not None and "gen_scheduler_state_dict" in ckpt:
        gen_scheduler.load_state_dict(ckpt["gen_scheduler_state_dict"])
    if disc_scheduler is not None and "disc_scheduler_state_dict" in ckpt:
        disc_scheduler.load_state_dict(ckpt["disc_scheduler_state_dict"])

    train_state = ckpt.get("train_state", {})
    epoch = int(train_state.get("epoch", 0))
    best_val_loss = float(train_state.get("best_val_loss", float("inf")))
    if accelerator.is_main_process:
        accelerator.print(f"Resumed: start_epoch={epoch+1}, best_val_loss={best_val_loss:.6f}")
    return TrainState(epoch=epoch + 1, best_val_loss=best_val_loss)


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


def _unpack_loss_parts(parts: Tuple[torch.Tensor, ...]) -> Dict[str, float]:
    if len(parts) == 5:
        names = ["recon_loss", "multi_spectral_recon_loss", "adversarial_loss", "feature_loss", "commit_loss"]
    else:
        names = ["recon_loss", "multi_spectral_recon_loss", "feature_loss", "commit_loss"]
    return {name: float(val.detach().item()) for name, val in zip(names, parts)}


@torch.no_grad()
def evaluate(
    accelerator: Accelerator,
    model,
    val_loader: DataLoader,
    args,
) -> Dict[str, float]:
    model.eval()
    totals = {}
    n = 0

    for batch in val_loader:
        x = batch.float()
        x = _normalize_batch(x, args.normalize)

        loss, parts = model(x, return_loss_breakdown=True)
        loss = accelerator.reduce(loss, "mean")
        totals["loss"] = totals.get("loss", 0.0) + loss.item()

        part_values = _unpack_loss_parts(parts)
        for k, v in part_values.items():
            totals[k] = totals.get(k, 0.0) + v
        n += 1

    if n == 0:
        return {"loss": 0.0}
    return {k: v / n for k, v in totals.items()}


def train(args) -> None:
    _maybe_set_matmul_precision(args.matmul_precision)

    accelerator = Accelerator(
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=args.log_dir,
    )
    set_seed(args.seed)
    accelerator.init_trackers(project_name=args.project_name, config=vars(args))

    train_loader, val_loader = build_dataloaders(args)
    model = build_model(args)

    gen_params = list(model.non_discr_parameters())
    disc_params = list(model.discriminators.parameters()) + list(model.stft_discriminator.parameters())

    gen_optimizer = torch.optim.AdamW(
        gen_params,
        lr=args.base_learning_rate,
        weight_decay=args.weight_decay,
    )
    disc_optimizer = torch.optim.AdamW(
        disc_params,
        lr=args.disc_learning_rate if args.disc_learning_rate is not None else args.base_learning_rate,
        weight_decay=args.weight_decay,
    )

    def _build_scheduler(optimizer: torch.optim.Optimizer):
        if not args.use_scheduler:
            return None
        total_steps = math.ceil(len(train_loader) / args.gradient_accumulation_steps) * args.num_epochs
        warmup_steps = int(math.ceil(len(train_loader) / args.gradient_accumulation_steps) * args.warmup_epochs)
        warmup_steps = max(0, min(warmup_steps, max(0, total_steps - 1)))

        if warmup_steps <= 0:
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps))
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
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )

    gen_scheduler = _build_scheduler(gen_optimizer)
    disc_scheduler = _build_scheduler(disc_optimizer)

    if gen_scheduler is not None and disc_scheduler is not None:
        train_loader, val_loader, model, gen_optimizer, disc_optimizer, gen_scheduler, disc_scheduler = accelerator.prepare(
            train_loader, val_loader, model, gen_optimizer, disc_optimizer, gen_scheduler, disc_scheduler
        )
    elif gen_scheduler is not None:
        train_loader, val_loader, model, gen_optimizer, disc_optimizer, gen_scheduler = accelerator.prepare(
            train_loader, val_loader, model, gen_optimizer, disc_optimizer, gen_scheduler
        )
        disc_scheduler = None
    elif disc_scheduler is not None:
        train_loader, val_loader, model, gen_optimizer, disc_optimizer, disc_scheduler = accelerator.prepare(
            train_loader, val_loader, model, gen_optimizer, disc_optimizer, disc_scheduler
        )
        gen_scheduler = None
    else:
        train_loader, val_loader, model, gen_optimizer, disc_optimizer = accelerator.prepare(
            train_loader, val_loader, model, gen_optimizer, disc_optimizer
        )

    os.makedirs(args.model_save_path, exist_ok=True)
    if accelerator.is_main_process:
        with open(os.path.join(args.model_save_path, "train_args.json"), "w") as f:
            json.dump(vars(args), f, indent=2, ensure_ascii=False)

    state = load_checkpoint(accelerator, args, model, gen_optimizer, disc_optimizer, gen_scheduler, disc_scheduler)

    global_step = 0
    for epoch in range(state.epoch, args.num_epochs):
        model.train()

        pbar = tqdm(total=len(train_loader), disable=not accelerator.is_main_process)
        pbar.set_description(f"epoch {epoch+1}/{args.num_epochs}")

        running_gen = {"loss": 0.0}
        running_disc = {"discr_loss": 0.0}
        n_steps = 0

        for batch in train_loader:
            batch = batch.float()
            batch = _normalize_batch(batch, args.normalize)

            unwrapped = accelerator.unwrap_model(model)

            # generator update
            _toggle_grad([unwrapped.discriminators, unwrapped.stft_discriminator], False)
            gen_optimizer.zero_grad()
            with accelerator.accumulate(model):
                gen_loss, parts = model(batch, return_loss_breakdown=True)
                accelerator.backward(gen_loss)

                if accelerator.sync_gradients:
                    if args.max_grad_norm > 0:
                        accelerator.clip_grad_norm_(gen_params, args.max_grad_norm)
                    gen_optimizer.step()
                    if gen_scheduler is not None:
                        gen_scheduler.step()
                    gen_optimizer.zero_grad(set_to_none=True)

            gen_parts = _unpack_loss_parts(parts)
            running_gen["loss"] += float(gen_loss.detach().item())
            for k, v in gen_parts.items():
                running_gen[k] = running_gen.get(k, 0.0) + v

            # discriminator update
            _toggle_grad([unwrapped.discriminators, unwrapped.stft_discriminator], True)
            disc_optimizer.zero_grad()
            with accelerator.accumulate(model):
                disc_loss = model(
                    batch,
                    return_discr_loss=True,
                    apply_grad_penalty=args.apply_grad_penalty,
                )
                accelerator.backward(disc_loss)

                if accelerator.sync_gradients:
                    if args.disc_max_grad_norm > 0:
                        accelerator.clip_grad_norm_(disc_params, args.disc_max_grad_norm)
                    disc_optimizer.step()
                    if disc_scheduler is not None:
                        disc_scheduler.step()
                    disc_optimizer.zero_grad(set_to_none=True)

            running_disc["discr_loss"] += float(disc_loss.detach().item())

            n_steps += 1
            global_step += 1

            if accelerator.is_main_process:
                pbar.update(1)
                pbar.set_postfix(
                    gen_loss=f"{gen_loss.detach().item():.4f}",
                    disc_loss=f"{disc_loss.detach().item():.4f}",
                    lr=f"{gen_optimizer.param_groups[0]['lr']:.2e}",
                )

            if accelerator.is_main_process and args.log_every > 0 and (global_step % args.log_every == 0):
                log_payload = {
                    "train/gen_loss": float(gen_loss.detach().item()),
                    "train/disc_loss": float(disc_loss.detach().item()),
                    "train/lr": gen_optimizer.param_groups[0]["lr"],
                }
                for k, v in gen_parts.items():
                    log_payload[f"train/{k}"] = v
                accelerator.log(log_payload, step=global_step)

        if accelerator.is_main_process:
            pbar.close()

        if n_steps > 0:
            sums = torch.tensor(
                [running_gen.get(k, 0.0) for k in sorted(running_gen.keys())] + [running_disc["discr_loss"], float(n_steps)],
                device=accelerator.device,
                dtype=torch.float32,
            )
            sums = accelerator.reduce(sums, "sum")
            denom = max(1.0, float(sums[-1].item()))

            keys = sorted(running_gen.keys())
            train_metrics = {f"gen/{k}": float(sums[i].item()) / denom for i, k in enumerate(keys)}
            train_metrics["disc/loss"] = float(sums[len(keys)].item()) / denom
        else:
            train_metrics = {"gen/loss": 0.0, "disc/loss": 0.0}

        val_metrics = evaluate(accelerator, model, val_loader, args)

        if accelerator.is_main_process:
            accelerator.print(
                f"\nEpoch {epoch+1} | "
                f"gen_loss={train_metrics.get('gen/loss', 0.0):.6f} | "
                f"val_loss={val_metrics.get('loss', 0.0):.6f} | "
                f"lr={gen_optimizer.param_groups[0]['lr']:.2e}"
            )
            log_payload = {"train/epoch": epoch + 1, "train/disc_loss": train_metrics.get("disc/loss", 0.0)}
            for k, v in train_metrics.items():
                log_payload[f"train/{k}"] = v
            for k, v in val_metrics.items():
                log_payload[f"val/{k}"] = v
            accelerator.log(log_payload, step=global_step)

        state.epoch = epoch
        if val_metrics.get("loss", float("inf")) < state.best_val_loss:
            state.best_val_loss = val_metrics["loss"]
            save_checkpoint(
                accelerator,
                args,
                model,
                gen_optimizer,
                disc_optimizer,
                gen_scheduler,
                disc_scheduler,
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
                gen_optimizer,
                disc_optimizer,
                gen_scheduler,
                disc_scheduler,
                state,
                ckpt_path=os.path.join(args.model_save_path, f"epoch_{epoch+1}.pth"),
            )

    accelerator.end_training()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SoundStream with accelerate")

    # Data
    p.add_argument("--data_path", type=str, required=True, help="Directory containing *.h5")
    p.add_argument("--segment_length", type=int, default=10240)
    p.add_argument("--train_ratio", type=float, default=0.9)
    p.add_argument("--max_files", type=int, default=0, help="0 = all")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--normalize", type=str, default="none", choices=["none", "zscore", "scale"])
    p.add_argument(
        "--warmup_h5_in_workers",
        action="store_true",
        help="Warmup-open H5 files in each DataLoader worker to avoid mid-epoch stalls on first file access.",
    )

    # Model
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--strides", type=int, nargs="+", default=[2, 4, 5, 8], help="Downsampling strides per encoder block")
    p.add_argument("--channel_mults", type=int, nargs="+", default=[2, 4, 8, 16], help="Channel multipliers per encoder block")
    p.add_argument("--codebook_dim", type=int, default=512)
    p.add_argument("--codebook_size", type=int, default=1024)
    p.add_argument("--rq_num_quantizers", type=int, default=8)
    p.add_argument("--rq_commitment_weight", type=float, default=1.0)
    p.add_argument("--rq_ema_decay", type=float, default=0.95)
    p.add_argument("--rq_groups", type=int, default=1)
    p.add_argument("--rq_stochastic_sample_codes", action="store_true")
    p.add_argument("--quantize_dropout_cutoff_index", type=int, default=1)
    p.add_argument("--target_sample_hz", type=int, default=4000)
    p.add_argument("--recon_loss_weight", type=float, default=1.0)
    p.add_argument("--multi_spectral_recon_loss_weight", type=float, default=1e-5)
    p.add_argument("--adversarial_loss_weight", type=float, default=1.0)
    p.add_argument("--feature_loss_weight", type=float, default=100.0)
    p.add_argument("--stft_normalized", action="store_true")
    p.add_argument("--disable_local_attn", action="store_true", help="Disable local attention blocks in encoder/decoder")
    p.add_argument("--attn_window_size", type=int, default=128)
    p.add_argument("--attn_dim_head", type=int, default=64)
    p.add_argument("--attn_heads", type=int, default=8)
    p.add_argument("--attn_depth", type=int, default=1)

    # Train
    p.add_argument("--num_epochs", type=int, default=100)
    p.add_argument("--optimizer", type=str, default="adamw", choices=["adamw"])
    p.add_argument("--base_learning_rate", type=float, default=3e-4)
    p.add_argument("--disc_learning_rate", type=float, default=None, help="If set, overrides generator lr for discriminators.")
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--disc_max_grad_norm", type=float, default=1.0)
    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--matmul_precision", type=str, default="high", choices=["high", "medium", "default"])
    p.add_argument("--use_scheduler", action="store_true")
    p.add_argument("--warmup_epochs", type=float, default=1.0, help="Warmup epochs for cosine scheduler (optimizer steps).")
    p.add_argument("--log_every", type=int, default=100, help="log step metrics every N steps (main process)")
    p.add_argument("--apply_grad_penalty", action="store_true", help="Use gradient penalty for discriminator losses")

    # I/O
    p.add_argument("--model_save_path", type=str, default="outputs/soundstream")
    p.add_argument("--log_dir", type=str, default="outputs/logs_soundstream")
    p.add_argument("--project_name", type=str, default="soundstream_pretrain")
    p.add_argument("--checkpoint_interval", type=int, default=10)
    p.add_argument("--resume_from", type=str, default=None)

    # Repro
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)

