#!/usr/bin/env python
"""
Evaluate VQVAEModel_wavesem_basecalling under different SNR levels by adding AWGN
to validation signals.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../models')))

from classes import basecalling_dataset
from bonito.vq_model import VQVAEModel_wavesem_basecalling,SingleCodebookVQVAEModel_stride_basecalling,BasecallingModel


def str_to_list(s):
    if isinstance(s, str):
        items = []
        for raw in s.split(","):
            item = raw.strip()
            if not item:
                continue
            if item.lower() == "none":
                items.append("none")
            else:
                items.append(float(item))
        return items
    return s


def add_awgn(x: torch.Tensor, snr_db: float) -> torch.Tensor:
    """
    Add white Gaussian noise to achieve target SNR (dB).
    x: [B, 1, L]
    """
    if snr_db is None:
        return x
    power = x.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-12)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = power / snr_linear
    noise = torch.randn_like(x) * torch.sqrt(noise_power)
    return x + noise


def build_model(args, device, train_loader, val_loader):
    if args.model == "wavesem":
        model = VQVAEModel_wavesem_basecalling(
            vqvae_model_path=args.vqvae_model_path,
            cnn_compression_ratio=args.cnn_compression_ratio,
            decoder_stride=args.decoder_stride,
            device=device,
            dataloader_train=train_loader,
            dataloader_validation=val_loader,
            use_amp=False,
        )
    elif args.model == "singlevq":
        model = SingleCodebookVQVAEModel_stride_basecalling(
            vqvae_model_path=args.vqvae_model_path,
            strides=args.strides,
            decoder_stride=args.decoder_stride,
            device=device,
            dataloader_train=train_loader,
            dataloader_validation=val_loader,
            use_amp=False,
        )
    elif args.model == "basecalling":
        model = BasecallingModel(
            vqvae_model_path=args.vqvae_model_path,
            decoder_stride=args.decoder_stride,
            device=device,
            dataloader_train=train_loader,
            dataloader_validation=val_loader,
            use_amp=False,
        )
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    model = model.to(device)
    ckpt = torch.load(args.basecalling_checkpoint, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.eval()
    return model


def evaluate_snr(model, dataloader, device, snr_db: float, max_batches: int | None = None):
    accs = []
    n_batches = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"SNR={snr_db}dB"):
            x = batch['x'].to(device).unsqueeze(1)
            y = batch['y'].to(device)
            x_noisy = add_awgn(x, snr_db)

            with torch.cuda.amp.autocast(enabled=False):
                p, _vq_loss = model.forward(x_noisy)

            predictions_decoded = model.decode(p, greedy=True)
            metrics = model.evaluate({'y': y}, predictions_decoded)
            if 'metric.accuracy' in metrics:
                accs.extend(metrics['metric.accuracy'])

            n_batches += 1
            if max_batches is not None and n_batches >= max_batches:
                break

    acc_mean = float(np.mean(accs)) if len(accs) > 0 else 0.0
    return acc_mean, len(accs)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate WaveSem basecalling under SNR noise.")
    parser.add_argument("--data-dir-val", type=str, required=True, help="Validation data directory")
    parser.add_argument("--vqvae_model_path", type=str, required=True, help="WaveSem pretrained checkpoint")
    parser.add_argument("--basecalling_checkpoint", type=str, required=True, help="Basecalling fine-tuned checkpoint")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--model", type=str, default="wavesem", choices=["wavesem", "singlevq", "basecalling"])
    parser.add_argument("--snr-list", type=str_to_list, default="none,35,30,25,20,15",
                        help="Comma-separated SNR list in dB. Use 'None' for clean.")
    parser.add_argument("--max-batches", type=int, default=None, help="Limit batches for quick test")
    parser.add_argument("--decoder_stride", type=str, default="[2,2,2]")
    parser.add_argument("--cnn_compression_ratio", type=int, default=4)
    parser.add_argument("--strides", type=str, default="[2,2,2,2,2,2]")
    parser.add_argument("--output-json", type=str, default="snr_eval.json")
    return parser.parse_args()


def main():
    args = parse_args()
    if isinstance(args.strides, str):
        try:
            import ast
            args.strides = ast.literal_eval(args.strides)
        except Exception:
            args.strides = [int(x.strip()) for x in args.strides.strip("[]").split(",") if x.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_ds = basecalling_dataset(data_path=args.data_dir_val)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=1)

    # Use val_loader for both to avoid dataset mismatch in evaluation
    model = build_model(args, device, val_loader, val_loader)

    results = []
    for snr in args.snr_list:
        if isinstance(snr, str) and snr.lower() == "none":
            snr_val = None
            snr_label = "clean"
        else:
            snr_val = float(snr)
            snr_label = f"{snr_val}dB"
        acc_mean, n = evaluate_snr(model, val_loader, device, snr_val, args.max_batches)
        results.append({"snr": snr_label, "accuracy": acc_mean, "samples": n})
        print(f"SNR={snr_label} | acc={acc_mean:.4f} | samples={n}")

    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved results to {args.output_json}")


if __name__ == "__main__":
    main()

