#!/usr/bin/env python3
"""
predict_methylation.py
=======================
使用训练好的甲基化检测模型对新的 ONT 数据进行预测。

用法:
  # 对 POD5 文件进行甲基化预测
  python predict_methylation.py \
    --model_checkpoint outputs/methylation_5mC/checkpoints/best_model.pth \
    --pretrained_checkpoint outputs/wave_vqmodel_Equal/checkpoints/best_model.pth \
    --input_pod5 /path/to/test.pod5 \
    --output predictions/5mC_predictions.tsv \
    --mod_type 5mC

  # 对 HDF5 数据进行预测
  python predict_methylation.py \
    --model_checkpoint outputs/methylation_5mC/checkpoints/best_model.pth \
    --pretrained_checkpoint outputs/wave_vqmodel_Equal/checkpoints/best_model.pth \
    --input_h5 methylation_data/5mC_test.h5 \
    --output predictions/5mC_test_results.tsv \
    --mod_type 5mC

输出格式 (TSV):
  read_id  position  mod_type  probability  prediction
"""

import os
import sys
import argparse
import numpy as np
import h5py
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent))


def predict_h5(
    model,
    h5_path: str,
    device: torch.device,
    batch_size: int = 64,
    normalize: str = "zscore",
) -> list:
    """对 HDF5 格式的信号数据进行预测"""
    results = []

    with h5py.File(h5_path, "r") as f:
        signals = f["signal_segments"][:]
        labels = f["labels"][:] if "labels" in f else None
        read_ids = f["read_ids"][:] if "read_ids" in f else [f"sample_{i}" for i in range(len(signals))]

    n = len(signals)
    model.eval()

    with torch.no_grad():
        for start in tqdm(range(0, n, batch_size), desc="Predicting"):
            end = min(start + batch_size, n)
            batch = signals[start:end].astype(np.float32)

            if normalize == "zscore":
                mu = batch.mean(axis=-1, keepdims=True)
                std = np.maximum(batch.std(axis=-1, keepdims=True), 1e-6)
                batch = (batch - mu) / std

            batch_tensor = torch.from_numpy(batch).unsqueeze(1).to(device)  # [B, 1, L]

            logits = model(batch_tensor)
            probs = F.softmax(logits, dim=-1)[:, 1]  # P(modified)
            preds = logits.argmax(dim=-1)

            for i in range(end - start):
                idx = start + i
                rid = read_ids[idx] if isinstance(read_ids[idx], str) else read_ids[idx].decode("utf-8")
                result = {
                    "read_id": rid,
                    "probability": float(probs[i].cpu()),
                    "prediction": int(preds[i].cpu()),
                }
                if labels is not None:
                    result["ground_truth"] = int(labels[idx])
                results.append(result)

    return results


def predict_pod5(
    model,
    pod5_path: str,
    device: torch.device,
    segment_length: int = 10240,
    batch_size: int = 64,
    normalize: str = "zscore",
    max_reads: int = 0,
) -> list:
    """对 POD5 文件中的 reads 进行全序列甲基化扫描"""
    try:
        import pod5
    except ImportError:
        logger.error("pod5 not installed. Run: pip install pod5")
        return []

    results = []
    model.eval()

    pod5_path = Path(pod5_path)
    if pod5_path.is_dir():
        pod5_files = list(pod5_path.glob("*.pod5"))
    else:
        pod5_files = [pod5_path]

    count = 0
    for p5_file in pod5_files:
        reader = pod5.Reader(p5_file)

        for read in tqdm(reader.reads(), desc=f"Scanning {p5_file.name}"):
            rid = str(read.read_id)
            signal = read.signal.astype(np.float32)

            if len(signal) < segment_length:
                continue

            # 滑窗扫描
            segments = []
            positions = []
            for start in range(0, len(signal) - segment_length + 1, segment_length // 2):
                seg = signal[start:start + segment_length]
                if normalize == "zscore":
                    mu = seg.mean()
                    std = max(seg.std(), 1e-6)
                    seg = (seg - mu) / std
                segments.append(seg)
                positions.append(start)

            if not segments:
                continue

            # 批量预测
            with torch.no_grad():
                for batch_start in range(0, len(segments), batch_size):
                    batch_end = min(batch_start + batch_size, len(segments))
                    batch = np.stack(segments[batch_start:batch_end])
                    batch_tensor = torch.from_numpy(batch).unsqueeze(1).to(device)

                    logits = model(batch_tensor)
                    probs = F.softmax(logits, dim=-1)[:, 1]
                    preds = logits.argmax(dim=-1)

                    for i in range(batch_end - batch_start):
                        results.append({
                            "read_id": rid,
                            "signal_position": positions[batch_start + i],
                            "probability": float(probs[i].cpu()),
                            "prediction": int(preds[i].cpu()),
                        })

            count += 1
            if max_reads > 0 and count >= max_reads:
                break

        reader.close()

    return results


def save_results(results: list, output_path: str):
    """保存预测结果为 TSV"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if not results:
        logger.warning("No results to save")
        return

    keys = results[0].keys()
    with open(output_path, "w") as f:
        f.write("\t".join(keys) + "\n")
        for r in results:
            f.write("\t".join(str(r[k]) for k in keys) + "\n")

    logger.info(f"Saved {len(results)} predictions to {output_path}")

    # 统计
    preds = [r["prediction"] for r in results]
    n_mod = sum(preds)
    n_unmod = len(preds) - n_mod
    logger.info(f"  Modified: {n_mod} ({100*n_mod/len(preds):.1f}%)")
    logger.info(f"  Unmodified: {n_unmod} ({100*n_unmod/len(preds):.1f}%)")

    if "ground_truth" in results[0]:
        gts = [r["ground_truth"] for r in results]
        correct = sum(1 for p, g in zip(preds, gts) if p == g)
        logger.info(f"  Accuracy: {100*correct/len(preds):.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Predict methylation using trained model")
    parser.add_argument("--model_checkpoint", type=str, required=True,
                        help="Trained methylation model checkpoint")
    parser.add_argument("--pretrained_checkpoint", type=str, required=True,
                        help="Pretrained WaveSem checkpoint")
    parser.add_argument("--input_h5", type=str, default=None,
                        help="Input HDF5 file with signals")
    parser.add_argument("--input_pod5", type=str, default=None,
                        help="Input POD5 file/directory")
    parser.add_argument("--output", type=str, required=True,
                        help="Output TSV file path")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--normalize", type=str, default="zscore")
    parser.add_argument("--segment_length", type=int, default=10240)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--wavelet", type=str, default="db4")
    parser.add_argument("--learnable_wavelet", action="store_true")
    parser.add_argument("--cnn_compression_ratio", type=int, default=4)
    parser.add_argument("--head_type", type=str, default="attention")
    parser.add_argument("--mod_type", type=str, default="5mC", choices=["5mC", "6mA"])
    parser.add_argument("--max_reads", type=int, default=0)

    args = parser.parse_args()

    if args.input_h5 is None and args.input_pod5 is None:
        parser.error("Must provide either --input_h5 or --input_pod5")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    from methylation_model import build_methylation_detector

    model = build_methylation_detector(
        pretrained_checkpoint=args.pretrained_checkpoint,
        device=str(device),
        head_type=args.head_type,
        freeze_encoder=True,
        hidden_dim=args.hidden_dim,
        wavelet=args.wavelet,
        learnable=args.learnable_wavelet,
        cnn_compression_ratio=args.cnn_compression_ratio,
    )

    # 加载训练好的权重
    ckpt = torch.load(args.model_checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info(f"Loaded trained model from {args.model_checkpoint}")

    # 预测
    if args.input_h5:
        results = predict_h5(model, args.input_h5, device, args.batch_size, args.normalize)
    else:
        results = predict_pod5(
            model, args.input_pod5, device,
            args.segment_length, args.batch_size,
            args.normalize, args.max_reads,
        )

    # 保存结果
    save_results(results, args.output)


if __name__ == "__main__":
    main()
