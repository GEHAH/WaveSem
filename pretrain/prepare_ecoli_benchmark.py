#!/usr/bin/env python3
"""
prepare_ecoli_benchmark.py
============================
从 ONT E. coli 5mC CpG 基准数据集生成甲基化检测的训练/测试数据。

实验设计 (基于生物学先验):
  - DM_MSssI: MSssI 处理的 E. coli → CpG 位点全甲基化 (label=1, 阳性对照)
  - DM:       dam-/dcm- 菌株 → 无 CpG 甲基化 (label=0, 阴性对照)
  - WT:       野生型 E. coli → 天然甲基化模式 (用于测试/评估)

信号提取策略:
  1. 从 POD5 读取原始信号
  2. 按 segment_length 滑窗切分 (stride = segment_length // 2)
  3. MAD 归一化 (Median Absolute Deviation)
  4. 根据来源分配标签

输出: HDF5 文件，兼容 train_methylation.py 的 MethylationDataset
"""

import os
import sys
import argparse
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
import logging
import random

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def mad_normalize(signal: np.ndarray) -> np.ndarray:
    """MAD (Median Absolute Deviation) 归一化 - ONT 信号标准方法"""
    median = np.median(signal)
    mad = np.median(np.abs(signal - median))
    mad = max(mad * 1.4826, 1e-6)  # 1.4826 使 MAD 与标准差一致
    return (signal - median) / mad


def extract_segments_from_pod5(
    pod5_path: str,
    segment_length: int = 10240,
    stride: int = 5120,
    max_reads: int = 0,
    normalize: str = "mad",
    min_signal_length: int = 10240,
) -> tuple:
    """
    从 POD5 文件中提取固定长度的信号片段。

    Args:
        pod5_path: POD5 文件路径
        segment_length: 信号片段长度 (与预训练一致)
        stride: 滑窗步长
        max_reads: 最大读取 read 数 (0=全部)
        normalize: 归一化方法 ("mad", "zscore", "none")
        min_signal_length: 最短信号长度过滤

    Returns:
        segments: list of np.ndarray [segment_length]
        read_ids: list of str
    """
    import pod5

    segments = []
    read_ids = []

    reader = pod5.Reader(Path(pod5_path))
    n_reads = 0
    n_short = 0

    for read in tqdm(reader.reads(), desc=f"Reading {Path(pod5_path).name}"):
        signal = read.signal.astype(np.float32)

        if len(signal) < min_signal_length:
            n_short += 1
            continue

        # 归一化
        if normalize == "mad":
            signal = mad_normalize(signal)
        elif normalize == "zscore":
            mu, std = signal.mean(), max(signal.std(), 1e-6)
            signal = (signal - mu) / std

        # 滑窗切分
        rid = str(read.read_id)
        for start in range(0, len(signal) - segment_length + 1, stride):
            seg = signal[start:start + segment_length]
            segments.append(seg)
            read_ids.append(rid)

        n_reads += 1
        if max_reads > 0 and n_reads >= max_reads:
            break

    reader.close()
    logger.info(
        f"  {Path(pod5_path).name}: {n_reads} reads, "
        f"{len(segments)} segments (skipped {n_short} short reads)"
    )
    return segments, read_ids


def build_dataset(
    positive_pod5: str,
    negative_pod5: str,
    output_path: str,
    segment_length: int = 10240,
    stride: int = 5120,
    max_reads_per_class: int = 0,
    max_segments_per_class: int = 50000,
    normalize: str = "mad",
    seed: int = 42,
):
    """
    构建甲基化检测训练数据集。

    将阳性 (DM_MSssI) 和阴性 (DM) 样本合并为一个平衡的 HDF5 数据集。
    """
    random.seed(seed)
    np.random.seed(seed)

    # 提取阳性样本 (DM_MSssI = 全 CpG 甲基化)
    logger.info("=== Extracting POSITIVE samples (DM_MSssI = fully methylated) ===")
    pos_segments, pos_read_ids = extract_segments_from_pod5(
        positive_pod5,
        segment_length=segment_length,
        stride=stride,
        max_reads=max_reads_per_class,
        normalize=normalize,
    )

    # 提取阴性样本 (DM = 无甲基化)
    logger.info("=== Extracting NEGATIVE samples (DM = unmethylated) ===")
    neg_segments, neg_read_ids = extract_segments_from_pod5(
        negative_pod5,
        segment_length=segment_length,
        stride=stride,
        max_reads=max_reads_per_class,
        normalize=normalize,
    )

    # 平衡采样: 取两类中较少的数量
    n_pos = len(pos_segments)
    n_neg = len(neg_segments)
    logger.info(f"Raw segments: positive={n_pos}, negative={n_neg}")

    # 限制每类最大片段数
    if max_segments_per_class > 0:
        if n_pos > max_segments_per_class:
            idx = np.random.choice(n_pos, max_segments_per_class, replace=False)
            pos_segments = [pos_segments[i] for i in idx]
            pos_read_ids = [pos_read_ids[i] for i in idx]
            n_pos = max_segments_per_class

        if n_neg > max_segments_per_class:
            idx = np.random.choice(n_neg, max_segments_per_class, replace=False)
            neg_segments = [neg_segments[i] for i in idx]
            neg_read_ids = [neg_read_ids[i] for i in idx]
            n_neg = max_segments_per_class

    # 平衡到相同数量
    n_balanced = min(n_pos, n_neg)
    if n_pos > n_balanced:
        idx = np.random.choice(n_pos, n_balanced, replace=False)
        pos_segments = [pos_segments[i] for i in idx]
        pos_read_ids = [pos_read_ids[i] for i in idx]
    if n_neg > n_balanced:
        idx = np.random.choice(n_neg, n_balanced, replace=False)
        neg_segments = [neg_segments[i] for i in idx]
        neg_read_ids = [neg_read_ids[i] for i in idx]

    logger.info(f"Balanced dataset: {n_balanced} per class, total={n_balanced * 2}")

    # 合并
    all_signals = np.stack(pos_segments + neg_segments, axis=0).astype(np.float32)
    all_labels = np.array(
        [1] * len(pos_segments) + [0] * len(neg_segments), dtype=np.int8
    )
    all_read_ids = pos_read_ids + neg_read_ids
    all_mod_types = ["5mC"] * len(all_signals)

    # Shuffle
    perm = np.random.permutation(len(all_signals))
    all_signals = all_signals[perm]
    all_labels = all_labels[perm]
    all_read_ids = [all_read_ids[i] for i in perm]
    all_mod_types = [all_mod_types[i] for i in perm]

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.create_dataset(
            "signal_segments", data=all_signals, dtype="float32",
            compression="gzip", compression_opts=4,
        )
        f.create_dataset("labels", data=all_labels, dtype="int8")

        dt = h5py.string_dtype()
        f.create_dataset("read_ids", data=all_read_ids, dtype=dt)
        f.create_dataset("mod_types", data=all_mod_types, dtype=dt)

        f.attrs["num_samples"] = len(all_signals)
        f.attrs["segment_length"] = segment_length
        f.attrs["num_positive"] = int(np.sum(all_labels == 1))
        f.attrs["num_negative"] = int(np.sum(all_labels == 0))
        f.attrs["source_positive"] = str(positive_pod5)
        f.attrs["source_negative"] = str(negative_pod5)

    logger.info(f"Saved {len(all_signals)} samples to {output_path}")
    logger.info(
        f"  positive={int(np.sum(all_labels==1))}, "
        f"  negative={int(np.sum(all_labels==0))}"
    )
    return output_path


def build_test_dataset(
    wt_pod5: str,
    output_path: str,
    segment_length: int = 10240,
    stride: int = 5120,
    max_reads: int = 0,
    max_segments: int = 20000,
    normalize: str = "mad",
    seed: int = 42,
):
    """
    构建 WT 测试数据集 (无标签，用于预测)。
    """
    np.random.seed(seed)

    logger.info("=== Extracting WT test samples ===")
    segments, read_ids = extract_segments_from_pod5(
        wt_pod5,
        segment_length=segment_length,
        stride=stride,
        max_reads=max_reads,
        normalize=normalize,
    )

    n = len(segments)
    if max_segments > 0 and n > max_segments:
        idx = np.random.choice(n, max_segments, replace=False)
        segments = [segments[i] for i in idx]
        read_ids = [read_ids[i] for i in idx]
        n = max_segments

    all_signals = np.stack(segments, axis=0).astype(np.float32)
    # WT 没有确定性标签，设为 -1
    all_labels = np.full(n, -1, dtype=np.int8)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.create_dataset(
            "signal_segments", data=all_signals, dtype="float32",
            compression="gzip", compression_opts=4,
        )
        f.create_dataset("labels", data=all_labels, dtype="int8")

        dt = h5py.string_dtype()
        f.create_dataset("read_ids", data=read_ids, dtype=dt)
        f.create_dataset("mod_types", data=["5mC"] * n, dtype=dt)

        f.attrs["num_samples"] = n
        f.attrs["segment_length"] = segment_length
        f.attrs["source"] = str(wt_pod5)
        f.attrs["dataset_type"] = "test_wt"

    logger.info(f"Saved {n} WT test samples to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Prepare E. coli 5mC CpG benchmark data for methylation detection"
    )
    parser.add_argument(
        "--benchmark_dir", type=str,
        default="/owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain/methylation_data/ont_benchmark",
        help="Path to the ONT benchmark data directory",
    )
    parser.add_argument("--output_dir", type=str,
                        default="/owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain/methylation_data",
                        help="Output directory for HDF5 files")
    parser.add_argument("--segment_length", type=int, default=10240)
    parser.add_argument("--stride", type=int, default=5120,
                        help="Sliding window stride (default: segment_length/2)")
    parser.add_argument("--max_reads", type=int, default=5000,
                        help="Max reads per POD5 file (0=all)")
    parser.add_argument("--max_segments_per_class", type=int, default=50000,
                        help="Max segments per class for training")
    parser.add_argument("--max_test_segments", type=int, default=20000,
                        help="Max segments for WT test set")
    parser.add_argument("--normalize", type=str, default="mad",
                        choices=["mad", "zscore", "none"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_train", action="store_true",
                        help="Skip training data generation")
    parser.add_argument("--skip_test", action="store_true",
                        help="Skip test data generation")

    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir)
    output_dir = Path(args.output_dir)

    # 数据路径
    pod5_dir = benchmark_dir / "pod5"
    pos_pod5 = pod5_dir / "Ecoli_DM_MSssI_5kHz.pod5"   # 阳性: 全 CpG 甲基化
    neg_pod5 = pod5_dir / "Ecoli_DM_5kHz.pod5"          # 阴性: 无甲基化
    wt_pod5 = pod5_dir / "Ecoli_WT_5kHz.pod5"           # 测试: 天然甲基化

    # 验证文件
    for f in [pos_pod5, neg_pod5, wt_pod5]:
        if not f.exists():
            logger.error(f"File not found: {f}")
            sys.exit(1)

    # 1. 生成训练数据 (DM_MSssI + DM)
    if not args.skip_train:
        train_h5 = str(output_dir / "ecoli_5mC_train.h5")
        build_dataset(
            positive_pod5=str(pos_pod5),
            negative_pod5=str(neg_pod5),
            output_path=train_h5,
            segment_length=args.segment_length,
            stride=args.stride,
            max_reads_per_class=args.max_reads,
            max_segments_per_class=args.max_segments_per_class,
            normalize=args.normalize,
            seed=args.seed,
        )

    # 2. 生成测试数据 (WT)
    if not args.skip_test:
        test_h5 = str(output_dir / "ecoli_5mC_test_wt.h5")
        build_test_dataset(
            wt_pod5=str(wt_pod5),
            output_path=test_h5,
            segment_length=args.segment_length,
            stride=args.stride,
            max_reads=args.max_reads,
            max_segments=args.max_test_segments,
            normalize=args.normalize,
            seed=args.seed,
        )

    logger.info("\n=== Data preparation complete! ===")
    logger.info(f"Training data: {output_dir / 'ecoli_5mC_train.h5'}")
    logger.info(f"Test data: {output_dir / 'ecoli_5mC_test_wt.h5'}")


if __name__ == "__main__":
    main()
