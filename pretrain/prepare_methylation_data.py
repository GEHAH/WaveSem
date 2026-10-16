#!/usr/bin/env python3
"""
prepare_methylation_data.py
=============================
从 ONT 测序数据中提取原始信号并与甲基化标签对齐，
生成用于甲基化检测下游任务的 HDF5 训练集。

用途：基于 WaveSem 预训练 encoder 的甲基化检测实验。

支持的修饰类型：
  - 5mC (CpG 上下文中的 5-甲基胞嘧啶)
  - 6mA (GATC 上下文中的 N6-甲基腺嘌呤)

数据来源：
  1. ONT POD5/FAST5 原始信号
  2. BAM 文件 (使用 dorado basecaller 生成，含 MM/ML 标签)
  3. 可选: 外部 ground truth (如 bisulfite-seq BED 文件)

输出格式：
  HDF5 文件，包含:
    - signal_segments: [N, segment_length] float32
    - labels: [N] int8, 0=未修饰, 1=已修饰
    - mod_types: [N] string, "5mC" or "6mA"
    - positions: [N] int64, 参考基因组上的位置
    - read_ids: [N] string
"""

import os
import sys
import argparse
import numpy as np
import h5py
import pysam
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ─── 方法 1: 从 Dorado BAM 的 MM/ML 标签提取修饰信息 ────────────────────────

def extract_mods_from_bam(
    bam_path: str,
    mod_code: str = "m",  # "m" = 5mC, "a" = 6mA
    min_qual: int = 0,
    min_prob: float = 0.8,
    max_reads: int = 0,
) -> Dict[str, List[Dict]]:
    """
    从 Dorado basecaller 输出的 BAM 文件中提取修饰碱基信息。

    Dorado 使用 SAM 规范的 MM/ML 标签存储修饰信息：
      - MM: 修饰类型和位置
      - ML: 修饰概率 (0-255, 映射到 0.0-1.0)

    Args:
        bam_path: BAM 文件路径
        mod_code: 修饰代码, "m"=5mC, "a"=6mA
        min_qual: 最低 mapping quality
        min_prob: 最低修饰概率阈值 (0-1)
        max_reads: 最大读取数 (0=不限)

    Returns:
        dict: {read_id: [{"pos": int, "prob": float, "label": int, "ref_pos": int}, ...]}
    """
    logger.info(f"Extracting modifications (code={mod_code}) from {bam_path}")

    mod_data = defaultdict(list)
    count = 0

    with pysam.AlignmentFile(bam_path, "rb") as bam:
        for read in tqdm(bam, desc="Scanning BAM for modifications"):
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            if read.mapping_quality < min_qual:
                continue

            read_id = read.query_name

            # 获取修饰信息
            if not read.has_tag("MM") or not read.has_tag("ML"):
                continue

            try:
                mod_bases = read.modified_bases
            except Exception:
                continue

            if mod_bases is None:
                continue

            # mod_bases 格式: {(base, strand, mod_code): [(pos, prob), ...]}
            for key, positions_probs in mod_bases.items():
                base_char, strand, code = key
                if code != ord(mod_code) and chr(code) != mod_code:
                    continue

                for query_pos, quality in positions_probs:
                    prob = quality / 255.0  # 转换为 0-1 概率

                    # 获取参考坐标
                    ref_positions = read.get_reference_positions(full_length=True)
                    if query_pos < len(ref_positions) and ref_positions[query_pos] is not None:
                        ref_pos = ref_positions[query_pos]
                    else:
                        ref_pos = -1

                    label = 1 if prob >= min_prob else 0

                    mod_data[read_id].append({
                        "query_pos": query_pos,
                        "ref_pos": ref_pos,
                        "prob": prob,
                        "label": label,
                        "strand": strand,
                    })

            count += 1
            if max_reads > 0 and count >= max_reads:
                break

    logger.info(f"Extracted modifications from {count} reads, {len(mod_data)} have mods")
    return dict(mod_data)


# ─── 方法 2: 从外部 BED 文件获取 ground truth ────────────────────────────────

def load_ground_truth_bed(
    bed_path: str,
    min_coverage: int = 10,
    methylation_threshold: float = 0.5,
) -> Dict[Tuple[str, int], int]:
    """
    加载 bisulfite-seq 或 modkit 生成的 BED 格式甲基化数据作为 ground truth。

    BED 格式 (bedMethyl):
      chrom  start  end  name  score  strand  ...  coverage  percent_modified

    Args:
        bed_path: BED 文件路径 (可以是 .gz 压缩)
        min_coverage: 最低覆盖度
        methylation_threshold: 甲基化比率阈值

    Returns:
        dict: {(chrom, position): label}
    """
    import gzip

    logger.info(f"Loading ground truth from {bed_path}")
    gt = {}

    open_fn = gzip.open if bed_path.endswith(".gz") else open
    with open_fn(bed_path, "rt") as f:
        for line in f:
            if line.startswith("#") or line.startswith("track"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 11:
                continue

            chrom = parts[0]
            start = int(parts[1])
            coverage = int(parts[9]) if len(parts) > 9 else 0
            pct_mod = float(parts[10]) if len(parts) > 10 else 0.0

            if coverage < min_coverage:
                continue

            label = 1 if pct_mod >= (methylation_threshold * 100) else 0
            gt[(chrom, start)] = label

    logger.info(f"Loaded {len(gt)} ground truth positions")
    return gt


# ─── 信号提取与标签对齐 ──────────────────────────────────────────────────────

def extract_signal_with_labels(
    pod5_path: str,
    mod_data: Dict[str, List[Dict]],
    segment_length: int = 10240,
    context_bases: int = 10,
    samples_per_base: int = 10,  # 典型值：ONT 每个碱基 ~8-15 个采样点
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    从 POD5 文件中提取原始信号，并与甲基化标签对齐。

    对齐策略：
      - 使用修饰位点周围 ±context_bases 的信号窗口
      - 窗口大小 = (2*context_bases + 1) * samples_per_base
      - 如果窗口小于 segment_length，进行 padding
      - 如果窗口大于 segment_length，居中裁剪

    Args:
        pod5_path: POD5 文件/目录路径
        mod_data: 从 BAM 提取的修饰信息
        segment_length: 信号片段长度 (应与预训练一致)
        context_bases: 修饰位点周围的碱基数
        samples_per_base: 每个碱基对应的采样点数

    Returns:
        signals: [N, segment_length] float32
        labels: [N] int8
        read_ids: list of str
        mod_types: list of str
    """
    try:
        import pod5
    except ImportError:
        logger.error("pod5 package not installed. Run: pip install pod5")
        return np.array([]), np.array([]), [], []

    signals_list = []
    labels_list = []
    read_ids_list = []
    mod_types_list = []

    # 计算窗口参数
    window_bases = 2 * context_bases + 1
    window_samples = window_bases * samples_per_base

    pod5_path = Path(pod5_path)
    if pod5_path.is_dir():
        pod5_files = list(pod5_path.glob("*.pod5"))
    else:
        pod5_files = [pod5_path]

    for p5_file in tqdm(pod5_files, desc="Processing POD5 files"):
        try:
            reader = pod5.Reader(p5_file)
        except Exception as e:
            logger.warning(f"Cannot open {p5_file}: {e}")
            continue

        for read in reader.reads():
            rid = str(read.read_id)
            if rid not in mod_data:
                continue

            signal = read.signal  # 原始 int16 信号
            signal_f = signal.astype(np.float32)

            # 归一化 (与预训练保持一致)
            # 你的预训练使用 none，这里也保持原始值
            # 但甲基化检测通常受益于归一化
            signal_f = (signal_f - np.median(signal_f)) / max(np.std(signal_f), 1e-6)

            sig_len = len(signal_f)

            for mod_info in mod_data[rid]:
                query_pos = mod_info["query_pos"]
                label = mod_info["label"]

                # 估算信号中对应的位置
                # 粗略估计：signal_position ≈ query_pos * samples_per_base
                center_sample = int(query_pos * samples_per_base)
                start_sample = max(0, center_sample - window_samples // 2)
                end_sample = min(sig_len, start_sample + segment_length)
                start_sample = max(0, end_sample - segment_length)

                # 提取信号窗口
                sig_window = signal_f[start_sample:end_sample]

                # Padding 到 segment_length
                if len(sig_window) < segment_length:
                    pad_len = segment_length - len(sig_window)
                    sig_window = np.pad(sig_window, (0, pad_len), mode="constant", value=0.0)

                signals_list.append(sig_window[:segment_length])
                labels_list.append(label)
                read_ids_list.append(rid)
                mod_types_list.append("5mC")

        reader.close()

    if len(signals_list) == 0:
        return np.array([]), np.array([]), [], []

    signals = np.stack(signals_list, axis=0).astype(np.float32)
    labels = np.array(labels_list, dtype=np.int8)

    return signals, labels, read_ids_list, mod_types_list


# ─── 保存为 HDF5 ─────────────────────────────────────────────────────────────

def save_methylation_h5(
    output_path: str,
    signals: np.ndarray,
    labels: np.ndarray,
    read_ids: List[str],
    mod_types: List[str],
):
    """保存甲基化训练数据为 HDF5 格式"""
    logger.info(f"Saving {len(signals)} samples to {output_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with h5py.File(output_path, "w") as f:
        f.create_dataset("signal_segments", data=signals, dtype="float32",
                         compression="gzip", compression_opts=4)
        f.create_dataset("labels", data=labels, dtype="int8")

        # 字符串数据
        dt = h5py.string_dtype()
        f.create_dataset("read_ids", data=read_ids, dtype=dt)
        f.create_dataset("mod_types", data=mod_types, dtype=dt)

        # 元信息
        f.attrs["num_samples"] = len(signals)
        f.attrs["segment_length"] = signals.shape[1]
        f.attrs["num_positive"] = int(np.sum(labels == 1))
        f.attrs["num_negative"] = int(np.sum(labels == 0))

    logger.info(
        f"Saved: {len(signals)} samples "
        f"(positive={int(np.sum(labels==1))}, negative={int(np.sum(labels==0))})"
    )


# ─── 使用 Dorado + modkit 生成 ground truth 的工作流 ─────────────────────────

def print_dorado_workflow():
    """打印使用 Dorado basecaller 生成修饰检测数据的完整工作流"""
    workflow = """
╔══════════════════════════════════════════════════════════════════════════╗
║           ONT 甲基化检测数据准备工作流 (Dorado + modkit)                  ║
╚══════════════════════════════════════════════════════════════════════════╝

步骤 1: 使用 Dorado 进行带修饰碱基检测的 basecalling
─────────────────────────────────────────────────────
  # 安装 Dorado (ONT 官方 basecaller)
  # https://github.com/nanoporetech/dorado

  # 5mC 检测 (CpG 上下文)
  dorado basecaller \\
    sup \\
    /path/to/pod5/ \\
    --modified-bases 5mCG \\
    --reference /path/to/hg38.fa \\
    > methylation_calls.bam

  # 6mA 检测
  dorado basecaller \\
    sup \\
    /path/to/pod5/ \\
    --modified-bases 6mA \\
    --reference /path/to/hg38.fa \\
    > 6mA_calls.bam

  # 排序和索引
  samtools sort methylation_calls.bam -o methylation_calls.sorted.bam
  samtools index methylation_calls.sorted.bam

步骤 2: 使用 modkit 生成 bedMethyl 格式的修饰汇总
─────────────────────────────────────────────────────
  # 安装 modkit: https://github.com/nanoporetech/modkit
  # cargo install modkit 或下载二进制

  # 5mC pileup
  modkit pileup \\
    methylation_calls.sorted.bam \\
    5mC_pileup.bed \\
    --ref /path/to/hg38.fa \\
    --cpg \\
    --combine-strands \\
    --filter-threshold 0.75

  # 6mA pileup
  modkit pileup \\
    6mA_calls.sorted.bam \\
    6mA_pileup.bed \\
    --ref /path/to/hg38.fa \\
    --filter-threshold 0.75

步骤 3: 运行本脚本生成训练数据
─────────────────────────────
  python prepare_methylation_data.py \\
    --pod5_path /path/to/pod5/ \\
    --bam_path methylation_calls.sorted.bam \\
    --mod_code m \\
    --output methylation_data/5mC_train.h5 \\
    --segment_length 10240

  python prepare_methylation_data.py \\
    --pod5_path /path/to/pod5/ \\
    --bam_path 6mA_calls.sorted.bam \\
    --mod_code a \\
    --output methylation_data/6mA_train.h5 \\
    --segment_length 10240

步骤 4 (可选): 使用 bisulfite-seq 作为 ground truth 验证
──────────────────────────────────────────────────────
  python prepare_methylation_data.py \\
    --pod5_path /path/to/pod5/ \\
    --bam_path methylation_calls.sorted.bam \\
    --ground_truth_bed NA12878_WGBS.bed.gz \\
    --mod_code m \\
    --output methylation_data/5mC_validated.h5
"""
    print(workflow)


# ─── 主函数 ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prepare methylation detection training data from ONT sequencing"
    )
    parser.add_argument("--pod5_path", type=str, required=True,
                        help="Path to POD5 file or directory")
    parser.add_argument("--bam_path", type=str, required=True,
                        help="Path to BAM file with MM/ML tags (from Dorado)")
    parser.add_argument("--mod_code", type=str, default="m",
                        choices=["m", "a"],
                        help="Modification code: 'm'=5mC, 'a'=6mA")
    parser.add_argument("--ground_truth_bed", type=str, default=None,
                        help="Optional: BED file with ground truth methylation calls")
    parser.add_argument("--output", type=str, required=True,
                        help="Output HDF5 file path")
    parser.add_argument("--segment_length", type=int, default=10240,
                        help="Signal segment length (should match pretrain)")
    parser.add_argument("--context_bases", type=int, default=10,
                        help="Number of bases around modification site")
    parser.add_argument("--min_prob", type=float, default=0.8,
                        help="Minimum modification probability for positive label")
    parser.add_argument("--max_reads", type=int, default=0,
                        help="Maximum reads to process (0=all)")
    parser.add_argument("--workflow", action="store_true",
                        help="Print Dorado + modkit workflow and exit")

    args = parser.parse_args()

    if args.workflow:
        print_dorado_workflow()
        return

    # Step 1: 提取修饰信息
    mod_data = extract_mods_from_bam(
        bam_path=args.bam_path,
        mod_code=args.mod_code,
        min_prob=args.min_prob,
        max_reads=args.max_reads,
    )

    if len(mod_data) == 0:
        logger.error("No modification data found in BAM file. "
                      "Make sure Dorado was run with --modified-bases flag.")
        sys.exit(1)

    # Step 2: 提取信号并对齐标签
    signals, labels, read_ids, mod_types = extract_signal_with_labels(
        pod5_path=args.pod5_path,
        mod_data=mod_data,
        segment_length=args.segment_length,
        context_bases=args.context_bases,
    )

    if len(signals) == 0:
        logger.error("No signal segments extracted. Check POD5 and BAM compatibility.")
        sys.exit(1)

    # Step 3: 保存
    save_methylation_h5(
        output_path=args.output,
        signals=signals,
        labels=labels,
        read_ids=read_ids,
        mod_types=mod_types,
    )

    logger.info("Done!")


if __name__ == "__main__":
    main()
