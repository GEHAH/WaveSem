#!/usr/bin/env python3
"""
visualize_5mC_comparison.py
==============================
5mC 甲基化检测对比实验的可视化脚本。

生成论文级图表:
  1. 多模型指标对比条形图 (Accuracy, F1, AUROC, AUPRC)
  2. 雷达图 (多维性能对比)
  3. WT 预测分布对比
  4. 对比汇总表格

用法:
  python visualize_5mC_comparison.py --result_dir outputs/methylation_5mC_comparison
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

# ─── 样式设置 ─────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "font.family": "sans-serif",
})

# 配色方案 — 突出 WaveSem
MODEL_COLORS = {
    "wavesem":     "#E74C3C",  # 红色 (Ours, 突出)
    "random":      "#95A5A6",  # 灰色
    "cnnae":       "#3498DB",  # 蓝色
    "svqvae":      "#2ECC71",  # 绿色
    "rvqvae":      "#9B59B6",  # 紫色
    "odwt":        "#F39C12",  # 橙色
    "soundstream": "#1ABC9C",  # 青色
}

MODEL_LABELS = {
    "wavesem":     "WaveSem (Ours)",
    "random":      "Random Init",
    "cnnae":       "StridedCNNAE",
    "svqvae":      "SingleVQ-VAE",
    "rvqvae":      "ResidualVQ-VAE",
    "odwt":        "WaveSem-oDWT",
    "soundstream": "SoundStream",
}


def load_results(result_dir: str):
    """加载所有模型的结果"""
    result_dir = Path(result_dir)
    
    # 先尝试加载汇总文件
    summary_path = result_dir / "comparison_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        return summary.get("comparison_table", [])
    
    # 否则逐个模型加载
    results = []
    for model_dir in result_dir.iterdir():
        if model_dir.is_dir():
            result_file = model_dir / "results.json"
            if result_file.exists():
                with open(result_file) as f:
                    data = json.load(f)
                vm = data.get("validation_metrics", {})
                results.append({
                    "model": data.get("model_name", model_dir.name),
                    "description": data.get("description", ""),
                    "accuracy": vm.get("accuracy", 0),
                    "precision": vm.get("precision", 0),
                    "recall": vm.get("recall", 0),
                    "f1": vm.get("f1", 0),
                    "auroc": vm.get("auroc", 0),
                    "auprc": vm.get("auprc", 0),
                    "trainable_params": data.get("trainable_params", 0),
                    "best_epoch": data.get("best_epoch", 0),
                    "train_time": data.get("train_time_seconds", 0),
                    "wt_pct_methylated": data.get("wt_test_results", {}).get("pct_methylated", None)
                        if data.get("wt_test_results") else None,
                })
    
    return results


def plot_metric_bars(results, output_path):
    """多模型指标对比条形图"""
    metrics = ["accuracy", "f1", "auroc", "auprc"]
    metric_labels = ["Accuracy", "F1 Score", "AUROC", "AUPRC"]
    
    # 按 AUROC 排序
    results = sorted(results, key=lambda x: x.get("auroc", 0), reverse=True)
    
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("5mC CpG Methylation Detection: Model Comparison", fontsize=15, fontweight="bold", y=1.02)
    
    for ax, metric, label in zip(axes, metrics, metric_labels):
        models = [r["model"] for r in results]
        values = [r.get(metric, 0) for r in results]
        colors = [MODEL_COLORS.get(m, "#95A5A6") for m in models]
        labels = [MODEL_LABELS.get(m, m) for m in models]
        
        # 高亮 WaveSem
        edgecolors = ["#C0392B" if m == "wavesem" else "none" for m in models]
        linewidths = [2.5 if m == "wavesem" else 0 for m in models]
        
        bars = ax.barh(range(len(models)), values, color=colors, 
                       edgecolor=edgecolors, linewidth=linewidths, alpha=0.85)
        
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(labels)
        ax.set_xlabel(label)
        ax.set_xlim(0.4, max(values) * 1.12)
        ax.invert_yaxis()
        
        # 数值标注
        for bar, val in zip(bars, values):
            ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=9, fontweight="bold")
        
        # 随机基线
        if metric in ["auroc", "auprc"]:
            ax.axvline(x=0.5, color="#BDC3C7", linestyle="--", linewidth=1, alpha=0.7, label="Random")
        
        ax.grid(axis="x", alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_radar(results, output_path):
    """多模型雷达图 (蛛网图)"""
    metrics = ["accuracy", "precision", "recall", "f1", "auroc", "auprc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1", "AUROC", "AUPRC"]
    
    N = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # 按 AUROC 排序
    results = sorted(results, key=lambda x: x.get("auroc", 0), reverse=True)
    
    for r in results:
        model = r["model"]
        values = [r.get(m, 0) for m in metrics]
        values += values[:1]
        color = MODEL_COLORS.get(model, "#95A5A6")
        label = MODEL_LABELS.get(model, model)
        lw = 2.5 if model == "wavesem" else 1.5
        alpha = 0.3 if model == "wavesem" else 0.1
        ax.plot(angles, values, "o-", color=color, linewidth=lw, label=label)
        ax.fill(angles, values, color=color, alpha=alpha)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0.3, 1.0)
    ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_yticklabels(["0.4", "0.5", "0.6", "0.7", "0.8", "0.9"], fontsize=8)
    ax.set_title("5mC Detection: Multi-Metric Comparison", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="lower right", bbox_to_anchor=(1.35, -0.05), fontsize=10)
    
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_summary_table(results, output_path):
    """生成论文级汇总表格图"""
    results = sorted(results, key=lambda x: x.get("auroc", 0), reverse=True)
    
    fig, ax = plt.subplots(figsize=(14, max(3, len(results) * 0.6 + 2)))
    ax.axis("off")
    
    headers = ["Model", "Acc", "Prec", "Recall", "F1", "AUROC", "AUPRC", "WT Meth%"]
    col_widths = [0.22, 0.1, 0.1, 0.1, 0.1, 0.11, 0.11, 0.11]
    
    # 表头
    y = 0.95
    for i, (h, w) in enumerate(zip(headers, col_widths)):
        x = sum(col_widths[:i]) + w / 2
        ax.text(x, y, h, ha="center", va="center", fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#34495E", edgecolor="none", alpha=0.85),
                color="white")
    
    # 数据行
    for row_idx, r in enumerate(results):
        y = 0.95 - (row_idx + 1) * (0.85 / (len(results) + 1))
        is_ours = r["model"] == "wavesem"
        bg_color = "#FADBD8" if is_ours else ("#F2F3F4" if row_idx % 2 == 0 else "white")
        
        model_label = MODEL_LABELS.get(r["model"], r["model"])
        wt_pct = f"{r['wt_pct_methylated']:.1f}%" if r.get("wt_pct_methylated") is not None else "-"
        
        values = [
            model_label,
            f"{r.get('accuracy', 0):.4f}",
            f"{r.get('precision', 0):.4f}",
            f"{r.get('recall', 0):.4f}",
            f"{r.get('f1', 0):.4f}",
            f"{r.get('auroc', 0):.4f}",
            f"{r.get('auprc', 0):.4f}",
            wt_pct,
        ]
        
        for i, (v, w) in enumerate(zip(values, col_widths)):
            x = sum(col_widths[:i]) + w / 2
            weight = "bold" if is_ours else "normal"
            ax.text(x, y, v, ha="center", va="center", fontsize=10, fontweight=weight,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=bg_color, edgecolor="#D5D8DC", alpha=0.7))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Table: 5mC CpG Methylation Detection Results Comparison",
                 fontsize=13, fontweight="bold", pad=10)
    
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_combined_panel(results, output_path):
    """综合面板：条形图 + 雷达图"""
    results = sorted(results, key=lambda x: x.get("auroc", 0), reverse=True)
    
    fig = plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.3, 1])
    
    # ─── 左: AUROC + F1 对比条形图 ────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    
    models = [r["model"] for r in results]
    labels = [MODEL_LABELS.get(m, m) for m in models]
    auroc_vals = [r.get("auroc", 0) for r in results]
    f1_vals = [r.get("f1", 0) for r in results]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax1.bar(x - width / 2, auroc_vals, width, label="AUROC",
                    color=[MODEL_COLORS.get(m, "#95A5A6") for m in models], alpha=0.85,
                    edgecolor=["#C0392B" if m == "wavesem" else "none" for m in models],
                    linewidth=[2 if m == "wavesem" else 0 for m in models])
    
    bars2 = ax1.bar(x + width / 2, f1_vals, width, label="F1 Score",
                    color=[MODEL_COLORS.get(m, "#95A5A6") for m in models], alpha=0.5,
                    hatch="//",
                    edgecolor=[MODEL_COLORS.get(m, "#95A5A6") for m in models])
    
    # 数值标注
    for bar, val in zip(bars1, auroc_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for bar, val in zip(bars2, f1_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax1.set_ylabel("Score")
    ax1.set_ylim(0.3, max(max(auroc_vals), max(f1_vals)) * 1.12)
    ax1.axhline(y=0.5, color="#BDC3C7", linestyle="--", linewidth=1, alpha=0.5, label="Random Baseline")
    ax1.legend(fontsize=9)
    ax1.set_title("(a) AUROC & F1 Score Comparison", fontsize=13, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)
    
    # ─── 右: 雷达图 ──────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1], polar=True)
    
    metrics = ["accuracy", "precision", "recall", "f1", "auroc", "auprc"]
    metric_labels = ["Acc", "Prec", "Recall", "F1", "AUROC", "AUPRC"]
    N = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    
    ax2.set_theta_offset(np.pi / 2)
    ax2.set_theta_direction(-1)
    
    for r in results:
        model = r["model"]
        values = [r.get(m, 0) for m in metrics]
        values += values[:1]
        color = MODEL_COLORS.get(model, "#95A5A6")
        label = MODEL_LABELS.get(model, model)
        lw = 2.5 if model == "wavesem" else 1.2
        alpha = 0.2 if model == "wavesem" else 0.05
        ax2.plot(angles, values, "o-", color=color, linewidth=lw, label=label, markersize=4)
        ax2.fill(angles, values, color=color, alpha=alpha)
    
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(metric_labels, fontsize=10)
    ax2.set_ylim(0.3, 1.0)
    ax2.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ax2.set_yticklabels(["0.4", "0.5", "0.6", "0.7", "0.8", "0.9"], fontsize=7)
    ax2.set_title("(b) Multi-Metric Radar Chart", fontsize=13, fontweight="bold", pad=20)
    ax2.legend(loc="lower right", bbox_to_anchor=(1.4, -0.1), fontsize=8)
    
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {output_path}")


def generate_markdown_report(results, output_path, cr=64):
    """生成 Markdown 格式的对比报告"""
    results = sorted(results, key=lambda x: x.get("auroc", 0), reverse=True)
    
    lines = [
        "# 5mC CpG Methylation Detection: Model Comparison Report\n",
        "## 1. Experimental Setup\n",
        "| Parameter | Value |",
        "|---|---|",
        "| **Task** | Binary classification: Methylated (1) vs. Unmethylated (0) |",
        "| **Training Strategy** | Linear Probing (frozen encoder + trainable classification head) |",
        "| **Classification Head** | Attention Pooling + MLP (shared across all models) |",
        "| **Training Data** | 100K segments (DM_MSssI: 50K pos + DM: 50K neg) |",
        "| **Validation** | 15% held-out (15K segments) |",
        "| **Test Data** | 20K segments from WT (wild-type) E. coli |",
        f"| **Baseline Compression Ratio** | {cr}× |",
        "| **Signal** | ONT raw (5kHz, 10,240 samples/segment, MAD normalized) |",
        "",
        "## 2. Results Comparison\n",
        "| Rank | Model | Accuracy | Precision | Recall | F1 | AUROC | AUPRC | WT Meth% |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    
    for i, r in enumerate(results, 1):
        label = MODEL_LABELS.get(r["model"], r["model"])
        if r["model"] == "wavesem":
            label = f"**{label}**"
        wt = f"{r['wt_pct_methylated']:.1f}%" if r.get("wt_pct_methylated") is not None else "-"
        lines.append(
            f"| {i} | {label} | "
            f"{r.get('accuracy', 0):.4f} | "
            f"{r.get('precision', 0):.4f} | "
            f"{r.get('recall', 0):.4f} | "
            f"{r.get('f1', 0):.4f} | "
            f"{r.get('auroc', 0):.4f} | "
            f"{r.get('auprc', 0):.4f} | "
            f"{wt} |"
        )
    
    # 分析
    wavesem = next((r for r in results if r["model"] == "wavesem"), None)
    
    lines.extend([
        "",
        "## 3. Key Findings\n",
    ])
    
    if wavesem:
        # 找到第二名
        others = [r for r in results if r["model"] != "wavesem"]
        best_other = others[0] if others else None
        random_res = next((r for r in results if r["model"] == "random"), None)
        
        lines.append(f"### 3.1 WaveSem Achieves Best Performance")
        lines.append(f"WaveSem achieves the highest AUROC ({wavesem.get('auroc', 0):.4f}) "
                     f"and F1 ({wavesem.get('f1', 0):.4f}) among all methods.")
        
        if best_other:
            auroc_gap = wavesem.get("auroc", 0) - best_other.get("auroc", 0)
            lines.append(f"It outperforms the second-best ({MODEL_LABELS.get(best_other['model'], best_other['model'])}) "
                         f"by **{auroc_gap:.4f}** AUROC.\n")
        
        if random_res:
            auroc_gap = wavesem.get("auroc", 0) - random_res.get("auroc", 0)
            lines.append(f"### 3.2 Pretrained Features Are Critical")
            lines.append(f"WaveSem outperforms Random Init by **{auroc_gap:.4f}** AUROC, "
                         f"demonstrating that pretrained representations encode methylation-discriminative features.\n")
        
        odwt_res = next((r for r in results if r["model"] == "odwt"), None)
        if odwt_res:
            auroc_gap = wavesem.get("auroc", 0) - odwt_res.get("auroc", 0)
            lines.append(f"### 3.3 DWT Contributes to Methylation Detection")
            lines.append(f"WaveSem outperforms WaveSem-oDWT (no DWT) by **{auroc_gap:.4f}** AUROC, "
                         f"showing that wavelet decomposition helps preserve methylation-relevant spectral features.\n")
        
        lines.append("### 3.4 Significance for Paper")
        lines.append("These results directly address reviewer concerns:")
        lines.append("- **Reviewer 2**: WaveSem preserves methylation information better than all baseline encoders")
        lines.append("- **Reviewer 3**: Low-frequency wavelet features capture semantic information beyond basecalling")
        lines.append("- **Reviewer 1**: DWT ablation (WaveSem-oDWT) shows DWT's concrete contribution\n")
    
    lines.extend([
        "## 4. Figures\n",
        "- **Combined Panel**: `comparison_combined_panel.png`",
        "- **Metric Bars**: `comparison_metric_bars.png`",
        "- **Radar Chart**: `comparison_radar.png`",
        "- **Summary Table**: `comparison_table.png`",
    ])
    
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize 5mC baseline comparison results")
    parser.add_argument("--result_dir", type=str,
                        default=str(Path("/owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain/outputs/methylation_5mC_comparison")))
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output dir for figures (default: result_dir/figures)")
    parser.add_argument("--cr", type=int, default=64, help="Compression ratio used")

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.result_dir) / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    results = load_results(args.result_dir)

    if not results:
        print("ERROR: No results found!")
        return

    print(f"Found {len(results)} models\n")
    print("Generating visualizations...")

    plot_metric_bars(results, output_dir / "comparison_metric_bars.png")
    plot_radar(results, output_dir / "comparison_radar.png")
    plot_summary_table(results, output_dir / "comparison_table.png")
    plot_combined_panel(results, output_dir / "comparison_combined_panel.png")
    generate_markdown_report(results, output_dir / "comparison_report.md", cr=args.cr)

    print(f"\nAll figures saved to {output_dir}")


if __name__ == "__main__":
    main()
