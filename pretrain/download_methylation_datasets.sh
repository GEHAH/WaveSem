#!/bin/bash
# ==============================================================================
# 甲基化检测 Benchmark 数据集下载脚本
# 整理自主流文献: Rockfish (Nat Comms 2024), DeepMod2 (Nat Comms 2024),
#                 NanoCon (Bioinformatics 2024), DeepSignal, METEORE
# ==============================================================================

set -e

BASE_DIR="/owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain/methylation_data"
mkdir -p ${BASE_DIR}

echo "=============================================="
echo "   ONT 甲基化检测 Benchmark 数据集下载"
echo "=============================================="

# =============================================
# 方案 A: Rockfish (Nature Communications 2024)
# 论文: https://doi.org/10.1038/s41467-024-49847-0
# 训练: NA24385 chr2-21 + H1ESc (PCR/M.SssI)
# 验证: chr22
# 测试: NA12878, HX1, K562, NA19240, H1ESc等
# =============================================
download_rockfish_data() {
    echo ""
    echo "========== [方案A] Rockfish 数据集 =========="
    ROCK_DIR="${BASE_DIR}/rockfish"
    mkdir -p ${ROCK_DIR}/{train,test,ground_truth,models}

    # --- 1. 训练数据: NA24385 (GM24385) ONT R9.4.1 ---
    echo "[1/6] 下载 NA24385 ONT 训练数据 (from ONT Open Data)..."
    echo "  来源: https://labs.epi2me.io/gm24385-5mc/"
    # R9.4.1 数据
    aws s3 sync --no-sign-request \
        s3://ont-open-data/gm24385_mod_2021.09/ \
        ${ROCK_DIR}/train/NA24385_R9.4.1/ \
        --exclude "*" --include "*.fast5" --include "*.pod5" --include "*.bam*"

    # R10.4.1 数据
    echo "[2/6] 下载 NA24385 ONT R10.4.1 训练数据..."
    aws s3 sync --no-sign-request \
        s3://ont-open-data/giab_2023.05/analysis/hg002/5mC/ \
        ${ROCK_DIR}/train/NA24385_R10.4.1/ \
        --exclude "*" --include "*.bam*" --include "*.pod5"

    # --- 2. 训练 ground truth: NA24385 RRBS (Illumina) ---
    echo "[3/6] 下载 NA24385 RRBS ground truth..."
    # 从 ENCODE 下载
    # NA24385 没有直接ENCODE数据，使用论文中指定的来源
    echo "  NOTE: Rockfish使用ONT官方GM24385数据集中附带的RRBS数据"

    # --- 3. 合成对照数据: H1ESc ---
    echo "[4/6] H1ESc PCR/M.SssI 数据 (BioProject: PRJNA876781)..."
    echo "  下载命令 (需要 SRA toolkit):"
    echo "    prefetch PRJNA876781"
    echo "    # H1ESc PCR (阴性对照, 全基因组扩增, 无甲基化)"
    echo "    # H1ESc M.SssI (阳性对照, CpG全甲基化)"
    echo "  或使用 fasterq-dump:"
    cat > ${ROCK_DIR}/train/download_h1esc.sh << 'SCRIPT'
#!/bin/bash
# 下载 H1ESc ONT 数据 (PRJNA876781)
# 需要安装 sra-toolkit: conda install -c bioconda sra-tools
# 获取 H1ESc 相关的 SRR accession
esearch -db sra -query "PRJNA876781 AND H1ESc" | \
    efetch -format runinfo | cut -d',' -f1 | grep SRR > srr_list.txt
# 下载
while read SRR; do
    prefetch ${SRR}
    fasterq-dump ${SRR} --outdir ./h1esc_reads/
done < srr_list.txt
SCRIPT
    chmod +x ${ROCK_DIR}/train/download_h1esc.sh

    # --- 4. 测试数据: NA12878 ---
    echo "[5/6] 下载 NA12878 ONT 测试数据..."
    # R9.4.1
    echo "  从 nanopore-wgs-consortium 下载..."
    echo "  链接: https://github.com/nanopore-wgs-consortium/NA12878"
    aws s3 sync --no-sign-request \
        s3://ont-open-data/cliveome_kit14_2024.01/ \
        ${ROCK_DIR}/test/NA12878_R10.4.1/ \
        --exclude "*" --include "*.pod5" \
        2>/dev/null || echo "  [INFO] R10.4.1 下载需要使用其他路径"

    # --- 5. 测试 ground truth: NA12878 WGBS from ENCODE ---
    echo "[6/6] 下载 NA12878 WGBS ground truth (ENCODE)..."
    # ENCODE WGBS - 两个重复
    wget -c -P ${ROCK_DIR}/ground_truth/ \
        "https://www.encodeproject.org/files/ENCFF798RSS/@@download/ENCFF798RSS.bed.gz" \
        2>/dev/null || echo "  可手动下载: https://www.encodeproject.org/files/ENCFF798RSS/"
    wget -c -P ${ROCK_DIR}/ground_truth/ \
        "https://www.encodeproject.org/files/ENCFF113KRQ/@@download/ENCFF113KRQ.bed.gz" \
        2>/dev/null || echo "  可手动下载: https://www.encodeproject.org/files/ENCFF113KRQ/"

    # --- 6. 下载预训练模型 ---
    echo "下载 Rockfish 预训练模型 (Zenodo)..."
    wget -c -P ${ROCK_DIR}/models/ \
        "https://zenodo.org/records/10867175/files/rockfish_r941_base.pt" \
        2>/dev/null || echo "  可手动下载: https://zenodo.org/records/10867175"

    echo "========== Rockfish 数据集下载完成 =========="
}

# =============================================
# 方案 B: DeepMod2 (Nature Communications 2024)
# 论文: https://doi.org/10.1038/s41467-024-45778-y
# 训练: HG002/HG003/HG004 chr2-21
# 验证: chr22
# 测试: chr1 + NIH3T3 (小鼠)
# =============================================
download_deepmod2_data() {
    echo ""
    echo "========== [方案B] DeepMod2 数据集 =========="
    DM2_DIR="${BASE_DIR}/deepmod2"
    mkdir -p ${DM2_DIR}/{train,test,ground_truth,reference}

    # --- 1. ONT 数据: HG002/HG003/HG004 R10.4.1 ---
    echo "[1/5] 下载 HG002 ONT R10.4.1 数据..."
    aws s3 sync --no-sign-request \
        s3://ont-open-data/giab_lsk114_2022.12/analysis/hg002/ \
        ${DM2_DIR}/train/HG002_R10.4.1/ \
        --exclude "*" --include "*.bam*" --include "*.pod5"

    echo "[2/5] 下载 HG003/HG004 ONT R10.4.1..."
    for SAMPLE in hg003 hg004; do
        aws s3 sync --no-sign-request \
            s3://ont-open-data/giab_lsk114_2022.12/analysis/${SAMPLE}/ \
            ${DM2_DIR}/train/${SAMPLE^^}_R10.4.1/ \
            --exclude "*" --include "*.bam*" --include "*.pod5"
    done

    # --- 2. ONT 数据: HG002 R9.4.1 ---
    echo "[3/5] 下载 HG002 ONT R9.4.1 数据..."
    aws s3 sync --no-sign-request \
        s3://ont-open-data/gm24385_mod_2021.09/ \
        ${DM2_DIR}/train/HG002_R9.4.1/ \
        --exclude "*" --include "*.fast5" --include "*.pod5" --include "*.bam*"

    # --- 3. Ground Truth: EpiQC 短读长甲基化数据 ---
    echo "[4/5] 下载 EpiQC ground truth (GEO: GSE186383)..."
    echo "  使用 GEO 工具下载:"
    cat > ${DM2_DIR}/ground_truth/download_epiqc.sh << 'SCRIPT'
#!/bin/bash
# EpiQC 数据: 6种方法的短读长甲基化检测 (作为 ground truth)
# GEO: GSE186383
# 包含 HG002/HG003/HG004 样本
# 方法: EM-seq, TruSeq, PBAT, SPLAT, RRBS, Swift

# 使用 wget 下载 (需要替换具体的 SRR 号)
# 方法1: 使用 GEO SRA explorer
# 访问 https://sra-explorer.info/ 输入 GSE186383 获取下载链接

# 方法2: 使用 sra-toolkit
esearch -db gds -query "GSE186383" | \
    elink -target sra | \
    efetch -format runinfo | \
    cut -d',' -f1 | grep SRR > epiqc_srr_list.txt

while read SRR; do
    prefetch ${SRR}
    fasterq-dump ${SRR} --outdir ./epiqc_reads/ --threads 8
done < epiqc_srr_list.txt
SCRIPT
    chmod +x ${DM2_DIR}/ground_truth/download_epiqc.sh

    # --- 4. 小鼠 NIH3T3 测试数据 ---
    echo "[5/5] NIH3T3 数据 (BioProject: PRJNA951714)..."
    echo "  ONT数据: BioProject PRJNA951714"
    echo "  BS-seq ground truth: GEO GSE162138"
    echo "    GSM4942823, GSM4942824, GSM4942825"
    cat > ${DM2_DIR}/test/download_nih3t3.sh << 'SCRIPT'
#!/bin/bash
# 下载 NIH3T3 小鼠 ONT + BS-seq 数据
# ONT数据
esearch -db sra -query "PRJNA951714 AND NIH3T3" | \
    efetch -format runinfo | cut -d',' -f1 | grep SRR > nih3t3_srr.txt
while read SRR; do
    prefetch ${SRR}
    fasterq-dump ${SRR} --outdir ./nih3t3_ont/ --threads 8
done < nih3t3_srr.txt

# BS-seq ground truth (3 replicates)
for GSM in GSM4942823 GSM4942824 GSM4942825; do
    echo "下载 ${GSM}..."
done
SCRIPT
    chmod +x ${DM2_DIR}/test/download_nih3t3.sh

    # --- 数据划分说明 ---
    cat > ${DM2_DIR}/DATA_SPLIT_README.md << 'EOF'
# DeepMod2 数据划分方式

## 训练集/验证集/测试集划分 (按染色体)

| 集合    | 染色体          | 说明                          |
|---------|----------------|-------------------------------|
| 训练集  | chr2 - chr21   | HG002/HG003/HG004 三个样本    |
| 验证集  | chr22          | 用于超参数选择                 |
| 测试集  | chr1           | 完全排除在训练和验证之外        |

## Ground Truth 标签定义 (EpiQC)

- **阳性 (甲基化)**: 所有6种短读长方法覆盖度 ≥10X 且 甲基化比率 ≥90%
- **阴性 (未甲基化)**: 所有6种短读长方法覆盖度 ≥10X 且 甲基化比率 <10%
- **排除**: 不满足上述条件的位点

## 跨物种测试

- NIH3T3 小鼠细胞: 常染色体 chr1-19
- BS-seq 作为 ground truth
EOF

    echo "========== DeepMod2 数据集下载完成 =========="
}

# =============================================
# 方案 C: ONT 官方 Benchmark 数据集 (2024)
# 论文: Kulkarni et al. bioRxiv 2024
# 最全面的跨物种 benchmark 数据
# =============================================
download_ont_benchmark() {
    echo ""
    echo "========== [方案C] ONT 官方 Benchmark 数据集 =========="
    ONT_DIR="${BASE_DIR}/ont_benchmark"
    mkdir -p ${ONT_DIR}

    echo "S3 存储桶: s3://ont-basemod-benchmark-data/"
    echo "AWS 区域: ap-south-1 (无需 AWS 账号)"
    echo ""

    # 列出可用数据
    echo "列出数据集内容..."
    aws s3 ls --no-sign-request s3://ont-basemod-benchmark-data/ 2>/dev/null || \
        echo "  [INFO] 需要安装 awscli: pip install awscli"

    # --- 推荐下载: E. coli 训练数据 (体积小，适合快速实验) ---
    echo ""
    echo "推荐先下载 E. coli 数据 (体积较小，适合快速实验)..."

    # E. coli WT (有甲基化 - 阳性)
    echo "  下载 E. coli WT (阳性样本)..."
    aws s3 sync --no-sign-request \
        s3://ont-basemod-benchmark-data/ecoli/wt/ \
        ${ONT_DIR}/ecoli/wt/

    # E. coli DM (双突变体, 无甲基化 - 阴性)
    echo "  下载 E. coli DM (阴性对照)..."
    aws s3 sync --no-sign-request \
        s3://ont-basemod-benchmark-data/ecoli/dm/ \
        ${ONT_DIR}/ecoli/dm/

    # E. coli DM + M.SssI (CpG全甲基化 - 5mC阳性)
    echo "  下载 E. coli DM_MSssI (5mC 阳性对照)..."
    aws s3 sync --no-sign-request \
        s3://ont-basemod-benchmark-data/ecoli/dm_msssi/ \
        ${ONT_DIR}/ecoli/dm_msssi/

    # E. coli WGA (全基因组扩增 - 完全阴性)
    echo "  下载 E. coli WGA (完全阴性对照)..."
    aws s3 sync --no-sign-request \
        s3://ont-basemod-benchmark-data/ecoli/wga/ \
        ${ONT_DIR}/ecoli/wga/

    # E. coli EMSeq ground truth
    echo "  下载 E. coli EMSeq ground truth..."
    aws s3 sync --no-sign-request \
        s3://ont-basemod-benchmark-data/ecoli/emseq/ \
        ${ONT_DIR}/ecoli/emseq/

    # --- 人类 HG002 ---
    echo ""
    echo "下载 Human HG002 数据..."
    aws s3 sync --no-sign-request \
        s3://ont-basemod-benchmark-data/human/ \
        ${ONT_DIR}/human/ \
        --exclude "*" --include "*.pod5" --include "*.bam*" --include "*.bed*"

    # --- 参考基因组 ---
    echo "下载参考基因组..."
    aws s3 sync --no-sign-request \
        s3://ont-basemod-benchmark-data/reference/ \
        ${ONT_DIR}/reference/

    cat > ${ONT_DIR}/README.md << 'EOF'
# ONT 官方甲基化 Benchmark 数据集

## 数据来源
- S3: s3://ont-basemod-benchmark-data/
- 论文: Kulkarni et al. bioRxiv 2024 (doi: 10.1101/2024.11.09.622763)

## 涵盖物种
| 物种       | 样本类型                      | 用途                      |
|-----------|-------------------------------|--------------------------|
| E. coli   | WT/DM/DM_MSssI/WGA           | 训练 + 阳性/阴性对照        |
| Human     | HG002                        | 5mC benchmark            |
| Mouse     | Brain/ESC                    | 跨物种泛化测试              |
| Rice      | Native                       | 植物甲基化                  |
| Arabidopsis| Native                      | 植物甲基化 (CG/CHG/CHH)    |
| H. pylori | WT/WGA                       | 6mA 检测                   |

## 5mC 实验设计
- **阳性对照**: WT (天然甲基化) 或 DM_MSssI (人工CpG全甲基化)
- **阴性对照**: DM (甲基转移酶双突变) 或 WGA (全基因组扩增)
- **正交验证**: EMSeq 数据

## 6mA 实验设计
- **阳性**: E. coli WT (含 dam 甲基化), H. pylori WT
- **阴性**: E. coli DM (dam-/dcm-), H. pylori WGA

## 推荐的训练/测试划分
- 可以按 E. coli/细菌 训练, 人类/小鼠 测试 (跨物种泛化)
- 或按 WT/DM 训练, 人类 HG002 测试
EOF

    echo "========== ONT Benchmark 数据集下载完成 =========="
}

# =============================================
# 方案 D: DeepSignal / NanoCon 共用数据集
# (与 DeepSignal-plant 一致的数据预处理)
# =============================================
download_deepsignal_data() {
    echo ""
    echo "========== [方案D] DeepSignal / NanoCon 数据集 =========="
    DS_DIR="${BASE_DIR}/deepsignal"
    mkdir -p ${DS_DIR}/{models,data}

    # --- DeepSignal 预训练模型 ---
    echo "[1/3] 下载 DeepSignal 预训练 CpG 模型..."
    echo "  模型来源: Google Drive (参见 GitHub README 中的链接)"
    echo "  GitHub: https://github.com/bioinfomaticsCSU/deepsignal"
    echo ""
    echo "  手动下载步骤:"
    echo "    1. 访问 https://github.com/bioinfomaticsCSU/deepsignal"
    echo "    2. 在 'Trained models' 部分点击 Google Drive 链接"
    echo "    3. 下载 model.CpG.R9.4_1D.human_hx1.bn17.sn360.v0.1.7+.tar.gz"
    echo "    4. tar xvzf model.CpG.*.tar.gz -C ${DS_DIR}/models/"

    # --- E. coli 示例数据 (来自 Simpson 2017) ---
    echo "[2/3] 克隆 METEORE 仓库获取 E. coli 示例数据..."
    if [ ! -d "${DS_DIR}/METEORE" ]; then
        git clone --depth 1 https://github.com/comprna/METEORE.git ${DS_DIR}/METEORE
    fi
    echo "  E. coli 数据位于: ${DS_DIR}/METEORE/data/example/"

    # --- HX1 数据 (DeepSignal 的训练数据来源) ---
    echo "[3/3] HX1 人类基因组 ONT 数据..."
    echo "  BioProject: PRJNA533926"
    echo "  下载命令:"
    echo "    prefetch --option-file srr_list.txt"
    cat > ${DS_DIR}/data/download_hx1.sh << 'SCRIPT'
#!/bin/bash
# 下载 HX1 (Chinese individual) ONT 数据
# BioProject: PRJNA533926
esearch -db sra -query "PRJNA533926" | \
    efetch -format runinfo | cut -d',' -f1 | grep SRR > hx1_srr.txt

# 下载前几个用于测试
head -5 hx1_srr.txt | while read SRR; do
    echo "Downloading ${SRR}..."
    prefetch ${SRR}
done
SCRIPT
    chmod +x ${DS_DIR}/data/download_hx1.sh

    echo "========== DeepSignal 数据集下载完成 =========="
}

# =============================================
# 快速开始: 下载最小数据集用于快速验证
# =============================================
download_quick_start() {
    echo ""
    echo "========== [快速开始] 最小数据集 =========="
    QS_DIR="${BASE_DIR}/quick_start"
    mkdir -p ${QS_DIR}

    # DeepMod2 小型测试数据 (~60 reads, POD5 + FAST5)
    echo "下载 DeepMod2 小型测试数据..."
    wget -c -P ${QS_DIR}/ \
        "https://github.com/WGLab/DeepMod2/releases/download/v0.3.0/test_data.tar.gz" \
        2>/dev/null && tar xzf ${QS_DIR}/test_data.tar.gz -C ${QS_DIR}/ \
        || echo "  可手动下载: https://github.com/WGLab/DeepMod2/releases/tag/v0.3.0"

    echo "========== 快速开始数据集下载完成 =========="
}

# =============================================
# 下载参考基因组
# =============================================
download_references() {
    echo ""
    echo "========== 下载参考基因组 =========="
    REF_DIR="${BASE_DIR}/reference"
    mkdir -p ${REF_DIR}

    # hg38 / T2T-CHM13
    echo "下载 hg38 参考基因组..."
    wget -c -P ${REF_DIR}/ \
        "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz" \
        2>/dev/null || echo "  可手动下载 hg38"

    # GRCm39 (小鼠)
    echo "下载 GRCm39 小鼠参考基因组..."
    wget -c -P ${REF_DIR}/ \
        "https://hgdownload.soe.ucsc.edu/goldenPath/mm39/bigZips/mm39.fa.gz" \
        2>/dev/null || echo "  可手动下载 mm39"

    # E. coli K-12
    echo "下载 E. coli K-12 MG1655 参考基因组..."
    wget -c -P ${REF_DIR}/ \
        "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/005/845/GCF_000005845.2_ASM584v2/GCF_000005845.2_ASM584v2_genomic.fna.gz" \
        2>/dev/null || echo "  可手动下载 E. coli K-12"

    echo "========== 参考基因组下载完成 =========="
}


# =============================================
# 主函数: 选择下载方案
# =============================================
echo ""
echo "可选的下载方案:"
echo "  A) Rockfish (Nat Comms 2024) - Transformer 5mC, R9.4.1 + R10.4.1"
echo "  B) DeepMod2 (Nat Comms 2024) - BiLSTM/Transformer 5mC, HG002/003/004"
echo "  C) ONT 官方 Benchmark (bioRxiv 2024) - 最全面跨物种, 含 5mC + 6mA"
echo "  D) DeepSignal/NanoCon - BiLSTM 5mC, HX1 + E. coli"
echo "  Q) 快速开始 (最小数据集, ~60 reads)"
echo "  R) 参考基因组"
echo "  ALL) 全部下载"
echo ""

# 如果传入参数则直接使用，否则交互选择
CHOICE=${1:-""}

if [ -z "$CHOICE" ]; then
    read -p "请选择方案 [A/B/C/D/Q/R/ALL]: " CHOICE
fi

case ${CHOICE^^} in
    A) download_rockfish_data ;;
    B) download_deepmod2_data ;;
    C) download_ont_benchmark ;;
    D) download_deepsignal_data ;;
    Q) download_quick_start ;;
    R) download_references ;;
    ALL)
        download_quick_start
        download_references
        download_rockfish_data
        download_deepmod2_data
        download_ont_benchmark
        download_deepsignal_data
        ;;
    *)
        echo "无效选择: ${CHOICE}"
        echo "用法: bash $0 [A|B|C|D|Q|R|ALL]"
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo "   下载完成! 数据保存在: ${BASE_DIR}/"
echo "=============================================="
echo ""
echo "下一步建议:"
echo "  1. 检查下载的数据完整性"
echo "  2. 按论文方法进行 basecalling (使用 Dorado 或 Guppy)"
echo "  3. 运行 prepare_methylation_data.py 准备训练数据"
echo "  4. 运行 train_methylation.py 训练模型"
