from turtle import hideturtle
import pywt
import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from vector_quantize_pytorch import ResidualVQ

def get_wavelet_filters_pywt(wavelet_name, in_channels):
    """
    使用 pywt 库生成适用于 PyTorch Conv1d/ConvTranspose1d 的小波滤波器张量。
    
    Args:
        wavelet_name (str): 'haar', 'db2', 'sym4' 等 pywt 支持的名字。
        in_channels (int): 输入通道数 (用于 repeat 权重以适应分组卷积)。
        
    Returns:
        (dec_lo, dec_hi, rec_lo, rec_hi)
        全部为 torch.Tensor, 且已经 reshape 好，可以直接赋值给 nn.Parameter 或 buffer。
    """
    # 1. 从 pywt 获取系数
    try:
        w = pywt.Wavelet(wavelet_name)
    except ValueError:
        raise ValueError(f"Unknown wavelet name: {wavelet_name}. Check pywt.wavelist().")
    # dec_lo: 分解低通, dec_hi: 分解高通
    # rec_lo: 重构低通, rec_hi: 重构高通
    dec_lo = np.array(w.dec_lo, dtype=np.float32)
    dec_hi = np.array(w.dec_hi, dtype=np.float32)
    rec_lo = np.array(w.rec_lo, dtype=np.float32)
    rec_hi = np.array(w.rec_hi, dtype=np.float32)

    # 2. 翻转 (Flip) - 关键步骤
    # 信号处理中的“卷积”通常需要将核翻转。
    # PyTorch 的 F.conv1d 执行的是“互相关 (Cross-correlation)”。
    # 为了数学严谨性，通常需要 flip。但在深度学习中，只要 DWT 和 IDWT 能对上即可。
    # 为了匹配 pywt 的标准定义，我们对分解滤波器进行 flip。
    dec_lo = np.flip(dec_lo)
    dec_hi = np.flip(dec_hi)
    # 重构滤波器通常不需要 flip (取决于具体实现标准，pywt 的 rec 系数通常是已经适配好的)
    # 但为了适配 ConvTranspose1d 的数学定义，我们保持原样或根据重构误差测试调整。
    # 经验证：适配 pytorch ConvTranspose1d 时，使用原始 rec 系数即可。

    # 3. 转换为 Tensor 并增加维度
    # 形状目标: [Out_Channels, In_Channels/Groups, Kernel_Size]
    # 这里 Out=1, In=1 (因为我们后面做 Group Conv, 对每个 channel 独立)
    dec_lo = torch.from_numpy(dec_lo.copy()).view(1, 1, -1)
    dec_hi = torch.from_numpy(dec_hi.copy()).view(1, 1, -1)
    rec_lo = torch.from_numpy(rec_lo).view(1, 1, -1)
    rec_hi = torch.from_numpy(rec_hi).view(1, 1, -1)

    # 4. 扩展到所有通道 (Repeat)
    # 使得 shape 变为 [in_channels, 1, kernel_size]
    dec_lo = dec_lo.repeat(in_channels, 1, 1)
    dec_hi = dec_hi.repeat(in_channels, 1, 1)
    rec_lo = rec_lo.repeat(in_channels, 1, 1)
    rec_hi = rec_hi.repeat(in_channels, 1, 1)

    return dec_lo, dec_hi, rec_lo, rec_hi

class DWT1D(nn.Module):
    def __init__(self, in_channels, wavelet='db2', learnable=False):
        super().__init__()
        self.in_channels = in_channels
        
        # 使用 pywt 获取系数 (解包前两个：分解滤波器)
        d_lo, d_hi, _, _ = get_wavelet_filters_pywt(wavelet, in_channels)
        
        # 填充计算：padding = (kernel_size - 1) // 2 (仅针对奇数核或 Haar)
        # 对于 db2 (len=4)，通常需要更复杂的 padding 逻辑 (如 Reflect Pad)
        self.padding = (d_lo.shape[-1]-2) // 2 
        self.stride = 2
        self.in_channels = in_channels
        
        if learnable:
            self.lo_conv = nn.Parameter(d_lo)
            self.hi_conv = nn.Parameter(d_hi)
        else:
            self.register_buffer('lo_conv', d_lo)
            self.register_buffer('hi_conv', d_hi)


    def forward(self, x):
        """
        Input: [B, C, L]
        Output: (low, high), both are [B, C, L/2]
        """
        # 处理填充以适应 stride=2
        # 如果长度是奇数，需要 pad
        if x.shape[-1] % 2 != 0:
            x = F.pad(x, (0, 1), mode='reflect')
            
        # 低频分量 (Approximation)
        low = F.conv1d(x, self.lo_conv, stride=self.stride, groups=self.in_channels,padding=self.padding)
        
        # 高频分量 (Detail)
        high = F.conv1d(x, self.hi_conv, stride=self.stride, groups=self.in_channels,padding=self.padding)
        
        return low, high

class IDWT1D(nn.Module):
    def __init__(self, in_channels, wavelet='db4', learnable=False):
        super().__init__()
        self.in_channels = in_channels
        
        # 对于 IDWT，我们需要逆滤波器
        # 正交小波中，IDWT 滤波器通常与 DWT 滤波器有时序反转关系
        # 这里简化处理，直接使用 DWT 滤波器进行转置卷积 (Transposed Conv) 即可实现重构
        _, _, rec_lo_kernel, rec_hi_kernel = get_wavelet_filters_pywt(wavelet, in_channels)
        
        # 转置卷积权重: [In_Channels, Out_Channels/Groups, Kernel_Size]
        # Groups=in_channels, 所以是 [in_channels, 1, kernel_size]
        # 注意：为了数学上的完美重构，通常 IDWT 的滤波器需要重新推导。
        # 对于 Haar，DWT和IDWT系数仅差一个缩放因子或顺序。
        # 这里为了确保完全重构，我们复用 DWT 的核，但在 forward 里相加。

        if learnable:
            self.lo_conv = nn.Parameter(rec_lo_kernel)
            self.hi_conv = nn.Parameter(rec_hi_kernel)
        else:
            self.register_buffer('lo_conv', rec_lo_kernel)
            self.register_buffer('hi_conv', rec_hi_kernel)
            
        self.stride = 2

    def forward(self, low, high):
        """
        Input: low [B, C, N], high [B, C, N]
        Output: recon [B, C, 2*N]
        """
        # 转置卷积实现上采样+滤波
        # 注意：PyTorch ConvTranspose1d 默认行为
        
        # Haar 反变换的特性：需要调整权重顺序或者符号来抵消。
        # 这里为了演示简单，我们假设使用 Haar 的特性。
        # 若是严格的数学重构，建议使用 pytorch_wavelets 库，
        # 但如果是“神经网络”，只要让这一层可逆或可学习即可。
        
        # 修正：为了保证 Haar 的完美重构，IDWT 的滤波器实际上是：
        # Lo_rec = Lo_dec
        # Hi_rec = -1 * Hi_dec (或者顺序互换，取决于具体定义)
        # 这里我们使用简单的加法重构逻辑
        
        rec_lo = F.conv_transpose1d(low, self.lo_conv, stride=self.stride, groups=self.in_channels)
        rec_hi = F.conv_transpose1d(high, self.hi_conv, stride=self.stride, groups=self.in_channels)
        target_len = low.shape[-1] * self.stride
        
        if rec_lo.shape[-1] > target_len:
            rec_lo = rec_lo[..., :target_len]
        elif rec_lo.shape[-1] < target_len:
            rec_lo = F.pad(rec_lo, (0, target_len - rec_lo.shape[-1]), mode='reflect')
            
        if rec_hi.shape[-1] > target_len:
            rec_hi = rec_hi[..., :target_len]
        elif rec_hi.shape[-1] < target_len:
            rec_hi = F.pad(rec_hi, (0, target_len - rec_hi.shape[-1]), mode='reflect')
        
        # Haar 重构通常是 Low + High (带系数调整)
        # 如果 DWT 系数是 normalized (除以 sqrt(2))，则 IDWT 直接相加即可
        return rec_lo + rec_hi

class PEG(nn.Module):
    """
    Positional Encoding Generator (PEG)
    引用: "Conditional Positional Encodings for Vision Transformers", ICCV 2021
    核心思想：用一个深度卷积(Depth-wise Conv)来动态生成位置信息。
    """
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        # Depth-wise convolution: groups=dim
        # 保持维度不变，只混合局部上下文信息作为位置提示
        self.proj = nn.Conv1d(dim, dim, kernel_size=kernel_size, stride=1, padding=kernel_size//2, groups=dim)

    def forward(self, x):
        """
        Input: [B, Length, Dim] (Transformer 格式)
        Output: [B, Length, Dim]
        """
        # Conv1d 需要 [B, Dim, Length]
        x_cnn = x.permute(0, 2, 1)
        
        # 加上卷积生成的位置偏置
        pos = self.proj(x_cnn)
        
        # 变回 [B, Length, Dim]
        return x + pos.permute(0, 2, 1)

class ResidualConv1dBlock(nn.Module):
    """Shallow residual Conv1d block (no downsampling) for pre/post processing."""
    def __init__(self, channels: int, kernel_size: int = 3, stride: int = 2):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, stride=stride, kernel_size=kernel_size, padding=pad)
        self.conv2 = nn.Conv1d(channels, channels, stride=1, kernel_size=kernel_size, padding=pad)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        self.skip = nn.Conv1d(channels, channels, 1, stride) if stride > 1  else None

    def forward(self, x):
        residual = x
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        if self.skip is not None:
            residual = self.skip(residual)
        x += residual
        return self.act(x)

class UpsampleResidualBlock(nn.Module):
    """Residual block for decoder with upsampling support"""
    def __init__(self, in_channels: int, kernel_size: int = 3, stride: int = 2):
        super().__init__()
        # Use ConvTranspose1d for upsampling
        self.conv1 = nn.ConvTranspose1d(in_channels, in_channels, kernel_size, stride, 
                                       padding=kernel_size//2, output_padding=stride-1)
        self.conv2 = nn.Conv1d(in_channels, in_channels, kernel_size, 1, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.bn2 = nn.BatchNorm1d(in_channels)
        self.relu = nn.GELU()
        
        # Skip connection with upsampling
        self.skip = nn.ConvTranspose1d(in_channels, in_channels, 1, stride, 
                                      output_padding=stride-1) if stride > 1  else None
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.skip is not None:
            residual = self.skip(residual)
            
        out += residual
        out = self.relu(out)
        
        return out

class CNN_encoder(nn.Module):
    """CNN encoder for feature extraction."""
    def __init__(self, input_dim: int = 1, hidden_dim: int = 256, compression_ratio: int = 8) -> None:
        super().__init__()
        self.compression_ratio = compression_ratio
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.num_downsamples = int(math.log2(compression_ratio))

        self.in_proj = nn.Sequential(
            nn.Conv1d(self.input_dim, self.hidden_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(self.hidden_dim),
            nn.GELU(),
        )
        self.shallow = nn.Sequential(*[ResidualConv1dBlock(self.hidden_dim, kernel_size=3, stride=2) for _ in range(self.num_downsamples)])

    def forward(self, x):
        x = self.in_proj(x)
        x = self.shallow(x)
        return x

class CNN_decoder(nn.Module):
    def __init__(self, in_channels, out_dim, compression_ratio=8):
        super().__init__()
        self.compression_ratio = compression_ratio
        self.in_channels = in_channels
        self.out_dim = out_dim
        self.num_upsamples = int(math.log2(compression_ratio))

        self.out_proj = nn.Sequential(
            nn.Conv1d(self.in_channels, self.out_dim, kernel_size=7, padding=3)
        )
        self.shallow = nn.Sequential(*[UpsampleResidualBlock(self.in_channels, kernel_size=3, stride=2) for _ in range(self.num_upsamples)])

    def forward(self, x):
        x = self.shallow(x)
        x = self.out_proj(x)
        return x
#多层小波分解
class MultiLevelDWT(nn.Module):
    def __init__(self, in_channels, levels=4, wavelet='db4', learnable=True):
        super().__init__()
        self.levels = levels
        self.wavelet = wavelet
        self.in_channels = in_channels
        self.learnable = learnable

        self.dwt = DWT1D(self.in_channels, wavelet=self.wavelet, learnable=self.learnable)

    def forward(self, x):
        """
        返回: 
        - low_final: 最低频 (Level 3 Low)
        - highs: 列表 [Level 1 High, Level 2 High, Level 3 High]
        """
        current_low = x
        highs = []
        
        for i in range(self.levels):
            # 递归分解：对上一级的低频继续分解
            current_low, current_high = self.dwt(current_low)
            highs.append(current_high)
            
        # highs 列表顺序: [H_lvl1, H_lvl2, H_lvl3]
        # 对应长度: [640, 320, 160] (以 1280 输入为例)
        return current_low, highs
#高频噪声添加
class NoiseInjector(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channel, 1)) # 学习噪声的强度
    
    def forward(self, x):
        noise = torch.randn_like(x)
        return x + noise * self.weight

class LowFreqStream(nn.Module):
    def __init__(self, in_channels=256,trans_channels=512,codebook_dim=256,num_layers=4,n_heads=8,num_quantizers=6,codebook_size=512):
        super().__init__()

        self.in_proj = nn.Linear(in_channels, trans_channels)
        #位置编码
        self.pos_generator = PEG(trans_channels, kernel_size=3)

        self.dropout = nn.Dropout(0.1)
        #transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=trans_channels, 
            nhead=n_heads, 
            dim_feedforward=trans_channels * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True 
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. 瓶颈投影
        self.bottleneck_proj = nn.Linear(trans_channels, codebook_dim)

        # 5. 残差 VQ
        self.vq_layer = ResidualVQ(
            dim=codebook_dim, codebook_size=codebook_size, 
            num_quantizers=num_quantizers, kmeans_init=True, 
            kmeans_iters=40,
            threshold_ema_dead_code=2,
        )
        # 6. 输出投影
        # self.out_proj = nn.Linear(codebook_dim, in_channels)

    def forward(self, x):
        #input size大小[B,C,L]
        x = x.transpose(1, 2)
        x = self.in_proj(x)
        x = self.pos_generator(x)
        x = self.dropout(x)
        feat_x = self.transformer(x)
        feat_x = self.bottleneck_proj(feat_x)
        z_q,indices,vq_loss = self.vq_layer(feat_x)
        # out = self.out_proj(z_q)
        out = z_q.transpose(1, 2)
        return out,indices,vq_loss.sum()

class HighFreqStream(nn.Module):
    def __init__(self,in_channels,codebook_dim=256,num_quantizers=2,codebook_size=512):
        super().__init__()
        self.in_channels = in_channels
        #对齐层
        self.align_h3 = nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
        self.align_h2 = nn.Conv1d(in_channels, in_channels, kernel_size=5, stride=4, padding=2)
        self.align_h1 = nn.Conv1d(in_channels, in_channels, kernel_size=5, stride=8, padding=2)
        
        #门控机制1:h4指导h3
        self.gate_h4 = nn.Conv1d(in_channels, in_channels, 1)
        self.process_h3 = nn.Sequential(
            nn.Conv1d(in_channels*2, in_channels, 3,1,1),
            nn.GELU(),
        )
        #门控机制2:f3指导h2
        self.gate_f3 = nn.Conv1d(in_channels, in_channels, 1)
        self.process_h2 = nn.Sequential(
            nn.Conv1d(in_channels*2, in_channels, 3,1,1),
            nn.GELU(),
        )  
        #门控机制3:f2指导h1
        self.gate_f2 = nn.Conv1d(in_channels, in_channels, 1)
        self.process_h1 = nn.Sequential(
            nn.Conv1d(in_channels*2, in_channels, 3,1,1),
            nn.GELU(),
        )
        self.vq_layer = ResidualVQ(
            dim=codebook_dim, codebook_size=codebook_size, 
            num_quantizers=num_quantizers, kmeans_init=True, 
            kmeans_iters=40,
            threshold_ema_dead_code=2,
        )

    def forward(self, h4,h3, h2, h1):
        """
        Input:
            h4: [B, C, 160]
            h3: [B, C, 320]
            h2: [B, C, 640]
            h1: [B, C, 1280]
        """
        # 1. 空间对齐 (Alignment)
        h3_prime = self.align_h3(h3) # -> 160
        h2_prime = self.align_h2(h2) # -> 160
        h1_prime = self.align_h1(h1) # -> 160
        
        gate_4 = torch.sigmoid(self.gate_h4(h4)) 
        h3_gated = h3_prime * gate_4 
        # 拼接并融合
        f3 = self.process_h3(torch.cat([h4, h3_gated], dim=1))

        gate_3 = torch.sigmoid(self.gate_f3(f3)) 
        h2_gated = h2_prime * gate_3 
        # 拼接并融合
        f2 = self.process_h2(torch.cat([f3, h2_gated], dim=1))
        
        gate_2 = torch.sigmoid(self.gate_f2(f2))
        h1_gated = h1_prime * gate_2
        # 拼接并融合
        h_fused = self.process_h1(torch.cat([f2, h1_gated], dim=1))
        x = h_fused.permute(0, 2, 1)

        z_q,indices,vq_loss = self.vq_layer(x)

        out = z_q.permute(0, 2, 1)
        
        return out,indices,vq_loss.sum()

class ResBlock1d(nn.Module):
    """简单的残差块，用于特征精炼"""
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(dim, dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv1d(dim, dim, 3, 1, 1)
        )
    def forward(self, x):
        return x + self.block(x)

class HighFreqStream_decoder(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.in_channels = in_channels
        #从 fused 特征中解耦出H4
        self.recover_h4 = nn.Sequential(
            ResBlock1d(in_channels),
        )
        #Upsample & Generate H3 (160 -> 320) ---
        # 转置卷积上采样
        self.upsample_3 = nn.ConvTranspose1d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)
        # 特征处理
        self.process_h3 = ResBlock1d(in_channels)

        #Upsample & Generate H2 (320-> 640) ---
        self.upsample_2 = nn.ConvTranspose1d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)
        # 特征处理
        self.process_h2 = ResBlock1d(in_channels)

        #Upsample & Generate H1 (640-> 1280) ---
        self.upsample_1 = nn.ConvTranspose1d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)
        # 特征处理
        self.process_h1 = ResBlock1d(in_channels)

    def forward(self, fused):
        """
        Input:
            fused: [B, C, 160]
        """
        #回复h4
        h4 = self.recover_h4(fused)
        #生成 H3
        # 先利用 h_fused 上采样得到 320 长度的中间特征
        # 注意：我们使用 h_fused 作为骨干继续生长，保持连贯性
        feat_h3 = self.upsample_3(fused)
        feat_h3 = self.process_h3(feat_h3)
        #生成 H2
        feat_h2 = self.upsample_2(feat_h3)
        feat_h2 = self.process_h2(feat_h2)
        #生成 H1
        feat_h1 = self.upsample_1(feat_h2)
        feat_h1 = self.process_h1(feat_h1)
        return [feat_h1,feat_h2,feat_h3,h4]

class MultiLevelIDWT(nn.Module):
    def __init__(self, in_channels, levels=3, wavelet='haar',learnable=True):
        super().__init__()
        self.levels = levels
        self.idwt = IDWT1D(in_channels, wavelet=wavelet,learnable=learnable)

    def forward(self, low_final, highs):
        """
        Input:
        - low_final: [B, C, L_smallest]
        - highs: list of [H_lvl1, H_lvl2, H_lvl3]
        """
        current_low = low_final
        
        # 逆序重构：从 Level 3 -> Level 1
        # highs 的输入顺序通常是 [H1, H2, H3]，我们需要反过来取
        
        for i in range(self.levels):
            level_idx = self.levels - 1 - i
            current_high = highs[level_idx]
            
            # 执行 IDWT 合并
            current_low = self.idwt(current_low, current_high)
            
        return current_low


class WaveletMambaDualStreamVQVAE(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=256, cnn_compression_ratio=4, wavelet='db4', learnable=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.cnn_compression_ratio = cnn_compression_ratio
        self.wavelet = wavelet
        self.learnable = learnable

        self.encoder = CNN_encoder(input_dim, hidden_dim, cnn_compression_ratio)
        #小波分解
        self.multi_level_dwt = MultiLevelDWT(hidden_dim, levels=4, wavelet=wavelet, learnable=learnable)
        #低频流
        self.low_stream = LowFreqStream(in_channels=hidden_dim)
        #高频流
        self.high_stream = HighFreqStream(in_channels=hidden_dim)
        #高频流解码
        self.high_stream_decoder = HighFreqStream_decoder(hidden_dim)
        #小波重构
        self.multi_level_idwt = MultiLevelIDWT(hidden_dim, levels=4, wavelet=wavelet, learnable=learnable)


        self.decoder = CNN_decoder(hidden_dim, input_dim, cnn_compression_ratio)

    def load_state_dict(self, state_dict, strict: bool = True):
        """
        Backward-compatible checkpoint loading.

        Some upstream dependencies (e.g., `vector_quantize_pytorch`) have changed
        certain 1-element buffers from scalar shape `[]` to length-1 shape `[1]`
        (or vice-versa). That creates size-mismatch errors when loading old
        checkpoints.

        We auto-fix any mismatch where both tensors have exactly 1 element by
        reshaping the checkpoint tensor to match the current model tensor.
        """
        try:
            own_state = self.state_dict()
            fixed = {}
            for k, v in state_dict.items():
                if (
                    k in own_state
                    and isinstance(v, torch.Tensor)
                    and isinstance(own_state[k], torch.Tensor)
                    and v.shape != own_state[k].shape
                    and v.numel() == 1
                    and own_state[k].numel() == 1
                ):
                    fixed[k] = v.reshape(own_state[k].shape)
                else:
                    fixed[k] = v
            state_dict = fixed
        except Exception:
            # If anything goes wrong, fall back to default behavior.
            pass

        return super().load_state_dict(state_dict, strict=strict)

    def forward(self, x):
        #input size大小[B,1,L]
        feat_cnn = self.encoder(x) #[B,1,L] -> [B,C,L/compression_ratio]
        feat_low_1,feat_high = self.multi_level_dwt(feat_cnn) #[B,C,L/compression_ratio] -> [B,C,L/compression_ratio/2]
        feat_low_2,indices_low,vq_loss_low = self.low_stream(feat_low_1) #[B,C,L/compression_ratio/2] -> [B,C,L/compression_ratio/2]
        feat_high_1,indices_high,vq_loss_high = self.high_stream(feat_high[3],feat_high[2],feat_high[1],feat_high[0]) #[B,C,L/compression_ratio/2] -> [B,C,L/compression_ratio/2]
        feat_high_2 = self.high_stream_decoder(feat_high_1) #[B,C,L/compression_ratio/2] -> [B,C,L/compression_ratio/2]
        feat_low = self.multi_level_idwt(feat_low_2,feat_high_2) #[B,C,L/compression_ratio/2] -> [B,C,L/compression_ratio]
        feat_cnn_out = self.decoder(feat_low) #[B,C,L/compression_ratio] -> [B,1,L]
        # Return order:
        #   - feat_cnn: reconstructed signal
        #   - indices_low/indices_high: code indices
        #   - vq_loss_low/vq_loss_high: RVQ losses
        #   - feat_high / feat_high_2: target highs and reconstructed highs (lists, low->high order)
        #   - feat_low_2 / feat_low_1: reconstructed low (quantized) and target low (DWT low_final)
        return feat_cnn_out,indices_low,indices_high,vq_loss_low,vq_loss_high,feat_high,feat_high_2,feat_cnn,feat_low


class MultiScaleSpectralLoss(nn.Module):
    def __init__(self, fft_sizes=[2048, 1024, 512, 256, 128, 64], overlap_ratio=0.75):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = [int(s * (1 - overlap_ratio)) for s in fft_sizes]
        self.win_lengths = fft_sizes 

    def stft(self, x, fft_size, hop_size, win_length):
        # x: [B, 1, L] -> [B, L]
        x = x.squeeze(1)
        # return: [B, Freq, Time, 2] (Complex)
        x_stft = torch.stft(x, n_fft=fft_size, hop_length=hop_size, win_length=win_length, 
                            window=torch.hann_window(win_length).to(x.device), 
                            return_complex=True)
        # 我们只关心幅度谱，不关心相位（对于生成的纹理来说，相位稍微错一点没关系）
        return torch.abs(x_stft)

    def forward(self, x_real, x_recon):
        """
        输入: 原始信号 x_real, 重构信号 x_recon
        输出: 综合频谱损失
        """
        loss = 0.0
        for i in range(len(self.fft_sizes)):
            mag_real = self.stft(x_real, self.fft_sizes[i], self.hop_sizes[i], self.win_lengths[i])
            mag_recon = self.stft(x_recon, self.fft_sizes[i], self.hop_sizes[i], self.win_lengths[i])
            
            # 1. 频谱收敛损失 (Spectral Convergence) - 关注整体能量分布
            sc_loss = torch.norm(mag_real - mag_recon, p="fro") / (torch.norm(mag_real, p="fro") + 1e-5)
            
            # 2. 对数幅度损失 (Log-Magnitude) - 关注微弱细节
            log_loss = F.l1_loss(torch.log(mag_real + 1e-5), torch.log(mag_recon + 1e-5))
            
            loss += (sc_loss + log_loss)
        
        return loss / len(self.fft_sizes)


class SubBandLoss(nn.Module):
    def __init__(self, high_freq_weights=[0.5, 1.0, 2.0,4.0]):
        """
        Args:
            high_freq_weights: 针对 [H1, H2, H3] 的权重列表。
            建议策略: H1(噪声)权重低，H3(结构细节)权重高。
            假设输入顺序是由细到粗: [H1(640), H2(320), H3(160)]
        """
        super().__init__()
        self.l1 = nn.L1Loss(reduction='mean') # 对整个矩阵求平均
        self.weights = high_freq_weights

    def forward(self, target_L3, recon_L3, target_Highs, recon_Highs):
        """
        Input:
            target_L3, recon_L3: Tensor [Batch, 128, 160]
            target_Highs: List of Tensors [H1, H2, H3] 
                          H1:[B,128,640], H2:[B,128,320], H3:[B,128,160]
            recon_Highs:  List of Tensors (同上)
        """
        
        # --- 1. 低频损失 (语义核心) ---
        # L1 Loss 会自动计算两个 [B, C, L] 矩阵之间所有元素的平均绝对误差
        loss_low = self.l1(target_L3, recon_L3)
        
        # --- 2. 高频损失 (纹理细节) ---
        loss_high = 0.0
        
        # 确保列表长度一致
        assert len(target_Highs) == len(recon_Highs), "高频分量列表长度不匹配"
        
        # 遍历每个尺度 (H1, H2, H3)
        for i, (t_h, r_h) in enumerate(zip(target_Highs, recon_Highs)):
            # 维度检查 (Debugging 友好)
            if t_h.shape != r_h.shape:
                raise ValueError(f"High-freq scale {i} shape mismatch: {t_h.shape} vs {r_h.shape}")
            
            # 计算该尺度的矩阵误差
            scale_loss = self.l1(t_h, r_h)
            
            # 加权累加
            # 如果 weights 是 [0.5, 1.0, 2.0]
            # H1(最细, 噪声多) * 0.5
            # H2(中等) * 1.0
            # H3(接近语义) * 2.0
            weight = self.weights[i] if i < len(self.weights) else 1.0
            loss_high += scale_loss * weight
            
        # 对高频总损失取个平均，防止数值过大（可选）
        loss_high = loss_high / len(target_Highs)
            
        return loss_low, loss_high

class WaveTransformerLoss(nn.Module):
    def __init__(self, lambdas={'time': 10.0, 'freq': 1.0, 'sub_low': 5.0, 'sub_high': 1.0, 'vq': 1.0}):
        super().__init__()
        self.lambdas = lambdas
        self.time_loss_fn = nn.L1Loss() # 全局时域
        self.spec_loss_fn = MultiScaleSpectralLoss() # 全局频域
        self.sub_loss_fn = SubBandLoss() # 局部子带监督

    def forward(self, 
                x, x_recon,               # 全局信号
                target_L3, recon_L3,      # 低频子带
                target_Hs, recon_Hs,      # 高频子带列表
                vq_loss):                 # VQ Loss
        
        # 1. 全局时域损失 (Time Domain)
        loss_time = self.time_loss_fn(x, x_recon)
        
        # 2. 全局频域损失 (Frequency Domain)
        loss_spec = self.spec_loss_fn(x, x_recon)
        
        # 3. 辅助子带损失 (Sub-band Supervision)
        loss_sub_low, loss_sub_high = self.sub_loss_fn(target_L3, recon_L3, target_Hs, recon_Hs)
        
        # 4. 总损失加权
        total_loss = (
            self.lambdas['time'] * loss_time +
            self.lambdas['freq'] * loss_spec +
            self.lambdas['sub_low'] * loss_sub_low +   # 给予低频更高的权重，因为它是语义核心
            self.lambdas['sub_high'] * loss_sub_high + # 高频权重低一点，允许有些许噪点差异
            self.lambdas['vq'] * vq_loss
        )
        
        return total_loss, {
            'loss_time': loss_time.item(),
            'loss_spec': loss_spec.item(),
            'loss_sub_low': loss_sub_low.item(),
            'loss_sub_high': loss_sub_high.item(),
            'loss_vq': vq_loss.item()
        }
