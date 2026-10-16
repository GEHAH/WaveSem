from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import ResidualVQ,VectorQuantize

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
        if self.compression_ratio == 1:
            x = self.in_proj(x)
            return x
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
        if self.compression_ratio == 1:
            x = self.out_proj(x)
            return x
        x = self.shallow(x)
        x = self.out_proj(x)
        return x

def _match_length(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    Crop or pad x on the right to match target_len.
    x: [B, C, L]
    """
    cur = x.size(-1)
    if cur == target_len:
        return x
    if cur > target_len:
        return x[..., :target_len]
    return F.pad(x, (0, target_len - cur), mode="constant", value=0.0)

class StridedCNNAE(nn.Module):
    """
    Strided CNN AutoEncoder baseline (naïve downsampling):
      - encoder: repeated Conv1d(stride=2)
      - decoder: repeated ConvTranspose1d(stride=2)
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 256,
        cnn_compression_ratio: int = 64,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.compression_ratio = cnn_compression_ratio
        self.encoder = CNN_encoder(self.input_dim, self.hidden_dim, self.compression_ratio)
        self.decoder = CNN_decoder(self.hidden_dim, self.input_dim, self.compression_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,1,L]
        target_len = x.size(-1)
        z = self.encoder(x)
        y = self.decoder(z)
        return _match_length(y, target_len)

class ReconSpecVQLoss(nn.Module):
    """
    Baseline loss for AE / single-codebook VQ-VAE:
      total = lambda_time * L1(x, x_recon) + lambda_freq * spectral(x, x_recon) + lambda_vq * vq_loss
    """

    def __init__(self, lambda_time: float = 10.0, lambda_freq: float = 1.0, lambda_vq: float = 1.0):
        super().__init__()
        self.lambda_time = float(lambda_time)
        self.lambda_freq = float(lambda_freq)
        self.lambda_vq = float(lambda_vq)
        self.time_loss_fn = nn.L1Loss()
        self.spec_loss_fn = MultiScaleSpectralLoss()

    def forward(self, x: torch.Tensor, x_recon: torch.Tensor, vq_loss: Optional[torch.Tensor] = None):
        loss_time = self.time_loss_fn(x, x_recon)
        loss_spec = self.spec_loss_fn(x, x_recon)
        if vq_loss is None:
            vq_loss = x.new_tensor(0.0)
        total = self.lambda_time * loss_time + self.lambda_freq * loss_spec + self.lambda_vq * vq_loss
        parts: Dict[str, float] = {
            "loss_time": float(loss_time.detach().item()),
            "loss_spec": float(loss_spec.detach().item()),
            "loss_vq": float(vq_loss.detach().item()),
        }
        return total, parts


class SingleCodebookVQVAE(nn.Module):
    """
    Single-codebook VQ-VAE baseline (no dual-stream):
      encoder -> (single RVQ codebook) -> decoder

    Returns:
      x_recon: [B,1,L]
      indices: [B, T, nq]
      vq_loss: scalar tensor
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 256,
        compression_ratio: int = 64,
        codebook_size: int = 512
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.compression_ratio = compression_ratio
        self.codebook_size = codebook_size
        self.encoder = CNN_encoder(self.input_dim, self.hidden_dim, self.compression_ratio)
        self.decoder = CNN_decoder(self.hidden_dim, self.input_dim, self.compression_ratio)
        self.vq = VectorQuantize(dim=self.hidden_dim, codebook_size=self.codebook_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target_len = x.size(-1)
        z_b_d_t = self.encoder(x)  # [B, D, T]
        z_b_t_d = z_b_d_t.transpose(1, 2)  # [B, T, D]
        quant_b_t_d, indices_b_t_nq, vq_loss = self.vq(z_b_t_d)
        quant_b_d_t = quant_b_t_d.transpose(1, 2)
        y = self.decoder(quant_b_d_t)
        y = _match_length(y, target_len)

        return y, indices_b_t_nq, vq_loss
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

class ReconSpecVQLoss(nn.Module):
    """
    Baseline loss for AE / single-codebook VQ-VAE:
      total = lambda_time * L1(x, x_recon) + lambda_freq * spectral(x, x_recon) + lambda_vq * vq_loss
    """

    def __init__(self, lambda_time: float = 10.0, lambda_freq: float = 1.0, lambda_vq: float = 1.0):
        super().__init__()
        self.lambda_time = float(lambda_time)
        self.lambda_freq = float(lambda_freq)
        self.lambda_vq = float(lambda_vq)
        self.time_loss_fn = nn.L1Loss()
        self.spec_loss_fn = MultiScaleSpectralLoss()

    def forward(self, x: torch.Tensor, x_recon: torch.Tensor, vq_loss: Optional[torch.Tensor] = None):
        loss_time = self.time_loss_fn(x, x_recon)
        loss_spec = self.spec_loss_fn(x, x_recon)
        if vq_loss is None:
            vq_loss = x.new_tensor(0.0)
        total = self.lambda_time * loss_time + self.lambda_freq * loss_spec + self.lambda_vq * vq_loss
        parts: Dict[str, float] = {
            "loss_time": float(loss_time.detach().item()),
            "loss_spec": float(loss_spec.detach().item()),
            "loss_vq": float(vq_loss.detach().item()),
        }
        return total, parts


class RCodebookVQVAE(nn.Module):
    """
    R-codebook VQ-VAE baseline (no dual-stream):
      encoder -> (single RVQ codebook) -> decoder

    Returns:
      x_recon: [B,1,L]
      indices: [B, T, nq]
      vq_loss: scalar tensor
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 256,
        compression_ratio: int = 64,
        codebook_size: int = 512
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.compression_ratio = compression_ratio
        self.codebook_size = codebook_size
        self.encoder = CNN_encoder(self.input_dim, self.hidden_dim, self.compression_ratio)
        self.decoder = CNN_decoder(self.hidden_dim, self.input_dim, self.compression_ratio)
        self.vq = ResidualVQ(
            dim=self.hidden_dim, codebook_size=self.codebook_size, 
            num_quantizers=8, kmeans_init=True, 
            kmeans_iters=40,
            threshold_ema_dead_code=2,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target_len = x.size(-1)
        z_b_d_t = self.encoder(x)  # [B, D, T]
        z_b_t_d = z_b_d_t.transpose(1, 2)  # [B, T, D]
        quant_b_t_d, indices_b_t_nq, vq_loss = self.vq(z_b_t_d)
        vq_loss = vq_loss.sum()
        quant_b_d_t = quant_b_t_d.transpose(1, 2)
        y = self.decoder(quant_b_d_t)
        y = _match_length(y, target_len)

        return y, indices_b_t_nq, vq_loss
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

#============================wavesem_oDWT=================================
class LowFreqStream_oDWT(nn.Module):
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

class HighFreqStream_oDWT(nn.Module):
    def __init__(self, in_channels=256,codebook_dim=256,num_quantizers=2,codebook_size=512):
        super().__init__()
        self.in_channels = in_channels
        self.vq = ResidualVQ(
            dim=codebook_dim, codebook_size=codebook_size, 
            num_quantizers=num_quantizers, kmeans_init=True, 
            kmeans_iters=40,
            threshold_ema_dead_code=2,
        )
    def forward(self, x):
        x = x.transpose(1, 2)
        z_q,indices,vq_loss = self.vq(x)
        out = z_q.transpose(1, 2)
        return out,indices,vq_loss.sum()

class wavesem_oDWT(nn.Module):
    def __init__(self, input_dim=1,hidden_dim=256):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.encoder = CNN_encoder(input_dim,hidden_dim,compression_ratio=4)
        self.encoder_1 = CNN_encoder(int(hidden_dim/2),hidden_dim,compression_ratio=16)
        self.encoder_2 = CNN_encoder(int(hidden_dim/2),hidden_dim,compression_ratio=16)
        self.low_stream = LowFreqStream_oDWT(in_channels=hidden_dim)
        self.high_stream = HighFreqStream_oDWT(in_channels=hidden_dim)

        self.decoder_1 = CNN_decoder(hidden_dim,int(hidden_dim/2),compression_ratio=16)
        self.decoder_2 = CNN_decoder(hidden_dim,int(hidden_dim/2),compression_ratio=16)
        self.decoder = CNN_decoder(hidden_dim,input_dim,compression_ratio=4)

    def forward(self, input_signal):
        x = self.encoder(input_signal)
        x_low_1,x_high_1 = x.split(int(self.hidden_dim/2),dim=1)
        x_1 = self.encoder_1(x_low_1)
        x_2 = self.encoder_2(x_high_1)
        x_low,indices_low,vq_loss_low = self.low_stream(x_1)
        x_high,indices_high,vq_loss_high = self.high_stream(x_2)
        x_low_2 = self.decoder_1(x_low)
        x_high_2 = self.decoder_2(x_high)
        x_fusion = torch.cat([x_low_2,x_high_2],dim=1)
        out = self.decoder(x_fusion)
        out = _match_length(out, input_signal.size(-1))
        return out,indices_low,indices_high,vq_loss_low,vq_loss_high,x_low_1,x_high_1,x_low_2,x_high_2

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

class WaveLoss_oDWT(nn.Module):
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
        
#============================wavesem_oDWT_end=================================

