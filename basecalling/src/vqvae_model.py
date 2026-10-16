import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import math
import sys
from vector_quantize_pytorch import ResidualVQ


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, ff_mult=4, dropout=0.2):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.ReLU(inplace=True),
            nn.Linear(dim * ff_mult, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, C, T] -> [B, T, C]
        x = x.transpose(1, 2)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        # return x
        return x.transpose(1, 2)

# 
class UpsampleResidualBlock(nn.Module):
    """Residual block for decoder with upsampling support"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 2):
        super().__init__()
        # Use ConvTranspose1d for upsampling
        self.conv1 = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, 
                                       padding=kernel_size//2, output_padding=stride-1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Skip connection with upsampling
        self.skip = nn.ConvTranspose1d(in_channels, out_channels, 1, stride, 
                                      output_padding=stride-1) if stride > 1 or in_channels != out_channels else None
        
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


class ResidualBlock(nn.Module):
    """Residual block for encoder/decoder"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Skip connection
        self.skip = nn.Conv1d(in_channels, out_channels, 1, stride) 
        # if stride > 1 or in_channels != out_channels else None
        
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


class Encoder(nn.Module):
    """Encoder for VQ-VAE"""
    def __init__(self, input_dim: int = 1, hidden_dim: int = 64, 
                 num_layers: int = 5, embedding_dim: int = 64,
                 compression_ratio: int = 64, 
                 stride: list = [2, 2,2, 2,2],
                 ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.compression_ratio = compression_ratio
        self.hidden_dim = hidden_dim
        # Calculate number of downsampling layers needed for desired compression ratio
        # compression_ratio = 2^(num_downsamples)
        # import math
        # num_downsamples = int(math.log2(compression_ratio))
        
        # Ensure we have enough layers for the compression ratio
        actual_num_layers = num_layers
        
        layers = []
        current_dim = input_dim
        
        # First layer
        layers.append(nn.Conv1d(input_dim, hidden_dim, kernel_size=7, stride=2, padding=7//2))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        current_dim = hidden_dim
        
        # Residual blocks (each with stride=2 for downsampling)
        for i in range(actual_num_layers):
            layers.append(ResidualBlock(current_dim, hidden_dim, kernel_size=3, stride=stride[i]))
            current_dim = hidden_dim
            
        # Final projection to embedding dimension
        layers.append(nn.Conv1d(current_dim, embedding_dim, kernel_size=1))
        
        self.encoder = nn.Sequential(*layers)
        self.compression_ratio = compression_ratio
        # self.transformer = TransformerBlock(hidden_dim, num_heads=4)
        # self.proj = nn.Conv1d(hidden_dim, embedding_dim, kernel_size=1)
        
    def forward(self, x):
        # x: [batch_size, seq_len] -> [batch_size, 1, seq_len]
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.encoder(x)
        # x = self.transformer(x)
        # x = self.proj(x)
        return x

class Decoder_basecalling(nn.Module):
    """Decoder for VQ-VAE"""
    def __init__(self, embedding_dim: int = 64, hidden_dim: int = 64, 
                 num_layers: int = 5, output_dim: int = 1, stride: list = [1, 2,2]):
        super().__init__()
        
        # Calculate number of upsampling layers needed for desired compression ratio
        # compression_ratio = 2^(num_upsamples)
        
        # Ensure we have enough layers for the compression ratio
        actual_num_layers = num_layers
        
        layers = []
        transformer_layers = []
        current_dim = embedding_dim
        
        # First layer
        layers.append(nn.ConvTranspose1d(embedding_dim, hidden_dim, kernel_size=7, stride=2, padding=3, output_padding=1))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        current_dim = hidden_dim
        
        # Residual blocks (each with stride=2 for upsampling)
        for i in range(len(stride)):
            layers.append(UpsampleResidualBlock(current_dim, hidden_dim, kernel_size=3, stride=stride[i]))
            current_dim = hidden_dim
            
        # Final projection to output dimension
        layers.append(nn.Conv1d(current_dim, output_dim, kernel_size=1))
        
        self.decoder = nn.Sequential(*layers)
        # for i in range(6):
        #     transformer_layers.append(TransformerBlock(embedding_dim, num_heads=8))
        # self.transformer = nn.Sequential(*transformer_layers)
    def forward(self, x):
        # x = self.transformer(x)
        x = self.decoder(x)
        # x = self.transformer(x)
        # x = x.permute(2, 0, 1)

        return x


class Decoder(nn.Module):
    """Decoder for VQ-VAE"""
    def __init__(self, embedding_dim: int = 64, hidden_dim: int = 64, 
                 num_layers: int = 5, output_dim: int = 1, compression_ratio: int = 64):
        super().__init__()
        
        # Calculate number of upsampling layers needed for desired compression ratio
        # compression_ratio = 2^(num_upsamples)
        import math
        num_upsamples = int(math.log2(compression_ratio))
        
        # Ensure we have enough layers for the compression ratio
        actual_num_layers = max(num_layers, num_upsamples - 1)
        
        layers = []
        # transformer_layers = []
        current_dim = embedding_dim
        
        # First layer
        layers.append(nn.ConvTranspose1d(embedding_dim, hidden_dim, kernel_size=7, stride=2, padding=3, output_padding=1))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        current_dim = hidden_dim
        
        # Residual blocks (each with stride=2 for upsampling)
        for i in range(actual_num_layers):
            layers.append(UpsampleResidualBlock(current_dim, hidden_dim, kernel_size=3, stride=2))
            current_dim = hidden_dim
            
        # Final projection to output dimension
        layers.append(nn.Conv1d(current_dim, output_dim, kernel_size=1))
        
        self.decoder = nn.Sequential(*layers)
        # for i in range(2):
        #     transformer_layers.append(TransformerBlock(embedding_dim, num_heads=4))
        # self.transformer = nn.Sequential(*transformer_layers)
    def forward(self, x):
        # x = self.transformer(x)
        x = self.decoder(x)
        return x


class VQVAE(nn.Module):
    """
    VQ-VAE model for nanopore signal compression and reconstruction
    """
    def __init__(self, input_dim: int = 1, hidden_dim: int = 64, 
                 num_layers: int = 5, embedding_dim: int = 64,
                 num_embeddings: int = 512, num_quantizers: int = 12, commitment_cost: float = 0.25,
                 compression_ratio: int = 64, stride: list = [2, 2,2, 2,2]):
        super().__init__()
        
        self.compression_ratio = compression_ratio
        self.codebook_size = num_embeddings
        self.num_quantizers = num_quantizers

        # 基础稳健型损失函数: rec_loss = 0.7·MAE + 0.3·Huber(beta=1.0)
        self.mae_loss = nn.L1Loss()
        self.huber_loss = nn.HuberLoss(delta=1.0)  # delta=1.0 corresponds to beta=1.0
        self.mae_weight = 0.7
        self.huber_weight = 0.3
        
        self.encoder = Encoder(input_dim, hidden_dim, num_layers, embedding_dim, compression_ratio, stride)
        self.decoder = Decoder(embedding_dim, hidden_dim, num_layers, input_dim, compression_ratio)
        self.vq_layer = ResidualVQ(
            dim=embedding_dim, codebook_size=num_embeddings, 
            num_quantizers=num_quantizers, kmeans_init=True, 
            kmeans_iters=40,
            threshold_ema_dead_code=2,
        )
        
    def forward(self, x):
        # Encode
        z = self.encoder(x)
        z = z.transpose(1, 2)
        
        # Vector quantization
        quantized, encoding_indices,vq_loss = self.vq_layer(z)
        quantized = quantized.transpose(1, 2)
        # Decode
        reconstructed = self.decoder(quantized)
        
        # 基础稳健型损失: 0.7·MAE + 0.3·Huber(beta=1.0)
        mae = self.mae_loss(reconstructed, x)
        huber = self.huber_loss(reconstructed, x)
        rec_loss = self.mae_weight * mae + self.huber_weight * huber
        
        vq_loss = torch.sum(vq_loss)
        loss = rec_loss + 0.5 * vq_loss
        return vq_loss, rec_loss, loss
    
    def encode(self, x):
        """
        Encode input signal to discrete codes
        Args:
            x: Input signal [batch_size, channels, seq_len] or [batch_size, seq_len]
        Returns:
            quantized: Quantized latent representation [batch_size, dimension, seq_len']
            encoding_indices: Discrete codes [batch_size, seq_len', num_quantizers]
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Encode
        z = self.encoder(x)
        
        # Transpose for VQ
        z = z.transpose(1, 2)
        
        # Vector quantization
        quantized, encoding_indices, _ = self.vq_layer(z)
        
        # Transpose back
        quantized = quantized.transpose(1, 2)
        
        return quantized, encoding_indices

    def decode(self, quantized):
        """
        Decode quantized latent representation to signal
        Args:
            quantized: Quantized latent [batch_size, dimension, seq_len']
        Returns:
            reconstructed: Reconstructed signal [batch_size, channels, seq_len]
        """
        return self.decoder(quantized)
    
    def reconstruct(self, x):
        """
        Full reconstruction pipeline
        Args:
            x: Input signal [batch_size, channels, seq_len] or [batch_size, seq_len]
        Returns:
            reconstructed: Reconstructed signal
            encoding_indices: Discrete codes
        """
        quantized, encoding_indices = self.encode(x)
        reconstructed = self.decode(quantized)
        return reconstructed, encoding_indices
    
    def compress(self, x):
        """
        Compress signal to discrete codes
        Args:
            x: Input signal [batch_size, channels, seq_len] or [batch_size, seq_len]
        Returns:
            encoding_indices: Discrete codes [batch_size, seq_len', num_quantizers]
        """
        _, encoding_indices = self.encode(x)
        return encoding_indices
    
    def decompress(self, encoding_indices):
        """
        Decompress discrete codes to signal
        Args:
            encoding_indices: Discrete codes [batch_size, seq_len', num_quantizers]
        Returns:
            reconstructed: Reconstructed signal [batch_size, channels, seq_len]
        """
        # Get embeddings from codebook
        quantized = self.vq_layer.get_output_from_indices(encoding_indices)
        # print('quantized shape: ', quantized.shape)
        
        # Transpose: [B, T', D] -> [B, D, T']
        quantized = quantized.transpose(1, 2)
        
        # Decode
        reconstructed = self.decoder(quantized)
        
        return reconstructed

