"""Core vector quantization implementation."""

import typing as tp
import warnings
import os
import sys,math
from einops import rearrange, repeat
import torch
from torch import nn
import torch.nn.functional as F
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import distrib


def default(val: tp.Any, d: tp.Any) -> tp.Any:
    return val if val is not None else d


def ema_inplace(moving_avg, new, decay: float):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def laplace_smoothing(x, n_categories: int, epsilon: float = 1e-5):
    return (x + epsilon) / (x.sum() + n_categories * epsilon)


def uniform_init(*shape: int):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t


def sample_vectors(samples, num: int):
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]


def kmeans(samples, num_clusters: int, num_iters: int = 10):
    dim, dtype = samples.shape[-1], samples.dtype

    means = sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        diffs = rearrange(samples, "n d -> n () d") - rearrange(
            means, "c d -> () c d"
        )
        dists = -(diffs ** 2).sum(dim=-1)

        buckets = dists.max(dim=-1).indices
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, repeat(buckets, "n -> n d", d=dim), samples)
        new_means = new_means / bins_min_clamped[..., None]

        means = torch.where(zero_mask[..., None], means, new_means)

    return means, bins


class EuclideanCodebook(nn.Module):
    """Codebook with Euclidean distance.
    Args:
        dim (int): Dimension.
        codebook_size (int): Codebook size.
        kmeans_init (bool): Whether to use k-means to initialize the codebooks.
            If set to true, run the k-means algorithm on the first training batch and use
            the learned centroids as initialization.
        kmeans_iters (int): Number of iterations used for k-means algorithm at initialization.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
    """
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        kmeans_init: int = False,
        kmeans_iters: int = 10,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        threshold_ema_dead_code: int = 2,
    ):
        super().__init__()
        self.decay = decay
        init_fn: tp.Union[tp.Callable[..., torch.Tensor], tp.Any] = uniform_init if not kmeans_init else torch.zeros
        embed = init_fn(codebook_size, dim)

        self.codebook_size = codebook_size

        self.kmeans_iters = kmeans_iters
        self.epsilon = epsilon
        self.threshold_ema_dead_code = threshold_ema_dead_code

        self.register_buffer("inited", torch.Tensor([not kmeans_init]))
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed", embed)
        self.register_buffer("embed_avg", embed.clone())

    @torch.jit.ignore
    def init_embed_(self, data):
        if self.inited:
            return

        embed, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters)
        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed.clone())
        self.cluster_size.data.copy_(cluster_size)
        self.inited.data.copy_(torch.Tensor([True]))
        # Make sure all buffers across workers are in sync after initialization
        distrib.broadcast_tensors(self.buffers())

    def replace_(self, samples, mask):
        modified_codebook = torch.where(
            mask[..., None], sample_vectors(samples, self.codebook_size), self.embed
        )
        self.embed.data.copy_(modified_codebook)

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired_codes):
            return

        batch_samples = rearrange(batch_samples, "... d -> (...) d")
        self.replace_(batch_samples, mask=expired_codes)
        distrib.broadcast_tensors(self.buffers())

    def preprocess(self, x):
        x = rearrange(x, "... d -> (...) d")
        return x

    def quantize(self, x):
        embed = self.embed.t()
        dist = -(
            x.pow(2).sum(1, keepdim=True)
            - 2 * x @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )
        embed_ind = dist.max(dim=-1).indices
        return embed_ind

    def postprocess_emb(self, embed_ind, shape):
        return embed_ind.view(*shape[:-1])

    def dequantize(self, embed_ind):
        quantize = F.embedding(embed_ind, self.embed)
        return quantize

    def encode(self, x):
        shape = x.shape
        # pre-process
        x = self.preprocess(x)
        # quantize
        embed_ind = self.quantize(x)
        # post-process
        embed_ind = self.postprocess_emb(embed_ind, shape)
        return embed_ind

    def decode(self, embed_ind):
        quantize = self.dequantize(embed_ind)
        return quantize

    def forward(self, x):
        shape, dtype = x.shape, x.dtype
        x = self.preprocess(x)

        self.init_embed_(x)

        embed_ind = self.quantize(x)
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind = self.postprocess_emb(embed_ind, shape)
        quantize = self.dequantize(embed_ind)

        if self.training:
            # We do the expiry of code at that point as buffers are in sync
            # and all the workers will take the same decision.
            self.expire_codes_(x)
            ema_inplace(self.cluster_size, embed_onehot.sum(0), self.decay)
            embed_sum = x.t() @ embed_onehot
            ema_inplace(self.embed_avg, embed_sum.t(), self.decay)
            cluster_size = (
                laplace_smoothing(self.cluster_size, self.codebook_size, self.epsilon)
                * self.cluster_size.sum()
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embed.data.copy_(embed_normalized)

        return quantize, embed_ind

class GumbelCodebook(nn.Module):
    """基于 Gumbel Softmax 的码本实现
    
    参数:
        dim (int): 输入特征维度
        codebook_size (int): 码本大小(码向量数量)
        temperature (float): Gumbel Softmax 温度参数
        straight_through (bool): 是否使用直通估计器
        kl_weight (float): KL 散度损失的权重
    """
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        temperature: float = 1.0,
        straight_through: bool = True,
        kl_weight: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.temperature = temperature
        self.straight_through = straight_through
        self.kl_weight = kl_weight
        
        # 创建可学习的码本
        self.embed = nn.Parameter(torch.randn(codebook_size, dim))
        nn.init.kaiming_uniform_(self.embed, a=math.sqrt(5))
        
        # 用于计算 logits 的线性层
        self.to_logits = nn.Linear(dim, codebook_size)

    def forward(self, x):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, dim)
            
        返回:
            quantized (torch.Tensor): 量化后的向量
            indices (torch.Tensor): 选择的码本索引
            loss (torch.Tensor): 损失值
        """
        # 计算 logits
        logits = self.to_logits(x)  # (batch_size, seq_len, codebook_size)
        
        # 计算 softmax 概率
        probs = F.softmax(logits, dim=-1)
        
        # 计算 KL 散度损失
        kl_loss = 0.0
        if self.kl_weight > 0 and self.training:
            uniform_dist = torch.ones_like(probs) / self.codebook_size
            kl_div = F.kl_div(
                F.log_softmax(logits, dim=-1), 
                uniform_dist, 
                reduction='batchmean'
            )
            kl_loss = self.kl_weight * kl_div
        
        # Gumbel Softmax 采样
        if self.training:
            # 添加 Gumbel 噪声
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
            gumbel_logits = (logits + gumbel_noise) / max(self.temperature, 1e-10)
            one_hot = F.softmax(gumbel_logits, dim=-1)
        else:
            # 推理时使用 argmax
            indices = torch.argmax(logits, dim=-1)
            one_hot = F.one_hot(indices, num_classes=self.codebook_size).float()
        
        # 直通估计器
        if self.straight_through and self.training:
            # 前向传播使用离散索引
            indices = torch.argmax(one_hot, dim=-1)
            quantized = self.embed[indices]
            
            # 反向传播使用连续近似
            quantized = quantized + (one_hot @ self.embed - quantized).detach()
        else:
            # 使用连续近似
            quantized = one_hot @ self.embed
            indices = torch.argmax(one_hot, dim=-1)
        
        # 重建损失
        recon_loss = F.mse_loss(quantized, x.detach())
        
        # 总损失
        loss = recon_loss + kl_loss
        
        return quantized, indices, loss

    def encode(self, x):
        """编码输入向量"""
        logits = self.to_logits(x)
        indices = torch.argmax(logits, dim=-1)
        return indices

    def decode(self, indices):
        """解码索引为量化向量"""
        quantized = self.embed[indices]
        return quantized

class GumbelSoftmaxVectorQuantization(nn.Module):
    """向量量化(VQ)模块实现
    支持 Gumbel Softmax 两种量化方式
    
    参数:
        dim (int): 输入特征维度
        codebook_size (int): 码本大小(码向量数量)
        codebook_dim (int): 码本维度。如未指定，则使用dim参数
        decay (float): 码本更新的指数移动平均衰减率（仅用于欧氏距离）
        epsilon (float): 数值稳定性常数
        kmeans_init (bool): 是否使用kmeans初始化码本（仅用于欧氏距离）
        kmeans_iters (int): kmeans初始化的迭代次数（仅用于欧氏距离）
        threshold_ema_dead_code (int): 死亡码向量的阈值（仅用于欧氏距离）
        commitment_weight (float): 承诺损失(commitment loss)的权重（仅用于欧氏距离）
        use_gumbel (bool): 是否使用 Gumbel Softmax 量化
        gumbel_temperature (float): Gumbel Softmax 温度参数
        gumbel_straight_through (bool): 是否使用直通估计器（Gumbel）
        gumbel_kl_weight (float): KL 散度损失的权重（Gumbel）
    """
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim: tp.Optional[int] = None,
        epsilon: float = 1e-5,
        commitment_weight: float = 1.,
        gumbel_temperature: float = 1.0,
        gumbel_straight_through: bool = True,
        gumbel_kl_weight: float = 0.0,
    ):
        super().__init__()
        _codebook_dim: int = default(codebook_dim, dim)

        # 如果码本维度与输入维度不同，需要投影层
        requires_projection = _codebook_dim != dim
        self.project_in = (nn.Linear(dim, _codebook_dim) if requires_projection else nn.Identity())
        self.project_out = (nn.Linear(_codebook_dim, dim) if requires_projection else nn.Identity())

        self.epsilon = epsilon
        self.commitment_weight = commitment_weight


        self._codebook = GumbelCodebook(
            dim=_codebook_dim,
            codebook_size=codebook_size,
            temperature=gumbel_temperature,
            straight_through=gumbel_straight_through,
            kl_weight=gumbel_kl_weight
            )
        
        self.codebook_size = codebook_size

    @property
    def codebook(self):
        return self._codebook.embed

    def encode(self, x):
        '''
        编码输入向量
        参数:
            x (torch.Tensor): 输入向量，形状为(batch_size, dim, seq_len)
        返回:
            torch.Tensor: 编码后的索引，形状为(batch_size, seq_len)
        '''
        x = rearrange(x, "b d n -> b n d")
        x = self.project_in(x)
        embed_in = self._codebook.encode(x)
        return embed_in

    def decode(self, embed_ind):
        quantize = self._codebook.decode(embed_ind)
        quantize = self.project_out(quantize)
        quantize = rearrange(quantize, "b n d -> b d n")
        return quantize

    def forward(self, x):
        x = rearrange(x, "b d n -> b n d")
        x = self.project_in(x)
        
        quantize, embed_ind, vq_loss = self._codebook(x)
        loss = vq_loss
        
        quantize = self.project_out(quantize)
        quantize = rearrange(quantize, "b n d -> b d n")
        return quantize, embed_ind, loss



class VectorQuantization(nn.Module):
    """向量量化(VQ)模块实现
    目前仅支持欧氏距离
    参数:
        dim (int): 输入特征维度
        codebook_size (int): 码本大小(码向量数量)
        codebook_dim (int): 码本维度。如未指定，则使用dim参数
        decay (float): 码本更新的指数移动平均衰减率
        epsilon (float): 数值稳定性常数
        kmeans_init (bool): 是否使用kmeans初始化码本
        kmeans_iters (int): kmeans初始化的迭代次数
        threshold_ema_dead_code (int): 死亡码向量的阈值。当某个码向量的EMA聚类size小于此值时，用当前batch中的随机向量替换
        commitment_weight (float): 承诺损失(commitment loss)的权重
    """
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim: tp.Optional[int] = None,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        kmeans_init: bool = True,
        kmeans_iters: int = 50,
        threshold_ema_dead_code: int = 2,
        commitment_weight: float = 1.,
    ):
        super().__init__()
        _codebook_dim: int = default(codebook_dim, dim)

        # 如果码本维度与输入维度不同，需要投影层
        requires_projection = _codebook_dim != dim
        self.project_in = (nn.Linear(dim, _codebook_dim) if requires_projection else nn.Identity())
        self.project_out = (nn.Linear(_codebook_dim, dim) if requires_projection else nn.Identity())

        self.epsilon = epsilon
        self.commitment_weight = commitment_weight

        # 创建欧氏距离码本模块
        self._codebook = EuclideanCodebook(dim=_codebook_dim, codebook_size=codebook_size,
                                           kmeans_init=kmeans_init, kmeans_iters=kmeans_iters,
                                           decay=decay, epsilon=epsilon,
                                           threshold_ema_dead_code=threshold_ema_dead_code)
        self.codebook_size = codebook_size

    @property
    def codebook(self):
        return self._codebook.embed

    def encode(self, x):
        '''
        编码输入向量
        参数:
            x (torch.Tensor): 输入向量，形状为(batch_size, dim, seq_len)
        返回:
            torch.Tensor: 编码后的索引，形状为(batch_size, seq_len)
        '''
        x = rearrange(x, "b d n -> b n d")
        x = self.project_in(x)
        embed_in = self._codebook.encode(x)
        return embed_in

    def decode(self, embed_ind):
        quantize = self._codebook.decode(embed_ind)
        quantize = self.project_out(quantize)
        quantize = rearrange(quantize, "b n d -> b d n")
        return quantize

    def forward(self, x):
        device = x.device
        x = rearrange(x, "b d n -> b n d")
        x = self.project_in(x)

        quantize, embed_ind = self._codebook(x)

        if self.training:
            quantize = x + (quantize - x).detach()

        loss = torch.tensor([0.0], device=device, requires_grad=self.training)

        if self.training:
            warnings.warn('When using RVQ in training model, first check '
                          'https://github.com/facebookresearch/encodec/issues/25 . '
                          'The bug wasn\'t fixed here for reproducibility.')
            if self.commitment_weight > 0:
                commit_loss = F.mse_loss(quantize.detach(), x)
                loss = loss + commit_loss * self.commitment_weight

        quantize = self.project_out(quantize)
        quantize = rearrange(quantize, "b n d -> b d n")
        return quantize, embed_ind, loss


class ResidualVectorQuantization(nn.Module):
    """Residual vector quantization implementation.
    Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf
    """
    def __init__(self, *, num_quantizers, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList(
            [VectorQuantization(**kwargs) for _ in range(num_quantizers)]
        )

    def forward(self, x, n_q: tp.Optional[int] = None):
        quantized_out = 0.0
        residual = x

        all_losses = []
        all_indices = []

        n_q = n_q or len(self.layers)

        for layer in self.layers[:n_q]:
            quantized, indices, loss = layer(residual)
            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized

            all_indices.append(indices)
            all_losses.append(loss)

        out_losses, out_indices = map(torch.stack, (all_losses, all_indices))
        return quantized_out, out_indices, out_losses

    def encode(self, x: torch.Tensor, n_q: tp.Optional[int] = None) -> torch.Tensor:
        residual = x
        all_indices = []
        n_q = n_q or len(self.layers)
        for layer in self.layers[:n_q]:
            indices = layer.encode(residual)
            quantized = layer.decode(indices)
            residual = residual - quantized
            all_indices.append(indices)
        out_indices = torch.stack(all_indices)
        return out_indices

    def decode(self, q_indices: torch.Tensor) -> torch.Tensor:
        quantized_out = torch.tensor(0.0, device=q_indices.device)
        for i, indices in enumerate(q_indices):
            layer = self.layers[i]
            quantized = layer.decode(indices)
            quantized_out = quantized_out + quantized
        return quantized_out

class VQ2(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.vq1 = GumbelSoftmaxVectorQuantization(**kwargs)    
        self.vq2 = GumbelSoftmaxVectorQuantization(**kwargs)
    
    def forward(self, x, n_q: tp.Optional[int] = None):
        all_indices = []

        quantized1, indices1, loss1 = self.vq1(x)
        quantized2, indices2, loss2 = self.vq2(x)
        quantized_out = torch.cat([quantized1, quantized2], dim=1)
        all_indices.append(indices1)
        all_indices.append(indices2)
        out_indices = torch.stack(all_indices)
        out_losses = torch.mean(loss1 + loss2)

        return quantized_out, out_indices, out_losses

    def encode(self, x: torch.Tensor, n_q: tp.Optional[int] = None) -> torch.Tensor:
        all_indices = []
        indices1 = self.vq1.encode(x)
        indices2 = self.vq2.encode(x)
        all_indices.append(indices1)
        all_indices.append(indices2)
        out_indices = torch.stack(all_indices)
        return out_indices

    def decode(self, q_indices: torch.Tensor) -> torch.Tensor:
        quantized1 = self.vq1.decode(q_indices[0])
        quantized2 = self.vq2.decode(q_indices[1])
        quantized_out = torch.cat([quantized1, quantized2], dim=1)
        
        return quantized_out
