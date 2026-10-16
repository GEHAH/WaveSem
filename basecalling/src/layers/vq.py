"""Implementations of the layers used in the Bonito model
"""

import torch
from torch import nn
import numpy as np
from constants import BASES
import warnings

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import torch.nn.functional as F

class Swish(nn.Module):
    def __init__(self,beta=1.0):
        super().__init__()
        self.beta = beta

    def forward(self,x):
        return x*torch.sigmoid(self.beta*x)

class CNN_encoder(nn.Module):
    def __init__(self,in_channels:int,hidden_channels:int):
        super(CNN_encoder,self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels = in_channels, out_channels = hidden_channels, 
                      kernel_size = 5, stride= 1, padding=5//2,bias=True),
            Swish(),
            nn.BatchNorm1d(hidden_channels),    
            nn.Conv1d(in_channels = hidden_channels, out_channels = hidden_channels, 
                      kernel_size = 5, stride= 1, padding=5//2,bias=True),
            Swish(),
            nn.BatchNorm1d(hidden_channels),
            nn.Conv1d(in_channels = hidden_channels, out_channels = 2*hidden_channels, 
                      kernel_size = 9, stride= 2, padding=9//2,bias=True),
            Swish(),
            nn.BatchNorm1d(2*hidden_channels),
            nn.Conv1d(in_channels = 2*hidden_channels, out_channels = 2*hidden_channels, 
                      kernel_size = 9, stride= 2, padding=9//2,bias=True),
            Swish(),
            nn.BatchNorm1d(2*hidden_channels),
            nn.Conv1d(in_channels = 2*hidden_channels, out_channels = 4*hidden_channels, 
                      kernel_size = 5, stride= 2, padding=5//2,bias=True),
            Swish(),
            nn.BatchNorm1d(4*hidden_channels),
            )
    def forward(self,x):
        return self.cnn(x)
    
class CNN_encoder_43(nn.Module):
    def __init__(self,in_channels:int,hidden_channels:int):
        super(CNN_encoder_43,self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels = in_channels, out_channels = 16, 
                      kernel_size = 5, stride= 1, padding=5//2,bias=True),
            Swish(),
            nn.BatchNorm1d(16),    
            nn.Conv1d(in_channels = 16, out_channels = 16, 
                      kernel_size = 5, stride= 2, padding=5//2,bias=True),
            Swish(),
            nn.BatchNorm1d(16),
            nn.Conv1d(in_channels = 16, out_channels = 16*hidden_channels, 
                      kernel_size = 19, stride= 3, padding=19//2,bias=True),
            Swish(),
            nn.BatchNorm1d(16*hidden_channels),
            )
    def forward(self,x):
        return self.cnn(x)

class CNN_encoder_vq(nn.Module): #压缩64倍
    def __init__(self,in_channels:int,hidden_channels:int):
        super(CNN_encoder_vq,self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels = in_channels, out_channels = hidden_channels, 
                      kernel_size = 5, stride= 1, padding=5//2,bias=True),
            Swish(),
            nn.BatchNorm1d(hidden_channels),    
            nn.Conv1d(in_channels = hidden_channels, out_channels = hidden_channels, 
                      kernel_size = 5, stride= 2, padding=5//2,bias=True),
            Swish(),
            nn.BatchNorm1d(hidden_channels),
            nn.Conv1d(in_channels = hidden_channels, out_channels = 2*hidden_channels, 
                      kernel_size = 9, stride= 4, padding=9//2,bias=True),
            Swish(),
            nn.BatchNorm1d(2*hidden_channels),
            nn.Conv1d(in_channels = 2*hidden_channels, out_channels = 2*hidden_channels, 
                      kernel_size = 9, stride= 4, padding=9//2,bias=True),
            Swish(),
            nn.BatchNorm1d(2*hidden_channels),
            nn.Conv1d(in_channels = 2*hidden_channels, out_channels = 8*hidden_channels, 
                      kernel_size = 5, stride= 4, padding=5//2,bias=True),
            Swish(),
            nn.BatchNorm1d(8*hidden_channels),
            )
    def forward(self,x):
        return self.cnn(x)

class CNN_decoder(nn.Module):
    def __init__(self,hidden_channels:int,out_channels:int):
        super(CNN_decoder,self).__init__()
        self.upsample_block1 = nn.Sequential(
            self._create_upsample_block(hidden_channels, 2*hidden_channels,scale_factor=2),
            self._create_upsample_block(2*hidden_channels, out_channels,scale_factor=2),
            self._create_upsample_block(out_channels, out_channels,scale_factor=4),
        )

    def _create_upsample_block(self, in_channels, out_channels, scale_factor=2):
        return nn.Sequential(
            # 首先使用反卷积进行上采样
            nn.ConvTranspose1d(in_channels, out_channels, 
                                kernel_size=2*scale_factor, 
                                stride=scale_factor, 
                                padding=scale_factor//2),
            Swish(),
            nn.BatchNorm1d(out_channels),  
                
            # 上采样后进行特征精炼
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=3//2),
            Swish(),
            nn.BatchNorm1d(out_channels)
        )
    def forward(self,x):
            return self.upsample_block1(x)
        




#向量量化模块
class VectorQuantizer(nn.Module):
    def __init__(self, embedding_dim,num_embedding,commitment_cost):
        super(VectorQuantizer,self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embedding
        self.commitment_cost = commitment_cost

		# initialize embeddings
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
    def forward(self, x):
        z = x.view(-1,self.embedding_dim)
        z = F.normalize(z, p=2, dim=-1)
        encoding_indices = self.get_code_indices(z)
        quantized = self.quantize(encoding_indices)
        q_latent_loss = F.mse_loss(quantized, z.detach())
        e_latent_loss = F.mse_loss(z, quantized.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        quantized = z + (quantized - z).detach().contiguous()
        quantized = quantized.view_as(x)
        return quantized, loss, encoding_indices
    def get_code_indices(self, x):
        distances = (
				torch.sum(x ** 2, dim=-1, keepdim=True) +
				torch.sum(F.normalize(self.embeddings.weight, p=2, dim=-1) ** 2, dim=1) -
				2. * torch.matmul(x, F.normalize(self.embeddings.weight.t(), p=2, dim=0))
		)

        encoding_indices = torch.argmin(distances, dim=1)

        return encoding_indices
    def quantize(self, encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        return F.normalize(self.embeddings(encoding_indices), p=2, dim=-1)

def get_stride(m):
    """Get the stride parameter if it has from a layer
    """
    if hasattr(m, 'stride'):
        return m.stride if isinstance(m.stride, int) else m.stride[0]
    if isinstance(m, torch.nn.Sequential):
        return int(np.prod([get_stride(x) for x in m]))
    try:
        get_stride(m.conv)
    except:
        return 1
    
class BonitoLinearCRFDecoder(nn.Module):
    """Linear decoder that has scores instead of bases to have
    base-base transitions to be able to use a CRF
    
    Copied from https://github.com/nanoporetech/bonito
    """

    def __init__(self, insize, n_base = 4, state_len = 4, bias=True, scale= 5.0, blank_score= 2.0):
        super(BonitoLinearCRFDecoder, self).__init__()
        """
        Args:
            insize (int): number of input channels
            n_base (int): number of bases
            state_len (int):
            bias (bool): whether the linear layer has bias
            scale (float): scores are multiplied by this value after tanh activation
            blank_score (float): 
            
        Defaults are based on the guppy configuration file
        """
        self.n_base = n_base
        self.state_len = state_len
        self.blank_score = blank_score
        size = (n_base + 1) * n_base**state_len if blank_score is None else n_base**(state_len + 1)
        self.linear = nn.Linear(insize, size, bias=bias)
        self.activation = nn.Tanh()
        self.scale = scale
        
    def forward(self, x):
        """
        Args:
            x (tensor): tensor with shape (timesteps, batch, channels)
        """
        scores = self.linear(x)
        if self.activation is not None:
            scores = self.activation(scores)
        if self.scale is not None:
            scores = scores * self.scale
        if self.blank_score is not None:
            T, N, C = scores.shape
            s = torch.tensor(self.blank_score, device=scores.device, dtype=scores.dtype)
            scores = torch.cat([s.expand(T, N, C//self.n_base, 1), scores.reshape(T, N, C//self.n_base, self.n_base)], 
                              axis=-1).reshape(T, N, -1)
            # T, N, C = scores.shape
            # scores = torch.nn.functional.pad(
            #     scores.view(T, N, C // self.n_base, self.n_base),
            #     (1, 0, 0, 0, 0, 0, 0, 0),
            #     value=self.blank_score
            # ).view(T, N, -1)
        return scores

try:
    # import seqdist.sparse
    # from seqdist.ctc_simple import logZ_cupy, viterbi_alignments
    # from seqdist.core import SequenceDist, Max, Log, semiring
    from koi.ctc import SequenceDist, Max, Log, semiring
    from koi.ctc import logZ_cu, viterbi_alignments, logZ_cu_sparse, bwd_scores_cu_sparse, fwd_scores_cu_sparse
    
    class CTC_CRF(SequenceDist):
        """CTC-CRF mix

        Copied from https://github.com/nanoporetech/bonito
        """

        def __init__(self, state_len = 4, alphabet = 'N'+''.join(BASES)):
            super().__init__()
            self.alphabet = alphabet
            self.state_len = state_len
            self.n_base = len(alphabet[1:])
            self.idx = torch.cat([
                torch.arange(self.n_base**(self.state_len))[:, None],
                torch.arange(
                    self.n_base**(self.state_len)
                ).repeat_interleave(self.n_base).reshape(self.n_base, -1).T
            ], dim=1).to(torch.int32)

            self.scores_dim = (self.n_base + 1) * self.n_base**self.state_len

        def n_score(self):
            return len(self.alphabet) * self.n_base**(self.state_len)

        def logZ(self, scores, S:semiring=Log):
            T, N, _ = scores.shape
            Ms = scores.reshape(T, N, -1, len(self.alphabet))
            alpha_0 = Ms.new_full((N, self.n_base**(self.state_len)), S.one)
            beta_T = Ms.new_full((N, self.n_base**(self.state_len)), S.one)
            return logZ_cu_sparse(Ms, self.idx, alpha_0, beta_T, S)

        def normalise(self, scores):
            return (scores - self.logZ(scores)[:, None] / len(scores))

        def forward_scores(self, scores, S: semiring=Log):
            T, N, _ = scores.shape
            Ms = scores.reshape(T, N, -1, self.n_base + 1)
            alpha_0 = Ms.new_full((N, self.n_base**(self.state_len)), S.one)
            return fwd_scores_cu_sparse(Ms, self.idx, alpha_0, S, K=1)

        def backward_scores(self, scores, S: semiring=Log):
            T, N, _ = scores.shape
            Ms = scores.reshape(T, N, -1, self.n_base + 1)
            beta_T = Ms.new_full((N, self.n_base**(self.state_len)), S.one)
            return bwd_scores_cu_sparse(Ms, self.idx, beta_T, S, K=1)

        def compute_transition_probs(self, scores, betas):
            T, N, C = scores.shape
            # add bwd scores to edge scores
            log_trans_probs = (scores.reshape(T, N, -1, self.n_base + 1) + betas[1:, :, :, None])
            # transpose from (new_state, dropped_base) to (old_state, emitted_base) layout
            log_trans_probs = torch.cat([
                log_trans_probs[:, :, :, [0]],
                log_trans_probs[:, :, :, 1:].transpose(3, 2).reshape(T, N, -1, self.n_base)
            ], dim=-1)
            # convert from log probs to probs by exponentiating and normalising
            trans_probs = torch.softmax(log_trans_probs, dim=-1)
            #convert first bwd score to initial state probabilities
            init_state_probs = torch.softmax(betas[0], dim=-1)
            return trans_probs, init_state_probs

        def reverse_complement(self, scores):
            T, N, C = scores.shape
            expand_dims = T, N, *(self.n_base for _ in range(self.state_len)), self.n_base + 1
            scores = scores.reshape(*expand_dims)
            blanks = torch.flip(scores[..., 0].permute(
                0, 1, *range(self.state_len + 1, 1, -1)).reshape(T, N, -1, 1), [0, 2]
            )
            emissions = torch.flip(scores[..., 1:].permute(
                0, 1, *range(self.state_len, 1, -1),
                self.state_len +2,
                self.state_len + 1).reshape(T, N, -1, self.n_base), [0, 2, 3]
            )
            return torch.cat([blanks, emissions], dim=-1).reshape(T, N, -1)

        def viterbi(self, scores):
            traceback = self.posteriors(scores, Max)
            paths = traceback.argmax(2) % len(self.alphabet)
            return paths

        def path_to_str(self, path):
            alphabet = np.frombuffer(''.join(self.alphabet).encode(), dtype='u1')
            seq = alphabet[path[path != 0]]
            return seq.tobytes().decode()

        def prepare_ctc_scores(self, scores, targets):
            # convert from CTC targets (with blank=0) to zero indexed
            targets = torch.clamp(targets - 1, 0)

            T, N, C = scores.shape
            scores = scores.to(torch.float32)
            n = targets.size(1) - (self.state_len - 1)
            stay_indices = sum(
                targets[:, i:n + i] * self.n_base ** (self.state_len - i - 1)
                for i in range(self.state_len)
            ) * len(self.alphabet)
            move_indices = stay_indices[:, 1:] + targets[:, :n - 1] + 1
            stay_scores = scores.gather(2, stay_indices.expand(T, -1, -1))
            move_scores = scores.gather(2, move_indices.expand(T, -1, -1))
            return stay_scores, move_scores

        def ctc_loss(self, scores, targets, target_lengths, loss_clip=None, reduction='mean', normalise_scores=True):
            if normalise_scores:
                # print(scores)
                scores = self.normalise(scores)

            stay_scores, move_scores = self.prepare_ctc_scores(scores, targets)
            logz = logZ_cu(stay_scores, move_scores, target_lengths + 1 - self.state_len)
            loss = - (logz / target_lengths)
            if loss_clip:
                loss = torch.clamp(loss, 0.0, loss_clip)
            if reduction == 'mean':
                return loss.mean()
            elif reduction in ('none', None):
                return loss
            else:
                raise ValueError('Unknown reduction type {}'.format(reduction))

        def ctc_viterbi_alignments(self, scores, targets, target_lengths):
            stay_scores, move_scores = self.prepare_ctc_scores(scores, targets)
            return viterbi_alignments(stay_scores, move_scores, target_lengths + 1 - self.state_len)

except ImportError as e:
    warnings.warn(str(e))    
    warnings.warn('CTC-CRF Decoder not available')

# try:
#     import seqdist.sparse
#     from seqdist.ctc_simple import logZ_cupy, viterbi_alignments
#     from seqdist.core import SequenceDist, Max, Log, semiring
    
#     class CTC_CRF(SequenceDist):
#         """CTC-CRF mix

#         Copied from https://github.com/nanoporetech/bonito
#         """

#         def __init__(self, state_len = 4, alphabet = 'N'+''.join(BASES)):
#             super().__init__()
#             self.alphabet = alphabet
#             self.state_len = state_len
#             self.n_base = len(alphabet[1:])
#             self.idx = torch.cat([
#                 torch.arange(self.n_base**(self.state_len))[:, None],
#                 torch.arange(
#                     self.n_base**(self.state_len)
#                 ).repeat_interleave(self.n_base).reshape(self.n_base, -1).T
#             ], dim=1).to(torch.int32)

#             self.scores_dim = (self.n_base + 1) * self.n_base**self.state_len

#         def n_score(self):
#             return len(self.alphabet) * self.n_base**(self.state_len)

#         def logZ(self, scores, S:semiring=Log):
#             T, N, _ = scores.shape
#             Ms = scores.reshape(T, N, -1, len(self.alphabet))
#             alpha_0 = Ms.new_full((N, self.n_base**(self.state_len)), S.one)
#             beta_T = Ms.new_full((N, self.n_base**(self.state_len)), S.one)
#             return seqdist.sparse.logZ(Ms, self.idx, alpha_0, beta_T, S)

#         def normalise(self, scores):
#             return (scores - self.logZ(scores)[:, None] / len(scores))

#         def forward_scores(self, scores, S: semiring=Log):
#             T, N, _ = scores.shape
#             Ms = scores.reshape(T, N, -1, self.n_base + 1)
#             alpha_0 = Ms.new_full((N, self.n_base**(self.state_len)), S.one)
#             return seqdist.sparse.fwd_scores_cupy(Ms, self.idx, alpha_0, S, K=1)

#         def backward_scores(self, scores, S: semiring=Log):
#             T, N, _ = scores.shape
#             Ms = scores.reshape(T, N, -1, self.n_base + 1)
#             beta_T = Ms.new_full((N, self.n_base**(self.state_len)), S.one)
#             return seqdist.sparse.bwd_scores_cupy(Ms, self.idx, beta_T, S, K=1)

#         def compute_transition_probs(self, scores, betas):
#             T, N, C = scores.shape
#             # add bwd scores to edge scores
#             log_trans_probs = (scores.reshape(T, N, -1, self.n_base + 1) + betas[1:, :, :, None])
#             # transpose from (new_state, dropped_base) to (old_state, emitted_base) layout
#             log_trans_probs = torch.cat([
#                 log_trans_probs[:, :, :, [0]],
#                 log_trans_probs[:, :, :, 1:].transpose(3, 2).reshape(T, N, -1, self.n_base)
#             ], dim=-1)
#             # convert from log probs to probs by exponentiating and normalising
#             trans_probs = torch.softmax(log_trans_probs, dim=-1)
#             #convert first bwd score to initial state probabilities
#             init_state_probs = torch.softmax(betas[0], dim=-1)
#             return trans_probs, init_state_probs

#         def reverse_complement(self, scores):
#             T, N, C = scores.shape
#             expand_dims = T, N, *(self.n_base for _ in range(self.state_len)), self.n_base + 1
#             scores = scores.reshape(*expand_dims)
#             blanks = torch.flip(scores[..., 0].permute(
#                 0, 1, *range(self.state_len + 1, 1, -1)).reshape(T, N, -1, 1), [0, 2]
#             )
#             emissions = torch.flip(scores[..., 1:].permute(
#                 0, 1, *range(self.state_len, 1, -1),
#                 self.state_len +2,
#                 self.state_len + 1).reshape(T, N, -1, self.n_base), [0, 2, 3]
#             )
#             return torch.cat([blanks, emissions], dim=-1).reshape(T, N, -1)

#         def viterbi(self, scores):
#             traceback = self.posteriors(scores, Max)
#             paths = traceback.argmax(2) % len(self.alphabet)
#             return paths

#         def path_to_str(self, path):
#             alphabet = np.frombuffer(''.join(self.alphabet).encode(), dtype='u1')
#             seq = alphabet[path[path != 0]]
#             return seq.tobytes().decode()

#         def prepare_ctc_scores(self, scores, targets):
#             # convert from CTC targets (with blank=0) to zero indexed
#             targets = torch.clamp(targets - 1, 0)

#             T, N, C = scores.shape
#             scores = scores.to(torch.float32)
#             n = targets.size(1) - (self.state_len - 1)
#             stay_indices = sum(
#                 targets[:, i:n + i] * self.n_base ** (self.state_len - i - 1)
#                 for i in range(self.state_len)
#             ) * len(self.alphabet)
#             move_indices = stay_indices[:, 1:] + targets[:, :n - 1] + 1
#             stay_scores = scores.gather(2, stay_indices.expand(T, -1, -1))
#             move_scores = scores.gather(2, move_indices.expand(T, -1, -1))
#             return stay_scores, move_scores

#         def ctc_loss(self, scores, targets, target_lengths, loss_clip=None, reduction='mean', normalise_scores=True):
#             if normalise_scores:
#                 scores = self.normalise(scores)
#             stay_scores, move_scores = self.prepare_ctc_scores(scores, targets)
#             logz = logZ_cupy(stay_scores, move_scores, target_lengths + 1 - self.state_len)
#             loss = - (logz / target_lengths)
#             if loss_clip:
#                 loss = torch.clamp(loss, 0.0, loss_clip)
#             if reduction == 'mean':
#                 return loss.mean()
#             elif reduction in ('none', None):
#                 return loss
#             else:
#                 raise ValueError('Unknown reduction type {}'.format(reduction))

#         def ctc_viterbi_alignments(self, scores, targets, target_lengths):
#             stay_scores, move_scores = self.prepare_ctc_scores(scores, targets)
#             return viterbi_alignments(stay_scores, move_scores, target_lengths + 1 - self.state_len)

# except ImportError as e:
#     warnings.warn(str(e))    
#     warnings.warn('CTC-CRF Decoder not available')