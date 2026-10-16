import os
import sys
from torch import nn
import torch
import ast
import math
import torch.nn.functional as F


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from classes import BaseModelImpl
from vqvae_model import VQVAE,TransformerBlock,Decoder_basecalling
from wave_vqmodel import  _pad_to_multiple,WaveletMambaDualStreamVQVAE,CNN_decoder_stride,WaveletMambaDualStreamVQVAE_ogate,WaveletMambaDualStreamVQVAE_stride
from baseline_models import wavesem_oDWT,SingleCodebookVQVAE_stride,RCodebookVQVAE_stride

from layers.vq import VectorQuantizer,CNN_encoder_vq,CNN_decoder,CNN_encoder,Swish,CNN_encoder_43
# from layers.core_vq import ResidualVectorQuantization,VectorQuantization,VQ2
from layers.bonito import BonitoLSTM

_AUDIO_LM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "pretrain", "soundstream", "audiolm-pytorch-main"))
if _AUDIO_LM_ROOT not in sys.path:
    sys.path.insert(0, _AUDIO_LM_ROOT)
try:
    from audiolm_pytorch import SoundStream
except Exception:
    SoundStream = None

class VQVAEModel(nn.Module):
    def __init__(self, vqvae_model_path, stride=[2, 2,2, 2,2], freeze_encoder=True):
        super(VQVAEModel, self).__init__()
        checkpoint = torch.load(vqvae_model_path)
        self.vqvae_model = VQVAE(input_dim=1,
        hidden_dim=256,
        num_layers=5,
        embedding_dim=512,
        num_embeddings=512,
        num_quantizers=8,
        commitment_cost=0.25,
        compression_ratio=64,
        stride=stride)
        self.vqvae_model.load_state_dict(checkpoint['model_state_dict'])
        if freeze_encoder:
            for param in self.vqvae_model.parameters():
                param.requires_grad = False
        self.encoder = self.vqvae_model.encoder
        self.vq_layer = self.vqvae_model.vq_layer

        for param in self.encoder.parameters():
            param.requires_grad = True
        for param in self.vq_layer.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.encoder(x)
        x = x.transpose(1, 2)
        x, encoding_indices,vq_loss = self.vq_layer(x)
        x = x.transpose(1, 2)
        return x,torch.mean(vq_loss)

class VQVAEModel_stride(nn.Module):
    def __init__(self, vqvae_model_path, cnn_compression_ratio=16, strides=[2, 2, 2, 2,2,2], freeze_encoder=True):
        super(VQVAEModel_stride, self).__init__()
        checkpoint = torch.load(vqvae_model_path)
        self.vqvae_model = SingleCodebookVQVAE_stride(input_dim=1,
        hidden_dim=256,
        compression_ratio=cnn_compression_ratio,
        codebook_size=512,
        strides=strides)
        self.vqvae_model.load_state_dict(checkpoint['model_state_dict'])
        if freeze_encoder:
            for param in self.vqvae_model.parameters():
                param.requires_grad = False
        for param in self.vqvae_model.encoder.parameters():
            param.requires_grad = True
        for param in self.vqvae_model.vq.parameters():
            param.requires_grad = True
    def forward(self, x):
        x = self.vqvae_model.encoder(x)
        x = x.transpose(1, 2)
        x, encoding_indices,vq_loss = self.vqvae_model.vq(x)
        x = x.transpose(1, 2)
        return x,vq_loss

class RCodebookVQVAEModel_stride(nn.Module):
    def __init__(self, vqvae_model_path, cnn_compression_ratio=32, strides=[2, 2, 2, 2,2,2], freeze_encoder=True):
        super(RCodebookVQVAEModel_stride, self).__init__()
        checkpoint = torch.load(vqvae_model_path)
        self.vqvae_model = RCodebookVQVAE_stride(input_dim=1,
        hidden_dim=256,
        compression_ratio=cnn_compression_ratio,
        codebook_size=512,
        strides=strides)
        self.vqvae_model.load_state_dict(checkpoint['model_state_dict'])
        if freeze_encoder:
            for param in self.vqvae_model.parameters():
                param.requires_grad = False
        for param in self.vqvae_model.encoder.parameters():
            param.requires_grad = True
        for param in self.vqvae_model.vq.parameters():
            param.requires_grad = True
    def forward(self, x):
        x = self.vqvae_model.encoder(x)
        x = x.transpose(1, 2)
        x, encoding_indices,vq_loss = self.vqvae_model.vq(x)
        vq_loss = vq_loss.sum()
        x = x.transpose(1, 2)
        return x,vq_loss

class VQVAEModel_wavesem(nn.Module):
    def __init__(self, vqvae_model_path,cnn_compression_ratio=4, freeze_encoder=True):
        super(VQVAEModel_wavesem, self).__init__()
        checkpoint = torch.load(vqvae_model_path)
        self.wavesem = WaveletMambaDualStreamVQVAE(input_dim=1,
        hidden_dim=256,
        cnn_compression_ratio=cnn_compression_ratio,
        wavelet='db4',
        learnable=True)
        self.wavesem.load_state_dict(checkpoint['model_state_dict'])
        if freeze_encoder:
            for param in self.wavesem.parameters():
                param.requires_grad = False
        

        for param in self.wavesem.low_stream.parameters():
            param.requires_grad = True
        for param in self.wavesem.encoder.parameters():
            param.requires_grad = True
        for param in self.wavesem.multi_level_dwt.parameters():
            param.requires_grad = True
    def forward(self, x):
        x = self.wavesem.encoder(x)
        feat_low_1,_ = self.wavesem.multi_level_dwt(x)
        output,_,vq_loss_low = self.wavesem.low_stream(feat_low_1)
        return output,vq_loss_low

class VQVAEModel_wavesem_high(nn.Module):
    def __init__(self, vqvae_model_path,cnn_compression_ratio=4, freeze_encoder=True):
        super(VQVAEModel_wavesem_high, self).__init__()
        checkpoint = torch.load(vqvae_model_path)
        self.wavesem = WaveletMambaDualStreamVQVAE(input_dim=1,
        hidden_dim=256,
        cnn_compression_ratio=cnn_compression_ratio,
        wavelet='db4',
        learnable=True)
        self.wavesem.load_state_dict(checkpoint['model_state_dict'])
        if freeze_encoder:
            for param in self.wavesem.parameters():
                param.requires_grad = False
        

        for param in self.wavesem.high_stream.parameters():
            param.requires_grad = True
        for param in self.wavesem.encoder.parameters():
            param.requires_grad = True
        for param in self.wavesem.multi_level_dwt.parameters():
            param.requires_grad = True
    def forward(self, x):
        x = self.wavesem.encoder(x)
        feat_low_1,feat_high = self.wavesem.multi_level_dwt(x)
        output,_,vq_loss_low = self.wavesem.high_stream(feat_high[3], feat_high[2], feat_high[1], feat_high[0])
        return output,vq_loss_low


class SoundStreamEncoder(nn.Module):
    def __init__(
        self,
        soundstream_model_path: str,
        channels: int = 32,
        strides: list[int] = [2, 4, 4, 2],
        channel_mults: list[int] = [2, 4, 8, 16],
        codebook_dim: int = 512,
        codebook_size: int = 1024,
        rq_num_quantizers: int = 8,
        rq_groups: int = 1,
        use_local_attn: bool = True,
        attn_window_size: int = 128,
        attn_dim_head: int = 64,
        attn_heads: int = 8,
        attn_depth: int = 1,
        target_sample_hz: int = 4000,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        if SoundStream is None:
            raise RuntimeError("SoundStream import failed. Check audiolm_pytorch path.")

        self.soundstream = SoundStream(
            channels=channels,
            strides=tuple(strides),
            channel_mults=tuple(channel_mults),
            codebook_dim=codebook_dim,
            codebook_size=codebook_size,
            rq_num_quantizers=rq_num_quantizers,
            rq_groups=rq_groups,
            target_sample_hz=target_sample_hz,
            recon_loss_weight=1.0,
            multi_spectral_recon_loss_weight=0.0,
            adversarial_loss_weight=0.0,
            feature_loss_weight=0.0,
            use_local_attn=use_local_attn,
            attn_window_size=attn_window_size,
            attn_dim_head=attn_dim_head,
            attn_heads=attn_heads,
            attn_depth=attn_depth,
            input_channels=1,
        )

        self._load_checkpoint(soundstream_model_path)

        if freeze_encoder:
            for param in self.soundstream.parameters():
                param.requires_grad = False
        for param in self.soundstream.encoder.parameters():
            param.requires_grad = True
        if self.soundstream.encoder_attn is not None:
            for param in self.soundstream.encoder_attn.parameters():
                param.requires_grad = True
        for param in self.soundstream.rq.parameters():
            param.requires_grad = True

    def _load_checkpoint(self, ckpt_path: str) -> None:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            self.soundstream.load_state_dict(ckpt["model_state_dict"], strict=False)
            return
        if isinstance(ckpt, dict) and any(k in ckpt for k in ("model", "ema_model", "config")):
            self.soundstream.load(ckpt_path, strict=False)
            return
        if isinstance(ckpt, dict):
            self.soundstream.load_state_dict(ckpt, strict=False)
            return
        raise ValueError(f"Unrecognized SoundStream checkpoint: {ckpt_path}")

    def forward(self, x):
        encoded, _indices, commit_loss = self.soundstream(x, return_encoded=True)
        encoded = encoded.transpose(1, 2)
        return encoded, commit_loss.sum()

class SingleCodebookVQVAEModel_stride_basecalling(BaseModelImpl):
    def __init__(self,vqvae_model_path,cnn_compression_ratio=64,strides=[2,2,2,2,2,2],decoder_stride=[2,2,2],*args, **kwargs):
        super(SingleCodebookVQVAEModel_stride_basecalling, self).__init__(*args, **kwargs)
        if isinstance(strides, str):
            strides = ast.literal_eval(strides)  # 如果是字符串，转换为列表
        strides = [int(s) for s in strides]
        self.vqvae_model = VQVAEModel_stride(vqvae_model_path, cnn_compression_ratio=cnn_compression_ratio, strides=strides)
        if isinstance(decoder_stride, str):
            decoder_stride = ast.literal_eval(decoder_stride)  # 如果是字符串，转换为列表
        decoder_stride = [int(s) for s in decoder_stride] 
        self.decoder_basecalling = CNN_decoder_stride(in_channels=256, out_dim=512, strides=decoder_stride)
        self.dropout = nn.Dropout(0.2)
        self.bonito_encoder = nn.Sequential(BonitoLSTM(512, 512, reverse = False),
                                    BonitoLSTM(512, 512, reverse = True),
                                    BonitoLSTM(512, 512, reverse = False),
                                    BonitoLSTM(512, 512, reverse = True),
                                    BonitoLSTM(512, 512, reverse = False))
        self.decoder = self.build_decoder(encoder_output_size = 512, decoder_type = 'crf')
        self.decoder_type = 'crf'
    def forward(self, x):
        x,vq_loss = self.vqvae_model(x)
        # print(x.shape)
        x = self.decoder_basecalling(x)
        x = x.permute(2,0,1).contiguous()
        x = self.dropout(x)
        x = self.bonito_encoder(x)
        x = self.decoder(x)
        return x,vq_loss

class RCodebookVQVAEModel_stride_basecalling(BaseModelImpl):
    def __init__(self,vqvae_model_path,cnn_compression_ratio=32,strides=[2,2,2,2,2,2],decoder_stride=[2,2,2],*args, **kwargs):
        super(RCodebookVQVAEModel_stride_basecalling, self).__init__(*args, **kwargs)
        if isinstance(strides, str):
            strides = ast.literal_eval(strides)  # 如果是字符串，转换为列表
        strides = [int(s) for s in strides]
        self.vqvae_model = RCodebookVQVAEModel_stride(vqvae_model_path, cnn_compression_ratio=cnn_compression_ratio, strides=strides)
        if isinstance(decoder_stride, str):
            decoder_stride = ast.literal_eval(decoder_stride)  # 如果是字符串，转换为列表
        decoder_stride = [int(s) for s in decoder_stride] 
        self.decoder_basecalling = CNN_decoder_stride(in_channels=256, out_dim=512, strides=decoder_stride)
        self.dropout = nn.Dropout(0.2)
        self.bonito_encoder = nn.Sequential(BonitoLSTM(512, 512, reverse = False),
                                    BonitoLSTM(512, 512, reverse = True),
                                    BonitoLSTM(512, 512, reverse = False),
                                    BonitoLSTM(512, 512, reverse = True),
                                    BonitoLSTM(512, 512, reverse = False))
        self.decoder = self.build_decoder(encoder_output_size = 512, decoder_type = 'crf')
        self.decoder_type = 'crf'
    def forward(self, x):
        x,vq_loss = self.vqvae_model(x)
        # print(x.shape)
        x = self.decoder_basecalling(x)
        x = x.permute(2,0,1).contiguous()
        x = self.dropout(x)
        x = self.bonito_encoder(x)
        x = self.decoder(x)
        return x,vq_loss


class VQVAEModel_wavesem_all(nn.Module):
    def __init__(self, vqvae_model_path, freeze_encoder=True):
        super(VQVAEModel_wavesem_all, self).__init__()
        checkpoint = torch.load(vqvae_model_path)
        self.wavesem = WaveletMambaDualStreamVQVAE(input_dim=1,
        hidden_dim=256,
        cnn_compression_ratio=4,
        wavelet='db4',
        learnable=True)
        self.wavesem.load_state_dict(checkpoint['model_state_dict'])
        if freeze_encoder:
            for param in self.wavesem.parameters():
                param.requires_grad = False
        

        for param in self.wavesem.low_stream.parameters():
            param.requires_grad = True
        for param in self.wavesem.encoder.parameters():
            param.requires_grad = True
        for param in self.wavesem.multi_level_dwt.parameters():
            param.requires_grad = True
        for param in self.wavesem.high_stream.parameters():
            param.requires_grad = True
        for param in self.wavesem.high_stream.parameters():
            param.requires_grad = True
    def forward(self, x):
        x = self.wavesem.encoder(x)
        feat_low_1,feat_high = self.wavesem.multi_level_dwt(x)
        feat_low_2,indices_low,vq_loss_low = self.wavesem.low_stream(feat_low_1)
        feat_high_1,indices_high,vq_loss_high = self.wavesem.high_stream(feat_high[3],feat_high[2],feat_high[1],feat_high[0])
        vq_loss = vq_loss_low + vq_loss_high
        feat = torch.add(feat_low_2,feat_high_1)
        return feat,vq_loss

class VQVAEModel_wavesem_oDWT(nn.Module):
    def __init__(self, vqvae_model_path, freeze_encoder=True):
        super(VQVAEModel_wavesem_oDWT, self).__init__()
        checkpoint = torch.load(vqvae_model_path)
        self.wavesem = wavesem_oDWT(input_dim=1,
        hidden_dim=256,
        )
        self.wavesem.load_state_dict(checkpoint['model_state_dict'])
        if freeze_encoder:
            for param in self.wavesem.parameters():
                param.requires_grad = False
        

        for param in self.wavesem.encoder.parameters():
            param.requires_grad = True
        for param in self.wavesem.encoder_1.parameters():
            param.requires_grad = True
        for param in self.wavesem.low_stream.parameters():
            param.requires_grad = True
    def forward(self, x):
        x = self.wavesem.encoder(x)
        x_low_1,x_high_1 = x.split(int(self.wavesem.hidden_dim/2),dim=1)
        x_1 = self.wavesem.encoder_1(x_low_1)
        x_low,indices_low,vq_loss_low = self.wavesem.low_stream(x_1)
        return x_low,vq_loss_low

class VQVAEModel_wavesem_ogate(nn.Module):
    def __init__(self, vqvae_model_path, freeze_encoder=True):
        super(VQVAEModel_wavesem_ogate, self).__init__()
        checkpoint = torch.load(vqvae_model_path)
        self.wavesem = WaveletMambaDualStreamVQVAE_ogate(input_dim=1,
        hidden_dim=256,
        cnn_compression_ratio=4,
        wavelet='db4',
        learnable=True)
        self.wavesem.load_state_dict(checkpoint['model_state_dict'])
        if freeze_encoder:
            for param in self.wavesem.parameters():
                param.requires_grad = False
        

        for param in self.wavesem.encoder.parameters():
            param.requires_grad = True
        for param in self.wavesem.multi_level_dwt.parameters():
            param.requires_grad = True
        for param in self.wavesem.low_stream.parameters():
            param.requires_grad = True
    def forward(self, x):
        x = self.wavesem.encoder(x)
        feat_low_1,_ = self.wavesem.multi_level_dwt(x)
        feat_low_2,indices_low,vq_loss_low = self.wavesem.low_stream(feat_low_1)
        return feat_low_2,vq_loss_low

class VQVAEModel_wavesem_stride(nn.Module):
    def __init__(self, vqvae_model_path, strides="[2, 2, 2, 2]", freeze_encoder=True):
        super(VQVAEModel_wavesem_stride, self).__init__()
        self.strides = strides
        checkpoint = torch.load(vqvae_model_path)
        self.wavesem = WaveletMambaDualStreamVQVAE_stride(input_dim=1,
        hidden_dim=256,
        strides=strides,
        wavelet='db4',
        learnable=True)
        self.wavesem.load_state_dict(checkpoint['model_state_dict'])
        if freeze_encoder:
            for param in self.wavesem.parameters():
                param.requires_grad = False
        

        for param in self.wavesem.encoder.parameters():
            param.requires_grad = True
        for param in self.wavesem.multi_level_dwt.parameters():
            param.requires_grad = True
        for param in self.wavesem.low_stream.parameters():
            param.requires_grad = True
    def forward(self, x):
        # total_stride = math.prod(self.strides)
        # dwt_factor = 2 ** self.wavesem.multi_level_dwt.levels
        # pad_factor = math.lcm(total_stride, dwt_factor)
        # x, original_len = _pad_to_multiple(x, pad_factor)
        x = self.wavesem.encoder(x)
        feat_low_1,_ = self.wavesem.multi_level_dwt(x)
        feat_low_2,indices_low,vq_loss_low = self.wavesem.low_stream(feat_low_1)
        return feat_low_2,vq_loss_low

class VQVAEModel_wavesem_reconstruct(nn.Module):
    def __init__(self, vqvae_model_path, freeze_encoder=True):
        super(VQVAEModel_wavesem_reconstruct, self).__init__()
        checkpoint = torch.load(vqvae_model_path)
        self.wavesem = WaveletMambaDualStreamVQVAE(input_dim=1,
        hidden_dim=256,
        cnn_compression_ratio=4,
        wavelet='db4',
        learnable=True)
        self.wavesem.load_state_dict(checkpoint['model_state_dict'])
        if freeze_encoder:
            for param in self.wavesem.parameters():
                param.requires_grad = False

        # for param in self.wavesem.low_stream.parameters():
        #     param.requires_grad = True
        # for param in self.wavesem.encoder.parameters():
        #     param.requires_grad = True
        # for param in self.wavesem.multi_level_dwt.parameters():
        #     param.requires_grad = True
        # for param in self.wavesem.high_stream_decoder.parameters():
        #     param.requires_grad = True
        # for param in self.wavesem.multi_level_idwt.parameters():
        #     param.requires_grad = True
        # for param in self.wavesem.decoder.parameters():
        #     param.requires_grad = True
    # def forward(self, x):
    #     x = self.wavesem.encoder(x)
    #     feat_low_1,_ = self.wavesem.multi_level_dwt(x)
    #     output,_,vq_loss_low = self.wavesem.low_stream(feat_low_1)
    #     return output,vq_loss_low
    def forward(self, x):
        x =F.pad(x, (0, 56))
        feat_cnn_out,_,_,vq_loss_low,vq_loss_high,_,_,_,_ = self.wavesem(x)
        vq_loss = vq_loss_low + vq_loss_high
        feat_cnn_out = F.pad(feat_cnn_out, (0, -56))
        return feat_cnn_out,vq_loss

class VQVAEModel_wavesem_reconstruct_basecalling(BaseModelImpl):
    def __init__(self,vqvae_model_path,decoder_stride=[2,2,2],*args, **kwargs):
        super(VQVAEModel_wavesem_reconstruct_basecalling, self).__init__(*args, **kwargs)
        self.vqvae_model = VQVAEModel_wavesem_reconstruct(vqvae_model_path)
        self.dropout = nn.Dropout(0.2)
        self.encoder = CNN_encoder(in_channels=1, hidden_channels=128)
        self.bonito_encoder = nn.Sequential(BonitoLSTM(512, 512, reverse = False),
                                    BonitoLSTM(512, 512, reverse = True),
                                    BonitoLSTM(512, 512, reverse = False),
                                    BonitoLSTM(512, 512, reverse = True),
                                    BonitoLSTM(512, 512, reverse = False))
        self.decoder = self.build_decoder(encoder_output_size = 512, decoder_type = 'crf')
        self.decoder_type = 'crf'
    def forward(self, x):
        x,vq_loss = self.vqvae_model(x)
        # print(x.shape)
        x = self.encoder(x)
        # print(x.shape)
        x = x.permute(2,0,1).contiguous()
        x = self.dropout(x)
        x = self.bonito_encoder(x)
        x = self.decoder(x)
        return x,vq_loss

class VQVAEModel_wavesem_stride_basecalling(BaseModelImpl):
    def __init__(self,vqvae_model_path,strides="[2, 2, 2, 2]",decoder_stride=[2,2,2],*args, **kwargs):
        super(VQVAEModel_wavesem_stride_basecalling, self).__init__(*args, **kwargs)
        self.vqvae_model = VQVAEModel_wavesem_stride(vqvae_model_path, strides=strides)
        if isinstance(decoder_stride, str):
            decoder_stride = ast.literal_eval(decoder_stride)  # 如果是字符串，转换为列表
        decoder_stride = [int(s) for s in decoder_stride] 
        self.decoder_basecalling = CNN_decoder_stride(in_channels=256, out_dim=512, strides=decoder_stride)
        self.dropout = nn.Dropout(0.2)
        self.bonito_encoder = nn.Sequential(BonitoLSTM(512, 512, reverse = False),
                                    BonitoLSTM(512, 512, reverse = True),
                                    BonitoLSTM(512, 512, reverse = False),
                                    BonitoLSTM(512, 512, reverse = True),
                                    BonitoLSTM(512, 512, reverse = False))
        self.decoder = self.build_decoder(encoder_output_size = 512, decoder_type = 'crf')
        self.decoder_type = 'crf'
    def forward(self, x):
        x,vq_loss = self.vqvae_model(x)
        # print(x.shape)
        x = self.decoder_basecalling(x)
        x = x.permute(2,0,1).contiguous()
        x = self.dropout(x)
        x = self.bonito_encoder(x)
        x = self.decoder(x)
        return x,vq_loss


class SoundStream_basecalling(BaseModelImpl):
    def __init__(
        self,
        vqvae_model_path,
        decoder_stride=[2,2,2],
        ss_channels=32,
        ss_strides=[2,4,4,2],
        ss_channel_mults=[2,4,8,16],
        ss_codebook_dim=512,
        ss_codebook_size=1024,
        ss_rq_num_quantizers=8,
        ss_rq_groups=1,
        ss_use_local_attn=True,
        ss_attn_window_size=128,
        ss_attn_dim_head=64,
        ss_attn_heads=8,
        ss_attn_depth=1,
        ss_target_sample_hz=4000,
        *args, **kwargs
    ):
        super(SoundStream_basecalling, self).__init__(*args, **kwargs)
        if isinstance(decoder_stride, str):
            decoder_stride = ast.literal_eval(decoder_stride)
        decoder_stride = [int(s) for s in decoder_stride]
        if isinstance(ss_strides, str):
            ss_strides = ast.literal_eval(ss_strides)
        ss_strides = [int(s) for s in ss_strides]
        if isinstance(ss_channel_mults, str):
            ss_channel_mults = ast.literal_eval(ss_channel_mults)
        ss_channel_mults = [int(s) for s in ss_channel_mults]

        self.soundstream_encoder = SoundStreamEncoder(
            soundstream_model_path=vqvae_model_path,
            channels=ss_channels,
            strides=ss_strides,
            channel_mults=ss_channel_mults,
            codebook_dim=ss_codebook_dim,
            codebook_size=ss_codebook_size,
            rq_num_quantizers=ss_rq_num_quantizers,
            rq_groups=ss_rq_groups,
            use_local_attn=ss_use_local_attn,
            attn_window_size=ss_attn_window_size,
            attn_dim_head=ss_attn_dim_head,
            attn_heads=ss_attn_heads,
            attn_depth=ss_attn_depth,
            target_sample_hz=ss_target_sample_hz,
        )
        self.decoder_basecalling = CNN_decoder_stride(in_channels=ss_codebook_dim, out_dim=512, strides=decoder_stride)
        self.dropout = nn.Dropout(0.2)
        self.bonito_encoder = nn.Sequential(
            BonitoLSTM(512, 512, reverse=False),
            BonitoLSTM(512, 512, reverse=True),
            BonitoLSTM(512, 512, reverse=False),
            BonitoLSTM(512, 512, reverse=True),
            BonitoLSTM(512, 512, reverse=False),
        )
        self.decoder = self.build_decoder(encoder_output_size=512, decoder_type='crf')
        self.decoder_type = 'crf'

    def forward(self, x):
        x, vq_loss = self.soundstream_encoder(x)
        x = self.decoder_basecalling(x)
        x = x.permute(2, 0, 1).contiguous()
        x = self.dropout(x)
        x = self.bonito_encoder(x)
        x = self.decoder(x)
        return x, vq_loss

class VQVAEModel_wavesem_all_basecalling(BaseModelImpl):
    def __init__(self,vqvae_model_path,decoder_stride=[2,2,2],*args, **kwargs):
        super(VQVAEModel_wavesem_all_basecalling, self).__init__(*args, **kwargs)
        self.vqvae_model = VQVAEModel_wavesem_all(vqvae_model_path)
        if isinstance(decoder_stride, str):
            decoder_stride = ast.literal_eval(decoder_stride)  # 如果是字符串，转换为列表
        decoder_stride = [int(s) for s in decoder_stride] 
        self.decoder_basecalling = CNN_decoder_stride(in_channels=256, out_dim=512, strides=decoder_stride)
        self.dropout = nn.Dropout(0.2)
        self.bonito_encoder = nn.Sequential(BonitoLSTM(512, 512, reverse = False),
                                    BonitoLSTM(512, 512, reverse = True),
                                    BonitoLSTM(512, 512, reverse = False),
                                    BonitoLSTM(512, 512, reverse = True),
                                    BonitoLSTM(512, 512, reverse = False))
        self.decoder = self.build_decoder(encoder_output_size = 512, decoder_type = 'crf')
        self.decoder_type = 'crf'
    def forward(self, x):
        x,vq_loss = self.vqvae_model(x)
        # print(x.shape)
        x = self.decoder_basecalling(x)
        x = x.permute(2,0,1).contiguous()
        x = self.dropout(x)
        x = self.bonito_encoder(x)
        x = self.decoder(x)
        return x,vq_loss


class BasecallingModel(BaseModelImpl):
    def __init__(self,vqvae_model_path,decoder_stride=[2,2,2],*args, **kwargs):
        super(BasecallingModel, self).__init__(*args, **kwargs)
        self.encoder = CNN_encoder(in_channels=1, hidden_channels=128)
        self.dropout = nn.Dropout(0.2)
        self.bonito_encoder = nn.Sequential(BonitoLSTM(512, 512, reverse = False),
                                    BonitoLSTM(512, 512, reverse = True),
                                    BonitoLSTM(512, 512, reverse = False),
                                    BonitoLSTM(512, 512, reverse = True),
                                    BonitoLSTM(512, 512, reverse = False))
        self.decoder = self.build_decoder(encoder_output_size = 512, decoder_type = 'crf')
        self.decoder_type = 'crf'
    def forward(self, x):
        loss = 0.
        x = self.encoder(x)
        # print(x.shape)
        x = x.permute(2,0,1).contiguous()
        x = self.dropout(x)
        x = self.bonito_encoder(x)
        x = self.decoder(x)
        return x,loss


class VQVAEModel_wavesem_basecalling(BaseModelImpl):
    def __init__(self,vqvae_model_path,cnn_compression_ratio=2,decoder_stride=[2,2,2],*args, **kwargs):
        super(VQVAEModel_wavesem_basecalling, self).__init__(*args, **kwargs)
        self.vqvae_model = VQVAEModel_wavesem(vqvae_model_path,cnn_compression_ratio=cnn_compression_ratio)
        if isinstance(decoder_stride, str):
            decoder_stride = ast.literal_eval(decoder_stride)  # 如果是字符串，转换为列表
        decoder_stride = [int(s) for s in decoder_stride] 
        self.decoder_basecalling = CNN_decoder_stride(in_channels=256, out_dim=512, strides=decoder_stride)
        self.dropout = nn.Dropout(0.2)
        self.bonito_encoder = nn.Sequential(BonitoLSTM(512, 512, reverse = False),
                                    BonitoLSTM(512, 512, reverse = True),
                                    BonitoLSTM(512, 512, reverse = False),
                                    BonitoLSTM(512, 512, reverse = True),
                                    BonitoLSTM(512, 512, reverse = False))
        self.decoder = self.build_decoder(encoder_output_size = 512, decoder_type = 'crf')
        self.decoder_type = 'crf'
    def forward(self, x):
        x,vq_loss = self.vqvae_model(x)
        # print(x.shape)
        x = self.decoder_basecalling(x)
        x = x.permute(2,0,1).contiguous()
        x = self.dropout(x)
        x = self.bonito_encoder(x)
        x = self.decoder(x)
        return x,vq_loss

class VQVAEModel_wavesem_high_basecalling(BaseModelImpl):
    def __init__(self,vqvae_model_path,cnn_compression_ratio=2,decoder_stride=[2,2,2],*args, **kwargs):
        super(VQVAEModel_wavesem_high_basecalling, self).__init__(*args, **kwargs)
        self.vqvae_model = VQVAEModel_wavesem_high(vqvae_model_path,cnn_compression_ratio=cnn_compression_ratio)
        if isinstance(decoder_stride, str):
            decoder_stride = ast.literal_eval(decoder_stride)  # 如果是字符串，转换为列表
        decoder_stride = [int(s) for s in decoder_stride] 
        self.decoder_basecalling = CNN_decoder_stride(in_channels=256, out_dim=512, strides=decoder_stride)
        self.dropout = nn.Dropout(0.2)
        self.bonito_encoder = nn.Sequential(BonitoLSTM(512, 512, reverse = False),
                                    BonitoLSTM(512, 512, reverse = True),
                                    BonitoLSTM(512, 512, reverse = False),
                                    BonitoLSTM(512, 512, reverse = True),
                                    BonitoLSTM(512, 512, reverse = False))
        self.decoder = self.build_decoder(encoder_output_size = 512, decoder_type = 'crf')
        self.decoder_type = 'crf'
    def forward(self, x):
        x,vq_loss = self.vqvae_model(x)
        # print(x.shape)
        x = self.decoder_basecalling(x)
        x = x.permute(2,0,1).contiguous()
        x = self.dropout(x)
        x = self.bonito_encoder(x)
        x = self.decoder(x)
        return x,vq_loss

class VQVAEModel_wavesem_oDWT_basecalling(BaseModelImpl):
    def __init__(self,vqvae_model_path,decoder_stride=[2,2,2],*args, **kwargs):
        super(VQVAEModel_wavesem_oDWT_basecalling, self).__init__(*args, **kwargs)
        self.vqvae_model = VQVAEModel_wavesem_oDWT(vqvae_model_path)
        if isinstance(decoder_stride, str):
            decoder_stride = ast.literal_eval(decoder_stride)  # 如果是字符串，转换为列表
        decoder_stride = [int(s) for s in decoder_stride] 
        self.decoder_basecalling = CNN_decoder_stride(in_channels=256, out_dim=512, strides=decoder_stride)
        self.dropout = nn.Dropout(0.2)
        self.bonito_encoder = nn.Sequential(BonitoLSTM(512, 512, reverse = False),
                                    BonitoLSTM(512, 512, reverse = True),
                                    BonitoLSTM(512, 512, reverse = False),
                                    BonitoLSTM(512, 512, reverse = True),
                                    BonitoLSTM(512, 512, reverse = False))
        self.decoder = self.build_decoder(encoder_output_size = 512, decoder_type = 'crf')
        self.decoder_type = 'crf'
    def forward(self, x):
        x,vq_loss = self.vqvae_model(x)
        # print(x.shape)
        x = self.decoder_basecalling(x)
        x = x.permute(2,0,1).contiguous()
        x = self.dropout(x)
        x = self.bonito_encoder(x)
        x = self.decoder(x)
        return x,vq_loss

class VQVAEModel_wavesem_ogate_basecalling(BaseModelImpl):
    def __init__(self,vqvae_model_path,decoder_stride=[2,2,2],*args, **kwargs):
        super(VQVAEModel_wavesem_ogate_basecalling, self).__init__(*args, **kwargs)
        self.vqvae_model = VQVAEModel_wavesem_ogate(vqvae_model_path)
        if isinstance(decoder_stride, str):
            decoder_stride = ast.literal_eval(decoder_stride)  # 如果是字符串，转换为列表
        decoder_stride = [int(s) for s in decoder_stride] 
        self.decoder_basecalling = CNN_decoder_stride(in_channels=256, out_dim=512, strides=decoder_stride)
        self.dropout = nn.Dropout(0.2)
        self.bonito_encoder = nn.Sequential(BonitoLSTM(512, 512, reverse = False),
                                    BonitoLSTM(512, 512, reverse = True),
                                    BonitoLSTM(512, 512, reverse = False),
                                    BonitoLSTM(512, 512, reverse = True),
                                    BonitoLSTM(512, 512, reverse = False))
        self.decoder = self.build_decoder(encoder_output_size = 512, decoder_type = 'crf')
        self.decoder_type = 'crf'
    def forward(self, x):
        x,vq_loss = self.vqvae_model(x)
        # print(x.shape)
        x = self.decoder_basecalling(x)
        x = x.permute(2,0,1).contiguous()
        x = self.dropout(x)
        x = self.bonito_encoder(x)
        x = self.decoder(x)
        return x,vq_loss



class VQVAEModel_basecalling(BaseModelImpl):
    def __init__(self,vqvae_stride=[2, 2,2, 2,2],decoder_stride=[2,2],*args, **kwargs):
        if isinstance(vqvae_stride, str):
            vqvae_stride = ast.literal_eval(vqvae_stride)  # 如果是字符串，转换为列表
        
        vqvae_stride = [int(s) for s in vqvae_stride] 
        if isinstance(decoder_stride, str):
            decoder_stride = ast.literal_eval(decoder_stride)  # 如果是字符串，转换为列表
        decoder_stride = [int(s) for s in decoder_stride] 
        super(VQVAEModel_basecalling, self).__init__(*args, **kwargs)
        self.vqvae_model = VQVAEModel(vqvae_model_path='/owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain/outputs/ont_pretrain/best_model.pth', stride=vqvae_stride, freeze_encoder=True)
        self.decoder_basecalling = Decoder_basecalling(embedding_dim=512, hidden_dim=256, num_layers=6, output_dim=512, stride=decoder_stride)
        self.dropout = nn.Dropout(0.2)
        self.bonito_encoder = nn.Sequential(BonitoLSTM(512, 512, reverse = False),
                                    BonitoLSTM(512, 512, reverse = True),
                                    BonitoLSTM(512, 512, reverse = False),
                                    BonitoLSTM(512, 512, reverse = True),
                                    BonitoLSTM(512, 512, reverse = False))
        self.decoder = self.build_decoder(encoder_output_size = 512, decoder_type = 'crf')
        self.decoder_type = 'crf'
    def forward(self, x):
        x,vq_loss = self.vqvae_model(x)
        # print(x.shape)
        x = self.decoder_basecalling(x)
        x = x.permute(2,0,1).contiguous()
        x = self.dropout(x)
        x = self.bonito_encoder(x)
        x = self.decoder(x)
        return x,vq_loss

class VQModel_zip(BaseModelImpl):
    def __init__(self, num_embeddings, embedding_dim, hidden_channels, in_channels, *args, **kwargs):
        super(VQModel_zip, self).__init__(*args, **kwargs)

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.hidden_channels = hidden_channels
        self.in_channels = in_channels

        self.vq = ResidualVectorQuantization(
            dim=512,
            codebook_size=512,
            codebook_dim=512,
            num_quantizers=12,
            decay=0.99,
            epsilon=1e-5,
            kmeans_init=True,
            kmeans_iters=10,
            threshold_ema_dead_code=2,
        )

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels = 512, out_channels = 512, 
                      kernel_size = 5, stride= 1, padding=5//2,bias=True),
            Swish(),
            nn.BatchNorm1d(512),    
            nn.Conv1d(in_channels = 512, out_channels = 512, 
                      kernel_size = 5, stride= 1, padding=5//2,bias=True))
        self.encoder = CNN_encoder_vq(self.in_channels,self.hidden_channels)
        self.cnn_decoder = CNN_decoder(8*self.hidden_channels,8*self.hidden_channels)
        self.decoder = self.build_decoder(encoder_output_size = 8*self.hidden_channels, decoder_type = 'crf')
        self.lstm_encoder = self.build_encoder(encoder_output_size = 8*self.hidden_channels)
    
        self.decoder_type = 'crf'
    def build_encoder(self,encoder_output_size):
        encoder = nn.Sequential(BonitoLSTM(encoder_output_size, 512, reverse = False),
                                    BonitoLSTM(512, 512, reverse = True),
                                    BonitoLSTM(512, 512, reverse = False),
                                    BonitoLSTM(512, 512, reverse = True),
                                    BonitoLSTM(512, 512, reverse = False))
        return encoder
    def forward(self,x):
        x = self.encoder(x)
        # x = x.permute(2,0,1).contiguous()#[len,bs,hidden_channels]
        # x = self.lstm_encoder(x)
        x, encoding_indices,vq_loss = self.vq(x)
        # x = x.permute(1,2,0).contiguous()#[bs,channel,len]
        x = self.cnn_decoder(x)
        x = x.permute(2,0,1).contiguous()
        # loss = 0.0
        x = self.lstm_encoder(x)
        x = self.decoder(x)
        return x,torch.mean(vq_loss)


