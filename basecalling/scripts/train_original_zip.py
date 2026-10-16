"""Script used for training for the CNN analysis experiment
"""

import os
import sys
import shutil

from torch.optim import lr_scheduler
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../models')))

from classes import BaseNanoporeDataset,basecalling_dataset
from schedulers import GradualWarmupScheduler
from constants import NON_RECURRENT_DECODING_DICT, NON_RECURRENT_ENCODING_DICT, RECURRENT_DECODING_DICT, RECURRENT_ENCODING_DICT

import torch
from torch.utils.data import DataLoader

import argparse
import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import ast

def str_to_list(s):
    """将字符串形式的列表转换为实际的列表"""
    if isinstance(s, str):
        return ast.literal_eval(s)
    return s

def generate_log_df(losses_keys, metrics_keys):
    """Creates a data.frame to store the logging values
    """
    
    header = ['epoch', # epoch number
              'step',  # step number
              'time']  # time it took
    # add losses and metrics for train and validation
    for k in losses_keys:
        header.append(k + '.train')
        header.append(k + '.val')
    for k in metrics_keys:
        header.append(k + '.train')
        header.append(k + '.val')
    # whether a checkpoint was saved at this step
    header.append('lr')
    header.append('checkpoint')
    
    log_dict = dict()
    for k in header:
        log_dict[k] = [None]
    return pd.DataFrame(log_dict)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir-train", type=str, default = '/owenbhe/buddy1/lqszchen/ONT/ont_basecalling/data/example_data_dna_r10.4.1_v0',help='Path where the training data for the dataloaders is stored')
    parser.add_argument("--data-dir-val", type=str, default = '/owenbhe/buddy1/lqszchen/ONT/ont_basecalling/data/example_data_dna_r10.4.1_v0/validation',help='Path where the validation data for the dataloaders is stored')
    parser.add_argument("--output-dir", type=str, default = '/owenbhe/buddy1/lqszchen/ONT/basecalling_architectures/results',help='Path where the model is saved')
    parser.add_argument("--model", type=str, choices=[
        'bonito',
        'catcaller',
        'causalcall',
        'mincall',
        'sacall',
        'urnano',
        'halcyon',
        'vq_model_rec_basecalling',
        'vq_model',
        'vq_model_wavesem_oDWT',
        'vq_model_wavesem_ogate',
        'vq_model_wavesem_all',
        'vq_model_wavesem_stride',
        'single_codebook_vqvae_stride_basecalling',
        'r_codebook_vqvae_stride_basecalling',
        'soundstream_basecalling',
        'basecalling',
        "VQVAEModel_wavesem_high_basecalling"
    ], help='Model')
    parser.add_argument("--window-size", type=int, choices=[400, 1000, 2000, 4000], help='Window size for the data')
    parser.add_argument("--num-epochs", type=int, default = 20)
    parser.add_argument("--batch-size", type=int, default = 128)
    parser.add_argument("--starting-lr", type=float, default = 0.001)
    parser.add_argument("--warmup-steps", type=int, default = 5000)
    parser.add_argument("--use-scaler", action='store_true', help='use 16bit float precision')
    parser.add_argument("--overwrite",default=True, action='store_true', help='delete existing files in folder')
    parser.add_argument("--checkpoint", type=str, help='checkpoint file to resume training')
    parser.add_argument("--vqvae_model_path", type=str, default = '/owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain/outputs/wave_vqmodel_small_cnn/checkpoints/best_model.pth', help='Path to the VQVAE model')
    parser.add_argument("--decoder_stride", type=str_to_list, default = [2,2,2], help='Stride for the decoder model')
    parser.add_argument("--strides", type=str, default = "[2, 2, 2, 2]", help='Stride for the VQVAE model')
    parser.add_argument("--cnn_compression_ratio", type=int, default = 4, help='Compression ratio for the CNN model')
    parser.add_argument("--ss_channels", type=int, default=32)
    parser.add_argument("--ss_strides", type=str_to_list, default=[2, 4, 4, 2])
    parser.add_argument("--ss_channel_mults", type=str_to_list, default=[2, 4, 8, 16])
    parser.add_argument("--ss_codebook_dim", type=int, default=512)
    parser.add_argument("--ss_codebook_size", type=int, default=1024)
    parser.add_argument("--ss_rq_num_quantizers", type=int, default=8)
    parser.add_argument("--ss_rq_groups", type=int, default=1)
    parser.add_argument("--ss_use_local_attn", action="store_true")
    parser.add_argument("--ss_attn_window_size", type=int, default=128)
    parser.add_argument("--ss_attn_dim_head", type=int, default=64)
    parser.add_argument("--ss_attn_heads", type=int, default=8)
    parser.add_argument("--ss_attn_depth", type=int, default=1)
    parser.add_argument("--ss_target_sample_hz", type=int, default=4000)
    args = parser.parse_args()
    
    validate_every = 500
    checkpoint_every = 20000

    data_dir_train = args.data_dir_train
    data_dir_val = args.data_dir_val

    if args.model == 'halcyon':
        from halcyon.model import HalcyonModelS2S as Model  # pyright: ignore[reportMissingImports]
        decoding_dict = RECURRENT_DECODING_DICT
        encoding_dict = RECURRENT_ENCODING_DICT
        s2s = True
    else:
        decoding_dict = NON_RECURRENT_DECODING_DICT
        encoding_dict = NON_RECURRENT_ENCODING_DICT
        s2s = False
        if args.model == 'bonito':
            from bonito.model import BonitoModel as Model  # pyright: ignore[reportMissingImports]
        elif args.model == 'catcaller':
            from catcaller.model import CATCallerModel as Model  # pyright: ignore[reportMissingImports]
        elif args.model == 'causalcall':
            from causalcall.model import CausalCallModel as Model  # pyright: ignore[reportMissingImports]
        elif args.model == 'mincall':
            from mincall.model import MinCallModel as Model  # pyright: ignore[reportMissingImports]
        elif args.model == 'sacall':
            from sacall.model import SACallModel as Model  # pyright: ignore[reportMissingImports]
        elif args.model == 'urnano':
            from urnano.model import URNanoModel as Model  # pyright: ignore[reportMissingImports]
        elif args.model == 'vq_model_rec_basecalling':
            from bonito.vq_model import VQVAEModel_wavesem_reconstruct_basecalling as Model  # pyright: ignore[reportMissingImports]
        elif args.model == 'basecalling':
            from bonito.vq_model import BasecallingModel as Model  # pyright: ignore[reportMissingImports]
        elif args.model == 'vq_model':
            from bonito.vq_model import VQVAEModel_wavesem_basecalling as Model  # pyright: ignore[reportMissingImports]
        elif args.model == 'vq_model_wavesem_oDWT':
            from bonito.vq_model import VQVAEModel_wavesem_oDWT_basecalling as Model  # pyright: ignore[reportMissingImports]
        elif args.model == 'vq_model_wavesem_ogate':
            from bonito.vq_model import VQVAEModel_wavesem_ogate_basecalling as Model  # pyright: ignore[reportMissingImports]
        elif args.model == 'vq_model_wavesem_all':
            from bonito.vq_model import VQVAEModel_wavesem_all_basecalling as Model  # pyright: ignore[reportMissingImports]
        elif args.model == 'vq_model_wavesem_stride':
            from bonito.vq_model import VQVAEModel_wavesem_stride_basecalling as Model  # pyright: ignore[reportMissingImports]
        elif args.model == 'single_codebook_vqvae_stride_basecalling':
            from bonito.vq_model import SingleCodebookVQVAEModel_stride_basecalling as Model  # pyright: ignore[reportMissingImports]
        elif args.model == 'r_codebook_vqvae_stride_basecalling':
            from bonito.vq_model import RCodebookVQVAEModel_stride_basecalling as Model  # pyright: ignore[reportMissingImports]
        elif args.model == 'soundstream_basecalling':
            from bonito.vq_model import SoundStream_basecalling as Model  # pyright: ignore[reportMissingImports]
        elif args.model == 'VQVAEModel_wavesem_high_basecalling':
            from bonito.vq_model import VQVAEModel_wavesem_high_basecalling as Model  # pyright: ignore[reportMissingImports]
    print('Creating dataset')
    train_dataset = basecalling_dataset(data_path = data_dir_train)
    val_dataset = basecalling_dataset(data_path = data_dir_val) 
    
    # dataset = BaseNanoporeDataset(
    #     data_dir = data_dir, 
    #     decoding_dict = decoding_dict, 
    #     encoding_dict = encoding_dict, 
    #     split = 0.95, 
    #     shuffle = True, 
    #     seed = 1,
    #     s2s = s2s,
    # )

    dataloader_train = DataLoader(
        train_dataset, 
        batch_size = args.batch_size, 
        num_workers = 1
    )
    dataloader_validation = DataLoader(
        val_dataset, 
        batch_size = args.batch_size, 
        num_workers = 1
    )


    if args.use_scaler:
        use_amp = True
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    else:
        use_amp = False
        scaler = None

    print('Creating model')
    # model = Model(
    #     num_embeddings=512,
    #     embedding_dim=256,
    #     hidden_channels=64,
    #     in_channels = 1,
    #     device = device,
    #     dataloader_train = dataloader_train, 
    #     dataloader_validation = dataloader_validation, 
    #     scaler = scaler,
    #     use_amp = use_amp,
    # )
    if args.model == 'soundstream_basecalling':
        model = Model(
            vqvae_model_path=args.vqvae_model_path,
            decoder_stride=args.decoder_stride,
            ss_channels=args.ss_channels,
            ss_strides=args.ss_strides,
            ss_channel_mults=args.ss_channel_mults,
            ss_codebook_dim=args.ss_codebook_dim,
            ss_codebook_size=args.ss_codebook_size,
            ss_rq_num_quantizers=args.ss_rq_num_quantizers,
            ss_rq_groups=args.ss_rq_groups,
            ss_use_local_attn=args.ss_use_local_attn,
            ss_attn_window_size=args.ss_attn_window_size,
            ss_attn_dim_head=args.ss_attn_dim_head,
            ss_attn_heads=args.ss_attn_heads,
            ss_attn_depth=args.ss_attn_depth,
            ss_target_sample_hz=args.ss_target_sample_hz,
            device = device,
            dataloader_train = dataloader_train,
            dataloader_validation = dataloader_validation,
            scaler = scaler,
            use_amp = use_amp
        )
    elif args.model == 'vq_model' or args.model == 'VQVAEModel_wavesem_high_basecalling':
        model = Model(
            vqvae_model_path=args.vqvae_model_path,
            cnn_compression_ratio=args.cnn_compression_ratio,
            decoder_stride=args.decoder_stride,
            device = device,
            dataloader_train = dataloader_train, 
            dataloader_validation = dataloader_validation, 
            scaler = scaler,
            use_amp = use_amp
        )
    elif args.model in (
        'vq_model_wavesem_stride',
    ):
        model = Model(
            vqvae_model_path=args.vqvae_model_path,
            strides=args.strides,
            decoder_stride=args.decoder_stride,
            device = device,
            dataloader_train = dataloader_train, 
            dataloader_validation = dataloader_validation, 
            scaler = scaler,
            use_amp = use_amp
        )
    elif args.model in('single_codebook_vqvae_stride_basecalling', 'r_codebook_vqvae_stride_basecalling'):
        model = Model(
            vqvae_model_path=args.vqvae_model_path,
            strides=args.strides,
            cnn_compression_ratio=args.cnn_compression_ratio,
            decoder_stride=args.decoder_stride,
            device = device,
            dataloader_train = dataloader_train, 
            dataloader_validation = dataloader_validation, 
            scaler = scaler,
            use_amp = use_amp
        )
    else:
        model = Model(
            vqvae_model_path=args.vqvae_model_path,
            decoder_stride=args.decoder_stride,
            device = device,
            dataloader_train = dataloader_train, 
            dataloader_validation = dataloader_validation, 
            scaler = scaler,
            use_amp = use_amp
        )
    model = model.to(device)

    print('Creating optimization')
    ##    OPTIMIZATION     #############################################
    optimizer = torch.optim.Adam(model.parameters(), lr=args.starting_lr)
    total_steps =  (int(len(train_dataset)+len(val_dataset))*args.num_epochs)/args.batch_size
    cosine_lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,total_steps, eta_min=0.00001, last_epoch=-1)
    lr_scheduler = GradualWarmupScheduler(optimizer, multiplier = 1.0, total_epoch = args.warmup_steps, after_scheduler=cosine_lr)
    schedulers = {'lr_scheduler': lr_scheduler}
    clipping_value = 2
    use_sam = False


    ##   MODEL PART2        #############################################
    model.optimizer = optimizer
    model.schedulers = schedulers
    model.clipping_value = clipping_value
    model.use_sam = use_sam

    if args.checkpoint is not None:
        model.load(args.checkpoint, initialize_lazy = True)
        model.to(device)

    print('Creating outputs')
    # output stuff
    output_dir = args.output_dir
    checkpoints_dir = os.path.join(output_dir, 'checkpoints')
    # os.makedirs(output_dir, exist_ok=True)
    # os.makedirs(checkpoints_dir, exist_ok=True)

    # check output dir
    if args.overwrite and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)

    if not os.path.isdir(output_dir):
        # create nested directories if parent paths don't exist
        os.makedirs(output_dir, exist_ok=True)
    else:
        if len(os.listdir(output_dir)) > 0 and not args.overwrite:
            raise FileExistsError('Output dir contains files')

    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # keep track of losses and metrics to take the average
    train_results = dict()
    
    print('Training')
    total_num_steps = 1
    best_val_acc = -float("inf")
    best_metric_key = None
    for epoch_num in range(args.num_epochs):
        
        loader_train = model.dataloader_train
        loader_validation = model.dataloader_validation
        # use this to restart the in case we finish all the validation data
        validation_iterator = iter(loader_validation) 
        
        st_time = time.time()
        # iterate over the train data
        for train_batch_num, train_batch in enumerate(loader_train):
            
            losses, predictions = model.train_step(train_batch)
            total_num_steps += 1
            
            for k, v in losses.items():
                if k not in train_results.keys():
                    train_results[k] = list()
                train_results[k].append(v)
            
            if total_num_steps % validate_every == 0:
                
                # calculate accuracy for the training only here since doing for every batch
                # is expensive and slow...
                predictions_decoded = model.decode(predictions, greedy = True)
                # print(predictions_decoded)
                metrics = model.evaluate(train_batch, predictions_decoded)
                
                # log the train results
                log_df = generate_log_df(list(losses.keys()), list(metrics.keys()))
                for k, v in train_results.items():
                    log_df[k + '.train'] = np.mean(v)
                for k, v in metrics.items():
                    log_df[k + '.train'] = np.mean(v)
                train_results = dict() # reset the dict
                
                try:
                    validation_batch = next(validation_iterator)
                except StopIteration:
                    validation_iterator = iter(loader_validation)
                    validation_batch = next(validation_iterator)
                                
                # calculate and log the validation results
                losses, predictions = model.validation_step(validation_batch)
                # print(losses)
                # print(predictions)
                predictions_decoded = model.decode(predictions, greedy = True)
                # print(predictions_decoded)
                metrics = model.evaluate(validation_batch, predictions_decoded)
                
                for k, v in losses.items():
                    log_df[k + '.val'] = v # do not need the mean as we only did it once
                for k, v in metrics.items():
                    log_df[k + '.val'] = np.mean(v)
                
                # determine which validation metric to use for checkpointing
                if best_metric_key is None and len(metrics) > 0:
                    if 'accuracy' in metrics:
                        best_metric_key = 'accuracy'
                    elif 'acc' in metrics:
                        best_metric_key = 'acc'
                    else:
                        # fall back to the first metric key
                        best_metric_key = list(metrics.keys())[0]
                
                # compute current validation accuracy (or chosen metric)
                current_val_acc = None
                if best_metric_key is not None:
                    v = metrics[best_metric_key]
                    try:
                        current_val_acc = float(np.mean(v))
                    except Exception:
                        current_val_acc = None
                    
                # calculate time it took since last validation step
                log_df['epoch'] = str(epoch_num)
                log_df['step'] = str(total_num_steps)
                log_df['time'] = int(time.time() - st_time)
                for param_group in model.optimizer.param_groups:
                    log_df['lr'] = param_group['lr']
                st_time = time.time()
                    
                # save the model if validation accuracy (or chosen metric) improves
                if current_val_acc is not None and current_val_acc > best_val_acc:
                    best_val_acc = current_val_acc
                    log_df['checkpoint'] = 'yes'
                    model.save(os.path.join(checkpoints_dir, 'best_checkpoint_' + str(total_num_steps) + '.pt'))
                else:
                    log_df['checkpoint'] = 'no'
                
                # write to log
                if not os.path.isfile(os.path.join(output_dir, 'train.log')):
                    log_df.to_csv(os.path.join(output_dir, 'train.log'), 
                                  header=True, index=False)
                else: # else it exists so append without writing the header
                    log_df.to_csv(os.path.join(output_dir, 'train.log'), 
                                  mode='a', header=False, index=False)
                    
                # write results to console
                print(log_df)
                
    
    model.save(os.path.join(checkpoints_dir, 'checkpoint_' + str(total_num_steps) + '.pt'))