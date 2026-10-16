from h5_data_process import NanoporeSegmentStorage
from tqdm import tqdm
import numpy as np
from bonito import pod5
import os
import h5py
import time
from concurrent.futures import ProcessPoolExecutor, as_completed


def segment_signal(signal, segment_length=10240):
    """
    将信号切分为固定长度的段，不足segment_length的剩余部分丢弃
    
    Args:
        signal: 原始信号数组
        segment_length: 每个段的长度，默认10240
        
    Returns:
        信号段的numpy数组列表
    """
    # 计算完整段的数量
    num_segments = len(signal) // segment_length
    
    # 如果信号长度不足一个段，返回空列表
    if num_segments == 0:
        return []
    
    # 使用reshape和切片，比循环快很多
    full_length = num_segments * segment_length
    signal_truncated = signal[:full_length].astype(np.float32)
    
    # 使用reshape直接切分，避免循环
    segments = signal_truncated.reshape(num_segments, segment_length)
    
    return list(segments)


def process_pod5_data(read_ids, pod5_directory, output_h5_path, segment_length=10240, 
                     chunk_size=500, batch_process=False):
    """
    处理POD5文件数据并保存为H5格式
    
    Args:
        read_ids: 读取ID列表
        pod5_directory: POD5文件目录路径
        output_h5_path: 输出H5文件路径
        segment_length: 信号段长度，默认10240
        chunk_size: H5写入chunk大小
        batch_process: 是否使用批量处理优化
    """
    # 读取POD5数据
    print("Loading POD5 reads...")
    data = pod5.get_reads(directory=pod5_directory, read_ids=read_ids)
    
    # 初始化H5存储
    storage = NanoporeSegmentStorage(
        file_path=output_h5_path,
        segment_length=segment_length,
        chunk_size=chunk_size
    )
    
    # 处理每个读取的信号
    total_segments = 0
    processed_count = 0
    skipped_count = 0
    
    # 批量处理模式：先收集所有segments再批量添加
    if batch_process:
        print("Processing reads batchly...")
        all_segments = []
        for read in tqdm(data, desc="Processing reads"):
            try:
                signal = read.signal
                signal_length = len(signal)
                
                if signal_length < 20000:
                    skipped_count += 1
                    continue
                
                # 切分信号（使用优化的向量化方法）
                segments = segment_signal(signal, segment_length)
                
                if segments:
                    all_segments.extend(segments)
                    processed_count += 1
                    
            except Exception as e:
                print(f"Error processing read {read.read_id}: {e}")
                skipped_count += 1
                continue
        
        # 批量添加到存储
        if all_segments:
            print(f"Adding {len(all_segments)} segments to storage...")
            storage.add_segments(all_segments)
            total_segments = len(all_segments)
    
    # 流式处理模式：逐个添加segments
    else:
        print("Processing reads streamly...")
        # 为了节省内存，立即删除不需要的数据
        for read in tqdm(data, desc="Processing reads"):
            try:
                signal = read.signal
                signal_length = len(signal)
                
                if signal_length < 20000:
                    skipped_count += 1
                    del signal  # 立即释放内存
                    continue
                
                # 切分信号
                segments = segment_signal(signal, segment_length)
                del signal  # 立即释放原始信号
                
                # 如果切分得到段，则添加到存储
                if segments:
                    storage.add_segments(segments)
                    total_segments += len(segments)
                    processed_count += 1
                del segments  # 释放segments内存
                    
            except Exception as e:
                print(f"Error processing read {read.read_id}: {e}")
                skipped_count += 1
                continue
    
    # 释放POD5数据
    del data
    
    # 完成写入并关闭文件
    storage.finalize()
    print(f"\nProcessing complete!")
    print(f"Processed reads: {processed_count}")
    print(f"Skipped reads: {skipped_count}")
    print(f"Total segments saved: {total_segments}")
    print(f"Output file: {output_h5_path}")


def process_single_read_batch(args):
    """
    处理单个read batch的函数（用于多进程）
    使用批量保存模式，避免内存累积
    """
    batch_idx, batch_read_ids, pod5_directory, batch_output, segment_length, chunk_size, save_batch_size = args
    
    try:
        start_time = time.time()
        
        # 加载POD5数据
        data = pod5.get_reads(directory=pod5_directory, read_ids=batch_read_ids)
        
        # 初始化H5存储
        storage = NanoporeSegmentStorage(
            file_path=batch_output,
            segment_length=segment_length,
            chunk_size=chunk_size
        )
        
        # 处理信号
        processed_count = 0
        skipped_count = 0
        total_segments = 0
        
        # 批量保存的segments列表
        batch_segments = []
        
        for read in data:
            try:
                signal = read.signal
                signal_length = len(signal)
                
                if signal_length < 20000:
                    skipped_count += 1
                    del signal
                    continue
                
                segments = segment_signal(signal, segment_length)
                del signal
                
                # 添加到批量列表
                if segments:
                    batch_segments.extend(segments)
                    total_segments += len(segments)
                    processed_count += 1
                    
                    # 达到批量大小时，立即写入
                    if len(batch_segments) >= save_batch_size:
                        storage.add_segments(batch_segments)
                        batch_segments = []  # 清空列表
                
                del segments
                    
            except Exception as e:
                skipped_count += 1
                continue
        
        # 写入剩余的segments
        if batch_segments:
            storage.add_segments(batch_segments)
        
        # 释放并保存
        del data
        del batch_segments
        storage.finalize()
        
        elapsed = time.time() - start_time
        
        return {
            'batch_idx': batch_idx,
            'success': True,
            'segments': total_segments,
            'processed': processed_count,
            'skipped': skipped_count,
            'time': elapsed
        }
        
    except Exception as e:
        elapsed = time.time() - start_time if 'start_time' in locals() else 0
        return {
            'batch_idx': batch_idx,
            'success': False,
            'error': str(e),
            'time': elapsed
        }


def merge_h5_files(h5_file_paths, output_path):
    """
    合并多个H5文件
    """
    print(f"\n{'='*70}")
    print(f"Merging {len(h5_file_paths)} H5 files...")
    print(f"{'='*70}")
    
    if not h5_file_paths:
        print("No files to merge!")
        return
    
    # 读取第一个文件获取属性
    with h5py.File(h5_file_paths[0], 'r') as f:
        segment_length = f.attrs['segment_length']
        chunk_size = f.attrs['chunk_size']
    
    # 计算总segments
    total_segments = 0
    for path in h5_file_paths:
        try:
            with h5py.File(path, 'r') as f:
                total_segments += f['signal_segments'].shape[0]
        except:
            continue
    
    print(f"Total segments to merge: {total_segments}")
    
    # 创建输出文件
    with h5py.File(output_path, 'w') as out_f:
        # 创建数据集
        ds = out_f.create_dataset(
            "signal_segments",
            shape=(total_segments, segment_length),
            dtype=np.float32,
            chunks=(chunk_size, segment_length),
            compression="gzip",
            compression_opts=3
        )
        out_f.attrs['segment_length'] = segment_length
        out_f.attrs['chunk_size'] = chunk_size
        
        # 复制数据
        offset = 0
        for h5_path in tqdm(h5_file_paths, desc="Merging"):
            try:
                with h5py.File(h5_path, 'r') as in_f:
                    data = in_f['signal_segments']
                    n_segments = data.shape[0]
                    ds[offset:offset+n_segments] = data[:]
                    offset += n_segments
            except Exception as e:
                print(f"Warning: Failed to merge {h5_path}: {e}")
                continue
    
    print(f"Successfully merged to {output_path}")
    print(f"Total segments: {total_segments}")
    print(f"{'='*70}\n")


def split_and_process(read_ids, pod5_directory, output_h5_path, segment_length=10240,
                     chunk_size=500, reads_per_batch=5000, num_workers=None, save_batch_size=1000):
    """
    分批处理并合并H5文件，支持并行处理
    
    Args:
        read_ids: read IDs列表
        pod5_directory: POD5目录路径
        output_h5_path: 最终输出H5文件路径
        segment_length: 段长度
        chunk_size: H5 chunk大小
        reads_per_batch: 每批处理的reads数量
        num_workers: 并行worker数量，None表示自动
    """
    print(f"\n{'='*70}")
    print(f"Split-and-merge processing mode")
    print(f"{'='*70}")
    
    # 创建临时目录
    output_dir = os.path.dirname(output_h5_path)
    base_name = os.path.splitext(os.path.basename(output_h5_path))[0]
    temp_dir = os.path.join(output_dir, f'temp_{base_name}')
    os.makedirs(temp_dir, exist_ok=True)
    
    # 计算批次
    num_batches = (len(read_ids) + reads_per_batch - 1) // reads_per_batch
    
    print(f"Total reads: {len(read_ids)}")
    print(f"Reads per batch: {reads_per_batch}")
    print(f"Number of batches: {num_batches}")
    print(f"Temp directory: {temp_dir}")
    
    # 准备批次参数
    batch_args = []
    h5_file_paths = []
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * reads_per_batch
        end_idx = min((batch_idx + 1) * reads_per_batch, len(read_ids))
        batch_read_ids = read_ids[start_idx:end_idx]
        
        batch_output = os.path.join(temp_dir, f'batch_{batch_idx:04d}.h5')
        h5_file_paths.append(batch_output)
        
        batch_args.append((
            batch_idx,
            batch_read_ids,
            pod5_directory,
            batch_output,
            segment_length,
            chunk_size,
            save_batch_size
        ))
    
    # 确定worker数量
    if num_workers is None:
        num_workers = min(os.cpu_count() or 4, 16)
    
    print(f"Parallel workers: {num_workers}\n")
    
    # 并行处理
    start_time = time.time()
    results = []
    
    if num_workers > 1:
        print(f"Starting parallel processing with {num_workers} workers...\n")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # 提交所有任务
            future_to_batch = {
                executor.submit(process_single_read_batch, args): args[0] 
                for args in batch_args
            }
            
            # 收集结果
            with tqdm(total=num_batches, desc="Processing batches") as pbar:
                for future in as_completed(future_to_batch):
                    result = future.result()
                    results.append(result)
                    
                    if result['success']:
                        pbar.set_postfix({
                            'Batch': result['batch_idx'],
                            'Segments': result['segments'],
                            'Time': f"{result['time']:.1f}s"
                        })
                        print(f"✓ Batch {result['batch_idx']}: {result['segments']} segments "
                              f"({result['processed']} reads) in {result['time']:.1f}s")
                    else:
                        print(f"✗ Batch {result['batch_idx']} failed: {result['error']}")
                    
                    pbar.update(1)
    else:
        # 单进程处理
        print("Processing sequentially...\n")
        for args in tqdm(batch_args, desc="Processing batches"):
            result = process_single_read_batch(args)
            results.append(result)
    
    # 统计结果
    elapsed_time = time.time() - start_time
    total_segments = sum(r['segments'] for r in results if r['success'])
    total_processed = sum(r['processed'] for r in results if r['success'])
    total_skipped = sum(r['skipped'] for r in results if r['success'])
    
    print(f"\n{'='*70}")
    print("Batch processing summary:")
    print(f"{'='*70}")
    print(f"Total batches: {num_batches}")
    print(f"Successful: {sum(1 for r in results if r['success'])}")
    print(f"Failed: {sum(1 for r in results if not r['success'])}")
    print(f"Total reads processed: {total_processed}")
    print(f"Total reads skipped: {total_skipped}")
    print(f"Total segments: {total_segments}")
    print(f"Total time: {elapsed_time:.1f}s ({elapsed_time/60:.1f} minutes)")
    print(f"{'='*70}\n")
    
    # 合并H5文件
    merge_h5_files(h5_file_paths, output_h5_path)
    
    # 清理临时文件
    print("Cleaning up temporary files...")
    for h5_path in h5_file_paths:
        try:
            if os.path.exists(h5_path):
                os.remove(h5_path)
        except:
            pass
    
    try:
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)
    except:
        pass
    
    print("Done!")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Process POD5 data to H5 format')
    parser.add_argument('--read_ids_file', type=str, required=False,
                        default='/owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain/raw_data/long_reads_id_2.txt',
                        help='Path to read IDs file')
    parser.add_argument('--pod5_dir', type=str, required=False,
                        default='/apdcephfs_qy3/share_301997302/chenchenqin/buddy1/jayjlzhou/giab_2025.01/flowcells/HG001/PAW79146/pod5',
                        help='POD5 directory path')
    parser.add_argument('--output', type=str, required=False,
                        default='/owenbhe/buddy1/lqszchen/ONT/ont_paper/ont_pretrain/raw_data/signal_segments_10240.h5',
                        help='Output H5 file path')
    parser.add_argument('--segment_length', type=int, default=10240,
                        help='Segment length (default: 10240)')
    parser.add_argument('--chunk_size', type=int, default=1000,
                        help='H5 chunk size (default: 500, smaller for memory efficiency)')
    parser.add_argument('--no_batch', action='store_true',
                        help='Disable batch processing mode (use streaming mode)')
    parser.add_argument('--split', action='store_true',
                        help='Split processing into multiple batches and merge (recommended for large datasets)')
    parser.add_argument('--reads_per_batch', type=int, default=10000,
                        help='Number of reads per batch when using --split (default: 10000)')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of parallel workers (default: auto, 0=sequential)')
    parser.add_argument('--save_batch_size', type=int, default=1000,
                        help='Number of segments to save in each batch within workers (default: 1000)')
    
    args = parser.parse_args()
    
    # 读取ID列表
    print("Loading read IDs...")
    with open(args.read_ids_file, 'r') as f:
        read_ids = f.readlines()
    read_ids = [read_id.strip() for read_id in read_ids]
    print(f"Total reads: {len(read_ids)}")
    
    # 处理数据
    if args.split:
        # 分批并行处理并合并
        split_and_process(
            read_ids=read_ids,
            pod5_directory=args.pod5_dir,
            output_h5_path=args.output,
            segment_length=args.segment_length,
            chunk_size=args.chunk_size,
            reads_per_batch=args.reads_per_batch,
            num_workers=args.num_workers,
            save_batch_size=args.save_batch_size
        )
    else:
        # 直接处理
        process_pod5_data(
            read_ids=read_ids,
            pod5_directory=args.pod5_dir,
            output_h5_path=args.output,
            segment_length=args.segment_length,
            chunk_size=args.chunk_size,
            batch_process=False  # 使用流式模式避免OOM
        )
