from threading import Lock
from collections import deque
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import torch.nn.functional as F
import os

class NanoporeSegmentStorage:
    """高效存储纳米孔信号分段的HDF5存储类"""
    
    def __init__(self, file_path, segment_length=10240, chunk_size=1000):
        """
        初始化存储对象
        :param file_path: HDF5文件路径
        :param segment_length: 每个信号段的长度
        :param chunk_size: HDF5分块大小，影响I/O效率
        """
        self.file_path = file_path
        self.segment_length = segment_length
        self.chunk_size = chunk_size
        self.file_lock = Lock()  # 线程/进程安全
        self.total_segments = 0
        self._current_chunk = []
        
        # 第一次创建文件时初始化
        if not h5py.is_hdf5(file_path):
            with h5py.File(file_path, 'w') as f:
                f.create_dataset("signal_segments", 
                                shape=(0, segment_length), 
                                maxshape=(None, segment_length),
                                dtype=np.float32,
                                chunks=(chunk_size, segment_length),
                                compression="gzip",
                                compression_opts=3)
                f.attrs['segment_length'] = segment_length
                f.attrs['chunk_size'] = chunk_size
            self.total_segments = 0
        else:
            # 文件已存在时正确初始化
            with h5py.File(file_path, 'r') as f:
                self.total_segments = f['signal_segments'].shape[0]
        
        self.closed = False  # 初始化关闭状态标志
        
    def add_segment(self, segment):
        """添加一个信号段到缓冲区"""
        if self.closed:
            raise RuntimeError("Cannot add segment to closed storage")
        self._current_chunk.append(segment)
        if len(self._current_chunk) >= self.chunk_size:
            self._flush_buffer()
            
    def add_segments(self, segments):
        """添加多个信号段"""
        if self.closed:
            raise RuntimeError("Cannot add segments to closed storage")
        self._current_chunk.extend(segments)
        if len(self._current_chunk) >= self.chunk_size:
            self._flush_buffer()
            
    def _flush_buffer(self):
        """将缓冲区数据写入文件"""
        if not self._current_chunk:
            return
            
        with self.file_lock:
            with h5py.File(self.file_path, 'a') as f:
                ds = f['signal_segments']
                
                # 获取当前数据集大小
                current_size = ds.shape[0]
                new_size = current_size + len(self._current_chunk)
                
                # 扩展数据集尺寸
                ds.resize(new_size, axis=0)
                
                # 创建堆叠的数组
                chunk_array = np.stack(self._current_chunk, axis=0)
                
                # 直接写入扩展部分
                ds[current_size:new_size] = chunk_array
                
                # 更新总段数
                self.total_segments = new_size
                
                # 清除缓冲区
                self._current_chunk = []

    def finalize(self):
        """处理剩余缓冲区数据并关闭文件"""
        if self.closed:
            return
            
        if self._current_chunk:
            self._flush_buffer()
        self.closed = True  # 标记为已关闭
        self._current_chunk = None  # 防止后续添加


class HDF5SignalDataset(Dataset):
    """用于加载H5信号数据的PyTorch数据集类，支持单个或多个H5文件"""
    
    def __init__(self, file_path, target_length=10240, lazy_load=False):
        """
        初始化数据集
        :param file_path: H5文件路径（字符串）或文件路径列表（支持多个文件）
        :param target_length: 目标信号长度，默认10240
        :param lazy_load: 是否延迟加载文件句柄（True：每次读取时打开文件，False：保持文件打开）
                         对于多文件，建议使用True以节省文件句柄资源
        """
        # 支持单个文件路径或文件路径列表
        if isinstance(file_path, str):
            self.file_paths = [file_path]
        elif isinstance(file_path, (list, tuple)):
            self.file_paths = list(file_path)
        else:
            raise ValueError(f"file_path must be str, list or tuple, got {type(file_path)}")
        
        self.target_length = target_length
        self.lazy_load = lazy_load
        
        # 预计算每个文件的数据量，建立索引映射
        self.file_lengths = []
        self.file_offsets = []
        self.files = []  # 文件句柄列表（仅在非lazy模式下使用）
        # lazy 模式下：在每个 DataLoader worker 进程内缓存打开的文件句柄，避免每个样本都反复 open/close
        # 注意：不要在主进程提前打开（fork 后不安全），因此这里先占位 None，在 worker 内首次访问时再打开。
        self._lazy_files = []  # type: ignore[var-annotated]
        
        total_length = 0
        for path in self.file_paths:
            if lazy_load:
                # lazy模式：只记录文件路径，不打开文件
                with h5py.File(path, 'r') as f:
                    length = f['signal_segments'].shape[0]
                self.file_lengths.append(length)
                self.file_offsets.append(total_length)
                self.files.append(None)
                self._lazy_files.append(None)
            else:
                # 非lazy模式：保持文件打开
                f = h5py.File(path, 'r')
                length = f['signal_segments'].shape[0]
                self.file_lengths.append(length)
                self.file_offsets.append(total_length)
                self.files.append(f)
                self._lazy_files.append(None)
            
            total_length += length
        
        self.length = total_length
        
        print(f"Loaded {len(self.file_paths)} H5 file(s), total {self.length} segments")
        for i, (path, length) in enumerate(zip(self.file_paths, self.file_lengths)):
            print(f"  File {i+1}: {os.path.basename(path)} - {length} segments")
    
    def _find_file_index(self, idx):
        """
        根据全局索引找到对应的文件和文件内索引
        :param idx: 全局索引
        :return: (文件索引, 文件内索引)
        """
        # 使用二分查找找到对应的文件
        for file_idx in range(len(self.file_offsets)):
            offset = self.file_offsets[file_idx]
            if file_idx == len(self.file_offsets) - 1:
                # 最后一个文件
                if offset <= idx < offset + self.file_lengths[file_idx]:
                    return file_idx, idx - offset
            else:
                # 非最后一个文件
                if offset <= idx < self.file_offsets[file_idx + 1]:
                    return file_idx, idx - offset
        
        raise IndexError(f"Index {idx} out of range [0, {self.length})")
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # 找到对应的文件和文件内索引
        file_idx, local_idx = self._find_file_index(idx)
        
        # 读取数据
        if self.lazy_load:
            # lazy模式：每个 worker 进程内首次访问时打开文件并缓存复用
            f = self._lazy_files[file_idx]
            if f is None:
                f = h5py.File(self.file_paths[file_idx], 'r')
                self._lazy_files[file_idx] = f
            signal = f['signal_segments'][local_idx]
        else:
            # 非lazy模式：使用已打开的文件句柄
            signal = self.files[file_idx]['signal_segments'][local_idx]
        
        # 转换为tensor并处理长度
        signal_tensor = torch.tensor(signal).unsqueeze(0)  # 转换为[1, segment_length]
        current_length = signal_tensor.size(1)
        
        # 处理长度不足的情况
        if current_length < self.target_length:
            # 计算需要填充的长度
            pad_length = self.target_length - current_length
            # 在时间维度右侧填充0
            signal_tensor = F.pad(signal_tensor, (0, pad_length), "constant", 0)
        # 处理长度超出的情况
        elif current_length > self.target_length:
            # 截断前target_length个点
            signal_tensor = signal_tensor[:, :self.target_length]
        
        return signal_tensor  # 返回形状[1, target_length]

    def warmup_open_files(self) -> None:
        """
        在当前进程中提前打开 lazy 模式下的所有 H5 文件句柄。
        典型用法：在 DataLoader worker_init_fn 里调用，把“首次打开文件”的耗时集中到开头，
        避免训练中途出现固定 step 的卡顿。
        """
        if not self.lazy_load:
            return
        for i, path in enumerate(self.file_paths):
            if self._lazy_files[i] is None:
                self._lazy_files[i] = h5py.File(path, 'r')
    
    def __del__(self):
        # 关闭所有打开的文件句柄
        if hasattr(self, 'files'):
            for f in self.files:
                if f is not None:
                    try:
                        f.close()
                    except:
                        pass
        if hasattr(self, '_lazy_files'):
            for f in self._lazy_files:
                if f is not None:
                    try:
                        f.close()
                    except:
                        pass