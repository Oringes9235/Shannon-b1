"""
PyTorch 数据集类
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Tuple, Optional


class TextDataset(Dataset):
    """文本数据集"""
    
    def __init__(self, texts: List[str], tokenizer, seq_len: int = 64, stride: Optional[int] = None):
        """
        初始化文本数据集

        Args:
            texts: 输入的文本列表
            tokenizer: 用于文本编码的分词器对象
            seq_len: 每个序列的长度，默认为64
            stride: 滑动窗口的步长，如果未指定则使用seq_len的一半
        """
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.stride = stride or seq_len // 2
        self.data = []
        
        # 遍历所有文本，将其转换为token序列
        for text in texts:
            tokens = tokenizer.encode(text)
            # 使用滑动窗口提取固定长度的序列片段
            for i in range(0, len(tokens) - seq_len - 1, self.stride):
                seq = tokens[i:i + seq_len + 1]
                if len(seq) == seq_len + 1:
                    self.data.append(seq)
        
        print(f"📊 Dataset created: {len(self.data)} sequences")
    
    def __len__(self) -> int:
        """返回数据集中序列的数量"""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取指定索引的数据项

        Args:
            idx: 数据项的索引

        Returns:
            包含输入序列和目标序列的元组
        """
        seq = self.data[idx]
        return torch.tensor(seq[:-1]), torch.tensor(seq[1:])
    
    def get_vocab_size(self) -> int:
        """获取词汇表大小"""
        return self.tokenizer.get_vocab_size()


class StreamingTextDataset(Dataset):
    """流式数据集 (适合大文件，不占用内存)"""
    
    def __init__(self, filepath: str, tokenizer, seq_len: int = 64):
        """
        初始化流式文本数据集

        Args:
            filepath: 文本文件的路径
            tokenizer: 用于文本编码的分词器对象
            seq_len: 每个序列的长度，默认为64
        """
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.filepath = filepath
        self._load_file()
    
    def _load_file(self):
        """
        从文件中加载文本数据并生成序列
        
        读取整个文件内容，进行tokenization，并使用滑动窗口创建训练序列
        """
        with open(self.filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        tokens = self.tokenizer.encode(text)
        self.data = []
        # 使用滑动窗口将文本切分为固定长度的序列
        for i in range(0, len(tokens) - self.seq_len - 1, self.seq_len // 2):
            seq = tokens[i:i + self.seq_len + 1]
            if len(seq) == self.seq_len + 1:
                self.data.append(seq)
        
        print(f"📊 Streaming dataset: {len(self.data)} sequences from {filepath}")
    
    def __len__(self):
        """返回数据集中序列的数量"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        获取指定索引的数据项

        Args:
            idx: 数据项的索引

        Returns:
            包含输入序列和目标序列的元组
        """
        seq = self.data[idx]
        return torch.tensor(seq[:-1]), torch.tensor(seq[1:])