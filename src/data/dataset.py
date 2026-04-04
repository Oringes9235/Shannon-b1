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
        Args:
            texts: 文本列表
            tokenizer: 分词器
            seq_len: 序列长度
            stride: 滑动窗口步长 (默认 seq_len // 2)
        """
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.stride = stride or seq_len // 2
        self.data = []
        
        for text in texts:
            tokens = tokenizer.encode(text)
            for i in range(0, len(tokens) - seq_len - 1, self.stride):
                seq = tokens[i:i + seq_len + 1]
                if len(seq) == seq_len + 1:
                    self.data.append(seq)
        
        print(f"📊 Dataset created: {len(self.data)} sequences")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.data[idx]
        return torch.tensor(seq[:-1]), torch.tensor(seq[1:])
    
    def get_vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()


class StreamingTextDataset(Dataset):
    """流式数据集 (适合大文件，不占用内存)"""
    
    def __init__(self, filepath: str, tokenizer, seq_len: int = 64):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.filepath = filepath
        self._load_file()
    
    def _load_file(self):
        """加载文件并生成序列"""
        with open(self.filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        tokens = self.tokenizer.encode(text)
        self.data = []
        for i in range(0, len(tokens) - self.seq_len - 1, self.seq_len // 2):
            seq = tokens[i:i + self.seq_len + 1]
            if len(seq) == self.seq_len + 1:
                self.data.append(seq)
        
        print(f"📊 Streaming dataset: {len(self.data)} sequences from {filepath}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        seq = self.data[idx]
        return torch.tensor(seq[:-1]), torch.tensor(seq[1:])