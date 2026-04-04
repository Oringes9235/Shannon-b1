"""
自定义神经网络层
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """正弦余弦位置编码"""
    
    def __init__(self, d_model: int, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """可学习位置编码"""
    
    def __init__(self, d_model: int, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
    
    def forward(self, x):
        x = x + self.pos_embedding[:, :x.size(1), :]
        return self.dropout(x)


class CausalMask(nn.Module):
    """因果掩码 (防止看到未来信息)"""
    
    def __init__(self, max_seq_len: int = 512):
        super().__init__()
        self.register_buffer("mask", self._create_mask(max_seq_len))
    
    def _create_mask(self, max_seq_len):
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, seq_len: int):
        return self.mask[:seq_len, :seq_len]


class RMSNorm(nn.Module):
    """RMSNorm (比 LayerNorm 更快)"""
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)