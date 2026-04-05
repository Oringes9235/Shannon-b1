"""
自定义神经网络层
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """正弦余弦位置编码"""
    
    def __init__(self, d_model: int, max_seq_len: int = 512, dropout: float = 0.1):
        """
        初始化正弦余弦位置编码层

        Args:
            d_model (int): 模型的维度
            max_seq_len (int): 最大序列长度，默认为512
            dropout (float): dropout概率，默认为0.1
        """
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
        """
        前向传播

        Args:
            x: 输入张量，形状为(batch_size, seq_len, d_model)

        Returns:
            经过位置编码和dropout处理后的张量
        """
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            # 如果序列更长，扩展位置编码
            import math
            import torch
            new_pe = torch.zeros(seq_len, self.pe.size(2), device=x.device)
            new_pe[:self.pe.size(1)] = self.pe[0]
            for i in range(self.pe.size(1), seq_len):
                new_pe[i] = new_pe[i - 1] + (new_pe[self.pe.size(1)-1] - new_pe[self.pe.size(1)-2])
            self.pe = new_pe.unsqueeze(0)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """可学习位置编码"""
    
    def __init__(self, d_model: int, max_seq_len: int = 512, dropout: float = 0.1):
        """
        初始化可学习位置编码层

        Args:
            d_model (int): 模型的维度
            max_seq_len (int): 最大序列长度，默认为512
            dropout (float): dropout概率，默认为0.1
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
    
    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量，形状为(batch_size, seq_len, d_model)

        Returns:
            经过可学习位置编码和dropout处理后的张量
        """
        x = x + self.pos_embedding[:, :x.size(1), :]
        return self.dropout(x)


class CausalMask(nn.Module):
    """因果掩码 (防止看到未来信息)"""
    
    def __init__(self, max_seq_len: int = 512):
        """
        初始化因果掩码层

        Args:
            max_seq_len (int): 最大序列长度，默认为512
        """
        super().__init__()
        self.register_buffer("mask", self._create_mask(max_seq_len))
    
    def _create_mask(self, max_seq_len):
        """
        创建因果掩码矩阵

        Args:
            max_seq_len (int): 序列长度

        Returns:
            掩码矩阵，上三角部分为负无穷，其余为0
        """
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, seq_len: int):
        """
        前向传播

        Args:
            seq_len (int): 当前序列长度

        Returns:
            大小为(seq_len, seq_len)的因果掩码
        """
        return self.mask[:seq_len, :seq_len]


class RMSNorm(nn.Module):
    """RMSNorm (比 LayerNorm 更快)"""
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        """
        初始化RMSNorm层

        Args:
            d_model (int): 模型的维度
            eps (float): 防止除零的小常数，默认为1e-6
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量，形状为(..., d_model)

        Returns:
            经过RMSNorm归一化后的张量
        """
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)