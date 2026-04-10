"""
自定义神经网络层
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


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
    
    def forward(self, x, start_pos: int = 0):
        """
        前向传播

        Args:
            x: 输入张量，形状为(batch_size, seq_len, d_model)
            start_pos: 起始位置（用于增量解码）

        Returns:
            经过位置编码和dropout处理后的张量
        """
        seq_len = x.size(1)
        end_pos = start_pos + seq_len
        
        if end_pos > self.pe.size(1):
            # 如果序列更长，扩展位置编码
            import math
            import torch
            new_pe = torch.zeros(end_pos, self.pe.size(2), device=x.device)
            new_pe[:self.pe.size(1)] = self.pe[0]
            for i in range(self.pe.size(1), end_pos):
                new_pe[i] = new_pe[i - 1] + (new_pe[self.pe.size(1)-1] - new_pe[self.pe.size(1)-2])
            self.pe = new_pe.unsqueeze(0)
        
        x = x + self.pe[:, start_pos:end_pos, :]
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
        self.max_seq_len = max_seq_len
        self.register_buffer("mask", self._create_mask(max_seq_len))
    
    def _create_mask(self, seq_len):
        """
        创建因果掩码矩阵

        Args:
            seq_len (int): 序列长度

        Returns:
            掩码矩阵，上三角部分为负无穷，其余为0
        """
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
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
        # 如果请求的序列长度超过当前掩码大小，动态扩展
        if seq_len > self.mask.size(0):
            # 计算新的掩码大小（向上取整到2的幂次，避免频繁扩展）
            import math
            new_size = max(seq_len, int(2 ** math.ceil(math.log2(seq_len))))
            new_mask = self._create_mask(new_size).to(device=self.mask.device, dtype=self.mask.dtype)
            self.max_seq_len = new_size
            self.register_buffer("mask", new_mask)
        
        return self.mask[:seq_len, :seq_len]


class MultiHeadAttentionWithCache(nn.Module):
    """支持KV缓存的多头注意力机制"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """
        初始化多头注意力层
        
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            dropout: dropout概率
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Q, K, V 投影
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # 输出投影
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        前向传播
        
        Args:
            query: (batch, seq_len_q, d_model)
            key: (batch, seq_len_k, d_model)
            value: (batch, seq_len_v, d_model)
            mask: 注意力掩码
            past_key_value: 过去的(K, V)缓存
            
        Returns:
            output: (batch, seq_len_q, d_model)
            present_key_value: 新的(K, V)缓存（如果启用）
        """
        batch_size, seq_len_q, _ = query.size()
        
        # 线性投影
        q = self.q_proj(query).view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 如果有past_key_value，拼接到当前的K和V上
        if past_key_value is not None:
            past_key, past_value = past_key_value
            k = torch.cat([past_key, k], dim=2)
            v = torch.cat([past_value, v], dim=2)
        
        # 返回当前的K和V作为新的缓存
        present_key_value = (k, v)
        
        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 应用掩码
        if mask is not None:
            # 确保mask的形状匹配
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len_q, seq_len_k)
            attn_scores = attn_scores.masked_fill(mask == float('-inf'), float('-inf'))
        
        # Softmax + Dropout
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # 加权求和
        output = torch.matmul(attn_probs, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
        
        # 输出投影
        output = self.out_proj(output)
        
        return output, present_key_value


class TransformerDecoderLayerWithCache(nn.Module):
    """支持KV缓存的Transformer Decoder层"""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = 'gelu',
    ):
        """
        初始化Transformer Decoder层
        
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            dim_feedforward: FFN维度
            dropout: dropout概率
            activation: 激活函数
        """
        super().__init__()
        
        # Self-attention with cache support
        self.self_attn = MultiHeadAttentionWithCache(d_model, num_heads, dropout)
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.GELU() if activation == 'gelu' else nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)
        
        # Layer norms (使用 RMSNorm 以保持一致性)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.dropout_attn = nn.Dropout(dropout)
    
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        前向传播
        
        Args:
            tgt: 目标序列 (batch, seq_len, d_model)
            memory: 记忆序列 (batch, seq_len, d_model) - 在此实现中未用于Self-Attention，但保留接口兼容性
            tgt_mask: 目标掩码
            past_key_value: 过去的(K, V)缓存
            
        Returns:
            output: (batch, seq_len, d_model)
            present_key_value: 新的(K, V)缓存
        """
        # Self-attention
        attn_output, present_key_value = self.self_attn(
            query=tgt,
            key=tgt,
            value=tgt,
            mask=tgt_mask,
            past_key_value=past_key_value,
        )
        
        # Add & Norm
        tgt = tgt + self.dropout_attn(attn_output)
        tgt = self.norm1(tgt)
        
        # Feed-forward
        ff_output = self.linear2(self.dropout1(self.activation(self.linear1(tgt))))
        ff_output = self.dropout2(ff_output)
        
        # Add & Norm
        tgt = tgt + ff_output
        tgt = self.norm2(tgt)
        
        return tgt, present_key_value


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