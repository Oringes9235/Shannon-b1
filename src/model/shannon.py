"""
Shannon-b1 主模型
"""

import torch
import torch.nn as nn
import math
from typing import Optional, List

from .config import ModelConfig
from .layers import PositionalEncoding, CausalMask


class ShannonB1(nn.Module):
    """
    Shannon-b1: 轻量级 GPT 风格语言模型
    
    架构:
    - Token Embedding
    - Positional Encoding
    - Transformer Decoder (多层)
    - Layer Norm
    - Output Projection
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config
        
        # 词嵌入
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(
            config.d_model, config.max_seq_len, config.dropout
        )
        
        # Transformer Decoder 层
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerDecoder(
            decoder_layer, num_layers=config.num_layers
        )
        
        # 因果掩码
        self.causal_mask = CausalMask(config.max_seq_len)
        
        # 最终 LayerNorm
        self.ln_f = nn.LayerNorm(config.d_model)
        
        # 输出投影
        self.output = nn.Linear(config.d_model, config.vocab_size)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            tokens: (batch, seq_len) 输入 token IDs
        
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        batch, seq_len = tokens.shape
        
        # 词嵌入 + 缩放
        x = self.token_embedding(tokens) * math.sqrt(self.config.d_model)
        
        # 位置编码
        x = self.pos_encoding(x)
        
        # 因果掩码
        mask = self.causal_mask(seq_len)
        
        # Transformer (自回归: 使用自己作为 memory)
        x = self.transformer(x, x, tgt_mask=mask)
        
        # LayerNorm
        x = self.ln_f(x)
        
        # 输出投影
        logits = self.output(x)
        
        return logits
    
    def generate(
        self, 
        start_tokens: List[int], 
        max_new_tokens: int, 
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0
    ) -> List[int]:
        """
        自回归生成文本
        
        Args:
            start_tokens: 起始 token 序列
            max_new_tokens: 最大生成 token 数
            temperature: 温度系数 (越高越随机)
            top_k: Top-K 采样
            top_p: Top-P (nucleus) 采样
        
        Returns:
            生成的 token 序列
        """
        self.eval()
        device = next(self.parameters()).device
        tokens = torch.tensor([start_tokens], device=device)
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # 前向传播
                logits = self.forward(tokens)
                
                # 获取最后一个位置
                last_logits = logits[0, -1, :] / temperature
                
                # Top-K 采样
                if top_k is not None:
                    indices_to_remove = last_logits < torch.topk(last_logits, top_k)[0][..., -1, None]
                    last_logits[indices_to_remove] = float('-inf')
                
                # Top-P (nucleus) 采样
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(last_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    last_logits = last_logits.masked_fill(indices_to_remove, float('-inf'))
                
                # 重复惩罚
                if repetition_penalty != 1.0:
                    for token in set(tokens[0].tolist()):
                        last_logits[token] /= repetition_penalty
                
                # Softmax 采样
                probs = torch.softmax(last_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # 拼接
                tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
        
        return tokens[0].tolist()


class ShannonB1Encoder(nn.Module):
    """编码器版本 (非自回归，用于理解任务)"""
    
    def __init__(self, config: ModelConfig):
        
        super().__init__()

        self.config = config
        
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_encoding = PositionalEncoding(config.d_model, config.max_seq_len, config.dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, config.num_layers)
        self.ln_f = nn.LayerNorm(config.d_model)
        self.output = nn.Linear(config.d_model, config.vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(tokens) * math.sqrt(self.config.d_model)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        x = self.ln_f(x)
        return self.output(x)