import numpy as np
from ..utils.functions import softmax
from .parameter import Parameter


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q, K, V: (batch, num_heads, seq_len, head_dim)
    """
    d_k = Q.shape[-1]
    K_t = np.transpose(K, (0, 1, 3, 2))
    scores = np.matmul(Q, K_t) / np.sqrt(d_k)
    
    if mask is not None:
        scores = scores + mask
    
    attention_weights = softmax(scores, axis=-1)
    output = np.matmul(attention_weights, V)
    return output, attention_weights


class MultiHeadAttention:
    """多头注意力"""
    
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert self.head_dim * num_heads == d_model, "d_model 必须能被 num_heads 整除"
        
        limit = np.sqrt(6.0 / d_model)
        self.W_q = Parameter(np.random.uniform(-limit, limit, (d_model, d_model)))
        self.W_k = Parameter(np.random.uniform(-limit, limit, (d_model, d_model)))
        self.W_v = Parameter(np.random.uniform(-limit, limit, (d_model, d_model)))
        self.W_o = Parameter(np.random.uniform(-limit, limit, (d_model, d_model)))
    
    def _split_heads(self, x):
        """(batch, seq_len, d_model) -> (batch, num_heads, seq_len, head_dim)"""
        batch, seq_len, _ = x.shape
        x = x.reshape(batch, seq_len, self.num_heads, self.head_dim)
        return np.transpose(x, (0, 2, 1, 3))
    
    def _combine_heads(self, x):
        """(batch, num_heads, seq_len, head_dim) -> (batch, seq_len, d_model)"""
        batch, num_heads, seq_len, head_dim = x.shape
        x = np.transpose(x, (0, 2, 1, 3))
        return x.reshape(batch, seq_len, self.d_model)
    
    def forward(self, x, mask=None):
        """x: (batch, seq_len, d_model)"""
        self.cache = {'x': x}
        
        # 线性投影
        Q = x @ self.W_q.data  # (batch, seq_len, d_model)
        K = x @ self.W_k.data
        V = x @ self.W_v.data
        
        # 拆分多头
        Q = self._split_heads(Q)  # (batch, num_heads, seq_len, head_dim)
        K = self._split_heads(K)
        V = self._split_heads(V)
        
        # 注意力
        attn_output, attention_weights = scaled_dot_product_attention(Q, K, V, mask)
        
        # 合并多头
        concat_attention = self._combine_heads(attn_output)  # (batch, seq_len, d_model)
        
        # 输出投影
        output = concat_attention @ self.W_o.data
        
        # 缓存用于反向传播
        self.cache.update({
            'Q': Q, 'K': K, 'V': V,
            'attention_weights': attention_weights,
            'concat_attention': concat_attention
        })
        
        return output
    
    def backward(self, d_output):
        """
        d_output: (batch, seq_len, d_model)
        返回: d_x: (batch, seq_len, d_model)
        """
        batch, seq_len, d_model = d_output.shape
        
        # 展平用于矩阵计算
        d_output_flat = d_output.reshape(-1, d_model)
        concat_attention_flat = self.cache['concat_attention'].reshape(-1, d_model)
        
        # 对 W_o 的梯度
        d_W_o = concat_attention_flat.T @ d_output_flat
        self.W_o.grad = d_W_o
        
        # 对 concat_attention 的梯度
        d_concat_attention_flat = d_output_flat @ self.W_o.data.T
        d_concat_attention = d_concat_attention_flat.reshape(batch, seq_len, d_model)
        
        # 拆分回多头
        d_attn_output = self._split_heads(d_concat_attention)  # (batch, num_heads, seq_len, head_dim)
        
        # 通过注意力层反向传播
        d_Q, d_K, d_V = self._attention_backward(d_attn_output)
        
        # 合并多头梯度 (保持形状为 batch, seq_len, d_model)
        d_Q_combined = self._combine_heads(d_Q)  # (batch, seq_len, d_model)
        d_K_combined = self._combine_heads(d_K)
        d_V_combined = self._combine_heads(d_V)
        
        # 对权重矩阵的梯度
        x_flat = self.cache['x'].reshape(-1, d_model)
        d_Q_flat = d_Q_combined.reshape(-1, d_model)
        d_K_flat = d_K_combined.reshape(-1, d_model)
        d_V_flat = d_V_combined.reshape(-1, d_model)
        
        d_W_q = x_flat.T @ d_Q_flat
        d_W_k = x_flat.T @ d_K_flat
        d_W_v = x_flat.T @ d_V_flat
        
        self.W_q.grad = d_W_q
        self.W_k.grad = d_W_k
        self.W_v.grad = d_W_v
        
        # 对输入的梯度
        d_x_flat = (d_Q_flat @ self.W_q.data.T + 
                    d_K_flat @ self.W_k.data.T + 
                    d_V_flat @ self.W_v.data.T)
        d_x = d_x_flat.reshape(batch, seq_len, d_model)
        
        return d_x
    
    def _attention_backward(self, d_output):
        """
        注意力层的反向传播
        d_output: (batch, num_heads, seq_len, head_dim)
        返回: d_Q, d_K, d_V 同形状
        """
        Q = self.cache['Q']  # (batch, num_heads, seq_len, head_dim)
        K = self.cache['K']
        V = self.cache['V']
        attention_weights = self.cache['attention_weights']  # (batch, num_heads, seq_len, seq_len)
        
        batch, num_heads, seq_len, head_dim = Q.shape
        d_k = head_dim
        
        # d_output 对 V 的梯度
        # attention_weights: (b, h, q, k) @ d_output: (b, h, q, d) -> (b, h, k, d)
        d_V = np.transpose(attention_weights, (0, 1, 3, 2)) @ d_output
        
        # d_output 对 attention_weights 的梯度
        # d_output: (b, h, q, d) @ V^T: (b, h, d, k) -> (b, h, q, k)
        d_attention_weights = d_output @ np.transpose(V, (0, 1, 3, 2))
        
        # d_attention_weights 对 scores 的梯度 (softmax 反向传播)
        # d_scores = attention_weights * (d_attention_weights - sum(d_attention_weights * attention_weights))
        d_scores = attention_weights * (d_attention_weights - np.sum(d_attention_weights * attention_weights, axis=-1, keepdims=True))
        
        # d_scores 对 Q, K 的梯度
        # scores = Q @ K^T / sqrt(d_k)
        # d_Q = d_scores @ K / sqrt(d_k)
        # d_K = d_scores^T @ Q / sqrt(d_k)
        d_Q = d_scores @ K / np.sqrt(d_k)
        d_K = np.transpose(d_scores, (0, 1, 3, 2)) @ Q / np.sqrt(d_k)
        
        return d_Q, d_K, d_V