import numpy as np
from .attention import MultiHeadAttention
from .feedforward import FeedForward
from .layer_norm import LayerNorm


class TransformerBlock:
    """完整的Transformer块"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        self.d_model = d_model
        self.dropout_rate = dropout
        
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)
        
        self.dropout_mask = None
    
    def dropout(self, x, training=True):
        if not training or self.dropout_rate == 0:
            return x
        self.dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, x.shape) / (1 - self.dropout_rate)
        return x * self.dropout_mask
    
    def dropout_backward(self, d_output, training=True):
        if not training or self.dropout_rate == 0:
            return d_output
        return d_output * self.dropout_mask
    
    def forward(self, x, mask=None, training=True):
        """x: (batch, seq_len, d_model)"""
        self.cache = {'training': training, 'x': x}
        
        # 第一个残差块: 注意力
        norm1 = self.ln1.forward(x)
        attn_output = self.attention.forward(norm1, mask)
        attn_output = self.dropout(attn_output, training)
        x1 = x + attn_output
        
        # 第二个残差块: FFN
        norm2 = self.ln2.forward(x1)
        ff_output = self.feed_forward.forward(norm2)
        ff_output = self.dropout(ff_output, training)
        x2 = x1 + ff_output
        
        self.cache.update({
            'norm1': norm1, 'attn_output': attn_output, 'x1': x1,
            'norm2': norm2, 'ff_output': ff_output
        })
        return x2
    
    def backward(self, d_output):
        """
        d_output: (batch, seq_len, d_model)
        返回: d_x: (batch, seq_len, d_model)
        """
        training = self.cache['training']
        
        # FFN 分支的梯度
        d_ff_output = self.dropout_backward(d_output, training)
        d_norm2 = self.feed_forward.backward(d_ff_output)
        d_x1 = self.ln2.backward(d_norm2)
        
        # 加上残差连接的梯度
        d_x1 = d_x1 + d_output
        
        # 注意力分支的梯度
        d_attn_output = self.dropout_backward(d_x1, training)
        d_norm1 = self.attention.backward(d_attn_output)
        d_x = self.ln1.backward(d_norm1)
        
        # 加上残差连接的梯度
        d_x = d_x + d_x1
        
        return d_x