import numpy as np
from ..utils.functions import gelu, gelu_backward, xavier_init
from .parameter import Parameter, zeros


class FeedForward:
    """前馈网络"""
    
    def __init__(self, d_model, d_ff):
        self.d_model = d_model
        self.d_ff = d_ff
        
        self.W1 = xavier_init(d_model, d_ff)
        self.b1 = zeros(d_ff)
        self.W2 = xavier_init(d_ff, d_model)
        self.b2 = zeros(d_model)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        self.cache = {'x': x}
        self.z = x @ self.W1.data + self.b1.data  # (batch, seq_len, d_ff)
        self.h = gelu(self.z)
        output = self.h @ self.W2.data + self.b2.data  # (batch, seq_len, d_model)
        return output
    
    def backward(self, d_output):
        """
        d_output: (batch, seq_len, d_model) 来自上层的梯度
        返回: d_x: (batch, seq_len, d_model)
        """
        # d_output 的形状: (batch, seq_len, d_model)
        batch, seq_len, d_model = d_output.shape
        
        # 对 W2 的梯度: (d_ff, d_model)
        # 需要将输入展平为 2D 进行计算
        h_flat = self.h.reshape(-1, self.d_ff)  # (batch * seq_len, d_ff)
        d_output_flat = d_output.reshape(-1, d_model)  # (batch * seq_len, d_model)
        
        d_W2 = h_flat.T @ d_output_flat  # (d_ff, d_model)
        d_b2 = np.sum(d_output_flat, axis=0)  # (d_model,)
        
        # 对 h 的梯度
        d_h_flat = d_output_flat @ self.W2.data.T  # (batch * seq_len, d_ff)
        d_h = d_h_flat.reshape(batch, seq_len, self.d_ff)
        
        # 通过 GELU
        d_z = d_h * gelu_backward(self.z)  # (batch, seq_len, d_ff)
        
        # 对 W1 的梯度
        x_flat = self.cache['x'].reshape(-1, self.d_model)  # (batch * seq_len, d_model)
        d_z_flat = d_z.reshape(-1, self.d_ff)  # (batch * seq_len, d_ff)
        
        d_W1 = x_flat.T @ d_z_flat  # (d_model, d_ff)
        d_b1 = np.sum(d_z_flat, axis=0)  # (d_ff,)
        
        # 对输入的梯度
        d_x_flat = d_z_flat @ self.W1.data.T  # (batch * seq_len, d_model)
        d_x = d_x_flat.reshape(batch, seq_len, self.d_model)
        
        # 保存梯度
        self.W1.grad = d_W1
        self.b1.grad = d_b1
        self.W2.grad = d_W2
        self.b2.grad = d_b2
        
        return d_x