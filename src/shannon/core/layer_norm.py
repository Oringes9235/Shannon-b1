import numpy as np
from .parameter import Parameter, ones, zeros


class LayerNorm:
    """带可学习参数的LayerNorm"""
    
    def __init__(self, d_model):
        self.gamma = ones(d_model)
        self.beta = zeros(d_model)
        self.eps = 1e-5
    
    def forward(self, x):
        # 提取数据
        x_data = x.data if isinstance(x, Parameter) else x
        
        self.cache = {'x': x_data, 'x_param': x}
        self.mean = np.mean(x_data, axis=-1, keepdims=True)
        self.var = np.var(x_data, axis=-1, keepdims=True)
        self.std_inv = 1.0 / np.sqrt(self.var + self.eps)
        self.normalized = (x_data - self.mean) * self.std_inv
        return self.gamma.data * self.normalized + self.beta.data
    
    def backward(self, d_output):
        x = self.cache['x']
        N = x.shape[-1]
        
        d_x_norm = d_output * self.gamma.data
        d_var = np.sum(d_x_norm * (x - self.mean) * -0.5 * (self.var + self.eps)**(-1.5), axis=-1, keepdims=True)
        d_mean = np.sum(d_x_norm * -self.std_inv, axis=-1, keepdims=True) + \
                 d_var * np.mean(-2 * (x - self.mean), axis=-1, keepdims=True)
        d_x = d_x_norm * self.std_inv + d_var * 2 * (x - self.mean) / N + d_mean / N
        
        d_gamma = np.sum(d_output * self.normalized, axis=(0, 1))
        d_beta = np.sum(d_output, axis=(0, 1))
        
        self.gamma.grad = d_gamma
        self.beta.grad = d_beta
        
        if isinstance(self.cache['x_param'], Parameter):
            return d_x
        return d_x