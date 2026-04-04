"""
优化器模块 - 完整功能
"""

import numpy as np


class SGD:
    """SGD优化器"""
    
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = {}
    
    def zero_grad(self):
        for name, param in self.params:
            if hasattr(param, 'grad') and param.grad is not None:
                param.grad = None
    
    def step(self):
        for name, param in self.params:
            if hasattr(param, 'grad') and param.grad is not None:
                grad = param.grad
                
                if self.weight_decay > 0:
                    grad = grad + self.weight_decay * param.data
                
                if self.momentum > 0:
                    if name not in self.velocities:
                        self.velocities[name] = np.zeros_like(grad)
                    self.velocities[name] = self.momentum * self.velocities[name] - self.lr * grad
                    param.data += self.velocities[name]
                else:
                    param.data -= self.lr * grad


class Adam:
    """Adam优化器"""
    
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = {}
        self.v = {}
        self.t = 0
    
    def zero_grad(self):
        for name, param in self.params:
            if hasattr(param, 'grad') and param.grad is not None:
                param.grad = None
    
    def step(self):
        self.t += 1
        
        bias_correction1 = 1 - self.beta1 ** self.t
        bias_correction2 = 1 - self.beta2 ** self.t
        
        for name, param in self.params:
            if hasattr(param, 'grad') and param.grad is not None:
                grad = param.grad
                
                if self.weight_decay > 0:
                    grad = grad + self.weight_decay * param.data
                
                if name not in self.m:
                    self.m[name] = np.zeros_like(grad)
                    self.v[name] = np.zeros_like(grad)
                
                self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
                self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (grad * grad)
                
                m_hat = self.m[name] / bias_correction1
                v_hat = self.v[name] / bias_correction2
                
                param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
    
    def get_state(self):
        return {'m': self.m, 'v': self.v, 't': self.t}
    
    def load_state(self, state):
        self.m = state.get('m', {})
        self.v = state.get('v', {})
        self.t = state.get('t', 0)


class AdamW:
    """AdamW优化器 (权重衰减解耦)"""
    
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = {}
        self.v = {}
        self.t = 0
    
    def zero_grad(self):
        for name, param in self.params:
            if hasattr(param, 'grad') and param.grad is not None:
                param.grad = None
    
    def step(self):
        self.t += 1
        
        bias_correction1 = 1 - self.beta1 ** self.t
        bias_correction2 = 1 - self.beta2 ** self.t
        
        for name, param in self.params:
            if hasattr(param, 'grad') and param.grad is not None:
                grad = param.grad
                
                if name not in self.m:
                    self.m[name] = np.zeros_like(grad)
                    self.v[name] = np.zeros_like(grad)
                
                self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
                self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (grad * grad)
                
                m_hat = self.m[name] / bias_correction1
                v_hat = self.v[name] / bias_correction2
                
                param.data = param.data * (1 - self.lr * self.weight_decay)
                param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
    
    def get_state(self):
        return {'m': self.m, 'v': self.v, 't': self.t}
    
    def load_state(self, state):
        self.m = state.get('m', {})
        self.v = state.get('v', {})
        self.t = state.get('t', 0)