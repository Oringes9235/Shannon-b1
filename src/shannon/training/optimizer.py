import numpy as np
from ..core.parameter import Parameter


class SGD:
    """SGD优化器"""
    
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        """
        params: list of (name, param_array) 元组
        lr: 学习率
        momentum: 动量系数
        weight_decay: 权重衰减系数
        """
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = {}
    
    def zero_grad(self):
        """清零所有梯度"""
        for name, param in self.params:
            if hasattr(param, 'grad'):
                param.grad = None
    
    def step(self):
        """更新参数"""
        for name, param in self.params:
            if hasattr(param, 'grad') and param.grad is not None:
                # 权重衰减
                if self.weight_decay > 0:
                    param.grad += self.weight_decay * param.data
                
                # 动量更新
                if self.momentum > 0:
                    if name not in self.velocity:
                        self.velocity[name] = np.zeros_like(param.grad)
                    self.velocity[name] = self.momentum * self.velocity[name] - self.lr * param.grad
                    param.data += self.velocity[name]
                else:
                    param.data -= self.lr * param.grad


class Adam:
    """
    Adam优化器 (Adaptive Moment Estimation)
    
    论文: https://arxiv.org/abs/1412.6980
    """
    
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        """
        params: list of (name, param) 元组
        lr: 学习率 (默认 0.001)
        betas: (beta1, beta2) 动量衰减率 (默认 (0.9, 0.999))
        eps: 数值稳定项 (默认 1e-8)
        weight_decay: 权重衰减系数 (默认 0.0)
        """
        self.params = params
        self.lr = lr
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.weight_decay = weight_decay
        
        # 存储动量项和二阶矩
        self.m = {}  # 一阶矩 (均值)
        self.v = {}  # 二阶矩 (未中心化方差)
        self.t = 0   # 时间步数
    
    def zero_grad(self):
        """清零所有梯度"""
        for name, param in self.params:
            if hasattr(param, 'grad'):
                param.grad = None
    
    def step(self):
        """执行参数更新"""
        self.t += 1
        
        for name, param in self.params:
            if hasattr(param, 'grad') and param.grad is not None:
                grad = param.grad
                
                # 权重衰减 (L2正则化)
                if self.weight_decay > 0:
                    grad = grad + self.weight_decay * param.data
                
                # 初始化动量项
                if name not in self.m:
                    self.m[name] = np.zeros_like(grad)
                    self.v[name] = np.zeros_like(grad)
                
                # 更新一阶矩和二阶矩
                self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
                self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (grad ** 2)
                
                # 偏差校正
                m_hat = self.m[name] / (1 - self.beta1 ** self.t)
                v_hat = self.v[name] / (1 - self.beta2 ** self.t)
                
                # 更新参数
                param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
    
    def get_state(self):
        """获取优化器状态 (用于保存检查点)"""
        return {
            'm': self.m,
            'v': self.v,
            't': self.t
        }
    
    def load_state(self, state):
        """加载优化器状态"""
        self.m = state.get('m', {})
        self.v = state.get('v', {})
        self.t = state.get('t', 0)


class AdamW:
    """
    AdamW优化器 (Adam with Weight Decay)
    
    论文: https://arxiv.org/abs/1711.05101
    将权重衰减与梯度更新解耦
    """
    
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        """
        params: list of (name, param) 元组
        lr: 学习率 (默认 0.001)
        betas: (beta1, beta2) 动量衰减率 (默认 (0.9, 0.999))
        eps: 数值稳定项 (默认 1e-8)
        weight_decay: 权重衰减系数 (默认 0.01)
        """
        self.params = params
        self.lr = lr
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.weight_decay = weight_decay
        
        self.m = {}
        self.v = {}
        self.t = 0
    
    def zero_grad(self):
        for name, param in self.params:
            if hasattr(param, 'grad'):
                param.grad = None
    
    def step(self):
        self.t += 1
        
        for name, param in self.params:
            if hasattr(param, 'grad') and param.grad is not None:
                grad = param.grad
                
                # 初始化动量项
                if name not in self.m:
                    self.m[name] = np.zeros_like(grad)
                    self.v[name] = np.zeros_like(grad)
                
                # 更新一阶矩和二阶矩
                self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
                self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (grad ** 2)
                
                # 偏差校正
                m_hat = self.m[name] / (1 - self.beta1 ** self.t)
                v_hat = self.v[name] / (1 - self.beta2 ** self.t)
                
                # 更新参数 (先做权重衰减，再做梯度更新)
                param.data = param.data * (1 - self.lr * self.weight_decay)
                param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
    
    def get_state(self):
        return {'m': self.m, 'v': self.v, 't': self.t}
    
    def load_state(self, state):
        self.m = state.get('m', {})
        self.v = state.get('v', {})
        self.t = state.get('t', 0)