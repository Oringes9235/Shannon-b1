"""
可训练参数包装类
"""

import numpy as np


class Parameter:
    """
    可训练参数，包装 numpy 数组并添加梯度
    """
    
    def __init__(self, data):
        """
        data: numpy 数组
        """
        self.data = data
        self.grad = None
    
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def dtype(self):
        return self.data.dtype
    
    def __add__(self, other):
        if isinstance(other, Parameter):
            return Parameter(self.data + other.data)
        return Parameter(self.data + other)
    
    def __sub__(self, other):
        if isinstance(other, Parameter):
            return Parameter(self.data - other.data)
        return Parameter(self.data - other)
    
    def __mul__(self, other):
        if isinstance(other, Parameter):
            return Parameter(self.data * other.data)
        return Parameter(self.data * other)
    
    def __truediv__(self, other):
        if isinstance(other, Parameter):
            return Parameter(self.data / other.data)
        return Parameter(self.data / other)
    
    def __iadd__(self, other):
        if isinstance(other, Parameter):
            self.data += other.data
        else:
            self.data += other
        return self
    
    def __isub__(self, other):
        if isinstance(other, Parameter):
            self.data -= other.data
        else:
            self.data -= other
        return self
    
    def __imul__(self, other):
        if isinstance(other, Parameter):
            self.data *= other.data
        else:
            self.data *= other
        return self
    
    def __itruediv__(self, other):
        if isinstance(other, Parameter):
            self.data /= other.data
        else:
            self.data /= other
        return self
    
    def __repr__(self):
        return f"Parameter(shape={self.shape}, dtype={self.dtype})"
    
    def numpy(self):
        """返回原始 numpy 数组"""
        return self.data
    
    def zero_grad(self):
        """清零梯度"""
        self.grad = None
    
    def copy(self):
        """复制参数"""
        return Parameter(self.data.copy())


def as_parameter(arr):
    """将 numpy 数组转换为 Parameter"""
    if isinstance(arr, Parameter):
        return arr
    return Parameter(arr.copy() if isinstance(arr, np.ndarray) else Parameter(np.array(arr)))


def to_numpy(param):
    """将 Parameter 转换回 numpy 数组"""
    if isinstance(param, Parameter):
        return param.data
    return param


def zeros(shape):
    """创建全零参数"""
    return Parameter(np.zeros(shape))


def ones(shape):
    """创建全一参数"""
    return Parameter(np.ones(shape))


def random_uniform(low, high, shape):
    """创建随机均匀分布参数"""
    return Parameter(np.random.uniform(low, high, shape))


def random_normal(mean=0, std=1, shape=None):
    """创建随机正态分布参数"""
    return Parameter(np.random.normal(mean, std, shape))