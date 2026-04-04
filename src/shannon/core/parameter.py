"""
可训练参数包装类 - 优化版本
"""

import numpy as np


class Parameter:
    """可训练参数，包装 numpy 数组并添加梯度"""
    
    __slots__ = ('data', 'grad', 'name')
    
    def __init__(self, data, name=None):
        if isinstance(data, np.ndarray):
            if data.dtype == np.float64:
                self.data = data.astype(np.float32)
            else:
                self.data = data
        else:
            self.data = np.array(data, dtype=np.float32)
        self.grad = None
        self.name = name
    
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
    
    def __neg__(self):
        return Parameter(-self.data)
    
    def __pos__(self):
        return Parameter(self.data)
    
    def __repr__(self):
        return f"Parameter(shape={self.shape}, dtype={self.dtype})"
    
    def numpy(self):
        return self.data
    
    def zero_grad(self):
        self.grad = None
    
    def copy(self):
        return Parameter(self.data.copy())
    
    def to(self, dtype):
        self.data = self.data.astype(dtype)
        if self.grad is not None:
            self.grad = self.grad.astype(dtype)
        return self


def as_parameter(arr):
    if isinstance(arr, Parameter):
        return arr
    return Parameter(arr.copy() if isinstance(arr, np.ndarray) else np.array(arr))


def to_numpy(param):
    if isinstance(param, Parameter):
        return param.data
    return param


def zeros(shape):
    return Parameter(np.zeros(shape, dtype=np.float32))


def ones(shape):
    return Parameter(np.ones(shape, dtype=np.float32))


def random_uniform(low, high, shape):
    return Parameter(np.random.uniform(low, high, shape).astype(np.float32))


def random_normal(mean, std, shape):
    return Parameter(np.random.normal(mean, std, shape).astype(np.float32))


def xavier_init(shape_in, shape_out):
    limit = np.sqrt(6.0 / (shape_in + shape_out))
    return Parameter(np.random.uniform(-limit, limit, (shape_in, shape_out)).astype(np.float32))


def he_init(shape_in, shape_out):
    std = np.sqrt(2.0 / shape_in)
    return Parameter(np.random.normal(0, std, (shape_in, shape_out)).astype(np.float32))