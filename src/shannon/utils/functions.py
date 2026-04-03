import numpy as np
from ..core.parameter import Parameter, as_parameter, to_numpy


def softmax(x, axis=-1):
    """数值稳定的softmax"""
    # 处理 Parameter 类型
    if isinstance(x, Parameter):
        x = x.data
    
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    result = e_x / np.sum(e_x, axis=axis, keepdims=True)
    
    # 如果输入是 Parameter，返回 Parameter
    if isinstance(x, Parameter):
        return Parameter(result)
    return result


def gelu(x):
    """GELU激活函数"""
    if isinstance(x, Parameter):
        x = x.data
    
    result = 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    
    if isinstance(x, Parameter):
        return Parameter(result)
    return result


def gelu_backward(x):
    """GELU的反向传播"""
    if isinstance(x, Parameter):
        x = x.data
    
    tanh_arg = np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)
    tanh_val = np.tanh(tanh_arg)
    result = 0.5 * (1 + tanh_val) + 0.5 * x * (1 - tanh_val**2) * np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * x**2)
    
    if isinstance(x, Parameter):
        return Parameter(result)
    return result


def xavier_init(shape_in, shape_out):
    """Xavier初始化，返回 Parameter"""
    limit = np.sqrt(6.0 / (shape_in + shape_out))
    data = np.random.uniform(-limit, limit, (shape_in, shape_out))
    return Parameter(data)


def zeros(shape):
    """创建零参数"""
    return Parameter(np.zeros(shape))


def ones(shape):
    """创建一参数"""
    return Parameter(np.ones(shape))


def random_uniform(low, high, shape):
    """创建随机均匀分布参数"""
    return Parameter(np.random.uniform(low, high, shape))