"""
工具函数
"""

import random
import numpy as np
import torch
from datetime import datetime


def set_seed(seed: int = 42):
    """
    设置随机种子以确保实验可复现性
    
    Args:
        seed (int): 随机种子值，默认为42
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # 仅在 CUDA 可用时设置 CUDA 相关种子与 cudnn 选项
    try:
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # 保持确定性以便复现（可能降低性能），如需性能可在命令行关闭
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def get_device() -> str:
    """
    获取当前可用的计算设备
    
    Returns:
        str: 可用设备名称，优先级为 cuda > mps > cpu
    """
    if torch.cuda.is_available():
        return "cuda"
    # 有些 PyTorch 版本可能不包含 mps backend，检查属性再调用
    if hasattr(torch.backends, 'mps') and callable(getattr(torch.backends.mps, 'is_available', None)):
        try:
            if torch.backends.mps.is_available():
                return 'mps'
        except Exception:
            pass
    return "cpu"


def format_time(seconds: float) -> str:
    """
    将秒数格式化为 HH:MM:SS 格式的时间字符串
    
    Args:
        seconds (float): 秒数
        
    Returns:
        str: 格式化后的时间字符串，格式为 HH:MM:SS
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def count_parameters(model: torch.nn.Module) -> int:
    """
    统计PyTorch模型中需要梯度更新的参数总量
    
    Args:
        model (torch.nn.Module): PyTorch模型
        
    Returns:
        int: 模型参数总数
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)