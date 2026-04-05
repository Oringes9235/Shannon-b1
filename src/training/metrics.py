"""
评估指标
"""

import torch
import math


def compute_perplexity(loss: float) -> float:
    """
    计算困惑度
    
    Args:
        loss (float): 损失值
        
    Returns:
        float: 困惑度值
    """
    return math.exp(loss)


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    计算准确率
    
    Args:
        logits (torch.Tensor): 模型输出的logits张量
        targets (torch.Tensor): 目标标签张量
        
    Returns:
        float: 准确率值
    """
    predictions = logits.argmax(dim=-1)
    correct = (predictions == targets).sum().item()
    total = targets.numel()
    return correct / total


def compute_top_k_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
    """
    计算 Top-K 准确率
    
    Args:
        logits (torch.Tensor): 模型输出的logits张量
        targets (torch.Tensor): 目标标签张量
        k (int): Top-K中的K值，默认为5
        
    Returns:
        float: Top-K准确率值
    """
    # 获取top-k预测结果
    top_k = logits.topk(k, dim=-1).indices
    correct = 0
    total = targets.numel()
    
    # 遍历每个样本和序列位置，检查目标是否在top-k预测中
    for i in range(targets.shape[0]):
        for j in range(targets.shape[1]):
            if targets[i, j] in top_k[i, j]:
                correct += 1
    
    return correct / total