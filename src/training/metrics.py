"""
评估指标
"""

import torch
import math


def compute_perplexity(loss: float) -> float:
    """计算困惑度"""
    return math.exp(loss)


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """计算准确率"""
    predictions = logits.argmax(dim=-1)
    correct = (predictions == targets).sum().item()
    total = targets.numel()
    return correct / total


def compute_top_k_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
    """计算 Top-K 准确率"""
    top_k = logits.topk(k, dim=-1).indices
    correct = 0
    total = targets.numel()
    
    for i in range(targets.shape[0]):
        for j in range(targets.shape[1]):
            if targets[i, j] in top_k[i, j]:
                correct += 1
    
    return correct / total