"""
学习率调度器
"""

import math
import torch
from torch.optim import Optimizer


class CosineAnnealingLR:
    """余弦退火学习率调度器"""
    
    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float = 0):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.base_lr = optimizer.param_groups[0]['lr']
        self.step_num = 0
    
    def step(self):
        self.step_num += 1
        lr = self.eta_min + (self.base_lr - self.eta_min) * \
             (1 + math.cos(math.pi * self.step_num / self.T_max)) / 2
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


class StepLR:
    """阶梯衰减学习率调度器"""
    
    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.step_num = 0
    
    def step(self):
        self.step_num += 1
        if self.step_num % self.step_size == 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= self.gamma


class LinearWarmupLR:
    """线性预热学习率调度器"""
    
    def __init__(self, optimizer: Optimizer, warmup_steps: int, target_lr: float):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.target_lr = target_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.step_num = 0
    
    def step(self):
        self.step_num += 1
        if self.step_num <= self.warmup_steps:
            lr = self.base_lr + (self.target_lr - self.base_lr) * (self.step_num / self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr