"""
学习率调度器 - 支持按 epoch 和按 step
"""

import math
import torch
from torch.optim import Optimizer


class CosineAnnealingLR:
    """余弦退火学习率调度器 (按 epoch)"""
    
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
    
    def state_dict(self):
        return {'step_num': self.step_num}
    
    def load_state_dict(self, state_dict):
        self.step_num = state_dict['step_num']


class CosineAnnealingWarmupLR:
    """带预热的余弦退火 (按 step，适合大模型)"""
    
    def __init__(self, optimizer: Optimizer, warmup_steps: int, total_steps: int, 
                 min_lr: float = 1e-6, initial_lr: float = 1e-7):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.initial_lr = initial_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.step_num = 0
    
    def step_per_batch(self):
        self.step_num += 1
        
        if self.step_num < self.warmup_steps:
            # 线性预热
            lr = self.initial_lr + (self.base_lr - self.initial_lr) * (self.step_num / self.warmup_steps)
        else:
            # 余弦退火
            progress = (self.step_num - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress)) / 2
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def step(self):
        """按 epoch 调度 (不做任何事)"""
        pass
    
    def state_dict(self):
        return {'step_num': self.step_num}
    
    def load_state_dict(self, state_dict):
        self.step_num = state_dict['step_num']


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
    
    def state_dict(self):
        return {'step_num': self.step_num}
    
    def load_state_dict(self, state_dict):
        self.step_num = state_dict['step_num']


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
    
    def state_dict(self):
        return {'step_num': self.step_num}
    
    def load_state_dict(self, state_dict):
        self.step_num = state_dict['step_num']


class ReduceLROnPlateau:
    """当验证损失停止下降时降低学习率"""
    
    def __init__(self, optimizer: Optimizer, patience: int = 5, factor: float = 0.5, 
                 min_lr: float = 1e-7, verbose: bool = True):
        self.optimizer = optimizer
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.verbose = verbose
        self.best_loss = float('inf')
        self.counter = 0
    
    def step(self, val_loss: float):
        if val_loss < self.best_loss - 1e-4:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                current_lr = self.optimizer.param_groups[0]['lr']
                new_lr = max(current_lr * self.factor, self.min_lr)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr
                if self.verbose:
                    print(f"📉 Reducing learning rate from {current_lr:.6f} to {new_lr:.6f}")
                self.counter = 0