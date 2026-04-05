"""
学习率调度器 - 支持按 epoch 和按 step
"""

import math
import torch
from torch.optim import Optimizer


class CosineAnnealingLR:
    """余弦退火学习率调度器 (按 epoch)"""
    
    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float = 0):
        """
        初始化余弦退火学习率调度器
        
        Args:
            optimizer: PyTorch优化器对象
            T_max: 余弦退火周期的最大步数
            eta_min: 学习率的最小值，默认为0
        """
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.base_lr = optimizer.param_groups[0]['lr']
        self.step_num = 0
    
    def step(self):
        """
        更新学习率，按照余弦退火公式计算新的学习率
        """
        self.step_num += 1
        lr = self.eta_min + (self.base_lr - self.eta_min) * \
             (1 + math.cos(math.pi * self.step_num / self.T_max)) / 2
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def state_dict(self):
        """
        获取调度器的状态字典
        
        Returns:
            包含当前步数的状态字典
        """
        return {'step_num': self.step_num}
    
    def load_state_dict(self, state_dict):
        """
        从状态字典加载调度器状态
        
        Args:
            state_dict: 包含调度器状态的字典
        """
        self.step_num = state_dict['step_num']


class CosineAnnealingWarmupLR:
    """带预热的余弦退火 (按 step，适合大模型)"""
    
    def __init__(self, optimizer: Optimizer, warmup_steps: int, total_steps: int, 
                 min_lr: float = 1e-6, initial_lr: float = 1e-7):
        """
        初始化带预热的余弦退火学习率调度器
        
        Args:
            optimizer: PyTorch优化器对象
            warmup_steps: 预热步数
            total_steps: 总训练步数
            min_lr: 最小学习率，默认为1e-6
            initial_lr: 初始学习率，默认为1e-7
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.initial_lr = initial_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.step_num = 0
    
    def step_per_batch(self):
        """
        按批次更新学习率，在预热阶段使用线性增长，在其余阶段使用余弦退火
        """
        self.step_num += 1
        
        if self.step_num < self.warmup_steps:
            # 线性预热阶段：从初始学习率线性增长到基础学习率
            lr = self.initial_lr + (self.base_lr - self.initial_lr) * (self.step_num / self.warmup_steps)
        else:
            # 余弦退火阶段：从基础学习率按余弦曲线衰减到最小学习率
            progress = (self.step_num - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress)) / 2
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def step(self):
        """按 epoch 调度 (不做任何事)"""
        pass
    
    def state_dict(self):
        """
        获取调度器的状态字典
        
        Returns:
            包含当前步数的状态字典
        """
        return {'step_num': self.step_num}
    
    def load_state_dict(self, state_dict):
        """
        从状态字典加载调度器状态
        
        Args:
            state_dict: 包含调度器状态的字典
        """
        self.step_num = state_dict['step_num']


class StepLR:
    """阶梯衰减学习率调度器"""
    
    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1):
        """
        初始化阶梯衰减学习率调度器
        
        Args:
            optimizer: PyTorch优化器对象
            step_size: 学习率衰减的步长间隔
            gamma: 学习率衰减的乘数因子，默认为0.1
        """
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.step_num = 0
    
    def step(self):
        """
        每隔step_size步将学习率乘以gamma进行衰减
        """
        self.step_num += 1
        if self.step_num % self.step_size == 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= self.gamma
    
    def state_dict(self):
        """
        获取调度器的状态字典
        
        Returns:
            包含当前步数的状态字典
        """
        return {'step_num': self.step_num}
    
    def load_state_dict(self, state_dict):
        """
        从状态字典加载调度器状态
        
        Args:
            state_dict: 包含调度器状态的字典
        """
        self.step_num = state_dict['step_num']


class LinearWarmupLR:
    """线性预热学习率调度器"""
    
    def __init__(self, optimizer: Optimizer, warmup_steps: int, target_lr: float):
        """
        初始化线性预热学习率调度器
        
        Args:
            optimizer: PyTorch优化器对象
            warmup_steps: 预热步数
            target_lr: 预热结束后的目标学习率
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.target_lr = target_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.step_num = 0
    
    def step(self):
        """
        在预热步数内线性增加学习率
        """
        self.step_num += 1
        if self.step_num <= self.warmup_steps:
            lr = self.base_lr + (self.target_lr - self.base_lr) * (self.step_num / self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
    
    def state_dict(self):
        """
        获取调度器的状态字典
        
        Returns:
            包含当前步数的状态字典
        """
        return {'step_num': self.step_num}
    
    def load_state_dict(self, state_dict):
        """
        从状态字典加载调度器状态
        
        Args:
            state_dict: 包含调度器状态的字典
        """
        self.step_num = state_dict['step_num']


class ReduceLROnPlateau:
    """当验证损失停止下降时降低学习率"""
    
    def __init__(self, optimizer: Optimizer, patience: int = 5, factor: float = 0.5, 
                 min_lr: float = 1e-7, verbose: bool = True):
        """
        初始化基于验证损失的学习率调度器
        
        Args:
            optimizer: PyTorch优化器对象
            patience: 在降低学习率之前等待的epoch数
            factor: 学习率衰减的乘数因子
            min_lr: 最小学习率阈值
            verbose: 是否打印学习率变化信息
        """
        self.optimizer = optimizer
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.verbose = verbose
        self.best_loss = float('inf')
        self.counter = 0
    
    def step(self, val_loss: float):
        """
        根据验证损失更新学习率
        
        Args:
            val_loss: 当前验证损失值
        """
        if val_loss < self.best_loss - 1e-4:
            # 验证损失有所改善，更新最佳损失并重置计数器
            self.best_loss = val_loss
            self.counter = 0
        else:
            # 验证损失未改善，增加计数器
            self.counter += 1
            if self.counter >= self.patience:
                # 计数器达到耐心值，降低学习率
                current_lr = self.optimizer.param_groups[0]['lr']
                new_lr = max(current_lr * self.factor, self.min_lr)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr
                if self.verbose:
                    print(f"📉 Reducing learning rate from {current_lr:.6f} to {new_lr:.6f}")
                self.counter = 0