"""
训练器模块
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, Dict, Any
import os


class Trainer:
    """模型训练器"""
    
    def __init__(self, model: nn.Module, dataloader: DataLoader, config, 
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 scheduler: Optional = None):
        self.model = model
        self.dataloader = dataloader
        self.config = config
        self.device = config.device if hasattr(config, 'device') else 'cpu'
        
        # 优化器
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        else:
            self.optimizer = optimizer
        
        # 学习率调度器
        self.scheduler = scheduler
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 历史记录
        self.history = {'loss': [], 'lr': []}
        self.best_loss = float('inf')
    
    def train_epoch(self) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.dataloader, desc="Training")
        for inputs, targets in pbar:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # 前向传播
            logits = self.model(inputs)
            loss = self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self, epochs: int) -> Dict[str, list]:
        """完整训练循环"""
        for epoch in range(epochs):
            # 训练一个epoch
            loss = self.train_epoch()
            
            # 更新学习率
            if self.scheduler:
                self.scheduler.step()
            
            # 记录
            self.history['loss'].append(loss)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            # 保存最佳模型
            if loss < self.best_loss:
                self.best_loss = loss
                if hasattr(self.config, 'save_path'):
                    self.save_checkpoint(self.config.save_path.replace('.pt', '_best.pt'))
            
            # 输出进度
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, LR: {self.history['lr'][-1]:.6f}")
        
        return self.history
    
    def save_checkpoint(self, path: str):
        """保存检查点"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_loss': self.best_loss
        }, path)
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        self.best_loss = checkpoint['best_loss']
        print(f"✅ Loaded checkpoint: {path}, Best loss: {self.best_loss:.4f}")