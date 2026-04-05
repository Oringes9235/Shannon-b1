"""
训练器模块 - 包含混合精度、梯度累积、早停、TensorBoard
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import amp
from tqdm import tqdm
import os
import time
from datetime import datetime
from typing import Optional, Dict, Any
import numpy as np
import torch.nn.functional as F

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("⚠️ TensorBoard not available. Install with: pip install tensorboard")


class ImprovedTrainer:
    """改进版训练器 - 支持混合精度、梯度累积、早停"""
    
    def __init__(self, model: nn.Module, dataloader: DataLoader, 
                 val_dataloader: Optional[DataLoader] = None,
                 config=None, optimizer: Optional[torch.optim.Optimizer] = None,
                 scheduler: Optional = None):
        self.model = model
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
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
        
        # 损失函数（若使用 label smoothing，将使用自定义计算）
        self.criterion = None
        
        # 混合精度训练
        self.use_amp = config.use_amp and self.device == 'cuda'
        # GradScaler 初始化兼容不同 PyTorch 版本
        if self.use_amp:
            try:
                self.scaler = amp.GradScaler(device_type='cuda')
            except TypeError:
                # 较旧/不同版本可能不接受 device_type 参数
                self.scaler = amp.GradScaler()
        else:
            self.scaler = None

        # 梯度累积
        self.grad_accum_steps = config.gradient_accumulation_steps

        # 历史记录
        self.history = {
            'train_loss': [], 
            'val_loss': [], 
            'lr': [],
            'time': [],
            'epoch_time': []
        }
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0

        # TensorBoard
        self.writer = None
        if TENSORBOARD_AVAILABLE and hasattr(config, 'tensorboard_dir'):
            self.writer = SummaryWriter(config.tensorboard_dir)

        # 统计信息
        self.global_step = 0
        self.start_time = None

    def _autocast(self):
        """Return a compatible autocast context manager across PyTorch versions."""
        if not self.use_amp:
            # no autocast when not using amp
            from contextlib import nullcontext
            return nullcontext()

        try:
            # try no-arg autocast (newer versions)
            return amp.autocast()
        except TypeError:
            # fall back to specifying device_type
            device_type = 'cuda' if self.device == 'cuda' else 'cpu'
            return amp.autocast(device_type=device_type)
    
    def train_epoch(self) -> float:
        """训练一个epoch (支持混合精度和梯度累积)"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # 进度条
        pbar = tqdm(self.dataloader, desc="Training")
        self.optimizer.zero_grad()
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # 前向传播并计算损失（支持 label smoothing）
            if self.use_amp:
                with self._autocast():
                    logits = self.model(inputs)
                    loss = self._compute_loss(logits, targets)
                    loss = loss / self.grad_accum_steps

                # 反向传播 (混合精度)
                self.scaler.scale(loss).backward()
            else:
                logits = self.model(inputs)
                loss = self._compute_loss(logits, targets)
                loss = loss / self.grad_accum_steps
                loss.backward()
            
            total_loss += loss.item() * self.grad_accum_steps
            num_batches += 1
            
            # 梯度累积更新
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                # 梯度裁剪
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # 更新学习率调度器 (按步数)
                if self.scheduler and hasattr(self.scheduler, 'step_per_batch'):
                    self.scheduler.step_per_batch()
            
            # 更新进度条
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({'loss': loss.item() * self.grad_accum_steps, 'lr': f'{current_lr:.2e}'})
            
            # TensorBoard 记录 (每 N 步)
            if self.writer and self.global_step % self.config.log_interval == 0:
                self.writer.add_scalar('train/loss_step', loss.item() * self.grad_accum_steps, self.global_step)
                self.writer.add_scalar('train/lr', current_lr, self.global_step)
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    @torch.no_grad()
    def validate(self) -> float:
        """验证"""
        if self.val_dataloader is None:
            return 0.0
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.val_dataloader, desc="Validating")
        for inputs, targets in pbar:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            logits = self.model(inputs)
            loss = self._compute_loss(logits, targets)
            
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'val_loss': loss.item()})
        
        return total_loss / num_batches

    def _compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算损失，支持 label smoothing（基于交叉熵的平滑实现）。

        logits: (batch, seq_len, vocab)
        targets: (batch, seq_len)
        返回: 标量 loss
        """
        smoothing = getattr(self.config, 'label_smoothing', 0.0)
        vocab_size = logits.size(-1)

        if smoothing is None or smoothing <= 0.0:
            # 标准交叉熵
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1), reduction='mean')
            return loss

        # label smoothing 实现（参考 HuggingFace/Transformer 常用实现）
        log_probs = F.log_softmax(logits, dim=-1)  # (B, L, V)
        nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)  # (B, L)
        smooth_loss = -log_probs.mean(dim=-1)  # (B, L)

        loss = (1.0 - smoothing) * nll_loss + smoothing * smooth_loss
        return loss.mean()
    
    def should_early_stop(self, val_loss: float) -> bool:
        """检查是否应该早停"""
        if val_loss < self.best_val_loss - self.config.early_stopping_min_delta:
            self.best_val_loss = val_loss
            self.best_epoch = len(self.history['val_loss']) - 1
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.config.early_stopping_patience
    
    def train(self, epochs: int) -> Dict[str, list]:
        """完整训练循环"""
        print("\n" + "=" * 70)
        print("🚀 Starting Training")
        print(f"   Device: {self.device.upper()}")
        print(f"   Mixed Precision: {'ON' if self.use_amp else 'OFF'}")
        print(f"   Gradient Accumulation: {self.grad_accum_steps}")
        print(f"   Batch Size (effective): {self.config.batch_size * self.grad_accum_steps}")
        print("=" * 70)
        
        self.start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # 训练
            train_loss = self.train_epoch()
            
            # 验证
            val_loss = self.validate()
            
            # 更新学习率调度器 (按 epoch)
            if self.scheduler and hasattr(self.scheduler, 'step'):
                self.scheduler.step()
            
            epoch_time = time.time() - epoch_start
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss if val_loss > 0 else train_loss)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            self.history['time'].append(epoch_time)
            self.history['epoch_time'].append(epoch_time)
            
            # TensorBoard 记录 (每 epoch)
            if self.writer:
                self.writer.add_scalar('train/loss_epoch', train_loss, epoch)
                self.writer.add_scalar('val/loss', val_loss, epoch)
                self.writer.add_scalar('train/lr_epoch', self.optimizer.param_groups[0]['lr'], epoch)
            
            # 输出进度
            print(f"\n📊 Epoch {epoch+1}/{epochs}")
            print(f"   Train Loss: {train_loss:.4f}")
            if val_loss > 0:
                print(f"   Val Loss: {val_loss:.4f}")
            print(f"   LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            print(f"   Time: {epoch_time:.1f}s")
            print(f"   Best Val Loss: {self.best_val_loss:.4f} (Epoch {self.best_epoch+1})")
            
            # 保存最佳模型
            if val_loss > 0 and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.save_checkpoint('checkpoints/shannon_b1_best.pt')
                print(f"   💾 Saved best model (val_loss: {val_loss:.4f})")
            
            # 定期保存检查点
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(f'checkpoints/shannon_b1_epoch{epoch+1}.pt')
            
            # 早停检查
            if val_loss > 0 and self.should_early_stop(val_loss):
                print(f"\n🛑 Early stopping triggered after {epoch+1} epochs")
                print(f"   Best val loss: {self.best_val_loss:.4f} at epoch {self.best_epoch+1}")
                break
            
            print("-" * 70)
        
        total_time = time.time() - self.start_time
        print(f"\n✅ Training completed!")
        print(f"   Total time: {total_time / 60:.1f} minutes")
        print(f"   Best val loss: {self.best_val_loss:.4f}")
        
        # 关闭 TensorBoard writer
        if self.writer:
            self.writer.close()
        
        return self.history
    
    def save_checkpoint(self, path: str):
        """保存检查点"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'config': self.config,
            'global_step': self.global_step
        }
        torch.save(checkpoint, path)
        print(f"   💾 Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        # 尝试显式允许加载包含自定义类的完整检查点（非 weights-only）
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            # 兼容较旧的 PyTorch 版本或不支持 weights_only 参数的环境
            checkpoint = torch.load(path, map_location=self.device)
        except Exception as e:
            # 如果受限于 safe globals，提醒用户并重抛
            print(f"⚠️ Failed to load checkpoint with weights_only=False: {e}")
            raise
        ckpt_state = checkpoint.get('model_state_dict', {})

        # 尝试按形状安全地加载模型权重：仅替换形状匹配的参数
        model_state = self.model.state_dict()
        matched_keys = []
        skipped_keys = []

        for k, v in ckpt_state.items():
            if k in model_state and v.size() == model_state[k].size():
                model_state[k] = v
                matched_keys.append(k)
            else:
                skipped_keys.append(k)

        # 用更新后的 state_dict 加载（包含被跳过的原始参数）
        self.model.load_state_dict(model_state)
        print(f"✅ Loaded model weights: {len(matched_keys)} params matched, {len(skipped_keys)} skipped")

        # 如果存在被跳过的参数（形状不匹配），不要加载 optimizer/scheduler/scaler 状态
        if skipped_keys:
            print("⚠️ Some parameters were skipped due to shape mismatch; skipping optimizer/scheduler/scaler state load to avoid errors.")
        else:
            # 尝试加载优化器和调度器状态，如果不兼容则跳过并提示
            try:
                if 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] is not None:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print("✅ Optimizer state loaded")
            except Exception as e:
                print(f"⚠️ Could not load optimizer state (skipped): {e}")

            try:
                if self.scheduler and checkpoint.get('scheduler_state_dict'):
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    print("✅ Scheduler state loaded")
            except Exception as e:
                print(f"⚠️ Could not load scheduler state (skipped): {e}")

            try:
                if self.scaler and checkpoint.get('scaler_state_dict'):
                    self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                    print("✅ AMP scaler state loaded")
            except Exception as e:
                print(f"⚠️ Could not load scaler state (skipped): {e}")
        
        self.history = checkpoint.get('history', self.history)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_epoch = checkpoint.get('best_epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        
        print(f"✅ Loaded checkpoint: {path}")
        print(f"   Best val loss: {self.best_val_loss:.4f} (Epoch {self.best_epoch+1})")