"""
Shannon-b1 改进版训练脚本
包含: 学习率调度、梯度裁剪、进度条、模型检查点、BPE分词器支持
"""

import numpy as np
import sys
import os
import argparse
import pickle
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shannon.model.shannon_b1 import ShannonB1
from shannon.training.loss import CrossEntropyLoss
from shannon.training.optimizer import Adam, AdamW, SGD
from shannon.utils.tokenizer import CharTokenizer, BPETokenizer, SimpleBPETokenizer
from data.dataset import TextDataset
from data.download import download_shakespeare, create_sample_data


class ProgressBar:
    """简单的进度条"""
    
    def __init__(self, total, width=50, desc="Progress"):
        self.total = total
        self.width = width
        self.desc = desc
        self.current = 0
        self.start_time = time.time()
    
    def update(self, n=1):
        self.current += n
        percent = self.current / self.total
        filled = int(self.width * percent)
        bar = '█' * filled + '░' * (self.width - filled)
        
        elapsed = time.time() - self.start_time
        if self.current > 0:
            eta = elapsed / self.current * (self.total - self.current)
            eta_str = f"{eta:.0f}s"
        else:
            eta_str = "?"
        
        print(f"\r{self.desc}: |{bar}| {percent*100:.1f}% [{self.current}/{self.total}] ETA: {eta_str}", end='')
        
        if self.current >= self.total:
            print()
    
    def close(self):
        print()


class CosineAnnealingLR:
    """余弦退火学习率调度器"""
    
    def __init__(self, optimizer, T_max, eta_min=0):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.base_lr = optimizer.lr
        self.t = 0
    
    def step(self):
        self.t += 1
        lr = self.eta_min + (self.base_lr - self.eta_min) * \
             (1 + np.cos(np.pi * self.t / self.T_max)) / 2
        self.optimizer.lr = lr
        return self.optimizer.lr


class StepLR:
    """阶梯衰减学习率调度器"""
    
    def __init__(self, optimizer, step_size, gamma=0.1):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.last_epoch = 0
    
    def step(self):
        self.last_epoch += 1
        if self.last_epoch % self.step_size == 0:
            self.optimizer.lr *= self.gamma
        return self.optimizer.lr


class ImprovedTrainer:
    """改进版训练器"""
    
    def __init__(self, model, optimizer, criterion, grad_clip=1.0, device='cpu'):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.grad_clip = grad_clip
        self.device = device
        self.history = {'loss': [], 'lr': [], 'time': []}
        self.best_loss = float('inf')
    
    def clip_gradients(self, params, max_norm):
        """梯度裁剪"""
        total_norm = 0.0
        for name, param in params:
            if hasattr(param, 'grad') and param.grad is not None:
                total_norm += np.sum(param.grad ** 2)
        total_norm = np.sqrt(total_norm)
        
        if total_norm > max_norm and total_norm > 0:
            clip_coef = max_norm / (total_norm + 1e-6)
            for name, param in params:
                if hasattr(param, 'grad') and param.grad is not None:
                    param.grad *= clip_coef
            return total_norm
        return 0.0
    
    def train_epoch(self, train_data, batch_size, seq_len, verbose=True):
        """训练一个epoch"""
        total_loss = 0.0
        num_batches = 0
        total_grad_norm = 0.0
        
        # 打乱数据
        indices = np.random.permutation(len(train_data))
        num_batches_total = len(indices) // batch_size
        
        if verbose:
            pbar = ProgressBar(num_batches_total, desc="  Training")
        
        for i in range(0, len(indices) - batch_size, batch_size):
            batch_indices = indices[i:i+batch_size]
            batch = [train_data[idx] for idx in batch_indices]
            
            # 准备批次
            max_len = min(max(len(seq) for seq in batch), seq_len)
            inputs, targets = self._prepare_batch(batch, max_len)
            
            # 前向传播
            logits = self.model.forward(inputs, training=True)
            loss = self.criterion.forward(logits, targets)
            
            # 反向传播
            d_logits = self.criterion.backward()
            self.model.backward(d_logits)
            
            # 梯度裁剪
            params = self.model.get_all_parameters()
            grad_norm = self.clip_gradients(params, self.grad_clip)
            total_grad_norm += grad_norm
            
            # 更新参数
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            total_loss += loss
            num_batches += 1
            
            if verbose:
                pbar.update(1)
        
        if verbose:
            pbar.close()
        
        avg_loss = total_loss / num_batches
        avg_grad_norm = total_grad_norm / num_batches
        
        return avg_loss, avg_grad_norm
    
    def _prepare_batch(self, batch, max_len):
        """准备批次数据"""
        inputs = []
        targets = []
        for seq in batch:
            if len(seq) > max_len + 1:
                seq = seq[:max_len + 1]
            input_seq = seq[:-1] if len(seq) > 1 else seq
            target_seq = seq[1:] if len(seq) > 1 else seq
            input_seq = input_seq + [0] * (max_len - len(input_seq))
            target_seq = target_seq + [0] * (max_len - len(target_seq))
            inputs.append(input_seq)
            targets.append(target_seq)
        return np.array(inputs, dtype=np.int32), np.array(targets, dtype=np.int32)
    
    def train(self, train_data, epochs, batch_size, seq_len, 
              lr_scheduler=None, save_path=None, save_every=10, verbose=True):
        """完整训练循环"""
        print("\n" + "=" * 60)
        print("开始训练")
        print("=" * 60)
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # 训练一个epoch
            avg_loss, avg_grad_norm = self.train_epoch(
                train_data, batch_size, seq_len, 
                verbose=(verbose and epoch % 5 == 0)
            )
            
            # 记录历史
            self.history['loss'].append(avg_loss)
            self.history['lr'].append(self.optimizer.lr)
            self.history['time'].append(time.time() - epoch_start)
            
            # 更新学习率
            if lr_scheduler is not None:
                current_lr = lr_scheduler.step()
            else:
                current_lr = self.optimizer.lr
            
            # 保存最佳模型
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                if save_path:
                    self.save_checkpoint(save_path.replace('.pkl', '_best.pkl'))
            
            # 定期保存检查点
            if save_path and (epoch + 1) % save_every == 0:
                self.save_checkpoint(save_path.replace('.pkl', f'_epoch{epoch+1}.pkl'))
            
            # 输出进度
            if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
                print(f"\n📊 Epoch {epoch+1}/{epochs}")
                print(f"   Loss: {avg_loss:.4f} (Best: {self.best_loss:.4f})")
                print(f"   LR: {current_lr:.6f}")
                print(f"   Grad Norm: {avg_grad_norm:.4f}")
                print(f"   Time: {self.history['time'][-1]:.1f}s")
                print("-" * 60)
        
        return self.history
    
    def save_checkpoint(self, path):
        """保存检查点"""
        params = self.model.get_all_parameters()
        checkpoint = {
            'model_weights': {name: p.data for name, p in params},
            'optimizer_state': self.optimizer.get_state() if hasattr(self.optimizer, 'get_state') else None,
            'history': self.history,
            'best_loss': self.best_loss
        }
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
        print(f"   💾 检查点已保存: {path}")
    
    def load_checkpoint(self, path):
        """加载检查点"""
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        params = self.model.get_all_parameters()
        for name, param in params:
            if name in checkpoint['model_weights']:
                param.data = checkpoint['model_weights'][name]
        
        if checkpoint['optimizer_state'] and hasattr(self.optimizer, 'load_state'):
            self.optimizer.load_state(checkpoint['optimizer_state'])
        
        self.history = checkpoint.get('history', self.history)
        self.best_loss = checkpoint.get('best_loss', self.best_loss)
        
        print(f"✅ 加载检查点: {path}")
        print(f"   最佳损失: {self.best_loss:.4f}")


def create_tokenizer(args, texts=None):
    """创建分词器"""
    if args.tokenizer == 'bpe':
        tokenizer = BPETokenizer(vocab_size=args.vocab_size)
        if texts:
            tokenizer.train(texts, verbose=True)
        return tokenizer
    elif args.tokenizer == 'simple_bpe':
        tokenizer = SimpleBPETokenizer(vocab_size=args.vocab_size)
        if texts:
            tokenizer.build_vocab(texts)
        return tokenizer
    else:
        # 默认使用字符级分词器
        tokenizer = CharTokenizer()
        if texts:
            tokenizer.build_vocab(texts, args.vocab_size)
        return tokenizer


def load_text_data(data_path, tokenizer, seq_len, max_samples=None):
    """加载文本数据"""
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # 限制数据量以便快速测试
    if max_samples:
        text = text[:max_samples * seq_len]
    
    dataset = TextDataset(tokenizer, seq_len)
    dataset.load_text(text, os.path.basename(data_path))
    
    return dataset.get_all_sequences()


def create_training_data(args, tokenizer):
    """创建训练数据"""
    print("\n📚 准备训练数据...")
    
    if args.dataset == 'shakespeare':
        data_path = download_shakespeare()
        if data_path and os.path.exists(data_path):
            train_data = load_text_data(data_path, tokenizer, args.seq_len, args.max_samples)
        else:
            print("⚠️ 莎士比亚数据下载失败，使用示例数据")
            data_path = create_sample_data()
            train_data = load_text_data(data_path, tokenizer, args.seq_len, args.max_samples)
    
    elif args.dataset == 'sample':
        data_path = create_sample_data()
        train_data = load_text_data(data_path, tokenizer, args.seq_len, args.max_samples)
    
    elif args.dataset == 'file' and args.data_path:
        if os.path.exists(args.data_path):
            train_data = load_text_data(args.data_path, tokenizer, args.seq_len, args.max_samples)
        else:
            raise FileNotFoundError(f"文件不存在: {args.data_path}")
    
    else:
        # 随机数据
        print("⚠️ 使用随机数据训练")
        train_data = [np.random.randint(0, tokenizer.get_vocab_size(), size=args.seq_len).tolist() 
                      for _ in range(args.train_size)]
    
    print(f"   ✅ 训练样本数: {len(train_data):,}")
    return train_data


def generate_samples(model, tokenizer, prompts, max_tokens=50, temperature=0.8):
    """生成文本样本"""
    print("\n" + "=" * 60)
    print("🎨 文本生成测试")
    print("=" * 60)
    
    for prompt in prompts:
        start_tokens = tokenizer.encode(prompt)
        if len(start_tokens) > 10:
            start_tokens = start_tokens[:10]
        
        generated = model.generate(start_tokens, max_tokens, temperature)
        text = tokenizer.decode(generated)
        
        print(f"\n📝 提示: {prompt}")
        print(f"💬 生成: {text[:150]}...")


def main():
    parser = argparse.ArgumentParser(description='Shannon-b1 训练')
    
    # 模型配置
    parser.add_argument('--d-model', type=int, default=128, help='模型维度')
    parser.add_argument('--num-heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--d-ff', type=int, default=512, help='FFN维度')
    parser.add_argument('--num-layers', type=int, default=4, help='Transformer层数')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout率')
    
    # 训练配置
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    parser.add_argument('--seq-len', type=int, default=64, help='序列长度')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'adam', 'adamw'])
    parser.add_argument('--weight-decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='梯度裁剪阈值')
    
    # 学习率调度
    parser.add_argument('--lr-scheduler', type=str, default='cosine', 
                        choices=['none', 'cosine', 'step'], help='学习率调度器')
    parser.add_argument('--lr-step-size', type=int, default=30, help='阶梯衰减步长')
    parser.add_argument('--lr-gamma', type=float, default=0.5, help='衰减系数')
    
    # 数据配置
    parser.add_argument('--dataset', type=str, default='shakespeare',
                        choices=['shakespeare', 'sample', 'random', 'file'])
    parser.add_argument('--data-path', type=str, default=None)
    parser.add_argument('--tokenizer', type=str, default='char',
                        choices=['char', 'bpe', 'simple_bpe'], help='分词器类型')
    parser.add_argument('--vocab-size', type=int, default=1000, help='词表大小')
    parser.add_argument('--max-samples', type=int, default=None, help='最大样本数(测试用)')
    parser.add_argument('--train-size', type=int, default=500, help='随机数据大小')
    
    # 保存配置
    parser.add_argument('--save-path', type=str, default='shannon_b1_improved.pkl')
    parser.add_argument('--load-path', type=str, default=None)
    parser.add_argument('--save-every', type=int, default=10, help='保存间隔')
    
    # 其他
    parser.add_argument('--eval-interval', type=int, default=20, help='评估间隔')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    
    print("=" * 60)
    print("Shannon-b1 改进版训练")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 打印配置
    print("\n📋 配置:")
    print(f"   模型: d_model={args.d_model}, num_heads={args.num_heads}, "
          f"num_layers={args.num_layers}")
    print(f"   训练: epochs={args.epochs}, batch_size={args.batch_size}, "
          f"seq_len={args.seq_len}, lr={args.lr}")
    print(f"   优化: optimizer={args.optimizer}, grad_clip={args.grad_clip}, "
          f"weight_decay={args.weight_decay}")
    print(f"   数据: dataset={args.dataset}, tokenizer={args.tokenizer}, vocab_size={args.vocab_size}")
    
    # 创建分词器 (先加载文本用于训练)
    if args.dataset == 'shakespeare':
        data_path = download_shakespeare()
        if data_path:
            with open(data_path, 'r', encoding='utf-8') as f:
                sample_text = f.read()[:10000]  # 取前10000字符用于构建词表
            texts = [sample_text]
        else:
            texts = ["Hello world! This is a sample text for building vocabulary."]
    elif args.dataset == 'sample':
        data_path = create_sample_data()
        with open(data_path, 'r', encoding='utf-8') as f:
            sample_text = f.read()
        texts = [sample_text]
    else:
        texts = ["Hello world! This is a sample text for building vocabulary."]
    
    tokenizer = create_tokenizer(args, texts)
    print(f"\n📝 词表大小: {tokenizer.get_vocab_size()}")
    
    # 创建训练数据
    train_data = create_training_data(args, tokenizer)
    
    # 更新词表大小
    actual_vocab_size = tokenizer.get_vocab_size()
    
    # 创建模型
    print("\n🏗️ 创建模型...")
    model = ShannonB1(
        vocab_size=actual_vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        max_seq_len=args.seq_len,
        dropout=args.dropout
    )
    
    # 统计参数量
    params = model.get_all_parameters()
    total_params = sum(p.data.size for _, p in params)
    print(f"   参数总量: {total_params:,}")
    print(f"   模型大小: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    # 创建优化器
    if args.optimizer == 'sgd':
        optimizer = SGD(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    
    # 加载检查点
    if args.load_path and os.path.exists(args.load_path):
        print(f"\n📂 加载检查点: {args.load_path}")
        # 这里需要实现加载逻辑
    
    # 创建学习率调度器
    lr_scheduler = None
    if args.lr_scheduler == 'cosine':
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
        print(f"\n📉 使用余弦退火学习率调度")
    elif args.lr_scheduler == 'step':
        lr_scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
        print(f"\n📉 使用阶梯衰减学习率调度")
    else:
        print(f"\n📉 使用固定学习率")
    
    # 创建训练器
    criterion = CrossEntropyLoss()
    trainer = ImprovedTrainer(model, optimizer, criterion, grad_clip=args.grad_clip)
    
    # 训练
    start_time = time.time()
    history = trainer.train(
        train_data, 
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr_scheduler=lr_scheduler,
        save_path=args.save_path,
        save_every=args.save_every,
        verbose=True
    )
    total_time = time.time() - start_time
    
    # 训练总结
    print("\n" + "=" * 60)
    print("✅ 训练完成！")
    print("=" * 60)
    print(f"   总时间: {total_time / 60:.1f} 分钟")
    print(f"   初始损失: {history['loss'][0]:.4f}")
    print(f"   最终损失: {history['loss'][-1]:.4f}")
    print(f"   最佳损失: {trainer.best_loss:.4f}")
    print(f"   损失下降: {history['loss'][0] - history['loss'][-1]:.4f}")
    
    # 生成测试样本
    test_prompts = ["The ", "Once upon ", "In the beginning ", "To be or "]
    generate_samples(model, tokenizer, test_prompts, max_tokens=80, temperature=0.7)
    
    # 保存最终模型
    final_weights = {name: p.data for name, p in params}
    with open(args.save_path, 'wb') as f:
        pickle.dump(final_weights, f)
    print(f"\n💾 最终模型已保存: {args.save_path}")
    
    # 保存分词器
    tokenizer.save(args.save_path.replace('.pkl', '_tokenizer.json'))
    
    return history


if __name__ == "__main__":
    history = main()