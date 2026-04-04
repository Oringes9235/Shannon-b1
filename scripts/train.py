#!/usr/bin/env python
"""
Shannon-b1 PyTorch 训练脚本
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from datetime import datetime

from src.model import ShannonB1, ModelConfig, TrainingConfig
from src.data import TextDataset, create_tokenizer, load_shakespeare
from src.training import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description='Shannon-b1 Training')
    
    # 模型参数
    parser.add_argument('--d-model', type=int, default=128, help='Model dimension')
    parser.add_argument('--num-heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num-layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--d-ff', type=int, default=512, help='FFN dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--seq-len', type=int, default=64, help='Sequence length')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    # 数据参数
    parser.add_argument('--tokenizer', type=str, default='char', choices=['char', 'bpe', 'simple_bpe'])
    parser.add_argument('--vocab-size', type=int, default=1000, help='Vocabulary size')
    
    # 其他
    parser.add_argument('--save-path', type=str, default='checkpoints/shannon_b1.pt')
    parser.add_argument('--seed', type=int, default=42)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    
    print("=" * 60)
    print("Shannon-b1 PyTorch Training")
    print(f"Device: {args.device}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 加载数据
    print("\n📚 Loading data...")
    text = load_shakespeare()
    tokenizer = create_tokenizer(text, args.tokenizer, args.vocab_size)
    
    # 创建数据集
    dataset = TextDataset([text], tokenizer, args.seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    # 创建模型配置
    model_config = ModelConfig(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        max_seq_len=args.seq_len,
        dropout=args.dropout,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seq_len=args.seq_len,
        device=args.device
    )
    
    print(f"\n📝 Model config:")
    print(f"   Vocab size: {model_config.vocab_size}")
    print(f"   Parameters: {model_config.d_model * model_config.num_layers * 4:,}")
    
    # 创建模型
    model = ShannonB1(model_config).to(args.device)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    # 创建训练器
    trainer = Trainer(model, dataloader, model_config)
    
    # 训练
    print("\n🚀 Starting training...")
    print("-" * 60)
    
    history = trainer.train(args.epochs)
    
    print("-" * 60)
    print(f"\n✅ Training completed!")
    print(f"   Initial loss: {history['loss'][0]:.4f}")
    print(f"   Final loss: {history['loss'][-1]:.4f}")
    
    # 保存模型
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model_config,
        'tokenizer_type': args.tokenizer,
        'vocab_size': tokenizer.get_vocab_size()
    }, args.save_path)
    print(f"\n💾 Model saved: {args.save_path}")
    
    # 保存分词器
    tokenizer.save(args.save_path.replace('.pt', '_tokenizer.json'))


if __name__ == "__main__":
    main()