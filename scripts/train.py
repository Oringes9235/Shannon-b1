#!/usr/bin/env python
"""
Shannon-b1 PyTorch 训练脚本 - 完整改进版
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, random_split
import argparse
from datetime import datetime

from src.model import ShannonB1, ModelConfig
from src.data import TextDataset, create_tokenizer, load_shakespeare
from src.training import ImprovedTrainer, CosineAnnealingWarmupLR
from src.utils import set_seed, get_device


def parse_args():
    """
    解析命令行参数并返回参数对象
    
    Returns:
        argparse.Namespace: 包含所有解析后的命令行参数的对象
    """
    parser = argparse.ArgumentParser(description='Shannon-b1 Training')
    
    # 模型架构参数
    parser.add_argument('--d-model', type=int, default=128)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--d-ff', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seq-len', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--grad-accum', type=int, default=1)
    
    # 高级训练选项
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--warmup-steps', type=int, default=1000)
    parser.add_argument('--gradient-checkpointing', action='store_true', help='Enable gradient checkpointing to save memory')
    parser.add_argument('--norm-type', type=str, default='layernorm', choices=['layernorm', 'rmsnorm'], help='Normalization type')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--label-smoothing', type=float, default=0.0)
    parser.add_argument('--tie-embeddings', action='store_true', help='Tie token embedding and output projection')
    parser.add_argument('--patience', type=int, default=10)
    
    # 分词器参数
    parser.add_argument('--tokenizer', type=str, default='char', choices=['char', 'bpe'])
    parser.add_argument('--vocab-size', type=int, default=2000)
    
    # 系统参数
    parser.add_argument('--device', type=str, default=get_device())
    parser.add_argument('--save-path', type=str, default='checkpoints/shannon_b1.pt')
    parser.add_argument('--seed', type=int, default=42)
    
    return parser.parse_args()


def main():
    """
    主训练函数：初始化模型、数据加载器、优化器，并执行完整的训练循环
    """
    args = parse_args()
    set_seed(args.seed)
    
    print("=" * 70)
    print("Shannon-b1 Improved Training")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {args.device.upper()}")
    print(f"Mixed Precision: {'OFF' if args.no_amp else 'ON'}")
    print(f"Grad Accum: {args.grad_accum}")
    print("=" * 70)
    
    if args.device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    print("\n📚 Loading data...")
    text = load_shakespeare()
    tokenizer = create_tokenizer(text, args.tokenizer, args.vocab_size)
    
    # 创建数据集并分割为训练集和验证集
    full_dataset = TextDataset([text], tokenizer, args.seq_len)
    val_size = int(len(full_dataset) * 0.1)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    vocab_size = tokenizer.get_vocab_size()
    print(f"   Vocab: {vocab_size}")
    print(f"   Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    config = ModelConfig(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        max_seq_len=args.seq_len,
        dropout=args.dropout,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.grad_accum,
        use_amp=not args.no_amp,
        seq_len=args.seq_len,
        device=args.device,
        early_stopping_patience=args.patience,
        label_smoothing=args.label_smoothing,
        lr_warmup_steps=args.warmup_steps,
        tie_word_embeddings=args.tie_embeddings,
        gradient_checkpointing=args.gradient_checkpointing,
        norm_type=args.norm_type,
    )
    
    print("\n🏗️ Creating model...")
    model = ShannonB1(config).to(args.device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params:,}")
    print(f"   Size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    # 优化器参数分组：对偏置、归一化层和嵌入不使用 weight decay
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lname = name.lower()
        if lname.endswith('bias') or 'norm' in lname or 'ln_' in lname or 'rmsnorm' in lname or 'embedding' in lname or 'pos_embedding' in lname:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {'params': decay_params, 'weight_decay': config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ],
        lr=config.learning_rate
    )
    
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    scheduler = CosineAnnealingWarmupLR(optimizer, warmup_steps=args.warmup_steps, total_steps=total_steps)
    
    trainer = ImprovedTrainer(model, train_loader, val_loader, config, optimizer, scheduler)
    # 恢复训练（如果提供 checkpoint）
    if args.resume:
        if os.path.exists(args.resume):
            trainer.load_checkpoint(args.resume)
        else:
            print(f"⚠️ Resume checkpoint not found: {args.resume}")

    history = trainer.train(args.epochs)
    
    trainer.save_checkpoint(args.save_path)
    tokenizer.save(args.save_path.replace('.pt', '_tokenizer.json'))
    
    print(f"\n💾 Saved: {args.save_path}")
    return history


if __name__ == "__main__":
    main()