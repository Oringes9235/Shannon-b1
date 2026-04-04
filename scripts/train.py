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
    parser = argparse.ArgumentParser(description='Shannon-b1 Training')
    
    parser.add_argument('--d-model', type=int, default=128)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--d-ff', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seq-len', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--grad-accum', type=int, default=1)
    
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--warmup-steps', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=10)
    
    parser.add_argument('--tokenizer', type=str, default='char', choices=['char', 'bpe'])
    parser.add_argument('--vocab-size', type=int, default=2000)
    
    parser.add_argument('--device', type=str, default=get_device())
    parser.add_argument('--save-path', type=str, default='checkpoints/shannon_b1.pt')
    parser.add_argument('--seed', type=int, default=42)
    
    return parser.parse_args()


def main():
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
    )
    
    print("\n🏗️ Creating model...")
    model = ShannonB1(config).to(args.device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params:,}")
    print(f"   Size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    scheduler = CosineAnnealingWarmupLR(optimizer, warmup_steps=args.warmup_steps, total_steps=total_steps)
    
    trainer = ImprovedTrainer(model, train_loader, val_loader, config, optimizer, scheduler)
    history = trainer.train(args.epochs)
    
    trainer.save_checkpoint(args.save_path)
    tokenizer.save(args.save_path.replace('.pt', '_tokenizer.json'))
    
    print(f"\n💾 Saved: {args.save_path}")
    return history


if __name__ == "__main__":
    main()