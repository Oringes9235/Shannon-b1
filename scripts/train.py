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
from src.data import TextDataset, create_tokenizer, create_tokenizer_streaming, load_all_data, download_shakespeare
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
    parser.add_argument('--gradient-checkpointing', action='store_true', help='Enable gradient checkpointing to save memory')
    parser.add_argument('--norm-type', type=str, default='layernorm', choices=['layernorm', 'rmsnorm'], help='Normalization type')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--label-smoothing', type=float, default=0.0)
    parser.add_argument('--tie-embeddings', action='store_true', help='Tie token embedding and output projection')
    parser.add_argument('--patience', type=int, default=10)
    
    parser.add_argument('--use-rope', action='store_true', help='Enable RoPE (Rotary Positional Embeddings)')
    parser.add_argument('--rope-base', type=float, default=10000.0)
    parser.add_argument('--sliding-window-size', type=int, default=None)
    parser.add_argument('--use-alibi', action='store_true', help='Enable ALiBi (not recommended with RoPE)')
    
    parser.add_argument('--lora', action='store_true', help='Enable LoRA fine-tuning')
    parser.add_argument('--lora-rank', type=int, default=8)
    parser.add_argument('--lora-alpha', type=float, default=16.0)
    parser.add_argument('--lora-dropout', type=float, default=0.0)
    parser.add_argument('--lora-target-modules', type=str, nargs='+', default=['q_proj', 'v_proj'])
    
    parser.add_argument('--tokenizer', type=str, default='char', choices=['char', 'bpe'])
    parser.add_argument('--vocab-size', type=int, default=2000)
    # 分块训练参数（避免大数据集 OOM）
    parser.add_argument('--stream-chunk-size', type=int, default=1_000_000,
                        help='Chunk size (chars) for streaming BPE training. Default: 1M')
    
    parser.add_argument('--device', type=str, default=get_device())
    parser.add_argument('--save-path', type=str, default='checkpoints/shannon_b1.pt')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--getdata', type=str, default=None,
                        help='Download dataset before training, e.g. --getdata shakespeare')
    
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
        try:
            from src.utils import get_cuda_info
            cuda_info = get_cuda_info()
            print(f"\n[93m[CONFIG][0m CUDA Environment:")
            print(f"   CUDA Version: {cuda_info['cuda_version']}")
            print(f"   cuDNN Version: {cuda_info['cudnn_version']}")
            print(f"   Device Count: {cuda_info['device_count']}")
            for device in cuda_info['devices']:
                print(f"\n   GPU {device['index']}:")
                print(f"      Name: {device['name']}")
                print(f"      Compute Capability: {device['compute_capability']}")
                print(f"      Total Memory: {device['memory_total'] / 1024**3:.2f} GB")
            print()
        except Exception as e:
            print(f"[93m[WARNING][0m Could not retrieve detailed CUDA info: {e}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n")
    
    if args.getdata:
        if args.getdata == 'shakespeare':
            download_shakespeare()
        else:
            print(f"[91m[ERROR][0m Unknown dataset: {args.getdata}")
            print("   Available: shakespeare")

    print("[96m[LOAD][0m Loading data...")
    texts = load_all_data()
    total_chars = sum(len(t) for t in texts)
    
    # 大数据集 (>50MB) 且使用 BPE 时，使用分块流式训练
    if args.tokenizer == 'bpe' and total_chars > 50_000_000:
        print(f"[94m[INFO][0m Large dataset detected ({total_chars:,} chars), using streaming BPE training...")
        tokenizer = create_tokenizer_streaming(
            tokenizer_type='bpe',
            vocab_size=args.vocab_size,
            data_dir='data',
            chunk_size=args.stream_chunk_size,
        )
    else:
        combined_text = "\n\n".join(texts)
        tokenizer = create_tokenizer(combined_text, args.tokenizer, args.vocab_size)
    
    full_dataset = TextDataset(texts, tokenizer, args.seq_len)
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
        use_rope=args.use_rope,
        rope_base=args.rope_base,
        sliding_window_size=args.sliding_window_size,
        use_alibi=args.use_alibi,
    )
    
    print("\n[96m[BUILD][0m Creating model...")
    model = ShannonB1(config).to(args.device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params:,}")
    print(f"   Size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    if args.lora:
        print("\n[93m[CONFIG][0m Applying LoRA...")
        model.apply_lora(
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
        )
        config.learning_rate = args.lr * 10
        print(f"   LoRA learning rate adjusted to: {config.learning_rate}")
    
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
    if args.resume:
        if os.path.exists(args.resume):
            trainer.load_checkpoint(args.resume)
        else:
            print(f"[93m[WARNING][0m Resume checkpoint not found: {args.resume}")

    history = trainer.train(args.epochs)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    arch_version = _generate_arch_version_string(config)
    base_path = args.save_path.replace('.pt', '')
    versioned_path = f"{base_path}_{arch_version}_{timestamp}.pt"
    
    trainer.save_checkpoint(versioned_path)
    tokenizer.save(versioned_path.replace('.pt', '_tokenizer.json'))
    
    print(f"\n[94m[SAVE][0m Saved: {versioned_path}")
    print(f"[94m[TEMPLATE][0m Architecture: {arch_version}")
    
    return history


def _generate_arch_version_string(config: ModelConfig) -> str:
    parts = []
    parts.append(f"dm{config.d_model}")
    parts.append(f"nl{config.num_layers}")
    parts.append(f"nh{config.num_heads}")
    
    if config.use_rope:
        rope_base_short = int(config.rope_base) if config.rope_base == int(config.rope_base) else config.rope_base
        parts.append(f"rope{int(rope_base_short)}")
    elif config.use_alibi:
        parts.append("alibi")
    else:
        parts.append("fixed")
    
    if config.sliding_window_size:
        parts.append(f"sw{config.sliding_window_size}")
    
    parts.append(config.norm_type.replace('norm', ''))
    
    if config.tie_word_embeddings:
        parts.append("tie")
    if config.gradient_checkpointing:
        parts.append("ckpt")
    
    return "_".join(parts)


if __name__ == "__main__":
    main()