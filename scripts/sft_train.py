"""
scripts/sft_train.py
SFT 微调脚本 - 让模型学会对话
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
import json

from src.model import ShannonB1, ModelConfig
from src.data import BPETokenizer


class SFTDataset(Dataset):
    """SFT 对话数据集（修复版）"""

    def __init__(self, filepath: str, tokenizer: BPETokenizer, seq_len: int = 128):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.samples = []

        # 特殊 token 的 ID
        self.pad_id = tokenizer.special_tokens.get('<PAD>', 0)
        self.eos_id = tokenizer.special_tokens.get('<EOS>', 3)

        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()

        # 按对话块分割
        blocks = text.strip().split('\n\n')
        for block in blocks:
            block = block.strip()
            if not block or '<|user|>' not in block or '<|assistant|>' not in block:
                continue

            # 提取用户和助手内容
            parts = block.split('<|assistant|>', 1)
            user_raw = parts[0].replace('<|user|>', '').strip()
            assistant_raw = parts[1].strip()

            # 构建完整输入：<BOS> user_text \n assistant_text <EOS>
            full_text = f"{user_raw}\n{assistant_raw}"
            full_ids = tokenizer.encode(full_text, add_bos=True, add_eos=True)

            # 只编码用户部分（用于确定 assistant 的起始位置）
            user_text_with_newline = f"{user_raw}\n"
            user_ids = tokenizer.encode(user_text_with_newline, add_bos=True)

            # 构建 labels：
            # - 用户部分（包括 BOS 和换行）：全部设为 -100（忽略）
            # - assistant 部分（包括 EOS）：保留原始 token ID（需要学习）
            user_len = len(user_ids)
            labels = [-100] * user_len + full_ids[user_len:]

            # 确保 labels 和 input_ids 长度一致
            assert len(full_ids) == len(labels), f"Length mismatch: {len(full_ids)} vs {len(labels)}"

            # 截断或填充到 seq_len
            if len(full_ids) > seq_len:
                full_ids = full_ids[:seq_len]
                labels = labels[:seq_len]
            else:
                pad_len = seq_len - len(full_ids)
                full_ids = full_ids + [self.pad_id] * pad_len
                labels = labels + [-100] * pad_len

            self.samples.append({
                'input_ids': torch.tensor(full_ids, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.long),
            })

        print(f"Loaded {len(self.samples)} SFT samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def train_sft(
    model_path: str,
    tokenizer_path: str,
    data_path: str,
    save_path: str,
    epochs: int = 5,
    batch_size: int = 2,
    lr: float = 1e-5,
    seq_len: int = 128,
    device: str = 'cuda',
):
    """SFT 微调主函数"""
    
    print("=" * 60)
    print("🚀 Shannon-b1 SFT Fine-tuning")
    print("=" * 60)
    
    # 加载模型
    print("📦 Loading model...")
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    config = ckpt['config']
    model = ShannonB1(config).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"   Loaded from: {model_path}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 加载分词器
    print("📝 Loading tokenizer...")
    tokenizer = BPETokenizer(vocab_size=config.vocab_size)
    tokenizer.load(tokenizer_path)
    print(f"   Vocab size: {tokenizer.get_vocab_size()}")
    
    # 加载数据
    print("📚 Loading SFT data...")
    dataset = SFTDataset(data_path, tokenizer, seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"   Batches: {len(dataloader)}")
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # 训练
    print(f"\n{'=' * 60}")
    print(f"🏋️ Starting SFT Training")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {lr}")
    print(f"   Device: {device}")
    print(f"{'=' * 60}\n")
    
    model.train()
    best_loss = float('inf')
    
    for epoch in range(epochs):
        total_loss = 0
        
        total_batches = len(dataloader)
        n_width = len(str(total_batches))
        bar_format = (
            f"SFT Epoch {epoch+1}/{epochs}: {{percentage:3.0f}}% [{{bar}}] "
            f"{{n:>{n_width}d}}/{{total}} "
            f"[{{elapsed}}<{{remaining}}, {{rate_fmt}}{{postfix}}]"
        )
        pbar = tqdm(enumerate(dataloader), total=total_batches, ascii=".#", bar_format=bar_format)
        
        for step, batch in pbar:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            # logits = model(input_ids)
            outputs = model(input_ids)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            
            # 计算损失（只计算 assistant 部分）
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100  # 忽略用户部分和 padding
            )
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            pbar.set_postfix({'loss': f'{loss.item():.2f}'})
        
        avg_loss = total_loss / len(dataloader)
        print(f"\n📊 SFT Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}\n")
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'sft_loss': avg_loss,
            }, save_path)
            print(f"💾 Saved best model: {save_path}")
    
    # 保存分词器
    tokenizer_path_out = save_path.replace('.pt', '_tokenizer.json')
    tokenizer.save(tokenizer_path_out)
    
    print(f"\n✅ SFT completed!")
    print(f"   Best loss: {best_loss:.4f}")
    print(f"   Model saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Shannon-b1 SFT Fine-tuning')
    
    parser.add_argument('--model-path', type=str, required=True, help='预训练模型路径')
    parser.add_argument('--tokenizer-path', type=str, required=True, help='分词器路径')
    parser.add_argument('--data-path', type=str, default='data/qa_samples.txt', help='SFT 数据路径')
    parser.add_argument('--save-path', type=str, default='checkpoints/shannon_b1_sft.pt', help='保存路径')
    parser.add_argument('--epochs', type=int, default=5, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=2, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-5, help='学习率')
    parser.add_argument('--seq-len', type=int, default=128, help='序列长度')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    
    args = parser.parse_args()
    
    train_sft(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        data_path=args.data_path,
        save_path=args.save_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seq_len=args.seq_len,
        device=args.device,
    )


if __name__ == '__main__':
    main()