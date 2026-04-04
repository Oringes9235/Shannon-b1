#!/usr/bin/env python
"""
模型评估脚本
计算困惑度、准确率等指标
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.model import ShannonB1, ModelConfig
from src.data import TextDataset, CharTokenizer, BPETokenizer
from src.training.metrics import compute_perplexity, compute_accuracy, compute_top_k_accuracy


def evaluate(model, dataloader, criterion, device='cpu'):
    """评估模型"""
    model.eval()
    total_loss = 0
    total_acc = 0
    total_top5_acc = 0
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        for inputs, targets in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            logits = model(inputs)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            total_loss += loss.item()
            total_acc += compute_accuracy(logits, targets)
            total_top5_acc += compute_top_k_accuracy(logits, targets, k=5)
            num_batches += 1
            
            pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    avg_top5_acc = total_top5_acc / num_batches
    perplexity = compute_perplexity(avg_loss)
    
    return {
        'loss': avg_loss,
        'accuracy': avg_acc,
        'top5_accuracy': avg_top5_acc,
        'perplexity': perplexity
    }


def load_model(model_path: str, device: str = 'cpu'):
    """加载模型"""
    checkpoint = torch.load(model_path, map_location=device)
    
    # 获取模型配置
    if 'model_config' in checkpoint:
        config = checkpoint['model_config']
    else:
        config = ModelConfig()
        # 从checkpoint推断
        for name, param in checkpoint['model_state_dict'].items():
            if 'token_embedding.weight' in name:
                config.vocab_size = param.shape[0]
                config.d_model = param.shape[1]
                break
    
    model = ShannonB1(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 加载分词器
    tokenizer_path = model_path.replace('.pt', '_tokenizer.json')
    if os.path.exists(tokenizer_path):
        tokenizer = CharTokenizer()
        tokenizer.load(tokenizer_path)
    else:
        tokenizer = CharTokenizer()
        tokenizer.build_vocab(["sample text"], 1000)
    
    return model, tokenizer, config


def main():
    parser = argparse.ArgumentParser(description='Evaluate Shannon-b1 model')
    parser.add_argument('--model-path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--test-file', type=str, default=None, help='Path to test text file')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--seq-len', type=int, default=64, help='Sequence length')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Shannon-b1 Model Evaluation")
    print(f"Model: {args.model_path}")
    print(f"Device: {args.device}")
    print("=" * 60)
    
    # 加载模型
    print("\n📦 Loading model...")
    model, tokenizer, config = load_model(args.model_path, args.device)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    # 加载测试数据
    print("\n📚 Loading test data...")
    if args.test_file and os.path.exists(args.test_file):
        with open(args.test_file, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        # 使用内置测试数据
        text = """To be or not to be, that is the question:
        Whether 'tis nobler in the mind to suffer
        The slings and arrows of outrageous fortune,
        Or to take arms against a sea of troubles
        And by opposing end them."""
    
    dataset = TextDataset([text], tokenizer, args.seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"   Test samples: {len(dataset)}")
    
    # 评估
    print("\n🔍 Evaluating...")
    criterion = torch.nn.CrossEntropyLoss()
    results = evaluate(model, dataloader, criterion, args.device)
    
    # 输出结果
    print("\n" + "=" * 60)
    print("📊 Evaluation Results")
    print("=" * 60)
    print(f"   Loss: {results['loss']:.4f}")
    print(f"   Perplexity: {results['perplexity']:.2f}")
    print(f"   Accuracy: {results['accuracy']:.2%}")
    print(f"   Top-5 Accuracy: {results['top5_accuracy']:.2%}")
    print("=" * 60)


if __name__ == "__main__":
    main()