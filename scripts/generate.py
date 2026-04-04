#!/usr/bin/env python
"""
文本生成脚本
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse

from src.model import ShannonB1, ModelConfig
from src.data import CharTokenizer, BPETokenizer


def load_model(model_path: str, device: str = 'cpu'):
    """加载模型"""
    checkpoint = torch.load(model_path, map_location=device)
    
    config = checkpoint.get('model_config')
    if config is None:
        # 从checkpoint推断配置
        config = ModelConfig()
    
    model = ShannonB1(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 加载分词器
    tokenizer_path = model_path.replace('.pt', '_tokenizer.json')
    if os.path.exists(tokenizer_path):
        tokenizer_type = checkpoint.get('tokenizer_type', 'char')
        if tokenizer_type == 'bpe':
            tokenizer = BPETokenizer()
        else:
            tokenizer = CharTokenizer()
        tokenizer.load(tokenizer_path)
    else:
        tokenizer = CharTokenizer()
    
    return model, tokenizer, config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--prompt', type=str, default="The ")
    parser.add_argument('--max-tokens', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--device', type=str, default='cpu')
    
    args = parser.parse_args()
    
    # 加载模型
    model, tokenizer, _ = load_model(args.model_path, args.device)
    model.eval()
    
    # 编码提示词
    start_tokens = tokenizer.encode(args.prompt)[:50]
    
    # 生成
    with torch.no_grad():
        generated = model.generate(
            start_tokens,
            args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )
    
    # 解码输出
    text = tokenizer.decode(generated)
    print(f"\n📝 Prompt: {args.prompt}")
    print(f"💬 Generated: {text}")


if __name__ == "__main__":
    main()