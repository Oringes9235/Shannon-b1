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
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # 从 checkpoint 加载配置
    if 'config' in checkpoint:
        config = checkpoint['config']
    elif 'model_config' in checkpoint:
        config = checkpoint['model_config']
    else:
        # 从模型状态字典推断配置
        state_dict = checkpoint['model_state_dict']
        vocab_size = state_dict['token_embedding.weight'].shape[0]
        d_model = state_dict['token_embedding.weight'].shape[1]
        max_seq_len = state_dict['pos_encoding.pe'].shape[1]
        
        config = ModelConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            max_seq_len=max_seq_len
        )
    
    model = ShannonB1(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--prompt', type=str, default="The ")
    parser.add_argument('--max-tokens', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--device', type=str, default='cpu')
    
    args = parser.parse_args()
    
    # 加载模型
    model, tokenizer, config = load_model(args.model_path, args.device)
    
    print(f"Model loaded: vocab_size={config.vocab_size}, d_model={config.d_model}")
    
    # 编码提示词
    start_tokens = tokenizer.encode(args.prompt)[:50]
    
    # 生成
    with torch.no_grad():
        generated = model.generate(
            start_tokens,
            args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k
        )
    
    # 解码输出
    text = tokenizer.decode(generated)
    print(f"\n📝 Prompt: {args.prompt}")
    print(f"💬 Generated: {text}")


if __name__ == "__main__":
    main()