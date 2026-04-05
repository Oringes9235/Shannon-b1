#!/usr/bin/env python
"""
文本生成脚本 - 支持重复惩罚
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
    
    if 'config' in checkpoint:
        config = checkpoint['config']
    elif 'model_config' in checkpoint:
        config = checkpoint['model_config']
    else:
        state_dict = checkpoint['model_state_dict']
        vocab_size = state_dict['token_embedding.weight'].shape[0]
        d_model = state_dict['token_embedding.weight'].shape[1]
        max_seq_len = state_dict['pos_encoding.pe'].shape[1]
        config = ModelConfig(vocab_size=vocab_size, d_model=d_model, max_seq_len=max_seq_len)
    
    model = ShannonB1(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    tokenizer_path = model_path.replace('.pt', '_tokenizer.json')
    if os.path.exists(tokenizer_path):
        import json
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if 'char_to_idx' in data:
            tokenizer = CharTokenizer()
            tokenizer.load(tokenizer_path)
        else:
            tokenizer = BPETokenizer()
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
    parser.add_argument('--top-p', type=float, default=None)
    parser.add_argument('--repetition-penalty', type=float, default=1.1, help='重复惩罚系数 (>1 减少重复)')
    parser.add_argument('--presence-penalty', type=float, default=0.0, help='Presence penalty (subtract from logits for seen tokens)')
    parser.add_argument('--frequency-penalty', type=float, default=0.0, help='Frequency penalty (subtract scaled by count)')
    parser.add_argument('--no-repeat-last', action='store_false', dest='ban_immediate_repeat', help='Allow repeating last token')
    parser.add_argument('--ngram-block', type=int, default=3, help='Block repeating n-grams of this size (0 to disable)')
    parser.add_argument('--best-of', type=int, default=1, help='Generate this many samples and pick best by log-prob')
    parser.add_argument('--max-repetition', type=int, default=3, help='Max times a single token may appear in generated output (hard cap)')
    parser.add_argument('--device', type=str, default='cpu')
    
    args = parser.parse_args()
    
    model, tokenizer, config = load_model(args.model_path, args.device)
    
    print(f"Model: vocab={config.vocab_size}, d_model={config.d_model}")
    print(f"Tokenizer: {'BPE' if hasattr(tokenizer, 'merges') else 'Char'}")
    
    start_tokens = tokenizer.encode(args.prompt)[:50]
    print(f"Start tokens: {start_tokens[:10]}...")
    
    with torch.no_grad():
        generated = model.generate(
            start_tokens,
            args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            presence_penalty=args.presence_penalty,
            frequency_penalty=args.frequency_penalty,
            ban_immediate_repeat=args.ban_immediate_repeat
            ,ngram_block_size=args.ngram_block, best_of=args.best_of, max_repetition=args.max_repetition
        )
    
    text = tokenizer.decode(generated)
    print(f"\n📝 Prompt: {args.prompt}")
    print(f"💬 Generated: {text}")


if __name__ == "__main__":
    main()