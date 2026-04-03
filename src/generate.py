#!/usr/bin/env python
"""
Shannon-b1 文本生成脚本
用法: python generate.py --prompt "Hello" --max-tokens 50 --temperature 0.8
"""

import argparse
import numpy as np
from shannon import ShannonB1


def load_model(model_path=None, config=None):
    """
    加载模型
    实际使用时需要实现模型权重的保存和加载
    """
    if config is None:
        config = {
            'vocab_size': 1000,
            'd_model': 64,
            'num_heads': 4,
            'd_ff': 256,
            'num_layers': 2,
            'max_seq_len': 50,
            'dropout': 0.0
        }
    
    model = ShannonB1(**config)
    
    if model_path:
        # 加载保存的权重
        import pickle
        with open(model_path, 'rb') as f:
            weights = pickle.load(f)
            # 这里需要实现权重加载逻辑
            print(f"从 {model_path} 加载权重")
    
    return model


def tokenize(text, vocab):
    """
    简单的分词器（示例）
    实际使用中需要实现真正的分词逻辑
    """
    # 这里是一个极其简化的示例
    # 实际应该使用 BPE 或 SentencePiece
    tokens = []
    for char in text:
        if char in vocab:
            tokens.append(vocab[char])
        else:
            tokens.append(vocab.get('<UNK>', 0))
    return tokens


def detokenize(tokens, idx_to_char):
    """
    去分词器（示例）
    """
    return ''.join([idx_to_char.get(t, '?') for t in tokens])


def create_simple_vocab():
    """
    创建一个简单的字符级词表（示例）
    """
    chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:"
    vocab = {ch: i+1 for i, ch in enumerate(chars)}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = len(vocab)
    idx_to_char = {i: ch for ch, i in vocab.items()}
    return vocab, idx_to_char


def generate_text(model, prompt, max_new_tokens=50, temperature=0.8, vocab=None, idx_to_char=None):
    """
    生成文本
    """
    if vocab is None:
        vocab, idx_to_char = create_simple_vocab()
    
    # 将提示词转换为token
    start_tokens = tokenize(prompt, vocab)
    
    if not start_tokens:
        print("警告: 提示词为空或无法分词")
        return prompt
    
    # 生成
    generated_tokens = model.generate(start_tokens, max_new_tokens, temperature)
    
    # 转换回文本
    generated_text = detokenize(generated_tokens, idx_to_char)
    
    return generated_text


def main():
    parser = argparse.ArgumentParser(description='Shannon-b1 文本生成')
    parser.add_argument('--prompt', '-p', type=str, default='Hello', 
                        help='生成文本的提示词')
    parser.add_argument('--max-tokens', '-n', type=int, default=50,
                        help='最大生成token数')
    parser.add_argument('--temperature', '-t', type=float, default=0.8,
                        help='温度系数 (0.1-2.0)')
    parser.add_argument('--model-path', '-m', type=str, default=None,
                        help='模型权重文件路径')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='交互式生成模式')
    
    args = parser.parse_args()
    
    # 创建模型
    print("加载 Shannon-b1 模型...")
    config = {
        'vocab_size': 1000,
        'd_model': 64,
        'num_heads': 4,
        'd_ff': 256,
        'num_layers': 2,
        'max_seq_len': 100,
        'dropout': 0.0
    }
    model = load_model(args.model_path, config)
    vocab, idx_to_char = create_simple_vocab()
    
    if args.interactive:
        # 交互式模式
        print("\n进入交互式生成模式 (输入 'quit' 退出)")
        print("-" * 50)
        while True:
            prompt = input("\nPrompt: ").strip()
            if prompt.lower() == 'quit':
                break
            if not prompt:
                continue
            
            generated = generate_text(
                model, prompt, 
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                vocab=vocab, idx_to_char=idx_to_char
            )
            print(f"生成: {generated}")
    else:
        # 单次生成模式
        generated = generate_text(
            model, args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            vocab=vocab, idx_to_char=idx_to_char
        )
        print(f"\n提示词: {args.prompt}")
        print(f"生成结果: {generated}")


if __name__ == "__main__":
    main()