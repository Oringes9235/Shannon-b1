#!/usr/bin/env python
"""
流式文本生成脚本 - 实时显示生成过程
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
import time

from src.model import ShannonB1, ModelConfig
from src.data import CharTokenizer, BPETokenizer


def load_model(model_path: str, device: str = 'cpu'):
    """
    加载模型、分词器和配置
    
    Args:
        model_path (str): 模型文件路径
        device (str): 运行设备，默认为'cpu'
    
    Returns:
        tuple: 包含模型、分词器和配置的元组 (model, tokenizer, config)
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # 尝试从检查点中获取配置信息
    if 'config' in checkpoint:
        config = checkpoint['config']
    elif 'model_config' in checkpoint:
        config = checkpoint['model_config']
    else:
        # 如果检查点中没有配置信息，则从状态字典中推断配置
        state_dict = checkpoint['model_state_dict']
        vocab_size = state_dict['token_embedding.weight'].shape[0]
        d_model = state_dict['token_embedding.weight'].shape[1]
        max_seq_len = state_dict['pos_encoding.pe'].shape[1]
        config = ModelConfig(vocab_size=vocab_size, d_model=d_model, max_seq_len=max_seq_len)
    
    model = ShannonB1(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 尝试加载对应的分词器文件
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
        # 如果没有找到分词器文件，创建一个基础分词器
        tokenizer = CharTokenizer()
        tokenizer.build_vocab(["sample text"], 1000)
    
    return model, tokenizer, config


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='流式文本生成')
    parser.add_argument('--model-path', '--checkpoint', type=str, required=True, help='模型文件路径')
    parser.add_argument('--prompt', type=str, default="The ", help='提示词')
    parser.add_argument('--max-tokens', '--max-new-tokens', type=int, default=100, help='最大生成token数')
    parser.add_argument('--temperature', type=float, default=0.8, help='温度参数')
    parser.add_argument('--top-k', type=int, default=50, help='Top-K采样参数')
    parser.add_argument('--top-p', type=float, default=None, help='Top-P采样参数')
    parser.add_argument('--repetition-penalty', type=float, default=1.1, help='重复惩罚系数')
    parser.add_argument('--device', type=str, default='cpu', help='运行设备')
    parser.add_argument('--delay', type=float, default=0.05, help='每个token之间的延迟（秒），模拟打字效果')
    
    args = parser.parse_args()
    
    print("🔄 加载模型...")
    model, tokenizer, config = load_model(args.model_path, args.device)
    
    print(f"✅ 模型加载完成: vocab={config.vocab_size}, d_model={config.d_model}")
    print(f"📝 分词器类型: {'BPE' if hasattr(tokenizer, 'merges') else 'Char'}")
    print(f"\n{'='*60}")
    print(f"💬 Prompt: {args.prompt}")
    print(f"{'='*60}\n")
    
    # 编码提示词
    start_tokens = tokenizer.encode(args.prompt)[:50]
    
    # 流式生成
    print("🚀 开始流式生成:\n")
    
    generated_tokens = list(start_tokens)
    start_time = time.time()
    
    try:
        for token_id, probability in model.generate_stream(
            start_tokens,
            args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty
        ):
            generated_tokens.append(token_id)
            
            # 解码当前文本
            current_text = tokenizer.decode(generated_tokens)
            current_text = current_text.replace('</w>', ' ').replace('  ', ' ')
            
            # 清屏并显示（简单的刷新效果）
            print(f"\r{current_text}", end='', flush=True)
            
            # 添加延迟模拟打字效果
            if args.delay > 0:
                time.sleep(args.delay)
        
        # 生成完成
        elapsed_time = time.time() - start_time
        tokens_generated = len(generated_tokens) - len(start_tokens)
        
        print(f"\n\n{'='*60}")
        print(f"✅ 生成完成!")
        print(f"📊 统计信息:")
        print(f"   - 生成token数: {tokens_generated}")
        print(f"   - 耗时: {elapsed_time:.2f}秒")
        print(f"   - 速度: {tokens_generated/elapsed_time:.2f} tokens/秒")
        print(f"{'='*60}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断生成")
    except Exception as e:
        print(f"\n\n❌ 生成出错: {e}")


if __name__ == "__main__":
    main()