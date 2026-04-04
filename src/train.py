"""
Shannon-b1 训练脚本 - 支持真实文本数据
"""

import numpy as np
import sys
import os
import argparse
import pickle

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shannon.model.shannon_b1 import ShannonB1
from shannon.training.loss import CrossEntropyLoss
from shannon.training.optimizer import SGD, Adam, AdamW
from shannon.training.trainer import Trainer
from shannon.utils.tokenizer import CharTokenizer, BPETokenizer
from data.dataset import TextDataset, load_training_data
from data.download import prepare_data, download_shakespeare, create_sample_data


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Shannon-b1 训练脚本')
    
    # 模型配置
    parser.add_argument('--vocab-size', type=int, default=1000, help='词表大小')
    parser.add_argument('--d-model', type=int, default=64, help='模型维度')
    parser.add_argument('--num-heads', type=int, default=4, help='注意力头数')
    parser.add_argument('--d-ff', type=int, default=256, help='FFN隐藏层维度')
    parser.add_argument('--num-layers', type=int, default=3, help='Transformer层数')
    parser.add_argument('--max-seq-len', type=int, default=128, help='最大序列长度')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout率')
    
    # 训练配置
    parser.add_argument('--optimizer', type=str, default='adam', 
                        choices=['sgd', 'adam', 'adamw'], help='优化器类型')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--betas', type=float, nargs=2, default=[0.9, 0.999], 
                        help='Adam的beta参数')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    parser.add_argument('--seq-len', type=int, default=64, help='序列长度')
    
    # 数据配置
    parser.add_argument('--dataset', type=str, default='shakespeare',
                        choices=['shakespeare', 'sample', 'random', 'file'],
                        help='数据集类型')
    parser.add_argument('--data-path', type=str, default=None, help='数据文件路径')
    parser.add_argument('--tokenizer', type=str, default='char',
                        choices=['char', 'bpe'], help='分词器类型')
    
    # 其他
    parser.add_argument('--save-path', type=str, default='shannon_b1_weights.pkl', 
                        help='模型保存路径')
    parser.add_argument('--load-path', type=str, default=None, help='加载模型路径')
    parser.add_argument('--eval-interval', type=int, default=100, help='评估间隔')
    
    return parser.parse_args()


def create_tokenizer(args, texts=None):
    """创建分词器"""
    if args.tokenizer == 'char':
        tokenizer = CharTokenizer()
        
        if texts is None:
            # 如果没有提供文本，使用示例文本构建词表
            sample_text = "Hello world! This is a sample text for building vocabulary."
            texts = [sample_text]
        
        tokenizer.build_vocab(texts, args.vocab_size)
    else:
        tokenizer = BPETokenizer(args.vocab_size)
        if texts:
            tokenizer.train(texts)
    
    return tokenizer


def prepare_training_data(args):
    """准备训练数据"""
    print("\n" + "=" * 60)
    print("准备训练数据")
    print("=" * 60)
    
    if args.dataset == 'shakespeare':
        # 下载并加载莎士比亚文本
        data_path = download_shakespeare()
        if data_path and os.path.exists(data_path):
            with open(data_path, 'r', encoding='utf-8') as f:
                text = f.read()
            print(f"✅ 加载莎士比亚文本: {len(text):,} 字符")
            
            # 创建分词器
            tokenizer = create_tokenizer(args, [text])
            
            # 创建数据集
            dataset = TextDataset(tokenizer, args.seq_len)
            dataset.load_text(text, "Shakespeare")
            
            return dataset, tokenizer
        else:
            print("⚠️ 莎士比亚文本下载失败，使用示例数据")
            args.dataset = 'sample'
    
    if args.dataset == 'sample':
        # 使用示例数据
        data_path = create_sample_data()
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"✅ 加载示例数据: {len(text):,} 字符")
        
        tokenizer = create_tokenizer(args, [text])
        dataset = TextDataset(tokenizer, args.seq_len)
        dataset.load_text(text, "Sample")
        
        return dataset, tokenizer
    
    if args.dataset == 'file' and args.data_path:
        # 从文件加载
        with open(args.data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"✅ 加载文件: {args.data_path} ({len(text):,} 字符)")
        
        tokenizer = create_tokenizer(args, [text])
        dataset = TextDataset(tokenizer, args.seq_len)
        dataset.load_file(args.data_path)
        
        return dataset, tokenizer
    
    # 随机数据
    print("⚠️ 使用随机数据训练")
    tokenizer = create_tokenizer(args, ["abc def ghi"])
    dataset = None
    return dataset, tokenizer


def generate_sample(model, tokenizer, prompt="", max_tokens=50, temperature=0.8):
    """生成文本样本"""
    if prompt:
        start_tokens = tokenizer.encode(prompt)
    else:
        start_tokens = [tokenizer.special_tokens.get('<BOS>', 0)]
    
    generated = model.generate(start_tokens, max_tokens, temperature)
    text = tokenizer.decode(generated)
    
    return text


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Shannon-b1 训练 - 真实文本数据版本")
    print("=" * 60)
    
    # 打印配置
    print("\n📋 配置:")
    print(f"  模型: vocab_size={args.vocab_size}, d_model={args.d_model}, "
          f"num_heads={args.num_heads}, num_layers={args.num_layers}")
    print(f"  训练: optimizer={args.optimizer}, lr={args.lr}, "
          f"epochs={args.epochs}, batch_size={args.batch_size}")
    print(f"  数据: dataset={args.dataset}, seq_len={args.seq_len}")
    
    # 准备数据
    dataset, tokenizer = prepare_training_data(args)
    
    # 更新词表大小
    actual_vocab_size = tokenizer.get_vocab_size()
    if actual_vocab_size != args.vocab_size:
        print(f"\n📝 实际词表大小: {actual_vocab_size}")
    
    # 创建模型
    print("\n🏗️ 创建模型...")
    model = ShannonB1(
        vocab_size=actual_vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout
    )
    
    # 加载预训练权重
    if args.load_path and os.path.exists(args.load_path):
        with open(args.load_path, 'rb') as f:
            weights = pickle.load(f)
            for name, param in model.get_all_parameters():
                if name in weights:
                    param.data = weights[name]
        print(f"✅ 加载模型权重: {args.load_path}")
    
    # 打印模型信息
    params = model.get_all_parameters()
    total_params = sum(p.data.size for _, p in params)
    print(f"   模型参数总数: {total_params:,}")
    
    # 创建优化器
    if args.optimizer == 'sgd':
        optimizer = SGD(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = Adam(params, lr=args.lr, betas=tuple(args.betas), 
                         weight_decay=args.weight_decay)
    else:
        optimizer = AdamW(params, lr=args.lr, betas=tuple(args.betas),
                          weight_decay=args.weight_decay)
    
    # 创建损失函数和训练器
    criterion = CrossEntropyLoss()
    trainer = Trainer(model, optimizer, criterion)
    
    # 训练
    print("\n🚀 开始训练...")
    print("-" * 60)
    
    if dataset is not None:
        # 使用真实数据训练
        train_data = dataset.get_all_sequences()
        history = trainer.train(
            train_data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            verbose=True
        )
    else:
        # 使用随机数据训练 (不推荐)
        train_data = [np.random.randint(0, actual_vocab_size, size=args.seq_len).tolist() 
                      for _ in range(500)]
        history = trainer.train(
            train_data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            verbose=True
        )
    
    print("-" * 60)
    print(f"\n✅ 训练完成！")
    print(f"   初始损失: {history[0]:.4f}")
    print(f"   最终损失: {history[-1]:.4f}")
    print(f"   损失下降: {history[0] - history[-1]:.4f}")
    
    # 生成测试
    print("\n" + "=" * 60)
    print("🎨 文本生成测试")
    print("=" * 60)
    
    test_prompts = [
        "Once upon a time",
        "The future of AI is",
        "In the beginning",
    ]
    
    for prompt in test_prompts:
        generated = generate_sample(model, tokenizer, prompt, max_tokens=30, temperature=0.8)
        print(f"\n📝 提示词: {prompt}")
        print(f"💬 生成: {generated}")
    
    # 保存模型
    print("\n💾 保存模型...")
    weights = {name: p.data for name, p in params}
    with open(args.save_path, 'wb') as f:
        pickle.dump(weights, f)
    print(f"   模型权重已保存到: {args.save_path}")
    
    # 保存词表
    tokenizer_save_path = args.save_path.replace('.pkl', '_tokenizer.json')
    if hasattr(tokenizer, 'save'):
        tokenizer.save(tokenizer_save_path)
    
    # 保存优化器状态
    if hasattr(optimizer, 'get_state'):
        opt_state = optimizer.get_state()
        opt_path = args.save_path.replace('.pkl', '_optimizer.pkl')
        with open(opt_path, 'wb') as f:
            pickle.dump(opt_state, f)
        print(f"   优化器状态已保存到: {opt_path}")
    
    return history


if __name__ == "__main__":
    history = main()