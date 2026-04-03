"""
Shannon-b1 训练脚本
支持多种优化器: SGD, Adam, AdamW
"""

import numpy as np
import sys
import os
import argparse

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shannon.model.shannon_b1 import ShannonB1
from shannon.training.loss import CrossEntropyLoss
from shannon.training.optimizer import SGD, Adam, AdamW
from shannon.training.trainer import Trainer


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Shannon-b1 训练脚本')
    
    # 模型配置
    parser.add_argument('--vocab-size', type=int, default=100, help='词表大小')
    parser.add_argument('--d-model', type=int, default=32, help='模型维度')
    parser.add_argument('--num-heads', type=int, default=4, help='注意力头数')
    parser.add_argument('--d-ff', type=int, default=128, help='FFN隐藏层维度')
    parser.add_argument('--num-layers', type=int, default=2, help='Transformer层数')
    parser.add_argument('--max-seq-len', type=int, default=20, help='最大序列长度')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout率')
    
    # 训练配置
    parser.add_argument('--optimizer', type=str, default='adam', 
                        choices=['sgd', 'adam', 'adamw'], help='优化器类型')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--betas', type=float, nargs=2, default=[0.9, 0.999], 
                        help='Adam的beta参数')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=8, help='批次大小')
    parser.add_argument('--seq-len', type=int, default=16, help='序列长度')
    
    # 数据配置
    parser.add_argument('--train-size', type=int, default=500, help='训练数据大小')
    parser.add_argument('--seq-length', type=int, default=20, help='序列长度')
    
    # 其他
    parser.add_argument('--save-path', type=str, default='shannon_b1_weights.pkl', 
                        help='模型保存路径')
    
    return parser.parse_args()


def create_training_data(vocab_size, num_samples=500, seq_length=20):
    """
    创建训练数据 (示例: 随机数据)
    实际使用时替换为真实文本数据
    """
    # 这里使用随机数据演示
    # 真实场景应该加载文本文件并进行分词
    train_data = [np.random.randint(0, vocab_size, size=seq_length).tolist() 
                  for _ in range(num_samples)]
    return train_data


def get_optimizer(name, params, lr, betas, weight_decay):
    """根据名称获取优化器"""
    if name == 'sgd':
        return SGD(params, lr=lr, weight_decay=weight_decay)
    elif name == 'adam':
        return Adam(params, lr=lr, betas=betas, weight_decay=weight_decay)
    elif name == 'adamw':
        return AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Shannon-b1 训练")
    print("=" * 60)
    
    # 打印配置
    print("\n配置:")
    print(f"  模型: vocab_size={args.vocab_size}, d_model={args.d_model}, "
          f"num_heads={args.num_heads}, d_ff={args.d_ff}, "
          f"num_layers={args.num_layers}, max_seq_len={args.max_seq_len}")
    print(f"  训练: optimizer={args.optimizer}, lr={args.lr}, "
          f"weight_decay={args.weight_decay}, epochs={args.epochs}, "
          f"batch_size={args.batch_size}")
    print(f"  数据: train_size={args.train_size}, seq_length={args.seq_length}")
    
    # 创建模型
    print("\n创建模型...")
    model = ShannonB1(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout
    )
    
    # 打印模型参数数量
    params = model.get_all_parameters()
    total_params = sum(p.data.size for _, p in params)
    print(f"模型参数总数: {total_params:,}")
    
    # 创建优化器
    optimizer = get_optimizer(
        args.optimizer, 
        params, 
        args.lr, 
        tuple(args.betas), 
        args.weight_decay
    )
    
    # 创建损失函数
    criterion = CrossEntropyLoss()
    
    # 创建训练器
    trainer = Trainer(model, optimizer, criterion)
    
    # 创建训练数据
    print("\n生成训练数据...")
    train_data = create_training_data(args.vocab_size, args.train_size, args.seq_length)
    
    # 训练
    print("\n开始训练...")
    print("-" * 60)
    history = trainer.train(
        train_data, 
        epochs=args.epochs, 
        batch_size=args.batch_size, 
        seq_len=args.seq_len,
        verbose=True
    )
    
    print("-" * 60)
    print(f"\n训练完成！")
    print(f"  初始损失: {history[0]:.4f}")
    print(f"  最终损失: {history[-1]:.4f}")
    print(f"  损失下降: {history[0] - history[-1]:.4f}")
    
    # 测试生成
    print("\n" + "=" * 60)
    print("生成测试")
    print("=" * 60)
    start_tokens = [1, 2, 3]
    generated = model.generate(start_tokens, max_new_tokens=10, temperature=0.8)
    print(f"起始序列: {start_tokens}")
    print(f"生成的序列: {generated}")
    
    # 保存模型
    import pickle
    weights = {name: p.data for name, p in params}
    with open(args.save_path, 'wb') as f:
        pickle.dump(weights, f)
    print(f"\n模型权重已保存到: {args.save_path}")
    
    # 保存优化器状态
    if hasattr(optimizer, 'get_state'):
        opt_state = optimizer.get_state()
        opt_path = args.save_path.replace('.pkl', '_optimizer.pkl')
        with open(opt_path, 'wb') as f:
            pickle.dump(opt_state, f)
        print(f"优化器状态已保存到: {opt_path}")
    
    return history


def compare_optimizers():
    """
    对比不同优化器的训练效果
    """
    import matplotlib.pyplot as plt
    
    optimizers = ['sgd', 'adam', 'adamw']
    histories = {}
    
    for opt_name in optimizers:
        print(f"\n{'='*60}")
        print(f"测试优化器: {opt_name.upper()}")
        print('='*60)
        
        # 创建模型
        model = ShannonB1(
            vocab_size=100,
            d_model=32,
            num_heads=4,
            d_ff=128,
            num_layers=2,
            max_seq_len=20,
            dropout=0.1
        )
        
        params = model.get_all_parameters()
        
        # 选择优化器
        if opt_name == 'sgd':
            optimizer = SGD(params, lr=0.01, weight_decay=0.01)
        elif opt_name == 'adam':
            optimizer = Adam(params, lr=0.001, weight_decay=0.01)
        else:
            optimizer = AdamW(params, lr=0.001, weight_decay=0.01)
        
        criterion = CrossEntropyLoss()
        trainer = Trainer(model, optimizer, criterion)
        
        # 训练数据
        train_data = [np.random.randint(0, 100, size=20).tolist() for _ in range(200)]
        
        # 训练
        history = trainer.train(train_data, epochs=5, batch_size=8, seq_len=16, verbose=False)
        histories[opt_name] = history
        
        print(f"最终损失: {history[-1]:.4f}")
    
    # 绘制对比图
    plt.figure(figsize=(10, 6))
    for opt_name, history in histories.items():
        plt.plot(history, label=opt_name.upper(), linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Optimizer Comparison on Shannon-b1')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('optimizer_comparison.png', dpi=150)
    plt.show()
    
    return histories


if __name__ == "__main__":
    # 基本训练
    history = main()
    
    # 可选: 对比优化器 (需要 matplotlib)
    # compare_optimizers()