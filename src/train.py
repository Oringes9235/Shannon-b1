"""
Shannon-b1 训练脚本
"""

import numpy as np
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shannon.model.shannon_b1 import ShannonB1
from shannon.training.loss import CrossEntropyLoss
from shannon.training.optimizer import SGD
from shannon.training.trainer import Trainer


# 配置
config = {
    'vocab_size': 100,
    'd_model': 32,
    'num_heads': 4,
    'd_ff': 128,
    'num_layers': 2,
    'max_seq_len': 20,
    'dropout': 0.1,
    'lr': 0.01,
    'epochs': 3,
    'batch_size': 4,
    'seq_len': 16
}

def main():
    print("=" * 50)
    print("Shannon-b1 训练")
    print("=" * 50)
    
    # 创建模型
    print("\n创建模型...")
    model = ShannonB1(**{k: v for k, v in config.items() 
                         if k not in ['lr', 'epochs', 'batch_size', 'seq_len']})
    
    # 打印模型参数数量
    params = model.get_all_parameters()
    total_params = sum(p.data.size for _, p in params)
    print(f"模型参数总数: {total_params:,}")
    
    # 创建优化器和损失函数
    optimizer = SGD(params, lr=config['lr'])
    criterion = CrossEntropyLoss()
    
    # 创建训练器
    trainer = Trainer(model, optimizer, criterion)
    
    # 生成假训练数据（实际使用真实文本数据）
    print("\n生成训练数据...")
    train_data = [np.random.randint(0, config['vocab_size'], size=20).tolist() 
                  for _ in range(100)]
    
    # 训练
    print("\n开始训练...")
    history = trainer.train(
        train_data, 
        epochs=config['epochs'], 
        batch_size=config['batch_size'], 
        seq_len=config['seq_len'],
        verbose=True
    )
    
    print(f"\n训练完成！最终损失: {history[-1]:.4f}")
    
    # 测试生成
    print("\n" + "=" * 50)
    print("生成测试")
    print("=" * 50)
    start_tokens = [1, 2, 3]
    generated = model.generate(start_tokens, max_new_tokens=10, temperature=0.8)
    print(f"起始序列: {start_tokens}")
    print(f"生成的序列: {generated}")
    
    # 保存模型
    import pickle
    save_path = "shannon_b1_weights.pkl"
    weights = {name: p.data for name, p in params}
    with open(save_path, 'wb') as f:
        pickle.dump(weights, f)
    print(f"\n模型权重已保存到: {save_path}")


if __name__ == "__main__":
    main()