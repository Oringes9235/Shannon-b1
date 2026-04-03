"""
测试不同优化器的效果
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shannon.model.shannon_b1 import ShannonB1
from shannon.training.loss import CrossEntropyLoss
from shannon.training.optimizer import SGD, Adam, AdamW
from shannon.training.trainer import Trainer


def test_optimizer(optimizer_class, optimizer_name, lr, **kwargs):
    """测试单个优化器"""
    print(f"\n{'='*50}")
    print(f"测试 {optimizer_name}")
    print('='*50)
    
    # 创建模型 (使用相同的随机种子)
    np.random.seed(42)
    model = ShannonB1(
        vocab_size=100,
        d_model=32,
        num_heads=4,
        d_ff=128,
        num_layers=2,
        max_seq_len=20,
        dropout=0.0  # 关闭 dropout 以便公平比较
    )
    
    params = model.get_all_parameters()
    optimizer = optimizer_class(params, lr=lr, **kwargs)
    criterion = CrossEntropyLoss()
    trainer = Trainer(model, optimizer, criterion)
    
    # 创建训练数据 (固定种子)
    np.random.seed(42)
    train_data = [np.random.randint(0, 100, size=20).tolist() for _ in range(200)]
    
    # 训练
    history = trainer.train(train_data, epochs=5, batch_size=8, seq_len=16, verbose=False)
    
    print(f"  初始损失: {history[0]:.4f}")
    print(f"  最终损失: {history[-1]:.4f}")
    print(f"  损失下降: {history[0] - history[-1]:.4f}")
    
    return history


def main():
    print("Shannon-b1 优化器对比测试")
    print("=" * 60)
    
    results = {}
    
    # 测试 SGD
    results['SGD'] = test_optimizer(SGD, 'SGD', lr=0.01, weight_decay=0.0)
    
    # 测试 SGD with momentum
    results['SGD+Momentum'] = test_optimizer(SGD, 'SGD+Momentum', lr=0.01, 
                                              momentum=0.9, weight_decay=0.0)
    
    # 测试 Adam
    results['Adam'] = test_optimizer(Adam, 'Adam', lr=0.001, 
                                      betas=(0.9, 0.999), weight_decay=0.0)
    
    # 测试 Adam with weight decay
    results['Adam+WD'] = test_optimizer(Adam, 'Adam+WD', lr=0.001,
                                         betas=(0.9, 0.999), weight_decay=0.01)
    
    # 测试 AdamW
    results['AdamW'] = test_optimizer(AdamW, 'AdamW', lr=0.001,
                                       betas=(0.9, 0.999), weight_decay=0.01)
    
    # 打印总结
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print(f"{'优化器':<20} {'初始损失':<12} {'最终损失':<12} {'损失下降':<12}")
    print("-" * 60)
    for name, history in results.items():
        print(f"{name:<20} {history[0]:<12.4f} {history[-1]:<12.4f} {history[0]-history[-1]:<12.4f}")
    
    return results


if __name__ == "__main__":
    main()