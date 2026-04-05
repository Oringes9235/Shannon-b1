"""
训练模块单元测试
"""

import sys
import os
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import unittest
from torch.utils.data import DataLoader, TensorDataset

from src.model import ShannonB1, ModelConfig
from src.training import Trainer, CosineAnnealingLR, StepLR, LinearWarmupLR
from src.training.metrics import compute_perplexity, compute_accuracy, compute_top_k_accuracy


class TestMetrics(unittest.TestCase):
    """测试评估指标"""
    
    def test_perplexity(self):
        """
        测试困惑度计算功能
        验证损失值转换为困惑度的正确性
        """
        loss = 2.0
        perplexity = compute_perplexity(loss)
        self.assertAlmostEqual(perplexity, 7.389, places=2)
    
    def test_accuracy(self):
        """
        测试准确率计算功能
        验证logits和targets计算准确率的正确性
        """
        logits = torch.randn(2, 4, 10)
        targets = torch.randint(0, 10, (2, 4))
        accuracy = compute_accuracy(logits, targets)
        self.assertGreaterEqual(accuracy, 0)
        self.assertLessEqual(accuracy, 1)
    
    def test_top_k_accuracy(self):
        """
        测试top-k准确率计算功能
        验证k=5时top-k准确率计算的正确性
        """
        logits = torch.randn(2, 4, 10)
        targets = torch.randint(0, 10, (2, 4))
        acc = compute_top_k_accuracy(logits, targets, k=5)
        self.assertGreaterEqual(acc, 0)
        self.assertLessEqual(acc, 1)


class TestSchedulers(unittest.TestCase):
    """测试学习率调度器"""
    
    def setUp(self):
        """
        设置测试环境
        初始化线性模型和Adam优化器用于调度器测试
        """
        self.model = torch.nn.Linear(10, 10)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
    
    def test_cosine_annealing(self):
        """
        测试余弦退火调度器
        验证调度器能够正确更新学习率
        """
        scheduler = CosineAnnealingLR(self.optimizer, T_max=100)
        initial_lr = self.optimizer.param_groups[0]['lr']
        scheduler.step()
        self.assertNotEqual(self.optimizer.param_groups[0]['lr'], initial_lr)
    
    def test_step_lr(self):
        """
        测试步长学习率调度器
        验证经过指定步数后学习率按比例衰减
        """
        scheduler = StepLR(self.optimizer, step_size=10, gamma=0.5)
        for _ in range(10):
            scheduler.step()
        self.assertEqual(self.optimizer.param_groups[0]['lr'], 0.005)
    
    def test_warmup_lr(self):
        """
        测试线性预热学习率调度器
        验证在预热步骤中学习率逐步增加
        """
        scheduler = LinearWarmupLR(self.optimizer, warmup_steps=100, target_lr=0.1)
        for _ in range(50):
            scheduler.step()
        self.assertLess(self.optimizer.param_groups[0]['lr'], 0.1)


class TestTrainer(unittest.TestCase):
    """测试训练器"""
    
    def setUp(self):
        """
        设置训练器测试环境
        创建小规模模型配置、模型实例和假数据集
        """
        # 创建小模型和假数据
        self.config = ModelConfig(
            vocab_size=100,
            d_model=32,
            num_heads=4,
            d_ff=128,
            num_layers=2,
            max_seq_len=32,
            dropout=0.0
        )
        self.model = ShannonB1(self.config)
        
        # 创建假数据
        inputs = torch.randint(0, 100, (64, 16))
        targets = torch.randint(0, 100, (64, 16))
        dataset = TensorDataset(inputs, targets)
        self.dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
    
    def test_trainer_creation(self):
        """
        测试训练器创建
        验证训练器实例能够成功初始化
        """
        trainer = Trainer(self.model, self.dataloader, self.config, self.optimizer)
        self.assertIsNotNone(trainer)
    
    def test_train_epoch(self):
        """
        测试单epoch训练
        验证单个训练周期能够正常执行并返回正损失值
        """
        trainer = Trainer(self.model, self.dataloader, self.config, self.optimizer)
        loss = trainer.train_epoch()
        self.assertGreater(loss, 0)
    
    def test_train(self):
        """
        测试完整训练
        验证多轮训练过程和历史记录的正确性
        """
        trainer = Trainer(self.model, self.dataloader, self.config, self.optimizer)
        history = trainer.train(epochs=3)
        self.assertEqual(len(history['loss']), 3)
    
    def test_save_load_checkpoint(self):
        """
        测试保存和加载检查点
        验证模型状态能够正确保存和恢复
        """
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            path = f.name
        
        try:
            trainer = Trainer(self.model, self.dataloader, self.config, self.optimizer)
            trainer.train_epoch()
            trainer.save_checkpoint(path)
            
            # 加载检查点
            new_trainer = Trainer(self.model, self.dataloader, self.config, self.optimizer)
            new_trainer.load_checkpoint(path)
            self.assertEqual(new_trainer.best_loss, trainer.best_loss)
        finally:
            os.unlink(path)


def run_tests():
    """
    运行所有测试用例
    
    Returns:
        bool: 所有测试是否通过
    """
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestSchedulers))
    suite.addTests(loader.loadTestsFromTestCase(TestTrainer))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)