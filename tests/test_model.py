"""
模型单元测试
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import unittest

from src.model import ShannonB1, ShannonB1Encoder, ModelConfig
from src.model.layers import PositionalEncoding, CausalMask, RMSNorm


class TestModelConfig(unittest.TestCase):
    """测试模型配置"""
    
    def test_default_config(self):
        config = ModelConfig()
        self.assertEqual(config.d_model, 128)
        self.assertEqual(config.num_heads, 8)
        self.assertEqual(config.num_layers, 4)
    
    def test_custom_config(self):
        config = ModelConfig(d_model=256, num_layers=6)
        self.assertEqual(config.d_model, 256)
        self.assertEqual(config.num_layers, 6)


class TestLayers(unittest.TestCase):
    """测试自定义层"""
    
    def test_positional_encoding(self):
        pe = PositionalEncoding(d_model=128, max_seq_len=512)
        x = torch.randn(2, 10, 128)
        output = pe(x)
        self.assertEqual(output.shape, (2, 10, 128))
    
    def test_causal_mask(self):
        mask_layer = CausalMask(max_seq_len=512)
        mask = mask_layer(seq_len=10)
        self.assertEqual(mask.shape, (10, 10))
        # 检查上三角为 -inf
        self.assertTrue(torch.isinf(mask[0, 1]))
        self.assertFalse(torch.isinf(mask[1, 0]))
    
    def test_rms_norm(self):
        rms_norm = RMSNorm(d_model=128)
        x = torch.randn(2, 10, 128)
        output = rms_norm(x)
        self.assertEqual(output.shape, (2, 10, 128))


class TestShannonB1(unittest.TestCase):
    """测试主模型"""
    
    def setUp(self):
        self.config = ModelConfig(
            vocab_size=1000,
            d_model=64,
            num_heads=4,
            d_ff=256,
            num_layers=2,
            max_seq_len=32,
            dropout=0.1
        )
        self.model = ShannonB1(self.config)
    
    def test_forward_shape(self):
        """测试前向传播输出形状"""
        tokens = torch.randint(0, 1000, (2, 16))
        logits = self.model(tokens)
        self.assertEqual(logits.shape, (2, 16, 1000))
    
    def test_generate(self):
        """测试生成功能"""
        start_tokens = [1, 2, 3]
        generated = self.model.generate(start_tokens, max_new_tokens=10, temperature=0.8)
        self.assertEqual(len(generated), 13)  # 3 + 10
        self.assertEqual(generated[:3], start_tokens)
    
    def test_parameter_count(self):
        """测试参数量"""
        total_params = sum(p.numel() for p in self.model.parameters())
        # 应该有合理的参数量
        self.assertGreater(total_params, 10000)
        self.assertLess(total_params, 1000000)
    
    def test_device_compatibility(self):
        """测试设备兼容性"""
        # 测试 CPU
        model = ShannonB1(self.config)
        self.assertEqual(next(model.parameters()).device.type, 'cpu')


class TestShannonB1Encoder(unittest.TestCase):
    """测试编码器版本"""
    
    def setUp(self):
        self.config = ModelConfig(
            vocab_size=1000,
            d_model=64,
            num_heads=4,
            num_layers=2,
            max_seq_len=32
        )
        self.model = ShannonB1Encoder(self.config)
    
    def test_forward_shape(self):
        tokens = torch.randint(0, 1000, (2, 16))
        logits = self.model(tokens)
        self.assertEqual(logits.shape, (2, 16, 1000))


def run_tests():
    """运行所有测试"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestModelConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestLayers))
    suite.addTests(loader.loadTestsFromTestCase(TestShannonB1))
    suite.addTests(loader.loadTestsFromTestCase(TestShannonB1Encoder))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)