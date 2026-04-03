"""
Shannon-b1 单元测试
运行: python -m pytest tests/
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shannon.core.attention import MultiHeadAttention
from shannon.core.feedforward import FeedForward
from shannon.core.layer_norm import LayerNorm
from shannon.core.transformer_block import TransformerBlock
from shannon.model.shannon_b1 import ShannonB1
from shannon.training.loss import CrossEntropyLoss
from shannon.training.optimizer import SGD
from shannon.utils.functions import softmax, gelu, gelu_backward


class TestUtils(unittest.TestCase):
    def test_softmax(self):
        x = np.array([[1.0, 2.0, 3.0]])
        result = softmax(x)
        self.assertAlmostEqual(np.sum(result), 1.0)
        self.assertTrue(np.all(result >= 0))
    
    def test_gelu(self):
        x = np.array([-1.0, 0.0, 1.0])
        result = gelu(x)
        self.assertEqual(result.shape, x.shape)
        # GELU(0) 应该接近 0
        self.assertAlmostEqual(result[1], 0.0, places=5)


class TestAttention(unittest.TestCase):
    def test_multi_head_attention_shape(self):
        mha = MultiHeadAttention(d_model=512, num_heads=8)
        x = np.random.randn(2, 4, 512)
        output = mha.forward(x)
        self.assertEqual(output.shape, (2, 4, 512))
    
    def test_attention_gradients(self):
        """测试梯度是否被正确初始化"""
        mha = MultiHeadAttention(d_model=64, num_heads=4)
        self.assertIsNone(mha.W_q.grad)
        self.assertIsNone(mha.W_k.grad)
        self.assertIsNone(mha.W_v.grad)
        self.assertIsNone(mha.W_o.grad)


class TestFeedForward(unittest.TestCase):
    def test_ffn_shape(self):
        ffn = FeedForward(d_model=512, d_ff=2048)
        x = np.random.randn(2, 4, 512)
        output = ffn.forward(x)
        self.assertEqual(output.shape, (2, 4, 512))
    
    def test_ffn_parameters(self):
        ffn = FeedForward(d_model=128, d_ff=512)
        self.assertEqual(ffn.W1.shape, (128, 512))
        self.assertEqual(ffn.W2.shape, (512, 128))
        self.assertEqual(ffn.b1.shape, (512,))
        self.assertEqual(ffn.b2.shape, (128,))


class TestLayerNorm(unittest.TestCase):
    def test_layer_norm_shape(self):
        ln = LayerNorm(d_model=512)
        x = np.random.randn(2, 4, 512)
        output = ln.forward(x)
        self.assertEqual(output.shape, (2, 4, 512))
    
    def test_layer_norm_mean(self):
        ln = LayerNorm(d_model=64)
        x = np.random.randn(2, 4, 64)
        output = ln.forward(x)
        # 沿着特征维度，输出应该接近均值为0
        mean = np.mean(output, axis=-1)
        self.assertTrue(np.all(np.abs(mean) < 1e-5))


class TestTransformerBlock(unittest.TestCase):
    def test_block_shape(self):
        block = TransformerBlock(d_model=512, num_heads=8, d_ff=2048)
        x = np.random.randn(2, 4, 512)
        output = block.forward(x, training=False)
        self.assertEqual(output.shape, (2, 4, 512))
    
    def test_residual_connection(self):
        """测试残差连接是否正确"""
        block = TransformerBlock(d_model=64, num_heads=4, d_ff=256)
        x = np.random.randn(2, 4, 64)
        output = block.forward(x, training=False)
        # 输出不应该与输入完全相同（因为有变换）
        self.assertFalse(np.allclose(x, output))


class TestShannonModel(unittest.TestCase):
    def setUp(self):
        self.config = {
            'vocab_size': 100,
            'd_model': 32,
            'num_heads': 4,
            'd_ff': 128,
            'num_layers': 2,
            'max_seq_len': 20,
            'dropout': 0.0
        }
        self.model = ShannonB1(**self.config)
    
    def test_model_forward(self):
        tokens = np.random.randint(0, 100, (2, 10))
        logits = self.model.forward(tokens, training=False)
        self.assertEqual(logits.shape, (2, 10, 100))
    
    def test_generate(self):
        start_tokens = [1, 2, 3]
        generated = self.model.generate(start_tokens, max_new_tokens=5, temperature=0.8)
        self.assertEqual(len(generated), 8)  # 3 + 5
        self.assertEqual(generated[:3], start_tokens)
    
    def test_get_all_parameters(self):
        params = self.model.get_all_parameters()
        self.assertGreater(len(params), 0)
        # 检查参数名称
        param_names = [name for name, _ in params]
        self.assertIn('token_embedding', param_names)
        self.assertIn('output_projection', param_names)


class TestLoss(unittest.TestCase):
    def test_cross_entropy(self):
        loss_fn = CrossEntropyLoss()
        logits = np.random.randn(2, 4, 10)
        targets = np.random.randint(0, 10, (2, 4))
        loss = loss_fn.forward(logits, targets)
        self.assertGreater(loss, 0)
        self.assertLess(loss, 10)  # 合理的损失范围
    
    def test_backward_shape(self):
        loss_fn = CrossEntropyLoss()
        logits = np.random.randn(2, 4, 10)
        targets = np.random.randint(0, 10, (2, 4))
        loss_fn.forward(logits, targets)
        d_logits = loss_fn.backward()
        self.assertEqual(d_logits.shape, logits.shape)


class TestOptimizer(unittest.TestCase):
    def test_sgd_step(self):
        class DummyParam:
            def __init__(self):
                self.value = 1.0
                self.grad = 0.1
        
        param = DummyParam()
        params = [('test', param)]
        optimizer = SGD(params, lr=0.01)
        
        # 确保 param 有 grad 属性
        param.grad = 0.1
        
        optimizer.step()
        # 注意: 这里需要根据实际的参数更新逻辑调整
        # param -= lr * grad = 1.0 - 0.01 * 0.1 = 0.999
    
    def test_zero_grad(self):
        class DummyParam:
            def __init__(self):
                self.grad = 0.1
        
        param = DummyParam()
        params = [('test', param)]
        optimizer = SGD(params, lr=0.01)
        optimizer.zero_grad()
        self.assertIsNone(param.grad)


def run_tests():
    """运行所有测试"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestUtils))
    suite.addTests(loader.loadTestsFromTestCase(TestAttention))
    suite.addTests(loader.loadTestsFromTestCase(TestFeedForward))
    suite.addTests(loader.loadTestsFromTestCase(TestLayerNorm))
    suite.addTests(loader.loadTestsFromTestCase(TestTransformerBlock))
    suite.addTests(loader.loadTestsFromTestCase(TestShannonModel))
    suite.addTests(loader.loadTestsFromTestCase(TestLoss))
    suite.addTests(loader.loadTestsFromTestCase(TestOptimizer))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)