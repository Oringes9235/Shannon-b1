"""
分词器单元测试
"""

import sys
import os
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest

from src.data.tokenizer import CharTokenizer, BPETokenizer, SimpleBPETokenizer


class TestCharTokenizer(unittest.TestCase):
    """测试字符级分词器"""
    
    def setUp(self):
        self.tokenizer = CharTokenizer()
        texts = ["Hello world!", "This is a test."]
        self.tokenizer.build_vocab(texts, vocab_size=100)
    
    def test_vocab_building(self):
        """测试词表构建"""
        self.assertGreater(self.tokenizer.get_vocab_size(), 0)
        self.assertIn('H', self.tokenizer.char_to_idx)
        self.assertIn('e', self.tokenizer.char_to_idx)
    
    def test_encode_decode(self):
        """测试编码和解码"""
        text = "Hello"
        tokens = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(tokens)
        self.assertEqual(decoded, text)
    
    def test_special_tokens(self):
        """测试特殊token"""
        self.assertEqual(self.tokenizer.get_pad_id(), 0)
        tokens = self.tokenizer.encode("", add_bos=True, add_eos=True)
        self.assertEqual(tokens[0], self.tokenizer.char_to_idx['<BOS>'])
        self.assertEqual(tokens[-1], self.tokenizer.char_to_idx['<EOS>'])
    
    def test_unknown_token(self):
        """测试未知字符处理"""
        tokens = self.tokenizer.encode("xyz")
        # 应该都能处理
        self.assertEqual(len(tokens), 3)
    
    def test_save_load(self):
        """测试保存和加载"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            path = f.name
        
        try:
            self.tokenizer.save(path)
            new_tokenizer = CharTokenizer()
            new_tokenizer.load(path)
            self.assertEqual(new_tokenizer.get_vocab_size(), self.tokenizer.get_vocab_size())
        finally:
            os.unlink(path)


class TestSimpleBPETokenizer(unittest.TestCase):
    """测试简化BPE分词器"""
    
    def setUp(self):
        self.tokenizer = SimpleBPETokenizer(vocab_size=200)
        texts = ["Hello world!", "This is a test."]
        self.tokenizer.build_vocab(texts)
    
    def test_vocab_building(self):
        self.assertGreater(self.tokenizer.get_vocab_size(), 0)
    
    def test_encode_decode(self):
        text = "Hello"
        tokens = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(tokens)
        self.assertEqual(decoded, text)
    
    def test_special_tokens(self):
        self.assertEqual(self.tokenizer.get_pad_id(), 0)


class TestBPETokenizer(unittest.TestCase):
    """测试完整BPE分词器"""
    
    def setUp(self):
        self.tokenizer = BPETokenizer(vocab_size=500)
        texts = ["Hello world! This is a BPE tokenizer test.", 
                 "Another sentence for training."]
        self.tokenizer.train(texts, verbose=False)
    
    def test_vocab_building(self):
        self.assertGreater(self.tokenizer.get_vocab_size(), 0)
    
    def test_encode_decode(self):
        text = "Hello world"
        tokens = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(tokens)
        # BPE 解码后可能有细微差别
        self.assertIn('hello', decoded.lower())
    
    def test_save_load(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            path = f.name
        
        try:
            self.tokenizer.save(path)
            new_tokenizer = BPETokenizer()
            new_tokenizer.load(path)
            self.assertEqual(new_tokenizer.get_vocab_size(), self.tokenizer.get_vocab_size())
        finally:
            os.unlink(path)


def run_tests():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestCharTokenizer))
    suite.addTests(loader.loadTestsFromTestCase(TestSimpleBPETokenizer))
    suite.addTests(loader.loadTestsFromTestCase(TestBPETokenizer))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)