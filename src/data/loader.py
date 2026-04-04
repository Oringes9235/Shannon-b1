"""
数据加载器
"""

import numpy as np
import os


class TextDataset:
    """文本数据集"""
    
    def __init__(self, tokenizer, seq_length=128):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.data = []
    
    def load_text_file(self, filepath):
        """加载文本文件"""
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # 分词
        tokens = self.tokenizer.encode(text)
        
        # 创建序列
        self.data = []
        for i in range(0, len(tokens) - self.seq_length - 1, self.seq_length // 2):
            seq = tokens[i:i + self.seq_length + 1]
            self.data.append(seq)
        
        print(f"加载 {filepath}: {len(self.data)} 个序列")
        return self
    
    def load_shakespeare(self, seq_length=128):
        """加载莎士比亚文本 (示例)"""
        # 下载莎士比亚文本
        import urllib.request
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        
        try:
            with urllib.request.urlopen(url) as response:
                text = response.read().decode('utf-8')
        except:
            # 如果下载失败，使用本地文件或简单文本
            text = "Hello world. This is a simple text for training."
        
        tokens = self.tokenizer.encode(text)
        
        self.data = []
        for i in range(0, len(tokens) - seq_length - 1, seq_length // 2):
            seq = tokens[i:i + seq_length + 1]
            self.data.append(seq)
        
        print(f"加载莎士比亚文本: {len(self.data)} 个序列")
        return self
    
    def get_batch(self, batch_size):
        """获取一个批次"""
        indices = np.random.choice(len(self.data), batch_size)
        
        batch = [self.data[i] for i in indices]
        max_len = max(len(seq) for seq in batch)
        
        inputs = []
        targets = []
        for seq in batch:
            if len(seq) > max_len:
                seq = seq[:max_len]
            inputs.append(seq[:-1] + [0] * (max_len - len(seq)))
            targets.append(seq[1:] + [0] * (max_len - len(seq)))
        
        return np.array(inputs), np.array(targets)
    
    def __len__(self):
        return len(self.data)