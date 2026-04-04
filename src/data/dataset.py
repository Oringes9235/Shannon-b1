"""
数据集加载和处理模块
"""

import numpy as np
import os
import urllib.request
from typing import List, Tuple, Optional


class TextDataset:
    """文本数据集"""
    
    def __init__(self, tokenizer, seq_length=128, stride=None):
        """
        tokenizer: 分词器
        seq_length: 序列长度
        stride: 滑动窗口步长 (默认为 seq_length // 2)
        """
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.stride = stride or seq_length // 2
        self.data = []
        self.vocab_size = tokenizer.get_vocab_size()
    
    def load_text(self, text: str, name: str = "text") -> 'TextDataset':
        """从文本字符串加载数据"""
        tokens = self.tokenizer.encode(text)
        self._create_sequences(tokens)
        print(f"✅ 加载 {name}: {len(self.data)} 个序列")
        return self
    
    def load_file(self, filepath: str) -> 'TextDataset':
        """从文件加载文本"""
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        return self.load_text(text, filepath)
    
    def load_shakespeare(self, download=True) -> 'TextDataset':
        """加载莎士比亚文本"""
        if download:
            url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            try:
                with urllib.request.urlopen(url) as response:
                    text = response.read().decode('utf-8')
                print("✅ 下载莎士比亚文本成功")
                return self.load_text(text, "Shakespeare")
            except Exception as e:
                print(f"⚠️ 下载失败: {e}")
        
        # 备用：使用内置示例文本
        text = """
        To be, or not to be, that is the question:
        Whether 'tis nobler in the mind to suffer
        The slings and arrows of outrageous fortune,
        Or to take arms against a sea of troubles
        And by opposing end them. To die: to sleep;
        No more; and by a sleep to say we end
        The heart-ache and the thousand natural shocks
        That flesh is heir to, 'tis a consummation
        Devoutly to be wish'd. To die, to sleep;
        To sleep: perchance to dream: ay, there's the rub;
        For in that sleep of death what dreams may come
        """
        return self.load_text(text, "Shakespeare (sample)")
    
    def load_wikipedia(self, sample_size=50000) -> 'TextDataset':
        """加载维基百科样本"""
        # 简化的维基百科示例文本
        text = """
        Artificial intelligence (AI) is intelligence demonstrated by machines, 
        as opposed to natural intelligence displayed by animals including humans. 
        Leading AI textbooks define the field as the study of "intelligent agents": 
        any device that perceives its environment and takes actions that maximize 
        its chance of successfully achieving its goals. Colloquially, the term 
        "artificial intelligence" is often used to describe machines that mimic 
        "cognitive" functions that humans associate with other human minds, such 
        as "learning" and "problem solving".
        """
        text = text * (sample_size // len(text) + 1)
        return self.load_text(text[:sample_size], "Wikipedia")
    
    def _create_sequences(self, tokens: List[int]) -> None:
        """创建训练序列"""
        self.data = []
        for i in range(0, len(tokens) - self.seq_length - 1, self.stride):
            seq = tokens[i:i + self.seq_length + 1]
            if len(seq) == self.seq_length + 1:
                self.data.append(seq)
    
    def get_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """获取一个批次的数据"""
        indices = np.random.choice(len(self.data), batch_size, replace=True)
        
        max_len = self.seq_length
        inputs = np.zeros((batch_size, max_len), dtype=np.int32)
        targets = np.zeros((batch_size, max_len), dtype=np.int32)
        
        for i, idx in enumerate(indices):
            seq = self.data[idx]
            inputs[i, :len(seq)-1] = seq[:-1]
            targets[i, :len(seq)-1] = seq[1:]
        
        return inputs, targets
    
    def get_all_sequences(self) -> List[List[int]]:
        """获取所有序列 (用于简单训练循环)"""
        return [seq[:-1] for seq in self.data]
    
    def get_all_targets(self) -> List[List[int]]:
        """获取所有目标"""
        return [seq[1:] for seq in self.data]
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        """获取单个样本"""
        seq = self.data[idx]
        return seq[:-1], seq[1:]


def load_training_data(args, tokenizer):
    """
    根据配置加载训练数据
    """
    dataset = TextDataset(tokenizer, args.seq_len)
    
    if args.dataset == 'shakespeare':
        dataset.load_shakespeare()
    elif args.dataset == 'wikipedia':
        dataset.load_wikipedia()
    elif args.dataset == 'file' and args.data_path:
        dataset.load_file(args.data_path)
    else:
        # 使用随机数据作为备用
        print("⚠️ 使用随机数据训练 (效果不佳，建议使用真实数据)")
        return [np.random.randint(0, tokenizer.get_vocab_size(), size=args.seq_length).tolist() 
                for _ in range(args.train_size)]
    
    # 返回序列列表格式，兼容现有训练器
    return dataset.get_all_sequences()