"""
分词器模块 - 支持字符级和BPE分词
"""

import re
import json
from collections import defaultdict


class CharTokenizer:
    """字符级分词器 (简单高效，适合入门)"""
    
    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3,
        }
    
    def build_vocab(self, texts, vocab_size=1000):
        """从文本构建词表"""
        chars = set()
        for text in texts:
            chars.update(text)
        
        # 添加特殊token
        all_chars = list(self.special_tokens.keys()) + sorted(chars)
        
        # 限制词表大小
        if len(all_chars) > vocab_size:
            all_chars = all_chars[:vocab_size]
        
        self.char_to_idx = {ch: i for i, ch in enumerate(all_chars)}
        self.idx_to_char = {i: ch for ch, i in self.char_to_idx.items()}
        
        print(f"✅ 词表构建完成: {len(self.char_to_idx)} 个字符")
        return self
    
    def encode(self, text, add_bos=False, add_eos=False):
        """文本转token序列"""
        tokens = []
        for ch in text:
            if ch in self.char_to_idx:
                tokens.append(self.char_to_idx[ch])
            else:
                tokens.append(self.char_to_idx['<UNK>'])
        
        if add_bos:
            tokens = [self.char_to_idx['<BOS>']] + tokens
        if add_eos:
            tokens = tokens + [self.char_to_idx['<EOS>']]
        
        return tokens
    
    def decode(self, tokens, skip_special=True):
        """token序列转文本"""
        chars = []
        for t in tokens:
            ch = self.idx_to_char.get(t, '<UNK>')
            if skip_special and ch in self.special_tokens:
                continue
            chars.append(ch)
        return ''.join(chars)
    
    def get_vocab_size(self):
        return len(self.char_to_idx)
    
    def get_pad_id(self):
        return self.char_to_idx['<PAD>']
    
    def save(self, path):
        """保存词表"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'char_to_idx': self.char_to_idx,
                'special_tokens': self.special_tokens
            }, f, ensure_ascii=False, indent=2)
        print(f"✅ 词表已保存到: {path}")
    
    def load(self, path):
        """加载词表"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.char_to_idx = data['char_to_idx']
        self.idx_to_char = {int(k): v for k, v in self.char_to_idx.items()}
        self.special_tokens = data['special_tokens']
        print(f"✅ 词表已加载: {len(self.char_to_idx)} 个字符")


class BPETokenizer:
    """BPE分词器 (更高效，适合生产环境)"""
    
    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size
        self.merges = {}  # (pair, new_token)
        self.vocab = {}
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3,
        }
        self.next_id = len(self.special_tokens)
    
    def _get_stats(self, word_counts):
        """统计相邻字符对频率"""
        pairs = defaultdict(int)
        for word, count in word_counts.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += count
        return pairs
    
    def _merge_vocab(self, pair, word_counts):
        """合并最频繁的字符对"""
        new_word_counts = {}
        bigram = ' '.join(pair)
        merged = ''.join(pair)
        for word, count in word_counts.items():
            new_word = word.replace(bigram, merged)
            new_word_counts[new_word] = count
        return new_word_counts
    
    def train(self, texts, min_frequency=2):
        """训练BPE"""
        print("开始训练BPE分词器...")
        
        # 初始化：每个字符作为一个token
        word_counts = {}
        for text in texts:
            # 在字符间添加空格，并在词尾添加特殊标记
            words = [' '.join(list(word)) + ' </w>' for word in text.split()]
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # 计算需要合并的次数
        unique_chars = set(''.join(texts))
        num_merges = min(self.vocab_size - len(self.special_tokens) - len(unique_chars), 1000)
        
        for i in range(num_merges):
            pairs = self._get_stats(word_counts)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            if pairs[best_pair] < min_frequency:
                break
            
            word_counts = self._merge_vocab(best_pair, word_counts)
            self.merges[best_pair] = self.next_id
            self.next_id += 1
            
            if (i + 1) % 100 == 0:
                print(f"  合并进度: {i+1}/{num_merges}")
        
        self._build_vocab()
        print(f"✅ BPE训练完成: {len(self.vocab)} 个token")
    
    def _build_vocab(self):
        """构建词表"""
        self.vocab = {**self.special_tokens}
        
        # 添加所有字符
        chars = set()
        for pair in self.merges.keys():
            chars.update(pair)
        for ch in chars:
            if ch not in self.vocab and ch != ' ' and ch != '</w>':
                self.vocab[ch] = len(self.vocab)
        
        # 添加合并后的token
        for pair, idx in self.merges.items():
            token = ''.join(pair)
            if token not in self.vocab:
                self.vocab[token] = idx
    
    def encode(self, text):
        """文本转token序列 (简化版)"""
        # 简化实现：返回字符级token
        tokens = []
        for ch in text:
            if ch in self.vocab:
                tokens.append(self.vocab[ch])
            else:
                tokens.append(self.vocab['<UNK>'])
        return tokens
    
    def decode(self, tokens):
        """token转文本"""
        idx_to_char = {v: k for k, v in self.vocab.items()}
        chars = []
        for t in tokens:
            ch = idx_to_char.get(t, '<UNK>')
            if ch not in self.special_tokens:
                chars.append(ch)
        return ''.join(chars)
    
    def get_vocab_size(self):
        return len(self.vocab)
    
    def get_pad_id(self):
        return self.vocab['<PAD>']