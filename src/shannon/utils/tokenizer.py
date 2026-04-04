"""
BPE (Byte Pair Encoding) 分词器
参考 GPT-2 的分词实现
"""

import json
import re
from collections import defaultdict
from typing import List, Dict, Tuple, Optional


class BPETokenizer:
    """BPE分词器 - 支持训练和编码/解码"""
    
    def __init__(self, vocab_size: int = 5000):
        self.vocab_size = vocab_size
        self.merges: Dict[Tuple[str, str], int] = {}  # (pair) -> new_token_id
        self.vocab: Dict[str, int] = {}  # token -> id
        self.idx_to_token: Dict[int, str] = {}  # id -> token
        
        # 特殊token
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3,
        }
        self.next_id = len(self.special_tokens)
        
        # 预编译正则表达式 (GPT-2风格)
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    
    def train(self, texts: List[str], min_frequency: int = 2, verbose: bool = True):
        """
        训练BPE分词器
        
        Args:
            texts: 训练文本列表
            min_frequency: 最小合并频率
            verbose: 是否显示进度
        """
        if verbose:
            print(f"开始训练BPE分词器 (目标词表大小: {self.vocab_size})")
        
        # 1. 预处理：将文本分割为单词
        word_counts = defaultdict(int)
        for text in texts:
            # 使用正则分割单词
            words = self.pat.findall(text)
            for word in words:
                # 在词尾添加特殊标记
                word_with_end = ' '.join(list(word)) + ' </w>'
                word_counts[word_with_end] += 1
        
        if verbose:
            print(f"  原始词数: {len(word_counts)}")
        
        # 2. 初始化：每个字符作为一个token
        chars = set()
        for word in word_counts.keys():
            for ch in word.split():
                if ch != '</w>':
                    chars.add(ch)
        
        # 添加所有字符到词表
        for ch in sorted(chars):
            if ch not in self.vocab and ch not in self.special_tokens:
                self.vocab[ch] = self.next_id
                self.next_id += 1
        
        # 3. 迭代合并
        num_merges = min(self.vocab_size - len(self.vocab) - len(self.special_tokens), 2000)
        
        for i in range(num_merges):
            # 统计所有相邻pair的频率
            pairs = defaultdict(int)
            for word, count in word_counts.items():
                symbols = word.split()
                for j in range(len(symbols) - 1):
                    pairs[(symbols[j], symbols[j + 1])] += count
            
            if not pairs:
                break
            
            # 找到最频繁的pair
            best_pair = max(pairs, key=pairs.get)
            if pairs[best_pair] < min_frequency:
                break
            
            # 合并
            new_token = ''.join(best_pair)
            self.merges[best_pair] = self.next_id
            self.vocab[new_token] = self.next_id
            self.next_id += 1
            
            # 更新所有单词
            new_word_counts = {}
            bigram = ' '.join(best_pair)
            merged = new_token
            for word, count in word_counts.items():
                new_word = word.replace(bigram, merged)
                new_word_counts[new_word] = count
            word_counts = new_word_counts
            
            if verbose and (i + 1) % 200 == 0:
                print(f"  合并进度: {i+1}/{num_merges}, 当前词表: {len(self.vocab)}")
        
        # 构建反向映射
        self.idx_to_token = {v: k for k, v in self.vocab.items()}
        
        if verbose:
            print(f"✅ BPE训练完成: {len(self.vocab)} 个token, {len(self.merges)} 次合并")
    
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """
        将文本编码为token序列
        """
        if not text:
            tokens = []
        else:
            # 分词
            words = self.pat.findall(text)
            tokens = []
            
            for word in words:
                # 对每个单词进行BPE编码
                word_tokens = self._encode_word(word)
                tokens.extend(word_tokens)
        
        if add_bos:
            tokens = [self.special_tokens['<BOS>']] + tokens
        if add_eos:
            tokens = tokens + [self.special_tokens['<EOS>']]
        
        return tokens
    
    def _encode_word(self, word: str) -> List[int]:
        """对单个单词进行BPE编码"""
        # 将单词分割为字符
        symbols = list(word) + ['</w>']
        
        # 应用所有合并规则
        while len(symbols) > 1:
            # 找到可以合并的pair
            min_pair = None
            min_idx = float('inf')
            
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                if pair in self.merges:
                    if self.merges[pair] < min_idx:
                        min_idx = self.merges[pair]
                        min_pair = (i, pair)
            
            if min_pair is None:
                break
            
            # 执行合并
            i, pair = min_pair
            symbols[i] = ''.join(pair)
            del symbols[i + 1]
        
        # 转换为token ID
        tokens = []
        for s in symbols:
            if s in self.vocab:
                tokens.append(self.vocab[s])
            else:
                # 未知token，使用字符级编码
                for ch in s:
                    if ch in self.vocab:
                        tokens.append(self.vocab[ch])
                    else:
                        tokens.append(self.special_tokens['<UNK>'])
        
        return tokens
    
    def decode(self, tokens: List[int], skip_special: bool = True) -> str:
        """
        将token序列解码为文本
        """
        chars = []
        for t in tokens:
            if t in self.idx_to_token:
                token = self.idx_to_token[t]
                if skip_special and token in self.special_tokens:
                    continue
                if token == '</w>':
                    chars.append(' ')
                else:
                    chars.append(token)
            else:
                chars.append('<UNK>')
        
        # 合并文本
        text = ''.join(chars)
        # 清理多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def get_vocab_size(self) -> int:
        """获取词表大小"""
        return len(self.vocab) + len(self.special_tokens)
    
    def get_pad_id(self) -> int:
        """获取PAD token ID"""
        return self.special_tokens['<PAD>']
    
    def get_bos_id(self) -> int:
        """获取BOS token ID"""
        return self.special_tokens['<BOS>']
    
    def get_eos_id(self) -> int:
        """获取EOS token ID"""
        return self.special_tokens['<EOS>']
    
    def save(self, path: str):
        """保存分词器"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'vocab': self.vocab,
                'merges': {f"{k[0]}|{k[1]}": v for k, v in self.merges.items()},
                'special_tokens': self.special_tokens,
                'vocab_size': self.vocab_size
            }, f, ensure_ascii=False, indent=2)
        print(f"✅ BPE分词器已保存到: {path}")
    
    def load(self, path: str):
        """加载分词器"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.vocab = data['vocab']
        self.special_tokens = data['special_tokens']
        self.vocab_size = data['vocab_size']
        
        # 恢复merges
        self.merges = {}
        for k, v in data['merges'].items():
            parts = k.split('|')
            self.merges[(parts[0], parts[1])] = v
        
        # 重建反向映射
        self.idx_to_token = {v: k for k, v in self.vocab.items()}
        for k, v in self.special_tokens.items():
            self.idx_to_token[v] = k
        
        self.next_id = max(self.vocab.values()) + 1 if self.vocab else len(self.special_tokens)
        
        print(f"✅ BPE分词器已加载: {len(self.vocab)} 个token")


class SimpleBPETokenizer:
    """简化的BPE分词器 (更轻量，适合快速测试)"""
    
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3,
        }
    
    def build_vocab(self, texts: List[str]):
        """构建字符级词表 (简化版)"""
        chars = set()
        for text in texts:
            chars.update(text)
        
        # 添加特殊token
        all_chars = list(self.special_tokens.keys()) + sorted(chars)
        
        if len(all_chars) > self.vocab_size:
            all_chars = all_chars[:self.vocab_size]
        
        self.char_to_idx = {ch: i for i, ch in enumerate(all_chars)}
        self.idx_to_char = {i: ch for ch, i in self.char_to_idx.items()}
        
        print(f"✅ 词表构建完成: {len(self.char_to_idx)} 个字符")
        return self
    
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
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
    
    def decode(self, tokens: List[int], skip_special: bool = True) -> str:
        """token序列转文本"""
        chars = []
        for t in tokens:
            ch = self.idx_to_char.get(t, '<UNK>')
            if skip_special and ch in self.special_tokens:
                continue
            chars.append(ch)
        return ''.join(chars)
    
    def get_vocab_size(self) -> int:
        return len(self.char_to_idx)
    
    def get_pad_id(self) -> int:
        return self.char_to_idx['<PAD>']
    
    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'char_to_idx': self.char_to_idx,
                'special_tokens': self.special_tokens
            }, f, ensure_ascii=False, indent=2)
        print(f"✅ 分词器已保存到: {path}")
    
    def load(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.char_to_idx = data['char_to_idx']
        self.idx_to_char = {int(k): v for k, v in self.char_to_idx.items()}
        self.special_tokens = data['special_tokens']
        print(f"✅ 分词器已加载: {len(self.char_to_idx)} 个字符")