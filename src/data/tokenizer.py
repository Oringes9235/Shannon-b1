"""
分词器模块 - 完整版
"""

import json
import re
from collections import defaultdict
from typing import List, Dict, Tuple, Optional


class CharTokenizer:
    """字符级分词器"""
    
    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3,
        }
    
    def build_vocab(self, texts: List[str], vocab_size: int = 1000):
        chars = set()
        for text in texts:
            chars.update(text)
        
        all_chars = list(self.special_tokens.keys()) + sorted(chars)
        
        if len(all_chars) > vocab_size:
            all_chars = all_chars[:vocab_size]
        
        self.char_to_idx = {ch: i for i, ch in enumerate(all_chars)}
        self.idx_to_char = {i: ch for ch, i in self.char_to_idx.items()}
        
        print(f"✅ CharTokenizer: {len(self.char_to_idx)} chars")
        return self
    
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
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
    
    def load(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.char_to_idx = data['char_to_idx']
        self.idx_to_char = {}
        # char_to_idx: mapping from character (str) -> index (int)
        # we need idx_to_char as index (int) -> character (str)
        for ch, idx in self.char_to_idx.items():
            try:
                self.idx_to_char[int(idx)] = ch
            except (ValueError, TypeError):
                # fallback: keep original
                self.idx_to_char[idx] = ch
        self.special_tokens = data.get('special_tokens', self.special_tokens)


class BPETokenizer:
    """BPE分词器"""
    
    def __init__(self, vocab_size: int = 5000):
        self.vocab_size = vocab_size
        self.merges: Dict[Tuple[str, str], int] = {}
        self.vocab: Dict[str, int] = {}
        self.idx_to_token: Dict[int, str] = {}
        
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3,
        }
        self.next_id = len(self.special_tokens)
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]|\s+(?!\S)|\s+""", re.UNICODE)
    
    def train(self, texts: List[str], min_frequency: int = 2, verbose: bool = True):
        if verbose:
            print(f"Training BPE tokenizer (target: {self.vocab_size})")
        
        word_counts = defaultdict(int)
        for text in texts:
            words = self.pat.findall(text)
            for word in words:
                word_with_end = ' '.join(list(word)) + ' </w>'
                word_counts[word_with_end] += 1
        
        chars = set()
        for word in word_counts.keys():
            for ch in word.split():
                if ch != '</w>':
                    chars.add(ch)
        
        for ch in sorted(chars):
            if ch not in self.vocab and ch not in self.special_tokens:
                self.vocab[ch] = self.next_id
                self.next_id += 1
        
        num_merges = min(self.vocab_size - len(self.vocab) - len(self.special_tokens), 2000)
        
        for i in range(num_merges):
            pairs = defaultdict(int)
            for word, count in word_counts.items():
                symbols = word.split()
                for j in range(len(symbols) - 1):
                    pairs[(symbols[j], symbols[j + 1])] += count
            
            if not pairs:
                break
            
            best_pair = max(pairs, key=pairs.get)
            if pairs[best_pair] < min_frequency:
                break
            
            new_token = ''.join(best_pair)
            self.merges[best_pair] = self.next_id
            self.vocab[new_token] = self.next_id
            self.next_id += 1
            
            new_word_counts = {}
            bigram = ' '.join(best_pair)
            merged = new_token
            for word, count in word_counts.items():
                new_word = word.replace(bigram, merged)
                new_word_counts[new_word] = count
            word_counts = new_word_counts
            
            if verbose and (i + 1) % 200 == 0:
                print(f"  Merges: {i+1}/{num_merges}, vocab: {len(self.vocab)}")
        
        self.idx_to_token = {v: k for k, v in self.vocab.items()}
        
        if verbose:
            print(f"✅ BPE: {len(self.vocab)} tokens, {len(self.merges)} merges")
    
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        if not text:
            tokens = []
        else:
            words = self.pat.findall(text)
            tokens = []
            for word in words:
                word_tokens = self._encode_word(word)
                tokens.extend(word_tokens)
        
        if add_bos:
            tokens = [self.special_tokens['<BOS>']] + tokens
        if add_eos:
            tokens = tokens + [self.special_tokens['<EOS>']]
        
        return tokens
    
    def _encode_word(self, word: str) -> List[int]:
        symbols = list(word) + ['</w>']
        
        while len(symbols) > 1:
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
            
            i, pair = min_pair
            symbols[i] = ''.join(pair)
            del symbols[i + 1]
        
        tokens = []
        for s in symbols:
            if s in self.vocab:
                tokens.append(self.vocab[s])
            else:
                for ch in s:
                    if ch in self.vocab:
                        tokens.append(self.vocab[ch])
                    else:
                        tokens.append(self.special_tokens['<UNK>'])
        
        return tokens
    
    def decode(self, tokens: List[int], skip_special: bool = True) -> str:
        chars: List[str] = []
        for t in tokens:
            if t in self.idx_to_token:
                token = self.idx_to_token[t]
                if skip_special and token in self.special_tokens:
                    continue

                # 处理以 '</w>' 结尾的 token（表示单词结尾），例如 'w</w>' -> 'w '
                if isinstance(token, str) and token.endswith('</w>'):
                    body = token[:-4]
                    if body:
                        chars.append(body)
                    chars.append(' ')
                elif token == '</w>':
                    chars.append(' ')
                else:
                    chars.append(token)
            else:
                chars.append('<UNK>')

        text = ''.join(chars)
        # 规范化空白并去掉首尾空格
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def get_vocab_size(self) -> int:
        return len(self.vocab) + len(self.special_tokens)
    
    def get_pad_id(self) -> int:
        return self.special_tokens['<PAD>']
    
    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'vocab': self.vocab,
                'merges': {f"{k[0]}|{k[1]}": v for k, v in self.merges.items()},
                'special_tokens': self.special_tokens,
                'vocab_size': self.vocab_size
            }, f, ensure_ascii=False, indent=2)
    
    def load(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.vocab = data['vocab']
        self.special_tokens = data['special_tokens']
        self.vocab_size = data['vocab_size']
        
        self.merges = {}
        for k, v in data['merges'].items():
            parts = k.split('|')
            self.merges[(parts[0], parts[1])] = v
        
        self.idx_to_token = {v: k for k, v in self.vocab.items()}
        for k, v in self.special_tokens.items():
            self.idx_to_token[v] = k
        
        self.next_id = max(self.vocab.values()) + 1 if self.vocab else len(self.special_tokens)


class SimpleBPETokenizer:
    """简化BPE分词器"""
    
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
        chars = set()
        for text in texts:
            chars.update(text)
        
        all_chars = list(self.special_tokens.keys()) + sorted(chars)
        
        if len(all_chars) > self.vocab_size:
            all_chars = all_chars[:self.vocab_size]
        
        self.char_to_idx = {ch: i for i, ch in enumerate(all_chars)}
        self.idx_to_char = {i: ch for ch, i in self.char_to_idx.items()}
        
        print(f"✅ SimpleBPETokenizer: {len(self.char_to_idx)} chars")
        return self
    
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
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
    
    def load(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.char_to_idx = data['char_to_idx']
        self.idx_to_char = {}
        for ch, idx in self.char_to_idx.items():
            try:
                self.idx_to_char[int(idx)] = ch
            except (ValueError, TypeError):
                self.idx_to_char[idx] = ch
        self.special_tokens = data.get('special_tokens', self.special_tokens)