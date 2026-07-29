"""
分词器模块 - 完整版
使用 HuggingFace tokenizers (Rust 实现) 进行高速 BPE 训练和编码
"""

import json
import re
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.decoders import BPEDecoder as HFBPEDecoder


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
        for ch, idx in self.char_to_idx.items():
            try:
                self.idx_to_char[int(idx)] = ch
            except (ValueError, TypeError):
                self.idx_to_char[idx] = ch
        self.special_tokens = data.get('special_tokens', self.special_tokens)


class BPETokenizer:
    """
    BPE分词器 — 双模式实现
    
    - 训练模式：使用 HuggingFace tokenizers (Rust)，速度极快
    - 回退模式：加载旧版 checkpoint tokenizer 时使用 Python BPE 逻辑确保 ID 一致
    """
    
    def __init__(self, vocab_size: int = 5000):
        self.vocab_size = vocab_size
        
        # 创建底层 HuggingFace Rust tokenizer (用于训练)
        self._hf = HFTokenizer(BPE(unk_token="<UNK>"))
        self._hf.pre_tokenizer = Whitespace()
        self._hf.decoder = HFBPEDecoder()
        
        # 特殊 token 映射
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3,
        }
        
        # 旧版数据（用于回退模式）
        self._legacy_mode = False
        self._legacy_vocab: Dict[str, int] = {}
        self._legacy_idx_to_token: Dict[int, str] = {}
        self._legacy_merges: Dict[Tuple[str, str], int] = {}
        
        # 兼容外部访问
        self.merges: Dict[Tuple[str, str], int] = {}
        self.vocab: Dict[str, int] = {}
        self.idx_to_token: Dict[int, str] = {}
    
    def _rebuild_mappings(self):
        """从底层 tokenizer 重建所有内部映射"""
        vocab = self._hf.get_vocab()
        for token in ['<PAD>', '<UNK>', '<BOS>', '<EOS>']:
            if token in vocab:
                self.special_tokens[token] = vocab[token]
        self.vocab = dict(vocab)
        self.idx_to_token = {v: k for k, v in vocab.items()}
    
    def train(self, texts: List[str], min_frequency: int = 2, verbose: bool = True):
        if verbose:
            print(f"Training BPE tokenizer (target: {self.vocab_size})")
        
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=min_frequency,
            special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"],
            show_progress=verbose,
            continuing_subword_prefix="",
            end_of_word_suffix="</w>",
        )
        
        self._hf.train_from_iterator(texts, trainer=trainer)
        self._legacy_mode = False
        self._rebuild_mappings()
        
        if verbose:
            real_vocab = self._hf.get_vocab_size()
            print(f"✅ BPE: {real_vocab} tokens")
    
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        if not text:
            return []
        
        if self._legacy_mode:
            token_ids = self._legacy_encode(text)
        else:
            enc = self._hf.encode(text)
            token_ids = enc.ids
        
        if add_bos:
            token_ids = [self.special_tokens['<BOS>']] + token_ids
        if add_eos:
            token_ids = token_ids + [self.special_tokens['<EOS>']]
        
        return token_ids
    
    def _legacy_encode(self, text: str) -> List[int]:
        """使用旧版 merge 规则编码，确保 token ID 与 checkpoint 一致"""
        # 先用预分词器切词 (与旧版 pat 逻辑一致)
        words = self._legacy_pat.findall(text) if hasattr(self, '_legacy_pat') else [text]
        
        token_ids = []
        for word in words:
            symbols = list(word) + ['</w>']
            
            # BPE 合并 (按 merge_id 优先级)
            while len(symbols) > 1:
                min_pair = None
                min_idx = float('inf')
                for i in range(len(symbols) - 1):
                    pair = (symbols[i], symbols[i + 1])
                    if pair in self._legacy_merges:
                        if self._legacy_merges[pair] < min_idx:
                            min_idx = self._legacy_merges[pair]
                            min_pair = (i, pair)
                
                if min_pair is None:
                    break
                
                i, pair = min_pair
                symbols[i] = ''.join(pair)
                del symbols[i + 1]
            
            # 查表转 ID
            for s in symbols:
                if s in self._legacy_vocab:
                    token_ids.append(self._legacy_vocab[s])
                else:
                    for ch in s:
                        token_ids.append(self._legacy_vocab.get(ch, self.special_tokens['<UNK>']))
        
        return token_ids
    
    # def decode(self, tokens: List[int], skip_special: bool = True) -> str:
    #     if skip_special:
    #         skip_ids = {v for k, v in self.special_tokens.items() if k in ['<PAD>', '<BOS>', '<EOS>']}
    #         tokens = [t for t in tokens if t not in skip_ids]
        
    #     if self._legacy_mode:
    #         return self._legacy_decode(tokens)
    #     else:
    #         return self._hf.decode(tokens)
    
    def decode(self, tokens: List[int], skip_special: bool = True) -> str:
        """
        将token序列解码为文本（修复词间粘连）
        """
        words: List[str] = []
        current_word: List[str] = []
        
        for t in tokens:
            if t in self.idx_to_token:
                token = self.idx_to_token[t]
                if skip_special and token in self.special_tokens:
                    continue
                
                # 处理以 '</w>' 结尾的 token（表示单词结尾）
                if isinstance(token, str) and token.endswith('</w>'):
                    body = token[:-4]  # 去掉 </w>
                    if body:
                        current_word.append(body)
                    # 单词结束，拼起来加空格
                    words.append(''.join(current_word))
                    current_word = []
                elif token == '</w>':
                    # 单独的单词结束符
                    words.append(''.join(current_word))
                    current_word = []
                else:
                    # 非结束符，追加到当前单词
                    current_word.append(token)
            else:
                current_word.append('<UNK>')
        
        # 处理最后一个不完整的单词
        if current_word:
            words.append(''.join(current_word))
        
        # 拼接并清理空白
        text = ' '.join(words)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _legacy_decode(self, tokens: List[int]) -> str:
        """旧版解码"""
        chars = []
        for t in tokens:
            token = self._legacy_idx_to_token.get(t, '<UNK>')
            if isinstance(token, str) and token.endswith('</w>'):
                body = token[:-4]
                if body:
                    chars.append(body)
                chars.append(' ')
            elif token == '</w>':
                chars.append(' ')
            else:
                chars.append(token)
        
        text = ''.join(chars)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def get_vocab_size(self) -> int:
        if self._legacy_mode:
            return len(self._legacy_vocab) + len(self.special_tokens)
        return self._hf.get_vocab_size()
    
    def get_pad_id(self) -> int:
        return self.special_tokens.get('<PAD>', 0)
    
    def save(self, path: str):
        """保存分词器为 HuggingFace tokenizer 格式"""
        self._hf.save(path)
    
    def load(self, path: str):
        """
        加载分词器，支持两种格式:
        1. HuggingFace tokenizer JSON (新格式，有 "model" 键)
        2. 旧版自定义 JSON (有 "vocab" + "merges" 键) → 回退模式
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if "model" in data:
            # HF 格式
            self._hf = HFTokenizer.from_file(path)
            self._legacy_mode = False
            self._rebuild_mappings()
            
        elif "vocab" in data and "merges" in data:
            # 旧版自定义格式 → 回退模式，确保 token ID 一致
            self._load_legacy_format(data)
            
        else:
            raise ValueError(f"Unknown tokenizer format in {path}. Keys: {list(data.keys())}")
    
    def _load_legacy_format(self, data: dict):
        """
        加载旧版自定义 JSON 格式
        
        {
            "vocab": {"a": 4, "b": 5, ...},
            "merges": {"a|b": 200, ...},
            "special_tokens": {"<PAD>": 0, ...},
            "vocab_size": 6494
        }
        """
        self._legacy_mode = True
        
        # 词表
        self._legacy_vocab = data["vocab"]
        self._legacy_idx_to_token = {int(v): k for k, v in self._legacy_vocab.items()}
        
        # 合并规则
        self._legacy_merges = {}
        for k, v in data["merges"].items():
            parts = k.split('|')
            self._legacy_merges[(parts[0], parts[1])] = v
        
        # 特殊 token
        self.special_tokens = data.get("special_tokens", {
            '<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3,
        })
        
        # 同步外部访问
        self.vocab = self._legacy_vocab
        self.idx_to_token = self._legacy_idx_to_token
        self.merges = self._legacy_merges
        
        # 构建与旧版相同的预分词器
        self._legacy_pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]|\s+(?!\S)|\s+""",
            re.UNICODE
        )
        
        # 用合成数据初始化 _hf (用于非 encode/decode 的内部调用)
        single_chars = sorted({
            t for t in self._legacy_vocab
            if len(t) == 1 and t not in ('<PAD>', '<UNK>', '<BOS>', '<EOS>', '</w>')
        })
        synthetic_text = ' '.join(single_chars) if single_chars else "a b c"
        
        trainer = BpeTrainer(
            vocab_size=len(self._legacy_vocab) + 4,
            min_frequency=1,
            special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"],
            show_progress=False,
            continuing_subword_prefix="",
            end_of_word_suffix="</w>",
        )
        self._hf.train_from_iterator([synthetic_text], trainer=trainer)


# SimpleBPETokenizer 保持原有实现不变
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