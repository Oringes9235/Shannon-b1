"""
模型管理器 - 加载和管理 Shannon-b1 模型
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import json
from typing import Optional, Dict, Any


class ModelManager:
    """模型管理器"""
    
    def __init__(self):
        """初始化模型管理器"""
        self.model = None
        self.tokenizer = None
        self.config = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def is_loaded(self) -> bool:
        """
        检查模型是否已加载
        
        Returns:
            bool: 模型是否已加载
        """
        return self.model is not None
    
    def get_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            Dict[str, Any]: 包含模型详细信息的字典
        """
        if not self.model:
            return {"loaded": False}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        return {
            "loaded": True,
            "vocab_size": self.config.vocab_size,
            "d_model": self.config.d_model,
            "num_layers": self.config.num_layers,
            "num_heads": self.config.num_heads,
            "parameters": total_params,
            "size_mb": total_params * 4 / 1024 / 1024,
            "device": str(self.device)
        }
    
    def load_model(self, model_path: str) -> bool:
        """
        加载模型
        
        Args:
            model_path (str): 模型文件路径
            
        Returns:
            bool: 加载是否成功
        """
        try:
            from src.model import ShannonB1, ModelConfig
            from src.data import CharTokenizer, BPETokenizer
            
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # 获取配置
            if 'config' in checkpoint:
                self.config = checkpoint['config']
            else:
                state_dict = checkpoint['model_state_dict']
                vocab_size = state_dict['token_embedding.weight'].shape[0]
                d_model = state_dict['token_embedding.weight'].shape[1]
                max_seq_len = state_dict['pos_encoding.pe'].shape[1]
                from src.model.config import ModelConfig
                self.config = ModelConfig(
                    vocab_size=vocab_size,
                    d_model=d_model,
                    max_seq_len=max_seq_len
                )
            
            # 创建模型
            self.model = ShannonB1(self.config).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # 加载分词器
            tokenizer_path = model_path.replace('.pt', '_tokenizer.json')
            if os.path.exists(tokenizer_path):
                with open(tokenizer_path, 'r') as f:
                    data = json.load(f)
                if 'char_to_idx' in data:
                    from src.data import CharTokenizer
                    self.tokenizer = CharTokenizer()
                else:
                    from src.data import BPETokenizer
                    self.tokenizer = BPETokenizer()
                self.tokenizer.load(tokenizer_path)
            else:
                from src.data import CharTokenizer
                self.tokenizer = CharTokenizer()
                self.tokenizer.build_vocab(["sample"], 200)
            
            print(f"Model loaded: {self.config.vocab_size} vocab, {self.config.d_model} dim")
            return True
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False
    
    def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.8,
                 top_k: int = 40, top_p: float = 0.9, repetition_penalty: float = 1.15) -> Dict[str, Any]:
        """
        生成文本
        
        Args:
            prompt (str): 提示文本
            max_tokens (int): 最大生成token数，默认100
            temperature (float): 温度参数，默认0.8
            top_k (int): Top-k采样参数，默认40
            top_p (float): Top-p采样参数，默认0.9
            repetition_penalty (float): 重复惩罚参数，默认1.15
            
        Returns:
            Dict[str, Any]: 包含生成结果的字典
        """
        if not self.model:
            raise ValueError("No model loaded")
        
        # 编码提示词
        start_tokens = self.tokenizer.encode(prompt)[:50]
        
        # 生成
        with torch.no_grad():
            generated = self.model.generate(
                start_tokens,
                max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty
            )
        
        # 解码
        text = self.tokenizer.decode(generated)
        text = text.replace('</w>', ' ').replace('  ', ' ').strip()
        
        return {
            "prompt": prompt,
            "generated_text": text,
            "tokens_generated": len(generated) - len(start_tokens),
            "temperature": temperature
        }