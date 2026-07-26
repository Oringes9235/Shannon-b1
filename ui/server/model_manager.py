"""
模型管理器 - 加载和管理 Shannon-b1 模型
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import json
from typing import Optional, Dict, Any, Generator, List
from src.utils import Conversation, SIMPLE_TEMPLATE, get_template_by_name


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

            # 加载模型权重，兼容旧格式 LoRA checkpoint
            ckpt_state = checkpoint['model_state_dict']
            if self._is_lora_checkpoint(ckpt_state):
                ckpt_state = self._remap_lora_to_standard(ckpt_state)
                print("[Info] Detected LoRA-format checkpoint, remapped to standard keys")

            self.model.load_state_dict(ckpt_state)
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
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None, max_tokens: int = 100, temperature: float = 0.8,
                 top_k: int = 40, top_p: float = 0.9, repetition_penalty: float = 1.15,
                 conversation: Optional[Conversation] = None,
                 conv_template: str = "simple") -> Dict[str, Any]:
        """
        生成文本（支持多轮对话）

        Args:
            prompt (str): 提示文本
            system_prompt (Optional[str]): 系统提示词，用于设定模型角色或行为准则，默认None
            max_tokens (int): 最大生成token数，默认100
            temperature (float): 温度参数，默认0.8
            top_k (int): Top-k采样参数，默认40
            top_p (float): Top-p采样参数，默认0.9
            repetition_penalty (float): 重复惩罚参数，默认1.15
            conversation (Optional[Conversation]): 多轮对话对象（传入则使用对话历史构建 prompt）
            conv_template (str): 对话模板名称（仅在未传入 conversation 时生效）

        Returns:
            Dict[str, Any]: 包含生成结果的字典
        """
        if not self.model:
            raise ValueError("No model loaded")

        # 多轮对话模式
        if conversation is not None:
            conversation.add_user(prompt)
            full_prompt = conversation.build_prompt()
        else:
            # 单轮模式：system_prompt + prompt
            full_prompt = prompt
            if system_prompt and system_prompt.strip():
                full_prompt = f"{system_prompt.strip()}\n\n{prompt}"

        # 编码提示词
        start_tokens = self.tokenizer.encode(full_prompt)[:50]

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

        # 提取助手的回复部分（仅保留新生成的内容）
        assistant_reply = self._extract_assistant_reply(text, full_prompt)

        # 更新对话历史
        if conversation is not None:
            conversation.add_assistant(assistant_reply)

        result = {
            "prompt": prompt,
            "generated_text": text,
            "assistant_reply": assistant_reply,
            "tokens_generated": len(generated) - len(start_tokens),
            "temperature": temperature,
        }
        if conversation is not None:
            result["conversation"] = conversation.to_dict()
        return result

    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None, max_tokens: int = 100, temperature: float = 0.8,
                        top_k: int = 40, top_p: float = 0.9, repetition_penalty: float = 1.15,
                        conversation: Optional[Conversation] = None,
                        conv_template: str = "simple") -> Generator[Dict[str, Any], None, None]:
        """
        流式生成文本（支持多轮对话）

        Args:
            prompt (str): 提示文本
            system_prompt (Optional[str]): 系统提示词，用于设定模型角色或行为准则，默认None
            max_tokens (int): 最大生成token数，默认100
            temperature (float): 温度参数，默认0.8
            top_k (int): Top-k采样参数，默认40
            top_p (float): Top-p采样参数，默认0.9
            repetition_penalty (float): 重复惩罚参数，默认1.15
            conversation (Optional[Conversation]): 多轮对话对象（传入则使用对话历史构建 prompt）
            conv_template (str): 对话模板名称（仅在未传入 conversation 时生效）

        Yields:
            Dict[str, Any]: 包含生成片段的字典
        """
        if not self.model:
            raise ValueError("No model loaded")

        # 多轮对话模式
        if conversation is not None:
            conversation.add_user(prompt)
            full_prompt = conversation.build_prompt()
        else:
            # 单轮模式：system_prompt + prompt
            full_prompt = prompt
            if system_prompt and system_prompt.strip():
                full_prompt = f"{system_prompt.strip()}\n\n{prompt}"

        # 编码提示词
        start_tokens = self.tokenizer.encode(full_prompt)[:50]

        # 流式生成
        generated_tokens = list(start_tokens)
        for token_id, probability in self.model.generate_stream(
            start_tokens,
            max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        ):
            generated_tokens.append(token_id)

            # 解码当前生成的文本
            current_text = self.tokenizer.decode(generated_tokens)
            current_text = current_text.replace('</w>', ' ').replace('  ', ' ').strip()

            # Yield当前状态
            yield {
                "token_id": token_id,
                "text": current_text,
                "probability": probability,
                "tokens_generated": len(generated_tokens) - len(start_tokens),
                "is_complete": False
            }

        # 提取助手回复
        assistant_reply = self._extract_assistant_reply(current_text, full_prompt)

        # 更新对话历史
        if conversation is not None:
            conversation.add_assistant(assistant_reply)

        conv_data = conversation.to_dict() if conversation is not None else None

        # 发送完成信号
        yield {
            "token_id": None,
            "text": current_text,
            "assistant_reply": assistant_reply,
            "probability": 0,
            "tokens_generated": len(generated_tokens) - len(start_tokens),
            "is_complete": True,
            "conversation": conv_data,
        }

    @staticmethod
    def _is_lora_checkpoint(state_dict: Dict[str, Any]) -> bool:
        """
        检测 checkpoint 是否为 LoRA 格式（即 state_dict 中包含 lora_A/lora_B/linear.weight 等 key）。

        Args:
            state_dict: checkpoint 中的 model_state_dict

        Returns:
            如果是 LoRA 格式返回 True
        """
        # 检查是否存在 LoRA 特有 key 且不存在标准 key
        sample_keys = list(state_dict.keys())
        for k in sample_keys:
            if 'lora_A' in k or 'lora_B' in k or '.linear.weight' in k:
                return True
        return False

    @staticmethod
    def _remap_lora_to_standard(state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        将 LoRA 格式的 state_dict 重映射为标准 nn.Linear 格式。

        LoRA 格式: decoder_layers.X.self_attn.{target}.linear.{weight,bias}
        标准格式:  decoder_layers.X.self_attn.{target}.{weight,bias}

        同时丢弃 lora_A、lora_B 等 LoRA 特有参数（因为已在保存前由 merge 步骤合并到 linear 中）。

        Args:
            state_dict: LoRA 格式的 state_dict

        Returns:
            标准格式的 state_dict
        """
        import re
        new_state = {}
        for key, value in state_dict.items():
            # 跳过 LoRA 参数和元数据
            if 'lora_A' in key or 'lora_B' in key:
                continue
            # 将 .linear.weight / .linear.bias 映射为标准 key
            # 例如: decoder_layers.0.self_attn.q_proj.linear.weight -> decoder_layers.0.self_attn.q_proj.weight
            new_key = key.replace('.linear.weight', '.weight').replace('.linear.bias', '.bias')
            new_state[new_key] = value
        return new_state

    @staticmethod
    def _extract_assistant_reply(full_text: str, input_prompt: str) -> str:
        """
        从完整输出中提取助手的回复部分。

        简单策略：去除输入 prompt 对应的前缀，得到纯回复。
        如果模板中有 [ASSISTANT] 等标记，也会尝试去除。

        Returns:
            助手的纯回复文本
        """
        # 去除输入 prompt 前缀
        if full_text.startswith(input_prompt):
            reply = full_text[len(input_prompt):].strip()
        else:
            reply = full_text.strip()

        # 去除模板标记
        markers = ["[ASSISTANT] ", "[ASSISTANT]", "<|im_start|>assistant\n", "assistant\n"]
        for marker in markers:
            if reply.startswith(marker):
                reply = reply[len(marker):].strip()

        return reply
