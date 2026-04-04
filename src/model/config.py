"""
模型配置类
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ModelConfig:
    """Shannon-b1 模型配置"""
    
    # 模型架构
    vocab_size: int = 10000
    d_model: int = 128
    num_heads: int = 8
    d_ff: int = 512
    num_layers: int = 4
    max_seq_len: int = 512
    dropout: float = 0.1
    
    # 训练配置
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    
    # 梯度累积
    gradient_accumulation_steps: int = 1
    
    # 混合精度训练
    use_amp: bool = True  # 自动混合精度
    
    # 数据配置
    tokenizer_type: str = "char"  # char, bpe, simple_bpe
    seq_len: int = 64
    
    # 早停配置
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    
    # 日志和保存
    log_interval: int = 10
    save_interval: int = 500
    eval_interval: int = 100
    
    # 其他
    device: str = "cuda"  # cpu, cuda
    seed: int = 42
    
    def __post_init__(self):
        if self.device == "cuda" and not self._has_cuda():
            self.device = "cpu"
            self.use_amp = False
    
    @staticmethod
    def _has_cuda():
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    @classmethod
    def from_dict(cls, config_dict):
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})


@dataclass
class TrainingConfig:
    """训练配置"""
    epochs: int = 50
    save_path: str = "checkpoints/shannon_b1.pt"
    resume_from: Optional[str] = None
    tensorboard_dir: str = "runs/shannon_b1"