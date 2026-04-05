"""
模型配置类
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ModelConfig:
    """Shannon-b1 模型配置

    该类定义了Shannon-b1模型的各种配置参数，包括模型架构、训练配置、数据配置等。
    """
    
    # 模型架构
    vocab_size: int = 10000  # 词汇表大小
    d_model: int = 128  # 模型维度
    num_heads: int = 8  # 注意力头数
    d_ff: int = 512  # 前馈网络隐藏层维度
    num_layers: int = 4  # Transformer层数
    max_seq_len: int = 512  # 最大序列长度
    dropout: float = 0.1  # Dropout概率
    
    # 训练配置
    batch_size: int = 32  # 批处理大小
    learning_rate: float = 0.001  # 学习率
    weight_decay: float = 0.01  # 权重衰减系数
    grad_clip: float = 1.0  # 梯度裁剪阈值
    
    # 梯度累积
    gradient_accumulation_steps: int = 1  # 梯度累积步数
    
    # 混合精度训练
    use_amp: bool = True  # 是否使用自动混合精度
    
    # 数据配置
    tokenizer_type: str = "char"  # 分词器类型：char, bpe, simple_bpe
    seq_len: int = 64  # 序列长度
    
    # 早停配置
    early_stopping_patience: int = 10  # 早停耐心值
    early_stopping_min_delta: float = 0.001  # 早停最小变化阈值
    
    # 日志和保存
    log_interval: int = 10  # 日志记录间隔
    save_interval: int = 500  # 模型保存间隔
    eval_interval: int = 100  # 评估间隔
    
    # 其他
    device: str = "cuda"  # 设备类型：cpu, cuda
    seed: int = 42  # 随机种子
    # 训练改进选项
    label_smoothing: float = 0.0  # 标签平滑系数
    lr_warmup_steps: int = 0  # 学习率预热步数
    use_cosine_scheduler: bool = True  # 是否使用余弦调度器
    total_steps: int = 0  # 总训练步数
    tie_word_embeddings: bool = True  # 是否绑定词嵌入权重
    # 额外选项
    gradient_checkpointing: bool = False  # 是否使用梯度检查点
    norm_type: str = "layernorm"  # 归一化类型：layernorm | rmsnorm
    
    def __post_init__(self):
        """初始化后处理，验证CUDA可用性并相应调整设备设置"""
        if self.device == "cuda" and not self._has_cuda():
            self.device = "cpu"
            self.use_amp = False
    
    @staticmethod
    def _has_cuda():
        """检查CUDA是否可用
        
        Returns:
            bool: CUDA是否可用
        """
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    @classmethod
    def from_dict(cls, config_dict):
        """从字典创建ModelConfig实例
        
        Args:
            config_dict (dict): 配置字典
            
        Returns:
            ModelConfig: ModelConfig实例
        """
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})


@dataclass
class TrainingConfig:
    """训练配置

    该类定义了模型训练过程中的各种配置参数，包括训练轮数、保存路径等。

    Attributes:
        epochs (int): 训练轮数，默认为50
        save_path (str): 模型保存路径，默认为"checkpoints/shannon_b1.pt"
        resume_from (Optional[str]): 从中断处恢复训练的路径，默认为None
        tensorboard_dir (str): TensorBoard日志目录，默认为"runs/shannon_b1"
    """
    epochs: int = 50
    save_path: str = "checkpoints/shannon_b1.pt"
    resume_from: Optional[str] = None
    tensorboard_dir: str = "runs/shannon_b1"