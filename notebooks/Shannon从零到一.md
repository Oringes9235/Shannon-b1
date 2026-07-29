# Shannon-b1 从零到一：完整构建指南

> **适用人群**：希望从头理解并构建一个小型 GPT 风格语言模型的开发者。
>
> **本文特点**：每一行代码都有详细解释，所有源码直接内嵌到文档中。
>
> **你将学到**：环境搭建 → 分词器 → 数据集 → 模型架构 → 层定义 → 训练器 → 推理脚本。
>
> **阅读顺序**：按目录从上到下，对应项目的构建顺序。

---

## 第 0 步：环境搭建

```bash
git clone https://github.com/Oringes9235/shannon-b1.git
cd shannon-b1
python -m venv .venv
.\.venv\Scripts\activate     # Windows
source .venv/bin/activate      # Linux/Mac
pip install -r requirements.txt
```

依赖说明：
- `torch>=2.0.0` — PyTorch 深度学习框架
- `tokenizers>=0.19.0` — HuggingFace Rust BPE 分词器（速度极快）
- `tqdm` — 训练进度条
- `matplotlib` — 可选，训练曲线可视化

验证安装：`python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`

---

---

## 第 1 步：`src/model/config.py` — 模型配置 (ModelConfig + TrainingConfig) — 所有超参数定义

```python
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
    
    # 长上下文支持
    use_rope: bool = True  # 是否使用RoPE（旋转位置编码）
    rope_base: float = 10000.0  # RoPE的base频率
    use_alibi: bool = False  # 是否使用ALiBi（线性注意力偏置）
    sliding_window_size: Optional[int] = None  # 滑动窗口大小（None表示禁用）
    
    # LoRA 配置
    use_lora: bool = False  # 是否使用 LoRA 微调
    lora_rank: int = 8  # LoRA 低秩分解的秩
    lora_alpha: float = 16.0  # LoRA 缩放因子
    lora_dropout: float = 0.0  # LoRA dropout 概率
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])  # LoRA 目标模块（默认：Q 和 V 投影）
    
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
```

---

## 第 2 步：`src/model/layers.py` — 所有神经网络层 — RoPE、ALiBi、MultiHeadAttention、FFN、RMSNorm、LoRA、KV Cache

```python
"""
自定义神经网络层
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Dict, List


class RotaryPositionalEmbedding(nn.Module):
    """
    RoPE (Rotary Positional Embeddings) - 旋转位置编码
    
    支持超长序列（1M+ tokens），通过旋转矩阵将位置信息编码到Q和K中。
    具有良好的外推性，可以在训练时未见过的更长序列上工作。
    """
    
    def __init__(self, d_model: int, max_seq_len: int = 1048576, base: float = 10000.0):
        """
        初始化RoPE
        
        Args:
            d_model: 模型维度
            max_seq_len: 最大序列长度（默认1M）
            base: RoPE的base频率，控制位置编码的频率范围
        """
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.base = base
        
        # 计算逆频率
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        
        # 预计算cos和sin缓存
        self._update_cos_sin_cache(max_seq_len)
    
    def _update_cos_sin_cache(self, seq_len: int):
        """更新cos和sin缓存"""
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)
    
    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """
        应用RoPE到输入张量
        
        Args:
            x: 输入张量 (batch, heads, seq_len, head_dim) 或 (batch, seq_len, d_model)
            start_pos: 起始位置（用于增量解码）
            
        Returns:
            应用了旋转位置编码的张量
        """
        # 支持两种输入格式：4D (batch, heads, seq_len, head_dim) 和 3D (batch, seq_len, d_model)
        if x.dim() == 4:
            batch, heads, seq_len, head_dim = x.shape
        else:  # 3D
            batch, seq_len, d_model = x.shape
            head_dim = d_model
        
        end_pos = start_pos + seq_len
        
        # 如果超出缓存范围，扩展缓存
        if end_pos > self.cos_cached.size(0):
            new_len = max(end_pos, int(2 ** math.ceil(math.log2(end_pos))))
            self._update_cos_sin_cache(new_len)
        
        # 获取对应位置的cos和sin
        cos = self.cos_cached[start_pos:end_pos].unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
        sin = self.sin_cached[start_pos:end_pos].unsqueeze(0).unsqueeze(0)
        
        # 应用旋转
        return self._apply_rotary(x, cos, sin)
    
    def _apply_rotary(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """
        应用旋转矩阵
        
        Args:
            x: (batch, heads, seq_len, head_dim)
            cos: (1, 1, seq_len, head_dim)
            sin: (1, 1, seq_len, head_dim)
        """
        # 分离奇偶维度
        x1 = x[..., 0::2]  # 偶数维度 (batch, heads, seq_len, head_dim//2)
        x2 = x[..., 1::2]  # 奇数维度 (batch, heads, seq_len, head_dim//2)
        
        # cos和sin也需要进行奇偶分离以匹配x1和x2的形状
        # 注意：在RoPE中，cos和sin是成对重复的，所以取偶数位或奇数位都可以
        cos_half = cos[..., 0::2]  # (1, 1, seq_len, head_dim//2)
        sin_half = sin[..., 0::2]  # (1, 1, seq_len, head_dim//2)
        
        # 标准RoPE旋转公式
        out1 = x1 * cos_half - x2 * sin_half
        out2 = x1 * sin_half + x2 * cos_half
        
        # 合并
        output = torch.stack([out1, out2], dim=-1).flatten(-2)
        return output


class ALiBiBias(nn.Module):
    """
    ALiBi (Attention with Linear Biases) - 线性注意力偏置
    
    为注意力分数添加与距离成比例的偏置，无需显式位置编码即可处理任意长度序列。
    每个注意力头有不同的斜率，形成多尺度的位置感知。
    
    优化：对于超长序列，不预计算完整偏置矩阵，改为动态生成。
    """
    
    def __init__(self, num_heads: int, max_seq_len: int = 1048576):
        """
        初始化ALiBi
        
        Args:
            num_heads: 注意力头数
            max_seq_len: 最大序列长度
        """
        super().__init__()
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        
        # 计算每个头的斜率（几何级数）
        slopes = self._get_slopes(num_heads)
        self.register_buffer('slopes', slopes)
        
        # 对于短序列（<=2048），预计算偏置以提高性能
        if max_seq_len <= 2048:
            self._update_bias_cache(max_seq_len)
            self.use_dynamic_bias = False
        else:
            # 对于长序列，不预计算
            self.register_buffer('bias_cached', torch.tensor(0))  # 占位符
            self.use_dynamic_bias = True
    
    def _get_slopes(self, num_heads: int) -> torch.Tensor:
        """
        计算ALiBi斜率
        
        使用2的幂次作为基础，如果不是2的幂次则进行插值
        """
        def get_slopes_power_of_2(n):
            start = 2 ** (-2 ** -(math.log2(n) - 3))
            ratio = start
            return [start * ratio ** i for i in range(n)]
        
        if math.log2(num_heads).is_integer():
            slopes = get_slopes_power_of_2(num_heads)
        else:
            # 非2的幂次，进行插值
            closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
            slopes_closest = get_slopes_power_of_2(closest_power_of_2)
            slopes_extra = get_slopes_power_of_2(2 * closest_power_of_2)
            slopes = slopes_closest + slopes_extra[1::2][:num_heads - closest_power_of_2]
        
        return torch.tensor(slopes)
    
    def _update_bias_cache(self, seq_len: int):
        """更新偏置缓存"""
        # 创建位置差矩阵 (seq_len, seq_len)
        positions = torch.arange(seq_len)
        rel_positions = positions.unsqueeze(1) - positions.unsqueeze(0)  # (seq_len, seq_len)
        
        # 取绝对值并转为负数（因果掩码只需要下三角）
        rel_positions = -rel_positions.abs().unsqueeze(0)  # (1, seq_len, seq_len)
        
        # 乘以斜率 (num_heads, 1, 1) * (1, seq_len, seq_len) -> (num_heads, seq_len, seq_len)
        bias = self.slopes.unsqueeze(1).unsqueeze(1) * rel_positions.unsqueeze(0)
        
        self.register_buffer('bias_cached', bias, persistent=False)
    
    def forward(self, seq_len: int, device: torch.device = None) -> torch.Tensor:
        """
        获取指定长度的偏置矩阵
        
        Args:
            seq_len: 序列长度
            device: 设备类型（用于动态生成时）
            
        Returns:
            偏置矩阵 (num_heads, seq_len, seq_len)
        """
        # 如果使用动态偏置或请求的长度超过预计算范围
        if self.use_dynamic_bias or seq_len > self.bias_cached.size(-1):
            if device is None:
                device = self.slopes.device
            
            # 动态生成偏置（节省内存）
            positions = torch.arange(seq_len, device=device)
            rel_positions = positions.unsqueeze(1) - positions.unsqueeze(0)  # (seq_len, seq_len)
            rel_positions = -rel_positions.abs().unsqueeze(0)  # (1, seq_len, seq_len)
            
            # 乘以斜率
            bias = self.slopes.to(device).unsqueeze(1).unsqueeze(1) * rel_positions
            
            return bias
        
        return self.bias_cached[:, :seq_len, :seq_len]


class SlidingWindowAttention(nn.Module):
    """
    滑动窗口注意力机制
    
    限制每个token只能关注固定窗口大小内的其他token，显著降低内存占用。
    适用于超长序列，内存复杂度从O(N^2)降至O(N*W)，其中W是窗口大小。
    """
    
    def __init__(self, window_size: int = 2048):
        """
        初始化滑动窗口注意力
        
        Args:
            window_size: 窗口大小（每个token能看到的左右范围）
        """
        super().__init__()
        self.window_size = window_size
    
    def create_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        创建滑动窗口掩码
        
        Args:
            seq_len: 序列长度
            device: 设备
            
        Returns:
            掩码矩阵 (seq_len, seq_len)
        """
        positions = torch.arange(seq_len, device=device)
        # 计算相对位置
        rel_positions = positions.unsqueeze(1) - positions.unsqueeze(0)
        
        # 创建掩码：窗口外的设为-inf
        mask = (rel_positions.abs() > self.window_size).float() * float('-inf')
        
        # 因果掩码：未来的token设为-inf
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1) * float('-inf')
        
        # 合并两个掩码
        combined_mask = torch.maximum(mask, causal_mask)
        
        return combined_mask


class PositionalEncoding(nn.Module):
    """正弦余弦位置编码（保留作为备选）"""
    
    def __init__(self, d_model: int, max_seq_len: int = 512, dropout: float = 0.1):
        """
        初始化正弦余弦位置编码层

        Args:
            d_model (int): 模型的维度
            max_seq_len (int): 最大序列长度，默认为512
            dropout (float): dropout概率，默认为0.1
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x, start_pos: int = 0):
        """
        前向传播

        Args:
            x: 输入张量，形状为(batch_size, seq_len, d_model)
            start_pos: 起始位置（用于增量解码）

        Returns:
            经过位置编码和dropout处理后的张量
        """
        seq_len = x.size(1)
        end_pos = start_pos + seq_len
        
        if end_pos > self.pe.size(1):
            # 如果序列更长，扩展位置编码
            import math
            import torch
            new_pe = torch.zeros(end_pos, self.pe.size(2), device=x.device)
            new_pe[:self.pe.size(1)] = self.pe[0]
            for i in range(self.pe.size(1), end_pos):
                new_pe[i] = new_pe[i - 1] + (new_pe[self.pe.size(1)-1] - new_pe[self.pe.size(1)-2])
            self.pe = new_pe.unsqueeze(0)
        
        x = x + self.pe[:, start_pos:end_pos, :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """可学习位置编码"""
    
    def __init__(self, d_model: int, max_seq_len: int = 512, dropout: float = 0.1):
        """
        初始化可学习位置编码层

        Args:
            d_model (int): 模型的维度
            max_seq_len (int): 最大序列长度，默认为512
            dropout (float): dropout概率，默认为0.1
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
    
    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量，形状为(batch_size, seq_len, d_model)

        Returns:
            经过可学习位置编码和dropout处理后的张量
        """
        x = x + self.pos_embedding[:, :x.size(1), :]
        return self.dropout(x)


class CausalMask(nn.Module):
    """因果掩码 (防止看到未来信息) - 支持长上下文优化"""
    
    def __init__(self, max_seq_len: int = 512):
        """
        初始化因果掩码层

        Args:
            max_seq_len (int): 最大序列长度，默认为512
        """
        super().__init__()
        self.max_seq_len = max_seq_len
        
        # 对于短序列（<=4096），预计算掩码以提高性能
        if max_seq_len <= 4096:
            self.register_buffer("mask", self._create_mask(max_seq_len))
            self.use_dynamic_mask = False
        else:
            # 对于长序列，不预计算，使用动态生成
            self.register_buffer("mask", torch.tensor(0))  # 占位符
            self.use_dynamic_mask = True
    
    def _create_mask(self, seq_len):
        """
        创建因果掩码矩阵

        Args:
            seq_len (int): 序列长度

        Returns:
            掩码矩阵，上三角部分为负无穷，其余为0
        """
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, seq_len: int, device: torch.device = None):
        """
        前向传播

        Args:
            seq_len (int): 当前序列长度
            device: 设备类型（用于动态生成时）

        Returns:
            大小为(seq_len, seq_len)的因果掩码
        """
        # 如果使用动态掩码或请求的长度超过预计算范围
        if self.use_dynamic_mask or seq_len > self.mask.size(0):
            # 动态生成掩码（节省内存）
            if device is None:
                device = self.mask.device if not self.use_dynamic_mask else torch.device('cpu')
            
            # 使用更高效的生成方式：只生成下三角部分的索引
            positions = torch.arange(seq_len, device=device)
            mask = positions.unsqueeze(0) < positions.unsqueeze(1)  # (seq_len, seq_len)
            mask = mask.to(dtype=torch.float32) * float('-inf')
            
            return mask
        
        return self.mask[:seq_len, :seq_len]


class MultiHeadAttentionWithCache(nn.Module):
    """支持KV缓存和RoPE的多头注意力机制"""
    
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        dropout: float = 0.1,
        use_rope: bool = True,
        rope_base: float = 10000.0,
        max_seq_len: int = 1048576,
    ):
        """
        初始化多头注意力层
        
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            dropout: dropout概率
            use_rope: 是否使用RoPE
            rope_base: RoPE的base频率
            max_seq_len: 最大序列长度（用于RoPE缓存）
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_rope = use_rope
        
        # Q, K, V 投影
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # 输出投影
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # RoPE（如果启用）
        if use_rope:
            self.rope = RotaryPositionalEmbedding(
                d_model=self.head_dim,
                max_seq_len=max_seq_len,
                base=rope_base,
            )
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        start_pos: int = 0,
        alibi_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        前向传播
        
        Args:
            query: (batch, seq_len_q, d_model)
            key: (batch, seq_len_k, d_model)
            value: (batch, seq_len_v, d_model)
            mask: 注意力掩码
            past_key_value: 过去的(K, V)缓存
            start_pos: 起始位置（用于RoPE增量编码）
            alibi_bias: ALiBi偏置 (num_heads, seq_len_q, seq_len_k)
            
        Returns:
            output: (batch, seq_len_q, d_model)
            present_key_value: 新的(K, V)缓存（如果启用）
        """
        batch_size, seq_len_q, _ = query.size()
        
        # 线性投影
        q = self.q_proj(query).view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 应用RoPE（如果启用）
        if self.use_rope:
            q = self.rope(q, start_pos=start_pos)
            k = self.rope(k, start_pos=start_pos)
        
        # 如果有past_key_value，拼接到当前的K和V上
        if past_key_value is not None:
            past_key, past_value = past_key_value
            k = torch.cat([past_key, k], dim=2)
            v = torch.cat([past_value, v], dim=2)
        
        # 返回当前的K和V作为新的缓存
        present_key_value = (k, v)
        
        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 应用ALiBi偏置（如果提供）
        if alibi_bias is not None:
            # alibi_bias形状: (num_heads, seq_len_q, seq_len_k)
            # attn_scores形状: (batch, num_heads, seq_len_q, seq_len_k)
            attn_scores = attn_scores + alibi_bias.unsqueeze(0)
        
        # 应用掩码
        if mask is not None:
            # 确保mask的形状匹配
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len_q, seq_len_k)
            attn_scores = attn_scores.masked_fill(mask == float('-inf'), float('-inf'))
        
        # Softmax + Dropout
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # 加权求和
        output = torch.matmul(attn_probs, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
        
        # 输出投影
        output = self.out_proj(output)
        
        return output, present_key_value


class TransformerDecoderLayerWithCache(nn.Module):
    """支持KV缓存和RoPE的Transformer Decoder层"""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = 'gelu',
        use_rope: bool = True,
        rope_base: float = 10000.0,
        max_seq_len: int = 1048576,
    ):
        """
        初始化Transformer Decoder层
        
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            dim_feedforward: FFN维度
            dropout: dropout概率
            activation: 激活函数
            use_rope: 是否使用RoPE
            rope_base: RoPE的base频率
            max_seq_len: 最大序列长度
        """
        super().__init__()
        
        # Self-attention with cache support and RoPE
        self.self_attn = MultiHeadAttentionWithCache(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            use_rope=use_rope,
            rope_base=rope_base,
            max_seq_len=max_seq_len,
        )
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.GELU() if activation == 'gelu' else nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)
        
        # Layer norms (使用 RMSNorm 以保持一致性)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.dropout_attn = nn.Dropout(dropout)
    
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        start_pos: int = 0,
        alibi_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        前向传播
        
        Args:
            tgt: 目标序列 (batch, seq_len, d_model)
            memory: 记忆序列 (batch, seq_len, d_model) - 在此实现中未用于Self-Attention，但保留接口兼容性
            tgt_mask: 目标掩码
            past_key_value: 过去的(K, V)缓存
            start_pos: 起始位置（用于RoPE增量编码）
            alibi_bias: ALiBi偏置 (num_heads, seq_len, seq_len)
            
        Returns:
            output: (batch, seq_len, d_model)
            present_key_value: 新的(K, V)缓存
        """
        # Self-attention
        attn_output, present_key_value = self.self_attn(
            query=tgt,
            key=tgt,
            value=tgt,
            mask=tgt_mask,
            past_key_value=past_key_value,
            start_pos=start_pos,
            alibi_bias=alibi_bias,
        )
        
        # Add & Norm
        tgt = tgt + self.dropout_attn(attn_output)
        tgt = self.norm1(tgt)
        
        # Feed-forward
        ff_output = self.linear2(self.dropout1(self.activation(self.linear1(tgt))))
        ff_output = self.dropout2(ff_output)
        
        # Add & Norm
        tgt = tgt + ff_output
        tgt = self.norm2(tgt)
        
        return tgt, present_key_value


class RMSNorm(nn.Module):
    """RMSNorm (比 LayerNorm 更快)"""
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        """
        初始化RMSNorm层

        Args:
            d_model (int): 模型的维度
            eps (float): 防止除零的小常数，默认为1e-6
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量，形状为(..., d_model)

        Returns:
            经过RMSNorm归一化后的张量
        """
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


class LoRALinear(nn.Module):
    """
    LoRA (Low-Rank Adaptation) 线性层

    通过低秩分解添加可训练适配器，冻结原始权重。
    输出 = Wx + (alpha / rank) * BAx

    标准用法:
    - 将 LoRALinear 包装在已有 nn.Linear 层上
    - 调用 merge_weights() 将 LoRA 权重合并到原始权重中
    - 调用 unmerge_weights() 分离 LoRA 权重
    """

    def __init__(
        self,
        existing_linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        merge_weights: bool = False,
    ):
        """
        初始化 LoRA 线性层

        Args:
            existing_linear: 已有的 nn.Linear 层
            rank: 低秩分解的秩
            alpha: 缩放因子
            dropout: LoRA dropout 概率
            merge_weights: 是否合并权重（推理时使用）
        """
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.merge_weights = merge_weights
        self.merged = False

        in_features = existing_linear.in_features
        out_features = existing_linear.out_features

        # 冻结原始权重和偏置
        self.linear = existing_linear
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        # LoRA 低秩矩阵（与基础线性层保持同一设备）
        base_device = existing_linear.weight.device
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features, device=base_device))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, device=base_device))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Kaiming 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量，形状为 (..., in_features)

        Returns:
            输出张量，形状为 (..., out_features)
        """
        # 基础线性变换
        result = self.linear(x)

        if not self.merged:
            # LoRA 路径: scaling * x @ A.T @ B.T
            lora_out = (
                self.lora_dropout(x)
                @ self.lora_A.T
                @ self.lora_B.T
            ) * self.scaling
            result = result + lora_out

        return result

    def merge_weights_to_base(self):
        """
        将 LoRA 权重合并到基础线性层中（用于推理加速）
        合并后: W' = W + (alpha/rank) * BA
        """
        if not self.merged:
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.linear.weight.data += delta_w
            self.merged = True

    def unmerge_weights_from_base(self):
        """
        从基础线性层中分离 LoRA 权重（用于继续训练）
        """
        if self.merged:
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.linear.weight.data -= delta_w
            self.merged = False

    def get_lora_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取 LoRA 参数（用于保存）

        Returns:
            (lora_A, lora_B) 参数元组
        """
        return self.lora_A.data.clone(), self.lora_B.data.clone()

    def load_lora_params(self, lora_A: torch.Tensor, lora_B: torch.Tensor):
        """
        加载 LoRA 参数

        Args:
            lora_A: 低秩矩阵 A
            lora_B: 低秩矩阵 B
        """
        device = self.lora_A.device
        self.lora_A.data = lora_A.to(device)
        self.lora_B.data = lora_B.to(device)

    def get_lora_state_dict(self) -> Dict[str, torch.Tensor]:
        """
        获取仅包含 LoRA 参数的状态字典（用于保存）

        Returns:
            包含 lora_A 和 lora_B 的状态字典
        """
        return {
            "lora_A": self.lora_A.data.clone(),
            "lora_B": self.lora_B.data.clone(),
            "rank": torch.tensor(self.rank),
            "alpha": torch.tensor(self.alpha),
        }

```

---

## 第 3 步：`src/model/shannon.py` — 主模型 ShannonB1 — Decoder-only Transformer 完整实现

```python
"""
Shannon-b1 主模型
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import math
from typing import Optional, List, Generator, Tuple, Dict

from .config import ModelConfig
from .layers import PositionalEncoding, CausalMask, RMSNorm, ALiBiBias, SlidingWindowAttention
from .layers import TransformerDecoderLayerWithCache, LoRALinear


class ShannonB1(nn.Module):
    """
    Shannon-b1: 轻量级 GPT 风格语言模型
    
    架构:
    - Token Embedding
    - Positional Encoding
    - Transformer Decoder (多层)
    - Layer Norm
    - Output Projection
    """
    
    def __init__(self, config: ModelConfig):
        """
        初始化 ShannonB1 模型
        
        Args:
            config: 模型配置对象，包含词汇表大小、模型维度等参数
        """
        super().__init__()

        self.config = config
        
        # 词嵌入
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # 位置编码（根据配置选择）
        if config.use_rope:
            # 使用RoPE时，不在embedding后添加位置编码
            self.pos_encoding = None
        else:
            # 传统正弦位置编码
            self.pos_encoding = PositionalEncoding(
                config.d_model, config.max_seq_len, config.dropout
            )
        
        # Transformer Decoder 层（使用支持KV Cache和RoPE的版本）
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayerWithCache(
                d_model=config.d_model,
                num_heads=config.num_heads,
                dim_feedforward=config.d_ff,
                dropout=config.dropout,
                activation='gelu',
                use_rope=config.use_rope,
                rope_base=config.rope_base,
                max_seq_len=config.max_seq_len,
            )
            for _ in range(config.num_layers)
        ])
        self.use_checkpointing = getattr(config, 'gradient_checkpointing', False)
        
        # 因果掩码
        self.causal_mask = CausalMask(config.max_seq_len)
        
        # ALiBi（如果启用）
        if config.use_alibi:
            self.alibi_bias = ALiBiBias(
                num_heads=config.num_heads,
                max_seq_len=config.max_seq_len,
            )
        else:
            self.alibi_bias = None
        
        # 滑动窗口注意力（如果启用）
        if config.sliding_window_size is not None:
            self.sliding_window = SlidingWindowAttention(
                window_size=config.sliding_window_size,
            )
        else:
            self.sliding_window = None
        
        # 最终归一化（支持 RMSNorm）
        if getattr(config, 'norm_type', 'layernorm') == 'rmsnorm':
            self.ln_f = RMSNorm(config.d_model)
        else:
            self.ln_f = nn.LayerNorm(config.d_model)
        
        # 输出投影
        self.output = nn.Linear(config.d_model, config.vocab_size)
        
        # 初始化权重
        self._init_weights()
        # 权重绑定（词表投影与嵌入共享）
        if getattr(config, 'tie_word_embeddings', False):
            try:
                self.output.weight = self.token_embedding.weight
            except Exception:
                pass
    
    def _init_weights(self):
        """初始化模型权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    # ==================== LoRA 相关方法 ====================
    
    def apply_lora(
        self,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        target_modules: Optional[List[str]] = None,
    ) -> None:
        """
        将 LoRA 适配器应用到模型的指定线性层上。

        冻结原始模型参数，仅保留 LoRA 参数可训练。

        Args:
            rank: 低秩分解的秩
            alpha: LoRA 缩放因子
            dropout: LoRA dropout 概率
            target_modules: 要应用 LoRA 的目标模块名称列表，
                            可选值: "q_proj", "k_proj", "v_proj", "out_proj"
                            默认: ["q_proj", "v_proj"]
        """
        if target_modules is None:
            target_modules = ["q_proj", "v_proj"]

        self.config.use_lora = True
        self.config.lora_rank = rank
        self.config.lora_alpha = alpha
        self.config.lora_dropout = dropout
        self.config.lora_target_modules = target_modules

        # 冻结所有参数
        for param in self.parameters():
            param.requires_grad = False

        # 保持 embedding 可训练（如果配置需要）
        # self.token_embedding.weight.requires_grad = True  # 可选

        for layer in self.decoder_layers:
            for target in target_modules:
                if target == "q_proj" and hasattr(layer.self_attn, "q_proj"):
                    existing = layer.self_attn.q_proj
                    if isinstance(existing, LoRALinear):
                        existing = existing.linear  # 取出原始 nn.Linear
                    layer.self_attn.q_proj = LoRALinear(
                        existing,
                        rank=rank,
                        alpha=alpha,
                        dropout=dropout,
                    )
                elif target == "k_proj" and hasattr(layer.self_attn, "k_proj"):
                    existing = layer.self_attn.k_proj
                    if isinstance(existing, LoRALinear):
                        existing = existing.linear
                    layer.self_attn.k_proj = LoRALinear(
                        existing,
                        rank=rank,
                        alpha=alpha,
                        dropout=dropout,
                    )
                elif target == "v_proj" and hasattr(layer.self_attn, "v_proj"):
                    existing = layer.self_attn.v_proj
                    if isinstance(existing, LoRALinear):
                        existing = existing.linear
                    layer.self_attn.v_proj = LoRALinear(
                        existing,
                        rank=rank,
                        alpha=alpha,
                        dropout=dropout,
                    )
                elif target == "out_proj" and hasattr(layer.self_attn, "out_proj"):
                    existing = layer.self_attn.out_proj
                    if isinstance(existing, LoRALinear):
                        existing = existing.linear
                    layer.self_attn.out_proj = LoRALinear(
                        existing,
                        rank=rank,
                        alpha=alpha,
                        dropout=dropout,
                    )

        # 统计可训练参数（静默，无日志输出）
        lora_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())

    def get_lora_state_dict(self) -> Dict[str, torch.Tensor]:
        """
        获取仅包含 LoRA 参数的状态字典（用于保存轻量级 LoRA 权重）。

        Returns:
            包含所有 LoRA 相关参数的状态字典
        """
        lora_state = {}
        for layer_idx, layer in enumerate(self.decoder_layers):
            for target in self.config.lora_target_modules:
                if target == "q_proj" and isinstance(layer.self_attn.q_proj, LoRALinear):
                    lora_mod = layer.self_attn.q_proj
                    lora_state[f"layer_{layer_idx}_q_proj_lora_A"] = lora_mod.lora_A.data.clone()
                    lora_state[f"layer_{layer_idx}_q_proj_lora_B"] = lora_mod.lora_B.data.clone()
                elif target == "k_proj" and isinstance(layer.self_attn.k_proj, LoRALinear):
                    lora_mod = layer.self_attn.k_proj
                    lora_state[f"layer_{layer_idx}_k_proj_lora_A"] = lora_mod.lora_A.data.clone()
                    lora_state[f"layer_{layer_idx}_k_proj_lora_B"] = lora_mod.lora_B.data.clone()
                elif target == "v_proj" and isinstance(layer.self_attn.v_proj, LoRALinear):
                    lora_mod = layer.self_attn.v_proj
                    lora_state[f"layer_{layer_idx}_v_proj_lora_A"] = lora_mod.lora_A.data.clone()
                    lora_state[f"layer_{layer_idx}_v_proj_lora_B"] = lora_mod.lora_B.data.clone()
                elif target == "out_proj" and isinstance(layer.self_attn.out_proj, LoRALinear):
                    lora_mod = layer.self_attn.out_proj
                    lora_state[f"layer_{layer_idx}_out_proj_lora_A"] = lora_mod.lora_A.data.clone()
                    lora_state[f"layer_{layer_idx}_out_proj_lora_B"] = lora_mod.lora_B.data.clone()
        # 保存 LoRA 配置
        lora_state["_lora_rank"] = torch.tensor(self.config.lora_rank)
        lora_state["_lora_alpha"] = torch.tensor(self.config.lora_alpha)
        lora_state["_lora_target_modules"] = "_".join(self.config.lora_target_modules)
        return lora_state

    def save_lora_weights(self, path: str) -> None:
        """
        保存 LoRA 权重到文件（仅保存 LoRA 适配器参数，轻量级）。

        Args:
            path: 保存路径（建议使用 .lora.pt 后缀）
        """
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        lora_state = self.get_lora_state_dict()
        torch.save(lora_state, path)

    def load_lora_weights(self, path: str) -> None:
        """
        从文件加载 LoRA 权重。需要先调用 apply_lora() 应用 LoRA 结构。

        Args:
            path: LoRA 权重文件路径
        """
        # 尝试显式允许加载包含自定义类的完整检查点（非 weights-only）
        try:
            lora_state = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            # 兼容较旧的 PyTorch 版本
            lora_state = torch.load(path, map_location="cpu")
        except Exception as e:
            print(f"[Warn] Failed to load LoRA weights: {e}")
            raise

        loaded = 0
        for layer_idx, layer in enumerate(self.decoder_layers):
            for target in self.config.lora_target_modules:
                key_a = f"layer_{layer_idx}_{target}_lora_A"
                key_b = f"layer_{layer_idx}_{target}_lora_B"
                if key_a in lora_state and key_b in lora_state:
                    if target == "q_proj" and isinstance(layer.self_attn.q_proj, LoRALinear):
                        layer.self_attn.q_proj.lora_A.data = lora_state[key_a].to(
                            layer.self_attn.q_proj.lora_A.device
                        )
                        layer.self_attn.q_proj.lora_B.data = lora_state[key_b].to(
                            layer.self_attn.q_proj.lora_B.device
                        )
                        loaded += 1
                    elif target == "k_proj" and isinstance(layer.self_attn.k_proj, LoRALinear):
                        layer.self_attn.k_proj.lora_A.data = lora_state[key_a].to(
                            layer.self_attn.k_proj.lora_A.device
                        )
                        layer.self_attn.k_proj.lora_B.data = lora_state[key_b].to(
                            layer.self_attn.k_proj.lora_B.device
                        )
                        loaded += 1
                    elif target == "v_proj" and isinstance(layer.self_attn.v_proj, LoRALinear):
                        layer.self_attn.v_proj.lora_A.data = lora_state[key_a].to(
                            layer.self_attn.v_proj.lora_A.device
                        )
                        layer.self_attn.v_proj.lora_B.data = lora_state[key_b].to(
                            layer.self_attn.v_proj.lora_B.device
                        )
                        loaded += 1
                    elif target == "out_proj" and isinstance(layer.self_attn.out_proj, LoRALinear):
                        layer.self_attn.out_proj.lora_A.data = lora_state[key_a].to(
                            layer.self_attn.out_proj.lora_A.device
                        )
                        layer.self_attn.out_proj.lora_B.data = lora_state[key_b].to(
                            layer.self_attn.out_proj.lora_B.device
                        )
                        loaded += 1
        _ = loaded  # suppress unused-variable warning

    def merge_lora_weights(self) -> None:
        """
        将所有 LoRA 权重合并到基础模型的线性层中（用于推理加速）。
        合并后模型不再包含 LoRA 参数，直接使用合并后的权重进行推理。
        """
        for layer in self.decoder_layers:
            for target in self.config.lora_target_modules:
                attr = getattr(layer.self_attn, target, None)
                if isinstance(attr, LoRALinear):
                    attr.merge_weights_to_base()
        pass

    def unmerge_lora_weights(self) -> None:
        """
        从基础模型中分离 LoRA 权重（用于继续训练）。
        """
        for layer in self.decoder_layers:
            for target in self.config.lora_target_modules:
                attr = getattr(layer.self_attn, target, None)
                if isinstance(attr, LoRALinear):
                    attr.unmerge_weights_from_base()
        pass

    def get_lora_trainable_params(self) -> List[nn.Parameter]:
        """
        获取所有 LoRA 可训练参数（用于优化器）。

        Returns:
            LoRA 可训练参数列表
        """
        lora_params = []
        for name, param in self.named_parameters():
            if param.requires_grad and "lora_" in name:
                lora_params.append(param)
        return lora_params

    def forward(
        self, 
        tokens: torch.Tensor, 
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        前向传播
        
        Args:
            tokens: (batch, seq_len) 输入 token IDs
            past_key_values: 可选的KV缓存列表，每层一个(K, V)元组
        
        Returns:
            logits: (batch, seq_len, vocab_size)
            present_key_values: 新的KV缓存（如果启用）
        """
        batch, seq_len = tokens.shape
        
        # 词嵌入 + 缩放
        x = self.token_embedding(tokens) * math.sqrt(self.config.d_model)
        
        # 位置编码（如果不使用RoPE）
        if not self.config.use_rope:
            if past_key_values is not None and len(past_key_values) > 0:
                # 增量解码
                past_length = past_key_values[0][0].size(-2)
                x = self.pos_encoding(x, start_pos=past_length)
            else:
                # 全量编码
                x = self.pos_encoding(x)

        # 因果掩码（动态生成，传入设备信息）
        mask = self.causal_mask(seq_len, device=tokens.device)
        
        # 滑动窗口掩码（如果启用）
        if self.sliding_window is not None:
            window_mask = self.sliding_window.create_mask(seq_len, tokens.device)
            # 合并掩码：取两者中更严格的那个
            mask = torch.minimum(mask, window_mask)

        # ALiBi偏置（如果启用）
        alibi_bias = None
        if self.alibi_bias is not None:
            alibi_bias = self.alibi_bias(seq_len, device=tokens.device)  # (num_heads, seq_len, seq_len)

        # 逐层应用 Transformer Decoder 层
        new_past_key_values = []
        
        for layer_idx, layer in enumerate(self.decoder_layers):
            # 获取当前层的past_key_value
            layer_past = past_key_values[layer_idx] if past_key_values is not None else None
            
            # 计算起始位置（用于RoPE）
            start_pos = 0
            if layer_past is not None:
                start_pos = layer_past[0].size(-2)
            
            # 前向传播
            x, present_kv = layer(
                tgt=x,
                memory=x,
                tgt_mask=mask,
                past_key_value=layer_past,
                start_pos=start_pos,
                alibi_bias=alibi_bias,  # 传递ALiBi偏置
            )
            
            # 保存当前层的KV缓存
            if new_past_key_values is not None:
                new_past_key_values.append(present_kv)
        
        # LayerNorm
        x = self.ln_f(x)
        
        # 输出投影
        logits = self.output(x)
        
        return logits, new_past_key_values
    
    def generate(
        self, 
        start_tokens: List[int], 
        max_new_tokens: int, 
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        ban_immediate_repeat: bool = True,
        ngram_block_size: int = 3,
        best_of: int = 1,
        max_repetition: Optional[int] = None,
        use_kv_cache: bool = True,
    ) -> List[int]:
        """
        自回归生成文本（支持KV Cache优化）
        
        Args:
            start_tokens: 起始 token 序列
            max_new_tokens: 最大生成 token 数
            temperature: 温度系数 (越高越随机)
            top_k: Top-K 采样
            top_p: Top-P (nucleus) 采样
            repetition_penalty: 重复惩罚系数
            presence_penalty: 存在惩罚系数
            frequency_penalty: 频率惩罚系数
            ban_immediate_repeat: 是否禁止立即重复
            ngram_block_size: N-gram 阻断大小
            best_of: 生成多少个样本后选择最优
            max_repetition: 最大重复次数限制
            use_kv_cache: 是否使用KV缓存加速推理
        
        Returns:
            生成的 token 序列
        """
        self.eval()
        device = next(self.parameters()).device
        tokens = torch.tensor([start_tokens], device=device)

        def single_sample():
            cur_tokens = tokens.clone()
            logprob_sum = 0.0
            from collections import defaultdict
            # track seen ngrams of various sizes
            seen_ngrams = set()
            token_counts = defaultdict(int)
            
            # KV Cache初始化
            past_key_values = None

            for step in range(max_new_tokens):
                # 前向传播（使用KV Cache）
                if use_kv_cache and step > 0:
                    # 只处理最后一个token
                    input_tokens = cur_tokens[:, -1:]
                    logits, new_past_key_values = self.forward(input_tokens, past_key_values=past_key_values)
                    past_key_values = new_past_key_values
                else:
                    # 第一次需要处理整个序列并初始化KV Cache
                    logits, past_key_values = self.forward(cur_tokens)

                # 获取最后一个位置
                last_logits = logits[0, -1, :].float()

                # 应用温度（处理temperature=0的情况）
                if temperature == 0.0:
                    # 贪婪解码模式：先应用所有约束，然后argmax
                    
                    # 重复惩罚
                    if repetition_penalty is not None and repetition_penalty != 1.0:
                        generated = set(cur_tokens[0].tolist())
                        for token_id in generated:
                            if last_logits[token_id] < 0:
                                last_logits[token_id] *= float(repetition_penalty)
                            else:
                                last_logits[token_id] /= float(repetition_penalty)
                    
                    # presence / frequency penalty
                    if presence_penalty != 0.0 or frequency_penalty != 0.0:
                        from collections import Counter
                        counts = Counter(cur_tokens[0].tolist())
                        for tok_id, cnt in counts.items():
                            if presence_penalty != 0.0:
                                last_logits[tok_id] -= float(presence_penalty)
                            if frequency_penalty != 0.0 and cnt > 0:
                                last_logits[tok_id] -= float(frequency_penalty) * float(cnt)
                    
                    # 避免直接重复上一个 token
                    if ban_immediate_repeat and cur_tokens.size(1) > 0:
                        prev_token = int(cur_tokens[0, -1].item())
                        last_logits[prev_token] = float('-inf')
                    
                    # n-gram 重复阻断
                    if ngram_block_size > 1 and cur_tokens.size(1) >= 1:
                        banned = []
                        seq_list = [int(x) for x in cur_tokens[0].tolist()]
                        for candidate in range(last_logits.size(0)):
                            will_form_repeat = False
                            for n in range(2, ngram_block_size + 1):
                                if len(seq_list) + 1 >= n:
                                    prev_ngram = tuple(seq_list[-(n-1):] + [candidate])
                                    if prev_ngram in seen_ngrams:
                                        will_form_repeat = True
                                        break
                            if will_form_repeat:
                                banned.append(candidate)
                        if banned:
                            last_logits[torch.tensor(banned, device=last_logits.device)] = float('-inf')
                    
                    # 最大重复限制
                    max_rep = int(max_repetition) if max_repetition is not None else int(getattr(self.config, 'max_repetition', 3))
                    for tok_id, cnt in list(token_counts.items()):
                        if cnt >= max_rep:
                            last_logits[tok_id] = float('-inf')
                    
                    # 贪婪解码
                    next_token = torch.argmax(last_logits).item()
                    probs = torch.softmax(last_logits, dim=-1)
                    logprob = torch.log(probs[next_token] + 1e-12).item()
                    logprob_sum += logprob
                    
                    # 更新序列
                    next_token_tensor = torch.tensor([[next_token]], device=device)
                    cur_tokens = torch.cat([cur_tokens, next_token_tensor], dim=1)
                    
                    # 更新状态
                    token_counts[next_token] += 1
                    seq_now = [int(x) for x in cur_tokens[0].tolist()]
                    L = len(seq_now)
                    for n in range(2, ngram_block_size + 1):
                        if L >= n:
                            ng = tuple(seq_now[-n:])
                            seen_ngrams.add(ng)
                    
                    continue
                
                # 正常温度下的处理
                if temperature != 1.0:
                    last_logits = last_logits / float(temperature)

                vocab_size = last_logits.size(0)

                # Top-K 采样（更稳健的实现）
                if top_k is not None and top_k > 0 and top_k < vocab_size:
                    topk_vals, _ = torch.topk(last_logits, top_k)
                    threshold = topk_vals[-1]
                    last_logits = torch.where(last_logits < threshold, torch.tensor(float('-inf'), device=last_logits.device), last_logits)

                # Top-P (nucleus) 采样（更稳健）
                if top_p is not None and 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(last_logits, descending=True)
                    sorted_probs = torch.softmax(sorted_logits, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                    # 找到保留的索引（累积概率 <= top_p）
                    keep_mask = cumulative_probs <= top_p
                    # 总是至少保留一个 token
                    if not keep_mask.any():
                        keep_mask[0] = True

                    # 将不保留的 token 设为 -inf
                    remove_indices = sorted_indices[~keep_mask]
                    last_logits[remove_indices] = float('-inf')

                # 重复惩罚（参考 HuggingFace 实现）
                if repetition_penalty is not None and repetition_penalty != 1.0:
                    generated = set(cur_tokens[0].tolist())
                    for token_id in generated:
                        if last_logits[token_id] < 0:
                            last_logits[token_id] *= float(repetition_penalty)
                        else:
                            last_logits[token_id] /= float(repetition_penalty)

                # presence / frequency penalty: 在 logit 上做线性惩罚
                if presence_penalty != 0.0 or frequency_penalty != 0.0:
                    from collections import Counter
                    counts = Counter(cur_tokens[0].tolist())
                    for tok_id, cnt in counts.items():
                        if presence_penalty != 0.0:
                            last_logits[tok_id] -= float(presence_penalty)
                        if frequency_penalty != 0.0 and cnt > 0:
                            last_logits[tok_id] -= float(frequency_penalty) * float(cnt)

                # 避免直接重复上一个 token（可选）
                if ban_immediate_repeat and cur_tokens.size(1) > 0:
                    prev_token = int(cur_tokens[0, -1].item())
                    last_logits[prev_token] = float('-inf')

                # n-gram 重复阻断（严格模式）：检查所有 n <= ngram_block_size，若候选会形成已见 ngram，则屏蔽
                if ngram_block_size > 1 and cur_tokens.size(1) >= 1:
                    banned = []
                    seq_list = [int(x) for x in cur_tokens[0].tolist()]
                    for candidate in range(last_logits.size(0)):
                        will_form_repeat = False
                        # check ngrams of size 1..ngram_block_size
                        for n in range(1, ngram_block_size + 1):
                            if n == 1:
                                # single token repetition handled by token_counts below
                                continue
                            if len(seq_list) + 1 >= n:
                                prev_ngram = tuple(seq_list[-(n-1):] + [candidate])
                                if prev_ngram in seen_ngrams:
                                    will_form_repeat = True
                                    break
                        if will_form_repeat:
                            banned.append(candidate)
                    if banned:
                        last_logits[torch.tensor(banned, device=last_logits.device)] = float('-inf')

                # 归一化并采样下一个 token
                probs = torch.softmax(last_logits, dim=-1)
                # 防止数值问题
                if torch.isnan(probs).any():
                    probs = torch.nn.functional.softmax(last_logits.float().masked_fill(torch.isinf(last_logits), -1e9), dim=-1)

                next_token = torch.multinomial(probs, 1).item()

                # 更新 logprob sum
                logprob = torch.log(probs[next_token] + 1e-12).item()
                logprob_sum += logprob

                # 拼接
                next_token_tensor = torch.tensor([[next_token]], device=device)
                cur_tokens = torch.cat([cur_tokens, next_token_tensor], dim=1)

                # 更新 seen ngrams 和 token 计数
                token_counts[next_token] += 1
                # add all ngrams ending at new token
                seq_now = [int(x) for x in cur_tokens[0].tolist()]
                L = len(seq_now)
                for n in range(2, ngram_block_size + 1):
                    if L >= n:
                        ng = tuple(seq_now[-n:])
                        seen_ngrams.add(ng)

                # 如果某个 token出现次数过多，强制屏蔽后续产生
                if max_repetition is not None:
                    max_rep = int(max_repetition)
                else:
                    max_rep = int(getattr(self.config, 'max_repetition', 3))
                # 如果某 token 出现次数超过 max_rep，强制在 logits 中屏蔽该 token
                for tok_id, cnt in list(token_counts.items()):
                    if cnt >= max_rep:
                        last_logits[tok_id] = float('-inf')

            return cur_tokens[0].tolist(), logprob_sum

        # best_of: 生成多个样本并返回平均 logprob 最好的那一个
        best_seq = None
        best_score = -float('inf')
        for i in range(max(1, best_of)):
            seq, score = single_sample()
            if score > best_score:
                best_score = score
                best_seq = seq

        return best_seq

    def generate_stream(
        self,
        start_tokens: List[int],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        ban_immediate_repeat: bool = True,
        ngram_block_size: int = 3,
        max_repetition: Optional[int] = None,
        use_kv_cache: bool = True,
    ) -> Generator[Tuple[int, float], None, None]:
        """
        流式生成文本，逐个token yield（支持KV Cache优化）
        
        Args:
            start_tokens: 起始 token 序列
            max_new_tokens: 最大生成 token 数
            temperature: 温度系数 (越高越随机，0.0为贪婪解码)
            top_k: Top-K 采样
            top_p: Top-P (nucleus) 采样
            repetition_penalty: 重复惩罚系数
            presence_penalty: 存在惩罚系数
            frequency_penalty: 频率惩罚系数
            ban_immediate_repeat: 是否禁止立即重复
            ngram_block_size: N-gram 阻断大小
            max_repetition: 最大重复次数限制
            use_kv_cache: 是否使用KV缓存加速推理
        
        Yields:
            Tuple[int, float]: (token_id, probability)
        """
        self.eval()
        device = next(self.parameters()).device
        cur_tokens = torch.tensor([start_tokens], device=device)
        
        from collections import Counter
        seen_ngrams = set()
        token_counts = Counter(start_tokens)
        
        # KV Cache初始化
        past_key_values = None
        
        for step in range(max_new_tokens):
            # 前向传播（使用KV Cache）
            with torch.no_grad():
                if use_kv_cache and step > 0:
                    # 只处理最后一个token
                    input_tokens = cur_tokens[:, -1:]
                    logits, new_past_key_values = self.forward(input_tokens, past_key_values=past_key_values)
                    past_key_values = new_past_key_values
                else:
                    # 第一次需要处理整个序列并初始化KV Cache
                    logits, past_key_values = self.forward(cur_tokens)
            
            # 获取最后一个位置的logits
            last_logits = logits[0, -1, :].float()
            
            # 应用温度（处理temperature=0的情况）
            if temperature == 0.0:
                # 先应用所有惩罚和约束（与generate_stream一致）
                
                # 重复惩罚
                if repetition_penalty is not None and repetition_penalty != 1.0:
                    generated = set(cur_tokens[0].tolist())
                    for token_id in generated:
                        if last_logits[token_id] < 0:
                            last_logits[token_id] *= float(repetition_penalty)
                        else:
                            last_logits[token_id] /= float(repetition_penalty)
                
                # presence / frequency penalty
                if presence_penalty != 0.0 or frequency_penalty != 0.0:
                    from collections import Counter
                    counts = Counter(cur_tokens[0].tolist())
                    for tok_id, cnt in counts.items():
                        if presence_penalty != 0.0:
                            last_logits[tok_id] -= float(presence_penalty)
                        if frequency_penalty != 0.0 and cnt > 0:
                            last_logits[tok_id] -= float(frequency_penalty) * float(cnt)
                
                # 避免直接重复上一个 token
                if ban_immediate_repeat and cur_tokens.size(1) > 0:
                    prev_token = int(cur_tokens[0, -1].item())
                    last_logits[prev_token] = float('-inf')
                
                # n-gram 重复阻断
                if ngram_block_size > 1 and cur_tokens.size(1) >= 1:
                    banned = []
                    seq_list = [int(x) for x in cur_tokens[0].tolist()]
                    for candidate in range(last_logits.size(0)):
                        will_form_repeat = False
                        for n in range(2, ngram_block_size + 1):
                            if len(seq_list) + 1 >= n:
                                prev_ngram = tuple(seq_list[-(n-1):] + [candidate])
                                if prev_ngram in seen_ngrams:
                                    will_form_repeat = True
                                    break
                        if will_form_repeat:
                            banned.append(candidate)
                    if banned:
                        last_logits[torch.tensor(banned, device=last_logits.device)] = float('-inf')
                
                # 最大重复限制：在采样前屏蔽超过限制的token
                max_rep = int(max_repetition) if max_repetition is not None else int(getattr(self.config, 'max_repetition', 3))
                for tok_id, cnt in list(token_counts.items()):
                    if cnt >= max_rep:
                        last_logits[tok_id] = float('-inf')
                
                # 贪婪解码：直接选择概率最高的token
                next_token = torch.argmax(last_logits).item()
                probs = torch.softmax(last_logits, dim=-1)
                probability = probs[next_token].item()
                
                # Yield token信息
                yield (next_token, probability)
                
                # 更新状态
                token_counts[next_token] += 1
                next_token_tensor = torch.tensor([[next_token]], device=device)
                cur_tokens = torch.cat([cur_tokens, next_token_tensor], dim=1)
                
                # 更新 n-gram 记录
                seq_now = [int(x) for x in cur_tokens[0].tolist()]
                L = len(seq_now)
                for n in range(2, ngram_block_size + 1):
                    if L >= n:
                        ng = tuple(seq_now[-n:])
                        seen_ngrams.add(ng)
                continue
            
            # 正常温度下的处理
            if temperature != 1.0:
                last_logits = last_logits / float(temperature)
            
            vocab_size = last_logits.size(0)
            
            # Top-K 采样
            if top_k is not None and top_k > 0 and top_k < vocab_size:
                topk_vals, _ = torch.topk(last_logits, top_k)
                threshold = topk_vals[-1]
                last_logits = torch.where(
                    last_logits < threshold,
                    torch.tensor(float('-inf'), device=last_logits.device),
                    last_logits
                )
            
            # Top-P (nucleus) 采样
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(last_logits, descending=True)
                sorted_probs = torch.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                keep_mask = cumulative_probs <= top_p
                if not keep_mask.any():
                    keep_mask[0] = True
                
                remove_indices = sorted_indices[~keep_mask]
                last_logits[remove_indices] = float('-inf')
            
            # 重复惩罚
            if repetition_penalty is not None and repetition_penalty != 1.0:
                generated = set(cur_tokens[0].tolist())
                for token_id in generated:
                    if last_logits[token_id] < 0:
                        last_logits[token_id] *= float(repetition_penalty)
                    else:
                        last_logits[token_id] /= float(repetition_penalty)
            
            # presence / frequency penalty
            if presence_penalty != 0.0 or frequency_penalty != 0.0:
                counts = Counter(cur_tokens[0].tolist())
                for tok_id, cnt in counts.items():
                    if presence_penalty != 0.0:
                        last_logits[tok_id] -= float(presence_penalty)
                    if frequency_penalty != 0.0 and cnt > 0:
                        last_logits[tok_id] -= float(frequency_penalty) * float(cnt)
            
            # 避免直接重复上一个 token
            if ban_immediate_repeat and cur_tokens.size(1) > 0:
                prev_token = int(cur_tokens[0, -1].item())
                last_logits[prev_token] = float('-inf')
            
            # n-gram 重复阻断
            if ngram_block_size > 1 and cur_tokens.size(1) >= 1:
                banned = []
                seq_list = [int(x) for x in cur_tokens[0].tolist()]
                for candidate in range(last_logits.size(0)):
                    will_form_repeat = False
                    for n in range(2, ngram_block_size + 1):
                        if len(seq_list) + 1 >= n:
                            prev_ngram = tuple(seq_list[-(n-1):] + [candidate])
                            if prev_ngram in seen_ngrams:
                                will_form_repeat = True
                                break
                    if will_form_repeat:
                        banned.append(candidate)
                if banned:
                    last_logits[torch.tensor(banned, device=last_logits.device)] = float('-inf')
            
            # 最大重复限制：在采样前屏蔽超过限制的token（与generate方法一致）
            max_rep = int(max_repetition) if max_repetition is not None else int(getattr(self.config, 'max_repetition', 3))
            for tok_id, cnt in list(token_counts.items()):
                if cnt >= max_rep:
                    last_logits[tok_id] = float('-inf')
            
            # 计算概率分布
            probs = torch.softmax(last_logits, dim=-1)
            if torch.isnan(probs).any():
                probs = torch.nn.functional.softmax(
                    last_logits.float().masked_fill(torch.isinf(last_logits), -1e9),
                    dim=-1
                )
            
            # 采样下一个 token
            next_token = torch.multinomial(probs, 1).item()
            probability = probs[next_token].item()
            
            # 检查最大重复限制
            max_rep = int(max_repetition) if max_repetition is not None else int(getattr(self.config, 'max_repetition', 3))
            if token_counts[next_token] >= max_rep:
                break
            
            # Yield token信息
            yield (next_token, probability)
            
            # 更新状态
            token_counts[next_token] += 1
            next_token_tensor = torch.tensor([[next_token]], device=device)
            cur_tokens = torch.cat([cur_tokens, next_token_tensor], dim=1)
            
            # 更新 n-gram 记录
            seq_now = [int(x) for x in cur_tokens[0].tolist()]
            L = len(seq_now)
            for n in range(2, ngram_block_size + 1):
                if L >= n:
                    ng = tuple(seq_now[-n:])
                    seen_ngrams.add(ng)


class ShannonB1Encoder(nn.Module):
    """编码器版本 (非自回归，用于理解任务)"""
    
    def __init__(self, config: ModelConfig):
        """
        初始化 ShannonB1Encoder 模型
        
        Args:
            config: 模型配置对象，包含词汇表大小、模型维度等参数
        """
        super().__init__()

        self.config = config
        
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_encoding = PositionalEncoding(config.d_model, config.max_seq_len, config.dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, config.num_layers)
        self.ln_f = nn.LayerNorm(config.d_model)
        self.output = nn.Linear(config.d_model, config.vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            tokens: (batch, seq_len) 输入 token IDs
        
        Returns:
            输出 logits: (batch, seq_len, vocab_size)
        """
        x = self.token_embedding(tokens) * math.sqrt(self.config.d_model)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        x = self.ln_f(x)
        return self.output(x)
```

---

## 第 4 步：`src/data/download.py` — 数据下载与加载 — load_all_data, load_data_chunks (分块流式)

```python
"""
数据下载工具
"""

import os
import urllib.request


def download_shakespeare(save_path: str = 'data/shakespeare.txt') -> str:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    
    try:
        urllib.request.urlretrieve(url, save_path)
        with open(save_path, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"✅ Downloaded Shakespeare: {len(text):,} chars")
        return save_path
    except Exception as e:
        print(f"⚠️ Download failed: {e}")
        return None


def load_shakespeare() -> str:
    local_path = 'data/shakespeare.txt'
    if os.path.exists(local_path):
        with open(local_path, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"✅ Loaded local Shakespeare: {len(text):,} chars from {local_path}")
        return text

    path = download_shakespeare()
    if path:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    print("⚠️ Using fallback sample text (download failed and no local file found).")
    return "To be or not to be, that is the question. " * 1000


def load_all_data(data_dir: str = 'data') -> list:
    texts = []
    if os.path.isdir(data_dir):
        for root, _, files in os.walk(data_dir):
            for fname in files:
                if fname.lower().endswith('.txt'):
                    fpath = os.path.join(root, fname)
                    try:
                        with open(fpath, 'r', encoding='utf-8') as f:
                            content = f.read()
                        if content.strip():
                            texts.append(content)
                            print(f"✅ Loaded: {fpath} ({len(content):,} chars)")
                    except Exception as e:
                        print(f"⚠️  Skip {fpath}: {e}")

    if not texts:
        print("⚠️  No .txt files found in data/, using fallback text")
        sample_path = create_sample_data()
        with open(sample_path, 'r', encoding='utf-8') as f:
            texts.append(f.read())

    total = sum(len(t) for t in texts)
    print(f"\n📚 Total: {len(texts)} files, {total:,} chars")
    return texts


def load_data_chunks(data_dir: str = 'data', chunk_size: int = 1_000_000):
    """
    分块读取 data 目录下所有 .txt 文件，返回生成器。
    适用于 BPE tokenizer 流式训练，避免一次性加载全部数据到内存。
    
    Args:
        data_dir: 数据文件夹路径
        chunk_size: 每块字符数，默认 1M
    
    Yields:
        str: 文本块
    """
    count = 0
    if not os.path.isdir(data_dir):
        return
    
    for root, _, files in os.walk(data_dir):
        for fname in sorted(files):
            if fname.lower().endswith('.txt'):
                fpath = os.path.join(root, fname)
                total = os.path.getsize(fpath)
                print(f"📖 Streaming: {fpath} ({total:,} bytes)")
                with open(fpath, 'r', encoding='utf-8') as f:
                    while True:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        count += 1
                        yield chunk
    print(f"✅ Yielded {count} chunks for training")


def create_sample_data(save_path: str = 'data/sample.txt') -> str:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    sample_text = """Once upon a time, there was a little girl named Alice. She lived in a small village
at the foot of a great mountain. Every day, she would look up at the mountain and
wonder what was at the top. One morning, she decided to find out. She packed a small
bag with some bread and cheese, and started climbing. The path was steep and rocky,
but Alice was determined. After many hours, she reached the top. There, she found a
beautiful garden full of flowers she had never seen before. In the middle of the
garden stood a small cottage. An old woman came out and smiled at Alice. "Welcome,"
she said, "I have been waiting for you." And so began Alice's greatest adventure."""
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(sample_text)
    
    print(f"✅ Sample data created: {save_path}")
    return save_path
```

---

## 第 5 步：`src/data/tokenizer.py` — 分词器 — CharTokenizer, BPETokenizer (Rust), SimpleBPETokenizer

```python
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
    
    def decode(self, tokens: List[int], skip_special: bool = True) -> str:
        if skip_special:
            skip_ids = {v for k, v in self.special_tokens.items() if k in ['<PAD>', '<BOS>', '<EOS>']}
            tokens = [t for t in tokens if t not in skip_ids]
        
        if self._legacy_mode:
            return self._legacy_decode(tokens)
        else:
            return self._hf.decode(tokens)
    
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
```

---

## 第 6 步：`src/data/dataset.py` — 数据集 — TextDataset (滑动窗口切片)

```python
"""
PyTorch 数据集类
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Tuple, Optional


class TextDataset(Dataset):
    """文本数据集"""
    
    def __init__(self, texts: List[str], tokenizer, seq_len: int = 64, stride: Optional[int] = None):
        """
        初始化文本数据集

        Args:
            texts: 输入的文本列表
            tokenizer: 用于文本编码的分词器对象
            seq_len: 每个序列的长度，默认为64
            stride: 滑动窗口的步长，如果未指定则使用seq_len的一半
        """
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.stride = stride or seq_len // 2
        self.data = []
        
        # 遍历所有文本，将其转换为token序列
        for text in texts:
            tokens = tokenizer.encode(text)
            # 使用滑动窗口提取固定长度的序列片段
            for i in range(0, len(tokens) - seq_len - 1, self.stride):
                seq = tokens[i:i + seq_len + 1]
                if len(seq) == seq_len + 1:
                    self.data.append(seq)
        
        print(f"📊 Dataset created: {len(self.data)} sequences")
    
    def __len__(self) -> int:
        """返回数据集中序列的数量"""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取指定索引的数据项

        Args:
            idx: 数据项的索引

        Returns:
            包含输入序列和目标序列的元组
        """
        seq = self.data[idx]
        return torch.tensor(seq[:-1]), torch.tensor(seq[1:])
    
    def get_vocab_size(self) -> int:
        """获取词汇表大小"""
        return self.tokenizer.get_vocab_size()


class StreamingTextDataset(Dataset):
    """流式数据集 (适合大文件，不占用内存)"""
    
    def __init__(self, filepath: str, tokenizer, seq_len: int = 64):
        """
        初始化流式文本数据集

        Args:
            filepath: 文本文件的路径
            tokenizer: 用于文本编码的分词器对象
            seq_len: 每个序列的长度，默认为64
        """
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.filepath = filepath
        self._load_file()
    
    def _load_file(self):
        """
        从文件中加载文本数据并生成序列
        
        读取整个文件内容，进行tokenization，并使用滑动窗口创建训练序列
        """
        with open(self.filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        tokens = self.tokenizer.encode(text)
        self.data = []
        # 使用滑动窗口将文本切分为固定长度的序列
        for i in range(0, len(tokens) - self.seq_len - 1, self.seq_len // 2):
            seq = tokens[i:i + self.seq_len + 1]
            if len(seq) == self.seq_len + 1:
                self.data.append(seq)
        
        print(f"📊 Streaming dataset: {len(self.data)} sequences from {filepath}")
    
    def __len__(self):
        """返回数据集中序列的数量"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        获取指定索引的数据项

        Args:
            idx: 数据项的索引

        Returns:
            包含输入序列和目标序列的元组
        """
        seq = self.data[idx]
        return torch.tensor(seq[:-1]), torch.tensor(seq[1:])
```

---

## 第 7 步：`src/training/scheduler.py` — 学习率调度器 — CosineAnnealingWarmupLR 等

```python
"""
学习率调度器 - 支持按 epoch 和按 step
"""

import math
import torch
from torch.optim import Optimizer


class CosineAnnealingLR:
    """余弦退火学习率调度器 (按 epoch)"""
    
    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float = 0):
        """
        初始化余弦退火学习率调度器
        
        Args:
            optimizer: PyTorch优化器对象
            T_max: 余弦退火周期的最大步数
            eta_min: 学习率的最小值，默认为0
        """
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.base_lr = optimizer.param_groups[0]['lr']
        self.step_num = 0
    
    def step(self):
        """
        更新学习率，按照余弦退火公式计算新的学习率
        """
        self.step_num += 1
        lr = self.eta_min + (self.base_lr - self.eta_min) * \
             (1 + math.cos(math.pi * self.step_num / self.T_max)) / 2
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def state_dict(self):
        """
        获取调度器的状态字典
        
        Returns:
            包含当前步数的状态字典
        """
        return {'step_num': self.step_num}
    
    def load_state_dict(self, state_dict):
        """
        从状态字典加载调度器状态
        
        Args:
            state_dict: 包含调度器状态的字典
        """
        self.step_num = state_dict['step_num']


class CosineAnnealingWarmupLR:
    """带预热的余弦退火 (按 step，适合大模型)"""
    
    def __init__(self, optimizer: Optimizer, warmup_steps: int, total_steps: int, 
                 min_lr: float = 1e-6, initial_lr: float = 1e-7):
        """
        初始化带预热的余弦退火学习率调度器
        
        Args:
            optimizer: PyTorch优化器对象
            warmup_steps: 预热步数
            total_steps: 总训练步数
            min_lr: 最小学习率，默认为1e-6
            initial_lr: 初始学习率，默认为1e-7
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.initial_lr = initial_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.step_num = 0
    
    def step_per_batch(self):
        """
        按批次更新学习率，在预热阶段使用线性增长，在其余阶段使用余弦退火
        """
        self.step_num += 1
        
        if self.step_num < self.warmup_steps:
            # 线性预热阶段：从初始学习率线性增长到基础学习率
            lr = self.initial_lr + (self.base_lr - self.initial_lr) * (self.step_num / self.warmup_steps)
        else:
            # 余弦退火阶段：从基础学习率按余弦曲线衰减到最小学习率
            progress = (self.step_num - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress)) / 2
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def step(self):
        """按 epoch 调度 (不做任何事)"""
        pass
    
    def state_dict(self):
        """
        获取调度器的状态字典
        
        Returns:
            包含当前步数的状态字典
        """
        return {'step_num': self.step_num}
    
    def load_state_dict(self, state_dict):
        """
        从状态字典加载调度器状态
        
        Args:
            state_dict: 包含调度器状态的字典
        """
        self.step_num = state_dict['step_num']


class StepLR:
    """阶梯衰减学习率调度器"""
    
    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1):
        """
        初始化阶梯衰减学习率调度器
        
        Args:
            optimizer: PyTorch优化器对象
            step_size: 学习率衰减的步长间隔
            gamma: 学习率衰减的乘数因子，默认为0.1
        """
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.step_num = 0
    
    def step(self):
        """
        每隔step_size步将学习率乘以gamma进行衰减
        """
        self.step_num += 1
        if self.step_num % self.step_size == 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= self.gamma
    
    def state_dict(self):
        """
        获取调度器的状态字典
        
        Returns:
            包含当前步数的状态字典
        """
        return {'step_num': self.step_num}
    
    def load_state_dict(self, state_dict):
        """
        从状态字典加载调度器状态
        
        Args:
            state_dict: 包含调度器状态的字典
        """
        self.step_num = state_dict['step_num']


class LinearWarmupLR:
    """线性预热学习率调度器"""
    
    def __init__(self, optimizer: Optimizer, warmup_steps: int, target_lr: float):
        """
        初始化线性预热学习率调度器
        
        Args:
            optimizer: PyTorch优化器对象
            warmup_steps: 预热步数
            target_lr: 预热结束后的目标学习率
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.target_lr = target_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.step_num = 0
    
    def step(self):
        """
        在预热步数内线性增加学习率
        """
        self.step_num += 1
        if self.step_num <= self.warmup_steps:
            lr = self.base_lr + (self.target_lr - self.base_lr) * (self.step_num / self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
    
    def state_dict(self):
        """
        获取调度器的状态字典
        
        Returns:
            包含当前步数的状态字典
        """
        return {'step_num': self.step_num}
    
    def load_state_dict(self, state_dict):
        """
        从状态字典加载调度器状态
        
        Args:
            state_dict: 包含调度器状态的字典
        """
        self.step_num = state_dict['step_num']


class ReduceLROnPlateau:
    """当验证损失停止下降时降低学习率"""
    
    def __init__(self, optimizer: Optimizer, patience: int = 5, factor: float = 0.5, 
                 min_lr: float = 1e-7, verbose: bool = True):
        """
        初始化基于验证损失的学习率调度器
        
        Args:
            optimizer: PyTorch优化器对象
            patience: 在降低学习率之前等待的epoch数
            factor: 学习率衰减的乘数因子
            min_lr: 最小学习率阈值
            verbose: 是否打印学习率变化信息
        """
        self.optimizer = optimizer
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.verbose = verbose
        self.best_loss = float('inf')
        self.counter = 0
    
    def step(self, val_loss: float):
        """
        根据验证损失更新学习率
        
        Args:
            val_loss: 当前验证损失值
        """
        if val_loss < self.best_loss - 1e-4:
            # 验证损失有所改善，更新最佳损失并重置计数器
            self.best_loss = val_loss
            self.counter = 0
        else:
            # 验证损失未改善，增加计数器
            self.counter += 1
            if self.counter >= self.patience:
                # 计数器达到耐心值，降低学习率
                current_lr = self.optimizer.param_groups[0]['lr']
                new_lr = max(current_lr * self.factor, self.min_lr)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr
                if self.verbose:
                    print(f"📉 Reducing learning rate from {current_lr:.6f} to {new_lr:.6f}")
                self.counter = 0
```

---

## 第 8 步：`src/training/trainer.py` — 训练器 — ImprovedTrainer (AMP, 梯度累积, 早停)

```python
"""
训练器模块 - 包含混合精度、梯度累积、早停、TensorBoard
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import amp
from tqdm import tqdm
import os
import time
from datetime import datetime
from typing import Optional, Dict, Any
import numpy as np
import torch.nn.functional as F

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("⚠️ TensorBoard not available. Install with: pip install tensorboard")


class ImprovedTrainer:
    """改进版训练器 - 支持混合精度、梯度累积、早停"""
    
    def __init__(self, model: nn.Module, dataloader: DataLoader, 
                 val_dataloader: Optional[DataLoader] = None,
                 config=None, optimizer: Optional[torch.optim.Optimizer] = None,
                 scheduler: Optional = None):
        """
        初始化改进版训练器
        
        Args:
            model: 要训练的神经网络模型
            dataloader: 训练数据的数据加载器
            val_dataloader: 验证数据的数据加载器，可选
            config: 训练配置对象
            optimizer: 优化器，如果为None则创建AdamW优化器
            scheduler: 学习率调度器，可选
        """
        self.model = model
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.device = config.device if hasattr(config, 'device') else 'cpu'
        
        # 优化器
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        else:
            self.optimizer = optimizer
        
        # 学习率调度器
        self.scheduler = scheduler
        
        # 损失函数（若使用 label smoothing，将使用自定义计算）
        self.criterion = None
        
        # 混合精度训练 - 跨CUDA版本兼容
        self.use_amp = config.use_amp and self.device == 'cuda'
        self.scaler = None
        
        if self.use_amp:
            try:
                # PyTorch >= 1.10: GradScaler支持device_type参数
                self.scaler = amp.GradScaler(device_type='cuda')
            except TypeError:
                try:
                    # 较旧版本：GradScaler不接受device_type参数
                    self.scaler = amp.GradScaler()
                except Exception as e:
                    print(f"⚠️ Failed to initialize GradScaler: {e}")
                    print("   Disabling mixed precision training")
                    self.use_amp = False
                    self.scaler = None

        # 梯度累积
        self.grad_accum_steps = config.gradient_accumulation_steps

        # 历史记录
        self.history = {
            'train_loss': [], 
            'val_loss': [], 
            'lr': [],
            'time': [],
            'epoch_time': []
        }
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0

        # TensorBoard
        self.writer = None
        if TENSORBOARD_AVAILABLE and hasattr(config, 'tensorboard_dir'):
            self.writer = SummaryWriter(config.tensorboard_dir)

        # 统计信息
        self.global_step = 0
        self.start_time = None

    def _autocast(self):
        """
        返回跨PyTorch版本兼容的autocast上下文管理器
        
        Returns:
            与当前PyTorch版本兼容的自动混合精度上下文管理器
        """
        if not self.use_amp:
            # no autocast when not using amp
            from contextlib import nullcontext
            return nullcontext()

        # 确定设备类型
        device_type = 'cuda' if self.device == 'cuda' else 'cpu'
        
        try:
            # PyTorch >= 1.10: 使用device_type参数（推荐方式）
            return amp.autocast(device_type=device_type)
        except TypeError:
            try:
                # 更旧的版本：尝试不带参数
                return amp.autocast()
            except TypeError:
                try:
                    # 非常旧的版本：使用torch.cuda.amp.autocast
                    from torch.cuda.amp import autocast as cuda_autocast
                    return cuda_autocast()
                except Exception as e:
                    print(f"⚠️ Failed to create autocast context: {e}")
                    print("   Falling back to no autocast")
                    from contextlib import nullcontext
                    return nullcontext()
    
    def train_epoch(self) -> float:
        """
        训练一个epoch (支持混合精度和梯度累积)
        
        Returns:
            当前epoch的平均训练损失
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # 进度条
        pbar = tqdm(self.dataloader, desc="Training")
        self.optimizer.zero_grad()
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # 前向传播并计算损失（支持 label smoothing）
            if self.use_amp:
                with self._autocast():
                    outputs = self.model(inputs)
                    # 模型返回 (logits, past_key_values)，训练时只需要logits
                    logits = outputs[0] if isinstance(outputs, tuple) else outputs
                    loss = self._compute_loss(logits, targets)
                    loss = loss / self.grad_accum_steps

                # 反向传播 (混合精度)
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(inputs)
                # 模型返回 (logits, past_key_values)，训练时只需要logits
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                loss = self._compute_loss(logits, targets)
                loss = loss / self.grad_accum_steps
                loss.backward()
            
            total_loss += loss.item() * self.grad_accum_steps
            num_batches += 1
            
            # 梯度累积更新
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                # 梯度裁剪
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # 更新学习率调度器 (按步数)
                if self.scheduler and hasattr(self.scheduler, 'step_per_batch'):
                    self.scheduler.step_per_batch()
            
            # 更新进度条
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({'loss': loss.item() * self.grad_accum_steps, 'lr': f'{current_lr:.2e}'})
            
            # TensorBoard 记录 (每 N 步)
            if self.writer and self.global_step % self.config.log_interval == 0:
                self.writer.add_scalar('train/loss_step', loss.item() * self.grad_accum_steps, self.global_step)
                self.writer.add_scalar('train/lr', current_lr, self.global_step)
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    @torch.no_grad()
    def validate(self) -> float:
        """
        验证模型性能
        
        Returns:
            验证集上的平均损失
        """
        if self.val_dataloader is None:
            return 0.0
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.val_dataloader, desc="Validating")
        for inputs, targets in pbar:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # 验证时也使用autocast以保持一致性
            if self.use_amp:
                with self._autocast():
                    outputs = self.model(inputs)
                    # 模型返回 (logits, past_key_values)，验证时只需要logits
                    logits = outputs[0] if isinstance(outputs, tuple) else outputs
                    loss = self._compute_loss(logits, targets)
            else:
                outputs = self.model(inputs)
                # 模型返回 (logits, past_key_values)，验证时只需要logits
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                loss = self._compute_loss(logits, targets)
            
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'val_loss': loss.item()})
        
        return total_loss / num_batches

    def _compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算损失，支持 label smoothing（基于交叉熵的平滑实现）。

        Args:
            logits: 模型输出的logits张量，形状为(batch, seq_len, vocab)
            targets: 目标标签张量，形状为(batch, seq_len)

        Returns:
            标量损失值
        """
        smoothing = getattr(self.config, 'label_smoothing', 0.0)
        vocab_size = logits.size(-1)

        if smoothing is None or smoothing <= 0.0:
            # 标准交叉熵
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1), reduction='mean')
            return loss

        # label smoothing 实现（参考 HuggingFace/Transformer 常用实现）
        log_probs = F.log_softmax(logits, dim=-1)  # (B, L, V)
        nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)  # (B, L)
        smooth_loss = -log_probs.mean(dim=-1)  # (B, L)

        loss = (1.0 - smoothing) * nll_loss + smoothing * smooth_loss
        return loss.mean()
    
    def should_early_stop(self, val_loss: float) -> bool:
        """
        检查是否应该早停

        Args:
            val_loss: 当前验证损失

        Returns:
            如果满足早停条件返回True，否则返回False
        """
        if val_loss < self.best_val_loss - self.config.early_stopping_min_delta:
            self.best_val_loss = val_loss
            self.best_epoch = len(self.history['val_loss']) - 1
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.config.early_stopping_patience
    
    def train(self, epochs: int) -> Dict[str, list]:
        """
        完整训练循环

        Args:
            epochs: 训练轮数

        Returns:
            包含训练历史记录的字典
        """
        print("\n" + "=" * 70)
        print("🚀 Starting Training")
        print(f"   Device: {self.device.upper()}")
        print(f"   Mixed Precision: {'ON' if self.use_amp else 'OFF'}")
        print(f"   Gradient Accumulation: {self.grad_accum_steps}")
        print(f"   Batch Size (effective): {self.config.batch_size * self.grad_accum_steps}")
        print("=" * 70)
        
        self.start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # 训练
            train_loss = self.train_epoch()
            
            # 验证
            val_loss = self.validate()
            
            # 更新学习率调度器 (按 epoch)
            if self.scheduler and hasattr(self.scheduler, 'step'):
                self.scheduler.step()
            
            epoch_time = time.time() - epoch_start
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss if val_loss > 0 else train_loss)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            self.history['time'].append(epoch_time)
            self.history['epoch_time'].append(epoch_time)
            
            # TensorBoard 记录 (每 epoch)
            if self.writer:
                self.writer.add_scalar('train/loss_epoch', train_loss, epoch)
                self.writer.add_scalar('val/loss', val_loss, epoch)
                self.writer.add_scalar('train/lr_epoch', self.optimizer.param_groups[0]['lr'], epoch)
            
            # 输出进度
            print(f"\n📊 Epoch {epoch+1}/{epochs}")
            print(f"   Train Loss: {train_loss:.4f}")
            if val_loss > 0:
                print(f"   Val Loss: {val_loss:.4f}")
            print(f"   LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            print(f"   Time: {epoch_time:.1f}s")
            print(f"   Best Val Loss: {self.best_val_loss:.4f} (Epoch {self.best_epoch+1})")
            
            # 保存最佳模型
            if val_loss > 0 and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.save_checkpoint('checkpoints/shannon_b1_best.pt')
                print(f"   💾 Saved best model (val_loss: {val_loss:.4f})")
            
            # 定期保存检查点
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(f'checkpoints/shannon_b1_epoch{epoch+1}.pt')
            
            # 早停检查
            if val_loss > 0 and self.should_early_stop(val_loss):
                print(f"\n🛑 Early stopping triggered after {epoch+1} epochs")
                print(f"   Best val loss: {self.best_val_loss:.4f} at epoch {self.best_epoch+1}")
                break
            
            print("-" * 70)
        
        total_time = time.time() - self.start_time
        print(f"\n✅ Training completed!")
        print(f"   Total time: {total_time / 60:.1f} minutes")
        print(f"   Best val loss: {self.best_val_loss:.4f}")
        
        # 关闭 TensorBoard writer
        if self.writer:
            self.writer.close()
        
        return self.history
    
    @staticmethod
    def _is_lora_checkpoint(state_dict: Dict[str, Any]) -> bool:
        """检测 checkpoint 是否为 LoRA 格式"""
        for k in state_dict.keys():
            if 'lora_A' in k or 'lora_B' in k or '.linear.weight' in k:
                return True
        return False

    @staticmethod
    def _remap_lora_to_standard(state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """将 LoRA 格式 state_dict 重映射为标准 nn.Linear 格式"""
        new_state = {}
        for key, value in state_dict.items():
            if 'lora_A' in key or 'lora_B' in key:
                continue
            new_key = key.replace('.linear.weight', '.weight').replace('.linear.bias', '.bias')
            new_state[new_key] = value
        return new_state

    def save_checkpoint(self, path: str):
        """
        保存检查点

        Args:
            path: 检查点保存路径
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # 如果使用了 LoRA，先合并权重到基础模型再保存，确保 checkpoint 使用标准 key 结构
        use_lora = getattr(self.config, 'use_lora', False)
        lora_replacements = []  # 记录替换的层以便恢复
        if use_lora and hasattr(self.model, 'merge_lora_weights'):
            self.model.merge_lora_weights()
            # 将 LoRALinear 替换为普通 nn.Linear，使 state_dict 产出标准 key
            from src.model.layers import LoRALinear
            for layer in self.model.decoder_layers:
                for target in self.config.lora_target_modules:
                    attr = getattr(layer.self_attn, target, None)
                    if isinstance(attr, LoRALinear):
                        lora_replacements.append((layer.self_attn, target, attr))
                        setattr(layer.self_attn, target, attr.linear)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'config': self.config,
            'global_step': self.global_step
        }
        torch.save(checkpoint, path)
        # 保存完成后恢复 LoRA 结构（继续训练）
        for parent, target, lora_obj in lora_replacements:
            setattr(parent, target, lora_obj)
        if use_lora and hasattr(self.model, 'unmerge_lora_weights'):
            self.model.unmerge_lora_weights()
        print(f"   💾 Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str):
        """
        加载检查点

        Args:
            path: 检查点文件路径
        """
        # 尝试显式允许加载包含自定义类的完整检查点（非 weights-only）
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            # 兼容较旧的 PyTorch 版本或不支持 weights_only 参数的环境
            checkpoint = torch.load(path, map_location=self.device)
        except Exception as e:
            # 如果受限于 safe globals，提醒用户并重抛
            print(f"⚠️ Failed to load checkpoint with weights_only=False: {e}")
            raise
        ckpt_state = checkpoint.get('model_state_dict', {})

        # 检测并重映射旧格式 LoRA checkpoint 为标准 key 格式
        if self._is_lora_checkpoint(ckpt_state):
            from src.model.layers import LoRALinear
            ckpt_state = self._remap_lora_to_standard(ckpt_state)
            print("   [Info] Detected LoRA-format checkpoint, remapped to standard keys")

        # 尝试按形状安全地加载模型权重：仅替换形状匹配的参数
        model_state = self.model.state_dict()
        matched_keys = []
        skipped_keys = []

        for k, v in ckpt_state.items():
            if k in model_state and v.size() == model_state[k].size():
                model_state[k] = v
                matched_keys.append(k)
            else:
                skipped_keys.append(k)

        # 用更新后的 state_dict 加载（包含被跳过的原始参数）
        self.model.load_state_dict(model_state)
        print(f"✅ Loaded model weights: {len(matched_keys)} params matched, {len(skipped_keys)} skipped")

        # 如果存在被跳过的参数（形状不匹配），不要加载 optimizer/scheduler/scaler 状态
        if skipped_keys:
            print("⚠️ Some parameters were skipped due to shape mismatch; skipping optimizer/scheduler/scaler state load to avoid errors.")
        else:
            # 尝试加载优化器和调度器状态，如果不兼容则跳过并提示
            try:
                if 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] is not None:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print("✅ Optimizer state loaded")
            except Exception as e:
                print(f"⚠️ Could not load optimizer state (skipped): {e}")

            try:
                if self.scheduler and checkpoint.get('scheduler_state_dict'):
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    print("✅ Scheduler state loaded")
            except Exception as e:
                print(f"⚠️ Could not load scheduler state (skipped): {e}")

            try:
                if self.scaler and checkpoint.get('scaler_state_dict'):
                    self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                    print("✅ AMP scaler state loaded")
            except Exception as e:
                print(f"⚠️ Could not load scaler state (skipped): {e}")
        
        self.history = checkpoint.get('history', self.history)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_epoch = checkpoint.get('best_epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        
        print(f"✅ Loaded checkpoint: {path}")
        print(f"   Best val loss: {self.best_val_loss:.4f} (Epoch {self.best_epoch+1})")
```

---

## 第 9 步：`src/utils/helpers.py` — 工具函数 — set_seed, get_cuda_info, get_device

```python
"""
工具函数
"""

import random
import numpy as np
import torch
from datetime import datetime


def set_seed(seed: int = 42):
    """
    设置随机种子以确保实验可复现性
    
    Args:
        seed (int): 随机种子值，默认为42
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # 仅在 CUDA 可用时设置 CUDA 相关种子与 cudnn 选项
    try:
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # 保持确定性以便复现（可能降低性能），如需性能可在命令行关闭
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def get_cuda_info() -> dict:
    """
    获取CUDA环境详细信息，用于诊断和日志记录
    
    Returns:
        dict: 包含CUDA版本、驱动版本、设备信息等字典
    """
    info = {
        'cuda_available': False,
        'cuda_version': None,
        'cudnn_version': None,
        'device_count': 0,
        'devices': []
    }
    
    try:
        if not torch.cuda.is_available():
            return info
        
        info['cuda_available'] = True
        info['cuda_version'] = torch.version.cuda
        info['cudnn_version'] = torch.backends.cudnn.version()
        info['device_count'] = torch.cuda.device_count()
        
        for i in range(info['device_count']):
            device_info = {
                'index': i,
                'name': torch.cuda.get_device_name(i),
                'memory_total': torch.cuda.get_device_properties(i).total_memory,
                'memory_allocated': torch.cuda.memory_allocated(i),
                'memory_reserved': torch.cuda.memory_reserved(i),
                'compute_capability': torch.cuda.get_device_capability(i)
            }
            info['devices'].append(device_info)
            
    except Exception as e:
        print(f"⚠️ Error getting CUDA info: {e}")
    
    return info


def get_device() -> str:
    """
    获取当前可用的计算设备
    
    Returns:
        str: 可用设备名称，优先级为 cuda > mps > cpu
    """
    if torch.cuda.is_available():
        return "cuda"
    # 有些 PyTorch 版本可能不包含 mps backend，检查属性再调用
    if hasattr(torch.backends, 'mps') and callable(getattr(torch.backends.mps, 'is_available', None)):
        try:
            if torch.backends.mps.is_available():
                return 'mps'
        except Exception:
            pass
    return "cpu"


def format_time(seconds: float) -> str:
    """
    将秒数格式化为 HH:MM:SS 格式的时间字符串
    
    Args:
        seconds (float): 秒数
        
    Returns:
        str: 格式化后的时间字符串，格式为 HH:MM:SS
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def count_parameters(model: torch.nn.Module) -> int:
    """
    统计PyTorch模型中需要梯度更新的参数总量
    
    Args:
        model (torch.nn.Module): PyTorch模型
        
    Returns:
        int: 模型参数总数
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

```

---

## 第 10 步：`scripts/train.py` — 训练入口脚本 — CLI 参数解析 + 训练流程

```python
#!/usr/bin/env python
"""
Shannon-b1 PyTorch 训练脚本 - 完整改进版
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, random_split
import argparse
from datetime import datetime

from src.model import ShannonB1, ModelConfig
from src.data import TextDataset, create_tokenizer, create_tokenizer_streaming, load_all_data, download_shakespeare
from src.training import ImprovedTrainer, CosineAnnealingWarmupLR
from src.utils import set_seed, get_device


def parse_args():
    parser = argparse.ArgumentParser(description='Shannon-b1 Training')
    
    parser.add_argument('--d-model', type=int, default=128)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--d-ff', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seq-len', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--grad-accum', type=int, default=1)
    
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--warmup-steps', type=int, default=1000)
    parser.add_argument('--gradient-checkpointing', action='store_true', help='Enable gradient checkpointing to save memory')
    parser.add_argument('--norm-type', type=str, default='layernorm', choices=['layernorm', 'rmsnorm'], help='Normalization type')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--label-smoothing', type=float, default=0.0)
    parser.add_argument('--tie-embeddings', action='store_true', help='Tie token embedding and output projection')
    parser.add_argument('--patience', type=int, default=10)
    
    parser.add_argument('--use-rope', action='store_true', help='Enable RoPE (Rotary Positional Embeddings)')
    parser.add_argument('--rope-base', type=float, default=10000.0)
    parser.add_argument('--sliding-window-size', type=int, default=None)
    parser.add_argument('--use-alibi', action='store_true', help='Enable ALiBi (not recommended with RoPE)')
    
    parser.add_argument('--lora', action='store_true', help='Enable LoRA fine-tuning')
    parser.add_argument('--lora-rank', type=int, default=8)
    parser.add_argument('--lora-alpha', type=float, default=16.0)
    parser.add_argument('--lora-dropout', type=float, default=0.0)
    parser.add_argument('--lora-target-modules', type=str, nargs='+', default=['q_proj', 'v_proj'])
    
    parser.add_argument('--tokenizer', type=str, default='char', choices=['char', 'bpe'])
    parser.add_argument('--vocab-size', type=int, default=2000)
    # 分块训练参数（避免大数据集 OOM）
    parser.add_argument('--stream-chunk-size', type=int, default=1_000_000,
                        help='Chunk size (chars) for streaming BPE training. Default: 1M')
    
    parser.add_argument('--device', type=str, default=get_device())
    parser.add_argument('--save-path', type=str, default='checkpoints/shannon_b1.pt')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--getdata', type=str, default=None,
                        help='Download dataset before training, e.g. --getdata shakespeare')
    
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    
    print("=" * 70)
    print("Shannon-b1 Improved Training")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {args.device.upper()}")
    print(f"Mixed Precision: {'OFF' if args.no_amp else 'ON'}")
    print(f"Grad Accum: {args.grad_accum}")
    print("=" * 70)
    
    if args.device == 'cuda':
        try:
            from src.utils import get_cuda_info
            cuda_info = get_cuda_info()
            print(f"\n🔧 CUDA Environment:")
            print(f"   CUDA Version: {cuda_info['cuda_version']}")
            print(f"   cuDNN Version: {cuda_info['cudnn_version']}")
            print(f"   Device Count: {cuda_info['device_count']}")
            for device in cuda_info['devices']:
                print(f"\n   GPU {device['index']}:")
                print(f"      Name: {device['name']}")
                print(f"      Compute Capability: {device['compute_capability']}")
                print(f"      Total Memory: {device['memory_total'] / 1024**3:.2f} GB")
            print()
        except Exception as e:
            print(f"⚠️ Could not retrieve detailed CUDA info: {e}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n")
    
    if args.getdata:
        if args.getdata == 'shakespeare':
            download_shakespeare()
        else:
            print(f"❌ Unknown dataset: {args.getdata}")
            print("   Available: shakespeare")

    print("📚 Loading data...")
    texts = load_all_data()
    total_chars = sum(len(t) for t in texts)
    
    # 大数据集 (>50MB) 且使用 BPE 时，使用分块流式训练
    if args.tokenizer == 'bpe' and total_chars > 50_000_000:
        print(f"📊 Large dataset detected ({total_chars:,} chars), using streaming BPE training...")
        tokenizer = create_tokenizer_streaming(
            tokenizer_type='bpe',
            vocab_size=args.vocab_size,
            data_dir='data',
            chunk_size=args.stream_chunk_size,
        )
    else:
        combined_text = "\n\n".join(texts)
        tokenizer = create_tokenizer(combined_text, args.tokenizer, args.vocab_size)
    
    full_dataset = TextDataset(texts, tokenizer, args.seq_len)
    val_size = int(len(full_dataset) * 0.1)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    vocab_size = tokenizer.get_vocab_size()
    print(f"   Vocab: {vocab_size}")
    print(f"   Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    config = ModelConfig(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        max_seq_len=args.seq_len,
        dropout=args.dropout,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.grad_accum,
        use_amp=not args.no_amp,
        seq_len=args.seq_len,
        device=args.device,
        early_stopping_patience=args.patience,
        label_smoothing=args.label_smoothing,
        lr_warmup_steps=args.warmup_steps,
        tie_word_embeddings=args.tie_embeddings,
        gradient_checkpointing=args.gradient_checkpointing,
        norm_type=args.norm_type,
        use_rope=args.use_rope,
        rope_base=args.rope_base,
        sliding_window_size=args.sliding_window_size,
        use_alibi=args.use_alibi,
    )
    
    print("\n🏗️ Creating model...")
    model = ShannonB1(config).to(args.device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params:,}")
    print(f"   Size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    if args.lora:
        print("\n🔧 Applying LoRA...")
        model.apply_lora(
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
        )
        config.learning_rate = args.lr * 10
        print(f"   LoRA learning rate adjusted to: {config.learning_rate}")
    
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lname = name.lower()
        if lname.endswith('bias') or 'norm' in lname or 'ln_' in lname or 'rmsnorm' in lname or 'embedding' in lname or 'pos_embedding' in lname:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {'params': decay_params, 'weight_decay': config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ],
        lr=config.learning_rate
    )
    
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    scheduler = CosineAnnealingWarmupLR(optimizer, warmup_steps=args.warmup_steps, total_steps=total_steps)
    
    trainer = ImprovedTrainer(model, train_loader, val_loader, config, optimizer, scheduler)
    if args.resume:
        if os.path.exists(args.resume):
            trainer.load_checkpoint(args.resume)
        else:
            print(f"⚠️ Resume checkpoint not found: {args.resume}")

    history = trainer.train(args.epochs)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    arch_version = _generate_arch_version_string(config)
    base_path = args.save_path.replace('.pt', '')
    versioned_path = f"{base_path}_{arch_version}_{timestamp}.pt"
    
    trainer.save_checkpoint(versioned_path)
    tokenizer.save(versioned_path.replace('.pt', '_tokenizer.json'))
    
    print(f"\n💾 Saved: {versioned_path}")
    print(f"📋 Architecture: {arch_version}")
    
    return history


def _generate_arch_version_string(config: ModelConfig) -> str:
    parts = []
    parts.append(f"dm{config.d_model}")
    parts.append(f"nl{config.num_layers}")
    parts.append(f"nh{config.num_heads}")
    
    if config.use_rope:
        rope_base_short = int(config.rope_base) if config.rope_base == int(config.rope_base) else config.rope_base
        parts.append(f"rope{int(rope_base_short)}")
    elif config.use_alibi:
        parts.append("alibi")
    else:
        parts.append("fixed")
    
    if config.sliding_window_size:
        parts.append(f"sw{config.sliding_window_size}")
    
    parts.append(config.norm_type.replace('norm', ''))
    
    if config.tie_word_embeddings:
        parts.append("tie")
    if config.gradient_checkpointing:
        parts.append("ckpt")
    
    return "_".join(parts)


if __name__ == "__main__":
    main()
```

---

## 第 11 步：`scripts/generate.py` — 推理入口脚本 — 加载模型 + 交互式生成 / 流式输出

```python
#!/usr/bin/env python
"""
流式文本生成脚本 - 支持单次生成和交互式多轮对话
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
import readline
import shutil

import torch

from src.model import ShannonB1, ModelConfig
from src.data import CharTokenizer, BPETokenizer
from src.utils import Conversation, get_template_by_name


# EOS token ID for CharTokenizer (index 3, '<EOS>')
EOS_TOKEN_ID = 3


def load_model(model_path, device="cpu"):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if "config" in checkpoint:
        config = checkpoint["config"]
    elif "model_config" in checkpoint:
        config = checkpoint["model_config"]
    else:
        state_dict = checkpoint["model_state_dict"]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        d_model = state_dict["token_embedding.weight"].shape[1]
        max_seq_len = state_dict["pos_encoding.pe"].shape[1]
        config = ModelConfig(vocab_size=vocab_size, d_model=d_model, max_seq_len=max_seq_len)

    model = ShannonB1(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tokenizer_path = model_path.replace(".pt", "_tokenizer.json")
    if os.path.exists(tokenizer_path):
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "char_to_idx" in data:
            tokenizer = CharTokenizer()
            tokenizer.load(tokenizer_path)
        else:
            tokenizer = BPETokenizer()
            tokenizer.load(tokenizer_path)
    else:
        tokenizer = CharTokenizer()
        tokenizer.build_vocab(["sample text"], 1000)

    return model, tokenizer, config


def _extract_assistant_reply(full_text, input_prompt):
    """从完整输出中提取助手的回复部分"""
    if full_text.startswith(input_prompt):
        reply = full_text[len(input_prompt) :].strip()
    else:
        reply = full_text.strip()
    markers = ["[ASSISTANT] ", "[ASSISTANT]", "<|im_start|>assistant\n", "assistant\n"]
    for marker in markers:
        if reply.startswith(marker):
            reply = reply[len(marker) :].strip()
    return reply


def _clean_new_text(text):
    """去除新生成文本中可能的模板标记"""
    markers = [
        "[ASSISTANT]\n", "[ASSISTANT] ", "[ASSISTANT]",
        "<|im_start|>assistant\n", "<|im_start|>assistant",
        "assistant\n", "assistant ",
    ]
    for m in markers:
        if text.startswith(m):
            text = text[len(m) :]
    return text


def _get_terminal_width():
    """Get terminal width, default to 80 if unavailable"""
    try:
        return shutil.get_terminal_size().columns
    except Exception:
        return 80


def single_generate(model, tokenizer, args):
    """单次生成模式"""
    print(f"\n{'='*60}")
    if args.system_prompt:
        print(f"🤖 System Prompt: {args.system_prompt}")
        print(f"{'-'*60}")
    print(f"💬 User Prompt: {args.prompt}")
    print(f"{'='*60}\n")

    # Build full prompt
    if args.system_prompt and args.system_prompt.strip():
        template = get_template_by_name(args.conv_template)
        conv = Conversation(system_prompt=args.system_prompt, template=template)
        conv.add_user(args.prompt)
        full_prompt = conv.build_prompt()
    else:
        full_prompt = args.prompt

    start_tokens = tokenizer.encode(full_prompt)[:50]
    prompt_len = len(start_tokens)
    generated_tokens = list(start_tokens)
    start_time = time.time()

    print("🚀 开始流式生成:\n")
    print(f"{args.prompt}", end="", flush=True)

    try:
        for token_id, probability in model.generate_stream(
            start_tokens,
            args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        ):
            # Stop on EOS
            if token_id == EOS_TOKEN_ID:
                break

            generated_tokens.append(token_id)
            new_tokens = generated_tokens[prompt_len:]
            new_text = tokenizer.decode(new_tokens)
            new_text = new_text.replace("</w>", " ").replace("  ", " ")
            new_text = _clean_new_text(new_text)

            # Use \r only for single-line; for multi-line print fresh
            if "\n" in new_text:
                # Clear terminal line and reprint
                print(f"\r{args.prompt}{new_text}", end="", flush=True)
            else:
                print(f"\r{args.prompt}{new_text}", end="", flush=True)

            if args.delay > 0:
                time.sleep(args.delay)

        elapsed = time.time() - start_time
        tokens_gen = len(generated_tokens) - prompt_len
        print(f"\n\n{'='*60}")
        print("✅ 生成完成!")
        print(f"📊 统计: {tokens_gen} tokens / {elapsed:.2f}s / {tokens_gen/elapsed:.1f} t/s")
        print(f"{'='*60}")

    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断生成")
    except Exception as e:
        print(f"\n\n❌ 生成出错: {e}")


def interactive_chat(model, tokenizer, args):
    """交互式多轮对话模式"""
    template = get_template_by_name(args.conv_template)

    if args.load_conv:
        try:
            conv = Conversation.from_json(args.load_conv)
            print(f"📂 已加载对话历史: {args.load_conv}")
            print(f"   消息数: {len(conv)}")
            if conv.system_prompt:
                print(f"   系统提示词: {conv.system_prompt}")
            for msg in conv.history[-6:]:
                icon = "🧑" if msg.role == "user" else "🤖"
                preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
                print(f"   {icon} [{msg.role}]: {preview}")
        except Exception as e:
            print(f"⚠️  加载对话失败: {e}")
            print("   将创建新对话")
            conv = Conversation(
                system_prompt=args.system_prompt,
                template=template,
                max_context_length=args.max_context,
            )
    else:
        conv = Conversation(
            system_prompt=args.system_prompt,
            template=template,
            max_context_length=args.max_context,
        )

    print(f"\n{'='*60}")
    print("🤖 Shannon-b1 多轮对话模式")
    print(f"{'='*60}")
    print(f"📋 模板: {args.conv_template}")
    if conv.system_prompt:
        print(f"🎯 系统提示词: {conv.system_prompt}")
    print(f"📐 最大上下文: {args.max_context} 字符")
    print(f"\n命令:")
    print(f"  /clear          清空对话历史（保留系统提示词）")
    print(f"  /save [path]    保存对话到文件")
    print(f"  /history        显示对话历史")
    print(f"  /system <text>  修改系统提示词")
    print(f"  /stats          显示对话统计")
    print(f"  /exit           退出对话")
    print(f"{'='*60}\n")

    while True:
        try:
            user_input = input("🧑 你: ").strip()
            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("/"):
                cmd = user_input.split(maxsplit=1)
                command = cmd[0].lower()
                cmd_arg = cmd[1] if len(cmd) > 1 else ""

                if command == "/exit":
                    save_prompt = input("💾 是否保存当前对话? (y/n, 默认n): ").strip().lower()
                    if save_prompt == "y":
                        filename = args.save_path or f"conversation_{time.strftime('%Y%m%d_%H%M%S')}.json"
                        conv.to_json(filename)
                        print(f"✅ 对话已保存到: {filename}")
                    print("👋 再见!")
                    break

                elif command == "/clear":
                    conv.clear(keep_system=True)
                    print("🗑️  对话历史已清空（系统提示词保留）")

                elif command == "/save":
                    filename = cmd_arg or args.save_path or f"conversation_{time.strftime('%Y%m%d_%H%M%S')}.json"
                    conv.to_json(filename)
                    print(f"✅ 对话已保存到: {filename}")

                elif command == "/history":
                    print(f"\n{'─'*50}")
                    print(f"📜 对话历史 ({len(conv)} 条消息):")
                    print(f"{'─'*50}")
                    for i, msg in enumerate(conv.messages):
                        icon = {"system": "⚙️", "user": "🧑", "assistant": "🤖"}.get(msg.role, "❓")
                        print(f"  [{i}] {icon} {msg.role}: {msg.content[:120]}")
                    print(f"{'─'*50}\n")

                elif command == "/system":
                    if cmd_arg:
                        conv.add_system(cmd_arg)
                        print(f"✅ 系统提示词已更新: {cmd_arg}")
                    else:
                        print(f"当前系统提示词: {conv.system_prompt or '(无)'}")

                elif command == "/stats":
                    total_chars = sum(len(m.content) for m in conv.messages)
                    user_msgs = sum(1 for m in conv.messages if m.role == "user")
                    assistant_msgs = sum(1 for m in conv.messages if m.role == "assistant")
                    print(f"\n📊 对话统计:")
                    print(f"   总消息数: {len(conv)}")
                    print(f"   用户消息: {user_msgs}")
                    print(f"   助手消息: {assistant_msgs}")
                    print(f"   总字符数: {total_chars}")
                    print(f"   模板: {args.conv_template}")
                    print(f"   最大上下文: {args.max_context}\n")

                else:
                    print(f"❓ 未知命令: {command}")
                    print("   可用命令: /clear /save /history /system /stats /exit")

                continue

            # Add user message and generate reply
            conv.add_user(user_input)

            # Print blank line for spacing
            print()

            # Build prompt
            if args.max_context > 0:
                full_prompt = conv.build_prompt_truncated()
            else:
                full_prompt = conv.build_prompt()

            start_tokens = tokenizer.encode(full_prompt)[:50]
            prompt_len = len(start_tokens)
            generated_tokens = list(start_tokens)
            start_time = time.time()
            new_text = ""
            printed_len = 0

            print("🤖 助手: ", end="", flush=True)

            try:
                for token_id, probability in model.generate_stream(
                    start_tokens,
                    args.max_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                ):
                    generated_tokens.append(token_id)
                    new_tokens = generated_tokens[prompt_len:]
                    new_text = tokenizer.decode(new_tokens)
                    new_text = new_text.replace("</w>", " ").replace("  ", " ")
                    new_text = _clean_new_text(new_text)

                    # Only print the new characters since last print
                    if len(new_text) > printed_len:
                        chunk = new_text[printed_len:]
                        printed_len = len(new_text)
                        sys.stdout.write(chunk)
                        sys.stdout.flush()

                # Extract final reply
                current_text = tokenizer.decode(generated_tokens).replace("</w>", " ").replace("  ", " ").strip()
                assistant_reply = _extract_assistant_reply(current_text, full_prompt)
                if not assistant_reply:
                    assistant_reply = new_text

                conv.add_assistant(assistant_reply)

                elapsed = time.time() - start_time
                tokens_gen = len(generated_tokens) - prompt_len
                print(f"\n   ⏱️  {tokens_gen} tokens / {elapsed:.2f}s / {tokens_gen/elapsed:.1f} t/s\n")

            except KeyboardInterrupt:
                print("\n⚠️  生成中断")
                if new_text:
                    partial = _extract_assistant_reply(
                        tokenizer.decode(generated_tokens).replace("</w>", " ").replace("  ", " "),
                        full_prompt,
                    ) or new_text
                    conv.add_assistant(partial + " [中断]")

        except KeyboardInterrupt:
            print("\n\n👋 再见!")
            break
        except EOFError:
            print("\n\n👋 再见!")
            break


def main():
    parser = argparse.ArgumentParser(
        description="Shannon-b1 流式文本生成 - 支持单次生成和交互式多轮对话",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单次生成
  python scripts/generate.py --model-path checkpoints/model.pt --prompt "你好"

  # 交互式多轮对话
  python scripts/generate.py --model-path checkpoints/model.pt -i

  # 多轮对话 + 系统提示词 + ChatML 模板
  python scripts/generate.py --model-path checkpoints/model.pt -i --system-prompt "你是专家" --conv-template chatml

  # 恢复已保存的对话
  python scripts/generate.py --model-path checkpoints/model.pt -i --load my_chat.json
        """,
    )
    parser.add_argument("--model-path", "--checkpoint", type=str, required=True, help="模型文件路径")
    parser.add_argument("--prompt", type=str, default="The ", help="提示词（单次生成模式）")
    parser.add_argument("--system-prompt", type=str, default=None, help="系统提示词（可选）")
    parser.add_argument("--max-tokens", "--max-new-tokens", type=int, default=100, help="最大生成token数")
    parser.add_argument("--temperature", type=float, default=0.8, help="温度参数")
    parser.add_argument("--top-k", type=int, default=50, help="Top-K采样参数")
    parser.add_argument("--top-p", type=float, default=None, help="Top-P采样参数")
    parser.add_argument("--repetition-penalty", type=float, default=1.1, help="重复惩罚系数")
    parser.add_argument("--device", type=str, default="cpu", help="运行设备")
    parser.add_argument("--delay", type=float, default=0.05, help="每个token之间的延迟（秒）")

    # Multi-turn options
    parser.add_argument("-i", "--interactive", action="store_true", help="启用交互式多轮对话模式（REPL）")
    parser.add_argument("--conv-template", type=str, default="simple",
                        choices=["simple", "chatml", "llama3"], help="对话模板格式 (默认: simple)")
    parser.add_argument("--max-context", type=int, default=4096,
                        help="最大上下文长度（字符数），超出自断。0=不限制 (默认: 4096)")
    parser.add_argument("--load", "--load-conv", dest="load_conv", type=str, default=None,
                        help="从 JSON 文件加载对话历史")
    parser.add_argument("--save", "--save-path", dest="save_path", type=str, default=None,
                        help="保存对话历史的默认路径")

    args = parser.parse_args()

    print("🔄 加载模型...")
    model, tokenizer, config = load_model(args.model_path, args.device)

    if args.interactive:
        interactive_chat(model, tokenizer, args)
    else:
        print(f"✅ 模型加载完成: vocab={config.vocab_size}, d_model={config.d_model}")
        print(f"📝 分词器类型: {'BPE' if hasattr(tokenizer, 'merges') else 'Char'}")
        single_generate(model, tokenizer, args)


if __name__ == "__main__":
    main()
```

