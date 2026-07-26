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
