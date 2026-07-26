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