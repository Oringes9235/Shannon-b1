"""
Shannon-b1 主模型
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import math
from typing import Optional, List

from .config import ModelConfig
from .layers import PositionalEncoding, CausalMask
from .layers import RMSNorm


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
        super().__init__()

        self.config = config
        
        # 词嵌入
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(
            config.d_model, config.max_seq_len, config.dropout
        )
        
        # Transformer Decoder 层（按层构建以便更灵活）
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=config.d_model,
                nhead=config.num_heads,
                dim_feedforward=config.d_ff,
                dropout=config.dropout,
                activation='gelu',
                batch_first=True
            )
            for _ in range(config.num_layers)
        ])
        self.use_checkpointing = getattr(config, 'gradient_checkpointing', False)
        
        # 因果掩码
        self.causal_mask = CausalMask(config.max_seq_len)
        
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
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            tokens: (batch, seq_len) 输入 token IDs
        
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        batch, seq_len = tokens.shape
        
        # 词嵌入 + 缩放
        x = self.token_embedding(tokens) * math.sqrt(self.config.d_model)
        
        # 位置编码
        x = self.pos_encoding(x)

        # 因果掩码
        mask = self.causal_mask(seq_len)

        # 逐层应用 Transformer Decoder 层，可选地使用 checkpoint 节省内存
        for layer in self.decoder_layers:
            if self.use_checkpointing and self.training:
                # 为 checkpoint 包装函数，closure 捕获 layer 与 mask
                def run_layer(tgt, memory, layer=layer, mask=mask):
                    return layer(tgt, memory, tgt_mask=mask)

                x = checkpoint.checkpoint(run_layer, x, x, use_reentrant=False)
            else:
                x = layer(x, x, tgt_mask=mask)
        
        # LayerNorm
        x = self.ln_f(x)
        
        # 输出投影
        logits = self.output(x)
        
        return logits
    
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
    ) -> List[int]:
        """
        自回归生成文本
        
        Args:
            start_tokens: 起始 token 序列
            max_new_tokens: 最大生成 token 数
            temperature: 温度系数 (越高越随机)
            top_k: Top-K 采样
            top_p: Top-P (nucleus) 采样
        
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

            for _ in range(max_new_tokens):
                # 前向传播
                logits = self.forward(cur_tokens)

                # 获取最后一个位置
                last_logits = logits[0, -1, :].float()

                # 应用温度
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
                    generated = set(tokens[0].tolist())
                    for token_id in generated:
                        if last_logits[token_id] < 0:
                            last_logits[token_id] *= float(repetition_penalty)
                        else:
                            last_logits[token_id] /= float(repetition_penalty)

                # presence / frequency penalty: 在 logit 上做线性惩罚
                if presence_penalty != 0.0 or frequency_penalty != 0.0:
                    from collections import Counter
                    counts = Counter(tokens[0].tolist())
                    for tok_id, cnt in counts.items():
                        if presence_penalty != 0.0:
                            last_logits[tok_id] -= float(presence_penalty)
                        if frequency_penalty != 0.0 and cnt > 0:
                            last_logits[tok_id] -= float(frequency_penalty) * float(cnt)

                # 避免直接重复上一个 token（可选）
                if ban_immediate_repeat and tokens.size(1) > 0:
                    prev_token = int(tokens[0, -1].item())
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


class ShannonB1Encoder(nn.Module):
    """编码器版本 (非自回归，用于理解任务)"""
    
    def __init__(self, config: ModelConfig):
        
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
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(tokens) * math.sqrt(self.config.d_model)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        x = self.ln_f(x)
        return self.output(x)