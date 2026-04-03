import numpy as np
from ..core.transformer_block import TransformerBlock
from ..core.parameter import Parameter, to_numpy
from ..utils.functions import softmax


class ShannonB1:
    """Shannon-b1: 完整的可训练LLM"""
    
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len, dropout=0.1):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        
        limit = np.sqrt(6.0 / vocab_size)
        self.token_embedding = Parameter(np.random.uniform(-limit, limit, (vocab_size, d_model)))
        self.position_embedding = Parameter(np.random.uniform(-0.01, 0.01, (max_seq_len, d_model)))
        
        self.blocks = [TransformerBlock(d_model, num_heads, d_ff, dropout) 
                       for _ in range(num_layers)]
        
        from ..core.layer_norm import LayerNorm
        self.final_ln = LayerNorm(d_model)
        
        limit_out = np.sqrt(6.0 / (d_model + vocab_size))
        self.output_projection = Parameter(np.random.uniform(-limit_out, limit_out, (d_model, vocab_size)))
    
    def get_all_parameters(self):
        """收集所有可训练参数"""
        params = [
            ('token_embedding', self.token_embedding),
            ('position_embedding', self.position_embedding),
        ]
        
        for i, block in enumerate(self.blocks):
            params.append((f'block_{i}_attn_W_q', block.attention.W_q))
            params.append((f'block_{i}_attn_W_k', block.attention.W_k))
            params.append((f'block_{i}_attn_W_v', block.attention.W_v))
            params.append((f'block_{i}_attn_W_o', block.attention.W_o))
            params.append((f'block_{i}_ff_W1', block.feed_forward.W1))
            params.append((f'block_{i}_ff_b1', block.feed_forward.b1))
            params.append((f'block_{i}_ff_W2', block.feed_forward.W2))
            params.append((f'block_{i}_ff_b2', block.feed_forward.b2))
            params.append((f'block_{i}_ln1_gamma', block.ln1.gamma))
            params.append((f'block_{i}_ln1_beta', block.ln1.beta))
            params.append((f'block_{i}_ln2_gamma', block.ln2.gamma))
            params.append((f'block_{i}_ln2_beta', block.ln2.beta))
        
        params.append(('final_ln_gamma', self.final_ln.gamma))
        params.append(('final_ln_beta', self.final_ln.beta))
        params.append(('output_projection', self.output_projection))
        
        return params
    
    def forward(self, tokens, training=True):
        batch, seq_len = tokens.shape
        
        # 1. 词嵌入 + 位置编码
        x = self.token_embedding.data[tokens]  # (batch, seq_len, d_model)
        pos_indices = np.arange(seq_len)
        x = x + self.position_embedding.data[pos_indices]  # 广播加法
        
        # 缓存嵌入输出
        self.cache = {'tokens': tokens, 'embedding_out': x.copy()}
        
        # 2. 通过所有Transformer块
        for i, block in enumerate(self.blocks):
            x = block.forward(x, training=training)
            self.cache[f'block_{i}_out'] = x.copy()
        
        # 3. 最终LayerNorm
        x = self.final_ln.forward(x)
        
        # 4. 输出投影得到logits
        logits = x @ self.output_projection.data  # (batch, seq_len, vocab_size)
        
        self.cache['final_x'] = x
        self.cache['logits'] = logits
        return logits
    
    def backward(self, d_logits):
        """
        反向传播
        d_logits: (batch, seq_len, vocab_size) 损失对logits的梯度
        """
        # 获取缓存的形状信息
        batch, seq_len, vocab_size = d_logits.shape
        
        # 梯度反向传播到输出投影
        # d_logits: (batch, seq_len, vocab_size)
        # self.cache['final_x']: (batch, seq_len, d_model)
        # d_output_projection: (d_model, vocab_size)
        d_output_projection = self.cache['final_x'].reshape(-1, self.d_model).T @ d_logits.reshape(-1, vocab_size)
        self.output_projection.grad = d_output_projection
        
        # 反向传播到 final_ln 的输入
        # d_x: (batch, seq_len, d_model)
        d_x = d_logits @ self.output_projection.data.T
        d_x = self.final_ln.backward(d_x)
        
        # 反向传播通过所有Transformer块
        for i in reversed(range(self.num_layers)):
            d_x = self.blocks[i].backward(d_x)
        
        # 梯度通过嵌入层
        d_token_embedding = np.zeros_like(self.token_embedding.data)
        d_position_embedding = np.zeros_like(self.position_embedding.data)
        tokens = self.cache['tokens']
        
        for b in range(batch):
            for s in range(seq_len):
                d_token_embedding[tokens[b, s]] += d_x[b, s]
                d_position_embedding[s] += d_x[b, s]
        
        self.token_embedding.grad = d_token_embedding
        self.position_embedding.grad = d_position_embedding
    
    def generate(self, start_tokens, max_new_tokens, temperature=1.0):
        tokens = np.array([start_tokens])
        
        for _ in range(max_new_tokens):
            logits = self.forward(tokens, training=False)
            last_logits = logits[0, -1, :] / temperature
            probs = softmax(last_logits)
            next_token = np.random.choice(len(probs), p=probs)
            tokens = np.append(tokens, [[next_token]], axis=1)
        
        return tokens[0].tolist()