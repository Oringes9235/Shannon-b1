import numpy as np


class Trainer:
    """训练器"""
    
    def __init__(self, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.history = []
    
    def train_epoch(self, train_data, batch_size=4, seq_len=16, verbose=True):
        total_loss = 0
        num_batches = 0
        
        # 打乱数据
        indices = np.random.permutation(len(train_data))
        
        for i in range(0, len(indices) - batch_size, batch_size):
            # 获取批次
            batch_indices = indices[i:i+batch_size]
            batch = [train_data[idx] for idx in batch_indices]
            
            # 找到批次中的最大长度
            max_len = min(max(len(seq) for seq in batch), seq_len)
            
            inputs, targets = self._prepare_batch(batch, max_len)
            
            # 前向传播
            logits = self.model.forward(inputs, training=True)
            
            # 计算损失
            loss = self.criterion.forward(logits, targets)
            total_loss += loss
            num_batches += 1
            
            # 反向传播
            d_logits = self.criterion.backward()
            self.model.backward(d_logits)
            
            # 更新参数
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            if verbose and num_batches % 10 == 0:
                print(f"  Batch {num_batches}, Loss: {loss:.4f}")
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self, train_data, epochs=10, batch_size=4, seq_len=16, verbose=True):
        for epoch in range(epochs):
            avg_loss = self.train_epoch(train_data, batch_size, seq_len, verbose)
            self.history.append(avg_loss)
            if verbose:
                print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
        return self.history
    
    def _prepare_batch(self, batch, max_len):
        """准备批次数据"""
        inputs = []
        targets = []
        
        for seq in batch:
            # 截断到 max_len + 1 (因为需要输入和目标)
            if len(seq) > max_len + 1:
                seq = seq[:max_len + 1]
            
            # 输入: 除了最后一个token
            if len(seq) > 1:
                input_seq = seq[:-1]
                target_seq = seq[1:]
            else:
                input_seq = seq
                target_seq = seq
            
            # 填充到相同长度
            input_seq = input_seq + [0] * (max_len - len(input_seq))
            target_seq = target_seq + [0] * (max_len - len(target_seq))
            
            inputs.append(input_seq)
            targets.append(target_seq)
        
        return np.array(inputs, dtype=np.int32), np.array(targets, dtype=np.int32)