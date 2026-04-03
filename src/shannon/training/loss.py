import numpy as np
from ..utils.functions import softmax


class CrossEntropyLoss:
    """交叉熵损失"""
    
    def forward(self, logits, targets):
        batch, seq_len, vocab_size = logits.shape
        
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)
        
        probs = softmax(logits_flat, axis=-1)
        target_probs = probs[np.arange(len(targets_flat)), targets_flat]
        loss = -np.mean(np.log(target_probs + 1e-8))
        
        self.logits = logits
        self.targets = targets
        self.probs = probs
        self.logits_flat = logits_flat
        self.targets_flat = targets_flat
        
        return loss
    
    def backward(self):
        batch, seq_len, vocab_size = self.logits.shape
        
        d_logits_flat = self.probs.copy()
        d_logits_flat[np.arange(len(self.targets_flat)), self.targets_flat] -= 1
        d_logits_flat = d_logits_flat / len(self.targets_flat)
        
        return d_logits_flat.reshape(batch, seq_len, vocab_size)