"""
saveTrain_Loss.py
从 checkpoint 中提取训练 loss 并保存为图片
"""

import torch
import matplotlib.pyplot as plt

ckpt_path = 'checkpoints/shannon_b1_best.pt'
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
history = ckpt.get('history', {})

train_loss = history.get('train_loss', [])
val_loss = history.get('val_loss', [])

if not train_loss and not val_loss:
    print('No loss history found in checkpoint.')
    print('Available keys:', list(ckpt.keys()))
    exit()

plt.figure(figsize=(10, 5))
if train_loss:
    plt.plot(train_loss, label='Train Loss', marker='o')
if val_loss:
    plt.plot(val_loss, label='Val Loss', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Shannon-b1 Training Progress')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('training_loss.png', dpi=150, bbox_inches='tight')
print('Saved: training_loss.png')