# CUDA 兼容性快速参考

## 🚀 快速开始

### 检查环境兼容性
```bash
python tests/test_cuda_compatibility.py
```

### 开始训练（自动适配CUDA版本）
```bash
# 基本训练（自动检测最佳配置）
python scripts/train.py

# 如果遇到问题，禁用混合精度
python scripts/train.py --no-amp
```

## 🔧 关键改进点

### 1. GradScaler 兼容性
```python
# 自动适配不同PyTorch版本
try:
    scaler = amp.GradScaler(device_type='cuda')  # PyTorch >= 2.0
except TypeError:
    scaler = amp.GradScaler()                     # PyTorch < 2.0
```

### 2. autocast 兼容性
```python
# 多层回退机制确保兼容
try:
    return amp.autocast()                         # PyTorch >= 2.0
except TypeError:
    return amp.autocast(device_type='cuda')       # PyTorch 1.10-1.13
except Exception:
    from contextlib import nullcontext
    return nullcontext()                          # 最终回退
```

### 3. 环境变量优化
```python
# UI后端自动设置
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
```

## ⚙️ 常用命令

### 查看CUDA信息
```bash
# Windows
nvidia-smi

# 或在Python中
python -c "import torch; print(f'CUDA: {torch.version.cuda}'); print(f'Device: {torch.cuda.get_device_name(0)}')"
```

### 显存不足时的解决方案
```bash
# 方案1: 减小batch size
python scripts/train.py --batch-size 8

# 方案2: 启用梯度检查点（节省50%显存）
python scripts/train.py --gradient-checkpointing

# 方案3: 梯度累积
python scripts/train.py --grad-accum 4

# 方案4: 组合使用
python scripts/train.py --batch-size 8 --grad-accum 4 --gradient-checkpointing
```

### 禁用混合精度
```bash
python scripts/train.py --no-amp
```

## 📊 支持的配置

| PyTorch版本 | CUDA支持 | 混合精度 | 备注 |
|------------|---------|---------|------|
| >= 2.0     | ✓       | ✓       | 完全支持 |
| 1.10-1.13  | ✓       | ✓       | 完全支持 |
| 1.6-1.9    | ✓       | ⚠       | 可能需要--no-amp |
| < 1.6      | ⚠       | ✗       | 建议升级 |

| CUDA版本   | 支持状态 | 最低计算能力 |
|-----------|---------|-------------|
| 11.x      | ✓       | 6.0         |
| 10.x      | ✓       | 6.0         |
| 9.x       | ⚠       | 6.0         |
| < 9.x     | ✗       | -           |

## 🐛 故障排除

### 问题：CUDA不可用
```bash
# 检查驱动
nvidia-smi

# 重装PyTorch（CUDA 11.8示例）
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 问题：混合精度失败
```bash
# 禁用AMP
python scripts/train.py --no-amp

# 或更新PyTorch
pip install --upgrade torch
```

### 问题：显存溢出
```bash
# 使用最小显存配置
python scripts/train.py \
  --batch-size 4 \
  --grad-accum 8 \
  --gradient-checkpointing \
  --d-model 64 \
  --num-layers 2
```

## 💡 最佳实践

1. **始终先运行兼容性测试**
   ```bash
   python tests/test_cuda_compatibility.py
   ```

2. **监控GPU使用情况**
   ```bash
   # Linux/Mac
   watch -n 1 nvidia-smi
   
   # Windows
   nvidia-smi -l 1
   ```

3. **使用TensorBoard监控训练**
   ```bash
   python scripts/train.py --tensorboard
   tensorboard --logdir runs
   ```

4. **保存和恢复训练**
   ```bash
   # 训练会自动保存checkpoint
   # 从checkpoint恢复
   python scripts/train.py --resume checkpoints/shannon_b1.pt
   ```

## 📝 训练日志解读

启动训练时会看到：
```
======================================================================
Shannon-b1 Improved Training
Start: 2026-04-06 17:51:37
Device: CUDA
Mixed Precision: ON          ← 混合精度已启用
Grad Accum: 1                ← 梯度累积步数
======================================================================

🔧 CUDA Environment:
   CUDA Version: 11.8        ← CUDA版本
   cuDNN Version: 8700       ← cuDNN版本
   Device Count: 1           ← GPU数量

   GPU 0:
      Name: NVIDIA GeForce RTX 3060
      Compute Capability: 8.6  ← 计算能力（>=7.0支持Tensor Cores）
      Total Memory: 12.00 GB
```

## 🔗 相关文档

- [完整CUDA兼容性指南](CUDA_COMPATIBILITY.md)
- [项目README](README.md)
- [PyTorch CUDA文档](https://pytorch.org/docs/stable/notes/cuda.html)
