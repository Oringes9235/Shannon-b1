# Shannon-b1 CUDA 兼容性指南

## 概述

Shannon-b1 现已支持多种 CUDA 版本和 PyTorch 版本的兼容训练。本文档说明了如何配置和优化不同环境下的训练。

## 自动兼容性特性

### 1. GradScaler 自动适配

训练器会自动检测并适配不同 PyTorch 版本的 `GradScaler` API：

- **PyTorch >= 2.0**: 使用 `GradScaler(device_type='cuda')`
- **PyTorch 1.10-1.13**: 使用 `GradScaler()` （不带参数）
- **更早版本**: 自动降级或禁用混合精度

### 2. autocast 上下文管理器兼容

`_autocast()` 方法会尝试多种初始化方式：

```python
# 优先级顺序：
1. amp.autocast()                    # PyTorch >= 2.0
2. amp.autocast(device_type='cuda')  # PyTorch 1.10-1.13
3. torch.cuda.amp.autocast()         # 更旧版本
4. nullcontext()                     # 最终回退（禁用AMP）
```

### 3. 设备自动检测

系统会自动检测可用的计算设备，优先级为：
1. CUDA (NVIDIA GPU)
2. MPS (Apple Silicon)
3. CPU

## 环境检查

运行兼容性测试脚本：

```bash
python tests/test_cuda_compatibility.py
```

该脚本会检查：
- ✓ PyTorch 版本
- ✓ CUDA 可用性
- ✓ CUDA 详细信息（版本、cuDNN、设备）
- ✓ 混合精度训练支持
- ✓ 内存分配
- ✓ 模型创建和前向传播

## 训练时的 CUDA 信息输出

启动训练时，系统会自动显示详细的 CUDA 环境信息：

```
======================================================================
Shannon-b1 Improved Training
Start: 2026-04-06 17:51:37
Device: CUDA
Mixed Precision: ON
Grad Accum: 1
======================================================================

🔧 CUDA Environment:
   CUDA Version: 11.8
   cuDNN Version: 8700
   Device Count: 1

   GPU 0:
      Name: NVIDIA GeForce RTX 3060
      Compute Capability: 8.6
      Total Memory: 12.00 GB
```

## 常见问题与解决方案

### 问题 1: CUDA 不可用

**症状**: 训练时使用 CPU，速度很慢

**解决方案**:
```bash
# 检查 CUDA 是否安装
nvidia-smi

# 重新安装带 CUDA 支持的 PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 问题 2: 混合精度训练失败

**症状**: 训练时报错 "CUDA error" 或 "autocast failed"

**解决方案**:
```bash
# 禁用混合精度训练
python scripts/train.py --no-amp

# 或使用梯度累积减少显存压力
python scripts/train.py --grad-accum 4
```

### 问题 3: 显存不足 (OOM)

**症状**: "CUDA out of memory" 错误

**解决方案**:
```bash
# 方案 1: 减小 batch size
python scripts/train.py --batch-size 8

# 方案 2: 启用梯度检查点（节省约 50% 显存）
python scripts/train.py --gradient-checkpointing

# 方案 3: 使用梯度累积
python scripts/train.py --grad-accum 4 --batch-size 32

# 方案 4: 组合使用
python scripts/train.py --batch-size 8 --grad-accum 4 --gradient-checkpointing
```

### 问题 4: 不同 CUDA 版本兼容性问题

**症状**: 在某些 CUDA 版本上训练不稳定

**解决方案**:

项目已内置以下兼容性措施：

1. **环境变量设置** (UI 后端自动应用):
   ```python
   env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
   ```

2. **GradScaler 异常处理**: 自动捕获初始化失败并降级

3. **autocast 多层回退**: 确保在不同 PyTorch 版本上都能工作

如果仍遇到问题，可以尝试：

```bash
# 手动设置环境变量
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Windows PowerShell
$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# 然后运行训练
python scripts/train.py
```

## 推荐的训练配置

### 低显存 GPU (< 4GB)

```bash
python scripts/train.py \
  --batch-size 4 \
  --grad-accum 8 \
  --gradient-checkpointing \
  --d-model 64 \
  --num-layers 2
```

### 中等显存 GPU (4-8GB)

```bash
python scripts/train.py \
  --batch-size 16 \
  --grad-accum 2 \
  --d-model 128 \
  --num-layers 4
```

### 高显存 GPU (> 8GB)

```bash
python scripts/train.py \
  --batch-size 32 \
  --d-model 256 \
  --num-layers 6 \
  --num-heads 8
```

## 性能优化建议

### 1. 启用 Tensor Cores

对于 Volta 架构及以上的 GPU (Compute Capability >= 7.0)：

```bash
# 混合精度训练会自动利用 Tensor Cores
python scripts/train.py  # 默认启用 AMP
```

### 2. 调整数据加载

```bash
# 增加 worker 数量加速数据加载（Linux/Mac）
# 注意：Windows 上建议使用 num_workers=0
python scripts/train.py --num-workers 4
```

### 3. 使用更快的存储

将数据集放在 SSD 上可以显著提升 I/O 性能。

## 调试技巧

### 查看详细错误信息

```python
# 在训练脚本开头添加
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"cuDNN: {torch.backends.cudnn.version()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
```

### 监控 GPU 使用情况

```bash
# Linux/Mac
watch -n 1 nvidia-smi

# Windows
nvidia-smi -l 1
```

### 验证混合精度是否生效

训练日志中应看到：
```
Mixed Precision: ON
```

并且在 TensorBoard 中可以观察到 loss 曲线的正常下降。

## 技术支持

如果遇到 CUDA 相关问题：

1. 首先运行兼容性测试：`python tests/test_cuda_compatibility.py`
2. 检查 PyTorch 官方文档：https://pytorch.org/docs/stable/notes/cuda.html
3. 确认 NVIDIA 驱动和 CUDA Toolkit 版本匹配
4. 在项目 issues 中提供完整的错误信息和环境详情

## 更新日志

### v1.0 (2026-04-06)
- ✓ 添加 GradScaler 多版本兼容初始化
- ✓ 实现 autocast 多层回退机制
- ✓ 增强 CUDA 环境信息输出
- ✓ 创建兼容性测试脚本
- ✓ 添加 PYTORCH_CUDA_ALLOC_CONF 环境变量支持
- ✓ 改进验证阶段的混合精度支持
