# Shannon-b1 训练完全指南

> **文档版本**: v1.0  
> **最后更新**: 2026-04-11  
> **适用模型**: Shannon-b1 (GPT-style Transformer)

---

## 📋 目录

- [1. 快速开始](#1-快速开始)
- [2. 参数完全说明](#2-参数完全说明)
- [3. 硬件环境配置](#3-硬件环境配置)
- [4. 长上下文训练](#4-长上下文训练)
- [5. 性能优化策略](#5-性能优化策略)
- [6. 故障排查](#6-故障排查)
- [7. 最佳实践](#7-最佳实践)

---

## 1. 快速开始

### 1.1 最小化测试（验证环境）

```bash
# CPU 或任意 GPU，快速验证代码是否正常工作
python scripts/train.py \
  --epochs 5 \
  --batch-size 4 \
  --seq-len 64 \
  --d-model 64 \
  --num-layers 2 \
  --lr 5e-4 \
  --warmup-steps 100 \
  --tokenizer char
```

**预期结果**: 5-10分钟内完成，Loss应明显下降

### 1.2 标准训练（推荐起点）

```bash
# 中等规模GPU (6-8GB显存)
python scripts/train.py \
  --epochs 50 \
  --batch-size 16 \
  --seq-len 256 \
  --d-model 256 \
  --num-layers 6 \
  --num-heads 8 \
  --lr 5e-4 \
  --warmup-steps 2000 \
  --tokenizer bpe \
  --vocab-size 5000 \
  --grad-accum 2 \
  --gradient-checkpointing \
  --norm-type rmsnorm \
  --label-smoothing 0.1
```

---

## 2. 参数完全说明

### 2.1 模型架构参数

| 参数 | 说明 | 推荐范围 | 默认值 | 影响 |
|------|------|----------|--------|------|
| `--d-model` | 模型维度（隐藏层大小） | 64-1024 | 128 | ⭐⭐⭐ 直接影响参数量和表达能力 |
| `--num-layers` | Transformer层数 | 2-16 | 4 | ⭐⭐⭐ 决定模型深度 |
| `--num-heads` | 注意力头数 | 4-16 | 8 | ⭐⭐ 必须能被d_model整除 |
| `--d-ff` | 前馈网络维度 | d_model×2 ~ d_model×4 | 512 | ⭐ 影响单层计算量 |
| `--vocab-size` | 词汇表大小 | 1000-50000 | 10000 | ⭐ 影响Embedding层大小 |
| `--max-seq-len` | 最大序列长度 | 64-1048576 | 512 | ⭐⭐⭐ 决定可处理的文本长度 |

**参数关系约束**:
```
d_model % num_heads == 0  （必须成立）
d_ff 通常为 d_model 的 2-4 倍
```

### 2.2 训练超参数

| 参数 | 说明 | 推荐范围 | 默认值 | 调优建议 |
|------|------|----------|--------|----------|
| `--lr` | 学习率 | 1e-5 ~ 1e-3 | 0.001 | ⭐⭐⭐ 最关键参数，小模型用5e-4 |
| `--warmup-steps` | 学习率预热步数 | 500-5000 | 0 | ⭐⭐ 总步数的5-10% |
| `--batch-size` | 批次大小 | 2-64 | 32 | ⭐⭐⭐ 受显存限制最大 |
| `--epochs` | 训练轮数 | 10-200 | 50 | 根据收敛情况调整 |
| `--weight-decay` | 权重衰减（L2正则） | 0-0.01 | 0.01 | ⭐ 防止过拟合 |
| `--grad-clip` | 梯度裁剪阈值 | 0.5-2.0 | 1.0 | ⭐ 稳定训练 |

**学习率选择指南**:
```
小模型 (d_model < 256):  lr = 5e-4 ~ 1e-3
中模型 (d_model 256-512): lr = 3e-4 ~ 5e-4
大模型 (d_model > 512):  lr = 1e-4 ~ 3e-4
```

### 2.3 优化与正则化参数

| 参数 | 说明 | 推荐值 | 启用条件 |
|------|------|--------|----------|
| `--label-smoothing` | 标签平滑系数 | 0.0-0.1 | 始终推荐 |
| `--gradient-checkpointing` | 梯度检查点 | True/False | 显存不足时必开 |
| `--norm-type` | 归一化类型 | layernorm/rmsnorm | 推荐rmsnorm |
| `--tie-word-embeddings` | 绑定词嵌入权重 | True/False | 推荐True节省参数 |
| `--use-amp` | 自动混合精度 | True/False | GPU上始终开启 |

### 2.4 长上下文参数（1M+支持）

| 参数 | 说明 | 推荐配置 | 适用场景 |
|------|------|----------|----------|
| `--use-rope` | 启用RoPE位置编码 | True | 所有长序列场景 |
| `--rope-base` | RoPE基频 | 见下方表格 | 根据序列长度选择 |
| `--sliding-window-size` | 滑动窗口大小 | 4096-8192 | seq_len > 32K时必开 |
| `--use-alibi` | 启用ALiBi | False | 不与RoPE同时使用 |

**RoPE Base频率选择表**:

| 目标序列长度 | rope_base | 说明 |
|-------------|-----------|------|
| < 8K | 10000.0 | 标准配置 |
| 8K - 64K | 100000.0 | 中长文档 |
| > 64K | 1000000.0+ | 超长上下文 |

---

## 3. 硬件环境配置

### 3.1 显存估算公式

```
基础显存需求 ≈ batch_size × seq_len × d_model × num_layers × 4 bytes (FP32)

优化后显存:
- 启用AMP (混合精度):     × 0.5
- 启用Gradient Checkpoint: × 0.3-0.4
- 综合优化:                × 0.15-0.2
```

**实际示例**:
```
配置: batch=16, seq_len=256, d_model=256, num_layers=6
基础显存: 16 × 256 × 256 × 6 × 4 ≈ 2.5 GB
启用AMP + Gradient Checkpoint: 2.5 × 0.2 ≈ 0.5 GB
加上优化器状态等: 总计约 1.5-2 GB
```

### 3.2 不同环境的推荐配置

#### 🔴 环境A: CPU / 极低显存 (< 2GB)

**适用硬件**: 
- 纯CPU训练
- 老旧GPU (如GT 710, MX110)

**推荐配置**:
```bash
python scripts/train.py \
  --batch-size 4 \
  --seq-len 64 \
  --d-model 64 \
  --num-layers 2 \
  --num-heads 4 \
  --lr 5e-4 \
  --warmup-steps 100 \
  --gradient-checkpointing \
  --grad-accum 8 \
  --no-amp  # CPU不支持AMP
```

**预期性能**:
- 训练速度: 极慢 (hours/epoch)
- 最大序列: ≤ 128 tokens
- 用途: 仅用于代码验证和学习

---

#### 🟡 环境B: 低显存GPU (2-4GB)

**适用硬件**:
- NVIDIA MX330, MX450
- GTX 1050 Ti, GTX 1650

**推荐配置**:
```bash
python scripts/train.py \
  --batch-size 8 \
  --seq-len 128 \
  --d-model 128 \
  --num-layers 4 \
  --num-heads 8 \
  --lr 5e-4 \
  --warmup-steps 1000 \
  --gradient-checkpointing \
  --grad-accum 4 \
  --norm-type rmsnorm \
  --label-smoothing 0.1
```

**预期性能**:
- 训练速度: 慢 (30min-1h/epoch)
- 最大序列: 128-256 tokens
- 可用功能: 基础训练、短文本生成

**极限配置** (勉强运行):
```bash
--batch-size 4 --seq-len 256 --d-model 128 --num-layers 4
```

---

#### 🟢 环境C: 中端GPU (6-8GB)

**适用硬件**:
- RTX 3050, RTX 3060 (6GB/8GB)
- GTX 1070, GTX 1660 Ti

**标准配置**:
```bash
python scripts/train.py \
  --batch-size 16 \
  --seq-len 256 \
  --d-model 256 \
  --num-layers 6 \
  --num-heads 8 \
  --lr 5e-4 \
  --warmup-steps 2000 \
  --grad-accum 2 \
  --gradient-checkpointing \
  --norm-type rmsnorm \
  --label-smoothing 0.1 \
  --tie-word-embeddings
```

**高性能配置** (关闭gradient checkpointing):
```bash
--batch-size 24 --seq-len 256 --d-model 256 --num-layers 6
# 移除 --gradient-checkpointing 以提升速度
```

**预期性能**:
- 训练速度: 中等 (5-15min/epoch)
- 最大序列: 256-512 tokens
- 可用功能: 完整训练、中等长度生成

---

#### 🔵 环境D: 高端GPU (12-24GB)

**适用硬件**:
- RTX 3080 (10GB), RTX 3090 (24GB)
- RTX 4070, RTX 4080, RTX 4090
- A100, A10G

**推荐配置**:
```bash
python scripts/train.py \
  --batch-size 32 \
  --seq-len 512 \
  --d-model 512 \
  --num-layers 8 \
  --num-heads 16 \
  --lr 3e-4 \
  --warmup-steps 4000 \
  --grad-accum 1 \
  --norm-type rmsnorm \
  --label-smoothing 0.1 \
  --tie-word-embeddings
```

**大规模配置**:
```bash
--batch-size 64 --seq-len 1024 --d-model 768 --num-layers 12
# 可能需要启用 gradient-checkpointing
```

**预期性能**:
- 训练速度: 快 (1-5min/epoch)
- 最大序列: 512-2048 tokens
- 可用功能: 大规模训练、长文本生成

---

#### 🟣 环境E: 长上下文专用 (> 32K tokens)

**适用硬件**:
- RTX 3090/4090 (24GB+)
- A100 (40GB/80GB)
- 多卡并行

**16K序列配置**:
```bash
python scripts/train.py \
  --batch-size 8 \
  --seq-len 16384 \
  --d-model 512 \
  --num-layers 8 \
  --num-heads 16 \
  --lr 3e-4 \
  --warmup-steps 4000 \
  --use-rope \
  --rope-base 100000.0 \
  --sliding-window-size 4096 \
  --gradient-checkpointing \
  --grad-accum 4
```

**64K序列配置** (需要A100 80GB或类似):
```bash
python scripts/train.py \
  --batch-size 4 \
  --seq-len 65536 \
  --d-model 512 \
  --num-layers 8 \
  --use-rope \
  --rope-base 1000000.0 \
  --sliding-window-size 8192 \
  --gradient-checkpointing \
  --grad-accum 8
```

**预期性能**:
- 训练速度: 较慢 (依赖序列长度)
- 最大序列: 32K-1M+ tokens (理论)
- 关键: 必须启用滑动窗口，否则OOM

---

### 3.3 是否可以"最大限度"训练？

**答案: 不能盲目最大化，需要权衡**

#### ❌ 错误做法
```bash
# 不要这样做！会导致OOM或极慢
--batch-size 128 --seq-len 4096 --d-model 1024 --num-layers 16
```

#### ✅ 正确策略

**原则1: 显存优先分配给 batch_size**
```
优先级: batch_size > seq_len > d_model > num_layers
```

**原则2: 使用梯度累积模拟大batch**
```bash
# 等效于 batch_size=64，但显存占用仅为batch_size=8
--batch-size 8 --grad-accum 8
```

**原则3: 序列长度按实际需求设定**
```
短文本任务 (对话):  seq_len = 128-256
中等文本 (文章):    seq_len = 512-1024
长文档 (书籍):      seq_len = 2048-8192 (需滑动窗口)
```

**原则4: 长序列必须优化**
```
seq_len > 8K:   必须启用 --use-rope --rope-base 10000
seq_len > 32K:  必须启用 --sliding-window-size 4096
seq_len > 64K:  必须启用 --gradient-checkpointing --rope-base 1000000
```

---

## 4. 长上下文训练

### 4.1 技术选型对比

| 方案 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **传统正弦编码** | 简单 | 外推性差 | seq_len < 512 |
| **RoPE** | 外推性好，主流方案 | 需要调整base频率 | **推荐: 所有长序列** |
| **ALiBi** | 无限外推 | 不与RoPE混用 | 特殊研究场景 |
| **滑动窗口** | 降低复杂度O(N²)→O(NW) | 丢失远距离依赖 | seq_len > 32K必选 |

### 4.2 RoPE配置最佳实践

``bash
# 命令行参数示例
python scripts/train.py \
  --use-rope \                    # 启用RoPE
  --rope-base 10000.0 \          # 根据序列长度调整
  --sliding-window-size 4096     # 长序列时启用
```

**Python API配置**:
```python
# 配置示例
config = ModelConfig(
    max_seq_len=1048576,  # 支持1M
    use_rope=True,
    rope_base=10000.0,    # 根据下表调整
)
```

**RoPE Base频率选择**:

| 应用场景 | 典型长度 | rope_base | 示例命令 |
|---------|---------|-----------|---------|
| 短对话 | < 2K | 10000.0 | `--rope-base 10000` |
| 文章摘要 | 2K-8K | 10000.0 | `--rope-base 10000` |
| 长文档 | 8K-64K | 100000.0 | `--rope-base 100000` |
| 书籍/代码库 | > 64K | 1000000.0 | `--rope-base 1000000` |

### 4.3 滑动窗口配置

```bash
# 推荐窗口大小
--sliding-window-size 4096   # 平衡性能和上下文
--sliding-window-size 8192   # 保留更多历史信息
```

**注意事项**:
- 窗口大小必须是2的幂次（性能优化）
- 窗口外的token完全不可见（硬截断）
- 与RoPE配合使用时效果最佳

### 4.4 长上下文训练示例

**8K序列训练**:
``bash
python scripts/train.py \
  --epochs 100 \
  --batch-size 8 \
  --seq-len 8192 \
  --d-model 512 \
  --num-layers 8 \
  --num-heads 16 \
  --lr 3e-4 \
  --warmup-steps 4000 \
  --use-rope \
  --rope-base 100000.0 \
  --sliding-window-size 4096 \
  --gradient-checkpointing \
  --grad-accum 4 \
  --norm-type rmsnorm
```

**预期资源**:
- 显存: ~12-16GB (RTX 3090级别)
- 时间: 较慢，建议从小序列预训练后再微调

---

## 5. 性能优化策略

### 5.1 显存优化技术对比

| 技术 | 显存节省 | 速度影响 | 推荐度 |
|------|---------|---------|--------|
| **AMP (混合精度)** | 40-50% | +10-30%加速 | ⭐⭐⭐ 必开 |
| **Gradient Checkpoint** | 60-70% | -20-30%减速 | ⭐⭐ 显存不足时 |
| **梯度累积** | 无直接节省 | 无影响 | ⭐⭐⭐ 模拟大batch |
| **Tie Embeddings** | 5-10% | 无影响 | ⭐ 推荐 |
| **滑动窗口** | O(N²)→O(NW) | 大幅加速长序列 | ⭐⭐⭐ 长序列必选 |

### 5.2 训练速度优化

**技巧1: 禁用不必要的功能**
```bash
# 如果不需要，关闭这些功能可以提升速度
--no-gradient-checkpointing  # 显存充足时
--grad-accum 1               # 避免累积开销
```

**技巧2: 使用更高效的数据加载**
```python
# 在 dataset.py 中设置
DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=4,      # 多进程加载
    pin_memory=True,    # 锁页内存加速传输
    prefetch_factor=2   # 预取批次
)
```

**技巧3: 调整日志频率**
```bash
--log-interval 50   # 减少日志写入频率 (默认10)
--eval-interval 500 # 减少评估频率 (默认100)
```

### 5.3 收敛优化

**学习率调度**:
```bash
# 余弦退火 + Warmup (推荐)
--lr 5e-4 \
--warmup-steps 2000 \
--use-cosine-scheduler
```

**正则化组合**:
```bash
--weight-decay 0.01 \
--label-smoothing 0.1 \
--dropout 0.1 \
--grad-clip 1.0
```

---

## 6. 故障排查

### 6.1 常见问题

#### ❌ CUDA Out of Memory

**症状**:
```
RuntimeError: CUDA out of memory. Tried to allocate X MiB
```

**解决方案** (按优先级):
1. 减小 `--batch-size`
2. 减小 `--seq-len`
3. 启用 `--gradient-checkpointing`
4. 增大 `--grad-accum` 补偿batch减小
5. 减小 `--d-model` 或 `--num-layers`

**诊断命令**:
```bash
# 监控显存使用
nvidia-smi -l 1

# PyTorch内部查询
python -c "import torch; print(torch.cuda.memory_summary())"
```

---

#### ❌ Loss不下降

**可能原因**:
1. 学习率过大/过小
2. 数据质量问题
3. 模型容量不足

**调试步骤**:
```bash
# 1. 降低学习率
--lr 1e-4

# 2. 检查数据
python -c "
from src.data import CharTokenizer
tok = CharTokenizer()
data = open('data/shakespeare.txt').read()[:1000]
print('Unique chars:', len(set(data)))
print('Sample:', data[:100])
"

# 3. 增大模型
--d-model 256 --num-layers 6
```

---

#### ❌ 生成长度超过max_seq_len时报错

**症状**:
```
IndexError: index out of range in self
```

**已修复**: 动态掩码扩展已实现，确保使用最新版本代码。

**验证**:
```bash
python tests/test_long_context.py
```

---

#### ❌ RoPE维度不匹配

**症状**:
```
RuntimeError: The size of tensor a (X) must match the size of tensor b (Y)
```

**原因**: cos/sin缓存未正确分离奇偶维度

**解决**: 已修复，确保 `layers.py` 中的 `_apply_rotary` 方法正确实现。

---

### 6.2 性能诊断

**检查训练瓶颈**:
```python
# 在训练脚本中添加
import time
start = time.time()
# ... 训练循环 ...
end = time.time()
print(f"Step time: {end-start:.3f}s")
print(f"Tokens/sec: {batch_size * seq_len / (end-start):.1f}")
```

**典型性能指标**:
```
RTX 3060 (8GB):
- batch=16, seq_len=256, d_model=256: ~1000 tokens/sec
- batch=8, seq_len=512, d_model=256:  ~800 tokens/sec

RTX 3090 (24GB):
- batch=32, seq_len=512, d_model=512: ~3000 tokens/sec
```

---

## 7. 最佳实践

### 7.1 训练流程建议

#### 阶段1: 快速原型 (1-2小时)
```bash
python scripts/train.py \
  --epochs 10 \
  --batch-size 8 \
  --seq-len 64 \
  --d-model 64 \
  --num-layers 2 \
  --lr 5e-4
```
**目标**: 验证代码、数据、环境正常

---

#### 阶段2: 小规模实验 (半天)
```bash
python scripts/train.py \
  --epochs 50 \
  --batch-size 16 \
  --seq-len 128 \
  --d-model 128 \
  --num-layers 4 \
  --lr 5e-4 \
  --warmup-steps 1000 \
  --tokenizer bpe \
  --vocab-size 2000
```
**目标**: 确定合适的学习率和模型规模

---

#### 阶段3: 中等规模训练 (1-3天)
```bash
python scripts/train.py \
  --epochs 100 \
  --batch-size 24 \
  --seq-len 256 \
  --d-model 256 \
  --num-layers 6 \
  --lr 5e-4 \
  --warmup-steps 2000 \
  --tokenizer bpe \
  --vocab-size 5000 \
  --grad-accum 2 \
  --gradient-checkpointing \
  --norm-type rmsnorm \
  --label-smoothing 0.1
```
**目标**: 获得可用的模型

---

#### 阶段4: 大规模训练 (可选，数天)
```bash
python scripts/train.py \
  --epochs 200 \
  --batch-size 32 \
  --seq-len 512 \
  --d-model 512 \
  --num-layers 8 \
  --lr 3e-4 \
  --warmup-steps 4000 \
  --tokenizer bpe \
  --vocab-size 10000 \
  --grad-accum 1 \
  --norm-type rmsnorm \
  --label-smoothing 0.1 \
  --tie-word-embeddings
```
**目标**: 追求最佳性能

---

### 7.2 监控与可视化

**启动TensorBoard**:
```bash
tensorboard --logdir runs --port 6006
# 访问 http://localhost:6006
```

**关键指标**:
- `train_loss`: 应持续下降
- `val_loss`: 应在某个点后趋于平稳
- `learning_rate`: 应按预期warmup然后decay
- `perplexity`: 越低越好

**早停判断**:
```
如果 val_loss 连续 10 个 epoch 不下降，考虑:
1. 降低学习率
2. 增加正则化
3. 停止训练并保存最佳模型
```

---

### 7.3 断点续训

**保存检查点**:
```bash
# 自动保存在 checkpoints/ 目录
checkpoints/shannon_b1_best.pt   # 最佳验证集模型
checkpoints/shannon_b1_last.pt   # 最后一个epoch
```

**从断点恢复**:
```bash
python scripts/train.py \
  --resume checkpoints/shannon_b1_last.pt \
  --epochs 100  # 继续训练到100个epoch
```

**注意事项**:
- 不要修改模型架构参数 (d_model, num_layers等)
- 可以调整学习率、batch_size等训练参数
- 确保分词器一致

---

### 7.4 模型评估

**生成测试**:
```bash
python scripts/generate.py \
  --checkpoint checkpoints/shannon_b1_best.pt \
  --prompt "To be, or not to be" \
  --max-new-tokens 200 \
  --temperature 0.8 \
  --top-k 40 \
  --top-p 0.9
```

**流式生成** (实时显示):
```bash
python scripts/generate.py \
  --checkpoint checkpoints/shannon_b1_best.pt \
  --prompt "Once upon a time" \
  --max-tokens 100 \
  --delay 0.05  # 打字机效果
```

---

### 7.5  checklist

训练前检查:
- [ ] 数据文件存在 (`data/shakespeare.txt`)
- [ ] 显存足够 (参考第3节)
- [ ] 学习率设置合理
- [ ] 启用AMP (`--use-amp`, GPU默认开启)
- [ ] 设置warmup步数
- [ ] 长序列时启用RoPE和滑动窗口

训练中监控:
- [ ] Loss正常下降
- [ ] 显存使用稳定
- [ ] TensorBoard记录正常
- [ ] 定期保存检查点

训练后验证:
- [ ] 生成文本质量可接受
- [ ] 模型文件大小合理
- [ ] 可以成功加载和推理

---

## 📚 附录

### A. 完整参数列表

```bash
python scripts/train.py --help
```

### B. 相关文档

- [流式输出生成功能](../notebooks/流式输出%20(Streaming)%20功能文档.md)
- [CUDA兼容性说明](../notebooks/CUDA_COMPATIBILITY.md)
- [长序列支持文档](../LONG_SEQUENCE_SUPPORT.md)

### C. 参考资料

- [RoPE论文](https://arxiv.org/abs/2104.09864)
- [Transformer架构](https://arxiv.org/abs/1706.03762)
- [混合精度训练](https://pytorch.org/docs/stable/notes/amp_examples.html)

---

**文档维护**: 如有问题或建议，请提交Issue或PR  
**最后更新**: 2026-04-11
