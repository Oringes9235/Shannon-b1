# Shannon-b1 架构版本字符串规范

## 📋 目录
- [什么是架构版本字符串？](#什么是架构版本字符串)
- [为什么需要架构版本？](#为什么需要架构版本)
- [命名格式规范](#命名格式规范)
- [参数对照表](#参数对照表)
- [使用示例](#使用示例)
- [实际应用场景](#实际应用场景)
- [常见问题](#常见问题)

---

## 🎯 什么是架构版本字符串？

**架构版本字符串（Architecture Version String）**是 Shannon-b1 项目在训练完成后自动生成的、用于标识模型架构配置的简短编码。它会附加在 checkpoint 文件名中，确保每个模型文件的架构信息一目了然。

### 基本格式

```
{base_name}_{arch_version}_{timestamp}.pt
```

**示例：**
```
shannon_b1_dm256_nl6_nh8_rope10000_rms_tie_20260411_164500.pt
```

其中：
- `shannon_b1` - 基础名称
- `dm256_nl6_nh8_rope10000_rms_tie` - **架构版本字符串**
- `20260411_164500` - 时间戳（YYYYMMDD_HHMMSS）

---

## 💡 为什么需要架构版本？

### 问题背景

在深度学习项目中，经常遇到以下问题：

1. **架构不兼容**：不同配置训练的模型无法互相加载
2. **文件管理混乱**：多个实验的 checkpoint 难以区分
3. **追溯困难**：忘记某个模型的具体配置参数
4. **协作障碍**：团队成员不清楚彼此的模型架构

### 解决方案

架构版本字符串通过**标准化命名**解决这些问题：

✅ **即时识别**：无需加载模型即可知道其架构  
✅ **避免冲突**：每次训练生成唯一文件名  
✅ **便于检索**：可按架构特征快速筛选模型  
✅ **团队协作**：统一的命名规范提升沟通效率  

---

## 📝 命名格式规范

### 完整结构

架构版本字符串由多个**短横线分隔的参数缩写**组成：

```
dm{d_model}_nl{num_layers}_nh{num_heads}_{position_encoding}_{norm_type}[_{optional_features}]
```

### 参数顺序（固定）

| 顺序 | 参数类别 | 必填 | 说明 |
|------|---------|------|------|
| 1 | 模型维度 | ✅ | `dm{value}` |
| 2 | 层数 | ✅ | `nl{value}` |
| 3 | 注意力头数 | ✅ | `nh{value}` |
| 4 | 位置编码 | ✅ | `rope/alibi/fixed` |
| 5 | 归一化类型 | ✅ | `layer/rms` |
| 6+ | 可选特性 | ❌ | `sw/tie/ckpt` 等 |

---

## 📊 参数对照表

### 核心架构参数

| 参数名 | 缩写前缀 | 示例值 | 完整格式 | 说明 |
|--------|---------|--------|---------|------|
| `d_model` | `dm` | 256 | `dm256` | 模型隐藏层维度 |
| `num_layers` | `nl` | 6 | `nl6` | Transformer 层数 |
| `num_heads` | `nh` | 8 | `nh8` | 多头注意力头数 |

### 位置编码类型

| 配置 | 格式 | 说明 | 适用场景 |
|------|------|------|---------|
| RoPE | `rope{base}` | `rope10000` | 默认推荐，支持外推 |
| ALiBi | `alibi` | `alibi` | 长序列线性偏置 |
| 固定正弦 | `fixed` | `fixed` | 传统 Transformer |

**RoPE Base 频率建议：**
- `< 8K tokens`: `rope10000`
- `8K - 64K tokens`: `rope100000`
- `> 64K tokens`: `rope1000000`

### 归一化类型

| 配置值 | 缩写 | 完整格式 | 特点 |
|--------|------|---------|------|
| `layernorm` | `layer` | `layer` | 标准 LayerNorm |
| `rmsnorm` | `rms` | `rms` | RMSNorm（推荐） |

### 可选特性标志

| 特性 | 标志 | 触发条件 | 说明 |
|------|------|---------|------|
| 滑动窗口 | `sw{size}` | `sliding_window_size > 0` | 如 `sw4096` |
| 权重共享 | `tie` | `tie_word_embeddings=True` | 输入输出嵌入共享 |
| 梯度检查点 | `ckpt` | `gradient_checkpointing=True` | 节省显存 |

---

## 🚀 使用示例

### 示例 1：标准配置（RoPE + RMSNorm）

**训练命令：**
```bash
python scripts/train.py \
  --d-model 256 \
  --num-layers 6 \
  --num-heads 8 \
  --use-rope \
  --rope-base 10000 \
  --norm-type rmsnorm \
  --epochs 50
```

**生成的文件名：**
```
checkpoints/shannon_b1_dm256_nl6_nh8_rope10000_rms_20260411_164500.pt
```

**架构版本解析：**
- `dm256` → d_model = 256
- `nl6` → num_layers = 6
- `nh8` → num_heads = 8
- `rope10000` → RoPE with base 10000
- `rms` → RMSNorm

---

### 示例 2：ALiBi + LayerNorm + 权重共享

**训练命令：**
```bash
python scripts/train.py \
  --d-model 128 \
  --num-layers 4 \
  --num-heads 8 \
  --use-alibi \
  --norm-type layernorm \
  --tie-embeddings
```

**生成的文件名：**
```
checkpoints/shannon_b1_dm128_nl4_nh8_alibi_layer_tie_20260411_170000.pt
```

**架构版本解析：**
- `dm128` → d_model = 128
- `nl4` → num_layers = 4
- `nh8` → num_heads = 8
- `alibi` → ALiBi position encoding
- `layer` → LayerNorm
- `tie` → Tied embeddings

---

### 示例 3：长上下文支持（滑动窗口 + Gradient Checkpointing）

**训练命令：**
```bash
python scripts/train.py \
  --d-model 512 \
  --num-layers 8 \
  --num-heads 16 \
  --use-rope \
  --rope-base 100000 \
  --sliding-window-size 4096 \
  --norm-type rmsnorm \
  --gradient-checkpointing \
  --seq-len 8192
```

**生成的文件名：**
```
checkpoints/shannon_b1_dm512_nl8_nh16_rope100000_sw4096_rms_ckpt_20260411_180000.pt
```

**架构版本解析：**
- `dm512` → d_model = 512
- `nl8` → num_layers = 8
- `nh16` → num_heads = 16
- `rope100000` → RoPE with base 100000 (for 8K-64K context)
- `sw4096` → Sliding window size 4096
- `rms` → RMSNorm
- `ckpt` → Gradient checkpointing enabled

---

### 示例 4：最小配置（快速测试）

**训练命令：**
```bash
python scripts/train.py \
  --d-model 64 \
  --num-layers 2 \
  --num-heads 4 \
  --epochs 5
```

**生成的文件名：**
```
checkpoints/shannon_b1_dm64_nl2_nh4_fixed_layer_20260411_120000.pt
```

**架构版本解析：**
- `dm64` → d_model = 64
- `nl2` → num_layers = 2
- `nh4` → num_heads = 4
- `fixed` → Fixed sinusoidal encoding (default when no RoPE/ALiBi)
- `layer` → LayerNorm (default)

---

## 🔍 实际应用场景

### 场景 1：快速识别模型架构

**问题：** checkpoints 目录中有多个模型文件，如何快速找到特定架构的模型？

**解决方案：**
```bash
# 查找所有 d_model=256 的模型
dir checkpoints\*dm256*.pt

# 查找所有使用 RoPE 的模型
dir checkpoints\*rope*.pt

# 查找所有启用梯度检查点的模型
dir checkpoints\*ckpt*.pt
```

---

### 场景 2：对比实验管理

**问题：** 进行了多组消融实验，如何组织结果？

**示例目录结构：**
```
checkpoints/
├── shannon_b1_dm128_nl4_nh8_rope10000_rms_20260410_100000.pt  # Baseline
├── shannon_b1_dm256_nl4_nh8_rope10000_rms_20260410_120000.pt  # ↑ d_model
├── shannon_b1_dm128_nl6_nh8_rope10000_rms_20260410_140000.pt  # ↑ layers
├── shannon_b1_dm128_nl4_nh8_alibi_layer_20260410_160000.pt    # ALiBi vs RoPE
└── shannon_b1_dm128_nl4_nh8_rope10000_rms_tie_20260410_180000.pt  # + Tie embeddings
```

通过文件名即可清晰对比各实验的配置差异。

---

### 场景 3：加载模型时验证兼容性

**问题：** 如何确保加载的 checkpoint 与当前代码兼容？

**解决方案：**
```python
import glob
import re

# 获取最新的 dm256 模型
checkpoint_files = glob.glob("checkpoints/*dm256*.pt")
latest_checkpoint = sorted(checkpoint_files)[-1]

print(f"Loading: {latest_checkpoint}")
# 输出: Loading: checkpoints/shannon_b1_dm256_nl6_nh8_rope10000_rms_20260411_164500.pt

# 从文件名提取架构信息
match = re.search(r'dm(\d+)_nl(\d+)_nh(\d+)_(rope\d+|alibi|fixed)_(layer|rms)', latest_checkpoint)
if match:
    d_model, num_layers, num_heads, pos_enc, norm_type = match.groups()
    print(f"Architecture: d_model={d_model}, layers={num_layers}, heads={num_heads}")
    print(f"Position Encoding: {pos_enc}, Normalization: {norm_type}")
```

---

### 场景 4：自动化脚本中的版本匹配

**问题：** 在 CI/CD 或自动化评估中，如何动态选择正确的模型？

**示例脚本：**
```bash
#!/bin/bash
# evaluate_latest.sh - 自动评估最新模型

# 查找最新的 RoPE + RMSNorm 模型
LATEST_MODEL=$(ls -t checkpoints/*rope*rms*.pt | head -n 1)

if [ -z "$LATEST_MODEL" ]; then
    echo "No compatible model found!"
    exit 1
fi

echo "Evaluating: $LATEST_MODEL"
python scripts/evaluate.py --model-path "$LATEST_MODEL" --dataset test_data.txt
```

---

## ❓ 常见问题

### Q1: 如果两个模型架构完全相同，文件名会冲突吗？

**A:** 不会。因为文件名包含**时间戳**（精确到秒），即使架构完全相同，文件名也会不同：

```
shannon_b1_dm256_nl6_nh8_rope10000_rms_20260411_164500.pt
shannon_b1_dm256_nl6_nh8_rope10000_rms_20260411_170000.pt  # 不同时间训练
```

---

### Q2: 可以自定义架构版本字符串的格式吗？

**A:** 当前实现是固定的格式。如需自定义，可以修改 [`scripts/train.py`](file://f:\Shannon-b1\scripts\train.py) 中的 `_generate_arch_version_string()` 函数。

**示例：添加学习率信息**
```python
def _generate_arch_version_string(config: ModelConfig) -> str:
    parts = []
    
    # 添加学习率（科学计数法简化）
    lr_short = f"{config.learning_rate:.0e}".replace('e-0', 'e-').replace('e-','e')
    parts.append(f"lr{lr_short}")
    
    return "_".join(parts)
```

---

### Q3: 旧的 checkpoint 没有架构版本怎么办？

**A:** 旧文件保持不变，新训练的文件会自动带版本。可以通过以下方式区分：

```bash
# 带版本的文件（新）
dir checkpoints\*dm*\_nl*\_nh*.pt

# 不带版本的文件（旧）
dir checkpoints\shannon_b1.pt
dir checkpoints\shannon_b1_.pt
```

建议逐步淘汰旧文件，或手动重命名以保持一致性。

---

### Q4: 架构版本字符串太长怎么办？

**A:** 当前设计已尽量精简，典型长度约 30-50 字符。如果觉得过长，可以：

1. **省略不重要的参数**（如去掉 `tie`、`ckpt` 等可选标志）
2. **使用更短的缩写**（如 `r` 代替 `rope`，但会降低可读性）
3. **仅保留核心参数**（`dm/nl/nh/pos_enc`）

**精简版示例：**
```python
# 只保留核心参数
parts = [
    f"dm{config.d_model}",
    f"nl{config.num_layers}",
    f"nh{config.num_heads}",
    pos_enc_short  # rope/alibi/fixed
]
```

---

### Q5: Web UI 能正确显示带版本的 checkpoint 吗？

**A:** 是的！Web UI 的 `/api/checkpoints` 端点会列出所有 `.pt` 文件，包括带版本的：

```json
[
  {
    "name": "shannon_b1_dm256_nl6_nh8_rope10000_rms_20260411_164500.pt",
    "path": "F:\\Shannon-b1\\checkpoints\\...",
    "size_mb": 12.5,
    "modified": "2026-04-11T16:45:00"
  }
]
```

前端会按修改时间倒序排列，最新版本在最上方。

---

### Q6: 如何从文件名反推训练配置？

**A:** 可以从架构版本字符串中提取关键信息，但**最可靠的方式**是加载 checkpoint 并读取其中的 `config` 对象：

```python
import torch

checkpoint = torch.load("checkpoints/shannon_b1_dm256_nl6_nh8_rope10000_rms_20260411_164500.pt")
config = checkpoint['config']

print(f"d_model: {config.d_model}")
print(f"num_layers: {config.num_layers}")
print(f"use_rope: {config.use_rope}")
print(f"rope_base: {config.rope_base}")
print(f"norm_type: {config.norm_type}")
```

架构版本字符串主要用于**快速筛选和识别**，完整配置应以 checkpoint 内保存的为准。

---

## 📚 相关文档

- [系统提示词功能文档](./SYSTEM_PROMPT_GUIDE.md)
- [训练操作指南](./TRAINING_GUIDE.md)
- [流式输出功能文档](./流式输出%20(Streaming)%20功能文档.md)
- [CUDA 兼容性说明](./CUDA_COMPATIBILITY.md)

---

## 🔄 更新日志

### v1.0.0 (2026-04-11)
- ✨ 首次实现架构版本字符串功能
- 📝 自动生成带版本的 checkpoint 文件名
- 🔧 支持 RoPE/ALiBi/Fixed 位置编码标识
- 🎯 包含核心架构参数（d_model, num_layers, num_heads）
- 🏷️ 支持可选特性标志（sliding_window, tie_embeddings, gradient_checkpointing）

---

**最后更新：** 2026-04-11  
**维护者：** Shannon-b1 开发团队