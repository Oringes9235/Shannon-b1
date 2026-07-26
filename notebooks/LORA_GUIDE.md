# LoRA 微调指南 (Low-Rank Adaptation)

## 概述

LoRA (Low-Rank Adaptation) 是一种参数高效的微调技术，通过在冻结的预训练权重旁添加低秩分解矩阵来实现领域适配。相比全量微调，LoRA 仅训练极少参数（通常 < 5%），大幅降低显存和训练成本。

Shannon-b1 内置完整的 LoRA 支持，覆盖训练、保存/加载、合并推理全流程。

---

## 快速开始

### 1. 应用 LoRA

```python
from src.model import ShannonB1, ModelConfig

config = ModelConfig(
    vocab_size=1000,
    d_model=128,
    num_heads=8,
    num_layers=4,
)
model = ShannonB1(config)

# 基本用法：Q + V projection 添加 LoRA
model.apply_lora(rank=8, alpha=16.0)

# 自定义目标模块
model.apply_lora(
    rank=8,
    alpha=16.0,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'out_proj']
)
```

### 2. 训练

```python
# 仅优化 LoRA 参数
lora_params = model.get_lora_trainable_params()
optimizer = torch.optim.AdamW(lora_params, lr=1e-3)

for epoch in range(epochs):
    for x, y in train_loader:
        logits, _ = model(x)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
```

### 3. CLI 训练

```bash
# LoRA 微调（低显存）
python scripts/train.py --epochs 10 --lora --lora-rank 8 --lr 0.001

# 全量微调（不加 --lora）
python scripts/train.py --epochs 10 --lr 0.0001
```

---

## API 参考

### `ShannonB1.apply_lora()`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `rank` | `int` | `8` | 低秩分解的秩，越大表达能力越强，参数越多 |
| `alpha` | `float` | `16.0` | 缩放因子，实际缩放 = alpha / rank |
| `dropout` | `float` | `0.0` | LoRA dropout 概率 |
| `target_modules` | `List[str]` | `["q_proj", "v_proj"]` | 目标模块：`q_proj`, `k_proj`, `v_proj`, `out_proj` |

### `ShannonB1.save_lora_weights(path)`

保存仅含 LoRA 参数的轻量级检查点（通常 < 1 MB）。

```python
model.save_lora_weights("checkpoints/lora_adapters.lora.pt")
```

### `ShannonB1.load_lora_weights(path)`

加载 LoRA 权重到已应用 LoRA 结构的模型。

```python
model2 = ShannonB1(config)
model2.load_state_dict(base_state_dict)  # 先加载基础权重
model2.apply_lora(rank=8, alpha=16.0)    # 再应用 LoRA 结构
model2.load_lora_weights("checkpoints/lora_adapters.lora.pt")
```

### `ShannonB1.merge_lora_weights()` / `unmerge_lora_weights()`

合并/分离 LoRA 权重到基础模型，用于推理加速或恢复训练。

```python
# 推理前合并
model.merge_lora_weights()   # W' = W + (alpha/rank) * BA
output = model(input_tokens)

# 继续训练前分离
model.unmerge_lora_weights()  # 还原 W
```

### `ShannonB1.get_lora_state_dict()`

获取仅包含 LoRA 参数的状态字典（含元数据）。

```python
state = model.get_lora_state_dict()
# keys: layer_0_q_proj_lora_A, layer_0_q_proj_lora_B, ...
# meta: _lora_rank, _lora_alpha, _lora_target_modules
```

### `ShannonB1.get_lora_trainable_params()`

返回所有 LoRA 可训练参数的列表，直接传给优化器。

```python
optimizer = torch.optim.AdamW(model.get_lora_trainable_params(), lr=1e-3)
```

---

## `LoRALinear` 底层模块

```python
from src.model.layers import LoRALinear

lora_linear = LoRALinear(
    existing_linear=nn.Linear(64, 128),
    rank=8,
    alpha=16.0,
    dropout=0.1,
)

# 前向: output = Wx + (alpha/rank) * B @ A @ x
output = lora_linear(input_tensor)

# 合并/分离
lora_linear.merge_weights_to_base()
lora_linear.unmerge_weights_from_base()

# 获取/加载 LoRA 参数
lora_A, lora_B = lora_linear.get_lora_params()
lora_linear.load_lora_params(lora_A, lora_B)
```

---

## 参数选择建议

| 场景 | rank | alpha | target_modules |
|------|------|-------|----------------|
| 极低资源（< 1% 参数） | 1-4 | 2× rank | `q_proj` |
| 标准配置 | 8 | 16 | `q_proj, v_proj` |
| 高质量适配 | 16-32 | 32 | `q_proj, k_proj, v_proj, out_proj` |

**学习率**：LoRA 通常使用比全量微调高 5-10 倍的学习率（如 `1e-3` vs `1e-4`）。

---

## 测试

```bash
python tests/test_lora.py
```

包含 29 个测试用例，覆盖：
- `LoRALinear` 创建/前向/合并/分离
- ShannonB1 LoRA 应用/冻结/训练/保存加载
- 边界情况（rank=1、空目标、多次调用、流式生成等）