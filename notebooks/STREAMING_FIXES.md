# 流式生成一致性修复说明

## 问题描述

在测试中发现 `generate_stream()` 和 `generate()` 方法即使在相同参数下（特别是 `temperature=0.0`）也会产生不同的结果。

## 根本原因

两个方法在处理**动态序列状态**时存在不一致：

### 1. `ban_immediate_repeat` 检查错误
- **错误代码**：使用固定的 `tokens` 变量（只包含start_tokens）
- **影响**：无法正确检测新生成token的立即重复

### 2. `repetition_penalty` 应用对象错误  
- **错误代码**：基于固定的 `tokens` 计算重复惩罚
- **影响**：只对start_tokens应用惩罚，忽略已生成的token

### 3. `max_repetition` 检查时机错误
- **错误代码**：在采样后使用 `break` 中断
- **影响**：与正确的"采样前屏蔽"逻辑不一致

## 修复方案

### 修复1：统一使用动态序列状态

```python
# 修复前（错误）
generated = set(tokens[0].tolist())  # ❌ 固定不变
if ban_immediate_repeat and tokens.size(1) > 0:  # ❌ 总是True
    prev_token = int(tokens[0, -1].item())

# 修复后（正确）
generated = set(cur_tokens[0].tolist())  # ✅ 动态更新
if ban_immediate_repeat and cur_tokens.size(1) > 0:  # ✅ 检查当前长度
    prev_token = int(cur_tokens[0, -1].item())
```

### 修复2：采样前应用所有约束

```python
# 修复前（错误）
next_token = torch.multinomial(probs, 1).item()
if token_counts[next_token] >= max_rep:
    break  # ❌ 采样后中断

# 修复后（正确）
# 在采样前屏蔽超过限制的token
for tok_id, cnt in list(token_counts.items()):
    if cnt >= max_rep:
        last_logits[tok_id] = float('-inf')  # ✅ 采样前屏蔽

next_token = torch.argmax(last_logits).item()  # 贪婪解码
```

### 修复3：temperature=0.0的特殊处理

```python
if temperature == 0.0:
    # 先应用所有惩罚和约束
    # ... (repetition_penalty, ban_immediate_repeat, etc.)
    
    # 最大重复限制
    for tok_id, cnt in list(token_counts.items()):
        if cnt >= max_rep:
            last_logits[tok_id] = float('-inf')
    
    # 贪婪解码
    next_token = torch.argmax(last_logits).item()
    probs = torch.softmax(last_logits, dim=-1)
    probability = probs[next_token].item()
    
    yield (next_token, probability)
    # 更新状态并继续
    continue
```

## 验证步骤

运行以下测试验证修复：

```bash
# 1. 严格一致性测试
python tests/strict_consistency.py

# 2. 详细调试测试
python tests/debug_consistency.py

# 3. 完整测试套件
python -m pytest tests/test_streaming.py -v
```

## 关键原则

1. **动态状态一致性**：所有基于序列状态的检查必须使用当前正在构建的序列（`cur_tokens`），而非初始输入（`tokens`）

2. **约束前置**：所有限制条件必须在采样/选择之前通过修改logits应用，而非之后中断

3. **确定性保证**：`temperature=0.0` 时必须使用 `torch.argmax`，并确保所有约束正确应用

## 影响范围

- ✅ `src/model/shannon.py` - ShannonB1.generate()
- ✅ `src/model/shannon.py` - ShannonB1.generate_stream()
- ✅ 两个方法现在行为完全一致
