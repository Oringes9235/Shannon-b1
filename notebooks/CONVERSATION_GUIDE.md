# 多轮对话管理指南

## 概述

多轮对话管理模块 (`src/utils/conversation.py`) 提供了完整的对话数据结构、历史管理和提示词格式化功能，使 Shannon-b1 能够维持多轮对话上下文。

核心组件：
- **Message** — 单条消息（角色/内容/时间戳）
- **Conversation** — 对话管理器（历史维护、prompt 构建、截断）
- **ConversationTemplate** — 可插拔的提示词格式化模板

---

## 快速开始

### 基本用法

```python
from src.utils import Conversation

# 创建对话（可选系统提示词）
conv = Conversation(system_prompt="你是一个翻译助手，将中文翻译为英文。")

# 添加消息
conv.add_user("你好")
conv.add_assistant("Hello")
conv.add_user("今天天气怎么样？")
conv.add_assistant("How's the weather today?")

# 构建模型输入
prompt = conv.build_prompt()
print(prompt)
# [SYSTEM] 你是一个翻译助手，将中文翻译为英文。
#
# [USER] 你好
# [ASSISTANT] Hello
# [USER] 今天天气怎么样？
# [ASSISTANT] How's the weather today?
# [ASSISTANT]
```

### 缩写 API

```python
conv = Conversation(system_prompt="你是助手")
conv.add_user("第一轮问题")

# 获取后更新
reply = model_generate(conv.build_prompt())
conv.add_assistant(reply)

conv.add_user("第二轮问题")
# ...
```

---

## 预置模板

### SIMPLE（默认）

```
[SYSTEM] 内容
[USER] 内容
[ASSISTANT] 内容
[ASSISTANT]
```

```python
from src.utils import Conversation, SIMPLE_TEMPLATE

conv = Conversation(template=SIMPLE_TEMPLATE)
```

### ChatML（OpenAI 风格）

```
<|im_start|>system
内容<|im_end|>
<|im_start|>user
内容<|im_end|>
<|im_start|>assistant
内容<|im_end|>
<|im_start|>assistant
```

```python
from src.utils import CHATML_TEMPLATE

conv = Conversation(template=CHATML_TEMPLATE)
```

### Llama3

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
内容<|eot_id|>
<|start_header_id|>user<|end_header_id|>
内容<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
内容<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
```

```python
from src.utils import LLAMA3_TEMPLATE

conv = Conversation(template=LLAMA3_TEMPLATE)
```

---

## 自定义模板

```python
from src.utils import ConversationTemplate, register_template

custom = ConversationTemplate(
    name="my_format",
    system_start="<system>",
    system_end="</system>\n",
    user_start="<human>",
    user_end="</human>\n",
    assistant_start="<bot>",
    assistant_end="</bot>\n",
    suffix="<bot>",
)

# 使用
conv = Conversation(system_prompt="你是助手", template=custom)

# 注册到全局，可按名称获取
register_template(custom)
```

---

## 上下文窗口截断

当对话历史增长超过 `max_context_length` 时，自动裁剪最早的对话轮次（始终保留 system prompt）。

```python
# 限制最大 500 字符
conv = Conversation(
    system_prompt="系统提示",
    max_context_length=500,
)

# 大量对话...
for i in range(100):
    conv.add_user(f"第 {i} 轮问题")
    conv.add_assistant(f"第 {i} 轮回答")

# 自动截断
prompt = conv.build_prompt_truncated()
assert len(prompt) <= 500
```

**截断策略**：二分查找最优起始位置，以保留 system prompt + 最近 N 轮对话。

---

## 与 ModelManager 集成

`model_manager.py` 的 `generate()` / `generate_stream()` 已集成多轮对话支持，传入 `conversation` 参数即可自动维护对话历史。

### 非流式

```python
from src.utils import Conversation
from model_manager import ModelManager

manager = ModelManager()
manager.load_model("checkpoints/model.pt")

# 多轮对话
conv = Conversation(system_prompt="你是助手")

# 第一轮
result = manager.generate(
    prompt="你好",
    conversation=conv,  # 自动使用对话历史
)
print(result["assistant_reply"])

# 第二轮（conv 已自动更新）
result = manager.generate(
    prompt="1+1 等于几？",
    conversation=conv,
)
print(result["assistant_reply"])

# 获取完整对话历史
print(conv.to_json())
```

### 流式

```python
for chunk in manager.generate_stream(
    prompt="讲个笑话",
    conversation=conv,
):
    if not chunk.get("is_complete"):
        print(chunk["text"], end="\r")
    else:
        # 完成时包含 assistant_reply 和 conversation
        print(f"\n完成: {chunk['assistant_reply']}")
```

---

## 序列化

### JSON 保存/加载

```python
# 保存到文件
conv.to_json("conversations/chat_001.json")

# 从文件恢复
conv = Conversation.from_json("conversations/chat_001.json")

# 字符串 ↔ 对象
json_str = conv.to_json()
conv = Conversation.from_json(json_str)
```

### 字典 ↔ 对象

```python
data = conv.to_dict()
conv = Conversation.from_dict(data)
```

### JSON 结构

```json
{
  "messages": [
    {
      "role": "system",
      "content": "你是助手",
      "timestamp": "2026-01-15T10:30:00"
    },
    {
      "role": "user",
      "content": "你好",
      "timestamp": "2026-01-15T10:30:05"
    },
    {
      "role": "assistant",
      "content": "你好！有什么可以帮你？",
      "timestamp": "2026-01-15T10:30:06"
    }
  ],
  "template": "simple",
  "max_context_length": 4096
}
```

---

## API 参考

### `Message`

| 属性/方法 | 说明 |
|-----------|------|
| `role: str` | 角色：`"system"` / `"user"` / `"assistant"` |
| `content: str` | 消息内容 |
| `timestamp: str` | ISO 格式时间戳（自动生成） |
| `to_dict()` | 序列化为字典 |
| `from_dict(d)` | 从字典反序列化 |

### `Conversation`

| 属性/方法 | 说明 |
|-----------|------|
| `system_prompt` | 获取/设置系统提示词 |
| `history` | 返回非 system 的消息列表 |
| `last_message` | 最后一条非 system 消息 |
| `add_message(role, content)` | 添加任意角色消息 |
| `add_user(content)` | 添加用户消息 |
| `add_assistant(content)` | 添加助手消息 |
| `add_system(content)` | 设置系统提示词 |
| `clear(keep_system=True)` | 清空对话历史 |
| `build_prompt(template?)` | 构建格式化 prompt |
| `build_prompt_truncated()` | 构建并自动截断 |
| `to_dict()` / `from_dict(d)` | 字典序列化 |
| `to_json(path?)` / `from_json(str)` | JSON 序列化 |
| `__len__()` | 消息总数 |

### `ConversationTemplate`

| 属性 | 说明 |
|------|------|
| `name` | 模板名称 |
| `system_start/end` | system 消息包裹 |
| `user_start/end` | user 消息包裹 |
| `assistant_start/end` | assistant 消息包裹 |
| `separator` | 消息间分隔符 |
| `suffix` | 末尾追加（提示模型开始回答） |

---

## 测试

```bash
python tests/test_conversation.py
```

24 个测试用例，覆盖：
- Message 创建/序列化/无效角色
- Conversation 增删消息/清空/长度管理
- 3 种模板 prompt 构建验证
- 上下文截断逻辑
- JSON/字典序列化往返
- 自定义模板注册

---

## 完整示例

```python
from src.utils import Conversation, CHATML_TEMPLATE

# 初始化
conv = Conversation(
    system_prompt="你是一个 Python 专家，回答要简洁准确。",
    template=CHATML_TEMPLATE,
    max_context_length=2048,
)

# 模拟多轮对话
conv.add_user("Python 中如何反转列表？")
conv.add_assistant("使用 list.reverse() 或切片 [::-1]")
conv.add_user("哪个更高效？")
conv.add_assistant("list.reverse() 是原地操作，O(n) 且不创建新对象")

# 构建输入
prompt = conv.build_prompt_truncated()

# 生成回复
# ... 调用模型生成 ...

# 持久化
conv.to_json("conversations/python_qa.json")