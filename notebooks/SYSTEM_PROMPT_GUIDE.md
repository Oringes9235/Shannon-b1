# Shannon-b1 系统提示词 (System Prompt) 功能文档

## 📖 目录
- [什么是系统提示词？](#什么是系统提示词)
- [为什么需要系统提示词？](#为什么需要系统提示词)
- [功能实现架构](#功能实现架构)
- [使用方法](#使用方法)
- [使用场景示例](#使用场景示例)
- [技术细节](#技术细节)
- [常见问题](#常见问题)

---

## 🎯 什么是系统提示词？

**系统提示词（System Prompt）**是一种特殊的文本前缀，用于在生成任务开始前为模型设定角色、行为准则或上下文背景。它会在用户输入的提示词之前自动添加，影响模型的生成风格和输出内容。

### 工作原理

```
完整输入 = 系统提示词 + "\n\n" + 用户提示词
```

例如：
```
系统提示词: "你是一个专业的翻译助手"
用户提示词: "Hello, how are you?"

实际发送给模型的文本:
"你是一个专业的翻译助手

Hello, how are you?"
```

---

## 💡 为什么需要系统提示词？

### 1. **角色定位**
让模型扮演特定角色，如：
- 编程专家
- 翻译官
- 教师
- 创意作家

### 2. **行为控制**
规范输出风格：
- 正式 vs 随意
- 详细 vs 简洁
- 学术化 vs 口语化

### 3. **上下文注入**
提供额外背景信息，提升生成质量：
- 领域知识
- 特殊要求
- 格式规范

### 4. **多轮对话基础**
为未来实现聊天功能做准备，区分系统指令和用户输入。

---

## 🏗️ 功能实现架构

系统提示词功能采用**四层架构**设计，确保前后端一致性：

```
┌─────────────────────────────────────┐
│  前端层 (TextGenerator.jsx)          │
│  - UI输入框                          │
│  - localStorage持久化                │
│  - API请求参数传递                   │
└──────────────┬──────────────────────┘
               │ HTTP POST /api/generate
               ▼
┌─────────────────────────────────────┐
│  API层 (app.py)                      │
│  - GenerateRequest模型定义           │
│  - 参数验证                          │
│  - 路由分发                          │
└──────────────┬──────────────────────┘
               │ 调用
               ▼
┌─────────────────────────────────────┐
│  服务层 (model_manager.py)           │
│  - generate() / generate_stream()   │
│  - 文本拼接逻辑                      │
│  - 模型推理                          │
└──────────────┬──────────────────────┘
               │ 编码
               ▼
┌─────────────────────────────────────┐
│  模型层 (ShannonB1)                  │
│  - Token编码                         │
│  - 自回归生成                        │
└─────────────────────────────────────┘
```

### 关键代码位置

| 层级 | 文件 | 关键函数/类 |
|------|------|------------|
| 前端 | `ui/client/src/components/TextGenerator.jsx` | `systemPrompt` state, `handleGenerate()` |
| API | `ui/server/app.py` | `GenerateRequest`, `/api/generate`, `/api/generate/stream` |
| 服务 | `ui/server/model_manager.py` | `generate()`, `generate_stream()` |
| CLI | `scripts/generate.py` | `--system-prompt` 参数, `main()` |

---

## 🚀 使用方法

### 方法一：Web UI 界面

1. **启动服务**
   ```bash
   # Windows
   ui\runUI.bat
   
   # Linux/Mac
   ui/runUI.sh
   ```

2. **访问界面**
   - 打开浏览器访问: `http://localhost:5173`
   - 进入"文本生成"页面

3. **填写系统提示词**
   - 在"🤖 系统提示词 (System Prompt)"输入框中输入角色设定
   - 例如: `"你是一个专业的Python程序员"`
   - 该字段为**可选**，不填写不影响正常功能

4. **填写用户提示词**
   - 在"📝 用户提示词 (User Prompt)"输入框中输入具体问题
   - 例如: `"请解释什么是装饰器"`

5. **调整参数并生成**
   - 设置温度、Top-K等参数
   - 点击"🚀 开始生成"按钮
   - 查看实时生成结果

6. **参数自动保存**
   - 系统提示词会自动保存到浏览器localStorage
   - 下次访问时自动恢复上次的设置

---

### 方法二：命令行工具 (CLI)

```bash
python scripts/generate.py \
  --model-path checkpoints/shannon_b1.pt \
  --system-prompt "你是一个翻译专家，将英文翻译成中文" \
  --prompt "Hello, how are you today?" \
  --max-tokens 50 \
  --temperature 0.8
```

#### 参数说明

| 参数 | 说明 | 默认值 | 必填 |
|------|------|--------|------|
| `--model-path` | 模型文件路径 | - | ✅ |
| `--system-prompt` | 系统提示词 | None | ❌ |
| `--prompt` | 用户提示词 | "The " | ❌ |
| `--max-tokens` | 最大生成token数 | 100 | ❌ |
| `--temperature` | 温度参数 | 0.8 | ❌ |
| `--top-k` | Top-K采样 | 50 | ❌ |
| `--repetition-penalty` | 重复惩罚 | 1.1 | ❌ |

#### 示例输出

```
🔄 加载模型...
✅ 模型加载完成: vocab=200, d_model=128
📝 分词器类型: Char

============================================================
🤖 System Prompt: 你是一个翻译专家，将英文翻译成中文
------------------------------------------------------------
💬 User Prompt: Hello, how are you today?
============================================================

🚀 开始流式生成:

你好，你今天好吗？

============================================================
✅ 生成完成!
📊 统计信息:
   - 生成token数: 12
   - 耗时: 2.35秒
   - 速度: 5.11 tokens/秒
============================================================
```

---

## 🎨 使用场景示例

### 场景1：角色扮演 - 编程助手

**系统提示词：**
```
你是一个经验丰富的Python程序员，擅长代码审查和最佳实践建议。请用专业但易懂的语言回答问题，并提供代码示例。
```

**用户提示词：**
```
如何优化以下列表推导式的性能？
result = [x**2 for x in range(1000) if x % 2 == 0]
```

---

### 场景2：翻译任务

**系统提示词：**
```
你是一名专业的中英翻译，精通两种语言的文化差异。请将英文翻译成流畅的中文，保持原意并符合中文表达习惯。
```

**用户提示词：**
```
The quick brown fox jumps over the lazy dog.
```

---

### 场景3：创意写作

**系统提示词：**
```
你是一位科幻小说作家，擅长构建未来世界和科技想象。请用生动的描写和富有张力的叙事风格创作故事。
```

**用户提示词：**
```
在2150年的火星殖民地，人类发现了...
```

---

### 场景4：学术写作

**系统提示词：**
```
你是一名学术论文编辑，请使用正式、客观的学术语言，避免口语化表达，注重逻辑严谨性和术语准确性。
```

**用户提示词：**
```
总结深度学习在自然语言处理中的应用现状。
```

---

### 场景5：代码格式化

**系统提示词：**
```
你是一个JSON格式化工具，只输出合法的JSON数据，不要添加任何解释性文字。
```

**用户提示词：**
```
创建一个包含姓名、年龄和城市的用户对象
```

**期望输出：**
```json
{"name": "张三", "age": 25, "city": "北京"}
```

---

## 🔧 技术细节

### 1. 文本拼接逻辑

在服务层 (`model_manager.py`) 中，系统提示词与用户提示词的拼接遵循以下规则：

```python
def generate(self, prompt: str, system_prompt: Optional[str] = None, ...):
    # 构建完整的输入文本
    full_prompt = prompt
    if system_prompt and system_prompt.strip():
        # 仅在system_prompt非空且非纯空白时生效
        full_prompt = f"{system_prompt.strip()}\n\n{prompt}"
    
    # 编码并生成
    start_tokens = self.tokenizer.encode(full_prompt)[:50]
    ...
```

**关键点：**
- ✅ 使用双换行符 `\n\n` 分隔，增强语义隔离
- ✅ 自动去除首尾空白 `.strip()`
- ✅ 空字符串或None时不添加
- ✅ 总长度限制为50个token（防止超出序列长度）

---

### 2. API请求格式

**请求体 (Request Body):**
```json
{
  "prompt": "用户提示词",
  "system_prompt": "系统提示词(可选)",
  "max_tokens": 100,
  "temperature": 0.8,
  "top_k": 40,
  "top_p": 0.9,
  "repetition_penalty": 1.15
}
```

**响应体 (Response Body) - 非流式:**
```json
{
  "success": true,
  "prompt": "用户提示词",
  "generated_text": "生成的完整文本",
  "tokens_generated": 45,
  "temperature": 0.8
}
```

**SSE流式响应:**
```
data: {"token_id": 123, "text": "...", "probability": 0.85, "tokens_generated": 1, "is_complete": false}

data: {"token_id": 456, "text": "...", "probability": 0.92, "tokens_generated": 2, "is_complete": false}

...

data: {"type": "complete"}
```

---

### 3. 前端状态管理

在 `TextGenerator.jsx` 中：

```javascript
// 从localStorage读取保存的系统提示词
const savedSystemPrompt = localStorage.getItem('shannon_system_prompt') || ''
const [systemPrompt, setSystemPrompt] = useState(savedSystemPrompt)

// 监听变化并自动保存
useEffect(() => {
  localStorage.setItem('shannon_system_prompt', systemPrompt)
}, [systemPrompt])

// API请求时传递参数
await axios.post(`${apiUrl}/generate`, {
  prompt,
  system_prompt: systemPrompt || undefined,  // 空字符串转为undefined
  max_tokens: maxTokens,
  ...
})
```

---

### 4. 参数命名规范

| 层级 | 参数名 | 说明 |
|------|--------|------|
| Python后端 | `system_prompt` | 下划线命名(snake_case) |
| JavaScript前端 | `systemPrompt` | 驼峰命名(camelCase) |
| JSON传输 | `system_prompt` | 保持与后端一致 |
| localStorage键 | `shannon_system_prompt` | 带项目前缀 |

**注意：** 前后端通过Pydantic模型自动转换命名风格，确保序列化正确。

---

## ❓ 常见问题

### Q1: 系统提示词会影响生成速度吗？

**A:** 会轻微影响。因为系统提示词会增加输入token数量，导致：
- 预填充阶段(Prefill)计算量增加
- 但解码阶段(Decoding)速度不变

**建议：** 保持系统提示词简洁（不超过20-30个token）。

---

### Q2: 可以不填系统提示词吗？

**A:** 完全可以！系统提示词是**可选参数**。不填写时，行为与之前版本完全一致。

---

### Q3: 系统提示词太长会怎样？

**A:** 当前实现会将完整提示词（系统+用户）截断为最多50个token：
```python
start_tokens = self.tokenizer.encode(full_prompt)[:50]
```

如果需要更长的上下文，可以修改此限制，但需注意：
- 不能超过模型的 `max_seq_len`
- 会减少可用于生成的token空间

---

### Q4: 如何在多轮对话中使用系统提示词？

**A:** 当前版本仅支持单轮生成。如需多轮对话，可以：
1. 手动拼接历史对话到用户提示词
2. 保持系统提示词不变

**未来规划：** 将在后续版本中原生支持多轮对话和会话管理。

---

### Q5: 修改系统提示词后需要重启服务吗？

**A:** 
- **Web UI:** 不需要，修改后立即生效
- **CLI:** 不需要，每次运行都是独立的
- **后端API:** 如果修改了代码逻辑（如拼接方式），需要重启 `app.py`

---

### Q6: 系统提示词能改变模型的知识吗？

**A:** 不能。系统提示词只能：
- ✅ 引导生成风格
- ✅ 设定角色定位
- ✅ 提供格式要求

但不能：
- ❌ 注入训练时未见的新知识
- ❌ 改变模型的底层能力
- ❌ 突破模型的能力边界

---

### Q7: 不同温度参数下，系统提示词的效果有差异吗？

**A:** 有差异：
- **低温度 (0.1-0.5):** 模型更严格遵守系统提示词的指令
- **高温度 (0.8-1.5):** 模型更有创造性，可能偏离系统提示词

**建议：** 对于需要严格遵循指令的任务（如翻译、格式化），使用较低温度。

---

## 📝 最佳实践

### 1. 编写有效的系统提示词

✅ **好的示例：**
```
"你是一名资深医生，请用通俗易懂的语言解释医学概念，避免使用过多专业术语。"
```

❌ **不好的示例：**
```
"你是一个AI。"  (太模糊)
"你必须严格按照以下10条规则..."  (过于复杂，模型可能无法全部记住)
```

### 2. 长度控制

- **推荐长度：** 10-50个字符
- **最大长度：** 不超过100个字符（约20-30个token）
- **原则：** 简洁明了，突出核心指令

### 3. 明确具体

✅ **明确：**
```
"请用三句话总结以下内容"
```

❌ **模糊：**
```
"总结一下"
```

### 4. 测试与迭代

1. 先用简单提示词测试效果
2. 根据输出调整措辞
3. 保存最有效的版本到localStorage

---

## 🔍 调试技巧

### 查看实际发送的完整提示词

在浏览器控制台（F12）中：

```javascript
// 拦截API请求
const originalFetch = window.fetch;
window.fetch = function(...args) {
  console.log('Request body:', args[1]?.body);
  return originalFetch.apply(this, args);
};
```

或在后端日志中查看：

```python
# 在 model_manager.py 中添加调试日志
print(f"[DEBUG] Full prompt: {full_prompt}")
print(f"[DEBUG] Tokens: {start_tokens}")
```

---

## 📚 相关文档

- [Shannon-b1 项目README](../README.md)
- [流式输出功能文档](../../notebooks/流式输出%20(Streaming)%20功能文档.md)
- [训练操作指南](../../notebooks/TRAINING_GUIDE.md)

---

## 🆕 更新日志

### v1.0.0 (2026-04-11)
- ✨ 首次实现系统提示词功能
- 🎨 Web UI添加系统提示词输入框
- 🔧 CLI支持 `--system-prompt` 参数
- 📝 前后端API同步更新
- 💾 localStorage持久化保存

---

## 📞 支持与反馈

如有问题或建议，请：
1. 检查本文档的"常见问题"部分
2. 查看项目GitHub Issues
3. 提交新的Issue描述你的问题

---

**最后更新：** 2026-04-11  
**维护者：** Shannon-b1 开发团队