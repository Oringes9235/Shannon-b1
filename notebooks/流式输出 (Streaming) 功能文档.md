# 流式输出 (Streaming) 功能文档

## 概述

Shannon-b1 现已支持流式文本生成功能，允许在生成过程中实时显示每个token，提供更好的用户体验。流式输出特别适用于：

- 🎯 **交互式应用**：用户可以看到AI正在"思考"和生成
- ⚡ **长文本生成**：无需等待全部生成完成即可开始阅读
- 🔍 **调试和监控**：实时观察模型的生成过程

## 技术实现

### 后端架构

1. **模型层** (`src/model/shannon.py`)
   - 新增 `generate_stream()` 方法，使用Python Generator实现
   - 逐个yield生成的token及其概率信息
   - 保持与原有`generate()`方法相同的采样策略和参数

2. **服务层** (`ui/server/model_manager.py`)
   - 新增 `generate_stream()` 方法包装模型流式生成
   - 实时解码token为文本并返回结构化数据

3. **API层** (`ui/server/app.py`)
   - 新增 `/api/generate/stream` 端点
   - 使用SSE (Server-Sent Events) 协议传输
   - 支持标准HTTP流式响应

### 前端实现

- **TextGenerator组件** (`ui/client/src/components/TextGenerator.jsx`)
  - 支持流式和非流式两种模式切换
  - 使用Fetch API + ReadableStream接收SSE数据
  - 实时更新的UI显示和光标动画
  - 支持中途停止生成

## 使用方法

### 1. Web UI 流式生成

启动Web界面后，在文本生成页面：

1. ✅ 勾选"启用流式输出"选项
2. 📝 输入提示词和调整参数
3. 🚀 点击"生成文本"按钮
4. 👀 实时观察文本逐字生成
5. ⏹️ 可随时点击"停止"按钮中断生成

### 2. 命令行流式生成

使用新增的流式生成脚本：

```bash
# 基本用法
python scripts/generate_stream.py \
    --model-path checkpoints/shannon_b1.pt \
    --prompt "The future of AI is" \
    --max-tokens 100 \
    --temperature 0.85

# 自定义打字速度
python scripts/generate_stream.py \
    --model-path checkpoints/shannon_b1.pt \
    --prompt "Once upon a time" \
    --delay 0.1 \
    --top-k 40

# 快速生成（无延迟）
python scripts/generate_stream.py \
    --model-path checkpoints/shannon_b1.pt \
    --prompt "In the beginning" \
    --delay 0
```

**参数说明：**
- `--model-path`: 模型检查点文件路径（必需）
- `--prompt`: 提示词文本
- `--max-tokens`: 最大生成token数，默认100
- `--temperature`: 温度参数，控制随机性
- `--top-k`: Top-K采样参数
- `--top-p`: Top-P采样参数
- `--repetition-penalty`: 重复惩罚系数
- `--device`: 运行设备（cpu/cuda）
- `--delay`: 每个token间的延迟（秒），模拟打字效果

### 3. API调用

#### SSE流式接口

```javascript
// JavaScript示例
const response = await fetch('http://localhost:8000/api/generate/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        prompt: "The answer is",
        max_tokens: 100,
        temperature: 0.85,
        top_k: 40,
        repetition_penalty: 1.15
    })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    
    const text = decoder.decode(value);
    // 解析SSE格式数据
    text.split('\n\n').forEach(line => {
        if (line.startsWith('data: ')) {
            const data = JSON.parse(line.slice(6));
            console.log('Generated:', data.text);
        }
    });
}
```

#### Python示例

```python
import requests
import json

response = requests.post(
    'http://localhost:8000/api/generate/stream',
    json={
        'prompt': 'The meaning of life is',
        'max_tokens': 50,
        'temperature': 0.9
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        line = line.decode('utf-8')
        if line.startswith('data: '):
            data = json.loads(line[6:])
            print(data['text'], end='', flush=True)
```

## SSE数据格式

流式接口返回的数据格式：

```json
{
    "token_id": 1234,
    "text": "当前生成的完整文本",
    "probability": 0.85,
    "tokens_generated": 10,
    "is_complete": false
}
```

**字段说明：**
- `token_id`: 当前生成的token ID
- `text`: 截至目前生成的完整文本
- `probability`: 当前token的生成概率
- `tokens_generated`: 已生成的token数量
- `is_complete`: 是否为最后一个chunk

完成信号：
```json
{
    "type": "complete"
}
```

错误信号：
```json
{
    "type": "error",
    "error": "错误信息"
}
```

## 性能优化建议

### 1. 批处理优化
对于高并发场景，可以考虑：
- 使用异步生成器
- 实现请求队列和批处理
- 添加缓存机制

### 2. 网络优化
- 启用GZIP压缩（SSE通常不压缩，但可考虑其他方案）
- 使用WebSocket替代SSE（双向通信需求时）
- CDN边缘计算（大规模部署）

### 3. 前端优化
- 使用虚拟滚动处理大量文本
- 防抖更新避免频繁重渲染
- Web Worker处理数据解析

## 故障排除

### 常见问题

**Q1: 流式输出卡顿或不流畅**
- 检查网络延迟
- 降低`--delay`参数值
- 确认服务器性能充足

**Q2: 浏览器不支持SSE**
- 现代浏览器均支持SSE
- 降级方案：使用轮询或WebSocket

**Q3: 生成中途断开**
- 检查网络连接稳定性
- 增加超时时间设置
- 实现自动重连机制

**Q4: 内存占用过高**
- 减少`max_tokens`参数
- 及时清理未使用的连接
- 监控服务器资源使用

## 进阶功能

### 1. 自定义回调

可以在模型生成过程中添加自定义回调：

```python
def custom_callback(token_id, text, probability):
    print(f"Token {token_id}: {text} (p={probability:.3f})")

for token_id, prob in model.generate_stream(...):
    custom_callback(token_id, tokenizer.decode([token_id]), prob)
```

### 2. 流式评估

实时监控生成质量：

```python
from collections import Counter

def monitor_quality(generated_tokens):
    # 计算重复率
    counter = Counter(generated_tokens)
    unique_ratio = len(counter) / len(generated_tokens)
    return unique_ratio
```

### 3. 多路复用

同时生成多个候选文本：

```python
import asyncio

async def generate_multiple(prompts):
    tasks = [generate_stream(prompt) for prompt in prompts]
    return await asyncio.gather(*tasks)
```

## 最佳实践

1. **用户体验**
   - 显示生成进度指示器
   - 提供停止按钮
   - 添加打字机音效（可选）

2. **错误处理**
   - 优雅处理网络中断
   - 显示友好的错误消息
   - 实现自动重试机制

3. **性能监控**
   - 记录生成速度（tokens/秒）
   - 监控服务器负载
   - 追踪用户交互数据

4. **安全性**
   - 限制最大token数
   - 实现速率限制
   - 验证输入内容

## 未来规划

- [ ] 支持WebSocket双向通信
- [ ] 实现增量渲染优化
- [ ] 添加生成质量实时评分
- [ ] 支持多模型并行生成
- [ ] 集成语音合成（TTS）

## 参考资料

- [Server-Sent Events规范](https://html.spec.whatwg.org/multipage/server-sent-events.html)
- [FastAPI StreamingResponse文档](https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse)
- [ReadableStream API](https://developer.mozilla.org/en-US/docs/Web/API/ReadableStream)

---

**最后更新**: 2026-04-06  
