# Shannon-b1 UI — 开发者指南

本文档面向开发者，说明如何在本地启动、调试和使用本仓库提供的 Web UI（包含后端 API 与前端界面），以及如何通过 UI 发起训练并监控训练日志/进度。

目录
- 简介
- 前置要求
- 安装与依赖
- 启动方法（开发/预览）
- UI 功能说明
- 后端训练行为说明
- 常见问题与排查

---

## 简介

`ui/` 提供一个轻量的 Web 控制面板：
- 后端：FastAPI（文件路径 `ui/server/app.py`），提供 REST + WebSocket。
- 前端：Vite + React（目录 `ui/client`），用于加载模型、启动/停止训练、可视化训练进度与日志。

后端在通过 UI 启动训练时，会以子进程运行仓库根目录下的 `scripts/train.py`（因此 UI 的训练行为与 CLI `scripts/train.py` 保持一致）。训练期间的 stdout 会被逐行转发到前端（通过 WebSocket `/ws`），并且 `checkpoints/` 会保存生成的模型与 tokenizer 文件。

## 前置要求

- Python 3.8+
- Node.js + npm
- 建议在虚拟环境中安装 Python 依赖
- 可选：CUDA 驱动与支持的 PyTorch（若要在 GPU 上训练）

## 安装与依赖

在项目根目录下安装 Python 依赖：

```bash
python -m pip install -r requirements.txt
```

前端依赖：

```bash
cd ui/client
npm install
```

（Windows 用户可以直接使用 `ui/runUI.bat` 一键启动后端与前端，见下）

## 启动方法

开发模式（推荐，用于调试前端/后端代码）

```powershell
# 在项目根（或在 ui 目录）使用一键脚本（Windows）
.\ui\runUI.bat

# 或者手动：
# 后端（在项目根）：
python ui/server/app.py
# 或使用 uvicorn（热重载）：
uvicorn ui.server.app:app --host 0.0.0.0 --port 8000 --reload

# 前端（新终端）：
cd ui/client
npm run dev
```

预览模式（production preview）

```bash
cd ui/client
npm run build
npm run preview -- --port 5173
```

UI 默认在 `http://localhost:5173`，后端监听 `http://0.0.0.0:8000`。

## UI 功能概览

- 模型加载：在前端可选择并加载 checkpoint（通过 `/api/model/load`）。
- 文本生成：使用加载的模型通过 `/api/generate` 接口生成文本。
- 训练控制：在“训练监控”界面可以配置超参数并点击“开始训练”或“停止训练”。
- 实时日志与进度：训练过程的 stdout 会被后端捕获并通过 WebSocket `/ws` 转发，前端会显示日志、训练进度条以及训练/验证损失曲线。

重要说明：前端使用一个共享的 WebSocket 连接（位于 `ui/client/src/ws.js`），可以保证在切换路由/组件卸载时不关闭连接，从而避免导航导致训练监控中断。

## 后端训练行为（关键点）

- 当从 UI 发起训练时，后端会在后台以子进程运行 `scripts/train.py`，命令行参数由前端配置映射而来（如 `--epochs`、`--batch-size`、`--seq-len` 等）。
- 后端会把训练子进程的 stdout/stderr 合并并逐行发送为 `training_log` 事件；同时会尝试从日志中解析 epoch/progress 并发送 `training_progress`、`training_epoch_complete` 等事件。前端订阅这些事件以更新界面。
- 检查点与 tokenizer 会保存在仓库根的 `checkpoints/` 目录（例如 `checkpoints/shannon_b1.pt`、`*_tokenizer.json`）。

## REST & WebSocket 简要接口

- GET `/api/status` — 服务与模型加载状态。
- POST `/api/model/load` — 加载模型（path）。
- POST `/api/generate` — 文本生成。
- POST `/api/train/start` — 发起训练（body 为训练配置 JSON）。
- POST `/api/train/stop` — 请求停止当前训练。
- GET `/api/train/status` — 当前训练状态（进度、epoch、loss）。
- GET `/api/checkpoints` — 列出 `checkpoints/` 下的文件。
- WebSocket `/ws` — 推送训练事件（`training_log`、`training_progress`、`training_epoch_complete`、`training_started`、`training_completed`、`training_error`、`training_stopped` 等）。

## 调试与常见问题

- 如果 UI 显示“训练已开始”但没有真实子进程：检查后端日志（运行 `ui/server/app.py` 的终端），确认 `train.py` 路径存在且可执行。
- Windows 控制台编码问题：若出现编码异常，可在 PowerShell 中运行 `chcp 65001` 或从 `runUI.bat` 启动（脚本已尝试设置 UTF-8 编码变量）。
- 若 WebSocket 无法连接：确认后端正常启动并监听 8000 端口；检查浏览器控制台与后端日志。
- 依赖问题：确保已在项目根安装 Python 依赖，并在 `ui/client` 下运行 `npm install`。

## 快速命令示例（从外部触发训练）

```bash
curl -X POST http://localhost:8000/api/train/start -H "Content-Type: application/json" -d '{"epochs":5,"batch_size":16,"seq_len":64}'

# 查询训练状态
curl http://localhost:8000/api/train/status

# 停止训练
curl -X POST http://localhost:8000/api/train/stop
```

## 开发者小贴士

- 要让 UI 的训练逻辑与 CLI 完全一致，优先修改 `scripts/train.py`（它是实际执行训练的入口）。
- 后端事件转发依赖 WebSocket 的稳定性；若需更可靠的持久化或日志检索，考虑将训练 stdout 同步写入文件或接入日志队列。
- 若要扩展前端的训练可视化（例如更细粒度的 batch-level loss），可以在 `scripts/train.py` 中打印约定格式的 JSON 行，然后在 `ui/server/training_worker.py` 中解析并转发为结构化事件。

---

如果需要，我可以：
- 把 README 翻译为英文版；
- 在 README 中加入示例截图或更详细的 API 示例；
- 或把“快速测试脚本”添加为 `ui/tools/quick_test.sh`（或 .bat）。
