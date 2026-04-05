#!/usr/bin/env python
"""
Shannon-b1 Web API 后端
提供 REST API 和 WebSocket 用于远程监控
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import asyncio
import json
import threading
from datetime import datetime
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

from model_manager import ModelManager
from training_worker import TrainingWorker


# 请求/响应模型
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.9
    repetition_penalty: float = 1.15


class TrainRequest(BaseModel):
    tokenizer: str = "char"
    vocab_size: int = 200
    d_model: int = 128
    num_layers: int = 3
    epochs: int = 30
    batch_size: int = 32
    seq_len: int = 64
    lr: float = 0.0005
    dropout: float = 0.3
    weight_decay: float = 0.1


# 全局状态
model_manager = ModelManager()
training_worker = None
websocket_connections = []
main_event_loop = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    print("Starting Shannon-b1 Web Server...")
    global main_event_loop
    try:
        main_event_loop = asyncio.get_event_loop()
    except Exception:
        main_event_loop = None
    yield
    print("Shutting down Shannon-b1 Web Server...")
    if training_worker:
        training_worker.stop()


app = FastAPI(lifespan=lifespan, title="Shannon-b1 API", version="1.0.0")

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# WebSocket 连接管理
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass


manager = ConnectionManager()


def broadcast_training_update(data: dict):
    """广播训练更新"""
    # run-safe broadcast: if called from background thread, schedule on main event loop
    try:
        loop = asyncio.get_running_loop()
        # we're in event loop thread — safe to create task
        loop.create_task(manager.broadcast(data))
    except RuntimeError:
        # not in event loop (likely called from training thread) — use run_coroutine_threadsafe if main loop available
        if main_event_loop and main_event_loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(manager.broadcast(data), main_event_loop)
            except Exception:
                pass
        else:
            # fallback: attempt to create a new task (may fail if no loop)
            try:
                asyncio.create_task(manager.broadcast(data))
            except Exception:
                pass


# API 路由
@app.get("/api/status")
async def get_status():
    """获取服务器状态"""
    return {
        "status": "running",
        "model_loaded": model_manager.is_loaded(),
        "model_info": model_manager.get_info(),
        "training_active": training_worker is not None and training_worker.is_running,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/model/info")
async def get_model_info():
    """获取模型信息"""
    return model_manager.get_info()


@app.post("/api/model/load")
async def load_model(model_path: str = "../../checkpoints/shannon_b1.pt"):
    """加载模型"""
    try:
        success = model_manager.load_model(model_path)
        if success:
            return {"success": True, "message": f"Model loaded from {model_path}", "info": model_manager.get_info()}
        else:
            raise HTTPException(status_code=404, detail=f"Model not found at {model_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate")
async def generate(request: GenerateRequest):
    """生成文本"""
    if not model_manager.is_loaded():
        raise HTTPException(status_code=400, detail="No model loaded. Please load a model first.")
    
    try:
        result = model_manager.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty
        )
        return {"success": True, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/train/start")
async def start_training(request: TrainRequest, background_tasks: BackgroundTasks):
    """开始训练"""
    global training_worker
    
    if training_worker and training_worker.is_running:
        raise HTTPException(status_code=400, detail="Training already in progress")
    
    training_worker = TrainingWorker(
        config=request.dict(),
        callback=broadcast_training_update
    )
    
    background_tasks.add_task(training_worker.run)
    
    return {"success": True, "message": "Training started"}


@app.post("/api/train/stop")
async def stop_training():
    """停止训练"""
    global training_worker
    
    if training_worker:
        training_worker.stop()
        return {"success": True, "message": "Training stop requested"}
    return {"success": False, "message": "No active training"}


@app.get("/api/train/status")
async def get_training_status():
    """获取训练状态"""
    if training_worker:
        return training_worker.get_status()
    return {"is_running": False, "progress": 0, "current_loss": None}


@app.get("/api/checkpoints")
async def list_checkpoints():
    """列出所有检查点"""
    import glob
    checkpoints = []
    for path in glob.glob("checkpoints/*.pt"):
        name = os.path.basename(path)
        size = os.path.getsize(path) / 1024 / 1024
        mtime = os.path.getmtime(path)
        checkpoints.append({
            "name": name,
            "path": path,
            "size_mb": round(size, 2),
            "modified": datetime.fromtimestamp(mtime).isoformat()
        })
    return sorted(checkpoints, key=lambda x: x["modified"], reverse=True)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket 连接用于实时监控"""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # 处理客户端消息
            if data == "ping":
                await websocket.send_json({"type": "pong", "timestamp": datetime.now().isoformat()})
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.get("/")
async def root():
    return {"message": "Shannon-b1 API Server", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)