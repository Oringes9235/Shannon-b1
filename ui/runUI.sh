#!/bin/bash

# 启动后端
echo "🚀 Starting backend server..."
cd server
pip install -r requirements.txt
python app.py &
BACKEND_PID=$!

# 等待后端启动
sleep 3

# 启动前端
echo "🎨 Starting frontend..."
cd ../client
npm install
npm run dev &
FRONTEND_PID=$!

echo ""
echo "=========================================="
echo "✅ Shannon-b1 Web UI 已启动"
echo "   前端: http://localhost:5173"
echo "   后端: http://localhost:8000"
echo "   API 文档: http://localhost:8000/docs"
echo "=========================================="
echo ""
echo "按 Ctrl+C 停止服务"

# 等待退出
wait $BACKEND_PID $FRONTEND_PID