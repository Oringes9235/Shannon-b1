#!/bin/bash
echo "========================================"
echo "Shannon-b1 训练启动"
echo "========================================"

# 使用莎士比亚文本训练
python train.py --dataset shakespeare --epochs 30 --batch-size 16 --seq-len 64

# 或使用示例数据
# python train.py --dataset sample --epochs 20 --batch-size 16 --seq-len 64