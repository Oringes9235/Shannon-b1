@echo off
echo ========================================
echo Shannon-b1 训练启动
echo ========================================

REM 使用莎士比亚文本训练
python train.py --dataset shakespeare --tokenizer bpe --vocab-size 2000 --epochs 50 --batch-size 32 --seq-len 64 --d-model 128 --num-layers 4

REM 从检查点继续训练
REM python train_improved.py --load-path shannon_b1_improved_best.pkl --epochs 200

pause