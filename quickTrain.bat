@echo off

python scripts/train.py --epochs 100 --batch-size 24 --seq-len 128 --d-model 256 --num-layers 16 --d-ff 1024 --lr 1e-3 --warmup-steps 2000 --label-smoothing 0.1 --tie-embeddings --patience 999 --grad-accum 4 --gradient-checkpointing --norm-type rmsnorm --dropout 0.1 --vocab-size 10000 --save-path checkpoints/shannon_b1_stronger.pt --seed 42 --resume checkpoints/shannon_b1.pt --device cuda

pause