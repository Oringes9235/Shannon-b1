#!/usr/bin/env bash
set -euo pipefail

# experiments/compare.sh
# 运行三组对比实验并把日志与检查点分别保存到 experiments/results

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$ROOT_DIR/logs"
mkdir -p "$ROOT_DIR/results"

run_exp(){
  name="$1"
  shift
  outdir="$ROOT_DIR/results/$name"
  mkdir -p "$outdir"
  logfile="$ROOT_DIR/logs/$name.log"

  echo "== Running experiment: $name =="
  python scripts/train.py "$@" --save-path "$outdir/${name}.pt" > "$logfile" 2>&1 || true

  # trainer may also save best/epoch checkpoints under checkpoints/ - move any recent files
  if [ -d checkpoints ]; then
    mv -f checkpoints/*.pt "$outdir/" 2>/dev/null || true
  fi

  # copy tokenizer if produced
  if [ -f "${outdir}/${name}_tokenizer.json" ]; then
    echo "tokenizer saved"
  fi
  echo "== Finished: $name (logs -> $logfile, results -> $outdir) =="
}

# 1) 小规模（快速）
run_exp exp_small --epochs 10 --batch-size 8 --seq-len 128 --d-model 128 --num-layers 4 \
  --lr 5e-4 --warmup-steps 1000 --tokenizer bpe --vocab-size 2000 --grad-accum 1

# 2) 中等（默认对比）
run_exp exp_medium --epochs 50 --batch-size 12 --seq-len 256 --d-model 256 --num-layers 6 \
  --lr 5e-4 --warmup-steps 2000 --tokenizer bpe --vocab-size 5000 --grad-accum 2 --gradient-checkpointing

# 3) 高质量（更长训练）
run_exp exp_large --epochs 200 --batch-size 16 --seq-len 512 --d-model 512 --num-layers 12 \
  --lr 3e-4 --warmup-steps 4000 --tokenizer bpe --vocab-size 10000 --grad-accum 4

echo "All experiments finished. Results are under $ROOT_DIR/results and logs under $ROOT_DIR/logs"
