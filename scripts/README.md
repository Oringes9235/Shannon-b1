## train.py
训练脚本
```powershell
python scripts/train.py `
  --tokenizer bpe `
  --vocab-size 10000 `
  --d-model 192 `
  --num-layers 4 `
  --d-ff 768 `
  --seq-len 96 `
  --batch-size 2 `
  --grad-accum 16 `
  --epochs 50 `
  --lr 0.0001 `
  --warmup-steps 1000 `
  --label-smoothing 0.1 `
  --tie-embeddings `
  --patience 10 `
  --gradient-checkpointing `
  --norm-type rmsnorm `
  --dropout 0.1 `
  --save-path checkpoints/shannon_b1_best.pt `
  --seed 42 `
  --device cuda
```

## generate.py
生成脚本

- 单次模型生成：
```powershell
python scripts/generate.py `
  --model-path checkpoints/shannon_b1_v2.pt `
  --prompt "The " `
  --max-new-tokens 150 `
  --temperature 0.8 `
  --top-k 50 `
  --device cuda
```

- 交互式生成：
```powershell
python scripts/generate.py `
  --model-path checkpoints/shannon_b1_v2.pt `
  -i `
  --system-prompt "You are a helpful assistant." `
  --max-new-tokens 200 `
  --temperature 0.8 `
  --top-k 50 `
  --device cuda
```

## evaluate.py
评估脚本

## countLines.py
统计代码行数脚本

```powershell
python scripts/countLines.py
```

## saveTokenizer.py
手动保存词表脚本

```powershell
python scripts/saveTokenizer.py
```


## merge_txt.py
- txt文本转码为UTF-8并整合脚本

- 注：请将所有txt文件放在同目录下的txt文件夹内

```powershell
python scripts/merge_txt.py
```

## saveTrain_Loss.py
保存训练集Loss曲线脚本

```powershell
python scripts/saveTrain_Loss.py
```

