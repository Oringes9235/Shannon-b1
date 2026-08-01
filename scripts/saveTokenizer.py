import json
import sys
import os
import torch
sys.path.insert(0, '.')
from src.data import BPETokenizer, CharTokenizer, load_all_data, load_data_chunks, create_tokenizer_streaming

# 读取 checkpoint 获取模型期望的 vocab_size
MODEL_PATH = 'checkpoints/shannon_b1_best.pt'
OUTPUT_PATH = MODEL_PATH.replace('.pt', '_tokenizer.json')

ckpt = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
expected_vocab_size = None
if 'config' in ckpt:
    expected_vocab_size = ckpt['config'].vocab_size
elif 'model_config' in ckpt:
    expected_vocab_size = ckpt['model_config'].vocab_size
print(f"模型期望 vocab_size: {expected_vocab_size}")

# 加载与训练时相同的数据（data/ 目录下所有 .txt 文件）
texts = load_all_data('data')
total_chars = sum(len(t) for t in texts)
print(f"训练文本总长度: {total_chars:,} 字符")

if expected_vocab_size and expected_vocab_size > 100:
    # BPE tokenizer — 大数据集用流式分块训练，小数据集直接训练
    print(f"训练 BPE tokenizer (target {expected_vocab_size})...")
    if total_chars > 50_000_000:
        # 超过 50MB 用流式训练，避免内存溢出
        tokenizer = create_tokenizer_streaming(
            tokenizer_type='bpe',
            vocab_size=expected_vocab_size,
            data_dir='data',
            chunk_size=1_000_000,
        )
    else:
        combined_text = "\n\n".join(texts)
        tokenizer = BPETokenizer(vocab_size=expected_vocab_size)
        tokenizer.train([combined_text], verbose=True)
else:
    # 小词汇表用字符级，保存为 BPETokenizer 兼容的 HF 格式
    print(f"训练 CharTokenizer (target {expected_vocab_size or 1000})...")
    tokenizer = CharTokenizer()
    combined_text = "\n\n".join(texts) if texts else "sample text"
    tokenizer.build_vocab([combined_text], expected_vocab_size or 1000)

# 保存
tokenizer.save(OUTPUT_PATH)
print(f"\n✅ 分词器已保存: {OUTPUT_PATH}")
print(f"   vocab_size={tokenizer.get_vocab_size()}")
print(f"   special_tokens={tokenizer.special_tokens}")
