import json
import sys
import torch
sys.path.insert(0, '.')
from src.data import BPETokenizer, load_shakespeare, CharTokenizer

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

# 加载莎士比亚文本
text = load_shakespeare()
print(f"训练文本长度: {len(text)} 字符")

# 创建并训练 tokenizer（与模型匹配的 vocab_size）
if expected_vocab_size and expected_vocab_size > 100:
    # 大于 100 用 BPE
    print(f"训练 BPE tokenizer (target {expected_vocab_size})...")
    tokenizer = BPETokenizer(vocab_size=expected_vocab_size)
    tokenizer.train([text], verbose=True)
else:
    # 小词汇表用字符级
    print(f"训练 CharTokenizer (target {expected_vocab_size or 1000})...")
    tokenizer = CharTokenizer()
    tokenizer.build_vocab([text], expected_vocab_size or 1000)

# 保存
tokenizer.save(OUTPUT_PATH)
print(f"\n✅ 分词器已保存: {OUTPUT_PATH}")
print(f"   vocab_size={tokenizer.get_vocab_size()}")
print(f"   special_tokens={tokenizer.special_tokens}")
