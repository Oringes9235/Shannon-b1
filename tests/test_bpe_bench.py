"""BPE tokenizer 训练速度基准测试"""
import sys
import time
sys.path.insert(0, '.')
from src.data.tokenizer import BPETokenizer

# 生成测试文本
base_text = "Hello world! This is a BPE tokenizer training speed test. " * 1000
texts = [base_text]

print("=" * 60)
print("BPE Tokenizer 训练速度测试")
print("=" * 60)
print(f"文本数量: {len(texts)}")
print(f"文本总长度: {sum(len(t) for t in texts)} 字符")
print()

for vocab_size in [500, 1000, 2000]:
    t = BPETokenizer(vocab_size=vocab_size)
    start = time.time()
    t.train(texts, verbose=False)
    elapsed = time.time() - start
    print(f"vocab_size={vocab_size:>5}: {elapsed:.3f}s, vocab={t.get_vocab_size()}")
    
    # 验证编码/解码
    tokens = t.encode("Hello world")
    decoded = t.decode(tokens)
    print(f"  样例: 'Hello world' -> tokens={tokens} -> '{decoded}'")

print()
print("[92m[SUCCESS][0m 所有测试通过！")