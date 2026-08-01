"""验证 Rust BPE tokenizer 正确性"""
import sys
sys.path.insert(0, '.')

from src.data.tokenizer import BPETokenizer
from src.data import load_shakespeare

print("=" * 60)
print("Rust BPE Tokenizer 验证")
print("=" * 60)

# Test 1: Basic training
print("\n[Test 1] 小规模训练 (vocab=1000)...")
t = BPETokenizer(vocab_size=1000)
t.train(['Hello world! This is a BPE tokenizer test. ' * 300], verbose=False)
print(f"  Vocab size: {t.get_vocab_size()}")
assert t.get_vocab_size() >= 50, f"Vocab too small: {t.get_vocab_size()}"
print(f"  Pad ID: {t.get_pad_id()}")

# Test 2: Encode/Decode
print("\n[Test 2] 编码/解码...")
tokens = t.encode('Hello world')
decoded = t.decode(tokens)
print(f"  'Hello world' -> {tokens} -> '{decoded}'")
assert len(tokens) > 0

# Test 3: Shakespeare data
print("\n[Test 3] Shakespeare 文本训练 (vocab=2000)...")
text = load_shakespeare()
print(f"  Text length: {len(text)} chars")
t2 = BPETokenizer(vocab_size=2000)
t2.train([text], verbose=False)
print(f"  Vocab size: {t2.get_vocab_size()}")
assert t2.get_vocab_size() >= 100, f"Vocab didn't grow: {t2.get_vocab_size()}"

# Test 4: Save/Load
print("\n[Test 4] 保存/加载...")
import tempfile, os
with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
    path = f.name
try:
    t2.save(path)
    t3 = BPETokenizer()
    t3.load(path)
    print(f"  Loaded vocab size: {t3.get_vocab_size()}")
    assert t3.get_vocab_size() == t2.get_vocab_size()
    # Verify encode produces same results
    test_text = "To be or not to be"
    tok2 = t2.encode(test_text)
    tok3 = t3.encode(test_text)
    assert tok2 == tok3, f"Mismatch: {tok2} != {tok3}"
    print(f"  Save/Load roundtrip OK")
finally:
    os.unlink(path)

# Test 5: Empty text
print("\n[Test 5] 空文本...")
assert t.encode('') == []
print(f"  Empty encode: []")

# Test 6: add_bos/add_eos
print("\n[Test 6] BOS/EOS 标记...")
tokens = t.encode('test', add_bos=True, add_eos=True)
bos_id = t.special_tokens['<BOS>']
eos_id = t.special_tokens['<EOS>']
assert tokens[0] == bos_id
assert tokens[-1] == eos_id
print(f"  OK: {tokens}")

# Test 7: Large vocab on Shakespeare
print("\n[Test 7] Shakespeare 大词汇量训练 (vocab=5000)...")
t4 = BPETokenizer(vocab_size=5000)
t4.train([text], verbose=True)
print(f"  Final vocab size: {t4.get_vocab_size()}")
assert t4.get_vocab_size() >= 1000, f"Vocab not growing: {t4.get_vocab_size()}"
print(f"  Special tokens: {t4.special_tokens}")

print("\n" + "=" * 60)
print("[92m[SUCCESS][0m 所有测试通过！")
print("=" * 60)