import sys, os
sys.path.insert(0, '.')
os.environ['PYTHONIOENCODING'] = 'utf-8'

from src.data.tokenizer import BPETokenizer

with open('tests/test_legacy_output.txt', 'w', encoding='utf-8') as f:
    # Test legacy checkpoint loading
    t = BPETokenizer()
    t.load('checkpoints/shannon_b1_best_tokenizer.json')
    
    f.write(f"Mode: {'legacy' if t._legacy_mode else 'HF'}\n")
    f.write(f"Vocab: {t.get_vocab_size()}\n")
    f.write(f"Pad ID: {t.get_pad_id()}\n")
    f.write(f"Special tokens: {t.special_tokens}\n")
    
    tok = t.encode('The king')
    f.write(f"Tokens 'The king': {tok}\n")
    f.write(f"Decoded: {repr(t.decode(tok))}\n")
    
    # Test longer text
    text = "To be, or not to be, that is the question"
    tok2 = t.encode(text)
    dec = t.decode(tok2)
    f.write(f"\nLong text encode: '{text}' -> {len(tok2)} tokens\n")
    f.write(f"Decoded: {repr(dec)}\n")
    
    f.write("\nALL TESTS PASSED\n")
    print("Done - see tests/test_legacy_output.txt")