import sys, os
sys.path.insert(0, '.')
os.environ['PYTHONIOENCODING'] = 'utf-8'

from src.data.tokenizer import BPETokenizer
from src.data import load_shakespeare

text = load_shakespeare()
with open('tests/verify_bpe_output.txt', 'w', encoding='utf-8') as f:
    f.write(f'Text: {len(text)} chars\n')
    
    t = BPETokenizer(vocab_size=5000)
    t.train([text], verbose=True)
    
    f.write(f'Final vocab: {t.get_vocab_size()}\n')
    f.write(f'Special tokens: {t.special_tokens}\n')
    
    tok = t.encode('To be or not to be')
    f.write(f'Tokens: {tok}\n')
    f.write(f'Decoded: {repr(t.decode(tok))}\n')
    
    # Save and reload
    t.save('tests/verify_bpe_temp.json')
    t2 = BPETokenizer()
    t2.load('tests/verify_bpe_temp.json')
    f.write(f'Reloaded vocab: {t2.get_vocab_size()}\n')
    assert t2.get_vocab_size() == t.get_vocab_size()
    f.write('Roundtrip OK\n')
    
    os.unlink('tests/verify_bpe_temp.json')
    f.write('ALL TESTS PASSED\n')