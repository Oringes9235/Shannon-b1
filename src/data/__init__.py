from .dataset import TextDataset
from .tokenizer import CharTokenizer, BPETokenizer, SimpleBPETokenizer
from .download import download_shakespeare, load_shakespeare, create_sample_data

def create_tokenizer(text, tokenizer_type='char', vocab_size=1000):
    if tokenizer_type == 'bpe':
        tokenizer = BPETokenizer(vocab_size=vocab_size)
        tokenizer.train([text], verbose=True)
    elif tokenizer_type == 'simple_bpe':
        tokenizer = SimpleBPETokenizer(vocab_size=vocab_size)
        tokenizer.build_vocab([text])
    else:
        tokenizer = CharTokenizer()
        tokenizer.build_vocab([text], vocab_size)
    return tokenizer

__all__ = [
    'TextDataset',
    'CharTokenizer',
    'BPETokenizer',
    'SimpleBPETokenizer',
    'download_shakespeare',
    'load_shakespeare',
    'create_sample_data',
    'create_tokenizer'
]
