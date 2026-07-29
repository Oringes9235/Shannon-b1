from .dataset import TextDataset
from .tokenizer import CharTokenizer, BPETokenizer, SimpleBPETokenizer
from .download import download_shakespeare, load_shakespeare, create_sample_data, load_all_data, load_data_chunks


def create_tokenizer(text_or_texts, tokenizer_type='char', vocab_size=1000):
    """
    根据指定的类型创建并训练分词器
    
    Args:
        text_or_texts: 用于训练的文本 (str/list) 或 可迭代对象
        tokenizer_type: 'char', 'bpe', 'simple_bpe'
        vocab_size: 词汇表大小
    """
    if tokenizer_type == 'bpe':
        tokenizer = BPETokenizer(vocab_size=vocab_size)
        # 支持 list 或 generator
        if isinstance(text_or_texts, list):
            tokenizer.train(text_or_texts, verbose=True)
        else:
            # 单字符串
            tokenizer.train([text_or_texts], verbose=True)
    elif tokenizer_type == 'simple_bpe':
        tokenizer = SimpleBPETokenizer(vocab_size=vocab_size)
        texts = text_or_texts if isinstance(text_or_texts, list) else [text_or_texts]
        tokenizer.build_vocab(texts)
    else:
        tokenizer = CharTokenizer()
        texts = text_or_texts if isinstance(text_or_texts, list) else [text_or_texts]
        tokenizer.build_vocab(texts, vocab_size)
    return tokenizer


def create_tokenizer_streaming(tokenizer_type='bpe', vocab_size=10000, data_dir='data', chunk_size=1_000_000):
    """
    使用分块流式训练 BPE tokenizer，避免内存溢出
    
    Args:
        tokenizer_type: 'bpe' (目前仅支持 bpe)
        vocab_size: 词汇表大小
        data_dir: 数据目录
        chunk_size: 每块字符数
    """
    if tokenizer_type != 'bpe':
        raise ValueError("流式训练目前仅支持 BPE tokenizer")
    
    tokenizer = BPETokenizer(vocab_size=vocab_size)
    chunks = load_data_chunks(data_dir, chunk_size=chunk_size)
    tokenizer.train(chunks, verbose=True)
    return tokenizer


__all__ = [
    'TextDataset',
    'CharTokenizer',
    'BPETokenizer',
    'SimpleBPETokenizer',
    'download_shakespeare',
    'load_shakespeare',
    'create_sample_data',
    'load_all_data',
    'load_data_chunks',
    'create_tokenizer',
    'create_tokenizer_streaming',
]