from .dataset import TextDataset
from .tokenizer import CharTokenizer, BPETokenizer, SimpleBPETokenizer
from .download import download_shakespeare, load_shakespeare, create_sample_data

def create_tokenizer(text, tokenizer_type='char', vocab_size=1000):
    """
    根据指定的类型创建并训练分词器
    
    Args:
        text (str): 用于训练分词器的文本数据
        tokenizer_type (str, optional): 分词器类型，可选值包括 'char'、'bpe'、'simple_bpe'。默认为 'char'
        vocab_size (int, optional): 词汇表大小，默认为 1000
    
    Returns:
        BaseTokenizer: 训练好的分词器实例，具体类型取决于 tokenizer_type 参数
    """
    # 根据分词器类型创建相应的分词器实例并进行训练或构建词汇表
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

# 定义模块公开接口列表，控制从该模块导入时可以访问的对象
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
