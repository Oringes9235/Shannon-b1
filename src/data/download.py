"""
数据下载工具
"""

import os
import urllib.request


def download_shakespeare(save_path: str = 'data/shakespeare.txt') -> str:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    
    try:
        urllib.request.urlretrieve(url, save_path)
        with open(save_path, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"[92m[SUCCESS][0m Downloaded Shakespeare: {len(text):,} chars")
        return save_path
    except Exception as e:
        print(f"[93m[WARNING][0m Download failed: {e}")
        return None


def load_shakespeare() -> str:
    local_path = 'data/shakespeare.txt'
    if os.path.exists(local_path):
        with open(local_path, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"[92m[SUCCESS][0m Loaded local Shakespeare: {len(text):,} chars from {local_path}")
        return text

    path = download_shakespeare()
    if path:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    print("[93m[WARNING][0m Using fallback sample text (download failed and no local file found).")
    return "To be or not to be, that is the question. " * 1000


def load_all_data(data_dir: str = 'data') -> list:
    texts = []
    if os.path.isdir(data_dir):
        for root, _, files in os.walk(data_dir):
            for fname in files:
                if fname.lower().endswith('.txt'):
                    fpath = os.path.join(root, fname)
                    try:
                        with open(fpath, 'r', encoding='utf-8') as f:
                            content = f.read()
                        if content.strip():
                            texts.append(content)
                            print(f"[92m[SUCCESS][0m Loaded: {fpath} ({len(content):,} chars)")
                    except Exception as e:
                        print(f"[93m[WARNING][0m  Skip {fpath}: {e}")

    if not texts:
        print("[93m[WARNING][0m  No .txt files found in data/, using fallback text")
        sample_path = create_sample_data()
        with open(sample_path, 'r', encoding='utf-8') as f:
            texts.append(f.read())

    total = sum(len(t) for t in texts)
    print(f"\n[96m[LOAD][0m Total: {len(texts)} files, {total:,} chars")
    return texts


def load_data_chunks(data_dir: str = 'data', chunk_size: int = 1_000_000):
    """
    分块读取 data 目录下所有 .txt 文件，返回生成器。
    适用于 BPE tokenizer 流式训练，避免一次性加载全部数据到内存。
    
    Args:
        data_dir: 数据文件夹路径
        chunk_size: 每块字符数，默认 1M
    
    Yields:
        str: 文本块
    """
    count = 0
    if not os.path.isdir(data_dir):
        return
    
    for root, _, files in os.walk(data_dir):
        for fname in sorted(files):
            if fname.lower().endswith('.txt'):
                fpath = os.path.join(root, fname)
                total = os.path.getsize(fpath)
                print(f"[96m[READ][0m Streaming: {fpath} ({total:,} bytes)")
                with open(fpath, 'r', encoding='utf-8') as f:
                    while True:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        count += 1
                        yield chunk
    print(f"[92m[SUCCESS][0m Yielded {count} chunks for training")


def create_sample_data(save_path: str = 'data/sample.txt') -> str:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    sample_text = """Once upon a time, there was a little girl named Alice. She lived in a small village
at the foot of a great mountain. Every day, she would look up at the mountain and
wonder what was at the top. One morning, she decided to find out. She packed a small
bag with some bread and cheese, and started climbing. The path was steep and rocky,
but Alice was determined. After many hours, she reached the top. There, she found a
beautiful garden full of flowers she had never seen before. In the middle of the
garden stood a small cottage. An old woman came out and smiled at Alice. "Welcome,"
she said, "I have been waiting for you." And so began Alice's greatest adventure."""
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(sample_text)
    
    print(f"[92m[SUCCESS][0m Sample data created: {save_path}")
    return save_path