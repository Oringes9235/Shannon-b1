"""
数据下载工具
"""

import os
import urllib.request


def download_shakespeare(save_path: str = 'data/shakespeare.txt') -> str:
    """
    下载莎士比亚文本
    
    Args:
        save_path (str): 保存路径，默认为'data/shakespeare.txt'
        
    Returns:
        str: 成功时返回保存路径，失败时返回None
    """
    # 创建目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    
    try:
        urllib.request.urlretrieve(url, save_path)
        with open(save_path, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"✅ Downloaded Shakespeare: {len(text):,} chars")
        return save_path
    except Exception as e:
        print(f"⚠️ Download failed: {e}")
        return None


def load_shakespeare() -> str:
    """
    加载莎士比亚文本

    优先使用本地文件 data/shakespeare.txt（如果存在），否则尝试下载；
    下载失败时使用内置的备用文本。
    
    Returns:
        str: 莎士比亚文本内容
    """
    local_path = 'data/shakespeare.txt'
    if os.path.exists(local_path):
        with open(local_path, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"✅ Loaded local Shakespeare: {len(text):,} chars from {local_path}")
        return text

    path = download_shakespeare()
    if path:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    print("⚠️ Using fallback sample text (download failed and no local file found).")
    return "To be or not to be, that is the question. " * 1000


def load_all_data(data_dir: str = 'data') -> list:
    """
    递归加载 data 文件夹下所有 .txt 文件的内容

    Args:
        data_dir: 数据文件夹路径，默认 'data'

    Returns:
        list[str]: 所有文本内容组成的列表，如果目录为空或无 .txt 则返回单个 fallback 文本
    """
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
                            print(f"✅ Loaded: {fpath} ({len(content):,} chars)")
                    except Exception as e:
                        print(f"⚠️  Skip {fpath}: {e}")

    if not texts:
        print("⚠️  No .txt files found in data/, using fallback text")
        sample_path = create_sample_data()
        with open(sample_path, 'r', encoding='utf-8') as f:
            texts.append(f.read())

    total = sum(len(t) for t in texts)
    print(f"\n📚 Total: {len(texts)} files, {total:,} chars")
    return texts


def create_sample_data(save_path: str = 'data/sample.txt') -> str:
    """
    创建示例数据
    
    Args:
        save_path (str): 保存路径，默认为'data/sample.txt'
        
    Returns:
        str: 保存路径
    """
    # 创建目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    sample_text = """Once upon a time, there was a little girl named Alice. She lived in a small village
at the foot of a great mountain. Every day, she would look up at the mountain and
wonder what was at the top. One morning, she decided to find out. She packed a small
bag with some bread and cheese, and started climbing. The path was steep and rocky,
but Alice was determined. After many hours, she reached the top. There, she found a
beautiful garden full of flowers she had never seen before. In the middle of the
garden stood a small cottage. An old woman came out and smiled at Alice. "Welcome,"
she said, "I have been waiting for you." And so began Alice's greatest adventure."""
    
    # 写入示例文本到文件
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(sample_text)
    
    print(f"✅ Sample data created: {save_path}")
    return save_path