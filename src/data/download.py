"""
数据下载工具
"""

import os
import urllib.request


def download_shakespeare(save_path: str = 'data/shakespeare.txt') -> str:
    """下载莎士比亚文本"""
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
    """加载莎士比亚文本

    优先使用本地文件 data/shakespeare.txt（如果存在），否则尝试下载；
    下载失败时使用内置的备用文本。
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


def create_sample_data(save_path: str = 'data/sample.txt') -> str:
    """创建示例数据"""
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
    
    print(f"✅ Sample data created: {save_path}")
    return save_path