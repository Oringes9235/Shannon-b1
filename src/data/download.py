"""
数据下载工具
"""

import os
import urllib.request
import gzip
import shutil


def download_shakespeare(save_path='data/shakespeare.txt'):
    """下载莎士比亚全集"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    
    try:
        urllib.request.urlretrieve(url, save_path)
        print(f"✅ 莎士比亚文本已下载到: {save_path}")
        
        # 统计信息
        with open(save_path, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"   字符数: {len(text):,}")
        print(f"   行数: {text.count(chr(10)):,}")
        
        return save_path
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return None


def download_tiny_stories(save_path='data/tiny_stories.txt'):
    """下载 TinyStories 数据集 (小故事，适合训练)"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # TinyStories 是一个小型故事数据集
    url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
    
    print("⚠️ TinyStories 数据集较大，建议手动下载")
    print(f"   下载地址: {url}")
    return None


def create_sample_data(save_path='data/sample.txt'):
    """创建示例训练数据"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    sample_text = """
    Once upon a time, there was a little girl named Alice. She lived in a small village
    at the foot of a great mountain. Every day, she would look up at the mountain and
    wonder what was at the top. One morning, she decided to find out. She packed a small
    bag with some bread and cheese, and started climbing. The path was steep and rocky,
    but Alice was determined. After many hours, she reached the top. There, she found a
    beautiful garden full of flowers she had never seen before. In the middle of the
    garden stood a small cottage. An old woman came out and smiled at Alice. "Welcome,"
    she said, "I have been waiting for you." And so began Alice's greatest adventure.
    
    The sun was setting over the ocean, painting the sky in shades of orange and pink.
    Sarah sat on the beach, watching the waves roll in. She had always loved the sea,
    with its mysterious depths and endless horizons. Today was special - it was the
    first time she had seen a dolphin in the wild. The creature leaped out of the water,
    its sleek body glistening in the golden light. Sarah smiled, feeling a deep
    connection to this intelligent creature. As the dolphin disappeared beneath the
    waves, she made a wish on the first star that appeared in the sky.
    
    In a laboratory far beneath the city, Dr. Chen was working on her greatest invention.
    The Quantum Processor hummed quietly, its surface glowing with an otherworldly blue
    light. Years of research had led to this moment. She took a deep breath and pressed
    the activation button. The machine whirred to life, and a holographic display
    appeared in the air. "It works," she whispered, tears forming in her eyes. The
    processor had just solved a problem that would have taken traditional computers
    a thousand years. The future had arrived.
    """
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(sample_text)
    
    print(f"✅ 示例数据已创建: {save_path}")
    return save_path


def prepare_data(args):
    """准备训练数据"""
    data_path = None
    
    if args.dataset == 'shakespeare':
        data_path = download_shakespeare()
    elif args.dataset == 'sample':
        data_path = create_sample_data()
    elif args.dataset == 'file' and args.data_path:
        if os.path.exists(args.data_path):
            data_path = args.data_path
        else:
            print(f"❌ 文件不存在: {args.data_path}")
            return None
    
    return data_path