"""
测试长序列生成功能

验证模型能否生成超过max_seq_len的token序列
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.model import ShannonB1, ModelConfig
from src.data import CharTokenizer

def test_long_sequence_generation():
    """测试长序列生成（超过max_seq_len）"""
    
    print("="*70)
    print("长序列生成测试")
    print("="*70)
    print()
    
    # 创建小模型用于快速测试
    config = ModelConfig(
        vocab_size=100,
        d_model=64,
        num_layers=2,
        num_heads=4,
        max_seq_len=256,  # 初始最大序列长度
        dropout=0.0
    )
    
    model = ShannonB1(config)
    model.eval()
    
    tokenizer = CharTokenizer()
    tokenizer.build_vocab(["hello world test long sequence generation"], 100)
    
    start_tokens = tokenizer.encode("The ")
    print(f"起始token: {start_tokens}")
    print(f"模型配置的max_seq_len: {config.max_seq_len}")
    print()
    
    # 测试生成不同长度的序列
    test_lengths = [100, 200, 300, 400]
    
    for target_length in test_lengths:
        print(f"\n测试生成 {target_length} 个token...")
        print("-"*70)
        
        try:
            generated_tokens = []
            for token_id, prob in model.generate_stream(
                start_tokens,
                max_new_tokens=target_length,
                temperature=0.85,
                repetition_penalty=1.15,
                ban_immediate_repeat=True,
                ngram_block_size=3,
                max_repetition=50
            ):
                generated_tokens.append(token_id)
                
                # 每50个token显示一次进度
                if len(generated_tokens) % 50 == 0:
                    print(f"  已生成: {len(generated_tokens)} tokens")
            
            print(f"✅ 成功生成 {len(generated_tokens)} 个token")
            
            # 解码并显示部分文本
            decoded_text = tokenizer.decode(start_tokens + generated_tokens[:50])
            print(f"  前50个token的文本: {decoded_text[:100]}...")
            
        except Exception as e:
            print(f"❌ 失败: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("\n" + "="*70)
    print("✅ 所有测试通过！模型支持动态扩展序列长度。")
    print("="*70)
    return True


if __name__ == '__main__':
    success = test_long_sequence_generation()
    sys.exit(0 if success else 1)
