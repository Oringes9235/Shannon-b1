"""
KV Cache 性能测试脚本
对比使用和不使用KV Cache的生成速度
"""

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import time
from src.model.config import ModelConfig
from src.model.shannon import ShannonB1


def test_kv_cache_performance():
    """测试KV Cache的性能提升"""
    
    # 创建一个小模型用于测试
    config = ModelConfig(
        vocab_size=500,
        d_model=128,
        num_layers=4,
        num_heads=8,
        d_ff=512,
        max_seq_len=256,
        dropout=0.1,
    )
    
    model = ShannonB1(config)
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"设备: {device}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 80)
    
    # 测试参数
    prompt_tokens = list(range(10))  # 模拟10个token的prompt
    max_new_tokens = 50
    
    # 测试1: 不使用KV Cache
    print("\n[94m[INFO][0m 测试1: 不使用 KV Cache")
    start_time = time.time()
    with torch.no_grad():
        result_no_cache = model.generate(
            start_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            use_kv_cache=False,
        )
    end_time = time.time()
    time_no_cache = end_time - start_time
    speed_no_cache = max_new_tokens / time_no_cache
    
    print(f"   生成token数: {max_new_tokens}")
    print(f"   耗时: {time_no_cache:.3f} 秒")
    print(f"   速度: {speed_no_cache:.2f} tokens/秒")
    
    # 测试2: 使用KV Cache
    print("\n[94m[INFO][0m 测试2: 使用 KV Cache")
    start_time = time.time()
    with torch.no_grad():
        result_with_cache = model.generate(
            start_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            use_kv_cache=True,
        )
    end_time = time.time()
    time_with_cache = end_time - start_time
    speed_with_cache = max_new_tokens / time_with_cache
    
    print(f"   生成token数: {max_new_tokens}")
    print(f"   耗时: {time_with_cache:.3f} 秒")
    print(f"   速度: {speed_with_cache:.2f} tokens/秒")
    
    # 计算加速比
    speedup = time_no_cache / time_with_cache
    improvement = ((time_no_cache - time_with_cache) / time_no_cache) * 100
    
    print("\n" + "=" * 80)
    print("[94m[INFO][0m 性能对比结果:")
    print("=" * 80)
    print(f"   无KV Cache: {time_no_cache:.3f}s ({speed_no_cache:.2f} tok/s)")
    print(f"   有KV Cache: {time_with_cache:.3f}s ({speed_with_cache:.2f} tok/s)")
    print(f"   加速比:     {speedup:.2f}x")
    print(f"   性能提升:   {improvement:.1f}%")
    print("=" * 80)
    
    # 验证结果一致性（由于采样的随机性，结果可能不同）
    print("\n[92m[SUCCESS][0m 注意: 由于采样随机性，两次生成的token序列可能不同")
    print(f"   这是正常现象，不影响性能测试结果")
    
    return {
        'time_no_cache': time_no_cache,
        'time_with_cache': time_with_cache,
        'speedup': speedup,
        'improvement_percent': improvement,
    }


def test_streaming_with_kv_cache():
    """测试流式生成与KV Cache的结合"""
    
    print("\n\n" + "=" * 80)
    print("[94m[LOADING][0m 测试流式生成 + KV Cache")
    print("=" * 80)
    
    config = ModelConfig(
        vocab_size=500,
        d_model=128,
        num_layers=4,
        num_heads=8,
        d_ff=512,
        max_seq_len=256,
        dropout=0.1,
    )
    
    model = ShannonB1(config)
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    prompt_tokens = list(range(10))
    max_new_tokens = 20
    
    print(f"\n开始流式生成 (use_kv_cache=True)...")
    start_time = time.time()
    
    token_count = 0
    with torch.no_grad():
        for token_id, prob in model.generate_stream(
            start_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            use_kv_cache=True,
        ):
            token_count += 1
            if token_count <= 5:  # 只显示前5个token
                print(f"   Token {token_count}: ID={token_id}, Prob={prob:.4f}")
    
    end_time = time.time()
    elapsed = end_time - start_time
    speed = token_count / elapsed
    
    print(f"\n   总token数: {token_count}")
    print(f"   总耗时: {elapsed:.3f} 秒")
    print(f"   平均速度: {speed:.2f} tokens/秒")
    print("=" * 80)


if __name__ == '__main__':
    print("[95m[START][0m Shannon-b1 KV Cache 性能测试")
    print("=" * 80)
    
    # 运行性能测试
    results = test_kv_cache_performance()
    
    # 运行流式生成测试
    test_streaming_with_kv_cache()
    
    print("\n[92m[SUCCESS][0m 所有测试完成!")
