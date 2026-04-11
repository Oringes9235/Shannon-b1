"""
长上下文（1M+ tokens）优化测试脚本
测试RoPE、ALiBi和滑动窗口注意力的效果
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


def test_rope_extrapolation():
    """测试RoPE的外推能力"""
    
    print("=" * 80)
    print("🧪 测试1: RoPE 外推能力")
    print("=" * 80)
    
    # 创建使用RoPE的模型
    config = ModelConfig(
        vocab_size=500,
        d_model=128,
        num_layers=4,
        num_heads=8,
        d_ff=512,
        max_seq_len=1048576,  # 1M
        dropout=0.1,
        use_rope=True,
        rope_base=10000.0,
    )
    
    model = ShannonB1(config)
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"设备: {device}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"RoPE启用: {config.use_rope}")
    print(f"最大序列长度: {config.max_seq_len:,}")
    
    # 测试不同长度的序列
    test_lengths = [512, 1024, 2048, 4096]
    
    for seq_len in test_lengths:
        print(f"\n   测试序列长度: {seq_len}")
        
        # 创建随机输入
        tokens = torch.randint(0, config.vocab_size, (1, seq_len)).to(device)
        
        start_time = time.time()
        with torch.no_grad():
            logits, _ = model(tokens)
        end_time = time.time()
        
        elapsed = end_time - start_time
        memory_mb = torch.cuda.memory_allocated(device) / 1024 / 1024 if device.type == 'cuda' else 0
        
        print(f"      耗时: {elapsed:.3f}s")
        print(f"      显存占用: {memory_mb:.1f} MB")
        print(f"      输出形状: {logits.shape}")
    
    print("\n✅ RoPE外推测试完成")


def test_alibi_bias():
    """测试ALiBi偏置"""
    
    print("\n\n" + "=" * 80)
    print("🧪 测试2: ALiBi 线性注意力偏置")
    print("=" * 80)
    
    config = ModelConfig(
        vocab_size=500,
        d_model=128,
        num_layers=4,
        num_heads=8,
        d_ff=512,
        max_seq_len=1048576,
        dropout=0.1,
        use_rope=False,  # ALiBi通常不与RoPE同时使用
        use_alibi=True,
    )
    
    model = ShannonB1(config)
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"设备: {device}")
    print(f"ALiBi启用: {config.use_alibi}")
    
    # 测试ALiBi偏置矩阵
    seq_len = 128
    alibi_bias = model.alibi_bias(seq_len)
    
    print(f"\n   ALiBi偏置矩阵形状: {alibi_bias.shape}")
    print(f"   偏置范围: [{alibi_bias.min():.2f}, {alibi_bias.max():.2f}]")
    print(f"   偏置均值: {alibi_bias.mean():.4f}")
    
    # 可视化第一个头的偏置
    print(f"\n   第一个头的偏置矩阵（前8x8）:")
    print(alibi_bias[0, :8, :8])
    
    print("\n✅ ALiBi测试完成")


def test_sliding_window():
    """测试滑动窗口注意力"""
    
    print("\n\n" + "=" * 80)
    print("🧪 测试3: 滑动窗口注意力")
    print("=" * 80)
    
    window_size = 512
    config = ModelConfig(
        vocab_size=500,
        d_model=128,
        num_layers=4,
        num_heads=8,
        d_ff=512,
        max_seq_len=1048576,
        dropout=0.1,
        use_rope=True,
        sliding_window_size=window_size,
    )
    
    model = ShannonB1(config)
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"设备: {device}")
    print(f"滑动窗口大小: {window_size}")
    
    # 测试滑动窗口掩码
    seq_len = 64
    mask = model.sliding_window.create_mask(seq_len, device)
    
    print(f"\n   滑动窗口掩码形状: {mask.shape}")
    print(f"   掩码中-inf的数量: {(mask == float('-inf')).sum().item()}")
    
    # 可视化掩码
    print(f"\n   掩码可视化（前8x8，-inf显示为X）:")
    vis_mask = mask[:8, :8]
    vis_str = ""
    for i in range(8):
        for j in range(8):
            if vis_mask[i, j] == float('-inf'):
                vis_str += "X "
            else:
                vis_str += ". "
        vis_str += "\n"
    print(vis_str)
    
    print("\n✅ 滑动窗口测试完成")


def compare_position_encodings():
    """对比不同位置编码方案"""
    
    print("\n\n" + "=" * 80)
    print("📊 测试4: 位置编码方案对比")
    print("=" * 80)
    
    seq_len = 2048
    configs = [
        ("传统正弦编码", ModelConfig(
            vocab_size=500, d_model=128, num_layers=4, num_heads=8, d_ff=512,
            max_seq_len=seq_len, use_rope=False, use_alibi=False,
        )),
        ("RoPE", ModelConfig(
            vocab_size=500, d_model=128, num_layers=4, num_heads=8, d_ff=512,
            max_seq_len=1048576, use_rope=True, rope_base=10000.0,
        )),
        ("ALiBi", ModelConfig(
            vocab_size=500, d_model=128, num_layers=4, num_heads=8, d_ff=512,
            max_seq_len=1048576, use_rope=False, use_alibi=True,
        )),
        ("RoPE + 滑动窗口", ModelConfig(
            vocab_size=500, d_model=128, num_layers=4, num_heads=8, d_ff=512,
            max_seq_len=1048576, use_rope=True, sliding_window_size=512,
        )),
    ]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokens = torch.randint(0, 500, (1, seq_len)).to(device)
    
    results = []
    
    for name, config in configs:
        print(f"\n   测试: {name}")
        
        model = ShannonB1(config).to(device)
        model.eval()
        
        # 预热
        with torch.no_grad():
            _ = model(tokens[:1, :64])
        
        # 正式测试
        torch.cuda.reset_peak_memory_stats() if device.type == 'cuda' else None
        start_time = time.time()
        
        with torch.no_grad():
            logits, _ = model(tokens)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        memory_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024 if device.type == 'cuda' else 0
        
        print(f"      耗时: {elapsed:.3f}s")
        print(f"      峰值显存: {memory_mb:.1f} MB")
        
        results.append((name, elapsed, memory_mb))
    
    # 汇总对比
    print("\n\n" + "=" * 80)
    print("📈 性能对比总结")
    print("=" * 80)
    print(f"{'方案':<25} {'耗时(s)':<15} {'显存(MB)':<15}")
    print("-" * 80)
    for name, elapsed, memory in results:
        print(f"{name:<25} {elapsed:<15.3f} {memory:<15.1f}")
    print("=" * 80)


def test_long_context_generation():
    """测试长上下文生成"""
    
    print("\n\n" + "=" * 80)
    print("🔄 测试5: 长上下文生成能力")
    print("=" * 80)
    
    config = ModelConfig(
        vocab_size=500,
        d_model=128,
        num_layers=4,
        num_heads=8,
        d_ff=512,
        max_seq_len=1048576,  # 1M
        dropout=0.1,
        use_rope=True,
        rope_base=10000.0,
    )
    
    model = ShannonB1(config)
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"设备: {device}")
    print(f"最大上下文: {config.max_seq_len:,} tokens")
    
    # 测试生成长文本
    prompt_len = 100
    gen_len = 200
    
    prompt_tokens = list(range(prompt_len))
    
    print(f"\n   Prompt长度: {prompt_len}")
    print(f"   生成长度: {gen_len}")
    
    start_time = time.time()
    with torch.no_grad():
        generated = model.generate(
            start_tokens=prompt_tokens,
            max_new_tokens=gen_len,
            temperature=0.7,
            use_kv_cache=True,
        )
    end_time = time.time()
    
    elapsed = end_time - start_time
    speed = gen_len / elapsed
    
    print(f"\n   总耗时: {elapsed:.3f}s")
    print(f"   生成速度: {speed:.2f} tokens/秒")
    print(f"   总序列长度: {len(generated)}")
    
    print("\n✅ 长上下文生成测试完成")


if __name__ == '__main__':
    print("🚀 Shannon-b1 长上下文（1M+）优化测试")
    print("=" * 80)
    
    # 运行所有测试
    test_rope_extrapolation()
    test_alibi_bias()
    test_sliding_window()
    compare_position_encodings()
    test_long_context_generation()
    
    print("\n\n" + "=" * 80)
    print("✅ 所有长上下文测试完成!")
    print("=" * 80)
