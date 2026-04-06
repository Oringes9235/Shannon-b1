"""
综合流式输出测试脚本

整合了以下测试功能：
1. 快速一致性验证 (temperature=0.0)
2. 深度调试分析 (logits输出、参数对比)
3. 多次迭代稳定性测试
4. 不同参数配置测试
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.model import ShannonB1, ModelConfig
from src.data import CharTokenizer


def create_model_and_tokenizer():
    """创建模型和分词器（禁用dropout以确保确定性）"""
    config = ModelConfig(
        vocab_size=100,
        d_model=64,
        num_layers=2,
        num_heads=4,
        max_seq_len=128,
        dropout=0.0  # 关键：禁用dropout
    )
    model = ShannonB1(config)
    model.eval()
    
    tokenizer = CharTokenizer()
    tokenizer.build_vocab(["hello world test streaming"], 100)
    
    return model, tokenizer


def test_quick_consistency(model, tokenizer):
    """
    测试1: 快速一致性验证
    检查 temperature=0.0 时流式生成与普通生成是否一致
    """
    print("\n" + "="*70)
    print("测试1: 快速一致性验证 (temperature=0.0)")
    print("="*70)
    
    start_tokens = tokenizer.encode("world")
    print(f"Start tokens: {start_tokens}\n")
    
    # 流式生成
    stream_tokens = []
    for token_id, prob in model.generate_stream(
        start_tokens,
        max_new_tokens=5,
        temperature=0.0,
        repetition_penalty=1.0,
        ban_immediate_repeat=False,
        ngram_block_size=0,
        max_repetition=100
    ):
        stream_tokens.append(token_id)
    
    # 普通生成
    normal_tokens = model.generate(
        start_tokens,
        max_new_tokens=5,
        temperature=0.0,
        repetition_penalty=1.0,
        ban_immediate_repeat=False,
        ngram_block_size=0,
        max_repetition=100
    )
    
    normal_new = normal_tokens[len(start_tokens):]
    
    print(f"Stream: {stream_tokens}")
    print(f"Normal: {normal_new}")
    print(f"Match:  {stream_tokens == normal_new}")
    
    if stream_tokens == normal_new:
        print("✅ PASSED: 两种方法产生一致结果\n")
        return True
    else:
        print("❌ FAILED: 两种方法结果不一致\n")
        return False


def test_debug_logits(model, tokenizer):
    """
    测试2: 深度调试分析
    检查logits输出和第一步生成的token
    """
    print("\n" + "="*70)
    print("测试2: 深度调试分析 (Logits检查)")
    print("="*70)
    
    start_tokens = tokenizer.encode("world")
    print(f"Start tokens: {start_tokens}\n")
    
    # 直接前向传播
    print("=== 直接前向传播 ===")
    tokens_tensor = torch.tensor([start_tokens])
    with torch.no_grad():
        logits = model.forward(tokens_tensor)
        last_logits = logits[0, -1, :].float()
    
    top5_values, top5_indices = torch.topk(last_logits, 5)
    print("Top 5 logits:")
    for i, (val, idx) in enumerate(zip(top5_values, top5_indices)):
        prob = torch.softmax(last_logits, dim=-1)[idx].item()
        print(f"  {i+1}. Token {idx.item()}: logit={val.item():.4f}, prob={prob:.4f}")
    
    greedy_token = torch.argmax(last_logits).item()
    print(f"\nGreedy token: {greedy_token}\n")
    
    # 流式生成第一步
    print("=== 流式生成第一步 ===")
    stream_iter = model.generate_stream(
        start_tokens,
        max_new_tokens=1,
        temperature=0.0,
        repetition_penalty=1.0,
        ban_immediate_repeat=False,
        ngram_block_size=0,
        max_repetition=100
    )
    stream_token, stream_prob = next(stream_iter)
    print(f"Stream first token: {stream_token} (prob={stream_prob:.4f})\n")
    
    # 普通生成第一步
    print("=== 普通生成第一步 ===")
    normal_tokens = model.generate(
        start_tokens,
        max_new_tokens=1,
        temperature=0.0,
        repetition_penalty=1.0,
        ban_immediate_repeat=False,
        ngram_block_size=0,
        max_repetition=100
    )
    normal_token = normal_tokens[len(start_tokens)]
    print(f"Normal first token: {normal_token}\n")
    
    match = (stream_token == normal_token == greedy_token)
    print(f"三者一致: {match}")
    
    if match:
        print("✅ PASSED: 所有方法选择相同的token\n")
        return True
    else:
        print("❌ FAILED: token选择不一致\n")
        return False


def test_multiple_iterations(model, tokenizer):
    """
    测试3: 多次迭代稳定性测试
    验证在相同条件下多次运行的一致性
    """
    print("\n" + "="*70)
    print("测试3: 多次迭代稳定性测试")
    print("="*70)
    
    start_tokens = tokenizer.encode("world")
    print(f"Start tokens: {start_tokens}\n")
    
    all_passed = True
    
    for iteration in range(3):
        print(f"Iteration {iteration + 1}:")
        
        # 流式生成
        stream_tokens = []
        for token_id, _ in model.generate_stream(
            start_tokens,
            max_new_tokens=5,
            temperature=0.0,
            repetition_penalty=1.0,
            ban_immediate_repeat=False,
            ngram_block_size=0,
            max_repetition=100
        ):
            stream_tokens.append(token_id)
        
        # 普通生成
        normal_tokens = model.generate(
            start_tokens,
            max_new_tokens=5,
            temperature=0.0,
            repetition_penalty=1.0,
            ban_immediate_repeat=False,
            ngram_block_size=0,
            max_repetition=100
        )
        
        normal_new = normal_tokens[len(start_tokens):]
        match = stream_tokens == normal_new
        
        print(f"  Stream: {stream_tokens}")
        print(f"  Normal: {normal_new}")
        print(f"  Match:  {match}\n")
        
        if not match:
            all_passed = False
    
    if all_passed:
        print("✅ PASSED: 所有迭代都保持一致\n")
        return True
    else:
        print("❌ FAILED: 存在不一致的迭代\n")
        return False


def test_different_parameters(model, tokenizer):
    """
    测试4: 不同参数配置测试
    测试ban_immediate_repeat等参数的影响
    """
    print("\n" + "="*70)
    print("测试4: 不同参数配置测试")
    print("="*70)
    
    start_tokens = tokenizer.encode("world")
    print(f"Start tokens: {start_tokens}\n")
    
    all_passed = True
    
    for ban_repeat in [True, False]:
        print(f"--- ban_immediate_repeat={ban_repeat} ---")
        
        # 流式生成
        stream_tokens = []
        for token_id, prob in model.generate_stream(
            start_tokens,
            max_new_tokens=5,
            temperature=0.0,
            repetition_penalty=1.0,
            ban_immediate_repeat=ban_repeat,
            ngram_block_size=0,
            max_repetition=100
        ):
            stream_tokens.append(token_id)
        
        # 普通生成
        normal_tokens = model.generate(
            start_tokens,
            max_new_tokens=5,
            temperature=0.0,
            repetition_penalty=1.0,
            ban_immediate_repeat=ban_repeat,
            ngram_block_size=0,
            max_repetition=100
        )
        
        normal_new = normal_tokens[len(start_tokens):]
        match = stream_tokens == normal_new
        
        print(f"  Stream: {stream_tokens}")
        print(f"  Normal: {normal_new}")
        print(f"  Match:  {match}\n")
        
        if not match:
            all_passed = False
    
    if all_passed:
        print("✅ PASSED: 所有参数配置下都保持一致\n")
        return True
    else:
        print("❌ FAILED: 某些参数配置下不一致\n")
        return False


def main():
    """主测试函数"""
    print("\n" + "="*70)
    print("Shannon-b1 流式输出综合测试套件")
    print("="*70)
    
    # 初始化
    print("\n初始化模型和分词器...")
    model, tokenizer = create_model_and_tokenizer()
    print("✅ 初始化完成\n")
    
    # 执行所有测试
    results = {}
    
    results['quick_consistency'] = test_quick_consistency(model, tokenizer)
    results['debug_logits'] = test_debug_logits(model, tokenizer)
    results['multiple_iterations'] = test_multiple_iterations(model, tokenizer)
    results['different_parameters'] = test_different_parameters(model, tokenizer)
    
    # 总结
    print("\n" + "="*70)
    print("测试总结")
    print("="*70)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:30s} {status}")
    
    print("-"*70)
    print(f"总计: {passed_tests}/{total_tests} 通过")
    print("="*70)
    
    if passed_tests == total_tests:
        print("\n🎉 所有测试通过！流式输出功能正常工作。")
        return 0
    else:
        print(f"\n⚠️  {total_tests - passed_tests} 个测试失败，请检查上述错误信息。")
        return 1


if __name__ == '__main__':
    sys.exit(main())
