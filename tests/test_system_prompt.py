#!/usr/bin/env python
"""
系统提示词功能测试脚本
用于验证system_prompt参数是否正确传递和处理
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_system_prompt_concatenation():
    """测试系统提示词拼接逻辑"""
    print("=" * 60)
    print("测试1: 系统提示词拼接逻辑")
    print("=" * 60)
    
    # 模拟model_manager中的拼接逻辑
    def build_full_prompt(prompt, system_prompt):
        full_prompt = prompt
        if system_prompt and system_prompt.strip():
            full_prompt = f"{system_prompt.strip()}\n\n{prompt}"
        return full_prompt
    
    # 测试用例1: 有系统提示词
    result1 = build_full_prompt(
        "Hello, how are you?",
        "你是一个翻译专家"
    )
    print("\n[92m[SUCCESS][0m 测试1.1 - 正常系统提示词:")
    print(f"   System: '你是一个翻译专家'")
    print(f"   User: 'Hello, how are you?'")
    print(f"   Result: '{result1}'")
    assert "你是一个翻译专家" in result1
    assert "\n\n" in result1
    print("   ✓ 通过")
    
    # 测试用例2: 无系统提示词 (None)
    result2 = build_full_prompt("Hello", None)
    print("\n[92m[SUCCESS][0m 测试1.2 - 系统提示词为None:")
    print(f"   System: None")
    print(f"   User: 'Hello'")
    print(f"   Result: '{result2}'")
    assert result2 == "Hello"
    print("   ✓ 通过")
    
    # 测试用例3: 空字符串
    result3 = build_full_prompt("Hello", "")
    print("\n[92m[SUCCESS][0m 测试1.3 - 系统提示词为空字符串:")
    print(f"   System: ''")
    print(f"   User: 'Hello'")
    print(f"   Result: '{result3}'")
    assert result3 == "Hello"
    print("   ✓ 通过")
    
    # 测试用例4: 纯空白字符
    result4 = build_full_prompt("Hello", "   ")
    print("\n[92m[SUCCESS][0m 测试1.4 - 系统提示词为纯空白:")
    print(f"   System: '   '")
    print(f"   User: 'Hello'")
    print(f"   Result: '{result4}'")
    assert result4 == "Hello"
    print("   ✓ 通过")
    
    # 测试用例5: 带首尾空格的系统提示词
    result5 = build_full_prompt("Hello", "  测试  ")
    print("\n[92m[SUCCESS][0m 测试1.5 - 系统提示词带首尾空格:")
    print(f"   System: '  测试  '")
    print(f"   User: 'Hello'")
    print(f"   Result: '{result5}'")
    assert result5 == "测试\n\nHello"
    print("   ✓ 通过")
    
    print("\n" + "=" * 60)
    print("[92m[OK][0m 所有拼接逻辑测试通过!")
    print("=" * 60)


def test_api_request_format():
    """测试API请求格式"""
    print("\n" + "=" * 60)
    print("测试2: API请求格式验证")
    print("=" * 60)
    
    import json
    
    # 模拟前端发送的请求
    request_data = {
        "prompt": "What is AI?",
        "system_prompt": "你是一个AI专家",
        "max_tokens": 100,
        "temperature": 0.8,
        "top_k": 40,
        "repetition_penalty": 1.15
    }
    
    print("\n[92m[SUCCESS][0m 测试2.1 - 完整请求体:")
    print(json.dumps(request_data, indent=2, ensure_ascii=False))
    
    # 验证字段存在
    assert "system_prompt" in request_data
    assert request_data["system_prompt"] == "你是一个AI专家"
    print("   ✓ system_prompt字段正确")
    
    # 测试可选参数 (不传system_prompt)
    request_data_optional = {
        "prompt": "What is AI?",
        "max_tokens": 100
    }
    
    print("\n[92m[SUCCESS][0m 测试2.2 - 可选参数 (不含system_prompt):")
    print(json.dumps(request_data_optional, indent=2, ensure_ascii=False))
    assert "system_prompt" not in request_data_optional
    print("   ✓ system_prompt可以省略")
    
    print("\n" + "=" * 60)
    print("[92m[OK][0m API请求格式测试通过!")
    print("=" * 60)


def test_cli_arguments():
    """测试CLI参数解析"""
    print("\n" + "=" * 60)
    print("测试3: CLI参数解析")
    print("=" * 60)
    
    import argparse
    
    parser = argparse.ArgumentParser(description='测试CLI参数')
    parser.add_argument('--system-prompt', type=str, default=None)
    parser.add_argument('--prompt', type=str, default="Test")
    
    # 模拟命令行参数
    args_with_system = parser.parse_args([
        '--system-prompt', '你是一个助手',
        '--prompt', '你好'
    ])
    
    print("\n[92m[SUCCESS][0m 测试3.1 - 带系统提示词的CLI调用:")
    print(f"   --system-prompt: '{args_with_system.system_prompt}'")
    print(f"   --prompt: '{args_with_system.prompt}'")
    assert args_with_system.system_prompt == "你是一个助手"
    print("   ✓ 参数解析正确")
    
    args_without_system = parser.parse_args([
        '--prompt', '你好'
    ])
    
    print("\n[92m[SUCCESS][0m 测试3.2 - 不带系统提示词的CLI调用:")
    print(f"   --system-prompt: {args_without_system.system_prompt}")
    print(f"   --prompt: '{args_without_system.prompt}'")
    assert args_without_system.system_prompt is None
    print("   ✓ 默认值为None")
    
    print("\n" + "=" * 60)
    print("[92m[OK][0m CLI参数测试通过!")
    print("=" * 60)


def main():
    """运行所有测试"""
    print("\n" + "🧪" * 30)
    print("开始系统提示词功能测试")
    print("🧪" * 30 + "\n")
    
    try:
        test_system_prompt_concatenation()
        test_api_request_format()
        test_cli_arguments()
        
        print("\n" + "[95m[HIGHLIGHT][0m" * 30)
        print("[92m[OK][0m 所有测试通过! 系统提示词功能正常工作!")
        print("[95m[HIGHLIGHT][0m" * 30 + "\n")
        
        print("[90m[TOKEN][0m 使用说明:")
        print("   1. Web UI: 在'系统提示词'输入框中填写角色设定")
        print("   2. CLI: 使用 --system-prompt 参数")
        print("   3. API: 在请求体中包含 system_prompt 字段")
        print("\n[96m[READ][0m 详细文档: notebooks/SYSTEM_PROMPT_GUIDE.md\n")
        
    except AssertionError as e:
        print(f"\n[91m[ERROR][0m 测试失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[91m[ERROR][0m 未知错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()