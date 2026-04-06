"""
流式输出实时性测试脚本

此脚本用于测试后端SSE流式输出是否真正实时推送数据
"""

import requests
import json
import time

def test_streaming_realtime():
    """测试流式输出的实时性"""
    
    print("="*70)
    print("流式输出实时性测试")
    print("="*70)
    print()
    
    # API端点
    url = "http://localhost:8000/api/generate/stream"
    
    # 请求参数
    payload = {
        "prompt": "The world is",
        "max_tokens": 20,
        "temperature": 0.85,
        "top_k": 40,
        "repetition_penalty": 1.15
    }
    
    print(f"发送请求到: {url}")
    print(f"提示词: {payload['prompt']}")
    print(f"最大token数: {payload['max_tokens']}")
    print()
    print("开始接收数据...")
    print("-"*70)
    
    start_time = time.time()
    token_count = 0
    
    try:
        # 发送流式请求
        response = requests.post(url, json=payload, stream=True)
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                
                # 解析SSE格式
                if line_str.startswith('data: '):
                    data_str = line_str[6:]  # 去掉 "data: " 前缀
                    
                    try:
                        data = json.loads(data_str)
                        
                        # 检查是否是完成信号
                        if data.get('type') == 'complete':
                            elapsed = time.time() - start_time
                            print()
                            print("-"*70)
                            print(f"✅ 生成完成！")
                            print(f"   总耗时: {elapsed:.2f}秒")
                            print(f"   Token数量: {token_count}")
                            if elapsed > 0:
                                print(f"   平均速度: {token_count/elapsed:.2f} tokens/秒")
                            break
                        
                        # 显示每个token的信息
                        token_count += 1
                        current_time = time.time() - start_time
                        text_preview = data.get('text', '')[-30:]  # 显示最后30个字符
                        
                        # 安全地获取token_id（可能是None）
                        token_id = data.get('token_id')
                        token_id_str = f"{token_id:3d}" if token_id is not None else "N/A"
                        probability = data.get('probability', 0)
                        prob_str = f"{probability:.4f}" if probability is not None else "N/A"
                        
                        print(f"[{current_time:6.3f}s] Token #{token_count:2d} | "
                              f"ID: {token_id_str} | "
                              f"Prob: {prob_str} | "
                              f"Text: ...{text_preview}")
                        
                    except json.JSONDecodeError as e:
                        print(f"❌ JSON解析错误: {e}")
                        print(f"   原始数据: {data_str[:100]}")
                        
    except requests.exceptions.RequestException as e:
        print(f"❌ 请求失败: {e}")
        return False
    
    print()
    print("="*70)
    
    if token_count > 0:
        print("✅ 测试通过：流式输出正常工作")
        print()
        print("如果看到上面的逐行输出，说明流式功能正常。")
        print("每个token都应该有独立的时间戳。")
        return True
    else:
        print("❌ 测试失败：没有接收到任何token")
        return False


if __name__ == '__main__':
    test_streaming_realtime()
