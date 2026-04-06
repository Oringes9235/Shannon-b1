#!/usr/bin/env python
"""
CUDA兼容性测试脚本
用于检测和验证当前环境的CUDA配置是否适合训练Shannon-b1模型
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def print_header():
    """打印美观的标题"""
    width = 70
    print()
    print("╔" + "═" * (width - 2) + "╗")
    print("║" + " " * 15 + "Shannon-b1 CUDA Compatibility Test" + " " * 19 + "║")
    print("╚" + "═" * (width - 2) + "╝")
    print()


def print_section(title):
    """打印分隔标题"""
    width = 70
    print()
    print("┌" + "─" * (width - 2) + "┐")
    print(f"│  {title:<{width-6}}│")
    print("└" + "─" * (width - 2) + "┘")


def print_success(msg):
    """打印成功信息"""
    print(f"  ✓ {msg}")


def print_warning(msg):
    """打印警告信息"""
    print(f"  ⚠ {msg}")


def print_error(msg):
    """打印错误信息"""
    print(f"  ✗ {msg}")


def print_info(label, value, indent=2):
    """打印键值对信息"""
    print(f"{' ' * indent}{label:<30} : {value}")


def print_subsection(title):
    """打印子章节标题"""
    print(f"\n  {'─' * 40}")
    print(f"  {title}")
    print(f"  {'─' * 40}")


def test_pytorch_version():
    """测试PyTorch版本"""
    print_section("PyTorch Version Information")
    
    version = torch.__version__
    python_version = sys.version.split()[0]
    
    print_info("PyTorch Version", version)
    print_info("Python Version", python_version)
    
    # 检查版本是否支持所需功能
    version_parts = version.split('+')[0].split('.')
    major, minor = int(version_parts[0]), int(version_parts[1])
    
    print()
    if major >= 2 or (major == 1 and minor >= 10):
        print_success("PyTorch version supports mixed precision training (>= 1.10)")
        return True
    else:
        print_warning("PyTorch version < 1.10, some features may not work optimally")
        return True


def test_cuda_availability():
    """测试CUDA可用性"""
    print_section("CUDA Availability Check")
    
    cuda_available = torch.cuda.is_available()
    print_info("CUDA Available", "Yes" if cuda_available else "No")
    
    if not cuda_available:
        print()
        print_warning("CUDA is not available. Training will use CPU.")
        print_warning("This will be significantly slower than GPU training.")
        print()
        print("  To enable CUDA:")
        print("    1. Install NVIDIA CUDA Toolkit")
        print("    2. Install PyTorch with CUDA support:")
        print("       pip install torch --index-url https://download.pytorch.org/whl/cu118")
        return False
    
    print_success("CUDA is available and ready to use")
    return True


def test_cuda_details():
    """测试CUDA详细信息"""
    print_section("CUDA Environment Details")
    
    try:
        from src.utils import get_cuda_info
        info = get_cuda_info()
        
        print_subsection("CUDA Runtime")
        print_info("CUDA Version", info['cuda_version'])
        print_info("cuDNN Version", info['cudnn_version'])
        print_info("Device Count", info['device_count'])
        
        if info['device_count'] == 0:
            print()
            print_warning("No CUDA devices found despite CUDA being available")
            return False
        
        for i, device in enumerate(info['devices']):
            print_subsection(f"GPU Device {i}")
            print_info("Name", device['name'])
            print_info("Compute Capability", f"{device['compute_capability'][0]}.{device['compute_capability'][1]}")
            print_info("Total Memory", f"{device['memory_total'] / 1024**3:.2f} GB")
            
            # 检查计算能力
            cc_major, cc_minor = device['compute_capability']
            print()
            if cc_major >= 7:
                print_success("Excellent! Supports Tensor Cores (CC >= 7.0)")
            elif cc_major >= 6:
                print_success("Good! Supports mixed precision (CC >= 6.0)")
            else:
                print_warning("Low compute capability, mixed precision may not work")
        
        return True
        
    except Exception as e:
        print_error(f"Error retrieving CUDA details: {e}")
        return False


def test_mixed_precision():
    """测试混合精度训练支持"""
    print_section("Mixed Precision Training Support")
    
    if not torch.cuda.is_available():
        print("  ⊘ Skipped (CUDA not available)")
        return True
    
    try:
        from torch import amp
        
        print_subsection("GradScaler Initialization")
        try:
            scaler = amp.GradScaler(device_type='cuda')
            print_success("Initialized with device_type parameter (PyTorch >= 2.0)")
        except TypeError:
            scaler = amp.GradScaler()
            print_success("Initialized in legacy mode (PyTorch < 2.0)")
        
        print_subsection("autocast Context Manager")
        autocast_works = False
        
        # 方式1: PyTorch >= 1.10 (推荐)
        try:
            with amp.autocast(device_type='cuda'):
                x = torch.randn(2, 3).cuda()
                y = x @ x.t()
            print_success("Works with device_type parameter (recommended)")
            autocast_works = True
        except TypeError:
            # 方式2: 更旧版本
            try:
                with amp.autocast():
                    x = torch.randn(2, 3).cuda()
                    y = x @ x.t()
                print_success("Works without parameters (legacy)")
                autocast_works = True
            except Exception:
                # 方式3: 使用torch.cuda.amp
                try:
                    from torch.cuda.amp import autocast as cuda_autocast
                    with cuda_autocast():
                        x = torch.randn(2, 3).cuda()
                        y = x @ x.t()
                    print_success("Works with torch.cuda.amp.autocast")
                    autocast_works = True
                except Exception as e:
                    print_error(f"autocast initialization failed: {e}")
        
        print()
        if autocast_works:
            print_success("Mixed precision training is fully supported")
            return True
        else:
            print_warning("autocast not working properly")
            print_warning("You can still train using --no-amp flag")
            return False
        
    except Exception as e:
        print_error(f"Mixed precision test failed: {e}")
        print_warning("You can still train without mixed precision using --no-amp flag")
        return False


def test_memory_allocation():
    """测试内存分配"""
    print_section("GPU Memory Allocation Test")
    
    if not torch.cuda.is_available():
        print("  ⊘ Skipped (CUDA not available)")
        return True
    
    try:
        device = torch.device('cuda')
        
        # 尝试分配一个小张量
        test_tensor = torch.randn(100, 100, device=device)
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        
        print_success("Successfully allocated tensor on GPU")
        print_info("Memory Allocated", f"{allocated:.2f} MB")
        print_info("Memory Reserved", f"{reserved:.2f} MB")
        
        # 清理
        del test_tensor
        torch.cuda.empty_cache()
        
        print()
        print_success("Memory cleanup successful")
        return True
        
    except Exception as e:
        print_error(f"Memory allocation failed: {e}")
        return False


def test_model_creation():
    """测试模型创建"""
    print_section("Model Creation & Forward Pass Test")
    
    try:
        from src.model import ShannonB1, ModelConfig
        
        # 创建一个小模型配置
        config = ModelConfig(
            vocab_size=100,
            d_model=64,
            num_heads=4,
            d_ff=256,
            num_layers=2,
            max_seq_len=32,
            dropout=0.1,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        model = ShannonB1(config)
        total_params = sum(p.numel() for p in model.parameters())
        
        print_success("Model created successfully")
        print_info("Total Parameters", f"{total_params:,}")
        print_info("Device", config.device)
        
        # 如果CUDA可用，测试模型在GPU上的运行
        if torch.cuda.is_available():
            model = model.cuda()
            test_input = torch.randint(0, 100, (2, 16)).cuda()
            
            with torch.no_grad():
                output = model(test_input)
            
            print()
            print_success("Forward pass successful on GPU")
            print_info("Output Shape", str(output.shape))
        
        return True
        
    except Exception as e:
        print_error(f"Model creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_summary(results):
    """打印测试总结"""
    print_section("Test Summary")
    
    all_passed = all(results.values())
    
    test_names = {
        'pytorch_version': 'PyTorch Version',
        'cuda_available': 'CUDA Availability',
        'cuda_details': 'CUDA Details',
        'mixed_precision': 'Mixed Precision',
        'memory_allocation': 'Memory Allocation',
        'model_creation': 'Model Creation'
    }
    
    print()
    for test_key, test_name in test_names.items():
        if test_key in results:
            passed = results[test_key]
            status = "✓ PASS" if passed else "✗ FAIL"
            icon = "✓" if passed else "✗"
            print(f"  {icon} {test_name:<30} {status}")
    
    print()
    print("─" * 70)
    
    if all_passed:
        print()
        print("  🎉 All tests passed! Your environment is ready for training.")
        print()
        print("  Quick start:")
        print("    python scripts/train.py")
        print()
    else:
        print()
        print("  ⚠ Some tests failed. Please check the details above.")
        print()
        print("  Recommendations:")
        if not results.get('cuda_available', True):
            print("    • Consider installing PyTorch with CUDA support")
        if not results.get('mixed_precision', True):
            print("    • Use --no-amp flag to disable mixed precision training")
        if not results.get('model_creation', True):
            print("    • Check your installation and dependencies")
        print()
    
    print("─" * 70)
    print()


def main():
    """运行所有测试"""
    print_header()
    
    results = {}
    
    # 运行测试
    results['pytorch_version'] = test_pytorch_version()
    results['cuda_available'] = test_cuda_availability()
    
    if results['cuda_available']:
        results['cuda_details'] = test_cuda_details()
        results['mixed_precision'] = test_mixed_precision()
        results['memory_allocation'] = test_memory_allocation()
    
    results['model_creation'] = test_model_creation()
    
    # 总结
    print_summary(results)
    
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
