"""
merge_txt.py - 合并同目录下 txt 文件夹内的所有 .txt 文件
自动检测编码格式并转换为 UTF-8 后合并
需要先安装 chardet: pip install chardet
直接双击运行，或在命令行执行：python merge_txt.py
"""

import os
import sys

def detect_encoding(filepath):
    """检测文件编码"""
    try:
        import chardet
        with open(filepath, 'rb') as f:
            raw = f.read(100000)  # 读取前100KB用于检测
            result = chardet.detect(raw)
            return result.get('encoding', 'utf-8')
    except ImportError:
        # 如果没有 chardet，尝试常见编码
        for enc in ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin-1']:
            try:
                with open(filepath, 'r', encoding=enc) as f:
                    f.read(1000)
                return enc
            except (UnicodeDecodeError, UnicodeError):
                continue
        return 'utf-8'  # 都失败了就用 utf-8


def read_file_safe(filepath):
    """安全读取文件，自动处理编码"""
    encoding = detect_encoding(filepath)
    try:
        with open(filepath, 'r', encoding=encoding) as f:
            return f.read(), encoding
    except (UnicodeDecodeError, UnicodeError):
        # 如果检测的编码失败，尝试用 errors='replace' 强制读取
        print(f"    ⚠️ 编码 {encoding} 失败，使用 replace 模式读取")
        with open(filepath, 'r', encoding=encoding, errors='replace') as f:
            return f.read(), f"{encoding}(replace)"


def merge_txt_files():
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    txt_dir = os.path.join(script_dir, "txt")
    output_file = os.path.join(script_dir, "merged_all.txt")
    
    # 检查 chardet
    try:
        import chardet
        has_chardet = True
    except ImportError:
        has_chardet = False
        print("⚠️ 未安装 chardet，编码检测准确度会降低")
        print("   建议安装: pip install chardet\n")
    
    # 检查 txt 目录是否存在
    if not os.path.exists(txt_dir):
        print(f"❌ 未找到 txt 目录: {txt_dir}")
        print("请在包含 txt 文件夹的目录下运行此脚本")
        input("按回车键退出...")
        return
    
    # 获取所有 .txt 文件（按文件名排序）
    txt_files = sorted([
        f for f in os.listdir(txt_dir) 
        if f.lower().endswith('.txt') and os.path.isfile(os.path.join(txt_dir, f))
    ])
    
    if not txt_files:
        print(f"❌ txt 目录下没有 .txt 文件")
        input("按回车键退出...")
        return
    
    # 显示文件信息
    print("=" * 60)
    print("📄 文本文件合并工具 (自动编码检测)")
    print("=" * 60)
    print(f"源目录: {txt_dir}")
    print(f"输出文件: {output_file}")
    print(f"编码检测: {'chardet (精准)' if has_chardet else '内置 (基础)'}")
    print(f"\n找到 {len(txt_files)} 个文件:")
    
    total_size = 0
    for fname in txt_files:
        fpath = os.path.join(txt_dir, fname)
        size_kb = os.path.getsize(fpath) / 1024
        total_size += os.path.getsize(fpath)
        # 快速检测编码
        enc = detect_encoding(fpath)
        print(f"  📄 {fname} ({size_kb:.1f} KB) [{enc}]")
    
    total_mb = total_size / (1024 * 1024)
    print(f"\n总大小: {total_mb:.2f} MB")
    print("合并中...")
    
    # 合并所有文件
    success_count = 0
    fail_files = []
    
    try:
        with open(output_file, 'w', encoding='utf-8') as out:
            for i, fname in enumerate(txt_files):
                fpath = os.path.join(txt_dir, fname)
                
                try:
                    content, used_enc = read_file_safe(fpath)
                    out.write(content)
                    out.write('\n')  # 文件间加换行
                    success_count += 1
                    
                    # 进度显示
                    percent = (i + 1) / len(txt_files) * 100
                    print(f"\r  进度: {i+1}/{len(txt_files)} ({percent:.0f}%) [{used_enc}]", end='', flush=True)
                
                except Exception as e:
                    fail_files.append((fname, str(e)))
                    print(f"\n  ⚠️ 跳过 {fname}: {e}")
                    continue
        
        print()  # 换行
        
        # 最终报告
        output_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"\n{'=' * 60}")
        print(f"✅ 合并完成!")
        print(f"   输出文件: {output_file}")
        print(f"   文件大小: {output_mb:.2f} MB")
        print(f"   成功合并: {success_count}/{len(txt_files)} 个文件")
        print(f"   输出编码: UTF-8")
        
        if fail_files:
            print(f"\n⚠️ 跳过的文件:")
            for fname, reason in fail_files:
                print(f"   - {fname}: {reason}")
    
    except Exception as e:
        print(f"\n❌ 合并失败: {e}")
    
    input("\n按回车键退出...")

if __name__ == "__main__":
    merge_txt_files()