import os
import pathlib
from datetime import datetime

base = pathlib.Path('f:/Shannon-b1')

# 日志保存路径 - 使用当前脚本所在目录
log_dir = pathlib.Path.cwd() / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / 'Shannon_codeLines.log'

print(f'📁 Log will be saved to: {log_file}')

exts = {'.py', '.js', '.jsx', '.ts', '.tsx', '.html', '.css',
        '.yaml', '.yml', '.json', '.md', '.bat', '.sh', '.txt', '.cfg', '.toml'}

skip_dirs = {'node_modules', 'checkpoints', '.git', '__pycache__', '.pytest_cache'}

cats = {
    'code': {'.py', '.js', '.jsx', '.ts', '.tsx', '.html', '.css'},
    'doc': {'.md'},
    'config': {'.yaml', '.yml', '.json', '.txt', '.toml', '.cfg'},
    'script': {'.bat', '.sh'},
}

sums = {k: 0 for k in cats}
cat_files = {k: 0 for k in cats}
rows = []

for p in base.rglob('*'):
    if p.suffix.lower() not in exts:
        continue
    if any(d in p.parts for d in skip_dirs):
        continue
    if 'package-lock' in p.name:
        continue
    rel = os.path.relpath(str(p), str(base))
    try:
        with open(str(p), encoding='utf-8', errors='ignore') as f:
            n = sum(1 for _ in f)
    except Exception:
        n = 0
    rows.append((rel, n))

for rel, n in rows:
    for cat, suffixes in cats.items():
        if pathlib.Path(rel).suffix.lower() in suffixes:
            sums[cat] += n
            cat_files[cat] += 1
            break

total_lines = sum(r[1] for r in rows)
total_files = len(rows)
code_and_script = sums['code'] + sums['script']

# ============================================================
# 构建输出内容
# ============================================================

output_lines = []

def add_line(text=''):
    """同时添加到输出列表和打印到控制台"""
    output_lines.append(text)
    print(text)

timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

add_line()
add_line('=' * 60)
add_line(f'         CODEBASE  LINE  COUNT')
add_line(f'         Generated: {timestamp}')
add_line('=' * 60)

# 核心数字 - 大字显示
add_line(f'''
   ███████╗██╗  ██╗ █████╗ ███╗   ██╗███╗   ██╗ ██████╗ ███╗   ██╗
   ██╔════╝██║  ██║██╔══██╗████╗  ██║████╗  ██║██╔═══██╗████╗  ██║
   ███████╗███████║███████║██╔██╗ ██║██╔██╗ ██║██║   ██║██╔██╗ ██║
   ╚════██║██╔══██║██╔══██║██║╚██╗██║██║╚██╗██║██║   ██║██║╚██╗██║
   ███████║██║  ██║██║  ██║██║ ╚████║██║ ╚████║╚██████╔╝██║ ╚████║
   ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═══╝ ╚═════╝ ╚═╝  ╚═══╝
                                                                  
    ██████╗ ██████╗ ██████╗ ███████╗                               
   ██╔════╝██╔═══██╗██╔══██╗██╔════╝                               
   ██║     ██║   ██║██║  ██║█████╗                                 
   ██║     ██║   ██║██║  ██║██╔══╝                                 
   ╚██████╗╚██████╔╝██████╔╝███████╗                               
    ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝                               
''')

add_line(f'        {sums["code"]:>10,}   lines of code')
add_line(f'        {sums["script"]:>10,}   lines of scripts')
add_line(f'        {"─" * 20}')
add_line(f'        {code_and_script:>10,}   TOTAL source')
add_line()

# 分类明细
add_line('─' * 60)
add_line(f'  {"Category":<20} {"Lines":>10}  {"Files":>8}  {"%":>6}')
add_line('─' * 60)

labels = {
    'code':   'Python/JS/TS/Web',
    'script': 'Bash/Bat Scripts',
    'config': 'YAML/JSON/TOML/CFG',
    'doc':    'Markdown Docs',
}

order = ['code', 'script', 'config', 'doc']
for cat in order:
    lines = sums[cat]
    files = cat_files[cat]
    pct = (lines / total_lines * 100) if total_lines > 0 else 0
    add_line(f'  {labels[cat]:<20} {lines:>10,}  {files:>8}  {pct:>5.1f}%')

add_line('─' * 60)
add_line(f'  {"ALL FILES":<20} {total_lines:>10,}  {total_files:>8}  {100.0:>5.1f}%')
add_line('─' * 60)

# 文件大小分布
buckets = {'  1-99': 0, '100-199': 0, '200-499': 0, '500-999': 0, '  1000+': 0}
for _, n in rows:
    if n < 100:     buckets['  1-99'] += 1
    elif n < 200:   buckets['100-199'] += 1
    elif n < 500:   buckets['200-499'] += 1
    elif n < 1000:  buckets['500-999'] += 1
    else:           buckets['  1000+'] += 1

add_line(f'\n  File size distribution:')
max_bucket = max(buckets.values()) if buckets else 1
for label, count in buckets.items():
    bar_len = int(count / max_bucket * 25) if max_bucket > 0 else 0
    bar = '#' * bar_len
    add_line(f'  {label} lines: {bar:<25} {count:>4} files')

# 详细文件列表
add_line(f'\n{"=" * 60}')
add_line(f'  ALL FILES (sorted by line count)')
add_line(f'{"=" * 60}')

for rel, n in sorted(rows, key=lambda x: -x[1]):
    ext = pathlib.Path(rel).suffix.lower()
    # 扩展名缩写
    ext_short = ext.lstrip('.')[:6]
    add_line(f'  {n:>6,} lines  [{ext_short:<6}]  {rel}')

add_line(f'\n{"=" * 60}')
add_line(f'  Project: {base.name}')
add_line(f'  Total: {total_lines:,} lines in {total_files} files')
add_line(f'  Log saved to: {log_file}')
add_line(f'{"=" * 60}')
add_line()

# ============================================================
# 保存到日志文件
# ============================================================

try:
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    print(f'\n[92m[SUCCESS][0m Log successfully saved to: {log_file}')
    print(f'   File size: {log_file.stat().st_size:,} bytes')
except Exception as e:
    print(f'\n[91m[ERROR][0m Failed to save log: {e}')