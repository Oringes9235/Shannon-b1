#!/usr/bin/env python3
"""Plot training/validation curves from saved checkpoint files.

Usage:
  python experiments/plot_history.py --checkpoints checkpoints --out plots/history.png
  python experiments/plot_history.py --glob "experiments/results/*/*.pt" --out plots/compare.png
"""
import argparse
import glob
import os
import torch
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Ensure project root is on sys.path so `src` package can be imported when loading checkpoints
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def find_ckpts(paths):
    files = []
    for p in paths:
        files.extend(glob.glob(p))
    # deduplicate and sort
    files = sorted(list(set(files)))
    return [f for f in files if f.endswith('.pt') or f.endswith('.pth')]


def load_history(ckpt_path):
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
    except Exception as e:
        # 尝试以 weights_only=False 重新加载，同时允许项目中已知的自定义类（如果信任该文件）
        try:
            from torch.serialization import add_safe_globals
            # 尝试引入可能的自定义类型并加入 allowlist
            try:
                from src.model.config import ModelConfig
                safe_list = [ModelConfig]
            except Exception:
                safe_list = []

            if safe_list:
                with add_safe_globals(safe_list):
                    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            else:
                ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        except Exception as e2:
            print(f"Failed to load {ckpt_path}: {e2}")
            return None
    history = ckpt.get('history') or {}
    train = history.get('train_loss', [])
    val = history.get('val_loss', [])
    return {'train': train, 'val': val}


def plot(histories, labels, out_path=None):
    plt.figure(figsize=(8, 5))
    for h, label in zip(histories, labels):
        if not h:
            continue
        train = h.get('train', [])
        val = h.get('val', [])
        x = list(range(1, len(train) + 1))
        if train:
            plt.plot(x, train, label=f"{label} train")
        if val:
            plt.plot(x[:len(val)], val, linestyle='--', label=f"{label} val")

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {out_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--glob', nargs='+', help='glob patterns to checkpoint files', default=[])
    parser.add_argument('--checkpoints', help='directory to search for checkpoints', default=None)
    parser.add_argument('--out', help='output image path', default='experiments/plots/history.png')
    args = parser.parse_args()

    patterns = []
    if args.checkpoints:
        patterns.append(os.path.join(args.checkpoints, '*.pt'))
    patterns += args.glob

    if not patterns:
        print('Provide --checkpoints or --glob patterns')
        return

    files = find_ckpts(patterns)
    if not files:
        print('No checkpoint files found for patterns:', patterns)
        return

    histories = []
    labels = []
    for f in files:
        hist = load_history(f)
        if hist:
            histories.append(hist)
            labels.append(os.path.basename(os.path.dirname(f)) or os.path.basename(f))

    if not histories:
        print('No valid history data found in checkpoints.')
        return

    plot(histories, labels, args.out)


if __name__ == '__main__':
    main()
