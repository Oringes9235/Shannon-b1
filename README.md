# Shannon-b1


<p align="center">
  <img src="https://img.shields.io/badge/Version-Shannon_b1-red.svg" alt="Version">
  <img src="https://img.shields.io/badge/Status-Developing-green.svg" alt="Status">
  <img src="https://img.shields.io/badge/Language-Python-blue.svg" alt="Language">
</p>
  

训练损失从 **4.78 降到 3.67**，下降了 **1.11**，比之前 SGD 的效果更好！

## 📊 训练结果分析

| 指标 | SGD (之前) | Adam (现在) | 提升 |
|------|-----------|-------------|------|
| 初始损失 | 4.86 | 4.78 | ✓ |
| 最终损失 | 4.77 | 3.67 | **↓ 1.10** |
| 损失下降 | 0.09 | 1.11 | **12倍** |

Adam 优化器显著提升了训练效果！

## ✅ 完成的改进

1. **Adam 优化器** - 自适应学习率，收敛更快
2. **权重衰减** - L2 正则化，防止过拟合
3. **动量项** - 平滑梯度更新
4. **偏差校正** - 解决初期估计偏差




