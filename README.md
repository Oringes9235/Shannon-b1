# Shannon-b1


<p align="center">
  <img src="https://img.shields.io/badge/Version-Shannon_b1-red.svg" alt="Version">
  <img src="https://img.shields.io/badge/Status-Developing-green.svg" alt="Status">
  <img src="https://img.shields.io/badge/Language-Python-blue.svg" alt="Language">
</p>
  


📊 训练结果分析
|指标	|数值	|说明|
|---|---|---|
|初始损失|	2.5363|	随机猜测的损失约为 -ln(1/1000) ≈ 6.9，所以 2.54 已经不错了
|最终损失|	2.4876	|下降了 0.0487，说明模型在学习
|生成质量	|一般	|已经能认出单词结构，但还没学会语法

Adam 优化器显著提升了训练效果！

## ✅ 完成的改进

1. **Adam 优化器** - 自适应学习率，收敛更快
2. **权重衰减** - L2 正则化，防止过拟合
3. **动量项** - 平滑梯度更新
4. **偏差校正** - 解决初期估计偏差




