import matplotlib.pyplot as plt
import numpy as np

# 1. 准备数据（这里填入你从分析脚本中得到的平均值）
# 示例数据来源于上一阶段的预测
labels = ['Raw Agentview', 'Occluded Agentview', 'Wristview']
correlations = [0.6521, 0.2104, 0.7842]  # 请替换为你实际运行得到的数值

# 2. 设置绘图风格
plt.style.use('seaborn-v0_8-muted') # 使用较美观的配色方案
fig, ax = plt.subplots(figsize=(10, 7))

# 3. 绘制柱状图
# 为不同视角分配颜色：蓝色（全局）、红色（遮挡警告）、绿色（手眼）
colors = ['#3498db', '#e74c3c', '#2ecc71']
bars = ax.bar(labels, correlations, color=colors, width=0.6, edgecolor='black', linewidth=1.2)

# 4. 在柱子上方添加具体的数值标签
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.4f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),  # 向上偏移 5 个点
                textcoords="offset points",
                ha='center', va='bottom', fontsize=12, fontweight='bold')

# 5. 添加轴标签和标题（使用 LaTeX 格式符合控制专业论文要求）
ax.set_ylabel(r'Correlation Score ($\cos \theta$)', fontsize=14)
ax.set_title('Feature-Action Correlation Analysis under Occlusion', fontsize=16, pad=20)
ax.set_ylim(0, 1.0) # 相关性分数通常在 0 到 1 之间显示
ax.grid(axis='y', linestyle='--', alpha=0.7)

# 6. 添加说明文本（Motivation 支撑）
ax.text(1, 0.05, 'Correlation Drop Due to Occlusion', 
        color='red', fontsize=12, ha='center', fontweight='italic')

# 7. 保存图片
output_file = "correlation_bar_chart.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()

print(f"柱状图已生成并保存至: {output_file}")
print("提示：你可以通过 VS Code 侧边栏下载此图片，或使用 scp 命令。")