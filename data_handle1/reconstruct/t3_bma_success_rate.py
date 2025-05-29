

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False

sns.set_style("whitegrid")
import pandas as pd

plt.rcParams['figure.dpi'] = 500

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))

# plt.figure(figsize=(12, 6))
df = pd.read_excel('plot.xlsx', sheet_name='t3_1')
df.to_csv('plot.csv', index=False, encoding='utf-8')
df = pd.read_csv('plot.csv', encoding='utf-8')




ax1 = sns.barplot(x='data', y='success', hue="method", data=df[0:8], ax=axes[0])
ax1.set_ylim(0.99, 1.0012)


font_size = 16
axes[0].tick_params(axis='x', labelsize=font_size)
axes[0].tick_params(axis='y', labelsize=font_size)
axes[0].set_xlabel('', fontsize=font_size)
axes[0].set_ylabel('sequence recovery rate', fontsize=font_size)

ax2 = sns.barplot(x='data', y='success', hue="method", data=df[8:16], ax=axes[1])
ax2.set_ylim(0.65, 1)


axes[1].tick_params(axis='x', labelsize=font_size)
axes[1].tick_params(axis='y', labelsize=font_size)
axes[1].set_xlabel('', fontsize=font_size)
axes[1].set_ylabel('sequence recovery rate', fontsize=font_size)


# axa = [ax1,ax2]
# for ax in axa:
#     for p in ax.patches:
#         height = p.get_height()  # 获取柱形的高度
#         if height > 0:  # 只显示大于零的柱子
#             # 将高度转换为百分比并格式化为两位小数
#             percentage = height * 100
#             ax.text(p.get_x() + p.get_width() / 2, height, f'{percentage:.2f}%',  # 显示百分比
#                     ha='center', va='bottom', fontsize=13)

ax1.legend(prop={'size': font_size}, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
ax2.legend(prop={'size': font_size}, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
plt.tight_layout()
plt.savefig('t3_bma_success_rate.png')


