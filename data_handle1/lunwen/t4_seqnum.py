

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib import ticker

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False

# sns.set_style("whitegrid")
sns.set_style("whitegrid")
import pandas as pd

plt.rcParams['figure.dpi'] =500

# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))


df = pd.read_excel('plot.xlsx', sheet_name='t4_density')
df.to_csv('plot.csv', index=False, encoding='utf-8')
df = pd.read_csv('plot.csv', encoding='utf-8')

# fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
# plt.figure(figsize=(10, 8))
# fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # 2行2列的子图布局
# ax = sns.scatterplot(x='file', y='density', hue="method", data=df)
ax = sns.barplot(x='file', y='seq_num', hue="method", data=df,gap=0.2)
# ax = sns.barplot(x='method', y='success', hue="method", data=df)
ax.set_xlabel('')
# ax2 = sns.barplot(x='data', y='success', hue="method", data=df, ax=axes[0,0])


# for p in ax.patches:
#     height = p.get_height()  # 获取柱形的高度
#     if height > 0:  # 只显示大于零的柱子
#         # 将高度转换为百分比并格式化为两位小数
#         percentage = height * 100
#         ax.text(p.get_x() + p.get_width() / 2, height, f'{percentage:.2f}%',  # 显示百分比
#                 ha='center', va='bottom', fontsize=12)
# plt.tight_layout()
# y_min, y_max = df['success'].min(), df['success'].max()
# axes[0].set_title('success rate')
# plt.ylim(0.99,1)
# plt.ylim(0.99, 1.02)
# ax.set_ylim(y_min-0.1, y_max + 0.1)
# sns.displot(x='data', y='rev', hue="method", data=df, ax=axes[1])
# axes[1].set_title('base recovery rate')

# def scientific_formatter(val, pos):
#     template = "{x:1.1e}"  # 1.1e表示保留一位小数，并显示为科学计数法
#     return template.format(x=val)


# 设置y轴的刻度标签格式器为自定义的科学计数法格式器
# ax.yaxis.set_major_formatter(ticker.FuncFormatter(scientific_formatter))
# ax.set_ylabel('Units (e.g., V/m)', loc='top')

font_size=11
plt.ylabel('dna sequence nums',fontsize=font_size)
plt.xticks(rotation=45,fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.tight_layout()
plt.savefig('t4_seqnum1.png')

# plt.subplots_adjust(bottom=0.8)

