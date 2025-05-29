

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 500
import pandas as pd
# sns.set(font_scale=1.5)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))


df = pd.read_excel('plot.xlsx', sheet_name='gc')
df.to_csv('plot.csv', index=False, encoding='utf-8')
df = pd.read_csv('plot.csv', encoding='utf-8')



ax1 = sns.violinplot(data=df, ax=axes[0])
# ax1 = sns.boxplot(data=df, ax=axes[0])
# ax1.set_ylim(0.99, 1.0012)
# ax1.set_xticks(rotation=45)
# plt.xticks(rotation=45)
# ax1.tick_params(axis='x', rotation=45)

font_size = 22
axes[0].set_ylabel('GC ratio', fontsize=font_size)
# ax1.ylabel('GC ratio', fontsize=font_size)
# plt.xticks(fontsize=font_size)
# plt.yticks(fontsize=font_size)
axes[0].tick_params(axis='x', rotation=45,labelsize=font_size)
axes[0].tick_params(axis='y',labelsize=font_size)

df = pd.read_excel('plot.xlsx', sheet_name='homo')
df.to_csv('plot.csv', index=False, encoding='utf-8')
df = pd.read_csv('plot.csv', encoding='utf-8')


# ax2 = sns.violinplot(data=df, ax=axes[1])
ax2 = sns.boxplot(data=df, ax=axes[1])
# ax2.set_ylim(0.65, 1)
# ax2.set_ylabel('homopolymers')

# ax2.set_xticks(rotation=45)

axes[1].tick_params(axis='x', rotation=45,labelsize=font_size)
axes[1].tick_params(axis='y',labelsize=font_size)
axes[1].set_ylabel('homopolymers', fontsize=font_size)
# plt.xticks(fontsize=font_size)
# plt.yticks(fontsize=font_size)

# plt.subplots_adjust(bottom=0.2)
# fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
# plt.figure(figsize=(10, 8))
# fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # 2行2列的子图布局
# ax = sns.scatterplot(x='file', y='density', hue="method", data=df)
# ax = sns.boxplot(data=df)
# sns.violinplot(data=df)
# ax = sns.barplot(x='method', y='success', hue="method", data=df)

# ax2 = sns.barplot(x='data', y='success', hue="method", data=df, ax=axes[0,0])


# for p in ax.patches:
#     height = p.get_height()  # 获取柱形的高度
#     if height > 0:  # 只显示大于零的柱子
#         # 将高度转换为百分比并格式化为两位小数
#         percentage = height * 100
#         ax.text(p.get_x() + p.get_width() / 2, height, f'{percentage:.2f}%',  # 显示百分比
#                 ha='center', va='bottom', fontsize=12)
# plt.tight_layout()
# plt.ylabel('信息密度')
# y_min, y_max = df['success'].min(), df['success'].max()
# axes[0].set_title('success rate')
# plt.ylim(0.99,1)
# plt.ylim(0.99, 1.02)
# ax.set_ylim(y_min-0.1, y_max + 0.1)
# sns.displot(x='data', y='rev', hue="method", data=df, ax=axes[1])
# axes[1].set_title('base recovery rate')

# plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('t4_gc_homo.png')


