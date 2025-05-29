import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.cm as cm

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False

sns.set_style("whitegrid")
import pandas as pd


# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
plt.rcParams['figure.dpi'] = 500

df = pd.read_excel('plot.xlsx', sheet_name='decodetime')
df.to_csv('plot.csv', index=False, encoding='utf-8')
df = pd.read_csv('plot.csv', encoding='utf-8')

# palette=pastel, muted, coolwarm, Blues
color_list = ['red', 'blue', 'green', 'orange', 'purple']
# fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
# plt.figure(figsize=(10, 8))
# fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # 2行2列的子图布局
# ax = sns.barplot(x='redundancy', y='r_success', hue="method", data=df)
# ax = sns.barplot(x='error', y='e_success', hue="method", data=df, errorbar="sd")
# ax = sns.barplot(x='time', y='method', data=df, orient='h',gap=0.2, width=0.7)
# ax = sns.barplot(x='time', y='method', data=df, orient='h', width=0.6)
ax = sns.barplot(x='time', y='method', data=df, orient='h', width=0.6, palette="muted")
# ax = sns.barplot(x='e_success', y='error', hue="method", data=df, orient='h',gap=0.2, width=0.7)
# ax = sns.barplot(x='r_success', y='redundancy', hue='method', data=df, orient='h')

# for p in ax.patches:
#     width = p.get_width()
#     if width > 0:  # 过滤掉宽度为 0 的柱子
#         ax.annotate(f'{width * 100:.2f}%', (width, p.get_y() + p.get_height() / 2.),
#                     ha='left', va='center', fontsize=11, color='black', xytext=(5, 0),
#                     textcoords='offset points')


# plt.xlim(0,1.05)
# plt.xlim(0,1.11)
font_size = 13
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)
ax.set_xlabel('average decode time(s)', fontsize=font_size)
ax.set_ylabel('method', fontsize=font_size)
# ax.set_xlabel('bit recovery rate', fontsize=font_size)
# ax.set_ylabel('error rate', fontsize=font_size)
# ax.legend(fontsize=font_size)
# plt.title('Redundancy vs. Success Rate by Method', fontsize=font_size)

# legend = ax.legend(prop={'size': font_size}, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)

# 调整图形布局，为图例留出空间
# plt.subplots_adjust(bottom=0.25)
# plt.tight_layout()
# plt.xlabel('bit recovery rate')
# plt.ylabel('redundancy')
# y_min, y_max = df['success'].min(), df['success'].max()
# axes[0].set_title('success rate')
# plt.ylim(0.4,1)
# plt.ylim(0.99, 1.02)
# ax.set_ylim(y_min-0.1, y_max + 0.1)
# sns.displot(x='data', y='rev', hue="method", data=df, ax=axes[1])
# axes[1].set_title('base recovery rate')


plt.tight_layout()
# plt.savefig('t4-1.png')
# plt.savefig('t4-3.png')
plt.savefig('t4_decode_time.png')
# plt.savefig('t4_fountain_err.png')


