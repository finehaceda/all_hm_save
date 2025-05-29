

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False

import pandas as pd


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
plt.rcParams['figure.dpi'] = 500

df = pd.read_excel('plot.xlsx', sheet_name='t3_ablation2')
df.to_csv('plot.csv', index=False, encoding='utf-8')
df = pd.read_csv('plot.csv', encoding='utf-8')

# fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
# plt.figure(figsize=(10, 8))
# fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # 2行2列的子图布局
# ax = sns.barplot(x='redundancy', y='r_success', hue="method", data=df)
# ax = sns.barplot(x='error', y='e_success', hue="method", data=df, errorbar="sd")
# ax1 = sns.barplot(x='r_success', y='redundancy', hue="method", data=df, orient='h',gap=0.2, width=0.7, ax=axes[0])
# ax2 = sns.barplot(x='e_success', y='error', hue="method", data=df, orient='h',gap=0.2, width=0.7, ax=axes[1])
ax1 = sns.barplot(x='model', y='seq_rate', data=df,gap=0.2, width=0.5, ax=axes[0],color='#3274A1')
# ax2 = sns.barplot(x='model', y='base_rate', data=df, gap=0.2, width=0.5, ax=axes[1],color='#ff7f0e')
ax2 = sns.barplot(x='model', y='base_rate', data=df, gap=0.2, width=0.5, ax=axes[1],color='#E1812C')
# ax = sns.barplot(x='r_success', y='redundancy', hue='method', data=df, orient='h')

# for p in ax1.patches:
#     width = p.get_width()
#     if width > 0:  # 过滤掉宽度为 0 的柱子
#         ax1.annotate(f'{width * 100:.2f}%', (width, p.get_y() + p.get_height() / 2.),
#                     ha='left', va='center', fontsize=11, color='black', xytext=(5, 0),
#                     textcoords='offset points')
#
# for p in ax2.patches:
#     width = p.get_width()
#     if width > 0:  # 过滤掉宽度为 0 的柱子
#         ax2.annotate(f'{width * 100:.2f}%', (width, p.get_y() + p.get_height() / 2.),
#                     ha='left', va='center', fontsize=11, color='black', xytext=(5, 0),
#                     textcoords='offset points')



font_size = 14
axes[0].tick_params(axis='x', labelsize=font_size, labelrotation=45)
axes[0].tick_params(axis='y', labelsize=font_size)
ax1.set_xlabel('',)
ax1.set_ylabel('seq recovery rate', fontsize=font_size)
ax1.set_ylim(0.7,0.87)
# ax1.legend(prop={'size': font_size}, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)


axes[1].tick_params(axis='x', labelsize=font_size, labelrotation=45)
axes[1].tick_params(axis='y', labelsize=font_size)
ax2.set_xlabel('',)
ax2.set_ylabel('base recovery rate', fontsize=font_size)
ax2.set_ylim(0.9985,0.9993)
# ax2.legend(prop={'size': font_size}, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)

plt.tight_layout()
plt.savefig('t3_ablation2.png')
# plt.savefig('t4_fountain_err.png')


