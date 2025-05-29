

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False

import pandas as pd


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

# plt.figure(figsize=(12, 6))
df = pd.read_excel('plot.xlsx', sheet_name='t3_4')
df.to_csv('plot.csv', index=False, encoding='utf-8')
df = pd.read_csv('plot.csv', encoding='utf-8')


#
#
# # fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
# # plt.figure(figsize=(10, 8))
# fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # 2行2列的子图布局
# ax1 = sns.barplot(x='method', y='success', hue="method", data=df[0:4], ax=axes[0,0])
# axes[0,0].set_xlabel('Organick')
# ax1.set_ylim(0.99, 1.0025)
# ax2 = sns.barplot(x='method', y='success', hue="method", data=df[4:8], ax=axes[0,1])
# axes[0,1].set_xlabel('Chandak')
# ax2.set_ylim(0.99, 1.0025)
# ax3 = sns.barplot(x='method', y='success', hue="method", data=df[8:12], ax=axes[1,0])
# axes[1,0].set_xlabel('Srinivasavardhan')
# ax3.set_ylim(0.7, 1)
# ax4 = sns.barplot(x='method', y='success', hue="method", data=df[12:16], ax=axes[1,1])
# axes[1,1].set_xlabel('Ding')
# ax4.set_ylim(0.99, 1.0025)
# # ax = sns.barplot(x='method', y='success', hue="method", data=df)
#
# # ax2 = sns.barplot(x='data', y='success', hue="method", data=df, ax=axes[0,0])
#
# axa = [ax1,ax2,ax3,ax4]
# for ax in axa:
#     for p in ax.patches:
#         height = p.get_height()  # 获取柱形的高度
#         if height > 0:  # 只显示大于零的柱子
#             # 将高度转换为百分比并格式化为两位小数
#             percentage = height * 100
#             ax.text(p.get_x() + p.get_width() / 2, height, f'{percentage:.2f}%',  # 显示百分比
#                     ha='center', va='bottom', fontsize=12)
#
#
# plt.tight_layout()
# plt.savefig('t3-1.png')




# # 填充 NaN 值
# df['copy_1'] = df['copy_1'].fillna(0)  # 填充为0
#
# # 替换 inf 和 -inf 为一个整数（例如0）
# df['copy_1'] = df['copy_1'].replace([float('inf'), float('-inf')], 0)
# df['copy_1'] = df['copy_1'].astype(int)
# sns.barplot(x='copy_1', y='Chandak', data=df[:13], ax=axes[0,0])
sns.lineplot(x='copy_1', y='Chandak', data=df[:13], ax=axes[0,0])
# ax1.set_ylim(0.95,1.01)
axes[0,0].set_xlabel('copy num')
axes[0,0].set_ylabel('edit error rate')
axes[0,0].set_title('Chandak et al.')
axes[0, 0].xaxis.set_major_locator(MaxNLocator(integer=True))
sns.lineplot(x='copy_1', y='Organick', data=df[:13], ax=axes[0,1],color='orange')
# ax1.set_ylim(0.95,1.01)
axes[0,1].set_xlabel('copy num')
axes[0,1].set_ylabel('edit error rate')
axes[0,1].set_title('Organick et al.')
axes[0, 0].xaxis.set_major_locator(MaxNLocator(integer=True))
sns.lineplot(x='copy_2', y='Srinivasavardhan', data=df, ax=axes[1,0],color='#98FF98')
# ax1.set_ylim(0.95,1.01)
axes[1,0].set_xlabel('copy num')
axes[1,0].set_ylabel('edit error rate')
axes[1,0].set_title('Srinivasavardhan et al.')
sns.lineplot(x='copy_2', y='Ding', data=df, ax=axes[1,1],color='#FFB6C1')
# ax2.set_ylim(0.83,1.01)
axes[1,1].set_xlabel('copy num')
axes[1,1].set_ylabel('edit error rate')
axes[1,1].set_title('Ding et al.')

# axa = [ax1,ax2]
# for ax in axa:
#     for p in ax.patches:
#         height = p.get_height()  # 获取柱形的高度
#         if height > 0:  # 只显示大于零的柱子
#             # 将高度转换为百分比并格式化为两位小数
#             percentage = height * 100
#             ax.text(p.get_x() + p.get_width() / 2, height, f'{percentage:.2f}%',  # 显示百分比
#                     ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.savefig('t3-4_merge_line.png')


# ax1 = sns.barplot(x='data', y='rev', hue="method", data=df[0:8], ax=axes[0])
# ax1.set_ylim(0.9997,1.0001)
# axes[0].set_xlabel('')
#
# ax2 = sns.barplot(x='data', y='rev', hue="method", data=df[8:16], ax=axes[1])
# ax2.set_ylim(0.99, 1.0012)
# axes[1].set_xlabel('')
#
# axa = [ax1,ax2]
# for ax in axa:
#     for p in ax.patches:
#         height = p.get_height()  # 获取柱形的高度
#         if height > 0:  # 只显示大于零的柱子
#             # 将高度转换为百分比并格式化为两位小数
#             percentage = height * 100
#             ax.text(p.get_x() + p.get_width() / 2, height, f'{percentage:.4f}%',  # 显示百分比
#                     ha='center', va='bottom', fontsize=12)
#
# plt.tight_layout()
# plt.savefig('t3-1_merge_rev.png')


