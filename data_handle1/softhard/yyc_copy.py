

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False

import pandas as pd

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 500
# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))


df = pd.read_excel('plot.xlsx', sheet_name='yyc')
df.to_csv('plot.csv', index=False, encoding='utf-8')
df = pd.read_csv('plot.csv', encoding='utf-8')

# fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
# plt.figure(figsize=(10, 8))
# fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # 2行2列的子图布局
# ax = sns.barplot(x='redundancy', y='r_success', hue="method", data=df)
# ax = sns.barplot(x='error', y='e_success', hue="method", data=df, errorbar="sd")

# ax = sns.barplot(x='copy', y='bitrev', hue="method", data=df)
ax = sns.barplot(x='copy', y='recovery1', hue="method1", data=df,gap=0.1)
# ax = sns.catplot(
#     df, kind="bar",
#     x="copy", y="recovery", hue="method", col="species",
#     height=4, aspect=.5,
# )
# for p in ax.patches:
#     height = p.get_height()  # 获取柱形的高度
#     if height > 0:  # 只显示大于零的柱子
#         # 将高度转换为百分比并格式化为两位小数
#         percentage = height * 100
#         ax.text(p.get_x() + p.get_width() / 2, height, f'{percentage:.2f}%',  # 显示百分比
#                 ha='center', va='bottom', fontsize=12)
# plt.tight_layout()


font_size = 13
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)
ax.set_xlabel('copy', fontsize=font_size)
ax.set_ylabel('bit recovery rate', fontsize=font_size)
# ax.legend(fontsize=10)
# legend = ax.legend(prop={'size': 11})

legend = ax.legend(prop={'size': 11}, loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=2)

# 调整图形布局，为图例留出空间
plt.subplots_adjust(bottom=0.25)
# plt.ylabel('bit recovery',fontsize=15)
# plt.xlabel('copy',fontsize=15)
# y_min, y_max = df['success'].min(), df['success'].max()
# axes[0].set_title('success rate')
# plt.ylim(0.4,1)
# plt.ylim(0.99, 1.02)
# ax.set_ylim(y_min-0.1, y_max + 0.1)
# sns.displot(x='data', y='rev', hue="method", data=df, ax=axes[1])
# axes[1].set_title('base recovery rate')






# plt.tight_layout()
# plt.savefig('t4-1.png')
plt.savefig('t4_yyc_copy.png')


