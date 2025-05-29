

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False

plt.rcParams['figure.dpi'] = 300
import pandas as pd
sns.set_style("whitegrid")
# sns.set_style("white")  # 只设置背景为白色
# data = sns.load_dataset("tips")
# g = sns.barplot(x="day", y="total_bill", data=data)

# 只显示 x 轴网格
plt.grid(axis='y')  # 设置仅显示 x 轴的网格
# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

plt.grid()
# df = pd.read_excel('plot.xlsx', sheet_name='t4_density')
# df.to_csv('plot.csv', index=False, encoding='utf-8')
# df = pd.read_csv('plot.csv', encoding='utf-8')

# fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
# plt.figure(figsize=(10, 8))
# fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # 2行2列的子图布局
# ax = sns.scatterplot(x='file', y='density', hue="method", data=df)
# ax = sns.scatterplot(x='method1', y='density1', hue="file_size", data=df, marker='D')
# ax = sns.scatterplot(x='method1', y='density_Illumina', hue="file_size", data=df, marker='D')

color_list = ['red', 'blue', 'green', 'orange', 'purple']
df = pd.read_excel('plot.xlsx', sheet_name='t4_density')
df.to_csv('plot.csv', index=False, encoding='utf-8')
df = pd.read_csv('plot.csv', encoding='utf-8')
ax = sns.barplot(x='avg_density3', y='avg_method', data=df, orient='h', width=0.6, palette="muted")


# ax = sns.barplot(x='method', y='success', hue="method", data=df)
ax.set_xlabel('')
# ax2 = sns.barplot(x='data', y='success', hue="method", data=df, ax=axes[0,0])
font_size = 17
for p in ax.patches:
    width = p.get_width()
    if width > 0:  # 过滤掉宽度为 0 的柱子
        ax.annotate(f'{width:.2f}', (width, p.get_y() + p.get_height() / 2.),
                    ha='left', va='center', fontsize=font_size, color='black', xytext=(5, 0),
                    textcoords='offset points')

# plt.tight_layout()


plt.ylabel('average density(bits/nt)', fontsize=font_size)
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)


# y_min, y_max = df['success'].min(), df['success'].max()
# axes[0].set_title('success rate')
plt.xlim(0,1.8)
# plt.ylim(0.99, 1.02)
# ax.set_ylim(y_min-0.1, y_max + 0.1)
# sns.displot(x='data', y='rev', hue="method", data=df, ax=axes[1])
# axes[1].set_title('base recovery rate')



# legend = ax.legend(prop={'size': 17}, loc='upper center', bbox_to_anchor=(0.5, -0.27), ncol=2)
legend = ax.legend(prop={'size': 17})

# 调整图形布局，为图例留出空间
# plt.subplots_adjust(bottom=0.3)


plt.xticks(rotation=45)
plt.tight_layout()
# plt.savefig('t4_density.png')
# plt.savefig('t4_avg_density_Illumina.png')
plt.savefig('t4_avg_density_nanopone.png')


