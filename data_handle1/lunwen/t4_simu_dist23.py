import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

plt.rcParams['figure.dpi'] = 500

import pandas as pd
# sns.set(font_scale=1.3)

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 8))
# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

lines1= [line.strip().split(' ') for line in open('distributed.txt', 'r')]


# transposed_list_irregular = [
#     [row[i] for row in lines if i < len(row)]
#     for i in range(len(lines[0])) if all(i < len(row) for row in lines) or any(row[i:i+1] for row in lines)  # 这个条件可以简化，但为了清晰保留
# ]
# 注意：上面的条件判断可能不是最优的，因为它试图处理所有情况，但在某些情况下可能会失败（例如，当原始列表为空或某些行完全为空时）。
# 一个更简单但可能不是最优的方法是：
lines = []
for i in range(max(len(row) for row in lines1)):
    column = [row[i] for row in lines1 if i < len(row)]
    if column:  # 只添加非空列
        lines.append(column)
font_size=17
def plotalldata(data,xlabel,ax):
    cluster_sizes = [int(n) for n in data]
    counts, bins = np.histogram(cluster_sizes, bins=range(min(cluster_sizes), max(cluster_sizes) + 2))

    # 计算占比
    total_count = len(cluster_sizes)
    percentages = counts / total_count
    # print(f"counts:{counts},bins:{bins},total_count:{total_count},percentages:{percentages}")
    # 绘制占比直方图
    sns.barplot(x=bins[:-1], y=percentages, ax=ax)

    ax.set_xlabel(xlabel, fontsize=font_size)
    ax.set_ylabel('Frequency', fontsize=font_size)
    ax.tick_params(axis='x', labelsize=font_size)
    ax.tick_params(axis='y', labelsize=font_size)
    num_ticks = 6
    tick_indices = np.linspace(0, len(bins) - 2, num_ticks, dtype=int)
    tick_labels = [bins[i] for i in tick_indices]
    ax.set_xticks([bins[i] for i in tick_indices])
    ax.set_xticklabels(tick_labels)


# print(f"{len(lines)}")
# 将字段列表转换为DataFrame，假设我们知道每行有3个字段（根据你的数据调整）
# 这里我们使用列表推导式中的结果作为DataFrame的行，并为列指定列名
df = pd.DataFrame(lines, columns=['Synthesis', 'Decay', 'PCR', 'Sampling','Illumina', 'Nanopone'])

# cluster_sizes = df['DNAfountain']

# sns.histplot(xx, kde=True, bins=range(min(cluster_sizes), max(cluster_sizes) + 2),ax=axes[0,0])
# sns.histplot(cluster_sizes, bins='auto',ax=axes[0,0])
# sns.histplot(cluster_sizes, kde=True, bins=range(min(cluster_sizes), max(cluster_sizes) + 2)/sum(cluster_sizes),ax=axes[0,0])
# sns.histplot(xs, kde=True, bins=range(min(cluster_sizes), max(cluster_sizes) + 2),ax=axes[0,0])
# sns.histplot(cluster_sizes, kde=True, bins='auto',ax=axes[0,0])

plotalldata(df['Synthesis'],'Synthesis',axes[0,0])
plotalldata(df['Decay'],'Decay',axes[0,1])
plotalldata(df['PCR'],'PCR',axes[0,2])
plotalldata(df['Sampling'],'Sampling',axes[1,0])
plotalldata(df['Illumina'],'Illumina Sequencing',axes[1,1])
plotalldata(df['Nanopone'],'Nanopone Sequencing',axes[1,2])
import matplotlib as mpl
#
# cluster_sizes = [int(n) for n in df['Synthesis']]
# sns.histplot(cluster_sizes, kde=True, bins=range(min(cluster_sizes), max(cluster_sizes) + 2),ax=axes[0,0])
# axes[0,0].set_xlabel('Synthesis')
# axes[0,0].set_ylabel('Count')
# cluster_sizes = [int(n) for n in df['Decay']]
# sns.histplot(cluster_sizes, kde=True, bins=range(min(cluster_sizes), max(cluster_sizes) + 2),ax=axes[0,1])
# axes[0,1].set_xlabel('Decay')
# axes[0,1].set_ylabel('Count')


sns.set_palette("deep")
palette = sns.color_palette("deep")
first_color = palette[0]
cluster_sizes = [int(n) for n in df['PCR']]
sns.histplot(cluster_sizes, bins=range(min(cluster_sizes), max(cluster_sizes) + 2),ax=axes[0,2],color=first_color)
axes[0,2].set_xlabel('PCR', fontsize=font_size)
axes[0,2].set_ylabel('Count', fontsize=font_size)
# cluster_sizes = [int(n) for n in df['Sampling']]
# sns.histplot(cluster_sizes, kde=True, bins=range(min(cluster_sizes), max(cluster_sizes) + 2),ax=axes[1,0])
# axes[1,0].set_xlabel('Sampling')
# axes[1,0].set_ylabel('Count')
# cluster_sizes = [int(n) for n in df['Illumina']]
# sns.histplot(cluster_sizes, kde=True, bins=range(min(cluster_sizes), max(cluster_sizes) + 2),ax=axes[1,1])
# axes[1,1].set_xlabel('Illumina Sequencing')
# axes[1,1].set_ylabel('Count')
# cluster_sizes = [int(n) for n in df['Nanopone']]
# sns.histplot(cluster_sizes, kde=True, bins=range(min(cluster_sizes), max(cluster_sizes) + 2),ax=axes[1,2])
# axes[1,2].set_xlabel('Nanopone Sequencing')
# axes[1,2].set_ylabel('Count')
plt.tight_layout()
plt.savefig('t4_simu_distri.png')


