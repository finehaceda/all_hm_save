
import matplotlib.font_manager as fm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 500

# font_path = '/home1/hongmei/.fonts/ADOBESONGSTD-LIGHT.OTF'
# font_path = '/home1/hongmei/.fonts/SIMSUN.TTC'
font_path = '/home1/hongmei/.fonts/ADOBEHEITISTD-REGULAR.OTF'
prop = fm.FontProperties(fname=font_path)

import pandas as pd
# sns.set(font_scale=1.5)
# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 10))
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))



lines1= [line.strip().split(' ') for line in open('distributed.txt', 'r')]
font_size = 17
lines = []
for i in range(max(len(row) for row in lines1)):
    column = [row[i] for row in lines1 if i < len(row)]
    if column:  # 只添加非空列
        lines.append(column)

def plotalldata(data,xlabel,ax,f=0):
    cluster_sizes = [int(n) for n in data]
    counts, bins = np.histogram(cluster_sizes, bins=range(min(cluster_sizes), max(cluster_sizes) + 2))
    # counts, bins = np.histogram(cluster_sizes, bins=range(min(cluster_sizes), 11))
    # print(f"min(cluster_sizes):{min(cluster_sizes)},max(cluster_sizes):{max(cluster_sizes)}")
    # print(f"bings:{bins}")
    # 计算占比
    total_count = len(cluster_sizes)
    # print(f"bings:{total_count}")
    percentages = counts / total_count
    # print(f"counts:{counts},bins:{bins},total_count:{total_count},percentages:{percentages}")
    # 绘制占比直方图
    sns.barplot(x=bins[:-1], y=percentages, ax=ax)
    # sns.barplot(x=percentages, y=bins[:-1], ax=ax, orient='h')
    # sns.lineplot(x=bins[:-1], y=total_count, ax=ax)
    # ax.tick_params(axis='both', which='major', labelsize=10)  # 'both' 表示 x 轴和 y 轴，'major' 表示主要刻度
    # ax.set_xlabel('X Axis Label', fontsize=16)  # 设置x轴标签字体大小
    ax.tick_params(axis='x', labelsize=font_size)
    ax.tick_params(axis='y', labelsize=font_size)
    # ax.set_xlabel('', fontsize=font_size)
    ax.set_ylabel('Frequency', fontsize=font_size)
    ax.set_xlabel(xlabel, fontproperties=prop, fontsize=25, labelpad=20)
    num_ticks = 6
    tick_indices = np.linspace(0, len(bins) - 2, num_ticks, dtype=int)
    tick_labels = [bins[i] for i in tick_indices]
    ax.set_xticks([bins[i] for i in tick_indices])
    ax.set_xticklabels(tick_labels)
    # ax.set_title(xlabel, fontsize=font_size)
    # if f==1:
    #     num_ticks = 5
    #     tick_indices = np.linspace(0, len(bins) - 2, num_ticks, dtype=int)
    #     # num_ticks = 10
    #     # tick_indices = np.linspace(0,9, num_ticks, dtype=int)
    #     tick_labels = [bins[i] for i in tick_indices]
    #     ax.set_xticks([bins[i] for i in tick_indices])
    #     ax.set_xticklabels(tick_labels)

df = pd.DataFrame(lines, columns=['Synthesis', 'Decay', 'PCR', 'Sampling','Illumina', 'Nanopone'])

plotalldata(df['Illumina'],'(a) Illumina测序每个类数量分布',axes[0,0])
plotalldata(df['Nanopone'],'(b) Nanopone测序每个类数量分布',axes[0,1])







lines1= [line.strip().split(' ') for line in open('edit_dis.txt', 'r')]
lines = []
for i in range(max(len(row) for row in lines1)):
    column = [row[i] for row in lines1 if i < len(row)]
    if column:  # 只添加非空列
        lines.append(column)
# print(f"lines:{len(lines)}")

df = pd.DataFrame(lines, columns=['Synthesis', 'Decay', 'PCR', 'Sampling','Illumina', 'Nanopore'])
first_none_index = df[df['Illumina'].isna()].index.min()
first_none_index1 = df[df['Nanopore'].isna()].index.min()
plotalldata(df['Illumina'][:first_none_index],'(c) Illumina测序剩余编辑错误分布',axes[1,0])
plotalldata(df['Nanopore'][:first_none_index1],'(d) Nanopore测序剩余编辑错误分布',axes[1,1],1)



#
# #图4 homo
# df = pd.read_excel('plot.xlsx', sheet_name='homo')
# df.to_csv('plot.csv', index=False, encoding='utf-8')
# df = pd.read_csv('plot.csv', encoding='utf-8')
#
# ax2 = sns.boxplot(data=df, ax=axes[1,1])
#
# # axes[1,1].tick_params(axis='x', rotation=45,labelsize=font_size)
# axes[1,1].tick_params(axis='x',labelsize=font_size)
# axes[1,1].tick_params(axis='y',labelsize=font_size)
# axes[1,1].set_ylabel('', fontsize=font_size)
# axes[1,1].set_xlabel('(d) 最大均聚物对比', fontproperties=prop, fontsize=25, labelpad=20)





plt.tight_layout()
plt.savefig('t4_simu_merge.png')


