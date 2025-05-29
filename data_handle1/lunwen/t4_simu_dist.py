

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False

import pandas as pd
# sns.set(font_scale=1.5)

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 16))
# fig, axes = plt.subplots(nrows=3, ncols=2)
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



# print(f"{len(lines)}")
# 将字段列表转换为DataFrame，假设我们知道每行有3个字段（根据你的数据调整）
# 这里我们使用列表推导式中的结果作为DataFrame的行，并为列指定列名
df = pd.DataFrame(lines, columns=['DNAfountain', 'YYC', 'derrick', 'PolarCode', 'Hedges'])

# df = pd.read_excel('plot.xlsx', sheet_name='distribt')
# df.to_csv('plot.csv', index=False, encoding='utf-8')
# df = pd.read_csv('plot.csv', encoding='utf-8')

# cluster_sizes = df['DNAfountain']
cluster_sizes = [int(n) for n in df['DNAfountain']]
sns.histplot(cluster_sizes, kde=True, bins=range(min(cluster_sizes), max(cluster_sizes) + 2),ax=axes[0,0])
axes[0,0].set_xlabel('Synthesis')
axes[0,0].set_ylabel('Frequency')
cluster_sizes = df['YYC']
cluster_sizes = [int(n) for n in df['YYC']]
sns.histplot(cluster_sizes, kde=True, bins=range(min(cluster_sizes), max(cluster_sizes) + 2),ax=axes[0,1])
axes[0,1].set_xlabel('Decay')
axes[0,1].set_ylabel('Frequency')
cluster_sizes = df['derrick']
cluster_sizes = [int(n) for n in df['derrick']]
sns.histplot(cluster_sizes, kde=True, bins=range(min(cluster_sizes), max(cluster_sizes) + 2),ax=axes[1,0])
axes[1,0].set_xlabel('PCR')
axes[1,0].set_ylabel('Frequency')
cluster_sizes = df['PolarCode']
cluster_sizes = [int(n) for n in df['PolarCode']]
sns.histplot(cluster_sizes, kde=True, bins=range(min(cluster_sizes), max(cluster_sizes) + 2),ax=axes[1,1])
axes[1,1].set_xlabel('Sampling')
axes[1,1].set_ylabel('Frequency')
cluster_sizes = df['Hedges']
cluster_sizes = [int(n) for n in df['Hedges']]
sns.histplot(cluster_sizes, kde=True, bins=range(min(cluster_sizes), max(cluster_sizes) + 2),ax=axes[2,0])
axes[2,0].set_xlabel('Sequencing')
axes[2,0].set_ylabel('Frequency')
fig.delaxes(axes[2,1])
plt.tight_layout()
plt.savefig('t4_simu_distri.png')


