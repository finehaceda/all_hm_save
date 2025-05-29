

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False

import pandas as pd
# sns.set(font_scale=1.5)

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 8))
# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))


df = pd.read_excel('plot.xlsx', sheet_name='distribt')
df.to_csv('plot.csv', index=False, encoding='utf-8')
df = pd.read_csv('plot.csv', encoding='utf-8')
font_size=15
# cluster_sizes = df['DNAfountain']
cluster_sizes = [int(n) for n in df['DNAfountain'][:2243]]
sns.histplot(cluster_sizes, kde=True, bins=range(min(cluster_sizes), max(cluster_sizes) + 2),ax=axes[0,0])
axes[0,0].set_xlabel('DNAfountain', fontsize=font_size)
axes[0,0].set_ylabel('Frequency', fontsize=font_size)
cluster_sizes = df['YYC']
cluster_sizes = [int(n) for n in df['YYC'][:2223]]
sns.histplot(cluster_sizes, kde=True, bins=range(min(cluster_sizes), max(cluster_sizes) + 2),ax=axes[0,1])
axes[0,1].set_xlabel('YYC', fontsize=font_size)
axes[0,1].set_ylabel('Frequency', fontsize=font_size)
cluster_sizes = df['derrick']
cluster_sizes = [int(n) for n in df['derrick'][:2295]]
sns.histplot(cluster_sizes, kde=True, bins=range(min(cluster_sizes), max(cluster_sizes) + 2),ax=axes[0,2])
axes[0,2].set_xlabel('derrick', fontsize=font_size)
axes[0,2].set_ylabel('Frequency', fontsize=font_size)
cluster_sizes = df['PolarCode']
cluster_sizes = [int(n) for n in df['PolarCode'][:2815]]
sns.histplot(cluster_sizes, kde=True, bins=range(min(cluster_sizes), max(cluster_sizes) + 2),ax=axes[1,0])
axes[1,0].set_xlabel('PolarCode', fontsize=font_size)
axes[1,0].set_ylabel('Frequency', fontsize=font_size)
cluster_sizes = df['Hedges']
cluster_sizes = [int(n) for n in df['Hedges'][:3315]]
sns.histplot(cluster_sizes, kde=True, bins=range(min(cluster_sizes), max(cluster_sizes) + 2),ax=axes[1,1])
axes[1,1].set_xlabel('Hedges', fontsize=font_size)
axes[1,1].set_ylabel('Frequency', fontsize=font_size)
fig.delaxes(axes[1,2])
plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig('t4_distri.png')


