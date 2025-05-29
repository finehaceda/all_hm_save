import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False

# sns.set_style("whitegrid")

import pandas as pd

plt.rcParams['figure.dpi'] = 800


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

# plt.figure(figsize=(12, 6))
df = pd.read_excel('plot.xlsx', sheet_name='t3_copy_edit')
df.to_csv('plot.csv', index=False, encoding='utf-8')
df = pd.read_csv('plot.csv', encoding='utf-8')


sns.lineplot(x='copy_1', y='error_rate_1',hue='dataset_1', data=df, ax=axes[0])

font_size = 20
axes[0].tick_params(axis='x', labelsize=font_size)
axes[0].tick_params(axis='y', labelsize=font_size)
axes[0].set_xlabel('copy num', fontsize=font_size)
axes[0].set_ylabel('edit error rate', fontsize=font_size)
axes[0].grid(axis='y', color='gray', linestyle='--', linewidth=0.5)
axes[0].set_xticks(np.arange(df['copy_1'].min(), df['copy_1'].max() + 1, 1))
axes[0].legend(prop={'size': font_size}, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)


sns.lineplot(x='copy_2', y='error_rate_2',hue='dataset_2', data=df, ax=axes[1])

font_size = 20
axes[1].tick_params(axis='x', labelsize=font_size)
axes[1].tick_params(axis='y', labelsize=font_size)
axes[1].set_xlabel('copy num', fontsize=font_size)
axes[1].set_ylabel('edit error rate', fontsize=font_size)
axes[1].set_xticks(np.arange(df['copy_2'].min(), df['copy_2'].max() + 1, 1))
axes[1].grid(axis='y', color='gray', linestyle='--', linewidth=0.5)
axes[1].legend(prop={'size': font_size}, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)

# plt.xlabel('copy num', fontsize=font_size)
# plt.ylabel('edit error rate', fontsize=font_size)
# # plt.xticks(fontsize=font_size)
# plt.yticks(fontsize=font_size)
# plt.xticks(fontsize=font_size,ticks=range(df['copy'].min(), df['copy'].max() + 1))
# plt.grid(axis='y', color='gray', linestyle='--', linewidth=0.5)

# plt.legend(prop={'size': font_size}, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)


plt.tight_layout()
plt.savefig('t3_copy_edit.png')

