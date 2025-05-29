

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


df = pd.read_excel('plot.xlsx', sheet_name='decodetime')
df.to_csv('plot.csv', index=False, encoding='utf-8')
df = pd.read_csv('plot.csv', encoding='utf-8')
# ax = sns.barplot(x='method2', y='spendtime2', hue="part2", data=df)
ax = sns.barplot(x='part2', y='spendtime2', hue="method2", data=df)


font_size = 13
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)
ax.set_xlabel('', fontsize=font_size)
ax.set_ylabel('Illumina decode time', fontsize=font_size)
legend = ax.legend(prop={'size': 11}, loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=2)

# 调整图形布局，为图例留出空间
plt.subplots_adjust(bottom=0.25)

plt.tight_layout()
plt.savefig('Illumina_decode_time.png')


#nanopone decode time

df = pd.read_excel('plot.xlsx', sheet_name='decodetime')
df.to_csv('plot.csv', index=False, encoding='utf-8')
df = pd.read_csv('plot.csv', encoding='utf-8')
# ax = sns.barplot(x='method3', y='spendtime3', hue="part3", data=df)
ax = sns.barplot(x='part3', y='spendtime3', hue="method3", data=df)


font_size = 13
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)
ax.set_xlabel('', fontsize=font_size)
ax.set_ylabel('Nanopone decode time', fontsize=font_size)
legend = ax.legend(prop={'size': 11}, loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=2)

# 调整图形布局，为图例留出空间
# plt.subplots_adjust(bottom=0.25,top=0.02)

plt.tight_layout()
plt.savefig('Nanopone_decode_time.png')