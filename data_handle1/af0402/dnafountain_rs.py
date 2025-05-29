

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False

import pandas as pd

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 500
# plt.rcParams['figure.dpi'] = 300
# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))


df = pd.read_excel('plot.xlsx', sheet_name='fountain_rs')
df.to_csv('plot.csv', index=False, encoding='utf-8')
df = pd.read_csv('plot.csv', encoding='utf-8')

# fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
# plt.figure(figsize=(10, 8))
# fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # 2行2列的子图布局
# ax = sns.barplot(x='redundancy', y='r_success', hue="method", data=df)
# ax = sns.barplot(x='error', y='e_success', hue="method", data=df, errorbar="sd")
# ValueError: Invalid `kind`: 'line'. Options are 'strip', 'swarm', 'box', 'boxen', 'violin', 'bar', 'count', and 'point'.
# ax = sns.barplot(x='copy', y='bitrev', hue="method", data=df)
# ax = sns.barplot(x='copy', y='recovery', hue="method", data=df ,gap=0.1)
g = sns.catplot(
    df, kind="point",
    x="copy", y="recovery", hue="method", col="rs",
    # height=4, aspect=0.8,gap=0.1
# , legend=False
)
# g.legend()
# g.add_legend()

font_size = 15
# plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)
g.set_xlabels('copy', fontsize=font_size)
g.set_ylabels('recovery rate', fontsize=font_size)
g.tick_params(axis='x', labelsize=font_size)
g.tick_params(axis='y', labelsize=font_size)
# ax.legend(fontsize=10)
# legend = g.legend(prop={'size': 11})
# g.add_legend(legend_data=g.legend, title='Method', loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=3)

# 调整子图布局，为图例留出空间
# plt.subplots_adjust(bottom=0.3)

for ax in g.axes.flatten():
    ax.set_title(ax.get_title(), fontsize=font_size)  # 将 14 替换为你想要的字体大小

plt.savefig('dnafountain_rs_0410.png')


