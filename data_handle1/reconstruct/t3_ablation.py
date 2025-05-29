

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
# plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
sns.set_style("whitegrid")
import pandas as pd

plt.rcParams['figure.dpi'] = 500


plt.figure(figsize=(10, 5))
df = pd.read_excel('plot.xlsx', sheet_name='t3_ablation')
df.to_csv('plot.csv', index=False, encoding='utf-8')
df = pd.read_csv('plot.csv', encoding='utf-8')



# ax1 = sns.barplot(x='data', y='success', hue="method", data=df[0:4], ax=axes[0])
# ax1.set_ylim(0.96,1.01)
# ax1 = sns.barplot(x='rate', y='model', orient='h', data=df,width=0.5,palette="muted")
# ax1 = sns.barplot(x='rate', y='model', hue="out", orient='h', data=df,width=0.5)
ax1 = sns.barplot(x='rate_0319', y='model', hue="out", orient='h', data=df,width=0.5)
# ax1.set_xlim(0.96,1.01)
# axes[0].set_xlabel('')
# axes[0].set_ylabel('success rate')

font_size = 15
ax1.tick_params(axis='x', labelsize=font_size)
ax1.tick_params(axis='y', labelsize=font_size)
ax1.set_xlabel('', fontsize=font_size)
ax1.set_ylabel('', fontsize=font_size)
ax1.set_xlim(0.8,1.05)
ax1.legend(prop={'size': font_size}, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)




for p in ax1.patches:
    width = p.get_width()
    if width > 0:  # 过滤掉宽度为 0 的柱子
        ax1.annotate(f'{width * 100:.2f}%', (width, p.get_y() + p.get_height() / 2.),
                    ha='left', va='center', fontsize=11, color='black', xytext=(5, 0),
                    textcoords='offset points')


# ax1.legend(prop={'size': font_size}, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
# ax2.legend(prop={'size': font_size}, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)

plt.tight_layout()
plt.savefig('t3_ablation.png')


