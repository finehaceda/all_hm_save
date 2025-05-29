

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False

sns.set_style("whitegrid")
import pandas as pd

plt.rcParams['figure.dpi'] = 500

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(24, 8))

# plt.figure(figsize=(12, 6))
df = pd.read_excel('plot.xlsx', sheet_name='endecode_rate')
df.to_csv('plot.csv', index=False, encoding='utf-8')
df = pd.read_csv('plot.csv', encoding='utf-8')



# ax1 = sns.barplot(x='data', y='success', hue="method", data=df[0:4], ax=axes[0])
# ax1.set_ylim(0.96,1.01)
# ax1 = sns.barplot(x='method', y='encode_rate', data=df, ax=axes[0])
ax1 = sns.barplot(x='method', y='simulate_time', data=df, ax=axes[0])
ax1.set_ylim(0.012,0.015)
# axes[0].set_xlabel('')
# axes[0].set_ylabel('success rate')

font_size = 22
ax1.tick_params(axis='x',rotation=45, labelsize=font_size)
ax1.tick_params(axis='y', labelsize=font_size)
ax1.set_xlabel('', fontsize=font_size)
ax1.set_ylabel('simulate cost time(s/read)', fontsize=font_size)


# ax2 = sns.barplot(x='data', y='success', hue="method", data=df[4:8], ax=axes[1])
# ax2.set_ylim(0.83,1.01)
# ax2 = sns.barplot(x='decode_rate', hue='method', orient='h',legend=False, data=df, ax=axes[1],width=0.5,palette="muted")
# ax2 = sns.barplot(x='method', y='decode_rate', data=df, ax=axes[1])
ax2 = sns.barplot(x='method', y='cluster_time', data=df, ax=axes[1])
# ax2.set_xlim(0.83,1.01)
ax2.set_ylim(0.012,0.017)
ax2.tick_params(axis='x', rotation=45, labelsize=font_size)
ax2.tick_params(axis='y', labelsize=font_size)
ax2.set_xlabel('', fontsize=font_size)
ax2.set_ylabel('cluster cost time(s/read)', fontsize=font_size)


ax3 = sns.barplot(x='method', y='reconstruct_time', data=df, ax=axes[2])
# ax2.set_xlim(0.83,1.01)
ax3.set_ylim(0.036,0.042)
ax3.tick_params(axis='x', rotation=45, labelsize=font_size)
ax3.tick_params(axis='y', labelsize=font_size)
ax3.set_xlabel('', fontsize=font_size)
ax3.set_ylabel('reconstruct cost time(s/read)', fontsize=font_size)


# axa = [ax1,ax2]
# for ax in axa:
#     for p in ax.patches:
#         height = p.get_height()  # 获取柱形的高度
#         if height > 0:  # 只显示大于零的柱子
#             # 将高度转换为百分比并格式化为两位小数
#             percentage = height * 100
#             ax.text(p.get_x() + p.get_width() / 2, height, f'{percentage:.2f}%',  # 显示百分比
#                     ha='center', va='bottom', fontsize=13)


# ax1.legend(prop={'size': font_size}, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
# ax2.legend(prop={'size': font_size}, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)

plt.tight_layout()
# plt.savefig('endecode_rate.png')
plt.savefig('cost_time_Illumina.png')


