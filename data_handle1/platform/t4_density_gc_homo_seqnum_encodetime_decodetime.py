
import matplotlib.font_manager as fm
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
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(24, 12))

font_size = 17

#图1 density
df = pd.read_excel('plot.xlsx', sheet_name='t4_density')
df.to_csv('plot.csv', index=False, encoding='utf-8')
df = pd.read_csv('plot.csv', encoding='utf-8')

# axes[0,0].set_ylabel('Density(bits/nt)', fontsize=font_size)
sns.scatterplot(x='method', y='density', hue="file", data=df, marker='D', ax=axes[0,0])
axes[0,0].set_ylabel('', fontsize=font_size)
axes[0,0].set_xlabel('（a）信息密度对比', fontproperties=prop, fontsize=25, labelpad=20)
axes[0,0].tick_params(axis='x',labelsize=font_size)
axes[0,0].tick_params(axis='y',labelsize=font_size)
axes[0,0].legend(prop={'size': 15})
axes[0,0].grid(axis='x')

#图2 encodetime
color_list = ['red', 'blue', 'green', 'orange', 'purple']
df = pd.read_excel('plot.xlsx', sheet_name='encodetime')
df.to_csv('plot.csv', index=False, encoding='utf-8')
df = pd.read_csv('plot.csv', encoding='utf-8')
ax = sns.barplot(x='time', y='method', data=df, orient='h', width=0.6, palette="muted", ax=axes[0,1])


axes[0,1].tick_params(axis='x',labelsize=font_size)
axes[0,1].tick_params(axis='y',labelsize=font_size)
# axes[2,0].set_xlabel('average decode time(s)', fontsize=font_size)
axes[0,1].set_xlabel('（b）平均编码时间对比', fontproperties=prop, fontsize=25, labelpad=20)
axes[0,1].set_ylabel('', fontsize=font_size)


#图2 decodetime
color_list = ['red', 'blue', 'green', 'orange', 'purple']
df = pd.read_excel('plot.xlsx', sheet_name='decodetime')
df.to_csv('plot.csv', index=False, encoding='utf-8')
df = pd.read_csv('plot.csv', encoding='utf-8')
sns.barplot(x='time0503', y='method', data=df, orient='h', width=0.6, palette="muted", ax=axes[0,2])


axes[0,2].tick_params(axis='x',labelsize=font_size)
axes[0,2].tick_params(axis='y',labelsize=font_size)
# axes[2,0].set_xlabel('average decode time(s)', fontsize=font_size)
axes[0,2].set_xlabel('（c）平均解码时间对比', fontproperties=prop, fontsize=25, labelpad=20)
axes[0,2].set_ylabel('', fontsize=font_size)

#
# #图2 seqnum
# df = pd.read_excel('plot.xlsx', sheet_name='t4_density')
# df.to_csv('plot.csv', index=False, encoding='utf-8')
# df = pd.read_csv('plot.csv', encoding='utf-8')
# ax = sns.barplot(x='file', y='seq_num', hue="method", data=df,gap=0.2, ax=axes[0,1])
# ax.set_xlabel('')
#
# # axes[0,1].tick_params(axis='x', rotation=45,labelsize=font_size)
# # axes[0,1].set_ylabel('dna sequence nums',fontsize=font_size)
# axes[0,1].set_xlabel('(b) 序列数量对比', fontproperties=prop, fontsize=25, labelpad=20)
# axes[0,1].set_ylabel('',fontsize=font_size)
# axes[0,1].tick_params(axis='x',labelsize=font_size)
# axes[0,1].tick_params(axis='y',labelsize=font_size)
#
# # axes[1,0].legend(prop={'size': 13}, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
# axes[0,1].legend(prop={'size': 15})



#图3 gc含量
df = pd.read_excel('plot.xlsx', sheet_name='gc')
df.to_csv('plot.csv', index=False, encoding='utf-8')
df = pd.read_csv('plot.csv', encoding='utf-8')

ax1 = sns.violinplot(data=df, ax=axes[1,0])

# axes[1,0].set_ylabel('GC ratio', fontsize=font_size)
# axes[1,0].tick_params(axis='x', rotation=45,labelsize=font_size)
axes[1,0].set_xlabel('（d）GC含量对比', fontproperties=prop, fontsize=25, labelpad=20)
axes[1,0].set_ylabel('', fontsize=font_size)
axes[1,0].tick_params(axis='x',labelsize=font_size)
axes[1,0].tick_params(axis='y',labelsize=font_size)



#图4 homo
df = pd.read_excel('plot.xlsx', sheet_name='homo')
df.to_csv('plot.csv', index=False, encoding='utf-8')
df = pd.read_csv('plot.csv', encoding='utf-8')

sns.boxplot(data=df, ax=axes[1,1])

# axes[1,1].tick_params(axis='x', rotation=45,labelsize=font_size)
axes[1,1].tick_params(axis='x',labelsize=font_size)
axes[1,1].tick_params(axis='y',labelsize=font_size)
axes[1,1].set_ylabel('', fontsize=font_size)
axes[1,1].set_xlabel('（e）最大均聚物对比', fontproperties=prop, fontsize=25, labelpad=20)



df = pd.read_excel('plot.xlsx', sheet_name='t4_density')
df.to_csv('plot.csv', index=False, encoding='utf-8')
df = pd.read_csv('plot.csv', encoding='utf-8')

sns.barplot(x='file', y='seq_num', hue="method", data=df,gap=0.2, ax=axes[1,2])
axes[1,2].set_xlabel('（f）平均序列数量对比', fontproperties=prop, fontsize=25, labelpad=20)
axes[1,2].set_ylabel('', fontsize=font_size)
axes[1,2].tick_params(axis='x',labelsize=font_size)
axes[1,2].tick_params(axis='y',labelsize=font_size)

plt.tight_layout()
plt.savefig('t4_all_merge_6.png')


