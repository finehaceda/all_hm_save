
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
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))

font_size = 20

#图1 density
df = pd.read_excel('plot.xlsx', sheet_name='t4_density')
df.to_csv('plot.csv', index=False, encoding='utf-8')
df = pd.read_csv('plot.csv', encoding='utf-8')
# df = df[1:4]
# axes[0,0].set_ylabel('Density(bits/nt)', fontsize=font_size)
# sns.scatterplot(x='method', y='density', hue="file", data=df, marker='D', ax=axes[0,0])
# ax = sns.barplot(x='avg_density3', y='avg_method', data=df, orient='h', width=0.6, palette="muted", ax=axes[0,0])
ax = sns.barplot(x='density.9.jpg.ill', y='avg_method', data=df, orient='h', width=0.6, palette="muted", ax=axes[0,0])
axes[0,0].set_ylabel('', fontsize=font_size)
axes[0,0].set_xlabel('(a) density(bits/nt)', fontproperties=prop, fontsize=25, labelpad=20)
axes[0,0].tick_params(axis='x',labelsize=font_size)
axes[0,0].tick_params(axis='y',labelsize=font_size)
axes[0,0].legend(prop={'size': 15})
axes[0,0].grid(axis='x')

for p in ax.patches:
    width = p.get_width()
    if width > 0:  # 过滤掉宽度为 0 的柱子
        ax.annotate(f'{width:.2f}', (width, p.get_y() + p.get_height() / 2.),
                    ha='left', va='center', fontsize=font_size, color='black', xytext=(5, 0),
                    textcoords='offset points')
ax.set_xlim(0,1.8)
#图2 cost
df = pd.read_excel('plot.xlsx', sheet_name='cost')
df.to_csv('plot.csv', index=False, encoding='utf-8')
df = pd.read_csv('plot.csv', encoding='utf-8')
# df = df[1:4]
# axes[0,0].set_ylabel('Density(bits/nt)', fontsize=font_size)
# sns.scatterplot(x='method', y='density', hue="file", data=df, marker='D', ax=axes[0,0])
ax = sns.barplot(x='cost.9.jpg.ill', y='method', data=df, orient='h', width=0.6, palette="muted", ax=axes[0,1])
axes[0,1].set_ylabel('', fontsize=font_size)
axes[0,1].set_xlabel('(b) cost($)', fontproperties=prop, fontsize=25, labelpad=20)
axes[0,1].tick_params(axis='x',labelsize=font_size)
axes[0,1].tick_params(axis='y',labelsize=font_size)
axes[0,1].legend(prop={'size': 15})
axes[0,1].grid(axis='x')


# #图3 encodetime
color_list = ['red', 'blue', 'green', 'orange', 'purple']
df = pd.read_excel('plot.xlsx', sheet_name='encodetime')
df.to_csv('plot.csv', index=False, encoding='utf-8')
df = pd.read_csv('plot.csv', encoding='utf-8')
# df = df[1:4]

ax = sns.barplot(x='time.9.jpg.ill', y='method', data=df, orient='h', width=0.6, palette="muted", ax=axes[1,0])

axes[1,0].tick_params(axis='x',labelsize=font_size)
axes[1,0].tick_params(axis='y',labelsize=font_size)
# axes[2,0].set_xlabel('average decode time(s)', fontsize=font_size)
axes[1,0].set_xlabel('(c) encode time(s)', fontproperties=prop, fontsize=25, labelpad=20)
axes[1,0].set_ylabel('', fontsize=font_size)



# 图4 decodetime
color_list = ['red', 'blue', 'green', 'orange', 'purple']
df = pd.read_excel('plot.xlsx', sheet_name='decodetime')
df.to_csv('plot.csv', index=False, encoding='utf-8')
df = pd.read_csv('plot.csv', encoding='utf-8')
# df = df[1:4]

sns.barplot(x='time.9.jpg.ill', y='method', data=df, orient='h', width=0.6, palette="muted", ax=axes[1,1])


axes[1,1].tick_params(axis='x',labelsize=font_size)
axes[1,1].tick_params(axis='y',labelsize=font_size)
# axes[2,0].set_xlabel('average decode time(s)', fontsize=font_size)
axes[1,1].set_xlabel('(d) decode time(s)', fontproperties=prop, fontsize=25, labelpad=20)
axes[1,1].set_ylabel('', fontsize=font_size)






plt.tight_layout()
# plt.savefig('t4_density_encodetime_decodetime_cost.nano.png')
plt.savefig('t4_density_encodetime_decodetime_cost.ill.png')


