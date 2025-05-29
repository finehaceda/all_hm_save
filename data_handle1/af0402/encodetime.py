
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
# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))

font_size = 20


# #图3 encodetime
color_list = ['red', 'blue', 'green', 'orange', 'purple']
df = pd.read_excel('plot.xlsx', sheet_name='encodetime')
df.to_csv('plot.csv', index=False, encoding='utf-8')
df = pd.read_csv('plot.csv', encoding='utf-8')
# df = df[1:4]

ax = sns.barplot(x='time.9.jpg.ill', y='method', data=df, orient='h', width=0.6, palette="muted")

ax.tick_params(axis='x',labelsize=font_size)
ax.tick_params(axis='y',labelsize=font_size)
# axes[2,0].set_xlabel('average decode time(s)', fontsize=font_size)
ax.set_xlabel('encode time(s)', fontproperties=prop, fontsize=25, labelpad=20)
ax.set_ylabel('', fontsize=font_size)



plt.tight_layout()
# plt.savefig('t4_density_encodetime_decodetime_cost.nano.png')
plt.savefig('encodetime.png')


