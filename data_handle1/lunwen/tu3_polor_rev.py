

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False

import pandas as pd


# fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(18, 8))
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(9, 8))


df = pd.read_excel('plot.xlsx', sheet_name='t3_6_polar')
df.to_csv('plot.csv', index=False, encoding='utf-8')
df = pd.read_csv('plot.csv', encoding='utf-8')

# fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))

# fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # 2行2列的子图布局
sns.lineplot(x='copy', y='rev_5', hue="method", data=df, ax=axes[0,0])
axes[0,0].set_title('rev:bit:5%')

sns.lineplot(x='copy', y='rev_6', hue="method", data=df, ax=axes[0,1])
axes[0,1].set_title('rev:bit:6%')

# sns.lineplot(x='copy', y='rev_8', hue="method", data=df, ax=axes[0,2])
# axes[0,2].set_title('rev:bit:8%')
#
# sns.lineplot(x='copy', y='rev_10', hue="method", data=df, ax=axes[0,3])
# axes[0,3].set_title('rev:bit:10%')

sns.lineplot(x='copy', y='bitrev_5', hue="method", data=df, ax=axes[1,0])
axes[1,0].set_title('bitrev:bit:5%')

sns.lineplot(x='copy', y='bitrev_6', hue="method", data=df, ax=axes[1,1])
axes[1,1].set_title('bitrev:bit:6%')

# sns.lineplot(x='copy', y='bitrev_8', hue="method", data=df, ax=axes[1,2])
# axes[1,2].set_title('bitrev:bit:8%')
#
# sns.lineplot(x='copy', y='bitrev_10', hue="method", data=df, ax=axes[1,3])
# axes[1,3].set_title('bitrev:bit:10%')
plt.tight_layout()
plt.savefig('tu3-7.png')


