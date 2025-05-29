

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False

import pandas as pd


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))


df = pd.read_excel('plot.xlsx', sheet_name='t3_5')
df.to_csv('plot.csv', index=False, encoding='utf-8')
df = pd.read_csv('plot.csv', encoding='utf-8')


df['bsalign3'] = df['bsalign_ldpc']
df['SeqTransformer3'] = df['SeqTransformer_ldpc']
# df['bsalign3'] = df['bsalign_id20']
# df['SeqTransformer3'] = df['SeqTransformer_id20']
# df['bsalign3'] = df['bsalign_trelli']
# df['SeqTransformer3'] = df['SeqTransformer_trelli']
# df['bsalign3'] = df['bsalign_derrick']
# df['SeqTransformer3'] = df['SeqTransformer_derrick']
df['x3'] = df['x_ldpc']

df_bsalign3 = [df['bsalign3'][i] for i in range(len(df['bsalign3'])) if df['bsalign3'][i] != 0]
df_bsalign3_x = [df['x3'][i] for i in range(len(df['bsalign3'])) if df['bsalign3'][i] != 0]
# df_bsalign3_x = [df['x3_bs'][i] for i in range(len(df['bsalign3'])) if df['bsalign3'][i] != 0]
df_SeqTransformer3 = [df['SeqTransformer3'][i] for i in range(len(df['SeqTransformer3'])) if df['SeqTransformer3'][i] != 0]
df_SeqTransformer3_x = [df['x3'][i] for i in range(len(df['SeqTransformer3'])) if df['SeqTransformer3'][i] != 0]

sns.scatterplot(x='x3', y='x3', label='standard', data=df, color='red', s=100, marker='s', ax=axes[0])
sns.scatterplot(x=df_bsalign3_x, y=df_bsalign3, label='bsalign', color='black', s=100, marker='H', ax=axes[0])
sns.scatterplot(x=df_SeqTransformer3_x, y=df_SeqTransformer3, label='SeqFormer', color='blue', s=100, marker='o', ax=axes[0])
sns.lineplot(x='x3', y='x3', color='red', data=df, ax=axes[0])
sns.lineplot(x=df_bsalign3_x, color='black', y=df_bsalign3, ax=axes[0])
sns.lineplot(x=df_SeqTransformer3_x, color='blue', y=df_SeqTransformer3, ax=axes[0])

# plt.title('SeqTransformer vs bsalign(derrick)')
# plt.title('SeqTransformer vs bsalign(Srinivasavaradhan)')
# plt.title('SeqTransformer vs bsalign(id20)')

# plt.xlabel('accuracy', fontsize=12)
# plt.ylabel('true accuracy(compute)', fontsize=12)

# axes[0].set_title('SeqTransformer vs bsalign(Srinivasavaradhan)')
# axes[0].set_title('Ding et al.')
axes[0].set_title('Chandak et al.')
# axes[0].set_title('SeqTransformer vs bsalign(Chandak)')
# axes[0].set_title('SeqTransformer vs bsalign(Organick)')
axes[0].set_xlabel('accuracy')
axes[0].set_ylabel('true accuracy(compute)')




df = pd.read_excel('plot.xlsx', sheet_name='t3_5abs')
df.to_csv('plot.csv', index=False, encoding='utf-8')
df = pd.read_csv('plot.csv', encoding='utf-8')


df['bsalign3'] = df['bsalign_ldpc']
df['SeqTransformer3'] = df['SeqTransformer_ldpc']
# df['bsalign3'] = df['bsalign_id20']
# df['SeqTransformer3'] = df['SeqTransformer_id20']
# df['bsalign3'] = df['bsalign_trelli']
# df['SeqTransformer3'] = df['SeqTransformer_trelli']
# df['bsalign3'] = df['bsalign_derrick']
# df['SeqTransformer3'] = df['SeqTransformer_derrick']
df['x3'] = df['x']
df['y3'] = [0]*len(df['x'])
df_bsalign3 = [df['bsalign3'][i] for i in range(len(df['bsalign3'])) if df['bsalign3'][i] != 0]
df_bsalign3_x = [df['x3'][i] for i in range(len(df['bsalign3'])) if df['bsalign3'][i] != 0]
df_SeqTransformer3 = [df['SeqTransformer3'][i] for i in range(len(df['SeqTransformer3'])) if df['SeqTransformer3'][i] != 0]
df_SeqTransformer3_x = [df['x3'][i] for i in range(len(df['SeqTransformer3'])) if df['SeqTransformer3'][i] != 0]

sns.scatterplot(x='x3', y=df['y3'], label='standard', data=df, color='orange', s=100, marker='^', ax=axes[1])
sns.scatterplot(x=df_SeqTransformer3_x, y=df_SeqTransformer3, label='SeqFormer', color='purple', s=100, marker='^', ax=axes[1])
sns.scatterplot(x=df_bsalign3_x, y=df_bsalign3, label='bsalign', color='green', s=100, marker='^', ax=axes[1])
sns.lineplot(x=df['x3'], y=df['y3'], color='orange', ax=axes[1])
sns.lineplot(x=df_bsalign3_x, color='green', y=df_bsalign3, ax=axes[1])
sns.lineplot(x=df_SeqTransformer3_x, color='purple', y=df_SeqTransformer3, ax=axes[1])

# # plt.title('The abs value of the difference(ldpc)')
# # plt.title('The abs value of the difference(id20)')
# # plt.title('The abs value of the difference(derrick)')
# plt.title('The abs value of the difference(Srinivasavaradhan)')
# plt.xlabel('accuracy', fontsize=12)
# plt.ylabel('diff', fontsize=12)
# plt.ylim(-0.1,0.4)

# axes[1].set_title('The abs value of the difference(Srinivasavaradhan)')
# axes[1].set_title('Ding et al.')
axes[1].set_title('Chandak et al.')
# axes[1].set_title('The abs value of the difference(Chandak)')
# axes[1].set_title('The abs value of the difference(Organick)')
axes[1].set_xlabel('accuracy')
axes[1].set_ylabel('diff')
axes[1].set_ylim(-0.1,0.4)
# axes[1].set_ylim(-0.1,0.3)


plt.tight_layout()
# plt.savefig('t3-5.png')
plt.savefig('t3-5Chandak.png')
