# # def ges(dna):
# #     for i in range(len(dna)):
# #         if str(dna[i]).endswith('2'):
# #             dna[i] = dna[i][:len(dna[i])-1]
# #     return dna
# # s = 'CGAGTAGCAGCCTGAAATGTCGAACCCCATCGGGGCTGGGACGACCATTGGCACTAGGTTGCGTCCTGAGATACAAATTGGCTGGAAGATAAGGTATTCTGTGTAAATACTCTAGGTAATCAGCCATGCGGTCAGCCACTGACTATACTTCGCATCTGCGCGCTTAGTAAGCGGCGTTTCCAGTCTGCTTCGATTCGCATTGAGTTGCAAAATGATAATTCTCCGGGAAGCCTGCGCCCACACAAGTTCGAAACC2'
# # dna = [s]
# # print(ges(dna))
# # import Levenshtein
# # seq1='CCGAATCTGGCCCATCAGATGTTCCTCTTTTTGTCCGGCGGGCTATGAGAAACTACGCTTTCTGAGGAAGGAAACACATTACCGATGGGGGAGATGCGGGAGGCTATCGTACGCCATAGGAGACTGTGTTATGCCAGGGTGGGTGAAGCCGTTGCCCGTCTTCGAACGACATGATCGCGCCTTCTTTTCTTCTAAAGAGCCTGGGAGGTGCGTGTCCCTCTCAATGTATTATTTCAGGAATGTTACCCCGGCGACCATCG'
# # seq2='CCGAATCTGGCCCATCAGATGTTCCTCTTTTTGTCCGGCGGGCTATGAGAAACTACGCTTTCTGAGGAAGGAAACACATTACCGATGGGGGAGATGCGGAGGCTATCGTACGCCATAGGAGACTGTGTTATGCCAGGGTGGGTGAAGCCGTTGCCCGTCTTCGAACGACATGATCGCGCCTTCTTTTCTTCTAAAGAGCCTGGGAGGTGCGTGTCCCTCTCAATGTATTATTTCAGGAATGTTACCCCGGCGACCATCGG'
# # edit_ops = Levenshtein.editops(seq1, seq2)
# # print(edit_ops)
# import csv
#
#
# def getgc(all_seqs):
#     avggc = 0
#     all_gc = []
#     for dna_sequence in all_seqs:
#         gc_count = dna_sequence.count('G') + dna_sequence.count('C')
#         gc = gc_count / len(dna_sequence)
#         avggc += gc
#         all_gc.append(gc)
#     with open(f'gc.csv', mode='w', newline='') as file:
#         writer = csv.writer(file)
#         for g in all_gc:
#             t = (g,)
#             writer.writerow(t)
#     return avggc / len(all_seqs), all_gc
#
# # getgc([0.51,0.49,0.5,])
# getgc(['AGCAACGAGTTAC','CAGATACGAA'])


import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter


def saveseqsdistributed_fig(ori_dna_sequences, all_seqs, output_filename='sequence_distribution.png'):
    # 假设 all_seqs 是一个列表的列表，每个子列表是一个聚类
    # 计算每个聚类中的序列数量
    cluster_sizes = [len(cluster) for cluster in all_seqs]

    # 绘制分布图
    plt.figure(figsize=(10, 6))
    sns.histplot(cluster_sizes, kde=True, bins=range(min(cluster_sizes), max(cluster_sizes) + 2))
    plt.title('Sequence Number Distribution in Clusters')
    plt.xlabel('Number of Sequences in Cluster')
    plt.ylabel('Frequency')

    # 保存图片
    plt.savefig(output_filename)
    plt.close()

    # 打印一些调试信息（可选）
    print(f"Distribution saved to {output_filename}")
    print(f"Cluster sizes: {Counter(cluster_sizes)}")


# 示例数据
ori_dna_sequences = ["ATCG", "GCAT", "TTAG"]
all_seqs = [
    ["ATCG", "GCAT"],  # 聚类1，包含2个序列
    ["TTAG", "GGAA", "CCTT"],  # 聚类2，包含3个序列
    ["AAAA", "TTTT"]  # 聚类3，包含2个序列
]

# 调用函数
saveseqsdistributed_fig(ori_dna_sequences, all_seqs)

