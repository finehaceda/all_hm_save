import random
import subprocess
from time import sleep

import numpy as np
import sklearn
import torch
from Bio import AlignIO
from scipy.special import softmax
from Bio.Align.Applications import ClustalwCommandline
# import tensorflow as tf


# 将 DNA 序列映射到独热编码
from sklearn.preprocessing import StandardScaler


def one_hot_encode(sequence):
    bases = 'ACGT-'
    base_dict = {base: i for i, base in enumerate(bases)}
    encoded = np.zeros((len(sequence), len(bases)))

    for i, base in enumerate(sequence.upper()):
        if base in base_dict:
            encoded[i, base_dict[base]] = 1
        else:
            # 处理未知的碱基（可以根据需求进行其他操作）
            # encoded[i, :] = 0.25  # 平均分配到四个碱基位置
            # encoded[i, :] = 0 # 平均分配到四个碱基位置
            encoded[i, 4] = 1 # 平均分配到四个碱基位置

    return encoded

def one_hot_encode_ori(sequence):
    bases = 'ACGT'
    base_dict = {base: i for i, base in enumerate(bases)}
    encoded = np.zeros((len(sequence), len(bases)))

    for i, base in enumerate(sequence):
        if base in base_dict:
            encoded[i, base_dict[base]] = 1
        else:
            # 处理未知的碱基（可以根据需求进行其他操作）
            # encoded[i, :] = 0.25  # 平均分配到四个碱基位置
            encoded[i, :] = 0 # 平均分配到四个碱基位置
            # encoded[i, 4] = 1 # 平均分配到四个碱基位置

    return encoded

# 将 DNA 序列填充到指定长度
def pad_sequence(sequence, max_length):
    padded_seq = sequence[:max_length] + '-' * max(0, max_length - len(sequence))
    return padded_seq


def calculate_base_proportions111(encoded_sequences):
    # 初始化一个数组来存储每个位置每个碱基的比例
    proportions_per_position = np.zeros((encoded_sequences.shape[1], encoded_sequences.shape[2]))

    # 对每个位置进行遍历
    for position in range(encoded_sequences.shape[1]):
        # 对每个碱基进行求和
        sum_per_base = np.sum(encoded_sequences[:, position, :], axis=0)

        # 计算每个碱基在该位置的比例
        total_positions = encoded_sequences.shape[0]
        proportions_per_position[position, :] = sum_per_base / total_positions

    return proportions_per_position

def calculate_base_proportions(encoded_sequences):
    # 初始化一个数组来存储每个位置每个碱基的比例
    # proportions_per_position = np.zeros((encoded_sequences.shape[0],encoded_sequences.shape[1], encoded_sequences.shape[2]))
    proportions_per_position = np.zeros((encoded_sequences.shape[1], encoded_sequences.shape[2]))

    # 对每个位置进行遍历
    for position in range(encoded_sequences.shape[1]):
        # 对每个碱基进行求和
        sum_per_base = np.sum(encoded_sequences[:, position, :], axis=0)

        # 计算每个碱基在该位置的比例
        total_positions = encoded_sequences.shape[0]
        proportions_per_position[position, :] = sum_per_base / total_positions

    return proportions_per_position

def normalize_list(input_list):
    # 找到列表中的最小值和最大值
    min_value = min(input_list)
    max_value = max(input_list)

    # 归一化列表中的每个元素
    normalized_list = [(x - min_value) / (max_value - min_value) for x in input_list]

    return normalized_list

def standardize_list(input_list):
    # 将列表转换为二维数组，这是许多机器学习库期望的输入格式
    data = [[x] for x in input_list]

    # 使用 StandardScaler 进行标准化
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data)

    # 提取标准化后的值并返回为一维列表
    standardized_list = [x[0] for x in standardized_data]

    return standardized_list
def softmax(x):
    exp_x = np.exp(x - np.max(x))  # 避免数值不稳定性，减去最大值
    return exp_x / exp_x.sum(axis=1, keepdims=True)


def trans_ori_hot(dna_sequences,max_sequence_length = 128):
    # 确定最大长度
    all_base_proportions = []
    save_seqs(dna_sequences,'myfiles/alltestoriseqs.fasta')
    all_base_proportionssum = []
    # max_sequence_length = max(len(seq) for seq in dna_sequences)
    # max_sequence_length = 128
    for seqs in dna_sequences:
        encoded_sequences = []
        # 对每条 DNA 序列进行独热编码和填充
        # max_sequence_length = max(len(seq) for seq in seqs)
        # padded_seq = pad_sequence(seqs, max_sequence_length)
        encoded_seq = one_hot_encode_ori(seqs)
        encoded_sequences.append(encoded_seq)

        # 转换为 numpy 数组
        encoded_sequences = np.array(encoded_sequences)
        sum_per_position = np.sum(encoded_sequences, axis=0)
        # all_base_proportions.append(sum_per_position)
        # sum_per_position = [(nums[0]*1+nums[1]*2+nums[2]*3+nums[3]*4) for nums in sum_per_position]
        sum_per_position = [(nums[1]*1+nums[2]*2+nums[3]*3) for nums in sum_per_position]
        all_base_proportions.append(sum_per_position)
        # sum_per_position = [((nums[0]*1+nums[1]*2+nums[2]*3+nums[3]*4)-1)/5 for nums in sum_per_position]

        # sum_per_position = normalize_list(sum_per_position)
        # sum_per_position = standardize_list(sum_per_position)
        # encoded_sequences = np.array(encoded_sequences)
        # base_proportions = calculate_base_proportions(encoded_sequences)
    return all_base_proportions

def readfile(path):
    sequences = []
    with open(path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if not line.startswith('>'):
                sequences.append(line.strip('\n'))
    return sequences

def writefile(data,path):
    with open(path, "w") as fasta_file:
        for i, seq in enumerate(data, 1):
            fasta_file.write(f">Seq{i}\n{seq}\n")

def get_aliseqs(seqs,path='input_sequences'):
    writefile(seqs, path+'.fasta' )
    clustalw_cmd = ClustalwCommandline("clustalw2", infile= path+'.fasta')
    clustalw_cmd()

def get_ori_ali(file_path,ori_seq = 'Seq6'):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    sequences = {}
    for line in lines:
        if line.startswith("Seq"):  # 原始序列以空格开头
            parts = line.split()
            sequence_name = parts[0]
            sequence_data = ''.join(parts[1:])
            if sequence_name not in sequences:
                sequences[sequence_name] = ''
            sequences[sequence_name] += sequence_data
    ori = sequences.get(ori_seq)
    sequences.pop(ori_seq, None)
    seqs =  list(sequences.values())
    return ori,seqs
def get_ori_alitest(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    sequences = {}
    for line in lines:
        if line.startswith("Seq"):  # 原始序列以空格开头
            parts = line.split()
            sequence_name = parts[0]
            sequence_data = ''.join(parts[1:])
            if sequence_name not in sequences:
                sequences[sequence_name] = ''
            sequences[sequence_name] += sequence_data
    # ori = sequences.get('Seq6')
    # sequences.pop('Seq6', None)
    seqs =  list(sequences.values())
    return seqs
def manage_oriali(seqs,sequence_length):
    # encoded_sequences = []
    padded_seq = pad_sequence(seqs, sequence_length)
    encoded_seq = one_hot_encode(padded_seq)
    # encoded_sequences.append(encoded_seq)

    # 转换为 numpy 数组
    encoded_sequences = np.array([encoded_seq],dtype=int)
    sum_per_position = np.sum(encoded_sequences, axis=0)
    # all_base_proportions.append(sum_per_position)
    # sum_per_position = [(nums[0]*1+nums[1]*2+nums[2]*3+nums[3]*4) for nums in sum_per_position]
    sum_per_position = [(nums[1] * 1 + nums[2] * 2 + nums[3] * 3 + nums[4] * 4) for nums in sum_per_position]
    return sum_per_position

def get_kmer_times(seqs,kmer_len=5, num = 1):
    all_kmers=[]
    for seq in seqs:
        kmers = []
        for i in range(0, len(seq) - kmer_len + 1):
            kmer = seq[i: i + kmer_len]
            kmers.append(kmer)
        all_kmers.append(kmers)
    return all_kmers

def save_seqs_special(seq, path):
    with open(path, 'w') as file:
        for j, cus in enumerate(seq):
            file.write('>' + str(j) + '\r\n')
            file.write(str(cus) + '\r\n')

def save_seqs(seq, path):
    with open(path, 'w') as file:
        for j, cus in enumerate(seq):
            file.write('>' + str(j) + '\n')
            file.write(str(cus) + '\n')

def read_alifiles(path,num):
    seq_inf = []
    with open(path, "r") as file:
        lines = file.readlines()
    for i in range(2,num+2):
        templine = lines[i].strip('\n').split(' ')[3].replace('.','-')
        seq_inf.append(templine)
    return seq_inf

def read_alifilestest(path,num):
    seq_inf = []
    with open(path, "r") as file:
        lines = file.readlines()
    for i in range(2,num+2):
        templine = lines[i].strip('\n').split(' ')[3].replace('.','-')
        seq_inf.append(templine)
    # lii = lines[num+7].strip('\n').split(' ')[3]
    qua = lines[num+7].strip('\n').split(' ')[1].split('\t')[2]
    consus = lines[num+6].strip('\n').split(' ')[1].split('\t')[2]
    errors = []
    if len(lines)>num+9:
        for i in range(len(lines)-(num+9)):
            line = lines[num+9+i].strip('\n').split(' ')[1].split('\t')
            dic = {}
            dic[line[5]] = int(line[6])
            dic[line[7]] = int(line[8])
            errors.append(dic)
    return seq_inf,qua,consus,errors


def bsalign_ali11(cluster_seqs,num):
    save_seqs(cluster_seqs, 'seqs.fasta')
    # save_seqs_special(cluster_seqs, 'seqs.fasta')
    shell = './deeplearning_align/bsalign-master/bsalign poa seqs.fasta -o consus.txt -L > ali.ali'  # 都要用这条语句
    # shell = './bsalign-master/bsalign poa seqs.fasta -o consus.txt -L > ali.ali'  # 146e用这条语句，试试
    result = subprocess.run(shell, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # sleep(300)
    seqs = read_alifiles('ali.ali', num)

    with open('consus.txt', 'r') as file:
        lines = file.readlines()
        for line in lines:
            consus = line.strip('\n')
    return seqs,consus

def bsalign_ali(cluster_seqs,num):
    # print()
    # print(cluster_seqs)
    if len(cluster_seqs)==0:
        return [],'',[],''
    save_seqs(cluster_seqs, '/seqs.fasta')
    # save_seqs_special(cluster_seqs, 'seqs.fasta')
    shell = './deeplearning_align/bsalign-master/bsalign poa seqs.fasta -o consus.txt -L > ali.ali'
    # shell = '../bsalign poa one_cluster.fasta -o consus.txt -L>consensus.bsalign'
    result = subprocess.run(shell, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # print(result)
    # sleep(300)
    seqs,_,aliconsus,errors = read_alifilestest('ali.ali', num)
    quasdict = []

    with open('./consus.txt', 'r') as file:
        lines = file.readlines()
        for line in lines:
            consus = line.strip('\n')
    # if len(consus) > len(cluster_seqs[0]) * 1.1:
    #     consus = cluster_seqs[0]
    #     return seqs, consus, consus, consus

    for i in range(len(seqs[0])):
        dict = {'A': 0, 'C': 0, 'G': 0, 'T': 0, '-': 0}
        for j in range(num):
            dict[seqs[j][i].upper()]+=1
        max_value = max(dict.values())
        max_key = max(dict, key=dict.get)
        max_value2 = sorted(dict.values())[-2]
        max_key2 = next(key for key, value in dict.items() if value == max_value2)
        if max_key == '-' and max_key2 == consus[i]:
            continue
        else:
            break
    newconsus = consus[i:]
    newlastindex = len(newconsus)-1
    for i in range(len(seqs[0])-1,0,-1):
        dict = {'A': 0, 'C': 0, 'G': 0, 'T': 0, '-': 0}
        for j in range(num):
            dict[seqs[j][i].upper()]+=1
        max_value = max(dict.values())
        max_key = max(dict, key=dict.get)
        max_value2 = sorted(dict.values())[-2]
        max_key2 = next(key for key, value in dict.items() if value == max_value2)
        if max_key == '-':
            if max_value==num:
                continue
            elif max_key2 == newconsus[newlastindex]:
                newlastindex-=1
                continue
        else:
            break
    newconsus = newconsus[:newlastindex+1]
    if len(newconsus) > len(cluster_seqs[0]) *1.07:
        # newconsus = cluster_seqs[random.randint(0,num-1)]
        newconsus = cluster_seqs[0]
    # if len(newconsus)<sequence_length-3:
    #     print(newconsus)
    return seqs,consus,newconsus
    # return seqs,myseqno_,quasno_

# def trans_seq_hot11(dna_sequences,ori_dna_sequences,sequence_length = 128):
#     ori_bases = []
#     all_base_proportions = []
#     len_max = []
#     for seqsi in range(len(dna_sequences)):
#         if seqsi%100 == 0:
#             print(seqsi)
#         seqs = dna_sequences[seqsi]
#         ori = ori_dna_sequences[seqsi]
#         seqs.append(ori)
#         encoded_sequences = []
#         get_aliseqs(seqs)
#         ori_ali_seq,seqs_up = get_ori_ali('./input_sequences.aln','Seq11')
#         len_max.append(len(seqs_up[0]))
#         ori_ali = manage_oriali(ori_ali_seq,sequence_length)
#         for seq in seqs_up:
#             padded_seq = pad_sequence(seq, sequence_length)
#             encoded_seq = one_hot_encode(padded_seq)
#             encoded_sequences.append(encoded_seq)
#
#         encoded_sequences = np.array(encoded_sequences)
#         sum_per_position = np.sum(encoded_sequences, axis=0)
#         sum_per_position = softmax(sum_per_position)
#         all_base_proportions.append(sum_per_position)
#         ori_bases.append(ori_ali)
#     print(f"max_num:{max(len_max)}")
#     return ori_bases,all_base_proportions

def getRadomSeqsNoQua(dna_sequences,select_nums=5):
    random.seed(27678)
    new_dna_sequences=[]
    for i in range(len(dna_sequences)):
        length = len(dna_sequences[i])
        if length >= select_nums:
            select_number = select_nums
        else:
            select_number = length
        indexs = random.sample(range(length), select_number)
        new_dna_sequences.append([dna_sequences[i][j] for j in indexs])
    # return np.array(new_dna_sequences),np.array(new_all_quas)
    return new_dna_sequences

def getRadomSeqs(dna_sequences,all_quas,select_nums=5):
    random.seed(6534533)
    new_dna_sequences,new_all_quas=[],[]
    for i in range(len(dna_sequences)):
        length = len(dna_sequences[i])
        if length >= select_nums:
            select_number = select_nums
        else:
            select_number = length
        indexs = random.sample(range(length), select_number)
        new_dna_sequences.append([dna_sequences[i][j] for j in indexs])
        new_all_quas.append([all_quas[i][j] for j in indexs])
    return new_dna_sequences,new_all_quas

def softmax0(your_array):
    normalized_array = (your_array) / (np.sum(your_array, axis=0))
    return normalized_array

def quasaliseqspass(quas,seqs_up):
    indices_list = [[i for i, char in enumerate(string) if char == '-'] for string in seqs_up]
    # means = np.mean(quas,axis=1)
    means = [sum(lst) / len(lst) for lst in quas]
    lens = len(seqs_up[0])
    new_quas = []
    for i in range(len(quas)):
        qua = quas[i]
        arr = []
        k = 0
        indeces = indices_list[i]
        # print(lens)
        # print(len(indeces))
        # print(seqs_up[i])
        # print(len(qua))
        for j in range(len(seqs_up[0])):
            if j in indeces:
                # arr.append(means[i])
                arr.append(5)
                k+=1
            else:
                # print(i)
                # print(j)
                # print(k)
                arr.append(qua[j-k])
        new_quas.append(arr)
    return new_quas

def quasaliseqs(quas,seqs_up):
    indices_list = [[i for i, char in enumerate(string) if char == '-'] for string in seqs_up]
    # means = np.mean(quas,axis=1)
    means = [sum(lst) / len(lst) for lst in quas]
    lens = len(seqs_up[0])
    new_quas = []
    for i in range(len(quas)):
        qua = quas[i]
        arr = []
        k = 0
        indeces = indices_list[i]
        # print(lens)
        # print(len(indeces))
        # print(seqs_up[i])
        # print(len(qua))
        for j in range(len(seqs_up[0])):
            if j in indeces:
                # arr.append(means[i])
                arr.append(0)
                # arr.append(5)
                k+=1
            else:
                # print(i)
                # print(j)
                # print(k)
                if j-k<len(qua):
                    arr.append(qua[j-k])
                else:
                    arr.append(1)
        new_quas.append(arr)
    new_quas = np.array(new_quas)
    for i in range(len(new_quas[0])):
        num,sumn = 0,0
        # ss = new_quas[:,i]
        index0s = []
        for k in range(len(new_quas[:,i])):
            if new_quas[k,i]!=0:
                num+=1
                sumn+=new_quas[k,i]
            else:
                index0s.append(k)
        if num == 0:
            for k in index0s:
                new_quas[k][i] = 1
        if 0 < num < 10 :
            avg = sumn / num
            for k in index0s:
                # if new_quas[k][i]==0:
                new_quas[k][i] = avg
    return new_quas

def promaxPhred(encoded_sequences,quas):
    seqs = []
    quas = softmax0(quas)
    for i in range(len(encoded_sequences[0])):
        base_scores = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        for j in range(len(encoded_sequences)):
            a = encoded_sequences[j][i]
            b = quas[j][i]
            # c = np.where(a == 1)
            c = np.where(a == 1)[0][0]

            base_scores[c] += b
        seqs.append(list(base_scores.values()))

    # return np.array(seqs)
    return seqs

def promaxPhreddata(encoded_sequences,quas):
    seqs = []
    # quas = softmax0(quas)
    for i in range(len(encoded_sequences[0])):
        base_scores = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        for j in range(len(encoded_sequences)):
            a = encoded_sequences[j][i]
            b = quas[j][i]
            c = np.where(a == 1)[0][0]

            base_scores[c] += b
        seqs.append(list(base_scores.values()))

    # return np.array(seqs)
    return seqs

def trans_seq_hot(dna_sequences,ori_dna_sequences,all_quas,copy_num=10,sequence_length = 128):
    # dna_sequences,all_quas_ = getRadomSeqs(dna_sequences,all_quas,10)
    dna_sequences,all_quas = getRadomSeqs(dna_sequences,all_quas,copy_num)
    # all_quas = np.array(all_quas_)
    ori_bases = []
    all_base_proportions = []
    all_base_proportions_phreds = []
    all_seqs_up = []
    all_ori_ali_seq = []
    all_ali_seqs_formaxlen = []
    # len_max = []
    for seqsi in range(len(dna_sequences)):
        if seqsi%1000 == 0:
            print(seqsi)
        seqs = dna_sequences[seqsi]
        ori = ori_dna_sequences[seqsi]
        seqs.append(ori)
        # get_aliseqs(seqs)
        bsalign_seqs,_ = bsalign_ali(seqs,copy_num+1)
        seqs_up,ori_ali_seq = bsalign_seqs[:copy_num], bsalign_seqs[copy_num]
        all_seqs_up.append(seqs_up)
        all_ori_ali_seq.append(ori_ali_seq)
        all_ali_seqs_formaxlen.append(len(seqs_up[0]))
    sequence_length = max(all_ali_seqs_formaxlen)
    print(f"trans_seq_hot max_num:{sequence_length}")

    for seqsi in range(len(dna_sequences)):
        if seqsi%1000 == 0:
            print(seqsi)
        # sequence_length = get_max_alilen(seqs_up)
        # all_bsalign_consus.append(consus)
        # ori_ali_seq,seqs_up = get_ori_ali('./input_sequences.aln','Seq11')
        # print(len(seqs_up[0]))
        # len_max.append(len(seqs_up[0]))
        # quas_ = all_quas[seqsi]
        quas = all_quas[seqsi]
        # quas = softmax0(quas_)
        ori_ali = manage_oriali(all_ori_ali_seq[seqsi],sequence_length)
        padded_seqs = []
        encoded_sequences = []
        for seq in all_seqs_up[seqsi]:
            # lens1 = len(seq)
            padded_seq = pad_sequence(seq, sequence_length)
            # lens2 = len(padded_seq)
            padded_seqs.append(padded_seq)
            encoded_seq = one_hot_encode(padded_seq)
            encoded_sequences.append(encoded_seq)

        encoded_sequences = np.array(encoded_sequences)
        quas = quasaliseqs(quas,padded_seqs)
        # sum_per_position = promaxPhred(encoded_sequences,quas)
        # all_base_proportions.append(sum_per_position)
        sum_per_position_phred = promaxPhreddata(encoded_sequences,quas)
        all_base_proportions_phreds.append(sum_per_position_phred)
        ori_bases.append(ori_ali)
        encoded_sequences = np.array(encoded_sequences)
        sum_per_position = np.sum(encoded_sequences, axis=0)
        all_base_proportions.append(sum_per_position)
        # # sum_per_position = softmax(sum_per_position)
        # ori_bases.append(ori_ali)
    all_base_proportions = np.array(all_base_proportions)
    all_base_proportions_phreds =  np.array(all_base_proportions_phreds)
    ori_bases = np.array(ori_bases)
    # return ori_bases,all_base_proportions
    return ori_bases,all_base_proportions,all_base_proportions_phreds
    # return ori_bases,all_base_proportions_phreds

def trans_seq_hottest(dna_sequences,all_quas,copy_num=10 ):
    # dna_sequences,all_quas_ = getRadomSeqs(dna_sequences,all_quas,10)
    dna_sequences,all_quas = getRadomSeqs(dna_sequences,all_quas,copy_num)
    save_seqs(dna_sequences,'./files/alltestseqs.fasta')
    # all_quas = np.array(all_quas_)
    all_base_proportions = []
    # len_max = []
    all_seqs_up = []
    # all_ori_ali_seq = []
    all_base_proportions_phreds = []
    all_ali_seqs_formaxlen = []
    all_consus = []
    for seqsi in range(len(dna_sequences)):
        if seqsi%500 == 0:
            print(seqsi)
        seqs = dna_sequences[seqsi]
        seqs_up,consus = bsalign_ali(seqs,copy_num)
        all_consus.append(consus)
        all_seqs_up.append(seqs_up)
        # all_ori_ali_seq.append(ori_ali_seq)
        all_ali_seqs_formaxlen.append(len(seqs_up[0]))
    sequence_length = max(all_ali_seqs_formaxlen)
    print(f"trans_seq_hottest max_num:{sequence_length}")

    for seqsi in range(len(dna_sequences)):
        if seqsi%500 == 0:
            print(seqsi)
        # seqs_up = get_ori_alitest('./input_sequences_test.aln')
        # save_seqsLine(seqs_up,'./testseqs_up.aln')
        # len_max.append(len(seqs_up[0]))
        # print(len(seqs_up[0]))
        # quas_ = all_quas[seqsi]
        quas = all_quas[seqsi]
        # quas = softmax0(quas_)
        padded_seqs = []
        encoded_sequences = []
        for seq in all_seqs_up[seqsi]:
            # lens = len(seq)
            padded_seq = pad_sequence(seq, sequence_length)
            padded_seqs.append(padded_seq)
            encoded_seq = one_hot_encode(padded_seq)
            encoded_sequences.append(encoded_seq)

        encoded_sequences = np.array(encoded_sequences)
        quas = quasaliseqs(quas,padded_seqs)
        # sum_per_position = promaxPhred(encoded_sequences,quas)
        # all_base_proportions.append(sum_per_position)
        sum_per_position_phred = promaxPhreddata(encoded_sequences,quas)
        all_base_proportions_phreds.append(sum_per_position_phred)

        encoded_sequences = np.array(encoded_sequences)
        sum_per_position = np.sum(encoded_sequences, axis=0)
        all_base_proportions.append(sum_per_position)
        # sum_per_position = softmax(sum_per_position)
        # ori_bases.append(ori_ali)
    all_base_proportions = np.array(all_base_proportions)
    all_base_proportions_phreds = np.array(all_base_proportions_phreds)
    # print(f"max_num:{max(len_max)}")
    print('testdata done!')
    # return all_base_proportions,all_consus
    return all_base_proportions,all_base_proportions_phreds,all_consus

def trans_seq_hottest_nophred(dna_sequences,copy_num):
    # 要修改这个函数
    dna_sequences = getRadomSeqsNoQua(dna_sequences,copy_num)
    save_seqs(dna_sequences,'./files/alltestseqs.fasta')
    all_base_proportions = []
    all_seqs_up = []
    all_base_proportions_phreds = []
    all_ali_seqs_formaxlen = []
    all_consus = []
    for seqsi in range(len(dna_sequences)):
        seqs = dna_sequences[seqsi]
        if len(seqs) >= copy_num:
            copy_number = copy_num
        else:
            copy_number = len(seqs)
        seqs_up,consus = bsalign_ali(seqs,copy_number)
        all_consus.append(consus)
        all_seqs_up.append(seqs_up)
        all_ali_seqs_formaxlen.append(len(seqs_up[0]))
    sequence_length = max(all_ali_seqs_formaxlen)
    print(f"trans_seq_hottest max_num:{sequence_length}")

    for seqsi in range(len(dna_sequences)):
        padded_seqs = []
        encoded_sequences = []
        for seq in all_seqs_up[seqsi]:
            # lens = len(seq)
            padded_seq = pad_sequence(seq, sequence_length)
            padded_seqs.append(padded_seq)
            encoded_seq = one_hot_encode(padded_seq)
            encoded_sequences.append(encoded_seq)

        encoded_sequences = np.array(encoded_sequences)
        sum_per_position = np.sum(encoded_sequences, axis=0)
        all_base_proportions.append(sum_per_position)
        # sum_per_position = softmax(sum_per_position)
        # ori_bases.append(ori_ali)
    all_base_proportions = np.array(all_base_proportions)
    # print(f"max_num:{max(len_max)}")
    print('testdata done!')
    # return all_base_proportions,all_consus
    return all_base_proportions,all_consus

def save_seqsLine(seq,path):
    with open(path, 'w') as file:
        for j,cus in enumerate(seq):
            file.write('>'+str(j) + ' ' + str(len(cus))+ '\n')
            file.write(str(cus) + '\n')
    # print(j)
def save_kmers(ori_ali_seq_kmers,seqs_up_kmers,path):
    with open(path, 'w') as file:
        file.write('>ori' + str(1) + ' oriseq' + '\n')
        file.write(str(ori_ali_seq_kmers) + '\n')
        file.write('>train' + str(1) + '\n')
        for i in range(len(seqs_up_kmers)):
            file.write(str(seqs_up_kmers[i]) + '\n')

def get_max_alilen(seqs_up):
    len_max = []
    for i in range(len(seqs_up)):
        len_max.append(len(seqs_up[i]))
    return max(len_max)

# def trans_seq_hot_0110up(dna_sequences,ori_dna_sequences,sequence_length = 128):
#     # 确定最大长度
#     all_base_proportions = []
#     for seqsi in range(len(dna_sequences)):
#         seqs = dna_sequences[seqsi]
#         ori = ori_dna_sequences[seqsi]
#         seqs.append(ori)
#         encoded_sequences = []
#         # seqs_up = get_aliseqs(seqs)
#         ori_ali,seqs_up = get_ori_ali('./input_sequences.aln')
#         # print(seqs_up)
#         # seqs_up = readfile('aliseqs.fasta')
#         for seq in seqs_up:
#             # lens = len(seq)
#             # padded_seq = pad_sequence(seq, max_sequence_length)
#             encoded_seq = one_hot_encode(seq)
#             encoded_sequences.append(encoded_seq)
#
#         # seqs_num = len(dna_sequences[0])
#         encoded_sequences = np.array(encoded_sequences)
#         sum_per_position = np.sum(encoded_sequences, axis=0)
#         sum_per_position_up = []
#         position = {}
#         length = len(sum_per_position)
#         nn = len(sum_per_position[0])
#         for k in range(length):
#             base_num = sum_per_position[k]
#             if base_num[4] != 0:
#                 position[k] = int(base_num[4])
#             # if base_num[4] < nn//2+1:
#             #     sum_per_position_up.append(base_num)
#         sorted_dict = sorted(position.items(), key=lambda item: item[1], reverse=True)
#         sorted_keys = [item[0] for item in sorted_dict]
#         sorted_values = np.array([item[1] for item in sorted_dict[:(length-sequence_length)]])
#         sorted_keys = sorted_keys[:(length-sequence_length)]
#         sorted_keys_numpy = np.array(sorted_keys)
#         for k in range(length):
#             if k not in sorted_keys:
#                 sum_per_position_up.append(sum_per_position[k])
#
#         sum_per_position_up = np.array(sum_per_position_up)[:,:4]
#         # print('----------------------------------------------------------------------------')
#         # print(len(sum_per_position_up))
#         writefile(seqs_up, 'aliseqs11.fasta')
#         print(len(sum_per_position_up))
#         sleep(300)
#         # if len(sum_per_position_up) != sequence_length:
#         #     writefile(seqs_up, 'aliseqs11.fasta')
#         #     print(len(sum_per_position_up))
#         #     sleep(300)
#         sum_per_position = softmax(sum_per_position_up)
#         all_base_proportions.append(sum_per_position)
#     return all_base_proportions

# def trans_seq_hot(dna_sequences,max_sequence_length = 128):
# def trans_seq_hot_beforeafter(dna_sequences,max_sequence_length = 128):
#     # 确定最大长度
#     med_num = max_sequence_length//2
#     # max_sequence_length = max(len(seq) for seq in dna_sequences)
#     # max_sequence_length = 128
#     all_base_proportions = []
#     all_base_proportionssum = []
#     for seqs in dna_sequences:
#         # 对每条 DNA 序列进行独热编码和填充
#         # max_sequence_length = max(len(seq) for seq in seqs)
#         encoded_sequences_before = []
#         encoded_sequences_after = []
#         encoded_sequences = []
#         for seq in seqs:
#             lens = len(seq)
#             # padded_seq = pad_sequence(seq, max_sequence_length)
#             encoded_seq1 = one_hot_encode(seq[:med_num])
#             encoded_seq2 = one_hot_encode(seq[-med_num:])
#             encoded_sequences_before.append(encoded_seq1)
#             encoded_sequences_after.append(encoded_seq2)
#         for i in range(len(encoded_sequences_before)):
#             merged_array = np.concatenate((encoded_sequences_before[i], encoded_sequences_after[i]), axis=0)
#             encoded_sequences.append(merged_array)
#         # 转换为 numpy 数组
#         seqs_num = len(dna_sequences[0])
#         encoded_sequences = np.array(encoded_sequences)
#         sum_per_position = np.sum(encoded_sequences, axis=0)
#         sum_per_position = softmax(sum_per_position)
#         all_base_proportions.append(sum_per_position)
#     return all_base_proportions
#
# def trans_seq_hot_ori(dna_sequences,max_sequence_length = 128):
#     # 确定最大长度
#     # max_sequence_length = max(len(seq) for seq in dna_sequences)
#     # max_sequence_length = 128
#     all_base_proportions = []
#     all_base_proportionssum = []
#     for seqs in dna_sequences:
#         # 对每条 DNA 序列进行独热编码和填充
#         # max_sequence_length = max(len(seq) for seq in seqs)
#         encoded_sequences = []
#         for seq in seqs:
#             # lens = len(seq)
#             padded_seq = pad_sequence(seq, max_sequence_length)
#             encoded_seq = one_hot_encode(padded_seq)
#             encoded_sequences.append(encoded_seq)
#
#         # 转换为 numpy 数组
#         seqs_num = len(dna_sequences[0])
#         encoded_sequences = np.array(encoded_sequences)
#         sum_per_position = np.sum(encoded_sequences, axis=0)
#         # all_base_proportions.append(sum_per_position)
#         # sum_per_position = [(nums[0]*1+nums[1]*2+nums[2]*3+nums[3]*4) for nums in sum_per_position]
#         sum_per_position = softmax(sum_per_position)
#         # sum_per_position = [(nums[0]/seqs_num*1+nums[1]/seqs_num*2+nums[2]/seqs_num*3+nums[3]/seqs_num*4) for nums in sum_per_position]
#         # sum_per_position = [(nums[0]*1+nums[1]*2+nums[2]*3+nums[3]*4) for nums in sum_per_position]
#         # sum_per_position = [(nums[1]*1+nums[2]*2+nums[3]*3) for nums in sum_per_position]
#         # sum_per_position = normalize_list(sum_per_position)
#         all_base_proportions.append(sum_per_position)
#         # sum_per_position = np.sum(sum_per_position, axis=0)
#         # sum_per_position = [((nums[0]*0+nums[1]*1+nums[2]*2+nums[3]*3)) for nums in sum_per_position]
#         # sum_per_position = [(nums[0]*1+nums[1]*2+nums[2]*3+nums[3]*4) for nums in sum_per_position]
#         # sum_per_position = [((nums[0]*1+nums[1]*2+nums[2]*3+nums[3]*4)-1)/5 for nums in sum_per_position]
#         # sum_per_position = [((nums[0]*0+nums[1]*1+nums[2]*2+nums[3]*3)) for nums in sum_per_position]
#         # from sklearn.preprocessing import StandardScaler
#         # from sklearn.preprocessing import StandardScaler
#         # sum_per_position = normalize_list(sum_per_position)
#         # sum_per_position = standardize_list(sum_per_position)
#         # base_proportions = calculate_base_proportions(encoded_sequences)
#         # base_proportions = calculate_base_proportions(np.array([sum_per_position]))
#     return all_base_proportions


# def trans_ones_hot111(dna_sequences):
#     # 确定最大长度
#     # max_sequence_length = max(len(seq) for seq in dna_sequences)
#     max_sequence_length = 128
#     encoded_sequences = []
#     for seq in dna_sequences:
#         padded_seq = pad_sequence(seq, max_sequence_length)
#         encoded_seq = one_hot_encode(padded_seq)
#         encoded_sequences.append(encoded_seq)
#
#     # 转换为 numpy 数组
#     encoded_sequences = np.array(encoded_sequences)
#     # sum_per_position = np.sum(encoded_sequences, axis=0)
#     # base_proportions = calculate_base_proportions(sum_per_position)
#     return base_proportions
# dna_sequences = read_dna_file('test.fasta')
# embedding_data = trans_one_hot(dna_sequences)




# print('done')
