import copy
import csv
import math
import random

import Levenshtein
from joblib import Parallel, delayed

from .glob_var import (segment_length, threads_number, index_base_matrix_id_length, index_base_row_id_length,
                       matrix_num, data_num, index_num_1, index_num_2, index_binary_length_1,
                       mark_list, index_base_length_new, threads_number, kmer_len, threshold, num_perm,
                       N_index_2, get_selected_number,
                       check_length_cluster, file_path_matrix_index, file_path_list_index, compute_edit_dis_len,
                       primer_left_len, primer_right_len, segment_length_parity, osdir, parity_length, check_length,
                       match, mismatch, score_tuple, gap)

def file_to_binary_sequence(file_path):
    # 读取任意文件的二进制序列
    try:
        with open(file_path, 'rb') as file:
            content = file.read()
            binary_sequence = ''.join(format(byte, '08b') for byte in content)
            return binary_sequence
    except FileNotFoundError:
        print(f"文件 '{file_path}' 未找到。")
    except Exception as e:
        print(f"发生错误：{e}")

def file_binary_to_random(binary_sequence_ori):

    # 与圆周率的前500000位，进行异或操作，以实现随机化
    # with open('./utils/pi_str.txt', 'rb') as file:
    #     content = file.read()
    #     pi_binary_sequence = ''.join(format(byte, '08b') for byte in content)
    # 以 r 的形式呢，这个不行，一定要rb才ok，用r，读到的是 3.14159.......
    # with open('./utils/pi_str.txt', 'r') as file:
    #     pi_binary_sequence = file.read().strip()
    # # print("pi_binary_sequence的长度：", len(pi_binary_sequence))
    # str_1, str_2 = binary_sequence_ori, pi_binary_sequence

    # # Pi的值直接转换成01序列，其随机性似乎不太好，有许多的连续0或1。下面是用random生成的随机01序列，试试效果
    # random_01_str = ''
    with open(osdir + './params/random_01string_1e7.txt', 'r') as file:   # 注意 r 与 rb 是不同的
        random_01_str = file.read().strip()  # 读取文件并移除首尾的空白字符（包括换行符）
        # content = file.read().strip()  # 读取文件并移除首尾的空白字符（包括换行符）
        # random_01_str = ''.join(format(byte, '01b') for byte in content)
    # print("random_01_str的长度：", len(random_01_str))
    str_1, str_2 = binary_sequence_ori, random_01_str

    len_str_1 = len(str_1)
    len_str_2 = len(str_2)
    # 使用列表推导式生成结果字符串,获取str_2中对应位置的字符，如果长度不够则循环使用
    binary_sequence_random = ''.join(str(int(str_1[i]) ^ int(str_2[i % len_str_2])) for i in range(len_str_1))
    # print("binary_sequence_ori的长度：", len(binary_sequence_ori))
    # print("binary_sequence_random的长度：", len(binary_sequence_random))
    return binary_sequence_random

def read_binary_new(binary_sequence):
    # 这里添加新功能，每行加 2 位奇偶校验位，两个校验位为：偶+奇。保证每行异或运算的结果为0，240816
    size = len(binary_sequence)
    # print("len(img_encode):", len(data_encode), type(data_encode))
    # print("img_encode的第一个元素类型：", type(data_encode[0]))
    size_B = math.ceil(size / 8)
    size_KB = math.ceil(size / (8 * 1024))
    print("size_KB = %d, size_B = %d" % (size_KB, size_B))
    # 因为szie已经是bit的位数，所以下面size不用 * 8
    # 下面这样初始化matrix是ok的，因为[0 for _ in range(segment_length)]每次都是新生成的
    matrix = [[0 for _ in range(segment_length_parity)] for _ in range(math.ceil(size / segment_length_parity))]
    # 显然这样的处理下，matrix的最后的一行与原来的数据相比会多一些0,多了多少个0，要记录下来，在按行排好序后，应该把对应的最大行的数据中的0去掉
    for idx in range(len(binary_sequence)):
        row = idx // segment_length_parity
        col = idx % segment_length_parity
        # 我还是有点不太理解这个bug，就算img_encode中的0/1是字符型，也只会影响编码啊，我看解码部分输出的是int型呢，怎么这里无int转换会影响
        matrix[row][col] = int(binary_sequence[idx])  # 红玫的建议，修好一个大bug
        # 如果用字符串的形式来保存每行数据，确实可以减少内存占用，但怎么实现，还得想想
    # print("原来matrix的行数：", len(matrix))
    more_zeros = segment_length_parity - size % segment_length_parity  # 注意：当more_zeros = segment_length时，表示more_zeros为0
    if more_zeros == segment_length_parity:
        more_zeros = 0
    print("编码前more_zeros:", more_zeros)
    # 最后一行的more_zeros还是有影响，进行随机化更好。
    max_true_row_num = len(matrix) - 1
    if more_zeros != 0:
        random_list = [random.randint(0, 1) for _ in range(more_zeros)]
        matrix[max_true_row_num][-more_zeros:] = random_list  # 将最后一行多余的0，进行随机化
    # print("随机化后，原数据的最后一行：", matrix[max_true_row_num], len(matrix[max_true_row_num]))
    # print("原数据最后一行数据的more_zeros：", more_zeros)
    # print("原来的最后一行数据：", matrix[max_true_row_num][:])
    # print("原来的最后一行数据,去掉多余的0：", matrix[max_true_row_num][:-more_zeros])
    # 把more_zeros信息，存放在matrix矩阵的最后的行，more_zeros为后10个bits
    # 一定要小心，要确保在matrix矩阵中的0/1是int类型，不然极化码会解码失败
    binary_string_more_zeros = list(int(bit) for bit in format(more_zeros, '010b'))  # more_zeros转化成二进制序列长为十
    # print("binary_string:", binary_string, len(binary_string))
    binary_string_len = len(binary_string_more_zeros)
    # 这里把多余的0的信息放在了后面
    last_row = [random.randint(0, 1) for _ in range(segment_length_parity - binary_string_len)] + binary_string_more_zeros
    # print("last_row:", last_row, len(last_row))
    matrix.append(last_row)
    print("原数据矩阵的最大行号：", len(matrix) - 1)
    # 对每一行添加 2 位的奇偶检验位，两个校验位为：偶+奇。保证每行异或后为0
    for my_list in matrix:
        even_parity, odd_parity = 0, 0   # 偶校验位， 奇校验位
        for i in range(0, len(my_list), 2):
            even_parity ^= my_list[i]
            odd_parity ^= my_list[i + 1]
        my_list.append(even_parity)   # 240
        my_list.append(odd_parity)  # 241
    return matrix

def add_matrix_and_row_numbers_new(matrices):  # 双层index方案中，index_dna也由两个小矩阵构成
    print("块地址与块内地址 分离！")
    # 使用块（矩阵）地址与块内地址分开的方案，可在保持index_base长15的情况下，增加可用矩阵数量
    # 打开文件并读取index_base的DNA序列
    # file_path = "./utils/index_base_have_complementary_and_reverse_1505_13312.txt"
    # file_path_matrix_index = "./utils/index_base_no_think_complementary_reverse_seq_0805_21.txt"  # 块地址
    # file_path_matrix_index = "./utils/index_base_no_think_complementary_reverse_seq_0705_11.txt"  # 块地址
    # 初始化一个空列表来存储DNA序列
    index_base_matrix = []
    try:
        with open(file_path_matrix_index, "r") as file:
            # 逐行读取文件，并将每行的内容添加到列表中
            for line in file:
                # 去除行末尾的换行符并将DNA序列添加到列表
                dna_sequence = line.strip()
                index_base_matrix.append(dna_sequence)
    except FileNotFoundError:
        print(f"File '{file_path_matrix_index}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    # file_path_list_index = "./utils/output_have_complementary_10_03_all_3500.txt"  # 块内地址
    # file_path_list_index = "./utils/index_base_no_think_complementary_reverse_seq_1305_2763.txt"  # 块内地址
    # 初始化一个空列表来存储DNA序列
    index_base_list = []
    try:
        with open(file_path_list_index, "r") as file:
            # 逐行读取文件，并将每行的内容添加到列表中
            for line in file:
                # 去除行末尾的换行符并将DNA序列添加到列表
                dna_sequence = line.strip()
                index_base_list.append(dna_sequence)
    except FileNotFoundError:
        print(f"File ???'{file_path_matrix_index}' found.{len(index_base_matrix)}")
        print(f"File ???'{file_path_list_index}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    # print(f'?????????????????????{len(index_base_list)}')
    # print("index_base_com_and_rev:", len(index_base_com_and_rev))
    # print("index_base_com_and_rev[100]:", index_base_com_and_rev[100])
    dna_sequences = []
    for i in range(len(matrices)):  # i 表示矩阵号
        # index_base_matrix表示矩阵号(块地址)，index_base_list表示行号(块内地址)
        # 分开矩阵号与行号的原因是：在聚类时，先聚到矩阵，再聚到行，每条DNA序列只用算2000次编辑距离，而不用算100万
        # 处理数据部分
        for j in range(len(matrices[i][0])):
            dna_sequence = index_base_matrix[i] + index_base_list[j] + matrices[i][0][j]
            dna_sequences.append(dna_sequence)
        # 处理index部分
        for j in range(len(matrices[i][1])):
            for k in range(len(matrices[i][1][j])):
                index_base_num = data_num + j * index_num_1 + k
                dna_sequence = index_base_matrix[i] + index_base_list[index_base_num] + matrices[i][1][j][k]
                dna_sequences.append(dna_sequence)
    # del matrices  # 原来的DNA序列矩阵不再需要
    return dna_sequences

def dna_seq2matrices_new(dna_sequences):  # 双层index方案, 实现块地址与块内地址的分离
    print("块地址与块内地址 分离！ 支持质量值+修改index_base")
    # file_path_matrix_index = "./utils/index_base_no_think_complementary_reverse_seq_0805_21.txt"  # 块地址
    # file_path_matrix_index = "./utils/index_base_no_think_complementary_reverse_seq_0705_11.txt"  # 块地址
    # 初始化一个空列表来存储DNA序列
    index_base_matrix = []
    try:
        with open(file_path_matrix_index, "r") as file:
            # 逐行读取文件，并将每行的内容添加到列表中
            for line in file:
                # 去除行末尾的换行符并将DNA序列添加到列表
                dna_sequence = line.strip()
                index_base_matrix.append(dna_sequence)
    except FileNotFoundError:
        print(f"File '{file_path_matrix_index}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    # file_path_list_index = "./utils/output_have_complementary_10_03_all_3500.txt"  # 块内地址
    # file_path_list_index = "./utils/index_base_no_think_complementary_reverse_seq_1305_2763.txt"  # 块内地址
    # 初始化一个空列表来存储DNA序列
    index_base_list = []
    try:
        with open(file_path_list_index, "r") as file:
            # 逐行读取文件，并将每行的内容添加到列表中
            for line in file:
                # 去除行末尾的换行符并将DNA序列添加到列表
                dna_sequence = line.strip()
                index_base_list.append(dna_sequence)
    except FileNotFoundError:
        print(f'1111111111111111111111111')
        print(f"File '{file_path_list_index}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    print(f'1111111111111111111111111{len(index_base_list)}')
    # 先对单位矩阵进行初始化
    matrix_data = [[[], []] for _ in range(data_num)]  # 注意到这里以DNA序列为目标，这个声明没有问题
    matrix_index_1 = [[[], []] for _ in range(index_num_1)]
    matrix_index_2 = [[[], []] for _ in range(index_num_2)]
    matrix_index = [matrix_index_1, matrix_index_2]
    my_matrix = [matrix_data, matrix_index]  # 这里应该没有问题
    # 这里其实有bug的风险，matrix_num比一个DNA单位矩阵中的DNA序列条数要大
    matrix_count_float = len(dna_sequences) / matrix_num
    matrix_count_up = math.ceil(len(dna_sequences) / matrix_num)
    matrix_count_down = len(dna_sequences) // matrix_num
    # 因为聚类后DNA序列的数量会稍大一点，所以当最后一个矩阵为last_3时，matrix_count_float会比真实matrix_count稍大。但通常来讲，因为matrix_num偏大，所以通常往上取才对。
    # 因此，不能以更靠近的方式来定，而是稍大于（eg:0.15）向下取，其它情况向上取，更合理。
    if matrix_count_float - matrix_count_down < 0.15:
        matrix_count = matrix_count_down
    else:
        matrix_count = matrix_count_up
    print("matrix_count_float = %.3f, matrix_count_up = %d, matrix_count_down = %d" % (
        matrix_count_float, matrix_count_up, matrix_count_down))
    print("matrix_count =", matrix_count)
    # my_matrices = [my_matrix for _ in range(matrix_count)]  # 有可能是这里有问题，就是python列表复制的问题, 终于找到了，就是这个bug
    # 上面的代码，会让my_matrices中的每一个单位矩阵是一样的
    my_matrices = []
    # 可以直接用这个大矩阵处理完，因为last_1和last_2的最后一个index矩阵只是DNA条数少一点，这样缺少的地方会有空列表，但空列表不会影响，因为后面在转01bits时，会先处理掉空列表
    for _ in range(matrix_count):
        matrix_temp = copy.deepcopy(my_matrix)  # 一定要用deepcopy()
        my_matrices.append(matrix_temp)
    # 这里用并行，会有点麻烦。可以输出matrix_idx和row_idx，然后根据这两个值去分类
    # results = Parallel(n_jobs=threads_number)(
    #     delayed(cal_matrix_and_row_idx_new)(dna_sequence, index_base_matrix, index_base_list)
    #     for dna_sequence in dna_sequences)
    # 下面尝试将块地址与块内地址一起考虑
    results = Parallel(n_jobs=threads_number)(
        delayed(cal_matrix_and_row_idx_new)(dna_sequence, index_base_matrix, index_base_list)
        for dna_sequence in dna_sequences)
    count = 0
    for i in range(len(results)):
        if results[i][0] >= matrix_count:  # 真的有这种情况
            print("出错啦!算出的块地址偏大。results[i][0] = %d, matrix_count = %d" % (results[i][0], matrix_count))
            count += 1
            continue
        dna_seq_have_phred_temp = [results[i][2][0][index_base_length_new:], results[i][2][1][index_base_length_new:]]
        # if i == 100:
        #     print("i == 100 时：dna_seq_have_phred_temp的类型为：", type(dna_seq_have_phred_temp))
        #     print("i == 100 时：dna_seq_have_phred_temp[0]为：", dna_seq_have_phred_temp[0], type(dna_seq_have_phred_temp[0]), len(dna_seq_have_phred_temp[0]))
        #     print("i == 100 时：dna_seq_have_phred_temp[1]为：", dna_seq_have_phred_temp[1],
        #           type(dna_seq_have_phred_temp[1]), len(dna_seq_have_phred_temp[1]))
        if results[i][1] < data_num:  # 数据部分的矩阵
            my_matrices[results[i][0]][0][results[i][1]] = dna_seq_have_phred_temp  # 去掉前面的index_base行号
        elif results[i][1] < data_num + index_num_1:  # index部分的第1个矩阵
            my_matrices[results[i][0]][1][0][results[i][1] - data_num] = dna_seq_have_phred_temp  # 去掉前面的index_base行号
        else:  # index部分的第2个矩阵
            temp_n = results[i][1] - data_num - index_num_1
            if temp_n >= index_num_2:
                # print("出错啦！row_id偏大，results[i][1] - data_num - index_num_1 = %d, index_num_c = %d" % (
                #     temp_n, index_num_2))
                count += 1
                continue
            my_matrices[results[i][0]][1][1][
                results[i][1] - data_num - index_num_1] = dna_seq_have_phred_temp  # 去掉前面的index_base行号
    print("恢复原来DNA矩阵时，发现算出块地址或row地址偏大的数量为 %d" % count)
    # 我们要注意到在这个matrices中，各个单位矩阵中，预留出来的地方会有一个空列表占位置，这不会影响后面将DNA序列还原成01bits的操作
    # my_matrices有问题，各个单位矩阵的DNA序列竟然是一样的，不知道为什么？ 答：已经解决，单位矩阵的初始化，要注意列表的复制规则。
    # 清洗最后一个矩阵的index_1的第2个矩阵，方便后面判断last_case
    # 只有最后一个矩阵的index_1的第2个矩阵全是空列表，才应该像下面这样清洗
    # 更新，考虑到聚类后，DNA序列数量可能会变多。所以，这里应该让index_1的第2个矩阵中序列数量大于一个阈值才可以判定为存在index_1的第2个矩阵
    # print("my_matrices[0][0][0]:", my_matrices[0][0][0], len(my_matrices[0][0][0]), type(my_matrices[0][0][0]))
    # print("my_matrices[0][1][0][0]:", my_matrices[0][1][0][0], len(my_matrices[0][1][0][0]), type(my_matrices[0][1][0][0]))
    # 数一数DNA矩阵中真实DNA序列的条数，方便统计DNA序列的丢失率
    dna_seqs_ture_num = 0
    for matrix in my_matrices:
        for dna_seq_and_phred in matrix[0]:  # 检查数据DNA矩阵部分
            if dna_seq_and_phred[0]:  # dna_seq_and_phred[0] ！= []
                dna_seqs_ture_num += 1
        for matrix_index in matrix[1]:  # 检查index_DNA矩阵部分
            for dna_seq_and_phred in matrix_index:
                if dna_seq_and_phred[0]:
                    dna_seqs_ture_num += 1
    print("发现恢复的DNA矩阵中，真实DNA条数 %d" % dna_seqs_ture_num)
    return my_matrices, dna_seqs_ture_num

def cal_matrix_and_row_idx_new(dna_seq_and_phred, index_base_matrix, index_base_list):
    # 支持质量值 + 对原DNA序列中index_base的修改
    min_distance_matrix, min_distance_row = 10000, 10000
    matrix_idx, row_idx = 100000, 100000
    for i in range(len(index_base_matrix)):
        edit_distance_temp = Levenshtein.distance(dna_seq_and_phred[0][:index_base_matrix_id_length],
                                                  index_base_matrix[i])
        if edit_distance_temp < min_distance_matrix:
            min_distance_matrix = edit_distance_temp
            matrix_idx = i
    _, _, _, dna_seq_and_phred_modify = check_and_edit_dna_sequence(index_base_matrix[matrix_idx], dna_seq_and_phred, 0)
    for i in range(len(index_base_list)):
        edit_distance_temp = Levenshtein.distance(
            dna_seq_and_phred_modify[0][index_base_matrix_id_length:index_base_length_new],
            index_base_list[i])
        if edit_distance_temp < min_distance_row:
            min_distance_row = edit_distance_temp
            row_idx = i
    _, _, _, dna_seq_and_phred_modify = check_and_edit_dna_sequence(index_base_list[row_idx], dna_seq_and_phred_modify,
                                                                    index_base_matrix_id_length)
    return matrix_idx, row_idx, dna_seq_and_phred_modify

def check_and_edit_dna_sequence(dna_sequence_decode, dna_sequence_receive, temp_n_0):
    # 20240831，考虑到可能会解码失败，试试尽量少动质量值。效果还可以，与适当修改对比一下。
    # 现在试试适当修改 20240913
    # 这里修改质量值的方法要再想想，主要是没有考虑若某列01序列解码失败的影响。是不是不去动质量值，会更好。
    # 支持质量值
    # 如果dna_sequence_receive是一个空列表会怎么样？
    # 答：看起空列表，没有办法修改。试试用全'C'。
    # 上面传进来的两个参数，应该都是string
    # dna_sequence_receive含两个元素，[0]是DNA序列string，[1]是质量值序列list
    phred_list_copy = dna_sequence_receive[1].copy()
    # print(type(phred_list_copy), phred_list_copy)
    dna_sequence_rec_copy = list(dna_sequence_receive[0])
    if not dna_sequence_rec_copy:
        for _ in range(segment_length):
            dna_sequence_rec_copy.append('C')  # 与初始全0能对上
            phred_list_copy.append(0.25)  # 碱基的可靠度，默认为0.25才对
    # print("dna_sequence_receive:", type(dna_sequence_receive), len(dna_sequence_receive))
    # print("dna_sequence_rec_copy:", type(dna_sequence_rec_copy), len(dna_sequence_rec_copy))
    num_sub, num_del, num_ins = 0, 0, 0
    for i in range(len(dna_sequence_decode)):
        j = i + temp_n_0
        if j >= len(dna_sequence_rec_copy):  # 发生了删除错误
            # print("i =", i)
            dna_sequence_rec_copy.append(dna_sequence_decode[i])
            phred_list_copy.append(0.5)   # 这里有一个问题，这里没有考虑该碱基对应的列是否有通过CRC检验。实际上，若没有通过，则解码得到的碱基可信度是较低的。
            # phred_list_copy.append(0.3)   # 这里还得加质量值，但也可能会出错。给一个较低的值吧，0.25是平均状态，无信息
        elif dna_sequence_rec_copy[j] != dna_sequence_decode[i]:
            # 依次尝试替换、删除、插入错误
            temp_char = dna_sequence_rec_copy[j]  # 这里真的有bug，如果dna_sequence_receive是空列表，这个就无效了
            # print("temp_char:", temp_char, type(temp_char))
            # 丁老师的建议，用二条序列比对来输出可能的错误类型
            # 尝试替换错误
            dna_sequence_rec_copy[j] = dna_sequence_decode[i]
            # Levenshtein.distance可以直接计算列表和string
            edit_distance_sub = Levenshtein.distance(dna_sequence_rec_copy[j:j + check_length],
                                                     dna_sequence_decode[i: i + check_length])
            dna_sequence_rec_copy[j] = temp_char  # 恢复原状
            # 尝试删除错误
            dna_sequence_rec_copy.insert(j, dna_sequence_decode[i])
            edit_distance_del = Levenshtein.distance(dna_sequence_rec_copy[j:j + check_length],
                                                     dna_sequence_decode[i: i + check_length])
            del dna_sequence_rec_copy[j]  # 恢复原状
            # 尝试插入错误
            del dna_sequence_rec_copy[j]
            edit_distance_ins = Levenshtein.distance(dna_sequence_rec_copy[j:j + check_length],
                                                     dna_sequence_decode[i: i + check_length])
            dna_sequence_rec_copy.insert(j, temp_char)  # 恢复原状
            if edit_distance_sub <= edit_distance_del and edit_distance_sub <= edit_distance_ins:  # 判定为替换错误
                num_sub += 1
                dna_sequence_rec_copy[j] = dna_sequence_decode[i]
                phred_list_copy[j] = 0.8  # 替换错误，修改对应的质量值，试试不再修改质量值，给个0.8试试
            # 判定为删除错误,试试优先判定为删除错误，对3代会不会更好？ 0605，似乎更差了，换回去试试
            elif edit_distance_del <= edit_distance_sub and edit_distance_del <= edit_distance_ins:
                # 在3代中，删除错误比插入错误略高一点
                num_del += 1
                dna_sequence_rec_copy.insert(j, dna_sequence_decode[i])
                phred_list_copy.insert(j, 0.8)
                # phred_list_copy.insert(j, 0.3)   # 这里还得加质量值，但也可能会出错。给一个较低的值吧
            else:  # 判定为插入错误
                num_ins += 1
                del dna_sequence_rec_copy[j]
                del phred_list_copy[j]
    # 检查phred_list_copy的长度，保证其长度为segment_length。不需要了，因为在函数base_phred_to_01_phred中会保证输出的质量值长度为segment_length
    # if len(phred_list_copy) < segment_length:
    #     num = segment_length - len(phred_list_copy)
    #     for _ in range(num):
    #         phred_list_copy.append(0.5)
    # else:
    #     phred_list_copy = phred_list_copy[:segment_length]
    dna_sequence_rec_copy = ''.join(dna_sequence_rec_copy)
    dna_sequence_modify = [dna_sequence_rec_copy, phred_list_copy]
    return num_sub, num_del, num_ins, dna_sequence_modify

def check_dna_vertical_error_hamming(matrices_dna, matrices_dna_ori, file_name, file_num):
    # 与上面的函数不同，这里是直接对比，类似于hamming距离
    # matrices_dna_ori 没有块地址和块内地址
    # 检查数据和index部分的列向错误，并统计每一个错误的行、列坐标，
    # 小心matrices_dna中的某行DNA序列可能丢失，长度可能也不对，matrices_dna的数据/index部分DNA矩阵的行数量会偏多，多的是空列表[]
    # matrices_dna的每一个DNA序列是和一个phred(list)结合起来的
    # matrices_dna的最后一个index矩阵一定包含了两个矩阵，后一个可能为空，matrices_dna_ori的最后一个index矩阵，也由2个小矩阵构成，当last_1 或 last_2时，后一个小矩阵为[]
    # 在matrices_dna_and_phred的最后一个矩阵中的index矩阵，其都用[[],[]]初始化了每一个DNA序列的位置
    print("开始竖向检查错误_hamming：在恢复DNA矩阵后，与原DNA矩阵对比，以检查错误分布")
    # matrices_dna = copy.deepcopy(matrices_dna_and_phred)
    mutate_num_all = 0
    matrix_num = len(matrices_dna_ori)
    errors_data_dict = dict()
    errors_index_dict = dict()
    for i in range(len(matrices_dna_ori)):
        # j = 0, 处理DNA数据矩阵
        results = Parallel(n_jobs=2 * threads_number)(
            delayed(handle_dna_matrix_hamming)(matrices_dna[i][0][k][0], matrices_dna_ori[i][0][k])
            for k in range(len(matrices_dna_ori[i][0])))
        for result in results:
            mutate_num_all += result[0]
            # 发生插入错误后，若该条DNA序列（横向）后面还有错误，则列号会+1，不过发生概率不高，先不管
            for [error_type, col_num] in result[1]:
                key = str(i) + '_data_' + str(0) + str(col_num)
                if key not in errors_data_dict:
                    errors_data_dict[key] = []
                errors_data_dict[key].append([error_type, col_num])
        # j = 1，处理index矩阵
        # j = 1 and k = 0, 处理index的第1个矩阵
        results = Parallel(n_jobs=2 * threads_number)(
            delayed(handle_dna_matrix_hamming)(matrices_dna[i][1][0][p][0], matrices_dna_ori[i][1][0][p])
            for p in range(len(matrices_dna_ori[i][1][0])))
        for result in results:
            mutate_num_all += result[0]
            for [error_type, col_num] in result[1]:
                key = str(i) + '_index_' + str(10) + str(col_num)
                if key not in errors_index_dict:
                    errors_index_dict[key] = []
                errors_index_dict[key].append([error_type, col_num])
        # j = 1 and k = 1, 处理index的第2个矩阵
        # matrices_dna_ori[i][1][1] 可能是[]
        if matrices_dna_ori[i][1][1]:   # 不为[]才需要处理
            if matrices_dna[i][1][1][0][0]:  # 注意到matrices_dna_and_phred的index_2的每一行DNA序列位置，都是用[[],[]]进行初始化的
                results = Parallel(n_jobs=2 * threads_number)(
                    delayed(handle_dna_matrix_hamming)(matrices_dna[i][1][1][p][0], matrices_dna_ori[i][1][1][p])
                    for p in range(len(matrices_dna_ori[i][1][1])))
                for result in results:
                    mutate_num_all += result[0]
                    for [error_type, col_num] in result[1]:
                        key = str(i) + '_index_' + str(11) + str(col_num)
                        if key not in errors_index_dict:
                            errors_index_dict[key] = []
                        errors_index_dict[key].append([error_type, col_num])
            else:
                if i == matrix_num - 1:  # 最后一个矩阵才需要检查
                    # 这种情况应该不会发生
                    print("出错了，请修改代码，matrices_dna_ori[i][1][1]不为空，但matrices_dna[i][1][1]为空，i = %d" % i)
    bp_num_all = 0
    for matrix in matrices_dna_ori:
        for seq in matrix[0]:
            bp_num_all += len(seq)
        for matrix_index in matrix[1]:
            for seq in matrix_index:
                bp_num_all += len(seq)
    # print("bp_num_all = %d" % bp_num_all)
    mutate_ration = mutate_num_all / bp_num_all
    all_ration = mutate_ration
    print("deeplearning清洗+恢复原DNA矩阵形式后所得DNA序列 不含primer，已经算上了丢失整条DNA序列的影响，直接对比，类似hamming距离")
    print("替换错误率为 %.6f, 总错误率为 %.6f"
          % (mutate_ration, all_ration))
    # 统计列方向上的错误情况
    errors_analysis_path =  file_name +'hamming_errors_analysis_dna_' + '_' + str(file_num) + '.txt'
    row_len = len(matrices_dna_ori[0][0][0])   # 横向长度，即列号，是一样的
    print("row_len =", row_len)
    col_data_num = len(matrices_dna_ori[0][0])
    min_disp_error = 0.002
    col_data_num_last = len(matrices_dna_ori[matrix_num - 1][0])
    col_index_1_num = len(matrices_dna_ori[0][1][0])
    col_index_2_num = len(matrices_dna_ori[0][1][1])
    col_index_1_num_last = len(matrices_dna_ori[matrix_num - 1][1][0])
    col_index_2_num_last = len(matrices_dna_ori[matrix_num - 1][1][1])   # 这个值可能为0
    with open(errors_analysis_path, 'w') as f:
        f.write("下面是各个DNA矩阵的 列 错误率, 未出现的没有出错\n")
        for i in range(len(matrices_dna_ori)):
            f.write('\n')
            f.write("第 %d 个单位矩阵的 列 错误率\n" % i)
            for j in range(len(matrices_dna_ori[i])):
                if j == 0:  # DNA数据矩阵
                    f.write("DNA数据矩阵的 列 错误率, 小于 %.3f 的不显示\n" % min_disp_error)
                    if i == matrix_num - 1:
                        col_data_num_real = col_data_num_last
                    else:
                        col_data_num_real = col_data_num
                    for k in range(row_len):
                        key = str(i) + '_data_' + str(0) + str(k)
                        if key in errors_data_dict:
                            column_error_ration = len(errors_data_dict[key]) / col_data_num_real
                            if column_error_ration >= min_disp_error:
                                f.write("第 %d 列的错误率为 %.6f \n" % (k, column_error_ration))
                else:  # j = 1, index 矩阵
                    for k in range(len(matrices_dna_ori[i][1])):
                        f.write("index_%d 矩阵的 列 错误率, 小于 %.3f 的不显示\n" % (k, min_disp_error))
                        if i == matrix_num - 1:
                            col_index_1_num_real = col_index_1_num_last
                            col_index_2_num_real = col_index_2_num_last
                        else:
                            col_index_1_num_real = col_index_1_num
                            col_index_2_num_real = col_index_2_num
                        if k == 0:
                            for p in range(row_len):
                                key = str(i) + '_index_' + str(10) + str(p)
                                if key in errors_index_dict:
                                    column_error_ration = len(errors_index_dict[key]) / col_index_1_num_real
                                    # f.write("第 %d 列的错误率为 %.6f \n" % (p, column_error_ration))
                                    if column_error_ration >= min_disp_error:
                                        f.write("第 %d 列的错误率为 %.6f \n" % (p, column_error_ration))
                        else:   # k = 1
                            for p in range(row_len):
                                key = str(i) + '_index_' + str(11) + str(p)
                                if key in errors_index_dict:
                                    column_error_ration = len(errors_index_dict[key]) / col_index_2_num_real
                                    if column_error_ration >= min_disp_error:
                                        f.write("第 %d 列的错误率为 %.6f \n" % (p, column_error_ration))
                                    # f.write("第 %d 列的错误率为 %.6f \n" % (p, column_error_ration))

def handle_dna_matrix_hamming(dna_seq_rec, dna_seq_ori):
    # 在直接对比中，只有替换错误，故不用返回错误类型
    mutate_num = 0
    errors_info = []
    if not dna_seq_rec:  # 收到的DNA序列为空，初始化全为'C'
        dna_seq_rec = ['C' for _ in range(len(dna_seq_ori))]
    len_dis = len(dna_seq_ori) - len(dna_seq_rec)
    if len_dis > 0:
        for _ in range(len_dis):
            dna_seq_rec += 'C'  # 不够长，则初上'C'
    for i in range(len(dna_seq_ori)):
        if dna_seq_ori[i] != dna_seq_rec[i]:
            mutate_num += 1
            errors_info.append([0, i])
    return mutate_num, errors_info

def check_dna_vertical_error_new(matrices_dna, matrices_dna_ori, file_name, file_num):
    # matrices_dna_ori 没有块地址和块内地址
    # 检查数据和index部分的列向错误，并统计每一个错误的行、列坐标，
    # 小心matrices_dna中的某行DNA序列可能丢失，长度可能也不对，matrices_dna的数据/index部分DNA矩阵的行数量会偏多，多的是空列表[]
    # matrices_dna的每一个DNA序列是和一个phred(list)结合起来的
    # matrices_dna的最后一个index矩阵一定包含了两个矩阵，后一个可能为空，matrices_dna_ori的最后一个index矩阵，也由2个小矩阵构成，当last_1 或 last_2时，后一个小矩阵为[]
    # 在matrices_dna_and_phred的最后一个矩阵中的index矩阵，其都用[[],[]]初始化了每一个DNA序列的位置
    print("开始竖向检查错误：在恢复DNA矩阵后，与原DNA矩阵对比，以检查错误分布")
    # matrices_dna = copy.deepcopy(matrices_dna_and_phred)
    mutate_num_all, insert_num_all, delete_num_all = 0, 0, 0
    matrix_num = len(matrices_dna_ori)
    errors_data_dict = dict()
    errors_index_dict = dict()
    for i in range(len(matrices_dna_ori)):
        # j = 0, 处理DNA数据矩阵
        results = Parallel(n_jobs=2 * threads_number)(
            delayed(handle_dna_matrix)(matrices_dna[i][0][k][0], matrices_dna_ori[i][0][k])
            for k in range(len(matrices_dna_ori[i][0])))
        for result in results:
            mutate_num_all += result[0]
            insert_num_all += result[1]
            delete_num_all += result[2]
            # 发生插入错误后，若该条DNA序列（横向）后面还有错误，则列号会+1，不过发生概率不高，先不管
            for [error_type, col_num] in result[3]:
                key = str(i) + '_data_' + str(0) + str(col_num)
                if key not in errors_data_dict:
                    errors_data_dict[key] = []
                errors_data_dict[key].append([error_type, col_num])
        # j = 1，处理index矩阵
        # j = 1 and k = 0, 处理index的第1个矩阵
        results = Parallel(n_jobs=2 * threads_number)(
            delayed(handle_dna_matrix)(matrices_dna[i][1][0][p][0], matrices_dna_ori[i][1][0][p])
            for p in range(len(matrices_dna_ori[i][1][0])))
        for result in results:
            mutate_num_all += result[0]
            insert_num_all += result[1]
            delete_num_all += result[2]
            for [error_type, col_num] in result[3]:
                key = str(i) + '_index_' + str(10) + str(col_num)
                if key not in errors_index_dict:
                    errors_index_dict[key] = []
                errors_index_dict[key].append([error_type, col_num])
        # j = 1 and k = 1, 处理index的第2个矩阵
        # matrices_dna_ori[i][1][1] 可能是[]
        if matrices_dna_ori[i][1][1]:   # 不为[]才需要处理
            if matrices_dna[i][1][1][0][0]:  # 注意到matrices_dna_and_phred的index_2的每一行DNA序列位置，都是用[[],[]]进行初始化的
                results = Parallel(n_jobs=2 * threads_number)(
                    delayed(handle_dna_matrix)(matrices_dna[i][1][1][p][0], matrices_dna_ori[i][1][1][p])
                    for p in range(len(matrices_dna_ori[i][1][1])))
                for result in results:
                    mutate_num_all += result[0]
                    insert_num_all += result[1]
                    delete_num_all += result[2]
                    for [error_type, col_num] in result[3]:
                        key = str(i) + '_index_' + str(11) + str(col_num)
                        if key not in errors_index_dict:
                            errors_index_dict[key] = []
                        errors_index_dict[key].append([error_type, col_num])
            else:
                if i == matrix_num - 1:  # 最后一个矩阵才需要检查
                    # 这种情况应该不会发生
                    print("出错了，请修改代码，matrices_dna_ori[i][1][1]不为空，但matrices_dna[i][1][1]为空，i = %d" % i)
    bp_num_all = 0
    for matrix in matrices_dna_ori:
        for seq in matrix[0]:
            bp_num_all += len(seq)
        for matrix_index in matrix[1]:
            for seq in matrix_index:
                bp_num_all += len(seq)
    print("bp_num_all = %d" % bp_num_all)
    mutate_ration = mutate_num_all / bp_num_all
    insert_ration = insert_num_all / bp_num_all
    delete_ration = delete_num_all / bp_num_all
    all_ration = mutate_ration + insert_ration + delete_ration
    print("deeplearning清洗+恢复原DNA矩阵形式后所得DNA序列 不含primer，已经算上了丢失整条DNA序列的影响")
    print("替换错误率为 %.6f,插入错误率为 %.6f,删除错误率为 %.6f, 总错误率为 %.6f"
          % (mutate_ration, insert_ration, delete_ration, all_ration))
    # 统计列方向上的错误情况
    errors_analysis_path =  file_name +'hamming_errors_analysis_dna_' + '_' + str(file_num) + '.txt'
    row_len = len(matrices_dna_ori[0][0][0])   # 横向长度，即列号，是一样的
    print("row_len =", row_len)
    col_data_num = len(matrices_dna_ori[0][0])
    min_disp_error = 0.002
    col_data_num_last = len(matrices_dna_ori[matrix_num - 1][0])
    col_index_1_num = len(matrices_dna_ori[0][1][0])
    col_index_2_num = len(matrices_dna_ori[0][1][1])
    col_index_1_num_last = len(matrices_dna_ori[matrix_num - 1][1][0])
    col_index_2_num_last = len(matrices_dna_ori[matrix_num - 1][1][1])   # 这个值可能为0
    with open(errors_analysis_path, 'w') as f:
        f.write("下面是各个DNA矩阵的 列 错误率, 未出现的没有出错\n")
        for i in range(len(matrices_dna_ori)):
            f.write('\n')
            f.write("第 %d 个单位矩阵的 列 错误率\n" % i)
            for j in range(len(matrices_dna_ori[i])):
                if j == 0:  # DNA数据矩阵
                    f.write("DNA数据矩阵的 列 错误率, 小于 %.3f 的不显示\n" % min_disp_error)
                    if i == matrix_num - 1:
                        col_data_num_real = col_data_num_last
                    else:
                        col_data_num_real = col_data_num
                    for k in range(row_len):
                        key = str(i) + '_data_' + str(0) + str(k)
                        if key in errors_data_dict:
                            column_error_ration = len(errors_data_dict[key]) / col_data_num_real
                            if column_error_ration >= min_disp_error:
                                f.write("第 %d 列的错误率为 %.6f \n" % (k, column_error_ration))
                else:  # j = 1, index 矩阵
                    for k in range(len(matrices_dna_ori[i][1])):
                        f.write("index_%d 矩阵的 列 错误率, 小于 %.3f 的不显示\n" % (k, min_disp_error))
                        if i == matrix_num - 1:
                            col_index_1_num_real = col_index_1_num_last
                            col_index_2_num_real = col_index_2_num_last
                        else:
                            col_index_1_num_real = col_index_1_num
                            col_index_2_num_real = col_index_2_num
                        if k == 0:
                            for p in range(row_len):
                                key = str(i) + '_index_' + str(10) + str(p)
                                if key in errors_index_dict:
                                    column_error_ration = len(errors_index_dict[key]) / col_index_1_num_real
                                    # f.write("第 %d 列的错误率为 %.6f \n" % (p, column_error_ration))
                                    if column_error_ration >= min_disp_error:
                                        f.write("第 %d 列的错误率为 %.6f \n" % (p, column_error_ration))
                        else:   # k = 1
                            for p in range(row_len):
                                key = str(i) + '_index_' + str(11) + str(p)
                                if key in errors_index_dict:
                                    column_error_ration = len(errors_index_dict[key]) / col_index_2_num_real
                                    if column_error_ration >= min_disp_error:
                                        f.write("第 %d 列的错误率为 %.6f \n" % (p, column_error_ration))
                                    # f.write("第 %d 列的错误率为 %.6f \n" % (p, column_error_ration))

def handle_dna_matrix(dna_seq_rec, dna_seq_ori):
    # 加入一个新的功能，记录每个错误的列地址，并且mutate_num, insert_num, delete_num错误分别标记为0，1，2
    mutate_num, insert_num, delete_num = 0, 0, 0
    errors_info = []
    if dna_seq_rec:  # 收到的DNA序列不为空，这样才等于 dna_seq_rec != []
        if dna_seq_rec != dna_seq_ori:
            # 我们注意到在check_dna_errors_ration_new函数中 dna_seq_ori要放左边
            mutate_num, insert_num, delete_num, errors_info = check_dna_errors_ration_new(dna_seq_ori, dna_seq_rec)
    else:
        dna_seq_rec = ['C' for _ in range(len(dna_seq_ori))]
        mutate_num, insert_num, delete_num, errors_info = check_all_C_to_dna_seq(dna_seq_rec, dna_seq_ori)

    return mutate_num, insert_num, delete_num, errors_info

def check_dna_errors_ration_new(seq_ori, seq_receive):
    # 加入一个新的功能，记录每个错误的列地址，并且mutate_num, insert_num, delete_num错误分别标记为0，1，2
    errors_info = []
    mutate_num, insert_num, delete_num = 0, 0, 0
    # seq_ori_copy = seq_ori.copy()  # str 没有copy，也不用copy，本身在函数调用时就是copy
    # seq_receive_copy = seq_receive.copy()
    # seq_ori_ali, seq_receive_ali = dp_align(seq_ori_copy, seq_receive_copy)
    seq_ori_ali, seq_receive_ali = dp_align(seq_ori, seq_receive)
    for i in range(len(seq_ori_ali)):
        if seq_ori_ali[i] != seq_receive_ali[i]:
            if seq_ori_ali[i] == '-':
                insert_num += 1
                errors_info.append([1, i])
            elif seq_receive_ali[i] == '-':
                delete_num += 1
                errors_info.append([2, i])
            else:
                mutate_num += 1
                errors_info.append([0, i])
    return mutate_num, insert_num, delete_num, errors_info

def dp_align(seq1, seq2,num=1):
    x, y = len(seq2) + 1, len(seq1) + 1
    array = compute(init_array(x, y), seq1, seq2)
    s1, s2 = backtrack(array, seq1, seq2)
    return s1, s2

def compute(array, seq1, seq2):
    row, col = len(seq2), len(seq1)
    for i in range(1, row + 1):
        for j in range(1, col + 1):
            if seq1[j - 1] == seq2[i - 1]:
                s = match
            else:
                s = mismatch
            lu = [array[i - 1][j - 1].score+s, [i - 1, j - 1]]
            left = [array[i - 1][j].score + gap, [i - 1, j]]
            up = [array[i][j - 1].score + gap, [i, j - 1]]
            max_choice = max([lu, left, up], key=lambda x: x[0])
            score = max_choice[0]
            point_position = max_choice[1]
            array[i][j] = score_tuple(score, point_position)
    return array

def init_array(x, y):
    array = [[0] * y for _ in range(x)]
    array[0][0] = score_tuple(0, None)
    for j in range(1, y):
        array[0][j] = score_tuple(gap * j, [0, j - 1])
    for i in range(1, x):
        array[i][0] = score_tuple(gap * i, [i - 1, 0])
    return array

def backtrack(array, seq1, seq2):
    s1 = []
    s2 = []
    row, col = len(seq2), len(seq1)
    while array[row][col].score != 0:
        i, j = array[row][col].point_position
        if i + 1 == row and j + 1 == col:
            s1.append(seq1[col - 1])
            s2.append(seq2[row - 1])
            row, col = i, j
        elif row == i + 1 and col == j:
            s1.append("-")
            s2.append(seq2[i])
            row, col = i, j
        elif row == i and col == j + 1:
            s1.append(seq1[j])
            s2.append("-")
            row, col = i, j
    s1 = ''.join(s1[::-1])
    s2 = ''.join(s2[::-1])
    return s1, s2

def check_all_C_to_dna_seq(dna_seq_rec, dna_seq_ori):
    mutate_num = 0
    errors_info = []
    for i in range(len(dna_seq_rec)):
        if dna_seq_rec[i] != dna_seq_ori[i]:
            mutate_num += 1
            errors_info.append([0, i])
    return mutate_num, 0, 0, errors_info

def check_recover_ration(matrices_ori, matrices_decode, file_name, file_num):
    # 为了方便分析最后解码结果的错误分布，这里对解码所得结果进行列上面的检查，并将记录每一个错误的坐标
    errors_analysis_path =  file_name +'hamming_errors_analysis_dna_' + '_' + str(file_num) + '.txt'
    errors_list = []
    print("开始检查恢复情况")
    true_num_block = 0
    true_num_bit = 0
    cnt_block, cnt_bit = 0, 0
    # print(f'parity_length:{parity_length},matrices_ori:{matrices_ori[0][0][0]}')
    # print(f'len(matrices_ori):{len(matrices_ori)},len(matrices_ori[0]):{len(matrices_ori[0])},len(matrices_ori[0][0]):{len(matrices_ori[0][0])}')
    for i in range(len(matrices_ori)):  # i 表示有多少个矩阵
        for j in range(len(matrices_ori[i])):  # j=0 表示第1层矩阵， j=1 表示第2层矩阵
            for k in range(len(matrices_ori[i][j])):  # k 表示该矩阵的第k个块，为list
                if matrices_ori[i][j][k][:-parity_length] == matrices_decode[i][j][k]:   # matrices_ori是有最后 parity_length 列的奇偶检验位的，要记得去掉
                    # if i == 0 and j == 0 and k == 0:
                    #     print("matrices_decode[0][0][0]:", matrices_decode[0][0][0], type(matrices_decode[0][0][0]), len(matrices_decode[0][0][0]))
                    true_num_block += 1
                cnt_block += 1
                # 下面检查bit的恢复率
                for m in range(len(matrices_ori[i][j][k][:-parity_length])):  # m 为第k个块的第m个元素，为0或1
                    if matrices_ori[i][j][k][m] == matrices_decode[i][j][k][m]:
                        true_num_bit += 1
                    else:
                        error_list_temp = [i, j, k, m, matrices_ori[i][j][k][m], matrices_decode[i][j][k][m]]
                        errors_list.append(error_list_temp)
                    cnt_bit += 1
    block_recover_ration = true_num_block / cnt_block
    print("块恢复率为 %.6f" % (block_recover_ration))
    print("bit恢复率为 %.6f" % (true_num_bit / cnt_bit))
    with open('out_put_Polar_DNA_two.txt', 'a') as file:
        file.write("块恢复率为 %.6f" % (block_recover_ration) + "\n")
        file.write("bit恢复率为 %.6f" % (true_num_bit / cnt_bit) + "\n")
    # 统计每个矩阵（每层）其每行和每列上的错误率
    # 先统计行上的错误率, 再统计列上的错误率
    errors_row_dict = dict()
    errors_column_dict = dict()
    for error in errors_list:
        key_row_temp = str(error[0]) + str(error[1]) + 'row_' + str(error[2])
        key_column_temp = str(error[0]) + str(error[1]) + 'column_' + str(error[3])
        if key_row_temp not in errors_row_dict:
            errors_row_dict[key_row_temp] = []
        if key_column_temp not in errors_column_dict:
            errors_column_dict[key_column_temp] = []
        errors_row_dict[key_row_temp].append(error)
        errors_column_dict[key_column_temp].append(error)
    row_cnt = len(matrices_ori[0][0][0][:-parity_length])  # 每行元素个数,每行的长度，是一样的，注意去掉最后 parity_length 列的奇偶检验位
    col_cnt_0 = len(matrices_ori[0][0])  # 前面矩阵，第0层矩阵每列元素个数
    col_cnt_1 = len(matrices_ori[0][1])  # 前面矩阵，第1层矩阵每列元素个数
    matrix_cnt = len(matrices_ori)
    col_last_cnt_0 = len(matrices_ori[matrix_cnt - 1][0])  # 最后一个矩阵，第0层矩阵每列元素个数
    col_last_cnt_1 = len(matrices_ori[matrix_cnt - 1][1])  # 最后一个矩阵，第1层矩阵每列元素个数
    print("row_cnt = %d, col_cnt_0 = %d, col_cnt_1 = %d, col_last_cnt_0 = %d, col_last_cnt_1 = %d"
          % (row_cnt, col_cnt_0, col_cnt_1, col_last_cnt_0, col_last_cnt_1))
    # 先写列的情况比较好
    with open(errors_analysis_path, 'w') as file:
        file.write("下面是各个矩阵的 列 错误率, 未出现的没有出错" + "\n")
        for i in range(len(matrices_ori)):
            file.write("\n")
            file.write("第 " + str(i) + " 个单位矩阵的 列 错误率\n")
            if i == len(matrices_ori) - 1:
                col_cnt_real_0 = col_last_cnt_0
                col_cnt_real_1 = col_last_cnt_1
            else:
                col_cnt_real_0 = col_cnt_0
                col_cnt_real_1 = col_cnt_1
            for j in range(len(matrices_ori[i])):
                file.write("第 " + str(j) + " 层矩阵的 列 错误率，未出现的没有出错\n")
                if j == 0:
                    col_cnt_real = col_cnt_real_0
                else:
                    col_cnt_real = col_cnt_real_1
                for m in range(row_cnt):
                    key_column_temp = str(i) + str(j) + 'column_' + str(m)
                    if key_column_temp in errors_column_dict:
                        column_error_ration = len(errors_column_dict[key_column_temp]) / col_cnt_real
                        file.write("第 %d 列的错误率为 %.2f \n" % (m, column_error_ration))
        file.write("\n")
        file.write("下面是各个矩阵的 行 错误率, 未出现的没有出错 \n")
        for i in range(len(matrices_ori)):
            file.write("第 " + str(i) + " 个单位矩阵的 行 错误率\n")
            for j in range(len(matrices_ori[i])):
                file.write("第 " + str(j) + " 层矩阵的 行 错误率，未出现的没有出错\n")
                for k in range(len(matrices_ori[i][j])):
                    key_row_temp = str(i) + str(j) + 'row_' + str(k)
                    if key_row_temp in errors_row_dict:
                        row_error_ration = len(errors_row_dict[key_row_temp]) / row_cnt
                        file.write("第 %d 行的错误率为 %.4f \n" % (k, row_error_ration))
        # file.write("\n")   # 感觉下面的可以先不写
        # file.write("下面输出所有的错误信息，格式说明如下：[单位矩阵号，所属层数，行号，列号，正确信息，解码信息] \n")
        # for error in errors_list:
        #     file.write(str(error) + '\n')

    data = [
        ('',get_selected_number(),block_recover_ration,(true_num_bit / cnt_bit))
    ]
    # f = open('files/testrev.csv', 'a', encoding='utf8', newline='')
    # writer = csv.writer(f)  # csv.writer()中可以传一个文件对象
    # for line in data:  # 该data既可以是列表嵌套列表的数据类型也可以是列表嵌套元组的数据类型
    #     writer.writerow(line)

    with open('files/testrev_8_5-15.csv', 'a', encoding='utf8', newline='') as f:
        writer = csv.writer(f)  # csv.writer()中可以传一个文件对象
        for line in data:  # 该data既可以是列表嵌套列表的数据类型也可以是列表嵌套元组的数据类型
            writer.writerow(line)

    return block_recover_ration,true_num_bit / cnt_bit,cnt_bit-true_num_bit,cnt_bit

def check_received_01_ration(matrices_01_ori, matrices_received_01, file_name, file_num):
    # 这里要实现最后一次解码时，接收到的01矩阵与原正确01矩阵在各列上的错误率
    print("开始竖向检查错误_hamming：最后一次解码时接收到的01矩阵，与原正确01矩阵矩阵对比，以检查 列 错误分布")
    # matrices_dna = copy.deepcopy(matrices_dna_and_phred)
    mutate_num_all = 0
    matrix_num = len(matrices_01_ori)
    errors_data_dict = dict()
    errors_index_dict = dict()
    for i in range(len(matrices_01_ori)):
        # j = 0, 处理01数据矩阵
        for data_n in range(2):  # data_n = 0 为第1层，data_n = 1 为第2层
            results = Parallel(n_jobs=2 * threads_number)(
                delayed(handle_01_matrix_hamming)(matrices_received_01[i][0][data_n][k], matrices_01_ori[i][0][data_n][k])
                for k in range(len(matrices_01_ori[i][0][data_n])))
            for result in results:
                mutate_num_all += result[0]
                # 发生插入错误后，若该条DNA序列（横向）后面还有错误，则列号会+1，不过发生概率不高，先不管
                for [error_type, col_num] in result[1]:
                    key = str(i) + '_data_' + str(data_n) + str(col_num)
                    if key not in errors_data_dict:
                        errors_data_dict[key] = []
                    errors_data_dict[key].append([error_type, col_num])

        # j = 1，处理index矩阵
        # j = 1 and k = 0, 处理index的第1个矩阵
        results = Parallel(n_jobs=2 * threads_number)(
            delayed(handle_01_matrix_hamming)(matrices_received_01[i][1][0][p], matrices_01_ori[i][1][0][p])
            for p in range(len(matrices_01_ori[i][1][0])))
        for result in results:
            mutate_num_all += result[0]
            for [error_type, col_num] in result[1]:
                key = str(i) + '_index_' + str(10) + str(col_num)
                if key not in errors_index_dict:
                    errors_index_dict[key] = []
                errors_index_dict[key].append([error_type, col_num])
        # j = 1 and k = 1, 处理index的第2个矩阵
        # matrices_dna_ori[i][1][1] 可能是[]
        if matrices_01_ori[i][1][1]:  # 不为[]才需要处理
            if matrices_received_01[i][1][1]:
                results = Parallel(n_jobs=2 * threads_number)(
                    delayed(handle_01_matrix_hamming)(matrices_received_01[i][1][1][p], matrices_01_ori[i][1][1][p])
                    for p in range(len(matrices_01_ori[i][1][1])))
                for result in results:
                    mutate_num_all += result[0]
                    for [error_type, col_num] in result[1]:
                        key = str(i) + '_index_' + str(11) + str(col_num)
                        if key not in errors_index_dict:
                            errors_index_dict[key] = []
                        errors_index_dict[key].append([error_type, col_num])
            else:
                if i == matrix_num - 1:  # 最后一个矩阵才需要检查
                    # 这种情况应该不会发生
                    print("出错了，请修改代码，matrices_01_ori[i][1][1]不为空，但matrices_received_01[i][1][1]为空，i = %d" % i)
    bits_num_all = 0
    for matrix in matrices_01_ori:
        for matrix_data in matrix[0]:
            for seq in matrix_data:
                bits_num_all += len(seq)
        for matrix_index in matrix[1]:
            for seq in matrix_index:
                bits_num_all += len(seq)
    # print("bp_num_all = %d" % bp_num_all)
    mutate_ration = mutate_num_all / bits_num_all
    all_ration = mutate_ration
    print(
        "最后1次解码时接收到的01矩阵与原正确01矩阵（极化码编码后），直接对比，类似hamming距离")
    print("替换错误率为 %.8f, 总错误率为 %.8f"
          % (mutate_ration, all_ration))
    # 统计列方向上的错误情况
    errors_analysis_path =  file_name +'hamming_errors_analysis_dna_' + '_' + str(file_num) + '.txt'
    row_len = len(matrices_01_ori[0][0][0][0])  # 横向长度，即列号，是一样的
    print("row_len =", row_len)
    col_data_num = len(matrices_01_ori[0][0][0])
    min_disp_error = 0.001
    col_data_num_last = len(matrices_01_ori[matrix_num - 1][0][0])
    col_index_1_num = len(matrices_01_ori[0][1][0])
    col_index_2_num = len(matrices_01_ori[0][1][1])
    col_index_1_num_last = len(matrices_01_ori[matrix_num - 1][1][0])
    col_index_2_num_last = len(matrices_01_ori[matrix_num - 1][1][1])  # 这个值可能为0
    with open(errors_analysis_path, 'w') as f:
        f.write("下面是各个01矩阵的 列 错误率, 未出现的没有出错\n")
        for i in range(len(matrices_01_ori)):
            f.write('\n')
            f.write("第 %d 个单位矩阵的 列 错误率\n" % i)
            for j in range(len(matrices_01_ori[i])):
                if j == 0:  # DNA数据矩阵
                    for data_n in range(2):
                        f.write("01数据矩阵,第 %d 层的 列 错误率, 小于 %.3f 的不显示\n" % (data_n, min_disp_error))
                        if i == matrix_num - 1:
                            col_data_num_real = col_data_num_last
                        else:
                            col_data_num_real = col_data_num
                        for k in range(row_len):
                            key = str(i) + '_data_' + str(data_n) + str(k)
                            if key in errors_data_dict:
                                column_error_ration = len(errors_data_dict[key]) / col_data_num_real
                                if column_error_ration >= min_disp_error:
                                    f.write("第 %d 列的错误率为 %.6f \n" % (k, column_error_ration))
                else:  # j = 1, index 矩阵
                    for k in range(len(matrices_01_ori[i][1])):
                        f.write("index_%d 矩阵的 列 错误率, 小于 %.3f 的不显示\n" % (k, min_disp_error))
                        if i == matrix_num - 1:
                            col_index_1_num_real = col_index_1_num_last
                            col_index_2_num_real = col_index_2_num_last
                        else:
                            col_index_1_num_real = col_index_1_num
                            col_index_2_num_real = col_index_2_num
                        if k == 0:
                            for p in range(row_len):
                                key = str(i) + '_index_' + str(10) + str(p)
                                if key in errors_index_dict:
                                    column_error_ration = len(errors_index_dict[key]) / col_index_1_num_real
                                    # f.write("第 %d 列的错误率为 %.6f \n" % (p, column_error_ration))
                                    if column_error_ration >= min_disp_error:
                                        f.write("第 %d 列的错误率为 %.6f \n" % (p, column_error_ration))
                        else:  # k = 1
                            for p in range(row_len):
                                key = str(i) + '_index_' + str(11) + str(p)
                                if key in errors_index_dict:
                                    column_error_ration = len(errors_index_dict[key]) / col_index_2_num_real
                                    if column_error_ration >= min_disp_error:
                                        f.write("第 %d 列的错误率为 %.6f \n" % (p, column_error_ration))
                                    # f.write("第 %d 列的错误率为 %.6f \n" % (p, column_error_ration))

def handle_01_matrix_hamming(seq_received, seq_ori):
    # 在直接对比中，只有替换错误，故不用返回错误类型
    mutate_num = 0
    errors_info = []
    for i in range(len(seq_ori)):
        if seq_ori[i] != seq_received[i]:
            mutate_num += 1
            errors_info.append([0, i])
    return mutate_num, errors_info

def matrices_bits_to_binary_sequence(matrices_bits):  # 将01矩阵恢复成原来的01序列
    binary_sequence = ''
    n = len(matrices_bits)
    # 先处理前面的矩阵
    for i in range(n - 1):  # 在双层极化码方案中，每个单位矩阵由两个小数据矩阵构成
        for j in range(len(matrices_bits[i])):
            for k in range(len(matrices_bits[i][j])):
                temp_str = ''.join([str(num) for num in matrices_bits[i][j][k]])
                binary_sequence += temp_str
    # 处理最后一个单位矩阵。
    # 先根据mark_list找到矩阵中真实数据的最后一行，以去掉矩阵中，多余的随机行；再根据more_zeros,处理真实最后一行中多余的0（其实也进行随机化了）
    mark_lists = mark_list * (segment_length // len(mark_list))  # 这里保证segment_length是mark_list整数倍
    mark_j, mark_row = -1, -1  # 分别记录mark_lists所在的子矩阵号和行号
    min_distance = 1000
    find_flag = False
    for j in range(len(matrices_bits[n - 1])):
        for k in range(len(matrices_bits[n - 1][j])):
            # 这里是根据mark_lists与最后一个矩阵所有行数据求最小编辑距离来确定的，故就算解码出来的最后一行数据有一些错误，也没有关系
            edit_distance = Levenshtein.distance(mark_lists, matrices_bits[n - 1][j][k])
            if edit_distance < min_distance:
                min_distance = edit_distance
                mark_j = j
                mark_row = k
                if min_distance == 0:  # 已经找到mark_lists
                    find_flag = True
                    break
        if find_flag:
            break
    # 去掉矩阵中多余的0行（已经随机化）
    if mark_j == 0:
        matrices_bits[n - 1][0] = matrices_bits[n - 1][0][:mark_row]  # 同时去掉了mark_lists这一行
        del matrices_bits[n - 1][1]
        # 去掉最后一行中多余的0
        more_zeros_binary = matrices_bits[n - 1][0][mark_row - 1][-10:]
        more_zeros = int(''.join([str(bit) for bit in more_zeros_binary]), 2)
        print("恢复的more_zeros：", more_zeros)
        del matrices_bits[n - 1][0][mark_row - 1]
        matrices_bits[n - 1][0][mark_row - 2] = matrices_bits[n - 1][0][mark_row - 2][:segment_length - more_zeros]
    else:
        matrices_bits[n - 1][1] = matrices_bits[n - 1][1][:mark_row]  # 同时去掉了mark_lists这一行
        # 去掉最后一行中多余的0
        more_zeros_binary = matrices_bits[n - 1][1][mark_row - 1][-10:]
        more_zeros = int(''.join([str(bit) for bit in more_zeros_binary]), 2)
        print("恢复的more_zeros：", more_zeros)
        del matrices_bits[n - 1][1][mark_row - 1]
        matrices_bits[n - 1][1][mark_row - 2] = matrices_bits[n - 1][1][mark_row - 2][:segment_length - more_zeros]
    # 把最后一个矩阵的数据加上去
    for i in range(len(matrices_bits[n - 1])):
        for j in range(len(matrices_bits[n - 1][i])):
            temp_str = ''.join([str(num) for num in matrices_bits[n - 1][i][j]])
            binary_sequence += temp_str
    return binary_sequence

def binary_sequence_to_file(binary_sequence, output_file):
    # 根据二进制序列，恢复文件
    try:
        with open(output_file, 'wb') as file:
            bytes_data = bytearray(int(binary_sequence[i:i + 8], 2) for i in range(0, len(binary_sequence), 8))
            file.write(bytes_data)
        print(f"文件 '{output_file}' 已成功写入。")
    except Exception as e:
        print(f"写入文件时发生错误：{e}")

def dna_seq_add_phred_scores(dna_sequences_clear, phred_scores):
    print("有使用质量值！！")
    dna_seq_and_phred = []
    if len(dna_sequences_clear) != len(phred_scores):
        print("出错啦！DNA序列数量与质量值数量不相等:len(dna_sequences_clear) = %d, len(phred_scores) = %d" % (
            len(dna_sequences_clear), len(phred_scores)))
    for i in range(len(dna_sequences_clear)):
        dna_seq_and_phred_temp = []
        dna_seq_and_phred_temp.append(dna_sequences_clear[i])
        dna_seq_and_phred_temp.append(phred_scores[i])
        dna_seq_and_phred.append(dna_seq_and_phred_temp)
    # print("dna_seq_and_phred[0]", dna_seq_and_phred[0], type(dna_seq_and_phred[0]))
    # print("dna_seq_and_phred[0][0]", dna_seq_and_phred[0][0], type(dna_seq_and_phred[0][0]))   # 为string
    # print("dna_seq_and_phred[0][1]", dna_seq_and_phred[0][1], type(dna_seq_and_phred[0][1]))  # 为list
    return dna_seq_and_phred




