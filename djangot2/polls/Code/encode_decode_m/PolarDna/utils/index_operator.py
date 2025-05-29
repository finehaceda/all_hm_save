import copy

import numpy as np
import math
import random
from .glob_var import (segment_length, mark_list,
                       matrix_row_num_data_1_last_1, matrix_row_num_data_2_last_1, matrix_row_num_data_1_last_2,
                       matrix_row_num_data_2_last_2, segment_length_parity, parity_length, N_data, crc_data)
from .chooseparams import PolarParams


def connect_all(matrix, index_binary_length):
    new_matrix = []
    for row in range(len(matrix)):
        # 下一行代码的作用是将一个二维列表matrix中的一行数据和该行的索引整合到一起
        # 也就是说每一行数据的后面将存下现在的行号
        new_matrix.append(connect(row, matrix[row], index_binary_length))
    del matrix
    return new_matrix

def connect(index, data, index_binary_length):
    # 作用是将行号与数据分开
    # 下面这行代码是不是搞了太多次list和str了，应该可以简化. 答：这是最简单的了。
    bin_index = list(map(int, list(str(bin(index))[2:].zfill(index_binary_length))))
    # 我们把行号存在数据的后面，以提升行号恢复的准确率
    one_list = data + bin_index
    return one_list

'''
# 下面的函数把不带行号的原数据矩阵切分成一个一个的小单位矩阵
def divide_to_matrices(input_matrix):
    # 这里应该也可以不用对matrices用0初始化
    num_of_matrix = math.ceil(len(input_matrix) / matrix_row_num_data)
    # 下面这样初始化one_matrix是ok的，因为[0 for _ in range(segment_length)]每次都是新生成的
    # one_matrix = [[0 for _ in range(segment_length)] for _ in range(matrix_row_num_data)]  # 这样写真的有问题，但在test里面测试又没有问题，不知道为什么，很奇怪
    # one_list = [0 for _ in range(segment_length)]
    one_matrix = []
    for _ in range(matrix_row_num_data):
        # one_list_temp = one_list.copy()
        one_list_temp = [0 for _ in range(segment_length)]
        one_matrix.append(one_list_temp)
    # print("one_matrix:", len(one_matrix), len(one_matrix[0]))  # 这里没有问题
    # matrices = [one_matrix for _ in range(num_of_matrix)]  # 这样是有bug的，考虑到python的列表复制特点，要用deepcopy才行
    matrices = []
    for _ in range(num_of_matrix):
        matrix_temp = copy.deepcopy(one_matrix)
        matrices.append(matrix_temp)
    for idx in range(len(input_matrix)):
        idx_matrix = idx // matrix_row_num_data
        idx_row = idx % matrix_row_num_data
        matrices[idx_matrix][idx_row] = input_matrix[idx]  # 这样写并没有真的把数据copy过去，而是用了同一个指针，看看行不行。可以！

    true_row = len(input_matrix) % matrix_row_num_data
    print("最后一个单位矩阵中真实数据的行数为(若为0，则表示所有数据均为真实数据)：", true_row)
    last_idx_matrix = num_of_matrix - 1
    # 在最后一个单位矩阵中加入mark_lists,用于标识最后一个单位矩阵中真实数据的行数（范围）
    if true_row != 0:  # 有空行才能添加
        mark_lists = mark_list * (segment_length // len(mark_list))
        matrices[last_idx_matrix][true_row] = mark_lists
        # 处理最后一个矩阵中，多的0行
        true_row = true_row + 1
        if true_row != matrix_row_num_data:  # 还有空行
            for row in range(true_row, matrix_row_num_data):
                # 新方案1115，不用在数据加行号
                # 生成的随机数据的行号将比原来真实数据的最大行号还要大4
                # random_index = random.randint(len(input_matrix) + 3, int(math.pow(2, index_binary_length) - 1))
                # index_list = list(map(int, list(str(bin(random_index))[2:].zfill(index_binary_length))))
                # random_list = [random.randint(0, 1) for _ in range(segment_length)] + index_list
                random_list = [random.randint(0, 1) for _ in range(segment_length)]
                matrices[last_idx_matrix][row] = random_list
    return matrices
'''

def generate_random_list_have_parity(segment_length_parity, parity_length):
    # 目前仅支持parity_length = 2
    random_list = [random.randint(0, 1) for _ in range(segment_length_parity)]
    if parity_length == 2:
        even_parity, odd_parity = 0, 0  # 偶校验位， 奇校验位
        for i in range(0, len(random_list), 2):
            even_parity ^= random_list[i]
            odd_parity ^= random_list[i + 1]
        random_list.append(even_parity)  # 240
        random_list.append(odd_parity)  # 241
    else:
        print("出错了，parity_length不为2，请修改对应代码")
    return random_list
def handle_last_matrix(input_matrix_last, matrix_row_num_data_1_last, matrix_row_num_data_2_last):
    # 支持 2 位的奇偶校验
    matrix_1, matrix_2 = [], []
    for _ in range(matrix_row_num_data_1_last):
        # one_list_temp = one_list.copy()
        one_list_temp = [0 for _ in range(segment_length)]
        matrix_1.append(one_list_temp)
    for _ in range(matrix_row_num_data_2_last):
        # one_list_temp = one_list.copy()
        one_list_temp = [0 for _ in range(segment_length)]
        matrix_2.append(one_list_temp)
    matrix_last = [matrix_1, matrix_2]
    if len(input_matrix_last) <= matrix_row_num_data_1_last:  # 全部放到第1个小矩阵
        print("mark_lists在第 1 个小矩阵中，其行号为： %d" % (len(input_matrix_last) - 1))
        for i in range(len(input_matrix_last)):
            matrix_last[0][i] = input_matrix_last[i]
        # 剩余行数的随机化
        for j in range(len(input_matrix_last), matrix_row_num_data_1_last):   # 这里没有错，一个是起始位置，一个是结束位置+1
            random_list = generate_random_list_have_parity(segment_length_parity, parity_length)
            matrix_last[0][j] = random_list
        for k in range(matrix_row_num_data_2_last):
            # random_list = [random.randint(0, 1) for _ in range(segment_length)]
            random_list = generate_random_list_have_parity(segment_length_parity, parity_length)
            matrix_last[1][k] = random_list
    else:  # 先放到第1个小矩阵，再放到第2个小矩阵
        print("mark_lists在第 2 个小矩阵中，其行号为： %d" % (len(input_matrix_last) - matrix_row_num_data_1_last - 1))
        for i in range(matrix_row_num_data_1_last):
            matrix_last[0][i] = input_matrix_last[i]
        for j in range(len(input_matrix_last) - matrix_row_num_data_1_last):
            matrix_last[1][j] = input_matrix_last[j + matrix_row_num_data_1_last]
        # 剩余行数的随机化
        for k in range(len(input_matrix_last) - matrix_row_num_data_1_last, matrix_row_num_data_2_last):
            # random_list = [random.randint(0, 1) for _ in range(segment_length)]
            random_list = generate_random_list_have_parity(segment_length_parity, parity_length)
            matrix_last[1][k] = random_list
    return matrix_last

# 下面的函数把不带行号的原数据矩阵切分成一个一个的小单位矩阵(二层方案)
def divide_to_matrices_two_layer_scheme(input_matrix,frozen_bits_len):
    polarParams = PolarParams(N_data, crc_data,frozen_bits_len)
    # 因为检查恢复情况时，是检查这里输出的矩阵。故为了检查结果的准确。引入的随机数据，也应该保证奇（偶）位的异或为0. 240902
    # 这个函数要更新，支持最后一个矩阵动态处理
    # 这里应该也可以不用对matrices用0初始化
    # mark_lists一定要添加，为了方便确定最后一个矩阵的真实数据行数
    # 已经保证mark_list_121的奇偶检验为0，且mark_lists长242，故mark_lists行，不用添加奇偶校验位
    mark_lists = mark_list * (segment_length // len(mark_list))  # 这里保证segment_length是mark_list整数倍
    input_matrix.append(mark_lists)  # 直接把mark_lists加到input_matrix后面是最方便的
    matrix_row_num_data_total = (polarParams.matrix_row_num_data_1 + polarParams.matrix_row_num_data_2)
    num_of_matrix = math.ceil(len(input_matrix) / matrix_row_num_data_total)  # 动态矩阵方案，num_of_matrix表示总矩阵数

    last_matrix_row = len(input_matrix) % matrix_row_num_data_total  # 表示最后一个单位矩阵中数据行数，若为0，则表示最后一个矩阵用2048的方案
    print("共有 %d 个单位矩阵（二层方案）" % num_of_matrix)
    # 下面这样初始化one_matrix是ok的，因为[0 for _ in range(segment_length)]每次都是新生成的
    # one_matrix = [[0 for _ in range(segment_length)] for _ in range(matrix_row_num_data)]  # 这样写真的有问题，但在test里面测试又没有问题，不知道为什么，很奇怪
    # one_list = [0 for _ in range(segment_length)]
    matrix_1, matrix_2 = [], []
    for _ in range(polarParams.matrix_row_num_data_1):
        # one_list_temp = one_list.copy()
        one_list_temp = [0 for _ in range(segment_length)]  # 考虑到奇偶检验位
        matrix_1.append(one_list_temp)
    for _ in range(polarParams.matrix_row_num_data_2):
        # one_list_temp = one_list.copy()
        one_list_temp = [0 for _ in range(segment_length)]  # 考虑到奇偶检验位
        matrix_2.append(one_list_temp)
    one_matrix = [matrix_1, matrix_2]
    matrices = []
    for _ in range(num_of_matrix - 1):  # 先分配前面的矩阵
        matrix_temp = copy.deepcopy(one_matrix)
        matrices.append(matrix_temp)
    if last_matrix_row == 0:
        last_matrix_row_temp = matrix_row_num_data_total
    else:
        last_matrix_row_temp = last_matrix_row

    for idx in range(len(input_matrix) - last_matrix_row_temp):  # 先分配前面的矩阵
        idx_matrix = idx // matrix_row_num_data_total
        idx_row = idx % matrix_row_num_data_total
        if idx_row < polarParams.matrix_row_num_data_1:  # 属于第1层
            matrices[idx_matrix][0][idx_row] = input_matrix[idx]  # 这样写并没有真的把数据copy过去，而是用了同一个指针，看看行不行。可以！
        else:  # 属于第2层
            matrices[idx_matrix][1][idx_row - polarParams.matrix_row_num_data_1] = input_matrix[idx]
    input_matrix_last = []  # 分配最后一个矩阵
    for idx in range(len(input_matrix) - last_matrix_row_temp, len(input_matrix)):
        input_matrix_last.append(input_matrix[idx])
    # 再根据最后一个矩阵中数据的行数，确定极化方案，分配最后一个矩阵中的两个小矩阵
    if last_matrix_row_temp <= matrix_row_num_data_1_last_1 + matrix_row_num_data_2_last_1:  # 分级情况1，last_1
        matrix_last = handle_last_matrix(input_matrix_last, matrix_row_num_data_1_last_1, matrix_row_num_data_2_last_1)
    elif last_matrix_row_temp <= matrix_row_num_data_1_last_2 + matrix_row_num_data_2_last_2:  # 分级情况2，last_2
        matrix_last = handle_last_matrix(input_matrix_last, matrix_row_num_data_1_last_2, matrix_row_num_data_2_last_2)
    else:  # 分级情况3，2048方案，与前面的矩阵相同
        matrix_last = handle_last_matrix(input_matrix_last, polarParams.matrix_row_num_data_1, polarParams.matrix_row_num_data_2)
    # 想到一个bug，如果原来的数据刚好填满整数个单位矩阵，这时，如何加mark_lists是一个问题。答：在input_matrix中先加入mark_lists，可解决
    matrices.append(matrix_last)
    return matrices

def binary_to_mode(mode_binary):
    mode_binary_mapping = {
        "001": "1",
        "010": "L",
        "011": "P",
        "100": "RGB",
        "101": "RGBA",
        "110": "CMYK",
        "111": "YCbCr"
    }

    if mode_binary in mode_binary_mapping:
        return mode_binary_mapping[mode_binary]
    else:
        return None


# 将数据与行号分开，并按行号排序，并且如果某一行号对应多个数据则采用投票原则确定真实数据
def divide_and_order(matrix, index_binary_length):
    matrix = np.array(matrix)
    data_dict = {}
    # 恢复数据并按行号分组
    for row in matrix:
        # row_number = int("".join(map(str, row[:index_binary_length])), 2)
        # 按照0812的优化，行号存在了后面
        row_number = int("".join(str(i) for i in row[-index_binary_length:]), 2)
        real_data = row[:-index_binary_length]
        if row_number not in data_dict:
            data_dict[row_number] = []
        data_dict[row_number].append(real_data)
    # 恢复数据并按投票原则确定每行的值
    real_data_len = len(matrix[0]) - index_binary_length
    sorted_items = sorted(data_dict.items())
    # 查找row_number中连续值的最大值
    max_row_num = -1
    for row_number, data_list in sorted_items:
        if row_number < 0:
            continue
        if row_number == max_row_num + 1:
            max_row_num = row_number
        else:
            break
    output_matrix = [[0 for _ in range(real_data_len)] for _ in range(max_row_num + 1)]
    for row_number, data_list in sorted_items:
        if row_number < 0 or row_number > max_row_num:
            continue
        recovered_row = np.zeros_like(data_list[0])
        row_length = len(data_list[0])
        if len(data_list) == 1 or len(data_list) == 2:
            recovered_row = data_list[0]
        elif len(data_list) > 2:
            for i in range(row_length):
                column_data = np.array([data[i] for data in data_list])
                vote_result = np.argmax(np.bincount(column_data))
                recovered_row[i] = vote_result
        output_matrix[row_number] = list(recovered_row)
    # 处理掉最后一行中多余的0
    binary_string_more_zeros = ''.join([str(bit) for bit in output_matrix[max_row_num][-10:]])
    random_bits_len = real_data_len - (3 + 10 + 20 + 20)   # 最后一行数据中前面随机bits的长度
    binary_string_mode = ''.join([str(bit) for bit in output_matrix[max_row_num][random_bits_len:random_bits_len+3]])
    binary_string_size_w = ''.join([str(bit) for bit in output_matrix[max_row_num][random_bits_len+3:random_bits_len+23]])
    binary_string_size_h = ''.join([str(bit) for bit in output_matrix[max_row_num][random_bits_len+23:random_bits_len+43]])
    # print("binary_string:", binary_string)
    more_zeros = int(binary_string_more_zeros, 2)
    mode = binary_to_mode(binary_string_mode)
    size_w = int(binary_string_size_w, 2)
    size_h = int(binary_string_size_h, 2)
    size_w_h = (size_w, size_h)
    print("恢复的mode：", mode, type(mode))
    print("恢复的像素：", size_w_h)
    print("恢复的最大行号:", max_row_num)
    print("恢复的more_zeros:", more_zeros)
    # 当more_zeros=0时，下面的代码会出问题，[:-0]会得到一个空List而不是完整的list，所以做个判断就行
    if more_zeros != 0:
        output_matrix[max_row_num - 1] = output_matrix[max_row_num - 1][:-more_zeros]
    del output_matrix[max_row_num]     # 把存more_zeros的行删掉
    del matrix
    return output_matrix, size_w_h, mode

def add_base_index(matrices):
    file_path = "index_of_dna_sequences.txt"
    with open(file_path, 'r') as file:
        # 读取每一行并去除末尾的换行符
        index_of_dna_sequence = [line.strip() for line in file]
        # print(len(index_of_dna_sequence), index_of_dna_sequence[0])
    file_path = "index_of_matrices.txt"
    with open(file_path, 'r') as file:
        index_of_matrices = [line.strip() for line in file]

    for i in range(matrices):
        k = 0
        for j in range(matrices[i]):
            for idx in range(matrices[i][j]):
                matrices[i][j][idx] = index_of_matrices[i] + index_of_dna_sequence[k] + matrices[i][j][idx]
                k += 1
    return matrices

