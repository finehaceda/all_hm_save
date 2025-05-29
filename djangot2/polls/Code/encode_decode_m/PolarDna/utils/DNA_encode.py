import copy
import random
from .Polar_encode import *
from .chooseparams import PolarParams
from .glob_var import (segment_length, max_homopolymer, max_gc_content,
                      search_count, crc_index_1, index_binary_length_1,
                      segment_length_index_1, matrix_row_num_index_1, frozen_bits_index_1, N_index_1, N_data, crc_data,
                      select_index_index_1, freeze_index_index_1, index_binary_length_2, segment_length_index_2,
                      matrix_row_num_index_2, select_index_index_2, freeze_index_index_2, frozen_bits_index_2, N_index_2,
                      crc_index_2, more_data_length_1, more_data_length_2, matrix_row_num_data_1_last_1, matrix_row_num_data_1_last_2,
                      select_index_data_1_last_1, freeze_index_data_1_last_1, select_index_data_2_last_1, freeze_index_data_2_last_1,
                      select_index_index_1_last_1, freeze_index_index_1_last_1, select_index_index_1_last_2, freeze_index_index_1_last_2,
                      select_index_data_1_last_2, freeze_index_data_1_last_2, select_index_data_2_last_2, freeze_index_data_2_last_2,
                      frozen_bits_data_1_last_1, frozen_bits_data_2_last_1, frozen_bits_data_1_last_2, frozen_bits_data_2_last_2,
                      N_data_last_1, N_index_1_last_1, N_data_last_2, N_index_1_last_2, crc_data_last_1, crc_data_last_2,
                      index_binary_length_last_1, index_binary_length_last_2, segment_length_index_1_last_1, matrix_row_num_index_1_last_1,
                      more_data_length_last_1, segment_length_index_1_last_2, matrix_row_num_index_1_last_2, more_data_length_last_2,
                      frozen_bits_index_1_last_1, crc_index_1_last, frozen_bits_index_1_last_2, add_random_list_threshold)
from .validity import *
from .index_operator import connect_all

# frozen_bits_data_1, frozen_bits_data_2, matrix_row_num_data_1, matrix_row_num_data_2,\
#            select_index_data_1, freeze_index_data_1, select_index_data_2, freeze_index_data_2  = getfrozen_bits(N_data,crc_data)

def dna_encode(bits_matrices,frozen_bits_len):
    polarParams = PolarParams(N_data, crc_data,frozen_bits_len)
    dna_matrices = []
    matrices_ori_01 = []
    num_matrix = len(bits_matrices)  # 记录单位矩阵的个数
    for idx in range(num_matrix):
        # flag = False
        dna_matrix, flag, matrix_ori_01 = encode_matrix(bits_matrices[idx], idx, num_matrix,polarParams)  # idx为单位矩阵的矩阵号
        dna_matrices.append(dna_matrix)
        matrices_ori_01.append(matrix_ori_01)
        if not flag:
            return dna_matrices, False
    return dna_matrices, True, matrices_ori_01

def encode_matrix(matrix, idx, num_matrix,polarParams):  # 在二层编码方案中，matrix有两个数据矩阵
    # select_index_data_1, freeze_index_data_1,frozen_bits_data_1 = polarParams.select_index_data_1,\
    #                    polarParams.freeze_index_data_1,polarParams.frozen_bits_data_1,
    # select_index_data_2, freeze_index_data_2,frozen_bits_data_2 = polarParams.select_index_data_2,\
    #                    polarParams.freeze_index_data_2,polarParams.frozen_bits_data_2,
    encode_matrix_all = []
    # 为了支持对最后一次解码时，接收到的01矩阵与原正确01矩阵做对比，计算各列的错误率，这里把encoder后，得到的两个矩阵保存下来，注意第0行的0，不要删除
    matrix_data_01 = []
    last_case = ''
    if idx < num_matrix - 1:
        for i in range(len(matrix)):
            matrix_temp = copy.deepcopy(matrix[i])
            matrix_temp = np.array(matrix_temp)
            transposed_matrix = matrix_temp.T
            encode_matrix_temp = []

            if i == 0:  # 双层极化码编码方案中，第一层与第二层的编码是不同的，frozen_bits长度不同
                for row in range(len(transposed_matrix)):
                    # 下面用polar_encode将每行数据进行编码
                    # 注意到编码与解码是不一样的。确实，竖向编码方案，在编码前冻结位置都是一样的。但是编码后，第一个位置，或者说冻结位置，不一定会全一样。
                    encode_list = polar_encode(transposed_matrix[row], polarParams.select_index_data_1, polarParams.freeze_index_data_1,
                                                            polarParams.frozen_bits_data_1, N_data, crc_data)
                    encode_matrix_temp.append(encode_list)
            else:  # i == 1
                for row in range(len(transposed_matrix)):
                    # 下面用polar_encode将每行数据进行编码
                    # 注意到编码与解码是不一样的。确实，竖向编码方案，在编码前冻结位置都是一样的。但是编码后，第一个位置，或者说冻结位置，不一定会全一样。
                    encode_list = polar_encode(transposed_matrix[row], polarParams.select_index_data_2, polarParams.freeze_index_data_2,
                                                            polarParams.frozen_bits_data_2, N_data, crc_data)
                    encode_matrix_temp.append(encode_list)

            encode_matrix_np_temp = np.array(encode_matrix_temp)
            encode_matrix_np_temp = encode_matrix_np_temp.T  # 再进行转置操作，为转化成DNA序列做准备
            # 因为CRC_16(偶数时),极化编码并转置后，得到的encode_matrix_np[0]总是0，所以把这一行删除，记得在解码时，把它加回来
            # 检查第0行是否全是0
            if idx == 0:  # 只检查第0个矩阵就行
                flag = True
                for bit in encode_matrix_np_temp[0]:
                    if bit != 0:
                        flag = False
                        break
                if flag:
                    print("i = %d时，数据部分encode_matrix_np_temp[0]全是0" % i)
                else:
                    print("出错了，i = %d时，数据部分encode_matrix_np_temp[0]不全是0" % i)
                    print("数据部分encode_matrix_np_temp[0]:", encode_matrix_np_temp[0])
            encode_matrix_new_temp = encode_matrix_np_temp.tolist()
            encode_matrix_save = copy.deepcopy(encode_matrix_new_temp)
            matrix_data_01.append(encode_matrix_save)  # 这里有2个矩阵，对应两层

            encode_matrix_new_temp = encode_matrix_new_temp[1:]  # 删除第0行数据，显然删除第0行数据后，其他数据的行号也会减1，所以解码时一定要插回来，不能靠初始化
            encode_matrix_all.append(encode_matrix_new_temp)

            # encode_matrix_np_save = copy.deepcopy(encode_matrix_np_temp)
            # matrix_data_01.append(encode_matrix_np_save)  # 这里有2个矩阵，对应两层
            # encode_matrix_np_temp = encode_matrix_np_temp[1:]  # 删除第0行数据，显然删除第0行数据后，其他数据的行号也会减1，所以解码时一定要插回来，不能靠初始化
            # encode_matrix_new_temp = encode_matrix_np_temp.tolist()
            # encode_matrix_all.append(encode_matrix_new_temp)
    else:  # 处理最后一个单位矩阵
        # 先判断是那种情况
        if len(matrix[0]) == matrix_row_num_data_1_last_1:
            last_case += 'last_1'
            select_index_data_1_last, select_index_data_2_last = select_index_data_1_last_1, select_index_data_2_last_1
            freeze_index_data_1_last, freeze_index_data_2_last = freeze_index_data_1_last_1, freeze_index_data_2_last_1
            frozen_bits_data_1_last, frozen_bits_data_2_last = frozen_bits_data_1_last_1, frozen_bits_data_2_last_1
            N_data_last = N_data_last_1
            crc_data_last = crc_data_last_1
        elif len(matrix[0]) == matrix_row_num_data_1_last_2:
            last_case += 'last_2'
            select_index_data_1_last, select_index_data_2_last = select_index_data_1_last_2, select_index_data_2_last_2
            freeze_index_data_1_last, freeze_index_data_2_last = freeze_index_data_1_last_2, freeze_index_data_2_last_2
            frozen_bits_data_1_last, frozen_bits_data_2_last = frozen_bits_data_1_last_2, frozen_bits_data_2_last_2
            N_data_last = N_data_last_2
            crc_data_last = crc_data_last_2
        else:   # 情况3，与其他矩阵一样，用2048极化
            last_case += 'last_3'
            select_index_data_1_last, select_index_data_2_last = polarParams.select_index_data_1, polarParams.select_index_data_2
            freeze_index_data_1_last, freeze_index_data_2_last = polarParams.freeze_index_data_1, polarParams.freeze_index_data_2
            frozen_bits_data_1_last, frozen_bits_data_2_last = polarParams.frozen_bits_data_1, polarParams.frozen_bits_data_2
            N_data_last = N_data
            crc_data_last = crc_data
        print("最后一个矩阵情况是" + last_case)
        for i in range(len(matrix)):
            matrix_temp = copy.deepcopy(matrix[i])
            matrix_temp = np.array(matrix_temp)
            transposed_matrix = matrix_temp.T
            encode_matrix_temp = []
            if i == 0:  # 双层极化码编码方案中，第一层与第二层的编码是不同的，frozen_bits长度不同
                for row in range(len(transposed_matrix)):
                    # 下面用polar_encode将每行数据进行编码
                    # 注意到编码与解码是不一样的。确实，竖向编码方案，在编码前冻结位置都是一样的。但是编码后，第一个位置，或者说冻结位置，不一定会全一样。
                    encode_list = polar_encode(transposed_matrix[row], select_index_data_1_last,
                                                            freeze_index_data_1_last,
                                                            frozen_bits_data_1_last, N_data_last, crc_data_last)
                    encode_matrix_temp.append(encode_list)
            else:  # i == 1
                for row in range(len(transposed_matrix)):
                    # 下面用polar_encode将每行数据进行编码
                    # 注意到编码与解码是不一样的。确实，竖向编码方案，在编码前冻结位置都是一样的。但是编码后，第一个位置，或者说冻结位置，不一定会全一样。
                    encode_list = polar_encode(transposed_matrix[row], select_index_data_2_last,
                                                            freeze_index_data_2_last,
                                                            frozen_bits_data_2_last, N_data_last, crc_data_last)
                    encode_matrix_temp.append(encode_list)

            encode_matrix_np_temp = np.array(encode_matrix_temp)
            encode_matrix_np_temp = encode_matrix_np_temp.T  # 再进行转置操作，为转化成DNA序列做准备
            # 检查第0行是否全是0
            flag = True
            for bit in encode_matrix_np_temp[0]:
                if bit != 0:
                    flag = False
                    break
            if flag:
                print("i = %d时，最后一个矩阵的数据部分encode_matrix_np_temp[0]全是0" % i)
            else:
                print("出错了，i = %d时，最后一个矩阵的数据部分encode_matrix_np_temp[0]不全是0" % i)
                print("最后一个矩阵的数据部分encode_matrix_np_temp[0]:", encode_matrix_np_temp[0])

            encode_matrix_new_temp = encode_matrix_np_temp.tolist()
            encode_matrix_save = copy.deepcopy(encode_matrix_new_temp)
            matrix_data_01.append(encode_matrix_save)  # 这里有2个矩阵，对应两层

            encode_matrix_new_temp = encode_matrix_new_temp[1:]  # 删除第0行数据，显然删除第0行数据后，其他数据的行号也会减1，所以解码时一定要插回来，不能靠初始化
            encode_matrix_all.append(encode_matrix_new_temp)

    # dna_matrix 里面有两个矩阵，上一个是数据矩阵，下一个是index矩阵
    dna_matrix, matrix_index_ori_01 = lists_to_dna(encode_matrix_all, idx, last_case, num_matrix)  # 对编码后的矩阵进行横向转换成DNA序列
    matrix_ori_01 = [matrix_data_01, matrix_index_ori_01]
    # print("单位矩阵 %d 的数据部分行数为 %d " %(idx, len(dna_matrix[0])))
    # print("单位矩阵 %d 的index部分行数为 %d " %(idx, len(dna_matrix[1])))
    # 检查生成的DNA序列，是否都满足兼容性
    flag = False
    max_homopolymer_f_count, gc_content_f_count = 0, 0
    # 双层index方案中，index矩阵由两个小矩阵构成
    # 先检查数据部分
    for row in range(len(dna_matrix[0])):
        max_homopolymer_f_num, gc_content_f_num = check_num(dna_matrix[0][row], max_homopolymer,
                                                                     max_gc_content)
        max_homopolymer_f_count += max_homopolymer_f_num
        gc_content_f_count += gc_content_f_num
    # 再检查index部分，如果是最后一个单位矩阵，在last_1和last_2两种情况下，index的DNA矩阵也是由两个小矩阵构成，后一个小矩阵为空
    for i in range(len(dna_matrix[1])):
        for row in range(len(dna_matrix[1][i])):
            max_homopolymer_f_num, gc_content_f_num = check_num(dna_matrix[1][i][row], max_homopolymer,
                                                                         max_gc_content)
            max_homopolymer_f_count += max_homopolymer_f_num
            gc_content_f_count += gc_content_f_num
    if max_homopolymer_f_count == 0 and gc_content_f_count == 0:
        # print("矩阵%d兼容性检查合格！" % idx)
        flag = True
    else:
        print("矩阵%d兼容性检查不合格！" % idx)
        print("在本矩阵中均聚物长度不合格的条数为：", max_homopolymer_f_count)
        print("在本矩阵中gc含量不合格的条数为：", gc_content_f_count)
    return dna_matrix, flag, matrix_ori_01

def lists_to_dna(matrix, idx, last_case, num_matrix):  # 在二层编码方案中，matrix有两个已经编码后的数据矩阵
    # 压缩后的数组0/1是比较平衡的，可以不用分good_data_set和bad_data_set，但若没有压缩，就值得分了
    # n为此过程的并行数量
    # 为了方便后面分割为good_data_set和bad_data_set，以及后面用表格存行号，这里对极化码编码后的矩阵加上行号
    dna_matrix, dna_matrix_data, index_numbers = [], [], []
    matrix_have_index = []
    # print("last_case:" + last_case)
    if last_case == '' or last_case == 'last_3':  # 该矩阵用2048极化
        for i in range(len(matrix)):
            matrix_index_temp = connect_all(matrix[i], index_binary_length_1)  # 行号index在数据的后面
            matrix_have_index.append(matrix_index_temp)
    elif last_case == 'last_1':  # 该矩阵用512极化
        for i in range(len(matrix)):
            matrix_index_temp = connect_all(matrix[i], index_binary_length_last_1)  # 行号index在数据的后面
            matrix_have_index.append(matrix_index_temp)
    else:   # 该矩阵用1024极化
        for i in range(len(matrix)):
            matrix_index_temp = connect_all(matrix[i], index_binary_length_last_2)  # 行号index在数据的后面
            matrix_have_index.append(matrix_index_temp)
    dna_matrix_data, index_numbers = divide_sets_to_dna_sequences(matrix_have_index[0], matrix_have_index[1], last_case)

    # print("dna_matrix_data:", len(dna_matrix_data))
    # print("dna_matrix_data[0]:", len(dna_matrix_data[0]), dna_matrix_data[0])
    dna_matrix.append(dna_matrix_data)   # 把数据矩阵存下来
    # 下面把index_numbers用DNA存下来
    if idx < num_matrix - 1:  # 前面的矩阵
        dna_matrix_index, matrix_index_ori_01 = index_to_dna_two_layers(index_numbers, idx)  # 改进版的，双层index方案
    else:  # 处理最后一个矩阵，如果是用512或1024极化（last_1或last_2）则用单层index方案，若用2048极化，则用双层index方案
        if last_case == 'last_1' or last_case == 'last_2':
            dna_matrix_index, matrix_index_ori_01 = index_to_dna_single_layer(index_numbers, last_case)  # 单层index方案
        else:
            dna_matrix_index, matrix_index_ori_01 = index_to_dna_two_layers(index_numbers, idx)  # 改进版的，双层index方案

    # print("dna_matrix_index:", len(dna_matrix_index))
    # print("dna_matrix_index[0]:", len(dna_matrix_index[0]), dna_matrix_index[0])
    dna_matrix.append(dna_matrix_index)   # 把index矩阵存下来
    return dna_matrix, matrix_index_ori_01

'''
def divide_library(matrix, n):
    good_data_set_total,  bad_data_set_total = [], []    # 它们是二维列表
    step = math.ceil(len(matrix) / n)
    for row in range(0, len(matrix), step):
        good_data_set, bad_data_set = [], []
        for i in range(step):
            if row + i == len(matrix):
                break
            handle_list = matrix[row+i][:segment_length]
            if np.sum(handle_list) > len(handle_list) * max_ratio \
                        or np.sum(handle_list) < len(handle_list) * (1 - max_ratio):
                bad_data_set.append(list(matrix[row+i]))
            else:
                good_data_set.append(list(matrix[row+i]))
        good_data_set_total.append(good_data_set)
        bad_data_set_total.append(bad_data_set)
    del matrix
    return good_data_set_total, bad_data_set_total
'''
def divide_sets_to_dna_sequences(matrix_1, matrix_2, last_case):
    if last_case == '' or last_case == 'last_3':
        index_binary_length = index_binary_length_1
        N = N_data
    elif last_case == 'last_1':
        index_binary_length = index_binary_length_last_1
        N = N_data_last_1
    else:
        index_binary_length = index_binary_length_last_2
        N = N_data_last_2
        # print("N = ", N)
    dna_sequences = []
    add_random_list_number = 0
    index_numbers = []   # 用于存储选择的行的行号，是一个三维列表，eg:[[[0,0,1], [1,1,0]],...],例子中index长为3
    while len(matrix_1) + len(matrix_2) > 0:
        if len(matrix_1) > 0 and len(matrix_2) > 0:
            fixed_list = random.sample(matrix_1, 1)[0]
            matrix_1.remove(fixed_list)
            dna_sequence, another_list, flag_1 = search_result(fixed_list, matrix_2, N, index_binary_length, add_random_list_number)
            if flag_1:   # flag_1 为True表示在另一个集合中找到了配对
                matrix_2.remove(another_list)
            else:
                add_random_list_number += 1
            dna_sequences.append(dna_sequence)
            # 此时，fixed_list一定在上面，即它的行号先存（存左边）
            index_number = [fixed_list[-index_binary_length:], another_list[-index_binary_length:]]
        else:  # matrix_1已经配完，但matrix_2还有List要配
            fixed_list = random.sample(matrix_2, 1)[0]
            matrix_2.remove(fixed_list)
            # 此时matrix_1已经空了，也没有关系，还是可以这样写
            # 为了实现对data部分添加了随机bits的DNA序列也能有修改能力，这里更新一下此时的转化为DNA序列的方案。
            # 更新为如下：生成的随机bits放上面,matrix_2的数据放下面。与上面出现随机bits的情况区分开。
            # dna_sequence, another_list, flag_1 = search_result(fixed_list, matrix_1, N, index_binary_length, add_random_list_number)
            # search_result_2应用于matrix_1已经空，此时，将生成的随机bits放上面,matrix_2的数据放下面。
            dna_sequence, another_list, flag_1 = search_result_2(fixed_list, N, index_binary_length,
                                                               add_random_list_number)
            # 一定是与随机bits配对
            add_random_list_number += 1
            dna_sequences.append(dna_sequence)
            # 此时，fixed_list一定在下面，即它的行号要后存（存右边）
            index_number = [another_list[-index_binary_length:], fixed_list[-index_binary_length:]]

        index_numbers.append(index_number)
    # 还有一个问题：在解码时，我怎么知道有多少对真实的index数据呢，这对下一步把数据DNA序列转换成0/1序列很重要
    # 答：可以这样解决。把真实的行的对数存在第一个位置，即真实行号（一对）从第二个开始存。在解码时，遇到的第一对信息，指的是有多少对真实行号。
    index_count = len(index_numbers)   # index_count表示有多少对行号存放在表格中
    # print("表格中行号的对数index_count:", index_count)
    index_count_binary = list(map(int, list(str(bin(index_count))[2:].zfill(index_binary_length))))  # 一样长index_binary_length
    index_count_list = [index_count_binary, index_count_binary]
    # 考虑到index_count很重要，这里把index_count_list插入两次，当这里出错时，可以用数bit的方式确定index_count, 240517
    index_numbers.insert(0, index_count_list)
    index_numbers.insert(0, index_count_list)
    if add_random_list_number > add_random_list_threshold:  # 添加的随机bits的条数较大时，才输出
        # index瘦身后，last_2的index_1预留最少，为14条。last_1,为18条，last_3为22条。
        print("添加的随机bits的条数（不可多过22条）：", add_random_list_number)
    return dna_sequences, index_numbers

def index_1_to_dna_sequences(matrix):
    dna_sequences = []
    add_random_list_number = 0
    index_numbers = []  # 用于存储选择的行的行号，是一个三维列表，eg:[[[0,0,1], [1,1,0]],...],例子中index长为3
    while len(matrix) > 0:
        fixed_list = random.sample(matrix, 1)[0]
        matrix.remove(fixed_list)
        dna_sequence, another_list, flag_1 = search_result(fixed_list, matrix, N_index_1, index_binary_length_2, add_random_list_number)
        if flag_1:  # flag_1 为True表示在另一个集合中找到了配对
            matrix.remove(another_list)
        else:
            add_random_list_number += 1
        dna_sequences.append(dna_sequence)
        # 存行号
        # fixed_list一定在上面，即它的行号先存（存左边）
        index_number = [fixed_list[-index_binary_length_2:], another_list[-index_binary_length_2:]]
        index_numbers.append(index_number)
    index_count = len(index_numbers)  # index_count表示有多少对行号存放在表格中
    # print("表格中行号的对数index_count:", index_count)
    index_count_binary = list(
        map(int, list(str(bin(index_count))[2:].zfill(index_binary_length_2))))  # 一样长index_binary_length
    index_count_list = [index_count_binary, index_count_binary]
    # 考虑到index_count很重要，这里把index_count_list插入两次，当这里出错时，可以用数bit的方式确定index_count, 240517
    index_numbers.insert(0, index_count_list)
    index_numbers.insert(0, index_count_list)
    if add_random_list_number > add_random_list_threshold:  # 添加的随机bits的条数较大时，才输出
        print("在处理index_1到DNA序列时，添加的随机bits的条数（不可多过28条）：", add_random_list_number)
    return dna_sequences, index_numbers

def search_result(fixed_list, other_lists, N, index_binary_length, add_random_list_number):
    flag_1 = False  # flag_1为False表示没有找到另一个匹配的数据，而是生成了一个随机bits
    if len(other_lists) > 0:  # 检查另一个行编号列表中是否还有待配对的数据
        for _ in range(search_count):
            another_list = random.sample(other_lists, 1)[0]
            # fixed_list = random.sample(good_data_set, 1)[0]
            # 这里[:segment_length]对应数据部分
            n_dna_sequence = two_lists_to_sequence(fixed_list[:segment_length], another_list[:segment_length])
            # 在二层极化编码的方案中，第一层数据总是在前面（即上面），第二层数据总是在后面（即下面）
            if check(n_dna_sequence,  # 我看check函数，传入列表和字符串都行，试试直接传n_dna_sequence
                              max_homopolymer=max_homopolymer,
                              max_content=max_gc_content, ):
                flag_1 = True  # 表示找了到另一个匹配的数据
                return n_dna_sequence, another_list, flag_1
        # 或者经过最大搜索次数后没有找到合适的匹配，则生成随机bits进行匹配
    while True:
        # 这里有个bug，随机生成的行号可能会相同，这个还是要解决一下。使用add_random_list_number来解决。
        # 生成的随机bits的行号相同也没有关系，因为这些随机数据不用恢复
        # random_index = random.randint(N + 3, int(math.pow(2, index_binary_length) - 1))
        random_index = N + 10 + add_random_list_number
        index_list = list(map(int, list(str(bin(random_index))[2:].zfill(index_binary_length))))
        best_score = -1
        best_dna_sequence = []
        best_random_list = []
        for _ in range(search_count):
            # 行号放后面
            random_list = [random.randint(0, 1) for _ in range(segment_length)] + index_list
            n_dna_sequence = two_lists_to_sequence(fixed_list[:segment_length], random_list[:segment_length])
            if check(n_dna_sequence,
                              max_homopolymer=max_homopolymer,
                              max_content=max_gc_content, ):
                return n_dna_sequence, random_list, flag_1
            n_score = list_score(n_dna_sequence)
            if n_score > best_score:
                best_score = n_score
                best_dna_sequence = n_dna_sequence
                best_random_list = random_list
        print("oh, my God!竟然真的出现了%d次随机也不满足兼容的情况，太不可思议了——1！" % search_count)
        print("fixed_list:", fixed_list, len(fixed_list))
        return best_dna_sequence, best_random_list, flag_1


def search_result_2(fixed_list, N, index_binary_length, add_random_list_number):
    flag_1 = False  # flag_1为False表示没有找到另一个匹配的数据，而是生成了一个随机bits
    while True:
        # 这里有个bug，随机生成的行号可能会相同，这个还是要解决一下。使用add_random_list_number来解决。
        # 生成的随机bits的行号相同也没有关系，因为这些随机数据不用恢复
        # random_index = random.randint(N + 3, int(math.pow(2, index_binary_length) - 1))
        random_index = N + 10 + add_random_list_number
        index_list = list(map(int, list(str(bin(random_index))[2:].zfill(index_binary_length))))
        best_score = -1
        best_dna_sequence = []
        best_random_list = []
        for _ in range(search_count):
            # 行号放后面
            random_list = [random.randint(0, 1) for _ in range(segment_length)] + index_list
            # 注意：此时，随机bits放上面，数据List放下面。
            n_dna_sequence = two_lists_to_sequence(random_list[:segment_length], fixed_list[:segment_length])
            if check(n_dna_sequence,
                              max_homopolymer=max_homopolymer,
                              max_content=max_gc_content, ):
                return n_dna_sequence, random_list, flag_1
            n_score = list_score(n_dna_sequence)
            if n_score > best_score:
                best_score = n_score
                best_dna_sequence = n_dna_sequence
                best_random_list = random_list
        print("oh, my God!竟然真的出现了%d次随机也不满足兼容的情况，太不可思议了——7！" % search_count)
        print("fixed_list:", fixed_list, len(fixed_list))
        return best_dna_sequence, best_random_list, flag_1


def two_lists_to_sequence(list_up, list_down):
    dna_string = ''
    # 更新01到碱基的转换规则：考虑到有随机bits，它可能在上面也可能在下面。
    # 好的规则应该是这样的，当上面固定为数据，下面为随机bits时，上面为0或1，都有可能转化为GC或非GC。当下面固定为数据时，上面为随机bits，则是下面为0或1，都有可能转化为GC或非GC。
    # 故新的转换规则如下：00>C,01>A,10>T,11>G 240418
    for i in range(len(list_up)):
        if list_up[i] == 0 and list_down[i] == 0:
            dna_string += 'C'
        elif list_up[i] == 0 and list_down[i] == 1:
            dna_string += 'A'
        elif list_up[i] == 1 and list_down[i] == 0:
            dna_string += 'T'
        else:  # list_up[i] == 1 and list_down[i] == 1
            dna_string += 'G'
    return dna_string


def index_to_dna_two_layers(index_numbers, idx):
    dna_sequences = []
    matrix_index_ori_01 = []
    index_str_1 = ''.join(str(num) for sublist1 in index_numbers for sublist2 in sublist1 for num in sublist2)
    index_matrix_1 = segment_index(index_str_1, segment_length_index_1, matrix_row_num_index_1, more_data_length_1)
    index_matrix_np_1 = np.array(index_matrix_1)
    transposed_index_matrix_1 = index_matrix_np_1.T
    encoded_index_matrix_1 = []
    for row in range(len(transposed_index_matrix_1)):
        encode_list = polar_encode(transposed_index_matrix_1[row], select_index_index_1,
                                                freeze_index_index_1, frozen_bits_index_1, N_index_1, crc_index_1)
        encoded_index_matrix_1.append(encode_list)
    del transposed_index_matrix_1
    encoded_index_np_1 = np.array(encoded_index_matrix_1)
    encoded_index_np_1 = encoded_index_np_1.T
    if idx == 0:  # 与数据部分一样，只用检查第0个矩阵就可以了
        flag = True
        for bit in encoded_index_np_1[0]:
            if bit != 0:
                flag = False
                break
        if flag:
            print("crc_8时，index_1部分，encoded_index_np_1[0]全是0")
        else:
            print("出错了，index_1部分，encoded_index_np_1[0]不全是0")
            print("encoded_index_np_1[0]:", encoded_index_np_1[0])
    # print("index部分，encoded_index_np[0]:", encoded_index_np[0])
    encoded_index_matrix_1 = encoded_index_np_1.tolist()
    save_index_1_ori = copy.deepcopy(encoded_index_matrix_1)
    matrix_index_ori_01.append(save_index_1_ori)

    encoded_index_matrix_1 = encoded_index_matrix_1[1:]  # 删除第0行数据
    encoded_index_matrix_1 = connect_all(encoded_index_matrix_1, index_binary_length_2)  # 先加上行号，行号在数据的后面
    dna_sequences_1, index_numbers_1 = index_1_to_dna_sequences(encoded_index_matrix_1)  # index_numbers_1是index_1的行号匹配情况
    dna_sequences.append(dna_sequences_1)

    # 处理index_numbers_1
    index_str_2 = ''.join(str(num) for sublist1 in index_numbers_1 for sublist2 in sublist1 for num in sublist2)
    index_matrix_2 = segment_index(index_str_2, segment_length_index_2, matrix_row_num_index_2, more_data_length_2)
    index_matrix_np_2 = np.array(index_matrix_2)
    transposed_index_matrix_2 = index_matrix_np_2.T
    encoded_index_matrix_2 = []
    for row in range(len(transposed_index_matrix_2)):
        encode_list = polar_encode(transposed_index_matrix_2[row], select_index_index_2,
                                                freeze_index_index_2, frozen_bits_index_2, N_index_2, crc_index_2)
        encoded_index_matrix_2.append(encode_list)
    del transposed_index_matrix_2
    encoded_index_np_2 = np.array(encoded_index_matrix_2)
    encoded_index_np_2 = encoded_index_np_2.T
    if idx == 0:  # 与数据部分一样，只用检查第0个矩阵就可以了
        flag = True
        for bit in encoded_index_np_2[0]:
            if bit != 0:
                flag = False
                break
        if flag:
            print("crc_4时，index_2部分，encoded_index_np_2[0]全是0")
        else:
            print("出错了，index_2部分，encoded_index_np_2[0]不全是0")
            print("encoded_index_np_2[0]:", encoded_index_np_2[0])
    # print("index部分，encoded_index_np[0]:", encoded_index_np[0])
    encoded_index_matrix_2 = encoded_index_np_2.tolist()
    save_index_2_ori = copy.deepcopy(encoded_index_matrix_2)
    matrix_index_ori_01.append(save_index_2_ori)

    encoded_index_matrix_2 = encoded_index_matrix_2[1:]  # 删除第0行数据
    # index_2的数据不用加行号，因为这里不用表格来存，而是每条数据转为1条DNA序列
    dna_sequences_2 = index_lists_to_dna(encoded_index_matrix_2)
    dna_sequences.append(dna_sequences_2)  # 在双层index方案中，index_dna矩阵也又两个矩阵构成
    return dna_sequences, matrix_index_ori_01


def index_to_dna_single_layer(index_numbers, last_case):
    dna_sequences = []
    # 使用两层嵌套列表推导式提取0和1，并将其组成字符串
    if last_case == 'last_1':
        segment_length_index_1_last = segment_length_index_1_last_1
        matrix_row_num_index_1_last = matrix_row_num_index_1_last_1
        more_data_length_last = more_data_length_last_1
        select_index_index, freeze_index_index = select_index_index_1_last_1, freeze_index_index_1_last_1
        frozen_bits_index, N_index, crc_index = frozen_bits_index_1_last_1, N_index_1_last_1, crc_index_1_last
    else:  # last_case == 'last_2':
        segment_length_index_1_last = segment_length_index_1_last_2
        matrix_row_num_index_1_last = matrix_row_num_index_1_last_2
        more_data_length_last = more_data_length_last_2
        select_index_index, freeze_index_index = select_index_index_1_last_2, freeze_index_index_1_last_2
        frozen_bits_index, N_index, crc_index = frozen_bits_index_1_last_2, N_index_1_last_2, crc_index_1_last

    index_str = ''.join(str(num) for sublist1 in index_numbers for sublist2 in sublist1 for num in sublist2)
    index_matrix = segment_index(index_str, segment_length_index_1_last, matrix_row_num_index_1_last, more_data_length_last)
    del index_str, index_numbers
    index_matrix_np = np.array(index_matrix)
    transposed_index_matrix = index_matrix_np.T
    encoded_index_matrix = []
    for row in range(len(transposed_index_matrix)):
        encode_list = polar_encode(transposed_index_matrix[row], select_index_index,
                                                freeze_index_index, frozen_bits_index, N_index, crc_index)
        encoded_index_matrix.append(encode_list)
    del transposed_index_matrix
    encoded_index_np = np.array(encoded_index_matrix)
    encoded_index_np = encoded_index_np.T

    flag = True
    for bit in encoded_index_np[0]:
        if bit != 0:
            flag = False
    if flag:
        print("最后一个矩阵的单层index方案，encoded_index_np[0]全是0 " + last_case)
    else:
        print("出错了，最后一个矩阵的单层index方案，encoded_index_np[0]不全是0 " + last_case)
        print("encoded_index_np[0]:", encoded_index_np[0])
    # print("index部分，encoded_index_np[0]:", encoded_index_np[0])
    encoded_index_matrix = encoded_index_np.tolist()
    save_index_1_ori = copy.deepcopy(encoded_index_matrix)
    encoded_index_matrix = encoded_index_matrix[1:]  # 删除第0行数据


    del encoded_index_np
    dna_sequences_1 = index_lists_to_dna(encoded_index_matrix)
    dna_sequences.append(dna_sequences_1)
    dna_sequences_2 = []
    dna_sequences.append(dna_sequences_2)  # 让index部分的DNA矩阵形式上保持统一，都由两个小矩阵构成
    matrix_index_ori_01 = [save_index_1_ori, []]  # 让index部分的DNA矩阵形式上保持统一，都由两个小矩阵构成
    return dna_sequences, matrix_index_ori_01


def index_lists_to_dna(encoded_index_matrix):
    dna_sequences = []
    for row in range(len(encoded_index_matrix)):
        dna_sequence = one_list_to_dna(encoded_index_matrix[row])
        dna_sequences.append(dna_sequence)
    # 有一个重要的问题，生成的DNA数据矩阵的行数是不确定的（每一个单位矩阵都不同），会略大于2^14，如此的话，怎么区分数据矩阵与index矩阵
    # 可以这样，就不用存数据矩阵的行数。在加碱基行号时，分为两类，一类是数据矩阵的，一类是index矩阵的。我的单位矩阵也由两个矩阵构成
    return dna_sequences
def one_list_to_dna(index_list):
    # 每条index_list与一条随机bits进行组合，index_list总是上面
    best_score = -1
    best_dna_sequence = []
    # best_random_list = []
    for _ in range(search_count):
        # 生成的随机bits后面，这里不需要行号
        random_list_index = [random.choice([0, 1]) for _ in range(segment_length)]
        n_dna_sequence = two_lists_to_sequence(index_list, random_list_index)
        if check(n_dna_sequence,
                          max_homopolymer=max_homopolymer,
                          max_content=max_gc_content, ):
            return n_dna_sequence
        n_score = list_score(n_dna_sequence)
        if n_score > best_score:
            best_score = n_score
            best_dna_sequence = n_dna_sequence
            # best_random_list = random_list_index
    print("oh, my God!竟然真的出现了%d次随机也不满足兼容的情况，太不可思议了——2！" % search_count)
    print("index_list:", index_list, len(index_list))
    return best_dna_sequence

def segment_index(index_str, segment_length_index, matrix_row_num_index, more_data_length):
    # 支持每行添加 2 位的奇偶校验，保证每行异或运算后为0
    size = len(index_str)
    # more_data_length = segment_length % (2 * index_binary_length)   # 表示每行行号数据与segment_length的差，这要用随机bits补上
    # segment_length_index = segment_length - more_data_length
    index_matrix = [[0 for _ in range(segment_length_index)] for _ in range(math.ceil(size / segment_length_index))]
    # index_matrix的最后一行一般会有多余的0，可以考虑把这些0换成随机bits，以方便转换成兼容的DNA序列
    # 答：看了编码的图，感觉还是换成随机bits更好。
    for idx in range(len(index_str)):
        row = idx // segment_length_index
        col = idx % segment_length_index
        index_matrix[row][col] = int(index_str[idx])
    # 将最后一行中多余的0，进行随机化
    more_0_num = segment_length_index - size % segment_length_index
    last_row = len(index_matrix) - 1
    if more_0_num != segment_length_index:  # 当size % segment_length_index = 0时，最后一行没有多余的0
        random_list = [random.randint(0, 1) for _ in range(more_0_num)]
        index_matrix[last_row][-more_0_num:] = random_list
    # 这里还有一个问题，信息位不够时，要用随机bits补全
    random_row_number = matrix_row_num_index - len(index_matrix)
    if random_row_number < 0:
        print("出错啦，index矩阵中，真实index的行数大于预留的行数，请增加matrix_row_num_index！")
    if random_row_number != 0:
        for _ in range(random_row_number):
            random_list = [random.randint(0, 1) for _ in range(segment_length_index)]
            index_matrix.append(random_list)
    # 把每行长度提高到segment_length
    if more_data_length != 0:
        for i in range(len(index_matrix)):
            random_list_index = [random.randint(0, 1) for _ in range(more_data_length)]
            index_matrix[i] = index_matrix[i] + random_list_index
    # 对每一行添加 2 位的奇偶检验位，两个校验位为：偶+奇。保证每行异或后为0
    for my_list in index_matrix:
        even_parity, odd_parity = 0, 0  # 偶校验位， 奇校验位
        for i in range(0, len(my_list), 2):
            even_parity ^= my_list[i]
            odd_parity ^= my_list[i + 1]
        my_list.append(even_parity)  # 240
        my_list.append(odd_parity)  # 241
    return index_matrix
