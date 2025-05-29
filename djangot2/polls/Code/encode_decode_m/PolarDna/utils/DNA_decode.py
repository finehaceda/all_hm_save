from collections import Counter

import numpy as np
import copy
import math
import Levenshtein
from joblib import Parallel, delayed

from .chooseparams import PolarParams
from .frozen_bits import frozen_bits_108, frozen_bits_97
from .select_good_channels_for_polar import SelectGoodChannels4Polar_two_layer_npy
from ..Polar_code import Polar_decode_cython as Polar_decode
from . import DNA_encode
from .glob_var import step_length as step, Index_data_two_layer
from .glob_var import (segment_length, segment_length_index_1, index_binary_length_1, N_data, freeze_index_index_1,
                            # freeze_index_data_1, select_index_index_1, select_index_data_1, frozen_bits_data_1,
                            select_index_index_1,
                            N_index_2,
                            crc_data, crc_index_1_last, segment_length_index_1_last_1, segment_length_index_1_last_2,
                            crc_index_1, segment_n, check_length, threads_number, N_index_1, back_length, List_data,
                            # frozen_bits_index_1, freeze_index_data_2, select_index_data_2, frozen_bits_data_2,
                            frozen_bits_index_1,
                            select_index_index_2, freeze_index_index_2, frozen_bits_index_2, List_index_1, crc_index_2,
                            segment_length_index_2, index_binary_length_2, List_index_2, N_index_1_last_1,
                            N_index_1_last_2,
                            select_index_index_1_last_1, freeze_index_index_1_last_1, frozen_bits_index_1_last_1,
                            list_last,
                            select_index_index_1_last_2, freeze_index_index_1_last_2, frozen_bits_index_1_last_2,
                            index_binary_length_last_1, index_binary_length_last_2, N_data_last_1, N_data_last_2,
                            select_index_data_1_last_1, freeze_index_data_1_last_1, frozen_bits_data_1_last_1,
                            crc_data_last_1,
                            select_index_data_1_last_2, freeze_index_data_1_last_2, frozen_bits_data_1_last_2,
                            crc_data_last_2,
                            select_index_data_2_last_2, freeze_index_data_2_last_2, frozen_bits_data_2_last_2,
                            select_index_data_2_last_1, freeze_index_data_2_last_1, frozen_bits_data_2_last_1,
                            parity_length, list_16, larger, )

# num_total_sub, num_total_del, num_total_ins = 0, 0, 0
# num_total_mis_dna_seqs = 0  # 表示丢失的DNA序列条数
# frozen_bits_data_1,frozen_bits_data_2 = np.array(frozen_bits_108, dtype=np.intc),np.array(frozen_bits_97, dtype=np.intc)
# matrix_row_num_data_1 = N_data - len(crc_data) + 1 - len(frozen_bits_data_1)
# matrix_row_num_data_2 = N_data - len(crc_data) + 1 - len(frozen_bits_data_2)
# select_index_data_1, freeze_index_data_1, select_index_data_2, freeze_index_data_2 = (
#     SelectGoodChannels4Polar_two_layer_npy(Index_data_two_layer, matrix_row_num_data_1 + len(crc_data) - 1, matrix_row_num_data_2 + len(crc_data) - 1))

def dna_decode_new(matrices_dna, dna_seqs_ture_num, frozen_bits_len):  # 新的函数将实现对接收到的DNA矩阵的修改功能，实现双层index，并且支持质量值
    print(f'frozen_bits_len:{frozen_bits_len}')
    polarParams = PolarParams(N_data, crc_data, frozen_bits_len)
    # 支持每行添加 2 位奇偶检验，保证每行 偶和奇 异或运算结果为0, 20240816
    # 添加一个新功能，记录下最后一次解码时的01矩阵。
    # 以完成两件事：1、支持与原正确01矩阵做对比，找到SCL解码时，收到01数据的真实错误率；2、支持某列未通过CRC检验时，其list * 2，再试一次。记录并反馈结果。240828
    global num_total_sub, num_total_del, num_total_ins, num_total_mis_dna_seqs


    num_total_sub, num_total_del, num_total_ins, num_total_mis_dna_seqs = 0, 0, 0, 0
    matrices_bits = []  # 解码后得到的矩阵
    matrices_received_01 = []  # 经过多轮修改后，最后1次解码前的01矩阵，可用与后面与原正确01矩阵做对比，确定解码时，各矩阵各列的错误率
    matrix_num = len(matrices_dna)
    for idx in range(matrix_num - 1):  # 先处理前面的矩阵
        # 将DNA矩阵与质量值矩阵进行分离
        # 有一个大问题：质量值是与DNA序列匹配的，但DNA序列转为0/1序列后，不是顺序往下放的，有行号控制。所以，DNA的质量值得与转化后的0/1序列匹配才行
        # 要解决这一个问题，就只能将DNA序列与质量值分离的操作放在下面的每一个操作过程中
        index_numbers_2, matrix_index_2_received_01 = matrix_dna2bits_2(matrices_dna[idx], polarParams)  # 恢复index_1的行号匹配，即表格index_2的内容
        index_numbers_1, matrix_index_1_received_01 = matrix_dna2bits_1(matrices_dna[idx],
                                                                        index_numbers_2, polarParams)  # 恢复data的行号匹配，即表格index_1的内容
        matrix_bits_decode_all, lists_data_received_01 = matrix_decode_new(matrices_dna[idx],
                                                                           index_numbers_1, polarParams)  # 恢复的数据矩阵
        matrix_received_01 = [lists_data_received_01, [matrix_index_1_received_01, matrix_index_2_received_01]]
        matrices_received_01.append(matrix_received_01)
        matrices_bits.append(matrix_bits_decode_all)
    # 处理最后一个矩阵
    last_true_num = count_last_num(matrices_dna[matrix_num - 1][0])  # 去数data部分更好
    print("last_true_num:", last_true_num)
    # 先根据最后一个index_DNA矩阵的第2个矩阵是否为空来判断是不是情况3,因为last_3与last_2在最后一个Index_DNA的第1个矩阵DNA条数会比较接近
    if last_true_num > N_data - 60:
        last_case = 'last_3'
        print("最后一个矩阵的情况是： " + last_case)
        # 如果最后一个矩阵的index_2的DNA序列删除了1条，只有62条了，会怎么样
        index_numbers_2, matrix_index_2_received_01 = matrix_dna2bits_2(matrices_dna[matrix_num - 1], polarParams)  # 恢复index_1的行号匹配，即表格index_2的内容
        index_numbers_1, matrix_index_1_received_01 = matrix_dna2bits_1(matrices_dna[matrix_num - 1],
                                                                        index_numbers_2, polarParams)  # 恢复data的行号匹配，即表格index_1的内容
        matrix_bits_decode_all, lists_data_received_01 = matrix_decode_new(matrices_dna[matrix_num - 1],
                                                                           index_numbers_1, polarParams)  # 恢复的数据矩阵
        matrix_received_01 = [lists_data_received_01, [matrix_index_1_received_01, matrix_index_2_received_01]]
    elif last_true_num < N_data_last_1 + 60:  # last_1
        last_case = 'last_1'
        print("最后一个矩阵的情况是： " + last_case)
        matrix_bits_decode_all, matrix_received_01 = matrix_decode_last(matrices_dna[matrix_num - 1], polarParams, last_case)
    else:
        last_case = 'last_2'
        print("最后一个矩阵的情况是： " + last_case)
        matrix_bits_decode_all, matrix_received_01 = matrix_decode_last(matrices_dna[matrix_num - 1], polarParams, last_case)
    matrices_bits.append(matrix_bits_decode_all)
    matrices_received_01.append(matrix_received_01)
    print("num_total_sub = %d, num_total_del = %d, num_total_ins = %d" % (num_total_sub, num_total_del, num_total_ins))
    print("发现整条DNA序列丢失的数量为 %d, 丢失率为 %.4f" % (
        num_total_mis_dna_seqs, num_total_mis_dna_seqs / dna_seqs_ture_num))
    return matrices_bits, matrices_received_01


def count_last_num(matrix_dna):  # 数数该矩阵中有多少条DNA序列，不算空列表
    num = 0
    for dna_and_phred in matrix_dna:
        if dna_and_phred[0]:
            num += 1
    return num


def matrix_decode_last(matrix_dna_last, polarParams, last_case):
    # 支持返回最后1次解码时的接收到的01矩阵，以及对未通过CRC列的larger_list再次解码
    index_numbers_1_last, matrix_index_1_received_01_last = matrix_dna2bits_1_last(matrix_dna_last, polarParams, last_case)
    matrix_bits_decode_all, lists_data_received_01 = matrix_decode_new(matrix_dna_last, index_numbers_1_last, polarParams, last_case)
    matrix_received_01 = [lists_data_received_01, [matrix_index_1_received_01_last, []]]
    return matrix_bits_decode_all, matrix_received_01


def matrix_dna2bits_1_last(matrix_dna, polarParams, last_case):
    # 本函数实现对最后一个矩阵的index_1内容的恢复，即data的行号匹配情况，只处理last_1和last_2两种情况
    global num_total_sub, num_total_del, num_total_ins
    # x_hat_merge = [[] for _ in range(N_index - 1)]  # -1是因为编码后删掉了第0行
    # matrix_list_decode_new = []
    # 支持任意切割
    matrix_list_decode_new = [[] for _ in range(segment_length)]
    matrix_list_decode_flag = [[] for _ in range(segment_length)]
    matrix_list_received_01_column = [[] for _ in range(segment_length)]  # 记录下最后1次解码时接收到的01信息
    matrix_list_received_phred_column = [[] for _ in range(segment_length)]  # 记录下最后1次解码时接收到的01信息的质量值
    # 先清洗掉index_1后面的空列表
    if last_case == 'last_1':
        matrix_dna[1][0] = matrix_dna[1][0][:N_index_1_last_1 - 1]  # -1是因为干掉了第0行
    else:  # last_2
        matrix_dna[1][0] = matrix_dna[1][0][:N_index_1_last_2 - 1]

    for i in range(1, segment_n + 1):
        matrix_index_bits, matrix_random_bits, matrix_index_phred, matrix_random_phred = matrix_dna_2_to_list_index_new(
            matrix_dna[1][0], last_case)
        # 对index_1_last部分数据进行解码
        insert_zeroes = [0 for _ in range(segment_length)]  # 用crc_16、crc_4和crc_8要插入
        matrix_index_bits.insert(0, insert_zeroes)  # 这个插入操作对修改DNA矩阵时，是有影响的
        phred_initial_value = [0.999999 for _ in range(segment_length)]
        matrix_index_phred.insert(0, phred_initial_value)
        matrix_index_bits = np.array(matrix_index_bits, dtype=object)  # 就算是多层List,也会全部变成np.array
        matrix_index_bits = matrix_index_bits.T
        matrix_index_phred = np.array(matrix_index_phred, dtype=object)
        matrix_index_phred = matrix_index_phred.T
        if i > back_length:
            temp_n_0 = (i - back_length) * step
        else:
            temp_n_0 = 0
        # temp_n_1 = i * step
        temp_n_1 = min(segment_length, i * step)
        x_hat_lists = []
        # 考虑并行
        # results = (
        #     Parallel(n_jobs=threads_number)(
        #         delayed(CASCL_decoder_index_1_last)(index_array, last_case) for index_array in
        #         matrix_index_bits[temp_n_0:temp_n_1]))
        results = (
            Parallel(n_jobs=threads_number)(
                delayed(CASCL_decoder_index_1_last)(matrix_index_bits[j], matrix_index_phred[j], last_case) for j in
                range(temp_n_0, temp_n_1)))
        # print(f"???")
        for result in results:
            x_hat_lists.append(result[1])
        x_hat_lists = np.array(x_hat_lists)
        x_hat_lists = x_hat_lists.T
        x_hat_lists = x_hat_lists.tolist()
        x_hat_lists = x_hat_lists[1:]  # 要转成DNA序列，就要去掉第0行
        if i >= back_length:  # 把回头部分的解码结果保存下来
            for j in range(step):
                # matrix_list_decode_new.append(results[j][0])
                matrix_list_decode_new[temp_n_0 + j] = results[j][0]
                matrix_list_decode_flag[temp_n_0 + j] = results[j][2]
                matrix_list_received_01_column[temp_n_0 + j] = matrix_index_bits[temp_n_0 + j]
                matrix_list_received_phred_column[temp_n_0 + j] = matrix_index_phred[temp_n_0 + j]
        # 开始检查并修改接收到的DNA矩阵中的错误
        for j in range(len(x_hat_lists)):
            # 怎么把解码的部分01bit给找出来，利用x_hat_merge
            # 要是随机bits(matrix_random_bits)错了呢，这样恢复的DNA序列就不对了，这显然也会影响对原DNA矩阵的修改成功率
            # 所以，从这个角度来讲，我们也不能处理太高的错误情况。不然，最后一个index矩阵会解码出错，导致前面的data部分也不能正确恢复
            dna_sequence_decode = DNA_encode.two_lists_to_sequence(x_hat_lists[j],
                                                                   matrix_random_bits[j][temp_n_0:temp_n_1])
            # print("dna_sequence_decode:", dna_sequence_decode, len(dna_sequence_decode))
            # print("matrix_dna[1][j][:temp_n]:", matrix_dna[1][j][:temp_n], len(matrix_dna[1][j][:temp_n]))
            if dna_sequence_decode == matrix_dna[1][0][j][0][temp_n_0:temp_n_1]:
                # pass
                for k in range(temp_n_0, temp_n_1):
                    matrix_dna[1][0][j][1][k] = 0.9998  # 得修改原DNA序列中的质量值才行
            else:  # 不相等时，才需要修改
                edit_distance = Levenshtein.distance(dna_sequence_decode, matrix_dna[1][0][j][0][temp_n_0:temp_n_1])
                if edit_distance > 6:
                    pass
                    # print("edit_distance太大了！在最后一个矩阵的index的修改中。j = %d, edit_distance = %d" % (
                    #     j, edit_distance))
                    # print("dna_sequence_decode:                   ", dna_sequence_decode, len(dna_sequence_decode))
                    # print("matrix_dna[1][0][j][0][temp_n_0:temp_n_1]:", matrix_dna[1][0][j][0][temp_n_0:temp_n_1],
                    #       len(matrix_dna[1][0][j][0][temp_n_0:temp_n_1]))
                num_sub, num_del, num_ins, dna_sequence_rec = check_and_edit_dna_sequence(dna_sequence_decode,
                                                                                          matrix_dna[1][0][j], temp_n_0)
                num_total_sub += num_sub
                num_total_del += num_del
                num_total_ins += num_ins
                matrix_dna[1][0][j] = dna_sequence_rec

    # 现在只用最后2个step的解码结果,看看能否再减少解码用时
    matrix_index_bits, _, matrix_index_phred, _ = matrix_dna_2_to_list_index_new(matrix_dna[1][0], last_case)
    insert_zeroes = [0 for _ in range(segment_length)]  # 用crc_16就又要插入
    matrix_index_bits.insert(0, insert_zeroes)  # 这个插入操作对修改DNA矩阵时，是有影响的
    matrix_index_bits = np.array(matrix_index_bits)  # 就算是多层List,也会全部变成np.array
    matrix_index_bits = matrix_index_bits.T

    phred_initial_value = [0.999999 for _ in range(segment_length)]
    matrix_index_phred.insert(0, phred_initial_value)  # 这里第0行一定是0的，其质量值应该是非常大才对，赶紧改过来240912
    matrix_index_phred = np.array(matrix_index_phred, dtype=object)
    matrix_index_phred = matrix_index_phred.T
    # results = (
    #     Parallel(n_jobs=threads_number)(
    #         delayed(CASCL_decoder_index_1_last)(index_array, last_case) for index_array in
    #         matrix_index_bits[-2 * step:]))
    results = (
        Parallel(n_jobs=threads_number)(
            delayed(CASCL_decoder_index_1_last)(matrix_index_bits[j], matrix_index_phred[j], last_case) for j in
            range(-2 * step, 0)))
    # for result in results:
    #     matrix_list_decode_new.append(result[0])
    temp_n_0 = -2 * step
    for j in range(len(results)):
        matrix_list_decode_new[temp_n_0 + j] = results[j][0]
        matrix_list_decode_flag[temp_n_0 + j] = results[j][2]
        matrix_list_received_01_column[temp_n_0 + j] = matrix_index_bits[temp_n_0 + j]  # 注意到这里也是完整矩阵
        matrix_list_received_phred_column[temp_n_0 + j] = matrix_index_phred[temp_n_0 + j]

    # 尝试对其中没有通过CRC检验的列，以增加list的方式，再进行SCL解码1次
    CASCL_decoder_name = "CASCL_decoder_index_1_last"
    matrix_list_decode_new, matrix_list_decode_flag = CASCL2failCRCwithlargerlist(CASCL_decoder_name, matrix_list_decode_new,
                                                         matrix_list_decode_flag, matrix_list_received_01_column,
                                                         matrix_list_received_phred_column, _, _, polarParams, last_case)  # 这里一定要输入last_case，不然会报错

    # 处理最后的解码结果
    matrix_list_decode_new = np.array(matrix_list_decode_new)
    matrix_list_decode_new = matrix_list_decode_new.T
    matrix_list_decode = matrix_list_decode_new.tolist()

    # 把matrix_list_encode_column也进行转置
    matrix_list_received_01_column = np.array(matrix_list_received_01_column)
    matrix_list_received_01_column = matrix_list_received_01_column.T
    matrix_list_received_01 = matrix_list_received_01_column.tolist()

    # 去掉每行后面添加的随机数据
    matrix_index_lists = []
    if last_case == 'last_1':
        segment_length_index_1_last = segment_length_index_1_last_1
        index_binary_length_last = index_binary_length_last_1
    else:
        segment_length_index_1_last = segment_length_index_1_last_2
        index_binary_length_last = index_binary_length_last_2
    for list_decode in matrix_list_decode:
        index_list = list_decode[:segment_length_index_1_last + parity_length]
        matrix_index_lists.append(index_list)
    # 进行奇偶校验，并去掉最后的奇偶检验位
    matrix_index_lists = handle_parity(matrix_index_lists, matrix_list_decode_flag)
    # 恢复表格index_1_last，即data_last中行号匹配情况
    index_numbers_1_last = []
    two_index_binary_length = 2 * index_binary_length_last
    index_numbers_count_binary = matrix_index_lists[0][:two_index_binary_length]
    # print("index_numbers_count_binary:", index_numbers_count_binary, len(index_numbers_count_binary), type(index_numbers_count_binary))

    if index_numbers_count_binary[:index_binary_length_last] != index_numbers_count_binary[index_binary_length_last:]:
        print("最后一个矩阵的记录index_1_last表格的行号对数，不一致！ " + last_case)
        print("采用投票算法，从4个行号对数中确定正确行号！")
        index_count_list_1 = matrix_index_lists[0][:index_binary_length_last]
        index_count_list_2 = matrix_index_lists[0][index_binary_length_last:two_index_binary_length]
        index_count_list_3 = matrix_index_lists[0][
                             two_index_binary_length:two_index_binary_length + index_binary_length_last]
        index_count_list_4 = matrix_index_lists[0][
                             two_index_binary_length + index_binary_length_last:2 * two_index_binary_length]

        index_count_lists = [index_count_list_1,
                             index_count_list_2,
                             index_count_list_3,
                             index_count_list_4]
        print("index_count_lists =", index_count_lists)
        index_count_binary = vote(index_count_list_1, index_count_list_2, index_count_list_3, index_count_list_4)
    else:
        # print("一切正常！记录index_2表格的行号对数是一致的！")   # 增加index部分的frozen_bits后，这里通常没有问题了
        index_count_binary = index_numbers_count_binary[:index_binary_length_last]
    index_numbers_count_string = ''.join([str(bit) for bit in index_count_binary])
    index_numbers_count = int(index_numbers_count_string, 2)  # 将二进制转换成整数
    matrix_index_lists[0] = matrix_index_lists[0][2 * two_index_binary_length:]  # 处理掉第0行中，存行号对数的数据, 支持存2对
    count = 0
    flag = True
    for list_index in matrix_index_lists:
        for i in range(0, len(list_index), two_index_binary_length):
            index_up = list_index[i: i + index_binary_length_last]
            index_down = list_index[i + index_binary_length_last: i + two_index_binary_length]
            index_up_int = int(''.join([str(bit) for bit in index_up]), 2)
            index_down_int = int(''.join([str(bit) for bit in index_down]), 2)
            index_numbers_1_last.append([index_up_int, index_down_int])
            count += 1
            if count == index_numbers_count:
                flag = False
                break
        if not flag:  # 已经恢复完index的表格
            break
    return index_numbers_1_last, matrix_list_received_01


def CASCL_decoder_index_1_last(index_array, phred_array, last_case, my_list=list_last):  # 专门用来对最后一个矩阵的第一层index_1进行解码
    llr = Polar_decode.SigReceive2llr_single_layer(index_array, phred_array)
    if last_case == 'last_1':
        array_decode, x_hat, crc_flag = Polar_decode.CASCL_decoder(N_index_1_last_1, llr, select_index_index_1_last_1,
                                                                   freeze_index_index_1_last_1,
                                                                   frozen_bits_index_1_last_1, my_list,
                                                                   crc_index_1_last)
    else:  # last_2
        array_decode, x_hat, crc_flag = Polar_decode.CASCL_decoder(N_index_1_last_2, llr, select_index_index_1_last_2,
                                                                   freeze_index_index_1_last_2,
                                                                   frozen_bits_index_1_last_2, my_list,
                                                                   crc_index_1_last)
    return array_decode, x_hat, crc_flag


def CASCL_decoder_index_2(index_array, phred_array, my_list=List_index_2):  # 专门用来对第二层index进行解码, 支持质量值
    llr = Polar_decode.SigReceive2llr_single_layer(index_array, phred_array)
    array_decode, x_hat, crc_flag = Polar_decode.CASCL_decoder(N_index_2, llr, select_index_index_2,
                                                               freeze_index_index_2,
                                                               frozen_bits_index_2, my_list, crc_index_2)
    return array_decode, x_hat, crc_flag


def CASCL_decoder_index_1(index_array, phred_array, my_list=List_index_1):  # 专门用来对第一层index进行解码, 支持质量值
    llr = Polar_decode.SigReceive2llr_single_layer(index_array, phred_array)
    array_decode, x_hat, crc_flag = Polar_decode.CASCL_decoder(N_index_1, llr, select_index_index_1,
                                                               freeze_index_index_1,
                                                               frozen_bits_index_1, my_list, crc_index_1)
    return array_decode, x_hat, crc_flag


# def check_and_edit_dna_sequence_new(dna_sequence_decode, dna_sequence_receive, temp_n_0):  # 这个版本用两DNA序列对比来做
#     # 上面传进来的两个参数，应该都是string
#     # dna_sequence_rec_copy = list(dna_sequence_receive)
#     dna_sequence_dec_long = list(dna_sequence_receive)
#     # 这样做的好处是用这个对比算法，如果前面3个碱基有2个以上的错误，其结果会很奇怪，这样就基本不会出现奇怪结果
#     for i in range(len(dna_sequence_decode)):
#         j = i + temp_n_0
#         if j >= len(dna_sequence_dec_long):
#             dna_sequence_dec_long.append(dna_sequence_decode[i])
#         else:
#             dna_sequence_dec_long[j] = dna_sequence_decode[i]
#     dna_sequence_dec_long = ''.join(dna_sequence_dec_long)
#     num_sub, num_del, num_ins = 0, 0, 0
#     n_1 = len(dna_sequence_decode)
#     seq_dec_align, seq_rec_align = two_dna_seq_align(dna_sequence_dec_long, dna_sequence_receive)
#     seq_dec_align = list(seq_dec_align)
#     seq_rec_align = list(seq_rec_align)
#     for i in range(len(seq_rec_align)):  # seq_dec_align, seq_rec_align 保证一样长
#         if seq_dec_align[i] != seq_rec_align[i]:
#             if seq_dec_align[i] != '-' and seq_rec_align[i] != '-':  # 发生替换错误
#                 seq_rec_align[i] = seq_dec_align[i]
#                 num_sub += 1
#             elif seq_dec_align[i] == '-':  # 发生插入错误
#                 seq_rec_align[i] = 'D'  # 为了防止直接删除导致碱基移动，先用D表示该位碱基要删除
#                 num_del += 1
#             else:  # seq_rec_align[i] = '-' 发生删除错误
#                 seq_rec_align[i] = seq_dec_align[i]
#                 num_ins += 1
#     dna_seq_rec = [base for base in seq_rec_align if base != 'D']
#     dna_seq_rec = ''.join(dna_seq_rec)
#     return num_sub, num_del, num_ins, dna_seq_rec


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


def check_and_edit_dna_sequence_simulator(dna_sequence_decode, dna_sequence_receive, temp_n_0):
    # 仅用于在badread后的聚类任务
    # 如果dna_sequence_receive是一个空列表会怎么样？
    # 答：看起空列表，没有办法修改。试试用全'C'。
    # 上面传进来的两个参数，应该都是string
    # dna_sequence_receive只有一个元素，即str的DNA序列
    # phred_list_copy = dna_sequence_receive[1].copy()
    # print(type(phred_list_copy), phred_list_copy)
    dna_sequence_rec_copy = list(dna_sequence_receive)
    if not dna_sequence_rec_copy:
        for _ in range(segment_length):
            dna_sequence_rec_copy.append('C')  # 与初始全0能对上
            # phred_list_copy.append(0.5)
    # print("dna_sequence_receive:", type(dna_sequence_receive), len(dna_sequence_receive))
    # print("dna_sequence_rec_copy:", type(dna_sequence_rec_copy), len(dna_sequence_rec_copy))
    num_sub, num_del, num_ins = 0, 0, 0
    for i in range(len(dna_sequence_decode)):
        j = i + temp_n_0
        if j >= len(dna_sequence_rec_copy):  # 发生了删除错误
            # print("i =", i)
            dna_sequence_rec_copy.append(dna_sequence_decode[i])
            # phred_list_copy.append(0.999)
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
                # phred_list_copy[j] = 0.999  # 替换错误，修改对应的质量值
            elif edit_distance_del <= edit_distance_sub and edit_distance_del <= edit_distance_ins:  # 判定为删除错误
                # 3代删除错误似乎比插入会略多一点，试试效果，240724
                num_del += 1
                dna_sequence_rec_copy.insert(j, dna_sequence_decode[i])
                # phred_list_copy.insert(j, 0.999)
            else:  # 判定为插入错误
                num_ins += 1
                del dna_sequence_rec_copy[j]
                # del phred_list_copy[j]
    # 检查phred_list_copy的长度，保证其长度为segment_length。不需要了，因为在函数base_phred_to_01_phred中会保证输出的质量值长度为segment_length
    # if len(phred_list_copy) < segment_length:
    #     num = segment_length - len(phred_list_copy)
    #     for _ in range(num):
    #         phred_list_copy.append(0.5)
    # else:
    #     phred_list_copy = phred_list_copy[:segment_length]
    dna_sequence_modify = ''.join(dna_sequence_rec_copy)
    # dna_sequence_modify = [dna_sequence_rec_copy, dna_sequence_receive[1]]
    return num_sub, num_del, num_ins, dna_sequence_modify


def matrix2dna_and_phred(matrix):
    # 实现将（非最后一个）含DNA和质量值的矩阵分离成只含DNA序列的矩阵和只含质量值的矩阵, 有index_1 和 index_2
    matrix_dna_data = [[] for _ in range(N_data)]
    matrix_dna_index_1 = [[] for _ in range(N_index_1)]
    matrix_dna_index_2 = [[] for _ in range(N_index_2)]
    matrix_dna_index = [matrix_dna_index_1, matrix_dna_index_2]
    matrix_dna = [matrix_dna_data, matrix_dna_index]

    matrix_phred_data = [[] for _ in range(N_data)]  # 先用空列表进行初始化，后面转成0/1可靠度时，再换成0.5
    matrix_phred_index_1 = [[] for _ in range(N_index_1)]
    matrix_phred_index_2 = [[] for _ in range(N_index_2)]
    matrix_phred_index = [matrix_phred_index_1, matrix_phred_index_2]
    matrix_phred = [matrix_phred_data, matrix_phred_index]
    # 处理data部分
    for i in range(len(matrix[0])):
        if not matrix[0][i]:
            continue
        matrix_dna[0][i] = matrix[0][i][0]
        matrix_phred[0][i] = matrix[0][i][1]
    # 处理index部分
    for i in range(len(matrix[1])):
        for j in range(len(matrix[1][i])):
            if not matrix[1][i][j]:
                continue
            matrix_dna[1][i][j] = matrix[1][i][j][0]
            matrix_phred[1][i][j] = matrix[1][i][j][1]
    # 将碱基的可靠度（质量值）转化为0/1bit的可靠度（质量值）
    # 通过计算，碱基的出错率为p，则其对应0/1出错率为2p/3；同理，碱基可靠度为x，则出错率为1-x，则对应0/1出错率为2(1-x)/3，可靠度为1-2(1-x)/3，即(1+2x)/3
    phred_initial_value = [0.5 for _ in range(segment_length)]
    # 这里有一个重要问题:DNA序列长度不等于segment_length时，已经能保证转换的0/1序列长度等于segment_length。
    # 同样，当质量值长度不等于segment_length时，也应该要保证转换的0/1序列长度等于segment_length
    # 处理data部分
    for i in range(len(matrix_phred[0])):
        if not matrix_phred[0][i]:  # 如果对应DNA序列和质量值都丢失了，则用0.5初始化
            matrix_phred[0][i] = phred_initial_value.copy()
        else:
            matrix_phred[0][i] = base_phred_to_01_phred(matrix_phred[0][i])
    # 处理index部分
    for i in range(len(matrix_phred[1])):
        for j in range(len(matrix_phred[1][i])):
            if not matrix_phred[1][i][j]:
                matrix_phred[1][i][j] = phred_initial_value.copy()
            else:
                matrix_phred[1][i][j] = base_phred_to_01_phred(matrix[1][i][j])
    return matrix_dna, matrix_phred


def base_phred_to_01_phred(base_phred_list):
    # 本方法适用于单层方案和双层方案中的第一层计算
    # 通过计算，碱基的出错率为p，则其对应0/1出错率为2p/3；同理，碱基可靠度为x，则出错率为1-x，则对应0/1出错率为2(1-x)/3，可靠度为1-2(1-x)/3，即(1+2x)/3
    # 同时要保证输出的0/1序列的质量值长度等于segment_length
    if len(base_phred_list) >= segment_length:
        handle_length = segment_length
    else:
        handle_length = len(base_phred_list)
    bits_phred = [0.5 for _ in range(segment_length)]
    for i in range(handle_length):
        bits_phred[i] = (1 + 2 * base_phred_list[i]) / 3
    return bits_phred


def base_phred_to_01_phred_second_layer(base_phred_list):
    # 对于双层极化码方案，从碱基可靠度转化为01可靠度的过程中，第一层和第二层计算公式是不同的
    # 第一层：与上面的普通算法一样.
    # 第二层：这里就输出碱基的可靠度，但是做一个处理，保证输出的base_phred_list长度为segment_length.然后，进一步的处理在函数SigReceive2llr_second_layer中
    if len(base_phred_list) >= segment_length:
        handle_length = segment_length
    else:
        handle_length = len(base_phred_list)
    base_phred = [0.25 for _ in range(segment_length)]
    for i in range(handle_length):
        base_phred[i] = base_phred_list[i]
    return base_phred


def matrix_dna2bits_2(matrix_dna, polarParams):  # 在这个新的函数中，将加入对index_DNA矩阵的修改功能，使其能应对indel错误,且实现双层index
    # 添加一个新功能，记录下最后1次解码时接收到的01信息。方便与原正确01矩阵对比SCL解码时的错误率，也实现出错的列以增加list再尝试解码。240828
    # 本函数实现对表格index_2内容的恢复，即index_1的行号匹配情况
    global num_total_sub, num_total_del, num_total_ins

    # x_hat_merge = [[] for _ in range(N_index - 1)]  # -1是因为编码后删掉了第0行
    # 　将DNA矩阵与质量值矩阵分离，以方便原来的代码执行
    # 支持任意切割
    matrix_list_decode_new = [[] for _ in range(segment_length)]  # 这里均以列为思考方向
    matrix_list_decode_flag = [[] for _ in range(segment_length)]
    matrix_list_received_01_column = [[] for _ in range(segment_length)]  # 记录下最后1次解码时接收到的01信息
    matrix_list_received_phred_column = [[] for _ in range(segment_length)]  # 记录下最后1次解码时接收到的01信息的质量值
    for i in range(1, segment_n + 1):
        matrix_index_bits, matrix_random_bits, matrix_index_phred, matrix_random_phred = matrix_dna_2_to_list_index_new(
            matrix_dna[1][1])
        # 对index_2部分数据进行解码
        # 注意：在用crc_8时，第0行是随机的，没有删掉第0行，但用crc_16或crc_4就又要插入
        insert_zeroes = [0 for _ in range(segment_length)]  # 用crc_16或crc_4就又要插入
        matrix_index_bits.insert(0, insert_zeroes)  # 这个插入操作对修改DNA矩阵时，是有影响的
        # 因为一定是0，故100%可靠。但插入1，在求似然比时会出现无穷大，故插入0.999999，方便后面操作
        insert_phred = [0.999999 for _ in range(segment_length)]
        matrix_index_phred.insert(0, insert_phred)  # 把质量值也插入

        matrix_phred_index_2 = np.array(matrix_index_phred, dtype=object)
        matrix_phred_index_2 = matrix_phred_index_2.T
        matrix_index_bits = np.array(matrix_index_bits, dtype=object)
        matrix_index_bits = matrix_index_bits.T
        if i > back_length:
            temp_n_0 = (i - back_length) * step
        else:
            temp_n_0 = 0
        # temp_n_1 = i * step
        temp_n_1 = min(segment_length, i * step)
        x_hat_lists = []
        # 考虑并行
        # results = (
        #     Parallel(n_jobs=threads_number)(
        #         delayed(CASCL_decoder_index_2)(index_array) for index_array in matrix_index_bits[temp_n_0:temp_n_1]))
        results = (
            Parallel(n_jobs=threads_number)(
                delayed(CASCL_decoder_index_2)(matrix_index_bits[j], matrix_phred_index_2[j]) for j in
                range(temp_n_0, temp_n_1)))

        for result in results:
            x_hat_lists.append(result[1])
        x_hat_lists = np.array(x_hat_lists)
        x_hat_lists = x_hat_lists.T
        x_hat_lists = x_hat_lists.tolist()
        x_hat_lists = x_hat_lists[1:]  # 要转成DNA序列，就要去掉第0行
        if i >= back_length:  # 把回头部分的解码结果保存下来
            for j in range(step):
                # matrix_list_decode_new.append(results[j][0])
                matrix_list_decode_new[temp_n_0 + j] = results[j][0]
                matrix_list_decode_flag[temp_n_0 + j] = results[j][2]  # 把该列是否通过CRC检验也记录下来
                matrix_list_received_01_column[temp_n_0 + j] = matrix_index_bits[temp_n_0 + j]
                matrix_list_received_phred_column[temp_n_0 + j] = matrix_phred_index_2[temp_n_0 + j]
        # 开始检查并修改接收到的DNA矩阵中的错误
        for j in range(len(x_hat_lists)):
            # 怎么把解码的部分01bit给找出来，利用x_hat_merge
            # 要是随机bits(matrix_random_bits)错了呢，这样恢复的DNA序列就不对了，这显然也会影响对原DNA矩阵的修改成功率
            # 所以，从这个角度来讲，我们也不能处理太高的错误情况。不然，最后一个index矩阵会解码出错，导致前面的data部分也不能正确恢复
            dna_sequence_decode = DNA_encode.two_lists_to_sequence(x_hat_lists[j],
                                                                   matrix_random_bits[j][temp_n_0:temp_n_1])
            # print("dna_sequence_decode:", dna_sequence_decode, len(dna_sequence_decode))
            # print("matrix_dna[1][j][:temp_n]:", matrix_dna[1][j][:temp_n], len(matrix_dna[1][j][:temp_n]))
            # if not matrix_dna[1][1][j]:  # 如果这条DNA序列刚好丢失会怎么样? 答：没有影响，下面matrix_dna[1][1][j][temp_n_0:temp_n_1]也会取出一个空列表
            # 关于质量值的修改，也许可以这样做：
            # 如果dna_sequence_decode等于接收到（或修改后）的DNA序列，可以给这些对应位置的质量值一个高可靠度（eg:0.999）;
            # 如果是修改后的碱基，替换错误，给到0.99，插入与删除错误，则对质量值矩阵做对应操作，使质量值与DNA序列保持对齐
            if dna_sequence_decode == matrix_dna[1][1][j][0][temp_n_0:temp_n_1]:
                # pass
                for k in range(temp_n_0, temp_n_1):
                    matrix_dna[1][1][j][1][k] = 0.9998  # 得修改原DNA序列中的质量值才行
            else:  # 不相等时，才需要修改
                edit_distance = Levenshtein.distance(dna_sequence_decode, matrix_dna[1][1][j][0][temp_n_0:temp_n_1])
                if edit_distance > 6:
                    pass
                    # print("edit_distance太大了！在index的修改中。j = %d, edit_distance = %d" % (j, edit_distance))
                    # print("dna_sequence_decode:                   ", dna_sequence_decode, len(dna_sequence_decode))
                    # print("matrix_dna[1][1][j][0][temp_n_0:temp_n_1]:", matrix_dna[1][1][j][0][temp_n_0:temp_n_1],
                    #       len(matrix_dna[1][1][j][0][temp_n_0:temp_n_1]))
                num_sub, num_del, num_ins, dna_sequence_modify = check_and_edit_dna_sequence(dna_sequence_decode,
                                                                                             matrix_dna[1][1][j],
                                                                                             temp_n_0)
                num_total_sub += num_sub
                num_total_del += num_del
                num_total_ins += num_ins
                matrix_dna[1][1][j] = dna_sequence_modify
    # 现在只用最后2个step的解码结果,看看能否再减少解码用时。答:可以减少解码时间。
    matrix_index_bits, _, matrix_index_phred, _ = matrix_dna_2_to_list_index_new(matrix_dna[1][1])
    insert_zeroes = [0 for _ in range(segment_length)]  # 用crc_16就又要插入
    matrix_index_bits.insert(0, insert_zeroes)  # 这个插入操作对修改DNA矩阵时，是有影响的
    matrix_index_bits = np.array(matrix_index_bits)  # 就算是多层List,也会全部变成np.array
    matrix_index_bits = matrix_index_bits.T
    # 因为一定是0，故100%可靠。但插入1，在求似然比时会出现无穷大，故插入0.999999，方便后面操作
    insert_phred = [0.999999 for _ in range(segment_length)]
    matrix_index_phred.insert(0, insert_phred)  # 把质量值也插入

    matrix_phred_index_2 = np.array(matrix_index_phred, dtype=object)
    matrix_phred_index_2 = matrix_phred_index_2.T

    # results = (
    #     Parallel(n_jobs=threads_number)(
    #         delayed(CASCL_decoder_index_2)(index_array) for index_array in matrix_index_bits[-2 * step:]))
    results = (
        Parallel(n_jobs=threads_number)(
            delayed(CASCL_decoder_index_2)(matrix_index_bits[j], matrix_phred_index_2[j]) for j in
            range(-2 * step, 0)))
    # for result in results:
    #     matrix_list_decode_new.append(result[0])
    # 所以最后一个step是解码了2次，其他step是解码了3次
    temp_n_0 = -2 * step
    for j in range(len(results)):
        matrix_list_decode_new[temp_n_0 + j] = results[j][0]
        matrix_list_decode_flag[temp_n_0 + j] = results[j][2]  # 把该列是否通过CRC检验也记录下来
        matrix_list_received_01_column[temp_n_0 + j] = matrix_index_bits[temp_n_0 + j]  # 我们要注意到这里matrix_index_bits 和 matrix_phred_index_2都是完整矩阵
        matrix_list_received_phred_column[temp_n_0 + j] = matrix_phred_index_2[temp_n_0 + j]

    # 尝试对其中没有通过CRC检验的列，以增加list的方式，再进行SCL解码1次
    CASCL_decoder_name = "CASCL_decoder_index_2"
    matrix_list_decode_new, matrix_list_decode_flag = CASCL2failCRCwithlargerlist(CASCL_decoder_name, matrix_list_decode_new,
                                                         matrix_list_decode_flag, matrix_list_received_01_column,
                                                         matrix_list_received_phred_column, _, _, polarParams)
    # 处理最后的解码结果
    matrix_list_decode_new = np.array(matrix_list_decode_new)
    matrix_list_decode_new = matrix_list_decode_new.T
    matrix_list_decode = matrix_list_decode_new.tolist()

    # 把matrix_list_encode_column也进行转置
    matrix_list_received_01_column = np.array(matrix_list_received_01_column)
    matrix_list_received_01_column = matrix_list_received_01_column.T
    matrix_list_received_01 = matrix_list_received_01_column.tolist()

    # 去掉每行后面添加的随机数据
    matrix_index_lists = []
    for list_decode in matrix_list_decode:
        index_list = list_decode[:segment_length_index_2 + parity_length]  # +1是为了把最后的奇偶检验位也加进来
        matrix_index_lists.append(index_list)
    # 进行奇偶校验，并去掉最后的奇偶检验位
    matrix_index_lists = handle_parity(matrix_index_lists, matrix_list_decode_flag)
    # 恢复表格index_2，即index_1中行号匹配情况
    index_numbers_2 = []
    two_index_binary_length = 2 * index_binary_length_2
    index_numbers_count_binary = matrix_index_lists[0][:two_index_binary_length]
    # print("index_numbers_count_binary:", index_numbers_count_binary, len(index_numbers_count_binary), type(index_numbers_count_binary))

    if index_numbers_count_binary[:index_binary_length_2] != index_numbers_count_binary[index_binary_length_2:]:
        print("记录index_2表格的行号对数，不一致！")
        # print("index_numbers_count_binary[:index_binary_length_2]:", index_numbers_count_binary[:index_binary_length_2])
        # print("index_numbers_count_binary[index_binary_length_2:]:", index_numbers_count_binary[index_binary_length_2:])
        print("采用投票算法，从4个行号对数中确定正确行号！")
        index_count_list_1 = matrix_index_lists[0][:index_binary_length_2]
        index_count_list_2 = matrix_index_lists[0][index_binary_length_2:two_index_binary_length]
        index_count_list_3 = matrix_index_lists[0][
                             two_index_binary_length:two_index_binary_length + index_binary_length_2]
        index_count_list_4 = matrix_index_lists[0][
                             two_index_binary_length + index_binary_length_2:2 * two_index_binary_length]

        index_count_lists = [index_count_list_1,
                             index_count_list_2,
                             index_count_list_3,
                             index_count_list_4]
        print("index_count_lists =", index_count_lists)

        index_count_binary = vote(index_count_list_1, index_count_list_2, index_count_list_3, index_count_list_4)
    else:
        # print("一切正常！记录index_2表格的行号对数是一致的！")   # 增加index部分的frozen_bits后，这里通常没有问题了
        index_count_binary = index_numbers_count_binary[:index_binary_length_2]
    index_numbers_count_string = ''.join([str(bit) for bit in index_count_binary])
    index_numbers_count = int(index_numbers_count_string, 2)  # 将二进制转换成整数
    matrix_index_lists[0] = matrix_index_lists[0][2 * two_index_binary_length:]  # 处理掉第0行中，存行号对数的数据, 支持存2对
    count = 0
    flag = True
    for list_index in matrix_index_lists:
        for i in range(0, len(list_index), two_index_binary_length):
            index_up = list_index[i: i + index_binary_length_2]
            index_down = list_index[i + index_binary_length_2: i + two_index_binary_length]

            index_up_int = int(''.join([str(bit) for bit in index_up]), 2)
            index_down_int = int(''.join([str(bit) for bit in index_down]), 2)

            index_numbers_2.append([index_up_int, index_down_int])
            count += 1
            if count == index_numbers_count:
                flag = False
                break
        if not flag:  # 已经恢复完index的表格
            break
    return index_numbers_2, matrix_list_received_01


def vote(*lists):
    num_lists = len(lists)
    result = []
    for i in range(len(lists[0])):
        # Count occurrences of 0s and 1s at position i across all lists
        counts = Counter([lst[i] for lst in lists])
        # Find the most common number (mode)
        mode = counts.most_common(1)[0][0]
        result.append(mode)
    return result


def matrix_dna2bits_1(matrix_dna, index_numbers_2, polarParams):
    # 本函数实现对表格index_1内容的恢复
    # 添加一个新功能，记录下最后1次解码时接收到的01信息。方便与原正确01矩阵对比SCL解码时的错误率，也实现出错的列以增加list再尝试解码。240828
    global num_total_sub, num_total_del, num_total_ins
    # x_hat_merge = [[] for _ in range(N_index - 1)]  # -1是因为编码后删掉了第0行
    # matrix_list_decode_new = []
    # 支持任意切割
    matrix_list_decode_new = [[] for _ in range(segment_length)]
    matrix_list_decode_flag = [[] for _ in range(segment_length)]
    matrix_list_received_01_column = [[] for _ in range(segment_length)]  # 记录下最后1次解码时接收到的01信息
    matrix_list_received_phred_column = [[] for _ in range(segment_length)]  # 记录下最后1次解码时接收到的01信息的质量值
    for i in range(1, segment_n + 1):
        bits_received_index_1, random_bits_index_1, matrix_phred_index_1, random_phred_index_1 = matrix_dna_index_1_to_bits(
            matrix_dna[1][0], index_numbers_2)
        # 对index_2部分数据进行解码
        # 注意：在用crc_8时，第0行是随机的，没有删掉第0行，但用crc_16或crc_4就又要插入
        # 上面matrix_dna_index_1_to_bits函数已经插入了第0行，这里不用再插入
        matrix_index_bits = np.array(bits_received_index_1, dtype=object)  # 就算是多层List,也会全部变成np.array
        matrix_index_bits = matrix_index_bits.T

        matrix_phred_index_1 = np.array(matrix_phred_index_1, dtype=object)
        matrix_phred_index_1 = matrix_phred_index_1.T
        if i > back_length:
            temp_n_0 = (i - back_length) * step
        else:
            temp_n_0 = 0
        # temp_n_1 = i * step
        temp_n_1 = min(segment_length, i * step)
        x_hat_lists = []
        # 考虑并行
        # results = (
        #     Parallel(n_jobs=threads_number)(
        #         delayed(CASCL_decoder_index_1)(index_array) for index_array in matrix_index_bits[temp_n_0:temp_n_1]))
        results = (
            Parallel(n_jobs=threads_number)(
                delayed(CASCL_decoder_index_1)(matrix_index_bits[j], matrix_phred_index_1[j]) for j in
                range(temp_n_0, temp_n_1)))
        for result in results:
            x_hat_lists.append(result[1])
        x_hat_lists = np.array(x_hat_lists)
        x_hat_lists = x_hat_lists.T
        x_hat_lists = x_hat_lists.tolist()
        x_hat_lists = x_hat_lists[1:]  # 要转成DNA序列，就要去掉第0行

        if i >= back_length:  # 把回头部分的解码结果保存下来
            for j in range(step):
                # matrix_list_decode_new.append(results[j][0])
                matrix_list_decode_new[temp_n_0 + j] = results[j][0]
                matrix_list_decode_flag[temp_n_0 + j] = results[j][2]
                matrix_list_received_01_column[temp_n_0 + j] = matrix_index_bits[temp_n_0 + j]
                matrix_list_received_phred_column[temp_n_0 + j] = matrix_phred_index_1[temp_n_0 + j]
        # 开始检查并修改接收到的DNA矩阵中的错误
        for j in range(len(index_numbers_2)):
            # 怎么把解码的部分01bit给找出来，利用x_hat_merge
            up_id, down_id = index_numbers_2[j][0], index_numbers_2[j][1]
            if up_id < len(x_hat_lists):
                up_list = x_hat_lists[up_id]
            else:  # 为生成的随机bits
                up_list = random_bits_index_1[up_id][temp_n_0:temp_n_1]

            if down_id < len(x_hat_lists):
                down_list = x_hat_lists[down_id]
            else:  # 为生成的随机bits
                # 下面一定要取random_bits_index_1[down_id]的[temp_n_0:temp_n_1]部分出来才行
                down_list = random_bits_index_1[down_id][temp_n_0:temp_n_1]
            dna_sequence_decode = DNA_encode.two_lists_to_sequence(up_list, down_list)
            if dna_sequence_decode == matrix_dna[1][0][j][0][temp_n_0:temp_n_1]:
                # pass   # 考虑到DNA序列会有解码失败的情况，还是相信deepleaning的质量值输出。试试不在解码时修改质量值
                for k in range(temp_n_0, temp_n_1):
                    matrix_dna[1][0][j][1][k] = 0.9998  # 得修改原DNA序列中的质量值才行
            else:  # 不相等时，才需要修改
                edit_distance = Levenshtein.distance(dna_sequence_decode, matrix_dna[1][0][j][0][temp_n_0:temp_n_1])
                if edit_distance > 6:
                    pass
                    # print("edit_distance太大了！在index_1的修改中。j = %d, edit_distance = %d" % (j, edit_distance))
                    # print("dna_sequence_decode:                   ", dna_sequence_decode, len(dna_sequence_decode))
                    # print("matrix_dna[1][0][j][0][temp_n_0:temp_n_1]:", matrix_dna[1][0][j][0][temp_n_0:temp_n_1],
                    #       len(matrix_dna[1][0][j][0][temp_n_0:temp_n_1]))
                num_sub, num_del, num_ins, dna_sequence_modify = check_and_edit_dna_sequence(dna_sequence_decode,
                                                                                             matrix_dna[1][0][j],
                                                                                             temp_n_0)
                num_total_sub += num_sub
                num_total_del += num_del
                num_total_ins += num_ins
                matrix_dna[1][0][j] = dna_sequence_modify
    # 现在只用最后2个step的解码结果,看看能否再减少解码用时
    bits_received_index_1, _, matrix_phred_index_1, _ = matrix_dna_index_1_to_bits(matrix_dna[1][0], index_numbers_2)
    # 上面的函数已经插入了第0行
    matrix_index_bits = np.array(bits_received_index_1)  # 就算是多层List,也会全部变成np.array
    matrix_index_bits = matrix_index_bits.T
    matrix_phred_index_1 = np.array(matrix_phred_index_1, dtype=object)
    matrix_phred_index_1 = matrix_phred_index_1.T
    results = (
        Parallel(n_jobs=threads_number)(
            delayed(CASCL_decoder_index_1)(matrix_index_bits[j], matrix_phred_index_1[j]) for j in
            range(-2 * step, 0)))
    # for result in results:
    #     matrix_list_decode_new.append(result[0])
    temp_n_0 = -2 * step
    for j in range(len(results)):
        matrix_list_decode_new[temp_n_0 + j] = results[j][0]
        matrix_list_decode_flag[temp_n_0 + j] = results[j][2]
        matrix_list_received_01_column[temp_n_0 + j] = matrix_index_bits[temp_n_0 + j]
        matrix_list_received_phred_column[temp_n_0 + j] = matrix_phred_index_1[temp_n_0 + j]
    # 尝试对其中没有通过CRC检验的列，以增加list的方式，再进行SCL解码1次
    CASCL_decoder_name = "CASCL_decoder_index_1"
    matrix_list_decode_new, matrix_list_decode_flag = CASCL2failCRCwithlargerlist(CASCL_decoder_name, matrix_list_decode_new,
                                                         matrix_list_decode_flag, matrix_list_received_01_column,
                                                         matrix_list_received_phred_column, _, _, polarParams)

    # 处理最后的解码结果
    matrix_list_decode_new = np.array(matrix_list_decode_new)
    matrix_list_decode_new = matrix_list_decode_new.T
    matrix_list_decode = matrix_list_decode_new.tolist()

    # 把matrix_list_encode_column也进行转置
    matrix_list_received_01_column = np.array(matrix_list_received_01_column)
    matrix_list_received_01_column = matrix_list_received_01_column.T
    matrix_list_received_01 = matrix_list_received_01_column.tolist()

    # 去掉每行后面添加的随机数据
    matrix_index_lists = []
    for list_decode in matrix_list_decode:
        index_list = list_decode[:segment_length_index_1 + parity_length]  # +parity_length 把奇偶检验位也加进来
        matrix_index_lists.append(index_list)
    # 进行奇偶校验，并去掉最后的奇偶检验位
    matrix_index_lists = handle_parity(matrix_index_lists, matrix_list_decode_flag)
    # 恢复表格index_1，即data中行号匹配情况
    index_numbers_1 = []
    two_index_binary_length = 2 * index_binary_length_1
    index_numbers_count_binary = matrix_index_lists[0][:two_index_binary_length]
    # print("index_numbers_count_binary:", index_numbers_count_binary, len(index_numbers_count_binary), type(index_numbers_count_binary))

    if index_numbers_count_binary[:index_binary_length_1] != index_numbers_count_binary[index_binary_length_1:]:
        print("记录index_1表格的行号对数，不一致！")
        print("采用投票算法，从4个行号对数中确定正确行号！")
        index_count_list_1 = matrix_index_lists[0][:index_binary_length_1]
        index_count_list_2 = matrix_index_lists[0][index_binary_length_1:two_index_binary_length]
        index_count_list_3 = matrix_index_lists[0][
                             two_index_binary_length:two_index_binary_length + index_binary_length_1]
        index_count_list_4 = matrix_index_lists[0][
                             two_index_binary_length + index_binary_length_1:2 * two_index_binary_length]

        index_count_lists = [index_count_list_1,
                             index_count_list_2,
                             index_count_list_3,
                             index_count_list_4]
        print("index_count_lists =", index_count_lists)
        index_count_binary = vote(index_count_list_1, index_count_list_2, index_count_list_3, index_count_list_4)
    else:
        # print("一切正常！记录index_1表格的行号对数是一致的！")   # 增加index部分的frozen_bits后，这里通常没有问题了
        index_count_binary = index_numbers_count_binary[:index_binary_length_1]
    index_numbers_count_string = ''.join([str(bit) for bit in index_count_binary])
    index_numbers_count = int(index_numbers_count_string, 2)  # 将二进制转换成整数
    matrix_index_lists[0] = matrix_index_lists[0][2 * two_index_binary_length:]  # 处理掉第0行中，存行号对数的数据, 支持存2对
    count = 0
    flag = True
    for list_index in matrix_index_lists:
        for i in range(0, len(list_index), two_index_binary_length):
            index_up = list_index[i: i + index_binary_length_1]
            index_down = list_index[i + index_binary_length_1: i + two_index_binary_length]
            index_up_int = int(''.join([str(bit) for bit in index_up]), 2)
            index_down_int = int(''.join([str(bit) for bit in index_down]), 2)
            index_numbers_1.append([index_up_int, index_down_int])
            count += 1
            if count == index_numbers_count:
                flag = False
                break
        if not flag:  # 已经恢复完index_1的表格
            break
    return index_numbers_1, matrix_list_received_01


'''
def matrix_dna2bits(matrix_dna):
    # 先处理index部分
    matrix_index_bits = matrix_dna2list_index(matrix_dna[1])
    # 对index部分数据进行解码
    # 在用crc_8时，第0行是随机的，没有删掉第0行，所以这里不用插入
    insert_zeroes = [0 for _ in range(segment_length)]  # 用crc_16就又要插入
    matrix_index_bits.insert(0, insert_zeroes)
    matrix_index_bits = np.array(matrix_index_bits)  # 就算是多层List,也会全部变成np.array
    matrix_index_bits = matrix_index_bits.T

    # matrix_list_decode = []
    # for index_array in matrix_index_bits:
    #     llr = Polar_decode.SigReceive2llr_single_layer(index_array)
    #     array_decode, _ = Polar_decode.CASCL_decoder(llr, select_index_index, freeze_index_index,
    #                                                  frozen_bits_index, List, crc_index)
    #     matrix_list_decode.append(array_decode)
    # 并行
    matrix_list_decode = (
        Parallel(n_jobs=10)(delayed(CASCL_decoder_index)(index_array) for index_array in matrix_index_bits))
    matrix_list_decode = np.array(matrix_list_decode)
    matrix_list_decode = matrix_list_decode.T
    matrix_list_decode = matrix_list_decode.tolist()

    # 去掉每行后面添加的随机数据
    matrix_index_lists = []
    for list_decode in matrix_list_decode:
        index_list = list_decode[:segment_length_index]
        matrix_index_lists.append(index_list)
    # 恢复表格
    index_numbers = []
    two_index_binary_length = 2 * index_binary_length
    index_numbers_count_binary = matrix_index_lists[0][:two_index_binary_length]
    # print("index_numbers_count_binary:", index_numbers_count_binary, len(index_numbers_count_binary), type(index_numbers_count_binary))

    if index_numbers_count_binary[:index_binary_length] != index_numbers_count_binary[index_binary_length:]:
        print("出错啦！！记录index表格的行号对数，不一致！")
        print("index_numbers_count_binary[:index_binary_length]:", index_numbers_count_binary[:index_binary_length])
        print("index_numbers_count_binary[index_binary_length:]:", index_numbers_count_binary[index_binary_length:])
    else:
        # print("一切正常！记录index表格的行号对数是一致的！")   # 增加index部分的frozen_bits后，这里通常没有问题了
        pass
    index_numbers_count_string = ''.join([str(bit) for bit in index_numbers_count_binary[:index_binary_length]])
    index_numbers_count = int(index_numbers_count_string, 2)  # 将二进制转换成整数
    matrix_index_lists[0] = matrix_index_lists[0][two_index_binary_length:]  # 处理掉第0行中，存行号对数的数据
    count = 0
    flag = True
    for list_index in matrix_index_lists:
        for i in range(0, len(list_index), two_index_binary_length):
            index_up = list_index[i: i + index_binary_length]
            index_down = list_index[i + index_binary_length: i + two_index_binary_length]
            index_up_int = int(''.join([str(bit) for bit in index_up]), 2)
            index_down_int = int(''.join([str(bit) for bit in index_down]), 2)
            index_numbers.append([index_up_int, index_down_int])
            count += 1
            if count == index_numbers_count:
                flag = False
                break
        if not flag:  # 已经恢复完index的表格
            break
    # 开始根据表格内容，去处理数据部分的dna序列
    # if index_numbers_count  < len(matrix_dna[0]):  # 可能这是正常的，因为有了预留后，恢复的matrix_dna[0]均有1100行。当然，可以考虑去掉里面的空[]
    #     print('出错了，index表格中行号对数小于数据部分DNA序列的行数')
    # elif index_numbers_count > len(matrix_dna[0]):
    #     print('出错了，index表格中行号对数大于数据部分DNA序列的行数')
    # else:
    #     print('正确了，index表格中行号对数等于数据部分DNA序列的行数')
    matrix_bits_one = []
    for _ in range(N_data - 1):  # -1是因为编码时删掉了第0行
        zeroes_list = [0 for _ in range(segment_length)]
        matrix_bits_one.append(zeroes_list)
    matrix_bits_all = []
    for _ in range(2):  # 二层方案中，matrix_bits中有两个数据矩阵matrix_bits_all[0]和matrix_bits_all[1]
        matrix_bits_temp = copy.deepcopy(matrix_bits_one)
        matrix_bits_all.append(matrix_bits_temp)
    for i in range(index_numbers_count):
        up_idx, down_idx = index_numbers[i][0], index_numbers[i][1]
        # 这里要考虑一个问题，某条DNA序列丢失，这的位置上是一个空列表；还有多的（预留的）空列表，会有什么影响
        # 答：我希望如果是某条DNA序列丢失，这个空列表，其对应bit位置上是zeroes_list。而预留的空列表，不要操作，直接跳过。
        if not matrix_dna[0][i]:
            continue
        up_list, down_list = dna2list(matrix_dna[0][i])
        # 这里还要检查行号是否合法，即对应数据是否是随机数据
        if up_idx < len(matrix_bits_one):
            matrix_bits_all[0][up_idx] = up_list
        if down_idx < len(matrix_bits_one):
            matrix_bits_all[1][down_idx] = down_list
    # 因为在编码时（竖向极化后）有删掉第0行（其值均为0），又因为对应行号也会发生变化，所以不能靠初始化0，还得插入0行
    zeroes_list = [0 for _ in range(segment_length)]
    matrix_bits_all[0].insert(0, zeroes_list.copy())  # 对列表的复制一定要特别小心，它与数组可不同
    matrix_bits_all[1].insert(0, zeroes_list.copy())
    return matrix_bits_all  # matrix_bits表示DNA序列的数据部分，恢复成0/1bits
'''


def matrix_dna2list_index(matrix_dna):
    matrix_lists = []
    for dna_seq in matrix_dna:
        list_up, list_down = dna2list(dna_seq)
        # 在处理index部分时，目前的方案（23-12-19）还是简单方案，list_down一定是随机数据
        matrix_lists.append(list_up)
    return matrix_lists


def matrix_dna_2_to_list_index_new(matrix_dna, last_case='last_3'):  # 还应该加入对某条DNA序列丢失的鲁棒性
    # 要在这里将DNA序列转化成01序列，同时，就分离质量值，并转化成对应01序列的质量值，因为要满足质量值与01序列相对应
    # 还应该加入对某条DNA序列丢失的鲁棒性，这里要注意到这个index矩阵，其后面的一些空列表，是不应该转化的01的
    # 这里有bug，如果index_2的DNA序列丢掉了1条，会导致解码时llr的长度为63，这是会报错的
    # 解决方案：一定要确保matrix_lists_index, matrix_lists_random中有63条01list，因为后面还会插入第0行
    global num_total_mis_dna_seqs
    if last_case == 'last_3':
        N = N_index_2
    elif last_case == 'last_2':
        N = N_index_1_last_2
    else:  # last_1
        N = N_index_1_last_1
    matrix_lists_index, matrix_lists_random = [], []
    matrix_index_phred, matrix_random_phred = [], []
    for _ in range(N - 1):
        zero_list = [0 for _ in range(segment_length)]
        matrix_lists_index.append(zero_list.copy())
        matrix_lists_random.append(zero_list.copy())
        phred_initial_value = [0.5 for _ in range(segment_length)]
        matrix_index_phred.append(phred_initial_value.copy())
        matrix_random_phred.append(phred_initial_value.copy())
    for i in range(len(matrix_dna)):
        if not matrix_dna[i][0]:
            num_total_mis_dna_seqs += 1
            continue  # 空列表就跳过，用默认值
        list_up, list_down = dna2list(matrix_dna[i][0])  # 我看这个dna2list函数，如果dna_seq是空列表，也会返回全0bits
        # 这样的话，后面的多余空列表，应该会导致错误啊，index_2不会，因为其后面没有空列表，但index_1(双层方案)和data会
        # 答：在index_1（双层index方案）中，其都是用N_index_1极化的，故用其来初始化，就能满足要求。data也一样。
        # 在处理index部分时，目前的方案（23-12-19）还是简单方案，list_down一定是随机数据
        matrix_lists_index[i] = list_up
        matrix_lists_random[i] = list_down
        matrix_index_phred[i] = base_phred_to_01_phred(matrix_dna[i][1])
        matrix_random_phred[i] = matrix_index_phred[i].copy()
    return matrix_lists_index, matrix_lists_random, matrix_index_phred, matrix_random_phred


def dna2list(dna_sequence):
    # list_up = [0 for _ in range(len(dna_sequence))]
    # list_down = [0 for _ in range(len(dna_sequence))]
    # 当dna_seq的长度长于segment_length时，也应该只将前面的segment_length长度的DNA序列转化为0/1序列，想想我们的竖向解码方案，多余的部分没有意义，也用不上
    # 另一方面，根据实验，如果一个01矩阵中各list的长度不同，是不能对其正常的，先转np.array，再用.T转置的
    # 这里的list_up和list_down长度必须等于segment_length，不然在进行CA-SCL解码时，会出问题
    list_up = [0 for _ in range(segment_length)]
    list_down = [0 for _ in range(segment_length)]
    if dna_sequence:  # 当dna_sequence为不为空时，才做下面的操作
        if len(dna_sequence) < segment_length:
            handle_num = len(dna_sequence)
        else:
            handle_num = segment_length
        for i in range(handle_num):
            # 更新01到碱基的转换规则：考虑到有随机bits，它可能在上面也可能在下面。
            # 好的规则应该是这样的，当上面固定为数据，下面为随机bits时，上面为0或1，都有可能转化为GC或非GC。当下面固定为数据时，上面为随机bits，则是下面为0或1，都有可能转化为GC或非GC。
            # 故新的转换规则如下：00>C,01>A,10>T,11>G 240418
            if dna_sequence[i] == 'G':
                list_up[i] = 1
                list_down[i] = 1
            elif dna_sequence[i] == 'T':
                list_up[i] = 1
            elif dna_sequence[i] == 'A':
                list_down[i] = 1
            # 当dna_sequence[i]=='C'时，不用处理
    return list_up, list_down


def CASCL_decoder_first_layer(y_1, y_1_phred, polarParams, last_case, my_list=list_16):
    llr = Polar_decode.SigReceive2llr_first_layer(y_1, y_1_phred)
    if last_case == 'last_3':
        sig_recover, x_hat, crc_flag = Polar_decode.CASCL_decoder(N_data, llr, polarParams.select_index_data_1, polarParams.freeze_index_data_1,
                                                                  polarParams.frozen_bits_data_1, my_list, crc_data)
    elif last_case == 'last_2':
        sig_recover, x_hat, crc_flag = Polar_decode.CASCL_decoder(N_data_last_2, llr, select_index_data_1_last_2,
                                                                  freeze_index_data_1_last_2,
                                                                  frozen_bits_data_1_last_2, my_list, crc_data_last_2)
    else:  # last_case == 'last_1':
        sig_recover, x_hat, crc_flag = Polar_decode.CASCL_decoder(N_data_last_1, llr, select_index_data_1_last_1,
                                                                  freeze_index_data_1_last_1,
                                                                  frozen_bits_data_1_last_1, my_list, crc_data_last_1)
    return sig_recover, x_hat, crc_flag


def CASCL_decoder_second_layer(x_hat, y_1, y_2, y_2_bases_phred, polarParams, last_case, my_list=list_16):
    # 支持质量值
    llr = Polar_decode.SigReceive2llr_second_layer(x_hat, y_1, y_2, y_2_bases_phred)
    if last_case == 'last_3':
        sig_recover, x_hat, crc_flag = Polar_decode.CASCL_decoder(N_data, llr, polarParams.select_index_data_2, polarParams.freeze_index_data_2,
                                                                  polarParams.frozen_bits_data_2, my_list, crc_data)
    elif last_case == 'last_2':
        sig_recover, x_hat, crc_flag = Polar_decode.CASCL_decoder(N_data_last_2, llr, select_index_data_2_last_2,
                                                                  freeze_index_data_2_last_2,
                                                                  frozen_bits_data_2_last_2, my_list, crc_data_last_2)
    else:  # last_1
        sig_recover, x_hat, crc_flag = Polar_decode.CASCL_decoder(N_data_last_1, llr, select_index_data_2_last_1,
                                                                  freeze_index_data_2_last_1,
                                                                  frozen_bits_data_2_last_1, my_list, crc_data_last_1)
    return sig_recover, x_hat, crc_flag


def matrix_dna_index_1_to_bits(matrix_dna_index_1, index_numbers_2):
    # 本函数要实现dna_index_1恢复成解码前的01bits,支持质量值
    global num_total_mis_dna_seqs
    matrix_bits_index_1, matrix_phred_index_1 = [], []  # 存放index_1的真实数据，和对应的质量值
    random_bits_index_1, random_phred_index_1 = [], []  # 存放与index_1的真实数据匹配而生成的随机bits，和对应的质量值
    # 对于双层index方案的index_1的恢复，还要把添加的随机bits也恢复好，因为在修改dna_index_1的矩阵中要用到
    zeroes_list = [0 for _ in range(segment_length)]
    phred_initial_value = [0.5 for _ in range(segment_length)]
    for _ in range(N_index_1 - 1):  # -1是因为编码时删掉了第0行
        matrix_bits_index_1.append(zeroes_list.copy())
        matrix_phred_index_1.append(phred_initial_value.copy())
    for _ in range(int(math.pow(2, index_binary_length_2))):
        random_bits_index_1.append(zeroes_list.copy())
        random_phred_index_1.append(phred_initial_value.copy())
    for i in range(len(index_numbers_2)):
        up_idx, down_idx = index_numbers_2[i][0], index_numbers_2[i][1]
        # 这里要考虑一个问题，某条DNA序列丢失，这的位置上是一个空列表；还有多的（预留的）空列表，会有什么影响
        # 答：我希望如果是某条DNA序列丢失，这个空列表，其对应bit位置上是zeroes_list。而预留的空列表，不要操作，直接跳过。
        if not matrix_dna_index_1[i][0]:  # 如果对应的DNA序列丢失了，就不处理，让其全部为默认值
            num_total_mis_dna_seqs += 1
            continue
        # 检查一下这条DNA序列的长度还是有必要的，因为长度不等于segment_length的dna，会转化成长度不同的0、1序列
        # 但是，也不对啊。我们就是要处理index错误啊。所以出现了长度不同的DNA序列，又该如何做呢
        # if len(matrix_dna_data[i]) != segment_length:
        #     print("出错了！")
        up_list, down_list = dna2list(matrix_dna_index_1[i][0])
        phred_list = base_phred_to_01_phred(matrix_dna_index_1[i][1])
        # 这里还要检查行号是否合法，即对应数据是否是随机数据
        if up_idx < len(matrix_bits_index_1):
            matrix_bits_index_1[up_idx] = up_list
            matrix_phred_index_1[up_idx] = phred_list.copy()
        else:  # 为随机数据
            random_bits_index_1[up_idx] = up_list
            random_phred_index_1[up_idx] = phred_list.copy()

        if down_idx < len(matrix_bits_index_1):
            matrix_bits_index_1[down_idx] = down_list
            matrix_phred_index_1[down_idx] = phred_list.copy()
        else:  # 为随机数据
            random_bits_index_1[down_idx] = down_list
            random_phred_index_1[down_idx] = phred_list.copy()
    # 因为在编码时（竖向极化后）有删掉第0行（其值均为0），又因为对应行号也会发生变化，所以不能靠初始化0，还得插入0行
    zeroes_list = [0 for _ in range(segment_length)]
    matrix_bits_index_1.insert(0, zeroes_list)  # 对列表的复制一定要特别小心，它与数组可不同
    phred_insert = [0.999999 for _ in range(segment_length)]
    matrix_phred_index_1.insert(0, phred_insert)  # 一样对，只对原数据矩阵进行插入操作
    return matrix_bits_index_1, random_bits_index_1, matrix_phred_index_1, random_phred_index_1


def matrix_dna_data2bits(matrix_dna_data, index_numbers, last_case):
    # 支持质量值
    # 添加新功能，支持修改含随机bits的DNA序列
    global num_total_mis_dna_seqs
    matrix_bits_one, matrix_phred_one = [], []
    if last_case == 'last_1':
        N = N_data_last_1
    elif last_case == 'last_2':
        N = N_data_last_2
    else:  # last_3
        N = N_data
    zeroes_list = [0 for _ in range(segment_length)]
    phred_initial_value = [0.5 for _ in range(segment_length)]
    for _ in range(N - 1):  # -1是因为编码时删掉了第0行
        matrix_bits_one.append(zeroes_list.copy())
        matrix_phred_one.append(phred_initial_value.copy())
    matrix_bits_all, matrix_phred_all = [], []
    for _ in range(2):  # 二层方案中，matrix_bits中有两个数据矩阵matrix_bits_all[0]和matrix_bits_all[1]
        matrix_bits_temp = copy.deepcopy(matrix_bits_one)
        matrix_bits_all.append(matrix_bits_temp)
        matrix_phred_temp = copy.deepcopy(matrix_phred_one)
        matrix_phred_all.append(matrix_phred_temp)
    random_dic_1, random_dic_2 = {}, {}
    # 看起来随机数据的质量值并不需要保存，用不着
    for i in range(len(index_numbers)):
        up_idx, down_idx = index_numbers[i][0], index_numbers[i][1]
        # 这里要考虑一个问题，某条DNA序列丢失，这的位置上是一个空列表；还有多的（预留的）空列表，会有什么影响
        # 答：我希望如果是某条DNA序列丢失，这个空列表，其对应bit位置上是zeroes_list。而预留的空列表，不要操作，直接跳过。
        if not matrix_dna_data[i][0]:
            num_total_mis_dna_seqs += 1
            continue
        # 检查一下这条DNA序列的长度还是有必要的，因为长度不等于segment_length的dna，会转化成长度不同的0、1序列
        # 但是，也不对啊。我们就是要处理index错误啊。所以出现了长度不同的DNA序列，又该如何做呢
        # if len(matrix_dna_data[i]) != segment_length:
        #     print("出错了！")
        up_list, down_list = dna2list(matrix_dna_data[i][0])
        phred_bits_list = base_phred_to_01_phred(matrix_dna_data[i][1])  # 第一层的计算公式与普通情况相同
        phred_bases_list = base_phred_to_01_phred_second_layer(matrix_dna_data[i][1])  # 第二层的计算公式是不一样的
        # 以字典的形式保存随机bits的数据，以节约内存开销。random_dic_1为与matrix_1匹配的随机bits，random_dic_2为与matrix_2匹配的随机bits
        # 这里还要检查行号是否合法，即对应数据是否是随机数据
        if up_idx < len(matrix_bits_one):
            matrix_bits_all[0][up_idx] = up_list
            matrix_phred_all[0][up_idx] = phred_bits_list.copy()  # 第一层
        else:
            # 一定是与matrix_2匹配的随机bits
            random_dic_2[up_idx] = up_list

        if down_idx < len(matrix_bits_one):
            matrix_bits_all[1][down_idx] = down_list
            matrix_phred_all[1][down_idx] = phred_bases_list.copy()  # 第二层
        else:
            # 一定是与matrix_1匹配的随机bits
            random_dic_1[down_idx] = down_list
    # 因为在编码时（竖向极化后）有删掉第0行（其值均为0），又因为对应行号也会发生变化，所以不能靠初始化0，还得插入0行
    zeroes_list = [0 for _ in range(segment_length)]
    matrix_bits_all[0].insert(0, zeroes_list.copy())  # 对列表的复制一定要特别小心，它与数组可不同
    matrix_bits_all[1].insert(0, zeroes_list.copy())
    phred_insert = [0.999999 for _ in range(segment_length)]
    matrix_phred_all[0].insert(0, phred_insert.copy())
    matrix_phred_all[1].insert(0, phred_insert.copy())
    return matrix_bits_all, random_dic_1, random_dic_2, matrix_phred_all


def matrix_decode_new(matrix_dna, index_numbers, polarParams, last_case='last_3'):  # 与动态处理最后一个单位矩阵兼容
    global num_total_sub, num_total_del, num_total_ins
    # X_hat_1, X_hat_2 = [], []
    # lists_decode_1_new, lists_decode_2_new = [], []
    # 支持任意切割
    lists_decode_1_new = [[] for _ in range(segment_length)]
    lists_decode_2_new = [[] for _ in range(segment_length)]
    lists_decode_1_flag = [[] for _ in range(segment_length)]
    lists_decode_2_flag = [[] for _ in range(segment_length)]
    lists_1_received_01_column = [[] for _ in range(segment_length)]  # 记录下最后1次解码时接收到的01信息
    lists_1_received_phred_column = [[] for _ in range(segment_length)]  # 记录下最后1次解码时接收到的01信息的质量值
    lists_2_received_01_column = [[] for _ in range(segment_length)]  # 记录下最后1次解码时接收到的01信息
    lists_2_received_phred_column = [[] for _ in range(segment_length)]  # 记录下最后1次解码时接收到的01信息的质量值
    X_hat_1_received_01_column = [[] for _ in range(segment_length)]  # 记录下最后1次解码时, 第1层的X_hat，更大list解码第2层要用
    for i in range(1, segment_n + 1):
        matrix_bits_all, random_dic_1, random_dic_2, matrix_phred_all = matrix_dna_data2bits(matrix_dna[0],
                                                                                             index_numbers, last_case)
        # 先解码第一层matrix_bits_all[0]
        X_hat_1 = []
        matrix_bits_array_1 = np.array(matrix_bits_all[0], dtype=object)
        matrix_bits_array_1 = matrix_bits_array_1.T
        matrix_phred_array_1 = np.array(matrix_phred_all[0], dtype=object)
        matrix_phred_array_1 = matrix_phred_array_1.T
        if i > back_length:
            temp_n_0 = (i - back_length) * step
        else:
            temp_n_0 = 0
        # temp_n_1 = i * step
        temp_n_1 = min(segment_length, i * step)
        # 用并行
        # results = (
        #     Parallel(n_jobs=threads_number)(
        #         delayed(CASCL_decoder_first_layer)(y_1, last_case) for y_1 in matrix_bits_array_1[temp_n_0:temp_n_1]))
        results = (
            Parallel(n_jobs=threads_number)(
                delayed(CASCL_decoder_first_layer)(matrix_bits_array_1[j], matrix_phred_array_1[j], polarParams, last_case) for j in
                range(temp_n_0, temp_n_1)))
        for result in results:  # 这里可能有问题,在修改DNA矩阵时，只当前解码的2个step就没有问题
            X_hat_1.append(result[1])
        if i >= back_length:
            for j in range(step):  # 只选最左边，3步中的第1步，其已经修改过2次，极化过3次，将其存入最终结果
                # lists_decode_1_new.append(results[j][0])
                lists_decode_1_new[temp_n_0 + j] = results[j][0]
                lists_decode_1_flag[temp_n_0 + j] = results[j][2]
                lists_1_received_01_column[temp_n_0 + j] = matrix_bits_array_1[temp_n_0 + j]
                lists_1_received_phred_column[temp_n_0 + j] = matrix_phred_array_1[temp_n_0 + j]
                X_hat_1_received_01_column[temp_n_0 + j] = results[j][1]
        # 再解码第二层matrix_bits_all[1]
        X_hat_2 = []
        matrix_bits_array_2 = np.array(matrix_bits_all[1], dtype=object)
        matrix_bits_array_2 = matrix_bits_array_2.T
        matrix_phred_array_2 = np.array(matrix_phred_all[1], dtype=object)
        matrix_phred_array_2 = matrix_phred_array_2.T
        X_hat_1 = np.array(X_hat_1)

        # results = (Parallel(n_jobs=threads_number)(delayed(CASCL_decoder_second_layer)
        #                                            (X_hat_1[j - temp_n_0], matrix_bits_array_1[j],
        #                                             matrix_bits_array_2[j], last_case) for j in
        #                                            range(temp_n_0, temp_n_1)))  # 这里的temp_n_1不用+1
        results = (Parallel(n_jobs=threads_number)(delayed(CASCL_decoder_second_layer)
                                                   (X_hat_1[j - temp_n_0], matrix_bits_array_1[j],
                                                    matrix_bits_array_2[j], matrix_phred_array_2[j], polarParams,last_case) for j in
                                                   range(temp_n_0, temp_n_1)))  # 这里的temp_n_1不用+1
        # 为什么上面的X_hat_1[j - temp_n_0],要j - temp_n_0？ 答：因为必须从下标0开始取数据。
        for result in results:
            X_hat_2.append(result[1])
        if i >= back_length:
            for j in range(step):  # 只选最左边，3步中的第1步，其已经修改过2次，极化过3次，将其存入最终结果
                # lists_decode_2_new.append(results[j][0])
                lists_decode_2_new[temp_n_0 + j] = results[j][0]
                lists_decode_2_flag[temp_n_0 + j] = results[j][2]
                lists_2_received_01_column[temp_n_0 + j] = matrix_bits_array_2[temp_n_0 + j]
                lists_2_received_phred_column[temp_n_0 + j] = matrix_phred_array_2[temp_n_0 + j]
        # 将X_hat_1和X_hat_2进行转置，并干掉各自的第0行
        X_hat_1 = X_hat_1.T
        X_hat_1 = X_hat_1.tolist()
        X_hat_1 = X_hat_1[1:]
        X_hat_2 = np.array(X_hat_2)
        X_hat_2 = X_hat_2.T
        X_hat_2 = X_hat_2.tolist()
        X_hat_2 = X_hat_2[1:]

        # for j in range(len(X_hat_1)):
        #     up_idx, down_idx = index_numbers[j][0], index_numbers[j][1]
        #     if up_idx >= len(X_hat_1) or down_idx >= len(X_hat_2):
        #         continue  # 遇到随机bits的行号就先不处理。这样做是有性能损失的，因为这样，凡是添加的随机bits的DNA序列都没有进行修改。补上这部分是应该的。
        # 下面实现对有随机bits的DNA序列也进行修改
        for j in range(len(index_numbers)):
            up_idx, down_idx = index_numbers[j][0], index_numbers[j][1]
            # 考虑到encode_data部分是这样操作的：先从data_1的矩阵中随机取一条数据，再去data_2的矩阵中找匹配数据，如果没有找到则生成随机数据与其匹配；当data_1数据取完后，从data_2随机取一条数据，再生成随机数据与之匹配。
            # 所以up_idx中大多数对应的是data_1的数据，但也有一些data_2的数据。另外，down_idx中大多数对应的是data_2的数据，但也有一些是随机生成的数据。
            # 但有以下规则：当up_idx与down_idx都<len(X_hat_1)时，up_idx是data_1的数据，do_idx对应的是data_2的数据。当down_idx>len(X_hat_1)时，up_idx对应的数据有可能是data_1的，也有可能是data_2的。这就比较麻烦了。
            # 应该要修改一下data部分的encode规则才行。
            if up_idx < len(X_hat_1) and down_idx < len(X_hat_2):
                up_list, down_list = X_hat_1[up_idx], X_hat_2[down_idx]
            elif up_idx >= len(X_hat_1):  # 上面是随机bits，下面是matrix_2的数据
                if up_idx in random_dic_2:
                    up_list = random_dic_2[up_idx][temp_n_0: temp_n_1]
                else:
                    up_list = [0 for _ in range(temp_n_1 - temp_n_0)]  # 用0补充
                if down_idx < len(X_hat_2):
                    down_list = X_hat_2[down_idx]  # 这里有报错？？可能是down_idx出错了，并超出了范围，发生的概率非常低，小于100万分之一
                else:
                    down_list = [0 for _ in range(temp_n_1 - temp_n_0)]  # 用0补充
            else:  # 上面是matrix_1的数据，下面是随机bits
                up_list = X_hat_1[up_idx]
                if down_idx in random_dic_1:
                    down_list = random_dic_1[down_idx][temp_n_0: temp_n_1]
                else:
                    down_list = [0 for _ in range(temp_n_1 - temp_n_0)]  # 用0补充
            dna_sequence_decode = DNA_encode.two_lists_to_sequence(up_list, down_list)
            # print("dna_sequence_decode:", dna_sequence_decode, len(dna_sequence_decode))
            # print("matrix_dna[0][j][:temp_n]", matrix_dna[0][j][:temp_n], len(matrix_dna[0][j][:temp_n]))
            # 只与当前极化码解码的部分与原DNA序列部分进行修改，不从头开始
            if dna_sequence_decode == matrix_dna[0][j][0][temp_n_0:temp_n_1]:
                # pass
                for k in range(temp_n_0, temp_n_1):
                    matrix_dna[0][j][1][k] = 0.9998  # 得修改原DNA序列中的质量值才行
            else:  # 不相等时，才需要修改
                edit_distance = Levenshtein.distance(dna_sequence_decode, matrix_dna[0][j][0][temp_n_0:temp_n_1])
                if edit_distance > 6:
                    pass
                #     print("edit_distance太大了！在DNA_data的修改中。j = %d, edit_distance = %d" % (j, edit_distance),
                #           last_case)
                #     print("dna_sequence_decode:              ", dna_sequence_decode)
                #     print("matrix_dna[0][j][0][temp_n_0:temp_n]:", matrix_dna[0][j][0][temp_n_0:temp_n_1])
                num_sub, num_del, num_ins, dna_sequence_modify = check_and_edit_dna_sequence(dna_sequence_decode,
                                                                                             matrix_dna[0][j], temp_n_0)
                # num_sub, num_del, num_ins, dna_sequence_rec = (
                #     check_and_edit_dna_sequence_new(dna_sequence_decode, matrix_dna[0][j], temp_n_0))  # 这个好像性能没有原来的好
                # print("dna_sequence_rec:", dna_sequence_rec, len(dna_sequence_rec))
                # print("matrix_dna[0][j]", matrix_dna[0][j], len(matrix_dna[0][j]))
                num_total_sub += num_sub
                num_total_del += num_del
                num_total_ins += num_ins
                matrix_dna[0][j] = dna_sequence_modify

    # 现在只用解码最后2个step的数据, 似乎这个方案还是没有再重新极化解码一次好，等等，好像差不多。
    matrix_bits_all, _, _, matrix_phred_all = matrix_dna_data2bits(matrix_dna[0], index_numbers, last_case)
    lists_decode = [[], []]
    # matrix_bits已经在函数matrix_dna_data2bits中插入了0行，故这里不用再插入
    # 先解码第一层matrix_bits_all[0]的最后一个step
    X_hat_1_new = []
    matrix_bits_array_1 = np.array(matrix_bits_all[0])
    matrix_bits_array_1 = matrix_bits_array_1.T
    Y1_bits_array_column = copy.deepcopy(matrix_bits_array_1)  # 在用larger_list再次解码的第2层解码要用
    matrix_bits_array_1 = matrix_bits_array_1[-2 * step:]
    matrix_phred_array_1 = np.array(matrix_phred_all[0], dtype=object)
    matrix_phred_array_1 = matrix_phred_array_1.T
    matrix_phred_array_1 = matrix_phred_array_1[-2 * step:]
    # 用并行
    # results = (
    #     Parallel(n_jobs=threads_number)(delayed(CASCL_decoder_first_layer)(y_1, last_case) for y_1 in matrix_bits_array_1))
    results = (
        Parallel(n_jobs=threads_number)(
            delayed(CASCL_decoder_first_layer)(matrix_bits_array_1[j], matrix_phred_array_1[j], polarParams, last_case) for j in
            range(len(matrix_bits_array_1))))
    # for result in results:
    #     lists_decode_1_new.append(result[0])
    #     X_hat_1_new.append(result[1])
    temp_n_0 = -2 * step
    for j in range(len(results)):
        lists_decode_1_new[temp_n_0 + j] = results[j][0]
        lists_decode_1_flag[temp_n_0 + j] = results[j][2]
        X_hat_1_new.append(results[j][1])
        lists_1_received_01_column[temp_n_0 + j] = matrix_bits_array_1[j]  # 这里要小心matrix_bits_array_1只取了整个矩阵的[-2 * step:]
        lists_1_received_phred_column[temp_n_0 + j] = matrix_phred_array_1[j]
        X_hat_1_received_01_column[temp_n_0 + j] = results[j][1]

    # 尝试对其中没有通过CRC检验的列，以增加list的方式，再进行SCL解码1次
    CASCL_decoder_name = "CASCL_decoder_first_layer"
    lists_decode_1_new, lists_decode_1_flag = CASCL2failCRCwithlargerlist(CASCL_decoder_name, lists_decode_1_new,
                                                     lists_decode_1_flag, lists_1_received_01_column,
                                                     lists_1_received_phred_column, _, _, polarParams, last_case)


    # 再解码第二层matrix_bits_all[1]的最后一个step
    matrix_bits_array_2 = np.array(matrix_bits_all[1])
    matrix_bits_array_2 = matrix_bits_array_2.T
    matrix_bits_array_2 = matrix_bits_array_2[-2 * step:]
    matrix_phred_array_2 = np.array(matrix_phred_all[1], dtype=object)
    matrix_phred_array_2 = matrix_phred_array_2.T
    matrix_phred_array_2 = matrix_phred_array_2[-2 * step:]
    X_hat_1_new = np.array(X_hat_1_new)
    # results = (Parallel(n_jobs=threads_number)(delayed(CASCL_decoder_second_layer)
    #                                            (X_hat_1_new[i], matrix_bits_array_1[i], matrix_bits_array_2[i], last_case)
    #                                            for i in range(len(matrix_bits_array_2))))
    results = (Parallel(n_jobs=threads_number)(delayed(CASCL_decoder_second_layer)
                                               (X_hat_1_new[j], matrix_bits_array_1[j], matrix_bits_array_2[j],
                                                matrix_phred_array_2[j],polarParams, last_case)
                                               for j in range(len(matrix_bits_array_2))))
    # for result in results:
    #     lists_decode_2_new.append(result[0])
    for j in range(len(results)):
        lists_decode_2_new[temp_n_0 + j] = results[j][0]
        lists_decode_2_flag[temp_n_0 + j] = results[j][2]
        lists_2_received_01_column[temp_n_0 + j] = matrix_bits_array_2[j]  # 注意到matrix_bits_array_2 和 matrix_phred_array_2 只包含[-2 * step:]
        lists_2_received_phred_column[temp_n_0 + j] = matrix_phred_array_2[j]

    # 尝试对其中没有通过CRC检验的列，以增加list的方式，再进行SCL解码1次
    CASCL_decoder_name = "CASCL_decoder_second_layer"
    X_hat_1_received_01_column = np.array(X_hat_1_received_01_column)
    lists_decode_2_new, lists_decode_2_flag = CASCL2failCRCwithlargerlist(CASCL_decoder_name, lists_decode_2_new,
                                                     lists_decode_2_flag, lists_2_received_01_column,
                                                     lists_2_received_phred_column, X_hat_1_received_01_column,
                                                     Y1_bits_array_column, polarParams, last_case)  # 要小心，这里要传入的Y1是最后1次解码时接收到的第一层的01序列，不是解码后的结果

    lists_decode_1_new = np.array(lists_decode_1_new)   # 转置操作放在这里，以免上面函数CASCL2failCRCwithlargerlist用到lists_decode_1_new时，会出错
    lists_decode_1_new = lists_decode_1_new.T
    lists_decode[0] = lists_decode_1_new.tolist()


    lists_decode_2_new = np.array(lists_decode_2_new)
    lists_decode_2_new = lists_decode_2_new.T
    lists_decode[1] = lists_decode_2_new.tolist()

    # 把lists_1_received_01_column也进行转置
    lists_1_received_01_column = np.array(lists_1_received_01_column)
    lists_1_received_01_column = lists_1_received_01_column.T
    lists_1_received_01 = lists_1_received_01_column.tolist()

    # 把lists_2_received_01_column也进行转置
    lists_2_received_01_column = np.array(lists_2_received_01_column)
    lists_2_received_01_column = lists_2_received_01_column.T
    lists_2_received_01 = lists_2_received_01_column.tolist()

    lists_received_01 = [lists_1_received_01, lists_2_received_01]
    # 下面对lists_decode的两层矩阵分别进行奇偶校验
    lists_decode[0] = handle_parity(lists_decode[0], lists_decode_1_flag)
    lists_decode[1] = handle_parity(lists_decode[1], lists_decode_2_flag)
    return lists_decode, lists_received_01


def handle_parity(matrix_lists_have_parity, matrix_list_decode_flag):
    # 经过考虑还是加偶和奇两位的奇偶校验，这样错1列肯定能纠，错2列时，若出错列号是偶+奇也一定能纠。240816
    # 若是 奇+奇 或 偶+偶 但出错bit有错开，即对具体某行只有1个错误，也不能纠，主要问题无法确定出错的列号。240816
    # 支持偶+奇 2 位奇偶校验位
    matrix_lists_new = []  # 没有了最后一列的奇偶检验位
    cnt_flag_false = 0
    column_false_num_1, column_false_num_2 = -1, -1  # 记录出错的列号
    for i in range(len(matrix_list_decode_flag)):
        if not matrix_list_decode_flag[i]:
            cnt_flag_false += 1
            if column_false_num_1 == -1:
                column_false_num_1 = i
            elif column_false_num_2 == -1:
                column_false_num_2 = i
    if cnt_flag_false == 1:
        # 只有1列出错，可以纠正
        # print("太好了，只有 1 列出错，进行奇偶校验修复，column_false_num_1 = %d" % column_false_num_1)
        # 感觉这里可以不区别奇偶情况
        if column_false_num_1 % 2 == 0:
            for my_list in matrix_lists_have_parity:
                if column_false_num_1 >= len(my_list):
                    continue
                sum_parity = 0
                for i in range(0, len(my_list), 2):
                    sum_parity ^= my_list[i]
                if sum_parity != 0:
                    my_list[column_false_num_1] ^= 1  # 修改对应位置的值
        else:
            for my_list in matrix_lists_have_parity:
                if column_false_num_1 >= len(my_list):
                    continue
                sum_parity = 0
                for i in range(1, len(my_list), 2):
                    sum_parity ^= my_list[i]
                if sum_parity != 0:
                    my_list[column_false_num_1] ^= 1  # 修改对应位置的值
    elif cnt_flag_false == 2:
        if (column_false_num_1 % 2) + (column_false_num_2 % 2) == 1:
            # print("太好了，有 奇+偶 2 列出错，进行奇偶校验修复, column_false_num_1 = %d, column_false_num_2 = %d" % (
            #     column_false_num_1, column_false_num_2))
            if column_false_num_1 % 2 == 0:
                even_column_num = column_false_num_1
                odd_column_num = column_false_num_2
            else:
                even_column_num = column_false_num_2
                odd_column_num = column_false_num_1
            for my_list in matrix_lists_have_parity:
                even_sum_parity, odd_sum_parity = 0, 0
                for i in range(0, len(my_list), 2):
                    even_sum_parity ^= my_list[i]
                    odd_sum_parity ^= my_list[i + 1]
                if even_column_num < len(my_list) and even_sum_parity != 0:
                    my_list[even_column_num] ^= 1  # 修改对应位置的值
                if odd_column_num < len(my_list) and odd_sum_parity != 0:
                    my_list[odd_column_num] ^= 1  # 修改对应位置的值
        else:
            pass
            # print("不好，有 非奇+偶 2 列出错，column_false_num_1 = %d, column_false_num_2 = %d" % (
            #     column_false_num_1, column_false_num_2))
    elif cnt_flag_false > 2:
        pass
        # print("不好，这个矩阵有 %d 个列出错了" % cnt_flag_false)
    # 去掉最右边的奇偶检验位
    for my_list in matrix_lists_have_parity:
        matrix_lists_new.append(my_list[:-parity_length])
    return matrix_lists_new


def CASCL2failCRCwithlargerlist(CASCL_decoder_name, matrix_list_decode_new, matrix_list_decode_flag,
                                matrix_list_received_01_column, matrix_list_received_phred_column,
                                X_hat_1_received_01_column, Y_1_received_01_column, polarParams, last_case="last_3"):
    # 不仅要返回larger_list后解码的结果01序列，如果解码成功，还应该修改其对应的flag。方便后面奇偶校验。240831
    global CASCL_decoder_largerList, larger_list
    if all(flag for flag in matrix_list_decode_flag):  # 如果全部是真，直接返回
        return matrix_list_decode_new, matrix_list_decode_flag
    flag = 0  # 用于区分不同情况下解码时，所要传的参数数量不同。
    # 有错误则进一步处理
    if CASCL_decoder_name == "CASCL_decoder_index_2":
        flag = 0
        CASCL_decoder_largerList = CASCL_decoder_index_2
        larger_list = larger * List_index_2
    elif CASCL_decoder_name == "CASCL_decoder_index_1":
        flag = 0
        CASCL_decoder_largerList = CASCL_decoder_index_1
        larger_list = larger * List_index_1
    elif CASCL_decoder_name == "CASCL_decoder_first_layer":
        flag = 1
        # CASCL_decoder_largerList = CASCL_decoder_first_layer
        larger_list = larger * list_16
    elif CASCL_decoder_name == "CASCL_decoder_second_layer":
        flag = 2
        # CASCL_decoder_largerList = CASCL_decoder_second_layer
        larger_list = larger * list_16
    elif CASCL_decoder_name == "CASCL_decoder_index_1_last":
        flag = 3
        # CASCL_decoder_largerList = CASCL_decoder_index_1_last
        larger_list = larger * list_16

    if flag == 0:  # index_0 or index_1 的情况
        for i in range(len(matrix_list_decode_flag)):
            if not matrix_list_decode_flag[i]:  # 如果该列没有通过CRC检验，尝试 larger_list 进行解码
                # print("不好，发现有1列错误，尝试用 larger_list = %d 再次解码" % larger_list)
                matrix_list_decode_new[i], _, crc_flag = CASCL_decoder_largerList(matrix_list_received_01_column[i],
                                                                                  matrix_list_received_phred_column[i],
                                                                                  larger_list)
                if crc_flag:
                    # print("太好了，尝试用 larger_list = %d 再次解码 成功" % larger_list)
                    matrix_list_decode_flag[i] = True   # 解码成功，当然也要修改其对应的flag，方便后面的奇偶检验
                else:
                    pass
                    # print("不好，尝试用 larger_list = %d 再次解码 失败" % larger_list)
    elif flag == 1:  # CASCL_decoder_first_layer
        for i in range(len(matrix_list_decode_flag)):
            if not matrix_list_decode_flag[i]:  # 如果该列没有通过CRC检验，尝试 larger_list 进行解码
                # print("不好，发现有1列错误，尝试用 larger_list = %d 再次解码" % larger_list)
                matrix_list_decode_new[i], _, crc_flag = CASCL_decoder_first_layer(matrix_list_received_01_column[i],
                                                                                  matrix_list_received_phred_column[i],
                                                                                  polarParams,last_case,  larger_list)
                if crc_flag:
                    # print("太好了，尝试用 larger_list = %d 再次解码 成功" % larger_list)
                    matrix_list_decode_flag[i] = True  # 解码成功，当然也要修改其对应的flag，方便后面的奇偶检验
                else:
                    pass
                    # print("不好，尝试用 larger_list = %d 再次解码 失败" % larger_list)
    elif flag == 2:  # CASCL_decoder_second_layer
        for i in range(len(matrix_list_decode_flag)):
            if not matrix_list_decode_flag[i]:  # 如果该列没有通过CRC检验，尝试 larger_list 进行解码
                # print("不好，发现有1列错误，尝试用 larger_list = %d 再次解码" % larger_list)
                matrix_list_decode_new[i], _, crc_flag = CASCL_decoder_second_layer(X_hat_1_received_01_column[i],
                                                                                  Y_1_received_01_column[i],
                                                                                  matrix_list_received_01_column[i],
                                                                                  matrix_list_received_phred_column[i],
                                                                                  polarParams,last_case, larger_list)
                if crc_flag:
                    # print("太好了，尝试用 larger_list = %d 再次解码 成功" % larger_list)
                    matrix_list_decode_flag[i] = True  # 解码成功，当然也要修改其对应的flag，方便后面的奇偶检验
                else:
                    pass
                    # print("不好，尝试用 larger_list = %d 再次解码 失败" % larger_list)
    elif flag == 3:  # CASCL_decoder_index_1_last
        for i in range(len(matrix_list_decode_flag)):
            if not matrix_list_decode_flag[i]:  # 如果该列没有通过CRC检验，尝试 larger_list 进行解码
                # print("不好，发现有1列错误，尝试用 larger_list = %d 再次解码" % larger_list)
                matrix_list_decode_new[i], _, crc_flag = CASCL_decoder_index_1_last(
                    matrix_list_received_01_column[i],
                    matrix_list_received_phred_column[i],
                    last_case, larger_list)
                if crc_flag:
                    # print("太好了，尝试用 larger_list = %d 再次解码 成功" % larger_list)
                    matrix_list_decode_flag[i] = True  # 解码成功，当然也要修改其对应的flag，方便后面的奇偶检验
                else:
                    pass
                    # print("不好，尝试用 larger_list = %d 再次解码 失败" % larger_list)
    return matrix_list_decode_new, matrix_list_decode_flag
