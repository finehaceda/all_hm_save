import time

from .utils import data_handle, index_operator, DNA_encode, DNA_decode


def file_to_dna(input_file_path,frozen_bits_len):
    # 可以考虑先判断文件类型再选用相应的压缩或处理函数，我们先跑通图像
    # 对图像的预处理，压缩+解压
    # 我在考虑是否不应该压缩，提出的方案越简洁越好，就只是Polar_DNA算法，是的，这个要改好

    print("开始读入文件的二进制序列new!")
    file_binary = data_handle.file_to_binary_sequence(input_file_path)
    print("对读入的二进制文件进行随机化")
    file_binary_random = data_handle.file_binary_to_random(file_binary)  # 经过对比，进行随机化后，添加的随机bits数量会少一点
    print("支持对每一行数据添加 2 位奇偶检验，偶+奇")
    input_matrix = data_handle.read_binary_new(file_binary_random)
    matrices_ori = index_operator.divide_to_matrices_two_layer_scheme(input_matrix,frozen_bits_len)  # 这里没有问题
    # 现在的编码方案还没有实现对单位矩阵中index部分的两次表格优化，打算后面再写 2023-12-18
    matrices_dna_ori, flag, matrices_01_ori = DNA_encode.dna_encode(matrices_ori,frozen_bits_len)
    # 其中matrices_ori_01是极化码编码后的01矩阵，与最后一次解码时接收到的01矩阵对应
    # print('matrices_dna_ori[0][0][0] =', matrices_dna_ori[0][0][0], type(matrices_dna_ori[0][0][0]))
    # print('matrices_dna_ori[0][1][0][0] =', matrices_dna_ori[0][1][0][0], type(matrices_dna_ori[0][1][0][0]))
    for matrix in matrices_dna_ori:
        if len(matrix) != 2:
            print("出错啦，有一个单位DNA矩阵不正常")
    if not flag:
        print("至少有一个矩阵的DNA序列不合格，编码失败！")
        return
    # 下面给每个单位矩阵的每行DNA序列加上碱基行号，矩阵号+行号
    # dna_sequences = data_handle.add_matrix_and_row_numbers(matrices_dna_ori)  # 块地址与块内地址在一起
    dna_sequences = data_handle.add_matrix_and_row_numbers_new(matrices_dna_ori)  # 块地址与块内地址分离
    print("copy前，dna_sequences的行数：", len(dna_sequences))

    # dna_sequences = data_handle.copy_dna_sequences(dna_sequences, copy_num=copy_number_pyhsics)
    # print("copy后，dna_sequences的行数：", len(dna_sequences))
    idx = 100
    print("第 %d 行DNA序列为：" % idx, dna_sequences[idx], len(dna_sequences[idx]))
    print("这里要更好的模拟拼接+3代模拟+拆分+清洗全过程，故长度为260是正确的！")
    for seq in dna_sequences:   # 检查是否有长度不符合预期的DNA序列
        if len(seq) != 260:
            print(seq, len(seq))
    # dna_sequences = data_handle.Copy_dna_sequences(dna_sequences, copy_number)
    print("编码成功！")

    return dna_sequences, matrices_ori, matrices_dna_ori, matrices_01_ori
    # return dna_sequences

def dna_to_file(dna_seq_and_phred, matrices_ori, matrices_dna_ori, matrices_01_ori, output_file_path, file_name, i,frozen_bits_len):
    # 这里没有聚类和清洗步骤，可以将dna_sequences看成是清洗后得到的consensus
    # 先按碱基行号恢复各单位矩阵（数据部分+index部分），得到有序的DNA序列
    print("解码时dna_seq_and_phred的条数：", len(dna_seq_and_phred))
    print("原DNA单位矩阵的个数：", len(matrices_ori))

    print("现在开始恢复DNA矩阵")
    start_time = time.time()
    # matrices_dna = data_handle.dna_seq2matrices(dna_seq_and_phred)  # 恢复成原来的DNA大矩阵,块地址与块内地址在一起
    matrices_dna, dna_seqs_ture_num = data_handle.dna_seq2matrices_new(dna_seq_and_phred)  # 恢复成原来的DNA大矩阵，块地址与块内地址分离
    # print('matrices_dna[0][0][0][0] =', matrices_dna[0][0][0][0], type(matrices_dna[0][0][0][0]))
    # print('matrices_dna[0][1][0][0][0] =', matrices_dna[0][1][0][0][0], type(matrices_dna[0][1][0][0][0]))
    end_time = time.time()
    print("恢复DNA矩阵用时为：{:.2f}秒".format(end_time - start_time))
    # 检查各个数据矩阵在列上的出错情况（直接对比）
    data_handle.check_dna_vertical_error_hamming(matrices_dna, matrices_dna_ori, file_name, i)
    # 检查各个数据矩阵在列上的出错情况（编辑距离）
    data_handle.check_dna_vertical_error_new(matrices_dna, matrices_dna_ori, file_name, i)  # 当DNA序列的各条DNA序列长度不一样时，是不能用np.array的方法实现转置的
    # 竖向对数据DNA部分有影响，是不是对index部分也会有影响。是的，这个index部分的影响，又会去影响data部分的恢复
    print("现在开始DNA矩阵进行解码")
    # matrices_decode = DNA_decode.dna_decode(matrices_dna)  # 恢复成原来的bits大矩阵（对应）
    matrices_decode, matrices_received_01 = DNA_decode.dna_decode_new(matrices_dna, dna_seqs_ture_num,frozen_bits_len)  # 恢复成原来的bits大矩阵（对应）,实现对接收到的DNA矩阵进行修改
    # 下面检查整体恢复情况
    block_recover_ration,bit_recover,bad_bits,bitnums = data_handle.check_recover_ration(matrices_ori, matrices_decode, file_name, i)
    # 下面检查最后一次解码时，接收到的01矩阵与原正确01矩阵之间的错误率
    data_handle.check_received_01_ration(matrices_01_ori, matrices_received_01, file_name, i)

    # 开始恢复文件
    file_binary_recover_random = data_handle.matrices_bits_to_binary_sequence(matrices_decode)
    print("对恢复的二进制序列再次进行异或操作，以恢复原来的01序列")
    file_binary_recover = data_handle.file_binary_to_random(file_binary_recover_random)
    data_handle.binary_sequence_to_file(file_binary_recover, output_file_path)
    # print("文件 " + output_file_path + " 恢复成功")
    return block_recover_ration,bit_recover,bad_bits,bitnums