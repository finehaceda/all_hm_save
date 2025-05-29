import numpy as np
import math


def crc_detect(signal, crc_poly):
    # crc_remainder = np.zeros(len(crc_poly) - 1)
    crc_remainder = np.zeros(len(crc_poly))
    for i in range(len(signal)):
        crc_remainder = np.hstack((crc_remainder[1:], signal[i]))
        if crc_remainder[0] == 1:
            crc_remainder = np.mod(np.add(crc_remainder, crc_poly), 2)
    crc_check = np.sum(crc_remainder) == 0
    return crc_check
def get_llr_layer(normal):
    layer_vec = np.zeros(normal, dtype=int)
    for phi in range(normal - 1):
        psi = phi + 1
        layer = 0
        while psi % 2 == 0:
            psi = int(psi / 2)
            layer = layer + 1
        layer_vec[phi + 1] = layer
    return layer_vec


def get_bit_layer(normal):
    layer_vec = np.zeros(normal, dtype=int)
    for phi in range(normal):
        psi = int(phi / 2)
        layer = 0
        while psi % 2 == 1:
            psi = int(psi / 2)
            layer = layer + 1
        layer_vec[phi] = layer
    return layer_vec


def power(base, pow_range):
    results = np.empty(pow_range.size, dtype=int)
    for i in range(pow_range.size):
        results[i] = int(math.pow(base, pow_range[i]))
    return results

def SigReceive2llr_single_layer(SigReceive, phred_array):  # 支持质量值
    N = len(SigReceive)
    Prob0 = np.ones(N, dtype=np.float64)
    Prob1 = np.ones(N, dtype=np.float64)
    temp = np.where(SigReceive == 1)[0]
    # Prob0[temp] = p_bit
    # Prob1[temp] = 1 - p_bit
    Prob0[temp] = 1 - phred_array[temp]
    Prob1[temp] = phred_array[temp]
    temp = np.where(SigReceive == 0)[0]
    # Prob0[temp] = 1 - p_bit
    # Prob1[temp] = p_bit
    Prob0[temp] = phred_array[temp]
    Prob1[temp] = 1 - phred_array[temp]
    llr = np.log2(Prob0 / Prob1)
    return llr

def SigReceive2llr_single_layer_new(SigReceive, Confidence_level):
    N = len(SigReceive)
    Prob0 = np.ones(N, dtype=np.float64)
    Prob1 = np.ones(N, dtype=np.float64)
    temp = np.where(SigReceive == 1)[0]
    Prob0[temp] = 1 - Confidence_level[temp]
    Prob1[temp] = Confidence_level[temp]
    temp = np.where(SigReceive == 0)[0]
    Prob0[temp] = Confidence_level[temp]
    Prob1[temp] = 1 - Confidence_level[temp]
    llr = np.log2(Prob0 / Prob1)
    return llr

def SigReceive2llr_first_layer(SigReceive_Y1, Y1_phred):
    # 支持质量值
    N = len(SigReceive_Y1)
    Prob0 = np.ones(N, dtype=np.float64)
    Prob1 = np.ones(N, dtype=np.float64)
    # p = 2 * p_basic_group / 3
    temp = np.where(SigReceive_Y1 == 1)[0]
    Prob0[temp] = 1 - Y1_phred[temp]
    Prob1[temp] = Y1_phred[temp]
    temp = np.where(SigReceive_Y1 == 0)[0]
    Prob0[temp] = Y1_phred[temp]
    Prob1[temp] = 1 - Y1_phred[temp]
    llr = np.log2(Prob0 / Prob1)
    return llr

def SigReceive2llr_second_layer(X_hat_1, SigReceive_Y1, SigReceive_Y2, Y2_bases_phred):
    # 支持质量值
    # 因为双层极化方案中，第二层从碱基可靠度换算成01可靠度的公式不同。Y2_bases_phred就是对应碱基的可靠度
    N = len(SigReceive_Y1)
    Prob0 = np.ones(N, dtype=np.float64)
    Prob1 = np.ones(N, dtype=np.float64)
    temp = np.where(SigReceive_Y2 == 1)[0]
    temp = np.array([index for index in temp if SigReceive_Y1[index] == X_hat_1[index]])
    if temp.size > 0:  # 当temp不为空时，才执行下面的代码，或者会报错
        # Prob0[temp] = p_basic_group / 3
        # Prob1[temp] = 1 - p_basic_group
        # 上面的做法是不对的，没有按照第二层的概率公式来算
        Prob0[temp] = (1 - Y2_bases_phred[temp]) / 3
        Prob1[temp] = Y2_bases_phred[temp]

    temp = np.where(SigReceive_Y2 == 0)[0]
    temp = np.array([index for index in temp if SigReceive_Y1[index] == X_hat_1[index]])
    if temp.size > 0:  # 当temp不为空时，才执行下面的代码，或者会报错
        # Prob0[temp] = 1 - p_basic_group
        # Prob1[temp] = p_basic_group / 3
        Prob0[temp] = Y2_bases_phred[temp]
        Prob1[temp] = (1 - Y2_bases_phred[temp]) / 3
    llr = np.log2(Prob0 / Prob1)
    return llr


def SC_decoder(llr, select_index, freeze_index, frozen_bits):
    N = len(llr)
    n = int(math.log(N, 2))
    # llr = SigReceive2llr(SigReceive)   # llr的转换放外面比较好
    frozen_flags = np.zeros(N, dtype=int)
    frozen_flags[freeze_index] = 1
    sig_in_all = np.zeros(N, dtype=int)
    sig_in_all[freeze_index] = frozen_bits   # 用sig_in_all把冻结bits和其位置存下来
    soft_info = np.zeros(N - 1)
    # hard_info = np.zeros([2, N - 1], dtype=int)
    hard_info = np.zeros([2, 2 * N - 1], dtype=int)
    u_hat = np.zeros(N, dtype=int)
    lambda_offset = power(2, np.arange(0, n + 1))
    # print("lambda_offset:", lambda_offset)
    llr_layer_vec = get_llr_layer(N)
    bit_layer_vec = get_bit_layer(N)
    for phi in range(N):
        if phi == 0:
            index_1 = lambda_offset[n - 1]
            for beta in range(index_1):
                soft_info[beta + index_1 - 1] = np.sign(llr[beta]) * np.sign(llr[beta + index_1]) * \
                    min(abs(llr[beta]), abs(llr[beta + index_1]))
            for i_layer in range(n - 2, -1, -1):
                index_1 = lambda_offset[i_layer]
                index_2 = lambda_offset[i_layer + 1]
                # for beta in range(index_1, index_2):
                #     soft_info[beta - 1] = (np.sign(soft_info[beta + index_1 - 1]) * np.sign(soft_info[beta + index_2 - 1])
                #                            * min(abs(soft_info[beta + index_1 - 1]), abs(soft_info[beta + index_2 - 1])))
                # 优化一下上面的for循环：
                for beta in range(index_1 - 1, index_2 - 1):
                    soft_info[beta] = (np.sign(soft_info[beta + index_1]) * np.sign(soft_info[beta + index_2])
                                           * min(abs(soft_info[beta + index_1]), abs(soft_info[beta + index_2])))
        elif phi == N // 2:
            index_1 = lambda_offset[n - 1]
            for beta in range(index_1):
                soft_info[beta + index_1 - 1] = (1 - 2 * hard_info[0][beta + index_1 - 1]) * llr[beta] + llr[beta + index_1]
            for i_layer in range(n - 2, -1, -1):
                index_1 = lambda_offset[i_layer]
                index_2 = lambda_offset[i_layer + 1]
                # for beta in range(index_1, index_2):
                #     soft_info[beta - 1] = np.sign(soft_info[beta + index_1 - 1]) * np.sign(soft_info[beta + index_2 - 1]) \
                #                          * min(abs(soft_info[beta + index_1 - 1]), abs(soft_info[beta + index_2 - 1]))
                # 优化一下上面的for循环：
                for beta in range(index_1 - 1, index_2 - 1):
                    soft_info[beta] = (np.sign(soft_info[beta + index_1]) * np.sign(soft_info[beta + index_2])
                                       * min(abs(soft_info[beta + index_1]), abs(soft_info[beta + index_2])))
        else:
            llr_layer = llr_layer_vec[phi]
            index_1 = lambda_offset[llr_layer]
            index_2 = lambda_offset[llr_layer + 1]
            # for beta in range(index_1, index_2):
            #     soft_info[beta - 1] = ((1 - 2 * hard_info[0][beta - 1]) * soft_info[beta + index_1 - 1] +
            #                            soft_info[beta + index_2 - 1])
            # 优化一下上面的for循环：
            for beta in range(index_1 - 1, index_2 - 1):
                soft_info[beta] = ((1 - 2 * hard_info[0][beta]) * soft_info[beta + index_1] + soft_info[beta + index_2])
            for i_layer in range(llr_layer - 1, -1, -1):
                index_1 = lambda_offset[i_layer]
                index_2 = lambda_offset[i_layer + 1]
                # for beta in range(index_1, index_2):
                #     soft_info[beta - 1] = np.sign(soft_info[beta + index_1 - 1]) * np.sign(soft_info[beta + index_2 - 1]) \
                #                             * min(abs(soft_info[beta + index_1 - 1]), abs(soft_info[beta + index_2 - 1]))
                # 优化一下上面的for循环：
                for beta in range(index_1 - 1, index_2 - 1):
                    soft_info[beta] = (np.sign(soft_info[beta + index_1]) * np.sign(soft_info[beta + index_2])
                                       * min(abs(soft_info[beta + index_1]), abs(soft_info[beta + index_2])))
        phi_mod_2 = phi % 2
        if frozen_flags[phi] == 1:
            # hard_info[0][phi_mod_2] = sig_in_all[phi]   # 这里的phi_mode_2可能有问题
            hard_info[phi_mod_2][0] = sig_in_all[phi]   # 注意到与MATLAB代码不同，在MATLAB中的行号对应这里的列号
            u_hat[phi] = sig_in_all[phi]
        else:
            if soft_info[0] < 0:
                hard_info[phi_mod_2][0] = 1
                u_hat[phi] = 1
            else:
                hard_info[phi_mod_2][0] = 0
                u_hat[phi] = 0
        # if phi_mod_2 == 1 and phi != N - 1:
        if phi_mod_2 == 1:   # 希望能输出X_hat
            bit_layer = bit_layer_vec[phi]
            for i_layer in range(bit_layer):
                index_1 = lambda_offset[i_layer]
                index_2 = lambda_offset[i_layer + 1]
                # for beta in range(index_1, index_2):
                #     hard_info[1][beta + index_1 - 1] = (hard_info[0][beta - 1] + hard_info[1][beta - 1]) % 2
                #     hard_info[1][beta + index_2 - 1] = hard_info[1][beta - 1]
                # 优化一下上面的for循环：
                for beta in range(index_1 - 1, index_2 - 1):
                    hard_info[1][beta + index_1] = (hard_info[0][beta] + hard_info[1][beta]) % 2
                    hard_info[1][beta + index_2] = hard_info[1][beta]
            index_1 = lambda_offset[bit_layer]
            index_2 = lambda_offset[bit_layer + 1]
            # for beta in range(index_1, index_2):
            #     hard_info[0][beta + index_1 - 1] = (hard_info[0][beta - 1] + hard_info[1][beta - 1]) % 2
            #     hard_info[0][beta + index_2 - 1] = hard_info[1][beta - 1]
            # 优化一下上面的for循环：
            for beta in range(index_1 - 1, index_2 - 1):
                hard_info[0][beta + index_1] = (hard_info[0][beta] + hard_info[1][beta]) % 2
                hard_info[0][beta + index_2] = hard_info[1][beta]
    sig_recover = u_hat[select_index]
    return sig_recover, hard_info[0][-N:]

def SCL_decoder(llr, select_index, freeze_index, frozen_bits, L):
    N = len(llr)
    n = int(math.log(N, 2))
    # llr = SigReceive2llr(SigReceive)   # llr的转换放外面比较好
    frozen_flags = np.zeros(N, dtype=int)
    frozen_flags[freeze_index] = 1
    sig_in_all = np.zeros(N, dtype=int)
    sig_in_all[freeze_index] = frozen_bits  # 用sig_in_all把冻结bits和其位置存下来
    lazy_copy = np.full([L, n], -1)
    soft_info = np.zeros([L, N - 1], dtype=np.float64)
    hard_info = np.zeros([2 * L, N - 1], dtype=int)
    # hard_info = np.zeros([2 * L, 2 * N - 1], dtype=int)  # 要输出X_hat，hard_info就得大点
    u_hat = np.zeros([L, N], dtype=int)
    PM = np.zeros(L, dtype=np.float64)   # path metrics
    active_path = np.zeros(L, dtype=int)  # Indicate if the path is active. '1'→active; '0' otherwise.
    # initialize
    active_path[0] = 1
    lazy_copy[0][:] = 0
    lambda_offset = power(2, np.arange(0, n + 1))
    # print("lambda_offset:", lambda_offset)
    llr_layer_vec = get_llr_layer(N)
    bit_layer_vec = get_bit_layer(N)
    # decoding starts
    for phi in range(N):
        layer = llr_layer_vec[phi]
        phi_mod_2 = phi % 2
        for l_index in range(L):  # List的编号为0,1,2,...,L-1
            if active_path[l_index] == 0:
                continue
            if phi == 0:
                index_1 = lambda_offset[n - 1]
                for beta in range(index_1):
                    soft_info[l_index][beta + index_1 - 1] = np.sign(llr[beta]) * np.sign(llr[beta + index_1]) * \
                                                    min(abs(llr[beta]), abs(llr[beta + index_1]))
                for i_layer in range(n - 2, -1, -1):
                    index_1 = lambda_offset[i_layer]
                    index_2 = lambda_offset[i_layer + 1]
                    for beta in range(index_1 - 1, index_2 - 1):
                        soft_info[l_index][beta] = (np.sign(soft_info[l_index][beta + index_1]) * np.sign(soft_info[l_index][beta + index_2])
                                           * min(abs(soft_info[l_index][beta + index_1]), abs(soft_info[l_index][beta + index_2])))
            elif phi == N // 2:
                index_1 = lambda_offset[n - 1]
                for beta in range(index_1):
                    x_tmp = hard_info[2 * l_index][beta + index_1 - 1]
                    soft_info[l_index][beta + index_1 - 1] = (1 - 2 * x_tmp) * llr[beta] + llr[beta + index_1]
                for i_layer in range(n - 2, -1, -1):
                    index_1 = lambda_offset[i_layer]
                    index_2 = lambda_offset[i_layer + 1]
                    for beta in range(index_1 - 1, index_2 - 1):
                        soft_info[l_index][beta] = (np.sign(soft_info[l_index][beta + index_1]) * np.sign(soft_info[l_index][beta + index_2])
                                           * min(abs(soft_info[l_index][beta + index_1]), abs(soft_info[l_index][beta + index_2])))
            else:
                index_1 = lambda_offset[layer]
                index_2 = lambda_offset[layer + 1]
                lazy_copy_value = lazy_copy[l_index][layer + 1]
                # 有优化下面的for循环的下标：
                for beta in range(index_1 - 1, index_2 - 1):
                    soft_info[l_index][beta] = (
                                (1 - 2 * hard_info[2 * l_index][beta]) * soft_info[lazy_copy_value][beta + index_1] +
                                soft_info[lazy_copy_value][beta + index_2])
                for i_layer in range(layer - 1, -1, -1):
                    index_1 = lambda_offset[i_layer]
                    index_2 = lambda_offset[i_layer + 1]
                    for beta in range(index_1 - 1, index_2 - 1):
                        soft_info[l_index][beta] = (np.sign(soft_info[l_index][beta + index_1]) * np.sign(soft_info[l_index][beta + index_2])
                                           * min(abs(soft_info[l_index][beta + index_1]), abs(soft_info[l_index][beta + index_2])))
        if frozen_flags[phi] == 0:  # if now we decode an unfrozen bit
            PM_pair = np.full((2, L), np.finfo(float).max)
            for l_index in range(L):    # 感觉是soft_info[l_index][0]的计算有问题。但是看起来上面的计算，也没有问题
                if active_path[l_index] == 0:
                    continue
                if soft_info[l_index][0] >= 0:
                    PM_pair[0][l_index] = PM[l_index]
                    PM_pair[1][l_index] = PM[l_index] + soft_info[l_index][0]
                else:
                    PM_pair[0][l_index] = PM[l_index] - soft_info[l_index][0]
                    PM_pair[1][l_index] = PM[l_index]
            middle = min(2 * sum(active_path), L)
            # print("middle = ", middle)
            PM_sort = sorted(PM_pair.flatten())  # 不知道这样行不行
            PM_cv = PM_sort[middle]
            # PM_cv = PM_sort[middle - 1]
            compare = np.zeros([2, L], dtype=int)
            for i in range(2):
                for j in range(L):
                    if PM_pair[i][j] < PM_cv:    # 这里一定要改成<，不然会报错。这时，上面PM_cv就应该等于PM_sort[middle]，即向后移一位
                        compare[i][j] = 1
            kill_index = np.full(L, L)  # 这里感觉比较怪，会不会有问题
            kill_cnt = -1
            for i in range(L):
                if (compare[0][i] == 0) and (compare[1][i] == 0):  # 这里没有问题，也会把active_path=0的list压入栈中
                    active_path[i] = 0
                    kill_cnt += 1
                    kill_index[kill_cnt] = i
            for l_index in range(L):
                if active_path[l_index] == 0:
                    continue
                path_state = compare[0][l_index] * 2 + compare[1][l_index]
                # path_state can equal to 0, but in this case we do no operation.
                if path_state == 1:  # 0, 1 , 即只保存向1走的这条路
                    u_hat[l_index][phi] = 1
                    hard_info[2 * l_index + phi_mod_2][0] = 1
                    PM[l_index] = PM_pair[1][l_index]
                elif path_state == 2:   # 1, 0, 即只保存向0走的这条路
                    u_hat[l_index][phi] = 0
                    hard_info[2 * l_index + phi_mod_2][0] = 0
                    PM[l_index] = PM_pair[0][l_index]
                elif path_state == 3:  # 1, 1, 同时保存向0和1两条路
                    index = kill_index[kill_cnt]
                    kill_cnt -= 1
                    active_path[index] = 1
                    # lazy copy
                    lazy_copy[index][:] = lazy_copy[l_index][:]
                    u_hat[index][:] = u_hat[l_index][:]
                    u_hat[l_index][phi] = 0  # 原来的List走0这条路
                    u_hat[index][phi] = 1  # 克隆的新list走1这条路
                    hard_info[2 * l_index + phi_mod_2][0] = 0
                    hard_info[2 * index + phi_mod_2][0] = 1
                    PM[l_index] = PM_pair[0][l_index]
                    PM[index] = PM_pair[1][l_index]
        else:  # frozen bit operation
            for l_index in range(L):
                if active_path[l_index] == 0:
                    continue
                # 如果硬判决结果与冻结bit不一致，则要加惩罚值
                if soft_info[l_index][0] < 0 and sig_in_all[phi] == 0:
                    PM[l_index] -= soft_info[l_index][0]
                if soft_info[l_index][0] >= 0 and sig_in_all[phi] == 1:
                    PM[l_index] += soft_info[l_index][0]
                hard_info[2 * l_index + phi_mod_2][0] = sig_in_all[phi]
                u_hat[l_index][phi] = sig_in_all[phi]
        for l_index in range(L):
            if active_path[l_index] == 0:
                continue
            if phi_mod_2 == 1 and phi != N - 1:
                bit_layer = bit_layer_vec[phi]
                for i_layer in range(bit_layer):
                    index_1 = lambda_offset[i_layer]
                    index_2 = lambda_offset[i_layer + 1]
                    lazy_copy_value = lazy_copy[l_index][i_layer]
                    for beta in range(index_1 - 1, index_2 - 1):
                        hard_info[2 * l_index + 1][beta + index_1] = (
                                (hard_info[2 * lazy_copy_value][beta] + hard_info[2 * l_index + 1][beta]) % 2)
                        hard_info[2 * l_index + 1][beta + index_2] = hard_info[2 * l_index + 1][beta]
                index_1 = lambda_offset[bit_layer]
                index_2 = lambda_offset[bit_layer + 1]
                lazy_copy_value = lazy_copy[l_index][bit_layer]
                # 有优化下面的for循环的下标：
                for beta in range(index_1 - 1, index_2 - 1):
                    hard_info[2 * l_index][beta + index_1] = (hard_info[2 * lazy_copy_value][beta] +
                                                              hard_info[2 * l_index + 1][beta]) % 2
                    hard_info[2 * l_index][beta + index_2] = hard_info[2 * l_index + 1][beta]
        # lazy copy (update lazy_copy)
        if phi < N - 1:
            for i_layer in range(1, llr_layer_vec[phi + 1] + 2):
                for l_index in range(L):
                    # if active_path[l_index] == 0:   # 这个if可加可不加
                    #     continue
                    lazy_copy[l_index][i_layer - 1] = l_index
    # path selection, 输出PM最小的路径结果
    select_List = -1
    min_PM = np.finfo(float).max
    for i in range(L):
        if PM[i] < min_PM:
            select_List = i
            min_PM = PM[i]
    # for i in range(L):
    #     print("sig_recover[%d] = " % i, u_hat[i][select_index])
    #     print("PM[%d] = " % i, PM[i])
    sig_recover = u_hat[select_List][select_index]
    return sig_recover

def CASCL_decoder(N, llr, select_index, freeze_index, frozen_bits, L, crc,):
    # N = len(llr)  # 想一想如果发生删除错误，会怎么样？
    if N != len(llr):
        print("出错啦！N != len(llr)")
        print("N = %d, len(llr) = %d" % (N, len(llr)))
    n = int(math.log(N, 2))
    # n = math.log(N, 2)  # 这样会不会更好一点，可以处理llr少1，修复434行的bug，也有风险，如果发生的插入错误呢，所以N得输入才对
    # llr = SigReceive2llr(SigReceive)  # llr的转换放外面比较好
    frozen_flags = np.zeros(N, dtype=int)
    frozen_flags[freeze_index] = 1
    sig_in_all = np.zeros(N, dtype=int)
    sig_in_all[freeze_index] = frozen_bits  # 用sig_in_all把冻结bits和其位置存下来
    lazy_copy = np.full([L, n], -1)
    soft_info = np.zeros([L, N - 1], dtype=np.float64)
    # hard_info = np.zeros([2 * L, N - 1], dtype=int)
    hard_info = np.zeros([2 * L, 2 * N - 1], dtype=int)  # 要输出X_hat，hard_info就得大点
    u_hat = np.zeros([L, N], dtype=int)
    PM = np.zeros(L, dtype=np.float64)   # path metrics
    active_path = np.zeros(L, dtype=int)  # Indicate if the path is active. '1'→active; '0' otherwise.
    # initialize
    active_path[0] = 1
    lazy_copy[0][:] = 0
    lambda_offset = power(2, np.arange(0, n + 1))
    # print("lambda_offset:", lambda_offset)
    llr_layer_vec = get_llr_layer(N)
    bit_layer_vec = get_bit_layer(N)
    # decoding starts
    for phi in range(N):
        layer = llr_layer_vec[phi]
        phi_mod_2 = phi % 2
        for l_index in range(L):  # List的编号为0,1,2,...,L-1
            if active_path[l_index] == 0:
                continue
            if phi == 0:
                index_1 = lambda_offset[n - 1]
                for beta in range(index_1):
                    soft_info[l_index][beta + index_1 - 1] = np.sign(llr[beta]) * np.sign(llr[beta + index_1]) * \
                                                    min(abs(llr[beta]), abs(llr[beta + index_1]))
                for i_layer in range(n - 2, -1, -1):
                    index_1 = lambda_offset[i_layer]
                    index_2 = lambda_offset[i_layer + 1]
                    for beta in range(index_1 - 1, index_2 - 1):
                        soft_info[l_index][beta] = (np.sign(soft_info[l_index][beta + index_1]) * np.sign(soft_info[l_index][beta + index_2])
                                           * min(abs(soft_info[l_index][beta + index_1]), abs(soft_info[l_index][beta + index_2])))
            elif phi == N // 2:
                index_1 = lambda_offset[n - 1]
                for beta in range(index_1):
                    x_tmp = hard_info[2 * l_index][beta + index_1 - 1]
                    soft_info[l_index][beta + index_1 - 1] = (1 - 2 * x_tmp) * llr[beta] + llr[beta + index_1]
                for i_layer in range(n - 2, -1, -1):
                    index_1 = lambda_offset[i_layer]
                    index_2 = lambda_offset[i_layer + 1]
                    for beta in range(index_1 - 1, index_2 - 1):
                        soft_info[l_index][beta] = (np.sign(soft_info[l_index][beta + index_1]) * np.sign(soft_info[l_index][beta + index_2])
                                           * min(abs(soft_info[l_index][beta + index_1]), abs(soft_info[l_index][beta + index_2])))
            else:
                index_1 = lambda_offset[layer]
                index_2 = lambda_offset[layer + 1]
                lazy_copy_value = lazy_copy[l_index][layer + 1]  # 偶尔会看到这行代码报错（看见了2次）, index 5 is out of bounds for axis 0 with size 5
                # 有优化下面的for循环的下标：
                for beta in range(index_1 - 1, index_2 - 1):
                    soft_info[l_index][beta] = (
                                (1 - 2 * hard_info[2 * l_index][beta]) * soft_info[lazy_copy_value][beta + index_1] +
                                soft_info[lazy_copy_value][beta + index_2])
                for i_layer in range(layer - 1, -1, -1):
                    index_1 = lambda_offset[i_layer]
                    index_2 = lambda_offset[i_layer + 1]
                    for beta in range(index_1 - 1, index_2 - 1):
                        soft_info[l_index][beta] = (np.sign(soft_info[l_index][beta + index_1]) * np.sign(soft_info[l_index][beta + index_2])
                                           * min(abs(soft_info[l_index][beta + index_1]), abs(soft_info[l_index][beta + index_2])))
        if frozen_flags[phi] == 0:  # if now we decode an unfrozen bit
            PM_pair = np.full((2, L), np.finfo(float).max)
            for l_index in range(L):    # 感觉是soft_info[l_index][0]的计算有问题。但是看起来上面的计算，也没有问题
                if active_path[l_index] == 0:
                    continue
                if soft_info[l_index][0] >= 0:
                    PM_pair[0][l_index] = PM[l_index]
                    PM_pair[1][l_index] = PM[l_index] + soft_info[l_index][0]
                else:
                    PM_pair[0][l_index] = PM[l_index] - soft_info[l_index][0]
                    PM_pair[1][l_index] = PM[l_index]
            middle = min(2 * sum(active_path), L)
            # print("middle = ", middle)
            PM_sort = sorted(PM_pair.flatten())  # 可以
            PM_cv = PM_sort[middle]
            # PM_cv = PM_sort[middle - 1]
            compare = np.zeros([2, L], dtype=int)
            for i in range(2):
                for j in range(L):
                    if PM_pair[i][j] < PM_cv:    # 这里一定要改成<，不然会报错。这时，上面PM_cv就应该等于PM_sort[middle]，即向后移一位
                        compare[i][j] = 1
            kill_index = np.full(L, L)  # 这里感觉比较怪，会不会有问题
            kill_cnt = -1
            for i in range(L):
                if (compare[0][i] == 0) and (compare[1][i] == 0):  # 这里没有问题，也会把active_path=0的list压入栈中
                    active_path[i] = 0
                    kill_cnt += 1
                    kill_index[kill_cnt] = i
            for l_index in range(L):
                if active_path[l_index] == 0:
                    continue
                path_state = compare[0][l_index] * 2 + compare[1][l_index]
                # path_state can equal to 0, but in this case we do no operation.
                if path_state == 1:  # 0, 1 , 即只保存向1走的这条路
                    u_hat[l_index][phi] = 1
                    hard_info[2 * l_index + phi_mod_2][0] = 1
                    PM[l_index] = PM_pair[1][l_index]
                elif path_state == 2:   # 1, 0, 即只保存向0走的这条路
                    u_hat[l_index][phi] = 0
                    hard_info[2 * l_index + phi_mod_2][0] = 0
                    PM[l_index] = PM_pair[0][l_index]
                elif path_state == 3:  # 1, 1, 同时保存向0和1两条路
                    index = kill_index[kill_cnt]
                    kill_cnt -= 1
                    active_path[index] = 1
                    # lazy copy
                    lazy_copy[index][:] = lazy_copy[l_index][:]
                    u_hat[index][:] = u_hat[l_index][:]
                    u_hat[l_index][phi] = 0  # 原来的List走0这条路
                    u_hat[index][phi] = 1  # 克隆的新list走1这条路
                    hard_info[2 * l_index + phi_mod_2][0] = 0
                    hard_info[2 * index + phi_mod_2][0] = 1
                    PM[l_index] = PM_pair[0][l_index]
                    PM[index] = PM_pair[1][l_index]
        else:  # frozen bit operation
            for l_index in range(L):
                if active_path[l_index] == 0:
                    continue
                # 如果硬判决结果与冻结bit不一致，则要加惩罚值
                if soft_info[l_index][0] < 0 and sig_in_all[phi] == 0:
                    PM[l_index] -= soft_info[l_index][0]
                if soft_info[l_index][0] >= 0 and sig_in_all[phi] == 1:
                    PM[l_index] += soft_info[l_index][0]
                hard_info[2 * l_index + phi_mod_2][0] = sig_in_all[phi]
                u_hat[l_index][phi] = sig_in_all[phi]
        for l_index in range(L):
            if active_path[l_index] == 0:
                continue
            # if phi_mod_2 == 1 and phi != N - 1:
            if phi_mod_2 == 1:  # 希望能输出x_hat
                bit_layer = bit_layer_vec[phi]
                for i_layer in range(bit_layer):
                    index_1 = lambda_offset[i_layer]
                    index_2 = lambda_offset[i_layer + 1]
                    lazy_copy_value = lazy_copy[l_index][i_layer]
                    for beta in range(index_1 - 1, index_2 - 1):
                        hard_info[2 * l_index + 1][beta + index_1] = (
                                (hard_info[2 * lazy_copy_value][beta] + hard_info[2 * l_index + 1][beta]) % 2)
                        hard_info[2 * l_index + 1][beta + index_2] = hard_info[2 * l_index + 1][beta]
                index_1 = lambda_offset[bit_layer]
                index_2 = lambda_offset[bit_layer + 1]
                lazy_copy_value = lazy_copy[l_index][bit_layer]
                # 有优化下面的for循环的下标：
                for beta in range(index_1 - 1, index_2 - 1):
                    hard_info[2 * l_index][beta + index_1] = (hard_info[2 * lazy_copy_value][beta] +
                                                              hard_info[2 * l_index + 1][beta]) % 2
                    hard_info[2 * l_index][beta + index_2] = hard_info[2 * l_index + 1][beta]
        # lazy copy (update lazy_copy)
        if phi < N - 1:
            for i_layer in range(1, llr_layer_vec[phi + 1] + 2):
                for l_index in range(L):
                    # if active_path[l_index] == 0:   # 这个if可加可不加
                    #     continue
                    lazy_copy[l_index][i_layer - 1] = l_index
    # path selection, 输出PM最小的路径结果
    # select_List = -1
    # min_PM = np.finfo(float).max
    # for i in range(L):
    #     if PM[i] < min_PM:
    #         select_List = i
    #         min_PM = PM[i]

    idx_list = [k for k in range(L)]
    idx_list.sort(key=lambda k: PM[k])
    u_hat[:] = [u_hat[k] for k in idx_list[:]]  # 按PM值的大小对u_hat重新排序
    x_hat = []
    for i in range(L):
        x_hat.append(hard_info[2 * i].copy())
    x_hat[:] = [x_hat[k] for k in idx_list[:]]  # 按PM值的大小对x_hat重新排序
    flag = False
    i = 0  # 这个最好不要省
    for i in range(L):
        if crc_detect(u_hat[i][select_index], crc):
            flag = True
            break
    if flag:
        sig_recover_with_crc = u_hat[i][select_index]
        x_hat_recover = x_hat[i][-N:]
    else:
        sig_recover_with_crc = u_hat[0][select_index]
        x_hat_recover = x_hat[0][-N:]
    sig_recover = sig_recover_with_crc[0:-(len(crc) - 1)]
    return sig_recover, x_hat_recover
    # return sig_recover
