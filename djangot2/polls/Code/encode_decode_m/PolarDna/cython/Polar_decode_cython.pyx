import cython
import numpy as np
# cimport numpy as np
from libc.math cimport log, fabs, copysign, pow
# import math
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int crc_detect(int[::1] signal, int[::1] crc_poly):  # int[::1]内存中连续的1维内存视图
    cdef int signal_length = signal.shape[0]
    cdef int crc_poly_length = crc_poly.shape[0]
    cdef int i, j, my_sum = 0
    crc_remainder = np.zeros(crc_poly_length, dtype=np.intc)
    cdef int[::1] crc_remainder_view = crc_remainder
    cdef int crc_check = 0   # 0表示假

    for i in range(signal_length):
        for j in range(crc_poly_length - 1):
            crc_remainder_view[j] = crc_remainder_view[j + 1]
        crc_remainder_view[crc_poly_length - 1] = signal[i]
        if crc_remainder_view[0] == 1:
            for i in range(crc_poly_length):
                crc_remainder_view[i] = (crc_remainder_view[i] + crc_poly[i]) % 2
    # crc_check = sum(crc_remainder) == 0
    for i in range(crc_poly_length):
        my_sum += crc_remainder_view[i]
    if my_sum == 0:
        crc_check = 1   # 1 表示真
    return crc_check

@cython.boundscheck(False)
@cython.wraparound(False)
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

@cython.boundscheck(False)
@cython.wraparound(False)
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

@cython.boundscheck(False)
@cython.wraparound(False)
def SigReceive2llr_second_layer(X_hat_1, SigReceive_Y1, SigReceive_Y2, Y2_bases_phred):
    # 支持质量值
    # 因为双层极化方案中，第二层从碱基可靠度换算成01可靠度的公式不同。Y2_bases_phred就是对应碱基的可靠度
    N = len(SigReceive_Y1)
    Prob0 = np.ones(N, dtype=np.float64)
    Prob1 = np.ones(N, dtype=np.float64)
    temp = np.where(SigReceive_Y2 == 1)[0]
    temp = np.array([index for index in temp if SigReceive_Y1[index] == X_hat_1[index]])
    if temp.size > 0:  # 当temp不为空时，才执行下面的代码，或者会报错
        Prob0[temp] = (1 - Y2_bases_phred[temp]) / 3
        Prob1[temp] = Y2_bases_phred[temp]

    temp = np.where(SigReceive_Y2 == 0)[0]
    temp = np.array([index for index in temp if SigReceive_Y1[index] == X_hat_1[index]])
    if temp.size > 0:  # 当temp不为空时，才执行下面的代码，或者会报错
        Prob0[temp] = Y2_bases_phred[temp]
        Prob1[temp] = (1 - Y2_bases_phred[temp]) / 3
    llr = np.log2(Prob0 / Prob1)
    return llr


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int[::1] get_llr_layer(int normal):
    layer_vec = np.zeros(normal, dtype=np.intc)
    cdef int[::1] layer_vec_view = layer_vec
    cdef int phi = 0
    cdef int psi, layer
    for phi in range(normal - 1):
        psi = phi + 1
        layer = 0
        while psi % 2 == 0:
            psi = int(psi / 2)
            layer = layer + 1
        layer_vec_view[phi + 1] = layer
    return layer_vec_view  # 直接返回内存视图，可能更好

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int[::1] get_bit_layer(int normal):
    layer_vec = np.zeros(normal, dtype=np.intc)
    cdef int[::1] layer_vec_view = layer_vec
    cdef int phi = 0
    cdef int psi, layer
    for phi in range(normal):
        psi = int(phi / 2)
        layer = 0
        while psi % 2 == 1:
            psi = int(psi / 2)
            layer = layer + 1
        layer_vec_view[phi] = layer
    return layer_vec_view

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int[::1] power(int base, int[::1] pow_range):
    results = np.empty(pow_range.shape[0], dtype=np.intc)
    cdef int[::1] results_view = results
    cdef int i = 0
    for i in range(pow_range.shape[0]):
        results_view[i] = int(pow(base, pow_range[i]))
    return results_view


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double my_min(double a, double b) :
    if a < b:
        return a
    else:
        return b


@cython.boundscheck(False)   # Deactivate bounds checking
@cython.wraparound(False)  # Deactivate negative indexing.
def CASCL_decoder(int N, double[::1] llr, int[::1] select_index, int[::1] freeze_index, int[::1] frozen_bits, int L, int[::1] crc):
    if N != llr.shape[0]:
        print("出错啦！ N != len(llr)")
        print("N = %d, len(llr) = %d" % (N, llr.shape[0]))
    cdef int n = int(log(N) / log(2))   # 求以2为底的对数值
    # print("N = %d, n = %d" % (N, n))
    frozen_flags = np.zeros(N, dtype=np.intc)
    cdef int[::1] frozen_flags_view = frozen_flags
    sig_in_all = np.zeros(N, dtype=np.intc)
    cdef int[::1] sig_in_all_view = sig_in_all
    cdef int i, f
    for i in range(freeze_index.shape[0]):
        f = freeze_index[i]
        frozen_flags_view[f] = 1
        sig_in_all_view[f] = frozen_bits[i]
    lazy_copy = np.full([L, n], -1, dtype=np.intc)
    cdef int[:,::1] lazy_copy_view = lazy_copy   # 二维连续内存视图
    soft_info = np.zeros([L, N - 1], dtype=np.float64)
    cdef double[:,::1] soft_info_view = soft_info
    hard_info = np.zeros([2 * L, 2 * N - 1], dtype=np.intc)
    cdef int[:,::1] hard_info_view = hard_info
    u_hat = np.zeros([L, N], dtype=np.intc)
    cdef int[:,::1] u_hat_view = u_hat
    PM = np.zeros(L, dtype=np.float64)
    cdef double[::1] PM_view = PM
    active_path = np.zeros(L, dtype=np.intc)
    cdef int[::1] active_path_view = active_path
    active_path_view[0] = 1
    lazy_copy_view[0][:] = 0
    cdef int[::1] lambda_offset = power(2, np.arange(0, n + 1, dtype=np.intc))
    cdef int[::1] llr_layer_vec = get_llr_layer(N)
    cdef int[::1] bit_layer_vec = get_bit_layer(N)
    PM_pair = np.full((2, L), np.finfo(float).max)
    cdef double[:,::1] PM_pair_view = PM_pair
    compare = np.zeros([2, L], dtype=np.intc)
    cdef int[:,::1] compare_view = compare
    kill_index = np.full(L, L, dtype=np.intc)
    cdef int[::1] kill_index_view = kill_index
    cdef int kill_cnt
    # decoding starts
    cdef int phi, layer, phi_mod_2, l_index, index_1, index_2, beta, x_tmp, i_layer, lazy_copy_value
    cdef int bit_layer, middle, j, sig_recover_length, path_state, index
    cdef double PM_cv
    for phi in range(N):
        layer = llr_layer_vec[phi]
        phi_mod_2 = phi % 2
        for l_index in range(L):
            if active_path_view[l_index] == 0:
                continue
            if phi == 0:
                index_1 = lambda_offset[n - 1]
                for beta in range(index_1):
                    # soft_info_view[l_index][beta + index_1 - 1] = np.sign(llr[beta]) * np.sign(llr[beta + index_1]) * \
                    #                                 min(abs(llr[beta]), abs(llr[beta + index_1]))
                    soft_info_view[l_index][beta + index_1 - 1] = copysign(1, llr[beta]) * copysign(1, llr[beta + index_1]) * \
                                                                  my_min(fabs(llr[beta]), fabs(llr[beta + index_1]))
                for i_layer in range(n - 2, -1, -1):
                    index_1 = lambda_offset[i_layer]
                    index_2 = lambda_offset[i_layer + 1]
                    for beta in range(index_1 - 1, index_2 - 1):
                        # soft_info_view[l_index][beta] = \
                        #     (np.sign(soft_info_view[l_index][beta + index_1]) *
                        #      np.sign(soft_info_view[l_index][beta + index_2]) *
                        #      min(abs(soft_info_view[l_index][beta + index_1]),abs(soft_info_view[l_index][beta + index_2])))
                        soft_info_view[l_index][beta] = \
                            (copysign(1, soft_info_view[l_index][beta + index_1]) *
                             copysign(1, soft_info_view[l_index][beta + index_2]) *
                             my_min(fabs(soft_info_view[l_index][beta + index_1]),
                                 fabs(soft_info_view[l_index][beta + index_2])))
            elif phi == N // 2:
                index_1 = lambda_offset[n - 1]
                for beta in range(index_1):
                    x_tmp = hard_info_view[2 * l_index][beta + index_1 - 1]
                    soft_info_view[l_index][beta + index_1 - 1] = (1 - 2 * x_tmp) * llr[beta] + llr[beta + index_1]
                for i_layer in range(n - 2, -1, -1):
                    index_1 = lambda_offset[i_layer]
                    index_2 = lambda_offset[i_layer + 1]
                    for beta in range(index_1 - 1, index_2 - 1):
                        # soft_info_view[l_index][beta] = (np.sign(soft_info_view[l_index][beta + index_1]) * np.sign(
                        #     soft_info_view[l_index][beta + index_2])
                        #                             * min(abs(soft_info_view[l_index][beta + index_1]),
                        #                                   abs(soft_info_view[l_index][beta + index_2])))
                        soft_info_view[l_index][beta] = (copysign(1, soft_info_view[l_index][beta + index_1]) * copysign(
                            1, soft_info_view[l_index][beta + index_2])
                                                         * my_min(fabs(soft_info_view[l_index][beta + index_1]),
                                                               fabs(soft_info_view[l_index][beta + index_2])))
            else:
                index_1 = lambda_offset[layer]
                index_2 = lambda_offset[layer + 1]
                lazy_copy_value = lazy_copy_view[l_index][layer + 1]
                for beta in range(index_1 - 1, index_2 - 1):
                    soft_info_view[l_index][beta] = (
                            (1 - 2 * hard_info_view[2 * l_index][beta]) * soft_info_view[lazy_copy_value][beta + index_1] +
                            soft_info_view[lazy_copy_value][beta + index_2])
                for i_layer in range(layer - 1, -1, -1):
                    index_1 = lambda_offset[i_layer]
                    index_2 = lambda_offset[i_layer + 1]
                    for beta in range(index_1 - 1, index_2 - 1):
                        # soft_info_view[l_index][beta] = (np.sign(soft_info_view[l_index][beta + index_1]) * np.sign(
                        #     soft_info_view[l_index][beta + index_2])
                        #                             * min(abs(soft_info_view[l_index][beta + index_1]),
                        #                                   abs(soft_info_view[l_index][beta + index_2])))
                        soft_info_view[l_index][beta] = (copysign(1, soft_info_view[l_index][beta + index_1]) * copysign(
                            1, soft_info_view[l_index][beta + index_2])
                                                         * my_min(fabs(soft_info_view[l_index][beta + index_1]),
                                                               fabs(soft_info_view[l_index][beta + index_2])))

        if frozen_flags_view[phi] == 0:  # if now we decode an unfrozen bit
            # 更新PM_pair的值
            for i in range(2):
                for j in range(L):
                    PM_pair_view[i][j] = np.finfo(float).max
            for l_index in range(L):  # 感觉是soft_info[l_index][0]的计算有问题。但是看起来上面的计算，也没有问题
                if active_path_view[l_index] == 0:
                    continue
                if soft_info_view[l_index][0] >= 0:
                    PM_pair_view[0][l_index] = PM_view[l_index]
                    PM_pair_view[1][l_index] = PM_view[l_index] + soft_info_view[l_index][0]
                else:
                    PM_pair_view[0][l_index] = PM_view[l_index] - soft_info_view[l_index][0]
                    PM_pair_view[1][l_index] = PM_view[l_index]
            middle = min(2 * sum(active_path), L)
            PM_sort = sorted(PM_pair.flatten())  # 可以
            PM_cv = PM_sort[middle]
            # for i in range(2):
            #     for j in range(L):
            #         compare_view[i][j] = 0  # 更新compare的值
            for i in range(2):
                for j in range(L):
                    if PM_pair_view[i][j] < PM_cv:  # 这里一定要改成<，不然会报错。这时，上面PM_cv就应该等于PM_sort[middle]，即向后移一位
                        compare_view[i][j] = 1
                    else:
                        compare_view[i][j] = 0
            for i in range(L):
                kill_index_view[i] = L  # 这里要更新kill_index的值
            kill_cnt = -1
            for i in range(L):
                # print("i = %d" % i)
                if (compare_view[0][i] == 0) and (compare_view[1][i] == 0):  # 这里没有问题，也会把active_path=0的list压入栈中
                    active_path_view[i] = 0
                    kill_cnt = kill_cnt + 1
                    kill_index_view[kill_cnt] = i
            for l_index in range(L):
                if active_path_view[l_index] == 0:
                    continue
                path_state = compare_view[0][l_index] * 2 + compare_view[1][l_index]
                # path_state can equal to 0, but in this case we do no operation.
                if path_state == 1:  # 0, 1 , 即只保存向1走的这条路
                    u_hat_view[l_index][phi] = 1
                    hard_info_view[2 * l_index + phi_mod_2][0] = 1
                    PM_view[l_index] = PM_pair_view[1][l_index]
                elif path_state == 2:  # 1, 0, 即只保存向0走的这条路
                    u_hat_view[l_index][phi] = 0
                    hard_info_view[2 * l_index + phi_mod_2][0] = 0
                    PM_view[l_index] = PM_pair_view[0][l_index]
                elif path_state == 3:  # 1, 1, 同时保存向0和1两条路
                    index = kill_index_view[kill_cnt]
                    # if index == 32:
                    #     print("index =", index)
                    #     print("kill_cnt =", kill_cnt)
                    kill_cnt = kill_cnt - 1
                    active_path_view[index] = 1
                    # lazy copy
                    lazy_copy_view[index][:] = lazy_copy_view[l_index][:]
                    u_hat_view[index][:] = u_hat_view[l_index][:]
                    u_hat_view[l_index][phi] = 0  # 原来的List走0这条路
                    u_hat_view[index][phi] = 1  # 克隆的新list走1这条路
                    hard_info_view[2 * l_index + phi_mod_2][0] = 0
                    hard_info_view[2 * index + phi_mod_2][0] = 1
                    PM_view[l_index] = PM_pair_view[0][l_index]
                    PM_view[index] = PM_pair_view[1][l_index]
        else:  # frozen bit operation
            for l_index in range(L):
                if active_path_view[l_index] == 0:
                    continue
                # 如果硬判决结果与冻结bit不一致，则要加惩罚值
                if soft_info_view[l_index][0] < 0 and sig_in_all_view[phi] == 0:
                    PM_view[l_index] -= soft_info_view[l_index][0]
                if soft_info_view[l_index][0] >= 0 and sig_in_all_view[phi] == 1:
                    PM_view[l_index] += soft_info_view[l_index][0]
                hard_info_view[2 * l_index + phi_mod_2][0] = sig_in_all_view[phi]
                u_hat_view[l_index][phi] = sig_in_all_view[phi]
        for l_index in range(L):
            if active_path_view[l_index] == 0:
                continue
            # if phi_mod_2 == 1 and phi != N - 1:
            if phi_mod_2 == 1:  # 希望能输出x_hat
                bit_layer = bit_layer_vec[phi]
                for i_layer in range(bit_layer):
                    index_1 = lambda_offset[i_layer]
                    index_2 = lambda_offset[i_layer + 1]
                    lazy_copy_value = lazy_copy_view[l_index][i_layer]
                    for beta in range(index_1 - 1, index_2 - 1):
                        hard_info_view[2 * l_index + 1][beta + index_1] = (
                                (hard_info_view[2 * lazy_copy_value][beta] + hard_info_view[2 * l_index + 1][beta]) % 2)
                        hard_info_view[2 * l_index + 1][beta + index_2] = hard_info_view[2 * l_index + 1][beta]
                index_1 = lambda_offset[bit_layer]
                index_2 = lambda_offset[bit_layer + 1]
                lazy_copy_value = lazy_copy_view[l_index][bit_layer]
                # 有优化下面的for循环的下标：
                for beta in range(index_1 - 1, index_2 - 1):
                    hard_info_view[2 * l_index][beta + index_1] = (hard_info_view[2 * lazy_copy_value][beta] +
                                                              hard_info_view[2 * l_index + 1][beta]) % 2
                    hard_info_view[2 * l_index][beta + index_2] = hard_info_view[2 * l_index + 1][beta]
        # lazy copy (update lazy_copy)
        if phi < N - 1:
            for i_layer in range(1, llr_layer_vec[phi + 1] + 2):
                for l_index in range(L):
                    # if active_path[l_index] == 0:   # 这个if可加可不加
                    #     continue
                    lazy_copy_view[l_index][i_layer - 1] = l_index
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
        x_hat_start = x_hat[i].shape[0] - N
        x_hat_recover = x_hat[i][x_hat_start:]
    else:
        sig_recover_with_crc = u_hat[0][select_index]
        x_hat_start = x_hat[0].shape[0] - N
        x_hat_recover = x_hat[0][x_hat_start:]
    sig_recover_length = sig_recover_with_crc.shape[0] - crc.shape[0] + 1
    sig_recover = sig_recover_with_crc[0:sig_recover_length]   # 已经禁用了负索引
    return sig_recover, x_hat_recover ,flag
