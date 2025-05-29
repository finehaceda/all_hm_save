import numpy as np
from .glob_var import (N_data)
from .frozen_bits import frozen_bits_205

#注意在编码部分，下标从0开始
def crc_encode(data_bits, divisor_bits):
    # data_bits = np.array(data_bits)
    # divisor_bits = np.array(divisor_bits)  # Ensure divisor_bits is a numpy array
    padded_data = np.concatenate((data_bits, np.zeros(len(divisor_bits) - 1, dtype=int)))
    for i in range(len(data_bits)):
        if padded_data[i] == 1:
            padded_data[i:i+len(divisor_bits)] ^= divisor_bits
    crc_code = padded_data[-len(divisor_bits)+1:]
    data_bits_crc = np.concatenate((data_bits, crc_code))
    return data_bits_crc

# 下面的编码算法是根据生成矩阵来做的，是正确的
def polar_encoder(u):
    # Encoding: x = u * FN.
    N = len(u)
    GN = get_GN(N)
    Y = np.dot(u, GN)
    x = np.mod(Y, 2).astype(int)
    return x

def get_GN(N):
    F = np.array([[1, 0], [1, 1]])
    FN = np.zeros((N, N))
    FN[:2, :2] = F
    for i in range(2, int(np.log2(N)) + 1):
        FN[:2**i, :2**i] = np.kron(FN[:2**(i-1), :2**(i-1)], F)
    return FN


# 确认，下面的编码算法是正确的，而且还很快，占用内存少
def Encoder4Polar(Miu):
    # parameter
    BL = len(Miu)
    Dim = int(np.log2(BL))
    Idx = np.arange(1, BL + 1)
    OH = np.arange(1, BL + 1, 2)
    EH = np.arange(2, BL + 1, 2)
    Temp = np.array(Miu)
    Temp1 = np.array(Miu)

    if Dim == 0:
        X = Miu
    else:
        for I in range(1, Dim + 1):
            # 改动，感觉这一部分也没有问题
            OIdx = [Idx[i - 1] - 1 for i in OH]
            EIdx = [Idx[i - 1] - 1 for i in EH]
            Temp1[EIdx] = Temp[EIdx]
            Temp1[OIdx] = np.bitwise_xor(Temp[EIdx], Temp[OIdx])
            Idx = np.reshape(Idx, (2, int(BL / 2)))
            Idx = np.reshape(Idx.T, BL)
            Temp = Temp1.copy()

        X = Temp1.copy()

    return X

def polar_encode(bits_arr, select_index, freeze_index, frozen_bits, N, crc):
    # print("bits_arr:", bits_arr, sum(bits_arr))
    crc_sig_in = crc_encode(bits_arr, crc)
    # 注意：当CRC_16(只要是偶数),得到的crc_sig_in中1的个数总是偶数
    # print("crc_sig_in:", crc_sig_in, sum(crc_sig_in) % 2)
    # print("crc_sig_in的长度：", len(crc_sig_in))
    # 注意到我写的CRC-SCL不需要BitReverse
    # encode
    SigInAll = np.zeros(N, dtype=int)  # 真实数据从0开始存放
    SigInAll[select_index] = crc_sig_in[0:len(select_index)]
    SigInAll[freeze_index] = frozen_bits
    encoded_list = Encoder4Polar(SigInAll)
    # encoded_list = polar_encoder(SigInAll)  # 这个方法与上面的相比，真的要慢很多哦，结果是一样的
    return encoded_list
'''
# 这个编码函数会不会有问题，就是下标问题，感觉没有问题
def polar_encode_data(bits_arr):
    # print("bits_arr:", bits_arr, sum(bits_arr))
    crc_sig_in = crc_encode(bits_arr, crc)
    # 注意：当CRC_16(只要是偶数),得到的crc_sig_in中1的个数总是偶数
    # print("crc_sig_in:", crc_sig_in, sum(crc_sig_in) % 2)
    # print("crc_sig_in的长度：", len(crc_sig_in))
    # encode
    SigInAll = np.zeros(N_data, dtype=int)  # 真实数据从0开始存放
    SigInAll[select_index_data] = crc_sig_in[0:len(select_index_data)]
    # SigInAll[freeze_index_data] = frozen_bits_data
    SigInAll[freeze_index_data] = frozen_bits_205
    # 上面部分没有问题，包括select_index_data和freeze_index_data
    # SigInAllRV = SigInAll[ReverseIndex_data]  # 我们试试用Bit reverse
    encoded_list = Encoder4Polar(SigInAll)
    # encoded_list = polar_encoder(SigInAll)  # 这个方法与上面的相比，真的要慢很多哦，结果是一样的
    return encoded_list

def polar_encode_index(index_arr):
    # 注意：当CRC_16(只要是偶数),得到的crc_sig_in中1的个数总是偶数
    # crc_sig_in = crc_encode(index_arr, crc_poly_8)
    crc_sig_in = crc_encode(index_arr, crc)
    # encode
    SigInAll = np.zeros(N_index, dtype=int)  # 真实数据从0开始存放
    SigInAll[select_index_index] = crc_sig_in[0:len(select_index_index)]
    SigInAll[freeze_index_index] = frozen_bits_index
    # SigInAll[freeze_index_index] = frozen_bits_66
    # SigInAllRV = SigInAll[ReverseIndex_index]  # 我们试试用Bit reverse
    encoded_index_list = Encoder4Polar(SigInAll)
    # encoded_index_list = polar_encoder(SigInAll)
    return encoded_index_list
'''
def polar_encode_data_no_crc(bits_arr, select_index, freeze_index, frozen_bits):
    SigInAll = np.zeros(N_data, dtype=int)  # 真实数据从0开始存放
    SigInAll[select_index] = bits_arr[0:len(select_index)]
    SigInAll[freeze_index] = frozen_bits
    # SigInAllRV = SigInAll[ReverseIndex_data]
    encoded_list = Encoder4Polar(SigInAll)
    # encoded_list = Encoder4Polar(SigInAllRV)
    return encoded_list

