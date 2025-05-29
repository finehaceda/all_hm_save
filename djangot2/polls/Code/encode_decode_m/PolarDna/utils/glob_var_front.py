import os

import numpy as np
from scipy.io import loadmat
import math
from .select_good_channels_for_polar import SelectGoodChannels4Polar, SelectGoodChannels4Polar_two_layer_npy
from .Bitreverse import BitReverse
from .frozen_bits import (frozen_bits_318, frozen_bits_297, frozen_bits_264, frozen_bits_578, frozen_bits_34,
                               frozen_bits_115, frozen_bits_110, frozen_bits_268, frozen_bits_200, frozen_bits_210,
                               frozen_bits_30, frozen_bits_73, frozen_bits_100, frozen_bits_95, frozen_bits_159,
                               frozen_bits_148, frozen_bits_232, frozen_bits_182, frozen_bits_230, frozen_bits_220,
                               frozen_bits_125, frozen_bits_131, frozen_bits_174, frozen_bits_154, frozen_bits_90,
                               frozen_bits_85, frozen_bits_138, frozen_bits_128, frozen_bits_79, frozen_bits_74,
                               frozen_bits_118, frozen_bits_108, frozen_bits_133, frozen_bits_113, frozen_bits_69,
                               frozen_bits_64,
                               frozen_bits_97, frozen_bits_87, frozen_bits_92, frozen_bits_59, frozen_bits_54,
                               frozen_bits_77,
                               frozen_bits_29, frozen_bits_44, frozen_bits_67, frozen_bits_38, frozen_bits_56,
                               frozen_bits_16,
                               frozen_bits_46, frozen_bits_49, frozen_bits_20)

p_bit = 0.00667
p_basic_group = 0.010
List_data, List_index_1, List_index_2 = 16, 16, 32  # List
list_16, list_32 = 16, 32  # 用这个指定list更清楚
larger = 4  # larger_list 中 larger_list = larger * 原list
k_data, k_index_1, k_index_2 = 11, 8, 6    # 在0、1bit替换错误率为0.00667（对应于碱基替换错误率为1%）
N_data = 2 ** k_data
N_index_1 = 2 ** k_index_1
N_index_2 = 2 ** k_index_2
# 新的方案是全面模拟真实3代测序流程，加入短序列拼接，3代测序，拆分，清洗过程。由 4 or 3 个长300的DNA序列拼接而形成
segment_length = 242  # 实现块地址与块内地址分离，块地址最小编辑距离为5+块内地址最小编辑距离为3，块地址长08，块内地址长10，增加行向1位奇偶校验，保证每行异或运算后为0
# 太多地方用到segment_length了，用segment_length_parity表示新的，不含奇偶检验位的segment_length
parity_length = 2   # 采用 1 or 2 位的奇偶方案
segment_length_parity = segment_length - parity_length   # 感觉加入2位奇偶校验，更好，分别加偶数位的和奇数位的，且用偶（240）+奇（241）的格式，均保证异或后为0
# segment_length = 240  # 实现块地址与块内地址分离，块地址最小编辑距离为5+块内地址最小编辑距离为8，块地址长07，块内地址长13
# segment_length改变后，index_1和index_2，以及最后一个矩阵的index_1_last的frozen_bits的长度得变，要算一下
# segment_length_index = 144  # index部分，每条DNA序列长144个碱基
index_binary_length_1 = k_data + 1  # 表格index_1中的行号长12位，也即data部分的行号
index_binary_length_2 = k_index_1 + 1  # 表格index_2中的行号长10位，也即index_1的真实数据部分的行号
# crc_poly_5 = [1, 0, 0, 1, 0, 1]
# crc_poly_12 = [1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1]
crc_poly_4 = [1, 1, 1, 0, 1]
crc_poly_8 = [1, 1, 0, 0, 1, 0, 0, 0, 1]
crc_poly_16 = [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
crc_poly_16 = np.array(crc_poly_16, dtype=np.intc)
crc_poly_8 = np.array(crc_poly_8, dtype=np.intc)
crc_poly_4 = np.array(crc_poly_4, dtype=np.intc)
# 重要！当选CRC-16时，经过无论什么信息数据，添加CRC后，其中1的个数总是偶数，这样极化编码后第1项总是0（或1）.但选CRC-5就不会这样。
crc_data = crc_poly_16
crc_index_1 = crc_poly_4
crc_index_2 = crc_poly_4

# 对于二层方案，编码时一二层frozen_bits_data的长度是不同的
# 5%
# frozen_bits_data_1 = frozen_bits_113  # 这两层冻结bits的长度，可以在构造后再确认F
# frozen_bits_data_2 = frozen_bits_92  # 好像第2层的表现会差一点，可以考虑变化冻结bit的长度

# 调5bit试试
# frozen_bits_data_1 = frozen_bits_108  # 这两层冻结bits的长度，可以在构造后再确认
# frozen_bits_data_2 = frozen_bits_97  # 好像第2层的表现会差一点，可以考虑变化冻结bit的长度

# 6%
# frozen_bits_data_1 = frozen_bits_133  # 这两层冻结bits的长度，可以在构造后再确认
# frozen_bits_data_2 = frozen_bits_113
# 8%
# frozen_bits_data_1 = frozen_bits_174  # 这两层冻结bits的长度，可以在构造后再确认
# frozen_bits_data_2 = frozen_bits_154
# 10%
# frozen_bits_data_1 = frozen_bits_232  # 这两层冻结bits的长度，可以在构造后再确认
# frozen_bits_data_2 = frozen_bits_182
# 15%
# frozen_bits_data_1 = frozen_bits_318  # 这两层冻结bits的长度，可以在构造后再确认
# frozen_bits_data_2 = frozen_bits_297

# frozen_bits_index_1 = frozen_bits_578
# frozen_bits_index_2 = frozen_bits_34
frozen_bits_index_1 = frozen_bits_46
frozen_bits_index_2 = frozen_bits_49

# frozen_bits_index = frozen_bits_74  # 在二层方案中，用64极化，用crc4,此时frozen_bit长15，占23.4%
# print("frozen_bits_data_1的长度：", len(frozen_bits_data_1))
# print("frozen_bits_data_2的长度：", len(frozen_bits_data_2))
# print("frozen_bits_index的长度：", len(frozen_bits_index))

# frozen_bits_data_1 = np.array(frozen_bits_data_1, dtype=np.intc)
# frozen_bits_data_2 = np.array(frozen_bits_data_2, dtype=np.intc)
frozen_bits_index_1 = np.array(frozen_bits_index_1, dtype=np.intc)
frozen_bits_index_2 = np.array(frozen_bits_index_2, dtype=np.intc)
# 信道错误率为2%，frozen_bits_data长88，占4%，CRC长16,信息bits长2048 - 88 = 1960, 占95%
# matrix_row_num_data_1 = N_data - len(crc_data) + 1 - len(frozen_bits_data_1)
# matrix_row_num_data_2 = N_data - len(crc_data) + 1 - len(frozen_bits_data_2)
# index极化为64，真的不行。还得用128极化
# 二层方案中，真实行号占行数约43，但考虑到有随机数据，把行号占行数留到46，用crc_8,frozen_bits_index长74，占58%
matrix_row_num_index_1 = N_index_1 - len(crc_index_1) + 1 - len(frozen_bits_index_1)
more_data_length_1 = segment_length_parity % (2 * index_binary_length_1)  # 表示每行行号数据与segment_length_parity的差，这要用随机bits补上
segment_length_index_1 = segment_length_parity - more_data_length_1   # segment_length_index 表示对index表格数据进行切割的长度

matrix_row_num_index_2 = N_index_2 - len(crc_index_2) + 1 - len(frozen_bits_index_2)
more_data_length_2 = segment_length_parity % (2 * index_binary_length_2)  # 表示每行行号数据与segment_length_parity的差，这要用随机bits补上
segment_length_index_2 = segment_length_parity - more_data_length_2   # segment_length_index 表示对index表格数据进行切割的长度


Normal_data = list(range(1, N_data + 1))  # 感觉这个BitReverse没有用到啊？答：我写的CRC-SCL没有用到bitreverse,所以编码也不用
ReverseIndex_data = BitReverse(Normal_data)
Normal_index_1 = list(range(1, N_index_1 + 1))
ReverseIndex_index_1 = BitReverse(Normal_index_1)
Normal_index_2 = list(range(1, N_index_2 + 1))
ReverseIndex_index_2 = BitReverse(Normal_index_2)
osdir = './polls/Code/encode_decode_m/PolarDna/'
# osdir = './'
# print("ReverseIndex_data", ReverseIndex_data)
# print("ReverseIndex_index", ReverseIndex_index)

# 这两个排序文件没有问题

# path = osdir + './params/Index_two_layer_N=2048_QSC_p=0.010.npy'
path = osdir + './params/Index_two_layer_N=2048_QSC_p=0.010.npy'
# Index_data_two_layer = np.load(osdir + './params/Index_two_layer_N=2048_SC_p=0.010.npy')
Index_data_two_layer = np.load(path)
# print("我的构造是：", path)
# select_index_data_1, freeze_index_data_1, select_index_data_2, freeze_index_data_2 = (
#     SelectGoodChannels4Polar_two_layer_npy(Index_data_two_layer, matrix_row_num_data_1 + len(crc_data) - 1, matrix_row_num_data_2 + len(crc_data) - 1))
# select_index_data_1 = np.array(select_index_data_1, dtype=np.intc)
# freeze_index_data_1 = np.array(freeze_index_data_1, dtype=np.intc)
# select_index_data_2 = np.array(select_index_data_2, dtype=np.intc)
# freeze_index_data_2 = np.array(freeze_index_data_2, dtype=np.intc)


# Zn_index = loadmat(osdir + './params/N=1024_BSC_p=0.00667_Pe.mat')
# Zn_index = loadmat(osdir + './params/N=512_BSC_p=0.00667_Pe.mat')
Zn_index = loadmat(osdir + './params/N=256_BSC_p=0.00667_Pe.mat')
select_index_index_1, freeze_index_index_1, _ = SelectGoodChannels4Polar(Zn_index['Pe'].flatten(), matrix_row_num_index_1 + len(crc_index_1) - 1)
select_index_index_1 = np.array(select_index_index_1, dtype=np.intc)
freeze_index_index_1 = np.array(freeze_index_index_1, dtype=np.intc)

# Zn_index = loadmat(osdir + './params/N=128_BSC_p=0.00667_Pe.mat')
Zn_index = loadmat(osdir + './params/N=64_BSC_p=0.00667_Pe.mat')
select_index_index_2, freeze_index_index_2, _ = SelectGoodChannels4Polar(Zn_index['Pe'].flatten(), matrix_row_num_index_2 + len(crc_index_2) - 1)
select_index_index_2 = np.array(select_index_index_2, dtype=np.intc)
freeze_index_index_2 = np.array(freeze_index_index_2, dtype=np.intc)

max_homopolymer = 4  # 当设为3时，真的会有数据尝试500次随机配对后，还无法满足兼容性要求，设为4与5比，添加的随机bits几乎是一样的
max_gc_content = 0.6
search_count = 500
max_ratio = 0.6
selected_number = 11  # 初始化全局变量
def get_selected_number():
    return selected_number

def set_selected_number(value):
    global selected_number
    selected_number = value
# global selected_number
# selected_number = 4
copy_number_pyhsics = 160  # 总的每条DNA序列的copy数量，为DNA清洗做准备, 最好是threads_number的整数倍,方便badread模拟
# with open("copy_num.txt",'r')as f:
#     selected_number = int(f.read())

# selected_number = 11   # 在DNA清洗过程中，选择selected_number条DNA序列来做多序列比对，偶数真的会差很多吗？从实验跑起来，好像是的。10 比 9 差？？
# selected_number = 10  # 在DNA清洗过程中，选择selected_number条DNA序列来做多序列比对，偶数真的会差很多吗？从实验跑起来，好像是的。10 比 9 差？？
# selected_number =
step_length = 6  # 先确定一步的长度，再确定分成了多少份
segment_n = math.ceil(segment_length / step_length)  #表示将dna矩阵分segment_n次进行处理，与整条DNA序列长度为242相匹配
check_length = step_length  # 检查修改位以及后面连续的碱基，共检查check_length个.check_length=5 or 6是ok的，但不能太长，比如为10 or 11，这会导致修改时性能下降。原因是：check_length过长时，解码出来的后面的DNA碱基很不可靠，故以其为准的修改不会有好结果。

threads_number = 20
back_length = 3  # 检查修改DNA矩阵时，回退的长度，注意：现在的方案支持back_length=2或3，其它值要小心检查
kmer_len = 11
threshold = 0.145
num_perm = 128
add_random_list_threshold = 0  # 打印显示添加随机bits的阈值
# n_pools = 4  # 在利用编辑距离进行聚类时，为减少运算时间，先把大池子分成 n_pools 个小池子, 无用
edit_dis_threshold = 0.2
compute_edit_dis_len = 60  # 为减少处理others_others到已有类别的时间，这里设置一个计算编辑距离的DNA序列长度，长度越小速度越快
min_num = 7  # 聚类后，对类中最少DNA数量的要求

# 对index_base进行分割的方法
# index_base_length = 15  # 原来的方案，矩阵id与内id放在一起，index_base长度

index_base_matrix_id_length = 8  # 新方案的矩阵id长度
index_base_row_id_length = 10   # 新方案的块内id长度
# index_base_matrix_id_length = 7  # 新方案的矩阵id长度
# index_base_row_id_length = 13   # 新方案的块内id长度
index_base_length_new = index_base_matrix_id_length + index_base_row_id_length
# 二层index中，N_data = 2048，DNA_data预留到2100;N_index_1 = 512，DNA_index_1预留到280;N_index_2 = 64，DNA_index_2预留到63（因为可能也会干掉第0行）;
data_num = 2060   # 当添加的随机bit较多时，这里可能不够
index_num_1 = 140  # 这里与segment_length=242匹配
index_num_2 = N_index_2 - 1  # 这里-1，是因为用crc_4，第0行被干掉了，要检查是不是会这样
matrix_num = data_num + index_num_1 + index_num_2
min_class_num = data_num // 2

# 用于识别最后一个单位矩阵中真实数据行数的mark_list,长503,此时segment_length长1006
# mark_list_503 = [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
mark_list_121 = [0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1]
mark_list_120 = [0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1]
# 已经保证mark_list_121的总奇偶校验为0，奇数位异或运算后为0，偶数位异或运算后也为0
mark_list = mark_list_121
# mark_list = mark_list_120




# 最后一个表格的动态处理
list_last = 16
k_data_last_1, k_index_1_last_1 = 9, 6  # 最后一个矩阵高度h<=463时，用512进行极化，crc8，用一层index方案
k_data_last_2, k_index_1_last_2 = 10, 7  # 最后一个矩阵高度463<h<=955时，用1024进行极化，crc8,用一层index方案
# 若最后一个矩阵高度大于824，则用2048极化，与其它矩阵一样处理
N_data_last_1, N_index_1_last_1 = 2 ** k_data_last_1, 2 ** k_index_1_last_1
N_data_last_2, N_index_1_last_2 = 2 ** k_data_last_2, 2 ** k_index_1_last_2
index_binary_length_last_1 = k_data_last_1 + 1
index_binary_length_last_2 = k_data_last_2 + 1
crc_index_1_last = crc_poly_4
crc_data_last_1, crc_data_last_2 = crc_poly_8, crc_poly_8

# last_1, 512 22.5% 21.5%  平均15%时，平均块恢复率为0.991867，表现比last_3（块恢复率0.999264）要差一点，故提升到25%试试，真的很好
# frozen_bits_data_1_last_1, frozen_bits_data_2_last_1 = frozen_bits_125, frozen_bits_131
# last_2, 1024 19.5% 18.5%  平均19%时，平均块恢复率为0.996902，表现比last_3（块恢复率0.999264）要差一点，故提升到21%试试，真的很好
# frozen_bits_data_1_last_2, frozen_bits_data_2_last_2 = frozen_bits_210, frozen_bits_200

# frozen_bits_data_1_last_1, frozen_bits_data_2_last_1 = frozen_bits_100, frozen_bits_95    # 512  19.5% 18.5%
# frozen_bits_data_1_last_2, frozen_bits_data_2_last_2 = frozen_bits_159, frozen_bits_148   # 1024 15.5% 14.5%

# frozen_bits_data_1_last_1, frozen_bits_data_2_last_1 = frozen_bits_90, frozen_bits_85    # 512   0.165 0.175
# frozen_bits_data_1_last_2, frozen_bits_data_2_last_2 = frozen_bits_138, frozen_bits_128   # 1024 0.125 0.135

# frozen_bits_data_1_last_1, frozen_bits_data_2_last_1 = frozen_bits_79, frozen_bits_74    # 512   0.155 0.145
# frozen_bits_data_1_last_2, frozen_bits_data_2_last_2 = frozen_bits_118, frozen_bits_108   # 1024 0.115 0.105

# frozen_bits_data_1_last_1, frozen_bits_data_2_last_1 = frozen_bits_69, frozen_bits_64    # 512   0.135 0.125
# frozen_bits_data_1_last_2, frozen_bits_data_2_last_2 = frozen_bits_97, frozen_bits_87    # 1024  0.095 0.085

# frozen_bits_data_1_last_1, frozen_bits_data_2_last_1 = frozen_bits_59, frozen_bits_54    # 512   0.115 0.105
# frozen_bits_data_1_last_2, frozen_bits_data_2_last_2 = frozen_bits_87, frozen_bits_77   # 1024   0.085 0.075

# frozen_bits_data_1_last_1, frozen_bits_data_2_last_1 = frozen_bits_49, frozen_bits_44    # 512   0.095 0.085
# frozen_bits_data_1_last_2, frozen_bits_data_2_last_2 = frozen_bits_77, frozen_bits_67   # 1024   0.075 0.065

frozen_bits_data_1_last_1, frozen_bits_data_2_last_1 = frozen_bits_44, frozen_bits_38    # 512   0.085 0.075
frozen_bits_data_1_last_2, frozen_bits_data_2_last_2 = frozen_bits_67, frozen_bits_56   # 1024   0.065 0.055

frozen_bits_data_1_last_1 = np.array(frozen_bits_data_1_last_1, dtype=np.intc)
frozen_bits_data_2_last_1 = np.array(frozen_bits_data_2_last_1, dtype=np.intc)
frozen_bits_data_1_last_2 = np.array(frozen_bits_data_1_last_2, dtype=np.intc)
frozen_bits_data_2_last_2 = np.array(frozen_bits_data_2_last_2, dtype=np.intc)

# frozen_bits_index_1_last_1, frozen_bits_index_1_last_2 = frozen_bits_148, frozen_bits_318

# frozen_bits_index_1_last_1, frozen_bits_index_1_last_2 = frozen_bits_16, frozen_bits_29
frozen_bits_index_1_last_1, frozen_bits_index_1_last_2 = frozen_bits_16, frozen_bits_20

# frozen_bits_index_1_last_1, frozen_bits_index_1_last_2 = frozen_bits_33, frozen_bits_76
frozen_bits_index_1_last_1 = np.array(frozen_bits_index_1_last_1, dtype=np.intc)
frozen_bits_index_1_last_2 = np.array(frozen_bits_index_1_last_2, dtype=np.intc)

matrix_row_num_data_1_last_1 = N_data_last_1 - len(crc_data_last_1) + 1 - len(frozen_bits_data_1_last_1)   # 分组情况1-512，第1层
matrix_row_num_data_2_last_1 = N_data_last_1 - len(crc_data_last_1) + 1 - len(frozen_bits_data_2_last_1)   # 分组情况1-512，第2层
matrix_row_num_index_1_last_1 = N_index_1_last_1 - len(crc_index_1_last) + 1 - len(frozen_bits_index_1_last_1)
more_data_length_last_1 = segment_length_parity % (2 * index_binary_length_last_1)  # 表示每行行号数据与segment_length_parity的差，这要用随机bits补上
segment_length_index_1_last_1 = segment_length_parity - more_data_length_last_1   # segment_length_index 表示对index表格数据进行切割的长度

matrix_row_num_data_1_last_2 = N_data_last_2 - len(crc_data_last_2) + 1 - len(frozen_bits_data_1_last_2)  # 分组情况2-1024，第1层
matrix_row_num_data_2_last_2 = N_data_last_2 - len(crc_data_last_2) + 1 - len(frozen_bits_data_2_last_2)  # 分组情况2-1024，第2层
matrix_row_num_index_1_last_2 = N_index_1_last_2 - len(crc_index_1_last) + 1 - len(frozen_bits_index_1_last_2)
more_data_length_last_2 = segment_length_parity % (2 * index_binary_length_last_2)  # 表示每行行号数据与segment_length_parity的差，这要用随机bits补上
segment_length_index_1_last_2 = segment_length_parity - more_data_length_last_2   # segment_length_index 表示对index表格数据进行切割的长度

path_last_1 = osdir + './params/Index_two_layer_N=512_QSC_p=0.010.npy'
# Index_data_two_layer = np.load(osdir + './params/Index_two_layer_N=2048_SC_p=0.010.npy')
Index_data_two_layer = np.load(path_last_1)
# print("我的构造是：", path)
select_index_data_1_last_1, freeze_index_data_1_last_1, select_index_data_2_last_1, freeze_index_data_2_last_1 = (
    SelectGoodChannels4Polar_two_layer_npy(Index_data_two_layer, matrix_row_num_data_1_last_1 + len(crc_data_last_1) - 1,
                                           matrix_row_num_data_2_last_1 + len(crc_data_last_1) - 1))
select_index_data_1_last_1 = np.array(select_index_data_1_last_1, dtype=np.intc)
freeze_index_data_1_last_1 = np.array(freeze_index_data_1_last_1, dtype=np.intc)
select_index_data_2_last_1 = np.array(select_index_data_2_last_1, dtype=np.intc)
freeze_index_data_2_last_1 = np.array(freeze_index_data_2_last_1, dtype=np.intc)
# Zn_index = loadmat(osdir + './params/N=128_BSC_p=0.00667_Pe.mat')
# Zn_index = loadmat(osdir + './params/N=256_BSC_p=0.00667_Pe.mat')
Zn_index = loadmat(osdir + './params/N=64_BSC_p=0.00667_Pe.mat')
select_index_index_1_last_1, freeze_index_index_1_last_1, _ = SelectGoodChannels4Polar(Zn_index['Pe'].flatten(), matrix_row_num_index_1_last_1 + len(crc_index_1_last) - 1)
select_index_index_1_last_1 = np.array(select_index_index_1_last_1, dtype=np.intc)
freeze_index_index_1_last_1 = np.array(freeze_index_index_1_last_1, dtype=np.intc)

path_last_2 = osdir + './params/Index_two_layer_N=1024_QSC_p=0.010.npy'
# Index_data_two_layer = np.load(osdir + './params/Index_two_layer_N=2048_SC_p=0.010.npy')
Index_data_two_layer = np.load(path_last_2)
# print("我的构造是：", path)
select_index_data_1_last_2, freeze_index_data_1_last_2, select_index_data_2_last_2, freeze_index_data_2_last_2 = (
    SelectGoodChannels4Polar_two_layer_npy(Index_data_two_layer, matrix_row_num_data_1_last_2 + len(crc_data_last_2) - 1,
                                           matrix_row_num_data_2_last_2 + len(crc_data_last_2) - 1))
select_index_data_1_last_2 = np.array(select_index_data_1_last_2, dtype=np.intc)
freeze_index_data_1_last_2 = np.array(freeze_index_data_1_last_2, dtype=np.intc)
select_index_data_2_last_2 = np.array(select_index_data_2_last_2, dtype=np.intc)
freeze_index_data_2_last_2 = np.array(freeze_index_data_2_last_2, dtype=np.intc)
# Zn_index = loadmat(osdir + './params/N=512_BSC_p=0.00667_Pe.mat')
# Zn_index = loadmat(osdir + './params/N=256_BSC_p=0.00667_Pe.mat')
Zn_index = loadmat(osdir + './params/N=128_BSC_p=0.00667_Pe.mat')
select_index_index_1_last_2, freeze_index_index_1_last_2, _ = SelectGoodChannels4Polar(Zn_index['Pe'].flatten(), matrix_row_num_index_1_last_2 + len(crc_index_1_last) - 1)
select_index_index_1_last_2 = np.array(select_index_index_1_last_2, dtype=np.intc)
freeze_index_index_1_last_2 = np.array(freeze_index_index_1_last_2, dtype=np.intc)

# start_protect_seq 长200, end_protect_seq 长100
start_protect_seq = 'CCTTGCCAGTATTGATCACGCGGCGTTGAGGCTCATGTACTGAGATGCAGAGCGATGCCGTTAATTGCCAGGCCTTATAGTACCTCATCTCATAAGTGAGGAGAACTATGCACGTAATGCTCAACGCGGACTGGAATCCATGGACTTAGCTCGGCCAACTATCGGTCCATAAGTTACATGGTTGCGGCTGTGGTAGATTA'
end_protect_seq = 'AGTATGCCTCAGCGTTGGCATCACTGCAATGAAGTACGAACTCTCTACAATGCTCTAATCCATGTATCCTACCTGCCATCTTCTGCTCAAGGAAGCGCTG'
# end_protect_seq =   'AGCTATGAGCAACGGTCAATTCTGCAGCCGCGGACTGCCATCCTAGTAAGCTGCCAGCAAGTCGAGTCTCGAAGGATTGGCTCGATTCGAAGAGCATATTAGGCGTTACATCTCACGGCATCCGATTCGAGTGACAATCCTAGGAGCAAGTCAACGAGAACGACCAAGATATGATTCGTGTCTTGATCATAATTAGACGA'
# start_primer = 'AAGCCGGAGTAAGGTAATAT'
# end_primer = 'ACAGATGTCGCGTAGGAGTA'
# start_primer = 'ATCATACATACGCGAGGAGT'   # 试试老师的primer, 对1
# end_primer = 'GCGTCTTAGCCTTACCAATC'
# start_primer = 'GTAGAGCCACCAGGAAGACT'   # 试试老师的primer, 对2
# end_primer = 'TCGTTGACTCTTCCAGATTG'



glue_20_0 = 'AAGTCAGTGCTGTCATTGTC'
glue_20_1 = 'TGCAAGCCAAGTTGTCGACG'
glue_20_2 = 'CGACCAGATCTCCTTGATGT'

# glue_20_0 = 'GTGATGAGACGAGTCTGTCA'
# glue_20_1 = 'CTCGTACGCACATGATATCT'
# glue_20_2 = 'GCTACATAGATCGATGCATG'

glue_lists = [glue_20_0, glue_20_1, glue_20_2]
link_num = 3   # 将多少条较短的DNA序列粘在一起，形成长的DNA序列，以进行3代模拟测序,建议为3 or 4
# 3代测序后以及清洗后，检查DNA序列的长度
# 很奇怪,check_length_1 = 8,check_length_2 = 7,会减少最少类内数量为8时的分类数量,因而降低整体表现
# check_length_1 = 10,check_length_2 = 8, 似乎是不错的组合
check_length_1 = 10  # 第一次检查，即3代模拟错误后，检查长度。左右检查长度总是一样的。
check_length_2 = 8  # 第二次检查，即去掉两端protect序列后，进一步去掉添加的胶水（glue）
check_length_3 = 4  # 第三次检查，用于去掉清洗后添加在两端的primer
check_length_cluster = 4  # 利用index_base聚类时，查找index_base块地址的检查长度
protect_residue_len = 15  # 3代模拟错误并去掉部分两端添加的保护DNA序列后剩余的长度，用于保护清洗过程对两端的影响
primer_left_len = 20
primer_right_len = 20
glue_len = len(glue_lists[0])

file_path_matrix_index = osdir + "./params/index_base_no_think_complementary_reverse_seq_0805_21.txt"  # 块地址
file_path_list_index = osdir + "./params/output_have_complementary_10_03_all_3500.txt"  # 块内地址
# file_path_matrix_index = "./params/index_base_no_think_complementary_reverse_seq_0705_11.txt"  # 块地址
# file_path_list_index = "./params/index_base_no_think_complementary_reverse_seq_1305_2763.txt"  # 块内地址


