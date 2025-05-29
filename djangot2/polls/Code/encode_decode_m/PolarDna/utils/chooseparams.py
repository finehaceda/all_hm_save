import numpy as np
from .frozen_bits import (frozen_bits_318, frozen_bits_297, frozen_bits_232, frozen_bits_182,
                          frozen_bits_174, frozen_bits_154, frozen_bits_108, frozen_bits_133,
                          frozen_bits_113, frozen_bits_97)
from .glob_var import osdir
from .select_good_channels_for_polar import SelectGoodChannels4Polar_two_layer_npy

class PolarParams():
    def __init__(self,N_data,crc_data,frozen_bits_len):
        self.N_data = N_data
        self.crc_data = crc_data
        self.frozen_bits_len = frozen_bits_len
        # self.frozen_bits_data_1 = frozen_bits_108  # 这两层冻结bits的长度，可以在构造后再确认
        # self.frozen_bits_data_2 = frozen_bits_97  # 好像第2层的表现会差一点，可以考虑变化冻结bit的长度
        if self.frozen_bits_len == 5:
            self.frozen_bits_data_1 = frozen_bits_108  # 这两层冻结bits的长度，可以在构造后再确认
            self.frozen_bits_data_2 = frozen_bits_97  # 好像第2层的表现会差一点，可以考虑变化冻结bit的长度
        elif self.frozen_bits_len == 6:
            self.frozen_bits_data_1 = frozen_bits_133  # 这两层冻结bits的长度，可以在构造后再确认
            self.frozen_bits_data_2 = frozen_bits_113
        elif self.frozen_bits_len == 8:
            self.frozen_bits_data_1 = frozen_bits_174  # 这两层冻结bits的长度，可以在构造后再确认
            self.frozen_bits_data_2 = frozen_bits_154
        elif self.frozen_bits_len == 10:
            self.frozen_bits_data_1 = frozen_bits_232  # 这两层冻结bits的长度，可以在构造后再确认
            self.frozen_bits_data_2 = frozen_bits_182
        elif self.frozen_bits_len == 15:
            self.frozen_bits_data_1 = frozen_bits_318  # 这两层冻结bits的长度，可以在构造后再确认
            self.frozen_bits_data_2 = frozen_bits_297
        self.frozen_bits_data_1 = np.array(self.frozen_bits_data_1, dtype=np.intc)
        self.frozen_bits_data_2 = np.array(self.frozen_bits_data_2, dtype=np.intc)

        self.matrix_row_num_data_1 = self.N_data - len(self.crc_data) + 1 - len(self.frozen_bits_data_1)
        self.matrix_row_num_data_2 = self.N_data - len(self.crc_data) + 1 - len(self.frozen_bits_data_2)
        path = osdir + './params/Index_two_layer_N=2048_QSC_p=0.010.npy'
        Index_data_two_layer = np.load(path)
        select_index_data_1, freeze_index_data_1, select_index_data_2, freeze_index_data_2 = (
            SelectGoodChannels4Polar_two_layer_npy(Index_data_two_layer, self.matrix_row_num_data_1 + len(self.crc_data) - 1,
                                                   self.matrix_row_num_data_2 + len(self.crc_data) - 1))
        self.select_index_data_1 = np.array(select_index_data_1, dtype=np.intc)
        self.freeze_index_data_1 = np.array(freeze_index_data_1, dtype=np.intc)
        self.select_index_data_2 = np.array(select_index_data_2, dtype=np.intc)
        self.freeze_index_data_2 = np.array(freeze_index_data_2, dtype=np.intc)


def getfrozen_bits(N_data,crc_data,frozen_bits_len=5):
    # 5%
    frozen_bits_data_1 = frozen_bits_108  # 这两层冻结bits的长度，可以在构造后再确认
    frozen_bits_data_2 = frozen_bits_97  # 好像第2层的表现会差一点，可以考虑变化冻结bit的长度
    if frozen_bits_len == 5:
        frozen_bits_data_1 = frozen_bits_108  # 这两层冻结bits的长度，可以在构造后再确认
        frozen_bits_data_2 = frozen_bits_97  # 好像第2层的表现会差一点，可以考虑变化冻结bit的长度
    elif frozen_bits_len == 6:
        frozen_bits_data_1 = frozen_bits_133  # 这两层冻结bits的长度，可以在构造后再确认
        frozen_bits_data_2 = frozen_bits_113
    elif frozen_bits_len == 8:
        frozen_bits_data_1 = frozen_bits_174  # 这两层冻结bits的长度，可以在构造后再确认
        frozen_bits_data_2 = frozen_bits_154
    elif frozen_bits_len == 10:
        frozen_bits_data_1 = frozen_bits_232  # 这两层冻结bits的长度，可以在构造后再确认
        frozen_bits_data_2 = frozen_bits_182
    elif frozen_bits_len == 15:
        frozen_bits_data_1 = frozen_bits_318  # 这两层冻结bits的长度，可以在构造后再确认
        frozen_bits_data_2 = frozen_bits_297
    frozen_bits_data_1 = np.array(frozen_bits_data_1, dtype=np.intc)
    frozen_bits_data_2 = np.array(frozen_bits_data_2, dtype=np.intc)

    matrix_row_num_data_1 = N_data - len(crc_data) + 1 - len(frozen_bits_data_1)
    matrix_row_num_data_2 = N_data - len(crc_data) + 1 - len(frozen_bits_data_2)
    path = './params/Index_two_layer_N=2048_QSC_p=0.010.npy'
    Index_data_two_layer = np.load(path)
    select_index_data_1, freeze_index_data_1, select_index_data_2, freeze_index_data_2 = (
        SelectGoodChannels4Polar_two_layer_npy(Index_data_two_layer, matrix_row_num_data_1 + len(crc_data) - 1,
                                               matrix_row_num_data_2 + len(crc_data) - 1))
    select_index_data_1 = np.array(select_index_data_1, dtype=np.intc)
    freeze_index_data_1 = np.array(freeze_index_data_1, dtype=np.intc)
    select_index_data_2 = np.array(select_index_data_2, dtype=np.intc)
    freeze_index_data_2 = np.array(freeze_index_data_2, dtype=np.intc)

    return frozen_bits_data_1, frozen_bits_data_2, matrix_row_num_data_1, matrix_row_num_data_2,\
           select_index_data_1, freeze_index_data_1, select_index_data_2, freeze_index_data_2
