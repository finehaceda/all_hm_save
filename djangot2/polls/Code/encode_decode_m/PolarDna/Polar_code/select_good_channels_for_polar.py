import numpy as np


def SelectGoodChannels4Polar(Zn, K):
    """
    利用巴氏参数Zn进行信道排序
    Gn: n*n generator matrix of Bhattacharyya parameters.

    Args:
    Zn:巴氏参数
    K:信息位长度，K < n.

    Returns:
    SelectIndex: k个好信道的排序
    FreezeIndex: n-k个冻结信道的排序
    ZnSmall: the sorted Bhattacharyya parameters of the good channels.
    """
    # 按巴氏参数升序排序
    index = np.argsort(Zn.flatten())  # 排序后的索引值
    Zn_sort = Zn[index]

    # 挑选巴氏参数最小的k个信道传输信息位
    SelectIndex = np.sort(index[:K])

    # 巴氏参数大的n-k个信道传输冻结位
    FreezeIndex = np.sort(index[K:])

    # k个最小的巴氏参数
    ZnSmall = Zn_sort[:K]
    return SelectIndex, FreezeIndex, ZnSmall

def SelectGoodChannels4Polar_two_layer_npy(lists, K1, K2):
    select_index_1 = lists[0][:K1]
    freeze_index_1 = lists[0][K1:]
    select_index_2 = lists[1][:K2]
    freeze_index_2 = lists[1][K2:]
    return select_index_1, freeze_index_1, select_index_2, freeze_index_2
