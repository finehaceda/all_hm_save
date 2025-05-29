import numpy as np
import math

def BitReverse(Normal):
    n = len(Normal)
    d = int(math.log(n, 2))
    # Reversal = np.array([int(bin(dec - 1)[2:].zfill(d)[::-1], 2) + 1 for dec in Normal])
    Reversal = np.array([int(bin(dec - 1)[2:].zfill(d)[::-1], 2) for dec in Normal])  # 在python中下标从1开始，所以不加1
    return Reversal

