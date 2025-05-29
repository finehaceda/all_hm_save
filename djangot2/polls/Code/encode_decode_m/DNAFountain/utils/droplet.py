"""
Copyright (C) 2016 Yaniv Erlich
License: GPLv3-or-later. See COPYING file for details.
"""

import random
import struct

from polls.Code.utils import crc16


class Droplet:
    def __init__(self, data, seed, num_chunks = None, rs = 0, rs_obj = None, degree = None,crc_num=None):
        self.crc_num = crc_num
        self.data = data
        self.seed = seed
        self.num_chunks = set(num_chunks)
        self.rs = rs
        self.rs_obj = rs_obj
        self.degree = degree
        self.DNA = None

    def toDNA(self, flag = None):
        if self.DNA is not None:
            return self.DNA
        self.DNA = self._int_to_four(self._package())
        return self.DNA
    def _int_to_four(self, a):
        bin_data = ''.join('{0:08b}'.format(element) for element in a) #convert to a long sring of binary values
        return ''.join(str(int(bin_data[t:t+2],2)) for t in range(0, len(bin_data),2)) #convert binary array to a string of 0,1,2,3

    # def _package(self):
    #     seed_ord = [c for c in struct.pack("!I", self.seed)]
    #     message = seed_ord + self.data
    #     if self.rs > 0:
    #         message = self.rs_obj.encode(message)  # adding RS symbols to the message
    #     return message

    # 加入crc校验码
    def _package(self):
        seed_ord = [c for c in struct.pack("!I", self.seed)]
        message = seed_ord + self.data
        if self.rs > 0:
            # 计算 CRC
            if self.crc_num > 0:
                crc = crc16(message)
                crc_bytes = crc.to_bytes(2, byteorder='big')
                # 将 CRC 附加到文件数据后面
                message = message + list(crc_bytes)
            message = self.rs_obj.encode(message) #adding RS symbols to the message
        return message

    def to_human_readable_DNA(self):
        return self.toDNA().replace('0','A').replace('1','C').replace('2','G').replace('3','T')
    def chunkNums(self):
        return self.num_chunks
    
        


