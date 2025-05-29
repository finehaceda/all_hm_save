# -*- coding: utf-8 -*-
import csv
import logging
import math
from abc import ABC, abstractmethod
from datetime import datetime
from os import path, stat
from typing import Tuple, List

import numpy as np
from numpy import array, NaN
# import numpy
from numpy import uint8

# from Evaluation_platform.hedges_master.get_decodedout import messbytesperstrand
# from .hedges_config import *
from .utils import bin_to_str_list
from ..utils import crc16

log = logging.getLogger('mylog')

from .tools import  EncodeParameter, DecodeParameter, FileTools, SplitTools, RsTools, BaseTools, CodecException
from .getPrimerPair import PrimerDesign, getOther

base_list = ['a', 't', 'c', 'g', 'A', 'T', 'C', 'G']

class AbstractEncode(ABC):
    """
    Abstract encode class
    """
    def __init__(self, input_file_path:str, output_dir:str, tool_name,seq_bit_to_base_ratio=1, index_redundancy=0, **encode_parameters):
        """
        Subclass may add some new parameters, 
        and add 'super().__init__(input_file_path, output_dir, **kwargs)' in its __init__.
        """
        self.tool_name = tool_name # in filename
        self.input_file_path = input_file_path
        self.output_dir = output_dir if output_dir[-1] in ['/','\\'] else output_dir+'/'
        if not path.exists(self.output_dir):
            raise CodecException('Output dir not exist : {}'.format(self.output_dir))
        self.file_base_name,self.file_extension =  FileTools.get_file_name_extension(self.input_file_path)
        self.output_file_path = self.output_dir + self.file_base_name + "_{}.fasta".format(tool_name)
        self.codec_param=EncodeParameter(**encode_parameters)
        self.seq_bit_to_base_ratio = seq_bit_to_base_ratio
        self.index_redundancy = index_redundancy
        self.total_bit =  8*stat(self.input_file_path).st_size
        self.seq_num = 0
        self.bin_split_len = 0
        self.index_length = 0
        self.run_time = '0:00:00'
        self.encode_time = '0:00:00'
        self.density = 0
        self.total_base = 0
        self.rs_group = 0
        self._check_params()
        self.codec_worker = None

    def __matrix_init__(self,matrix_n,matrix_r,coderatecode):
        # self.coderates = coderates

        # self.coderatecode =1
        self.coderatecode = coderatecode
        self.npackets = 1
        self.totstrandlen = self.codec_param.sequence_length
        coderates = array([NaN, 0.75, 0.6, 0.5, 1. / 3., 0.25, 1. / 6.])
        self.strandIDbytes = 2
        self.strandrunoutbytes = 2
        self.hlimit = 1000000
        # self.leftprimer = "T"
        # self.rightprimer = "T"
        self.leftprimer = ""
        self.rightprimer = ""
        self.max_hpoly_run = self.codec_param.max_homopolymer
        self.strandsperpacket = matrix_n
        self.strandsperpacketcheck = matrix_r
        # leftlen = len(leftprimer)
        # rightlen = len(rightprimer)
        self.strandlen = self.codec_param.sequence_length - len(self.leftprimer) - len(self.rightprimer)
        self.strandsperpacketmessage = self.strandsperpacket - self.strandsperpacketcheck
        self.bytesperstrand = int(self.strandlen * coderates[self.coderatecode]  / 4.)
        # bytesperstrand = int(self.strandlen * coderates[coderatecode] / 4.)
        self.messbytesperstrand = self.bytesperstrand - self.strandIDbytes - self.strandrunoutbytes  # payload bytes per strand
        self.messbytesperpacket = self.strandsperpacket * self.messbytesperstrand  # payload bytes per packet of 255 strands


    def _check_params(self):
        if self.codec_param.max_homopolymer<=0:
            raise CodecException("homopolymer:{} must be greater than 0".format(self.codec_param.homopolymer))
        if self.codec_param.min_gc<=0 or self.codec_param.max_gc>=1:
            raise CodecException("min_gc:{} must be greater than 0 and max_gc:{} must be less than 1".format(self.codec_param.min_gc, self.codec_param.max_gc))
        if self.codec_param.min_gc > self.codec_param.max_gc:
            raise CodecException("min_gc:{} could not be greater than max_gc:{}".format(self.codec_param.min_gc, self.codec_param.max_gc))
        if self.codec_param.primer_length<18 and self.codec_param.primer_length>24: # set in primer3Settiong.py
            raise CodecException("primer_length:{} should be between 18 and 24")
        if self.codec_param.sequence_length<=0:
            raise CodecException("sequence_length must be greater than 0")


    def getgc(self,all_seqs):
        avggc = 0
        all_gc = []
        for dna_sequence in all_seqs:
            try:
                gc_count = dna_sequence.count('G') + dna_sequence.count('C')
                gc = gc_count / len(dna_sequence)
                avggc += gc
                all_gc.append(gc)
            except:
                print(f"getgc遇到了错误，序列为：{dna_sequence}")
                continue
        with open(f'gcfiles/{self.tool_name}_{self.file_base_name}.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            for g in all_gc:
                t = (g,)
                writer.writerow(t)
        return avggc / len(all_seqs)

    def max_homopolymer_length(self,all_dna_sequence):
        all_homopolymer = []
        maxh = 1
        for dna_sequence in all_dna_sequence:
            try:
                current_length = 1  # 当前检查的均聚物长度
                max_length = 1  # 最短均聚物长度为1（单个核苷酸）
                # 遍历DNA序列，从第二个核苷酸开始（因为单个核苷酸本身就是一个长度为1的均聚物）
                for i in range(1, len(dna_sequence)):
                    if dna_sequence[i] == dna_sequence[i - 1]:
                        # 如果当前核苷酸与前一个核苷酸相同，则增加当前均聚物的长度
                        current_length += 1
                    else:
                        # 如果不同，则重置当前均聚物的长度，并检查是否需要更新最大长度
                        if current_length > max_length:
                            max_length = current_length
                        current_length = 1  # 重置为1，因为新序列片段从当前核苷酸开始
                # 检查循环结束后的最后一个均聚物长度（以防它是最长的）
                if current_length > max_length:
                    max_length = current_length
                if max_length > maxh:
                    maxh = max_length
                all_homopolymer.append(max_length)
            except:
                print(f"max_homopolymer_length遇到了错误，序列为：{dna_sequence}")
                continue
        with open(f'homopolymerfiles/{self.tool_name}_{self.file_base_name}.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            for g in all_homopolymer:
                t = (g,)
                writer.writerow(t)
        return maxh

    def _set_density(self):
        if self.total_base!=0:
            self.density = self.total_bit/self.total_base
        else:
            log.error("set density error, total base is 0")
    
    @abstractmethod
    def _encode(self, bit_segments: list, bit_size:int) -> list:
        """
        This part is implemented according to different algorithms
        Processed Binary Sequence(str) List >> Base Sequence(str) List
        """
        # return base_segments
        pass
    
    def _file_to_bin(self):
        """
        get binary string from file
        """
        bin_str = FileTools.file_to_bin(self.input_file_path)
        return bin_str

    def _split_bin(self, binstring:str) -> Tuple[int,int,List]:
        """
        split binstring to bin segments
        """
        index_length, bin_split_len, seq_num, rs_group = SplitTools.get_bin_split_length(len(binstring),
                                                                               self.seq_bit_to_base_ratio,
                                                                               self.codec_param,
                                                                               index_redundancy=self.index_redundancy)
        bit_segments = SplitTools.split(binstring, bin_split_len, self.codec_param.add_redundancy,
                                        index_redundancy=self.index_redundancy)
        return bin_split_len, index_length, rs_group, bit_segments

    def _add_crc_code(self,segment_list, group=1):
        # bin_to_str_list = list()
        # for i in range(256):
        #     bin_to_str_list.append(bin(i)[2:].rjust(8, '0'))
        res_list = list()
        for binstring in segment_list:
            # If the binary length is not a multiple of 8, it will be left-padded with zeros
            self.remainder = 0
            if len(binstring) % 8 != 0:
                self.remainder = 8 - len(binstring) % 8
                binstring = '0' * self.remainder + binstring

            data_byte = bytearray([int(binstring[i:i + 8], 2) for i in range(0, len(binstring), 8)])
            # data_byte = data.to_bytes(1, byteorder='big')  # 将整数转换为字节
            crc_byte = crc16(data_byte).to_bytes(2, byteorder='big')  # 计算CRC并转换为字节
            combined_data = data_byte + crc_byte  # 将原始数据和CRC合并
            str_list = ''.join([bin_to_str_list[i] for i in combined_data])
            res_list.append(str_list[self.remainder:])
        return res_list

    def _add_rscode(self,bit_segments):
        """
        add rscode
        """
        if self.crc_num > 0:
            log.info('add crccode')
            bit_segments = self._add_crc_code(bit_segments, group=self.rs_group)
            # print(f'len:{len(bit_segments[0])}')
        log.info('add rs')
        res_segments = RsTools.add_rs(bit_segments, self.codec_param.rs_num, self.rs_group)
        return res_segments

    def _add_rscode11(self, bit_segments):
        """
        add rscode
        """
        log.info('add rs')

        if self.crc_num > 0:
            bin_to_str_list = list()
            for i in range(256):
                bin_to_str_list.append(bin(i)[2:].rjust(8, '0'))
            processed_data_list = []
            for data in bit_segments:
                data_byte = bytearray([int(data[i:i + 8], 2) for i in range(0, len(data), 8)])
                # data_byte = data.to_bytes(1, byteorder='big')  # 将整数转换为字节
                crc_byte = crc16(data_byte).to_bytes(2, byteorder='big') # 计算CRC并转换为字节
                combined_data = data_byte + crc_byte  # 将原始数据和CRC合并
                str_list = ''.join([bin_to_str_list[i] for i in combined_data])
                # str_list = ''.join(str_list)
                processed_data_list.append(str_list)  # 转换回整数形式以便存储或传输
                # processed_data_list.append(int.from_bytes(combined_data, byteorder='big'))  # 转换回整数形式以便存储或传输
            bit_segments = processed_data_list

        res_segments = RsTools.add_rs(bit_segments, self.codec_param.rs_num, self.rs_group)
        return res_segments

    def _add_primer(self):
        log.info('add primer')
        primer_designer = PrimerDesign([self.output_file_path], iBreakNum=1, iBreakSec=300,iPrimerLen=self.codec_param.primer_length, sBaseDir=self.output_dir, bTimeTempName=False, bLenStrict=True)
        res = primer_designer.getPrimer()
        if len(res)>0:
            left_primer = res.left[0]
            right_primer = res.right[0]
            right_primer_reverse = getOther(right_primer) # add reverse complementary at the right
            pool = []
            with open(self.output_file_path, 'r') as file:
                param = self._set_param_line(left_primer, right_primer)
                pool.append(param)
                index = 0
                for line in file.readlines()[1:]:
                    if line[0] in base_list:
                        pool.append(">seq_{}\n".format(index+1) + left_primer + line.strip('\n') + right_primer_reverse + '\n')
                        index += 1
            with open(self.output_file_path, 'w') as f:
                f.writelines(pool)
            return self.output_file_path
        else:
            log.error("No primer")
            return self.output_file_path
    
    def _set_param_line(self, left_primer="", right_primer=""):
        param = ">totalBit:{},binSegLen:{},leftPrimer:{},rightPrimer:{},fileExtension:{},bRedundancy:{},RSNum:{}\n".format(
            self.total_bit, self.bin_split_len, left_primer, right_primer, self.file_extension,
            int(self.codec_param.add_redundancy), int(self.codec_param.rs_num))
        return param


    def _set_matrix_param_line(self, left_primer="", right_primer=""):
        param = "totalBit:{},strandsperpacket:{},leftPrimer:{},rightPrimer:{},strandsperpacketcheck:{},totstrandlen:{}," \
                "coderatecode:{},max_hpoly_run:{},messbytesperstrand:{},npackets:{}\n".format(
            self.total_bit, self.strandsperpacket, left_primer, right_primer, self.strandsperpacketcheck,
            self.totstrandlen, self.coderatecode,self.max_hpoly_run,self.messbytesperstrand,self.npackets)
        return param

    # def getgc(self, all_seqs):
    #     avggc = 0
    #     for dna_sequence in all_seqs:
    #         if dna_sequence is None:
    #             break
    #         gc_count = dna_sequence.count('G') + dna_sequence.count('C')
    #         total_count = len(dna_sequence)
    #         avggc += gc_count / total_count
    #     return avggc / len(all_seqs)
    #

    def common_encode(self):
        """
        common encode flow
        """
        # 序列形式编码：1.读取文件；2.根据文件大小，rs码个数，序列长度计算总共有多少条序列及index长度；3.编码并保存文件

        self._check_params()
        tm_run = datetime.now()
        log.debug('read file')
        bin_str = self._file_to_bin()

        # split
        self.bin_split_len, self.index_length, self.rs_group, bit_segments = self._split_bin(bin_str)
        self.seq_num = len(bit_segments)
        # new_bit_segments = [None] * self.seq_num
        # for i in range(self.seq_num):
        #     new_bit_segments[i]=bit_segments[i][self.index_length:]
        #add
        with open('messplains.txt', 'w') as file:
            for data in bit_segments:
                file.write(str(data)+'\n')
        if self.codec_param.rs_num>0:
            # bit_segments = self._add_rscode(new_bit_segments)
            bit_segments = self._add_rscode(bit_segments)
        self.decode_packets = len(bit_segments)
        # with open('messplains.txt', 'w') as file:
        #     for data in bit_segments:
        #         file.write(str(data)+'\n')
        self.total_count = len(bit_segments)
        # encode
        tm_encode = datetime.now()
        base_segments = self._encode(bit_segments,len(bin_str))

        print(f"编码时间为{str(datetime.now() - tm_encode)}")
        print(f"编码的序列共有{len(base_segments)}条，每条{len(base_segments[0])}nt")
        self.total_base = len(base_segments)  * len(base_segments[0])
        self._set_density()
        mygc = self.getgc(base_segments)
        self.gc = mygc
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print(f"total_bit:{self.total_bit},total_base:{self.total_base},dnaseqslen:{len(base_segments[0])},density:{self.density},mygc:{mygc}")

        param = self._set_param_line()
        log.debug('write')
        with open(self.output_file_path, 'w') as f:
            f.write(param)
            for index,seg in enumerate(base_segments):
                if seg is not None:
                    f.write(">seq_{}\n".format(index+1) + str(''.join(seg)) + '\n')
        
        # add primer (blast need fasta file to compare)
        if self.codec_param.add_primer:
            self._add_primer()
            self.total_base += (len(base_segments)*2*self.codec_param.primer_length)
            
        self.run_time = str(datetime.now()-tm_run)
        self._set_density()
        self.index_length = int(len(str(bin(len(bit_segments)))) - 2)
        # return self.output_file_path, self.index_length


    def getwiz(self,n):  # return next n chars from wiztext
        # global wizoffset, wizlen
        if self.wizoffset + n > self.wizlen:
            self.wizoffset = 0
        bytes = self.wizbytes[self.wizoffset:self.wizoffset + n]
        self.wizoffset += n
        return bytes

    def createmesspacket(self, packno):  # packno in range 0..255 with value 2 for strandIDbytes
        packet = np.zeros([self.strandsperpacket, self.bytesperstrand], dtype=uint8)
        # plaintext = np.zeros(self.strandsperpacketmessage * self.messbytesperstrand, dtype=uint8)
        plaintext = np.zeros([self.strandsperpacketmessage , self.bytesperstrand-self.strandrunoutbytes-self.strandIDbytes], dtype=uint8)
        for i in range(self.strandsperpacket):
            packet[i, 0] = packno  # note assumes value 2 for strandIDbytes
            packet[i, 1] = i
            if i < self.strandsperpacketmessage:
                ptext = self.getwiz(self.messbytesperstrand)
                packet[i, self.strandIDbytes:self.strandIDbytes + self.messbytesperstrand] = ptext
                # plaintext[i] = packet[i,:self.strandIDbytes + self.messbytesperstrand]
                plaintext[i] = packet[i,self.strandIDbytes: self.strandIDbytes + self.messbytesperstrand]
        return (packet, plaintext)

    def myprotectmesspacket(self, packetin):  # fills in the RS check strands
        packet = packetin.copy()
        # regins = np.zeros([self.messbytesperstrand,self.strandsperpacketmessage],dtype=uint8)
        regins = ['']*self.messbytesperstrand

        regin = np.zeros(self.strandsperpacketmessage, dtype=uint8)
        for j in range(self.messbytesperstrand):
            for i in range(self.strandsperpacketmessage):
                regin[i] = packet[i, ((j + i) % self.messbytesperstrand) + self.strandIDbytes]
            binstring = [bin(d)[2:].zfill(8) for d in regin]
            regins[j] = ''.join(binstring)
        self.rs_group = 1
        if self.codec_param.rs_num >= 0:
            bytes_num = math.ceil(len(regins[0]) / 8)
            self.rs_group = math.ceil(bytes_num / 255)
        if self.codec_param.rs_num > 0:
            bit_segments = self._add_rscode(regins)
        # print()
        for j in range(len(bit_segments)):
            bytearr = ([int(bit_segments[j][i:i + 8], 2) for i in range(0, len(bit_segments[j]), 8)])
            for i in range(self.strandsperpacket):
                packet[i, ((j + i) % self.messbytesperstrand) + self.strandIDbytes] = bytearr[i]
        return packet

    def writemessplain(self,path, datas):
        with open(path, 'w') as file:
            for data in datas:
                for d in data:
                    file.write(str(d) + ' ')
                file.write('\n')

    def file_binary_to_random(self,binary_sequence_ori):
        # 与圆周率的前500000位，进行异或操作，以实现随机化
        with open('./pi_str.txt', 'rb') as file:
            content = file.read()
            pi_binary_sequence = ''.join(format(byte, '08b') for byte in content)
        # print("pi_binary_sequence的长度：", len(pi_binary_sequence))
        str_1, str_2 = binary_sequence_ori, pi_binary_sequence
        len_str_1 = len(str_1)
        len_str_2 = len(str_2)
        # 使用列表推导式生成结果字符串,获取str_2中对应位置的字符，如果长度不够则循环使用
        binary_sequence_random = ''.join(str(int(str_1[i]) ^ int(str_2[i % len_str_2])) for i in range(len_str_1))
        # print("binary_sequence_ori的长度：", len(binary_sequence_ori))
        # print("binary_sequence_random的长度：", len(binary_sequence_random))
        return binary_sequence_random

    # def getgc(self,all_seqs):
    #     avggc = 0
    #     for dna_sequence in all_seqs:
    #         if dna_sequence is None:
    #             break
    #         gc_count = dna_sequence.count('G') + dna_sequence.count('C')
    #         total_count = len(dna_sequence)
    #         avggc += gc_count / total_count
    #     return avggc/len(all_seqs)

    def common_encode_matrix(self):
        pass
        # # # 矩阵形式编码
        # # 1.读取文件；2.根据文件大小，rs码个数，序列长度（及码率）计算总共有多少个矩阵；3.填充矩阵（+rs）4.编码并保存序列
        # tm_encode = datetime.now()
        # self._check_params()
        # self.__matrix_init__(self.matrix_n, self.matrix_r, self.coderatecode)
        # self.norsway = False
        # log.debug('read file')
        # # bin_to_str_list = list()
        # # for i in range(256):
        # #     bin_to_str_list.append(int(bin(i)[2:],2))
        # bin_to_str_list = list()
        # for i in range(256):
        #     bin_to_str_list.append(bin(i)[2:].rjust(8, '0'))
        # with open(self.input_file_path, 'rb') as f:
        #     bytesarr = f.read()
        # # self.wizbytes = [bin_to_str_list[i] for i in bytesarr]
        # str_list = [bin_to_str_list[i] for i in bytesarr]
        # binstring = ''.join(str_list)
        # binstring = self.file_binary_to_random(binstring)
        # self.wizbytes = [int(binstring[i:i + 8], 2) for i in range(0, len(binstring), 8)]
        #
        #
        # self.wizlen = len(self.wizbytes)
        # self.npackets = int(math.ceil(float(self.wizlen) / self.messbytesperstrand / self.strandsperpacketmessage))
        # self.bytesseqnum = int(math.ceil(float(self.wizlen) / self.messbytesperstrand))
        # npackets=self.npackets
        # # 填充字节数据到矩阵
        # messplains = np.zeros([npackets * self.strandsperpacketmessage,self.bytesperstrand-self.strandrunoutbytes-self.strandIDbytes], dtype=uint8)
        # segmentfor_encode = np.zeros([npackets * self.strandsperpacket,self.bytesperstrand], dtype=uint8)
        # self.wizoffset = 0
        # for ipacket in range(npackets):
        #     messpack, messplain = self.createmesspacket(ipacket)
        #     messplains[ipacket * self.strandsperpacketmessage:(ipacket + 1) * self.strandsperpacketmessage] = messplain
        #     segmentfor_encode[ipacket*self.strandsperpacket:(ipacket+1)*self.strandsperpacket] = messpack
        # if self.norsway:
        #     for ipacket in range(npackets):
        #         segmentfor_encode[ipacket * self.strandsperpacket:(ipacket + 1) * self.strandsperpacket] = self.myprotectmesspacket(messpack)
        #
        # # 编码 + rs
        # # self.writemessplain('messplains.txt', messplains)
        # self.writemessplain('messbytesnors.txt', messplains[:self.bytesseqnum])
        # outfilepath = self._encode(segmentfor_encode,self.coderatecode)
        #
        # # 计算 信息密度
        # with open(outfilepath, 'r', encoding='utf-8') as f:
        #     old_content = f.read()
        # tseqs = old_content.split('\n')[1::2]
        # self.total_base = npackets * self.strandsperpacket * self.totstrandlen
        # self._set_density()
        # dnaseqs = []
        # for i in range(len(tseqs)):
        #     dnaseqs.append(tseqs[i])
        # mygc = self.getgc(dnaseqs)
        # print(f"total_bit:{self.total_bit},total_base:{self.total_base},density:{self.density},mygc:{mygc}")
        # # print(f"total_bit:{self.total_bit},total_base:{self.total_base},density:{self.density}")
        #
        # # 保存序列，首行添加编码的一些重要参数
        # param = self._set_matrix_param_line()
        # log.debug('write')
        # with open(self.output_file_path, 'w') as f:
        #     f.write(param)
        #     f.write(old_content)
        # # self.encode_time = str(datetime.now() - tm_encode)
        # # return outfilepath


class AbstractDecode(ABC):
    def __init__(self, input_file_path:str, output_dir:str, index_redundancy=0):
        """
        Subclass may add some new parameters, 
        and add 'super().__init__(input_file_path, output_dir)' in its __init__.
        """
        self.input_file_path = input_file_path
        self.output_dir = output_dir if output_dir[-1] in ['/','\\'] else output_dir+'/'
        if not path.exists(self.output_dir):
            raise CodecException('Output dir not exist')
        self.file_base_name,self.file_extension =  path.splitext(path.basename(self.input_file_path))
        self.rs_err_rate = 0.0 # error correction failed
        self.rs_err_indexs = [] # error correction failed
        self.repaired_rate = 0.0 # missing but repaired segments
        self.miss_err_rate = 0.0 # missing and unrepaired segments
        self.repaired_indexs = [] # missing but repaired segments
        self.miss_err_indexs = [] # missing and unrepaired segments
        self.run_time = 0.0
        self.index_redundancy = index_redundancy
        self.codec_worker = None




    def _check_file_param(self, param_list:list):
        with open(self.input_file_path,'r') as file:
            first_line = file.readline().strip()
            param_line = first_line.strip(">").strip(';').split(',')

        param_dict = dict()
        for param in param_line:
            temp = param.split(":")
            param_dict[temp[0]] = temp[1]
        # list_input = set(param_dict.keys())
        # list_default = set(param_list)
        # err_list = list(list_input.difference(list_default)) + list(list_default.difference(list_input))
        # log.info(err_list)
        # if len(err_list)!=0:
        #     log.error("wrong para : {}".format((',').join(err_list)))
        #     raise CodecException("Failed to parse parameter. Please Check that the input file and the selected codec algorithm are consistent.")
        return param_dict

    def _parse_param(self):
        """
        parse parameter from init input and input_file
        """
        param_dict = self._check_file_param(["totalBit","binSegLen","leftPrimer","rightPrimer","fileExtension","bRedundancy","RSNum"])  
        param = {
            'total_bit': int(param_dict['totalBit']),
            'bin_seg_len' : int(param_dict['binSegLen']),
            'add_redundancy' : bool(int(param_dict['bRedundancy'])),
            'rs_num' :int(param_dict['RSNum']),
            'left_primer_len' : len(param_dict['leftPrimer']),
            'right_primer_len' : len(param_dict['rightPrimer']),
            'file_extension' : param_dict['fileExtension']
            }
        self.codec_param = DecodeParameter(**param)
        self.index_length,self.seq_num = self._get_indexlen_splitnum()
        self.ori_len = self.codec_param.bin_seg_len + self.index_length
        self.output_file_path = self.output_dir + self.file_base_name + "_decode" + self.codec_param.file_extension

    def _parse_matrix_param(self):
        """
        parse parameter from init input and input_file

        param = ">totalBit:{},strandsperpacket:{},leftPrimer:{},rightPrimer:{},strandsperpacketcheck:{},totstrandlen:{},coderatecode:{},max_hpoly_run:{}\n".format(
            self.total_bit, self.strandsperpacket, left_primer, right_primer, self.strandsperpacketcheck, self.totstrandlen, self.coderatecode,self.max_hpoly_run)
        """
        param_dict = self._check_file_param(["totalBit","strandsperpacket","leftPrimer","rightPrimer","strandsperpacketcheck",
                                             "totstrandlen","coderatecode","max_hpoly_run","messbytesperstrand","npackets"])
        param = {
            'total_bit': int(param_dict['totalBit']),
            'left_primer_len': len(param_dict['leftPrimer']),
            'right_primer_len': len(param_dict['rightPrimer']),
            'file_extension': '.jpg'
        }
        self.total_bit = int(param_dict['totalBit'])
        self.strandsperpacket = int(param_dict['strandsperpacket'])
        self.strandsperpacketcheck = int(param_dict['strandsperpacketcheck'])
        self.totstrandlen = int(param_dict['totstrandlen'])
        self.coderatecode = int(param_dict['coderatecode'])
        self.max_hpoly_run = int(param_dict['max_hpoly_run'])
        self.messbytesperstrand = int(param_dict['messbytesperstrand'])
        self.npackets = int(param_dict['npackets'])
        # self.file_extension = ""
        self.codec_param = DecodeParameter(**param)
        # self.index_length,self.seq_num = self._get_indexlen_splitnum()
        # self.ori_len = self.codec_param.bin_seg_len + self.index_length
        self.output_file_path = self.output_dir + self.file_base_name + "_decode" + self.codec_param.file_extension

    def _get_indexlen_splitnum(self):
        index_length, seq_num = SplitTools.get_indexlen_and_segnum(self.codec_param.total_bit,
                                                                     self.codec_param.bin_seg_len,
                                                                     self.codec_param.add_redundancy,
                                                                     index_redundancy=self.index_redundancy)
        return index_length, seq_num
    
    def _get_base_line_list(self):
        base_line_list = []
        with open(self.input_file_path,'r') as file:
            if self.codec_param.right_primer_len==0:
                def get_line():
                    base_line_list.append(line.strip()[self.codec_param.left_primer_len:])
            else:
                def get_line():
                    base_line_list.append(line.strip()[self.codec_param.left_primer_len:-self.codec_param.right_primer_len])
            # del primer
            for line in file.readlines():
                if line[0] in base_list:
                    get_line()
        base_line_list = [x.strip() for x in base_line_list]
        return base_line_list
        
    @abstractmethod
    def _decode(self, base_line_list:list()) -> list():
        """
        The input parameter is a list of base sequences, and the return value is a list of binary sequences
        """
        # return bit_segments
        pass


    def _del_rscode_crc(self, bit_segs, phreds=[]):
        res_dict = RsTools.del_rs_crc(bit_segs, self.ori_len, self.codec_param.rs_num,phreds,self.crc_num)
        err_segs = [bit_segs[i] for i in res_dict.get("err_index")]
        self.rs_err_rate = res_dict.get("err_rate")
        err_lists_tmp = [int(bit_segs[i][:self.index_length], 2) for i in res_dict["err_index"]]
        for index in err_lists_tmp:
            if index < self.seq_num:
                self.rs_err_indexs.append(index)
        return res_dict['validate_bit_seg'], err_segs, res_dict['err_rate']


    def _del_rscode(self, bit_segs):
        log.debug('del rscode')
        res_dict = RsTools.del_rs(bit_segs, self.ori_len, self.codec_param.rs_num)
        err_segs = [bit_segs[i] for i in res_dict.get("err_index")]
        self.rs_err_rate = res_dict.get("err_rate")
        err_lists_tmp = [int(bit_segs[i][:self.index_length],2) for i in res_dict["err_index"]]
        for index in err_lists_tmp:
            if index < self.seq_num:
                self.rs_err_indexs.append(index)
        return res_dict['validate_bit_seg'], err_segs

    def _repair_segment(self, bit_segs):
        res_dict = SplitTools.repair_segment(self.index_length, bit_segs, self.seq_num,
                                             self.codec_param.add_redundancy)
        self.repaired_rate = res_dict['repaired_rate']
        self.repaired_indexs = res_dict['repaired_indexs']
        self.miss_err_rate = res_dict['failed_rate']
        self.miss_err_indexs = res_dict['failed_indexs']
        return res_dict["res_segments"]
    
    def _bin_to_file(self, bit_str):
        FileTools.bin_to_file(bit_str, self.output_file_path)

    def check_recover_ration(self,validate_bit_segs,err_bit_segs):
        try:
            messplainbits = self.readmessplain('messplains.txt')
        except FileNotFoundError:
            return 0
        for segment_index, bit_segment in enumerate(messplainbits):
            messplainbits[segment_index] = np.array([int(bit) for bit in bit_segment if bit != ' '], uint8)
            messplainbits[segment_index] = messplainbits[segment_index][self.index_length:]
        messplainbits = np.array(messplainbits)
        messlen = len(messplainbits)
        numt = 0
        decode_bits1 = np.full((len(messplainbits), len(messplainbits[0])), -1)
        for segment_index, bit_segment in enumerate(validate_bit_segs):
            # decode_bits1.append([int(biti) for biti in bit_segment][:])
            pos = int(bit_segment[:self.index_length], 2)
            if pos >= messlen:
                numt += 1
                continue
            thisbit = bit_segment[self.index_length:]
            for biti in range(len(thisbit)):
                if biti >= len(messplainbits[0]):
                    break
                #     # decode_bits1[segment_index][biti] = np.array([int(bit) for bit in bit_segment], uint8)
                decode_bits1[pos][biti] = int(thisbit[biti])
        for segment_index, bit_segment in enumerate(err_bit_segs):
            # decode_bits1.append([int(biti) for biti in bit_segment][:])
            pos = int(bit_segment[:self.index_length], 2)
            if pos >= messlen:
                numt += 1
                continue
            thisbit = bit_segment[self.index_length:]
            for biti in range(len(thisbit)):
                if biti >= len(messplainbits[0]):
                    break
                #     # decode_bits1[segment_index][biti] = np.array([int(bit) for bit in bit_segment], uint8)
                decode_bits1[pos][biti] = int(thisbit[biti])

        with open(self.file_base_name+'_messplains_dec.txt', 'w') as file:
            for data in decode_bits1:
                file.write(str(data) + '\n')

        badbits = np.count_nonzero(decode_bits1 - messplainbits)
        # badbits = np.count_nonzero(decode_bits - messplainbits)
        allbits = len(messplainbits) * len(messplainbits[0])
        recov = 1 - badbits / allbits
        print(f"badbits:{badbits},allbits:{allbits},bits recov:{recov}")
        self.infos['badbits']=badbits
        self.infos['allbits']=allbits
        self.infos['bits_recov']=recov
        return recov

    def common_decode(self):
        pass
        # """
        # common decode flow
        # """
        # # 序列形式解码：1.读取dna序列；2.解码并排序；3.rs纠错并删除；4.恢复原文件
        # tm_decode = datetime.now()
        # self._parse_param()
        # self.infos = dict()
        # # self.ori_len = self.codec_param.bin_seg_len
        # # tm_run = datetime.now()
        # base_line_list = self._get_base_line_list()
        # bit_segments = self._decode(base_line_list)
        # print(f"解码时间为{str(datetime.now() - tm_decode)}")
        # log.debug('sort')
        # # sorted_binstr = bit_segments
        # sorted_binstr = BaseTools.sort_segment(bit_segments, self.index_length)
        # sorted_binstr = sorted_binstr[:self.seq_num]
        #
        # # del rscode
        # if self.codec_param.rs_num>0:
        #     validate_bit_segs, err_bit_segs = self._del_rscode(sorted_binstr)
        # else:
        #     validate_bit_segs, err_bit_segs = sorted_binstr, []
        # self.infos['decode_time']=str(datetime.now() - tm_decode)
        # self.infos['decodefile']=self.output_file_path
        # self.infos['code']=0
        #
        # print(f"原有{len(sorted_binstr)}条，经过rs解码后，剩余：{len(validate_bit_segs)}条")
        # #repair and count missing segments
        # # log.debug('repair')
        # # validate_bit_segs = self._repair_segment(validate_bit_segs)
        #
        # recov = self.check_recover_ration(validate_bit_segs,err_bit_segs)
        #
        # log.debug('merge')
        # res_bit_str = SplitTools.merge(validate_bit_segs)
        # res_bit_str = res_bit_str[:self.codec_param.total_bit]
        #
        # log.debug('write')
        # self._bin_to_file(res_bit_str)
        #
        # return self.infos

    def getnumbers(self,dnaseqs):
        dna_sequences = []
        # thisseq = zeros([len(dnaseqs)], dtype=uint8)
        dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        for i in range(len(dnaseqs)):
            thisseq = np.zeros([self.totstrandlen], dtype=uint8)
            # thisseq = []
            for j in range(len(dnaseqs[i])):
                if j < self.totstrandlen:
                    thisseq[j] = dict[dnaseqs[i][j]]
            # dna_sequences.append(np.array(thisseq,dtype=uint8)))
            dna_sequences.append(thisseq[:self.totstrandlen])
        return np.array(dna_sequences, dtype=uint8)

    def readmessplain(self,path):
        returnarr = []
        with open(path, 'r') as file:
            lines = file.readlines()
        for i in range(len(lines)):
            returnarr.append(str(lines[i].strip('\n').strip(' ')))
        #     returnarr.append(line)
        return returnarr


    def extractplaintext(self,cpacket):
        # extract plaintext from a corrected packet
        plaintext=[]
        # plaintext = zeros(strandsperpacketmessage * messbytesperstrand, dtype=uint8)
        for i in range(0,len(cpacket)//self.strandsperpacket):
            for j in range((self.strandsperpacket - self.strandsperpacketcheck)):
                # plaintext.append(cpacket[i*self.strandsperpacket+j, self.strandIDbytes:self.strandIDbytes + self.messbytesperstrand])
                # plaintext.append(cpacket[i*self.strandsperpacket+j, 0:self.strandIDbytes + self.messbytesperstrand])
                plaintext.append(cpacket[i*self.strandsperpacket+j, self.strandIDbytes:self.strandIDbytes + self.messbytesperstrand])
                # plaintext[i * self.messbytesperstrand:(i + 1) * self.messbytesperstrand] = (cpacket[i, strandIDbytes:strandIDbytes + self.messbytesperstrand])
        return plaintext



    def readmessplainlist(self,path):
        returnarr = []
        with open(path, 'r') as file:
            lines = file.readlines()
        for i in range(len(lines)):
            line = lines[i].strip(' \n').split(' ')
            returnarr.append(np.array(line,dtype=int))
            # returnarr.append(line)
        return np.array(returnarr, dtype=int)

    def bytesTobits(self,decode_bytes):
        bit_segments = []
        for i in range(len(decode_bytes)):
            str = ''
            for j in range(len(decode_bytes[i])):
                str += bin(decode_bytes[i][j])[2:].rjust(8, '0')
            bit_segments.append(str)
        return np.array(bit_segments)

    # def common_decode_matrix_consus(self):
    #     # 矩阵形式解码：1.读取dna序列；2.将dna序列转换为矩阵形式的数据；3.对每个矩阵使用特定的解码方法进行解码；
    #     #               4.rs纠错并删除（可替换）5.数据转换为bit并排序；6.恢复原文件
    #
    #     self._parse_matrix_param()
    #     self.strandIDbytes = 2
    #     strandsperpacket = self.strandsperpacket
    #     tm_run = datetime.now()
    #     all_dna_sequences = self._get_base_line_list()
    #     npackets = int(np.ceil(len(all_dna_sequences) / strandsperpacket))
    #     segmentfor_decode = np.zeros([npackets * strandsperpacket, self.totstrandlen], dtype=np.uint8)
    #     for ipacket in range(npackets):
    #
    #         dna_sequences = all_dna_sequences[ipacket * strandsperpacket:(ipacket + 1) * strandsperpacket]
    #         dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    #         for i in range(len(dna_sequences)):
    #             thisseq = np.zeros([self.totstrandlen], dtype=uint8)
    #             for j in range(len(dna_sequences[i])):
    #                 if j < self.totstrandlen:
    #                     thisseq[j] = dict[dna_sequences[i][j]]
    #             segmentfor_decode[ipacket * strandsperpacket+i] = thisseq
    #
    #     # decode
    #     outfilepath = self._decode(segmentfor_decode)
    #     decode_numbers_ori = self.readmessplainlist(outfilepath)
    #     decode_bytes = self.extractplaintext(decode_numbers_ori)
    #     decode_bits = self.bytesTobits(decode_bytes)
    #     self.judgeoutcomes(decode_bytes,decode_bits)
    #
    #     self.seq_num = len(decode_numbers_ori)
    #     self.index_length = self.strandIDbytes*8
    #     log.debug('sort')
    #     sorted_bit_segments = BaseTools.sort_segment(decode_bits, self.index_length)
    #     log.debug('repair')
    #     validate_bit_segs = self._repair_segment(sorted_bit_segments)
    #     log.debug('merge')
    #     res_bit_str = SplitTools.merge(validate_bit_segs)
    #     res_bit_str = res_bit_str[:self.codec_param.total_bit]
    #
    #     log.debug('write')
    #     self._bin_to_file(res_bit_str)
    #
    #     self.run_time = datetime.now() - tm_run
    #     return self.output_file_path
    #
    #
    #     # sorted_binstr = []
    #     # for ipacket in range(npackets):
    #     #     thisbit_segments = sorted_bit_segments[
    #     #                        ipacket * strandsperpacket:(ipacket + 1) * strandsperpacket]
    #     #     sorted_binstr.append(thisbit_segments[:(strandsperpacket - self.strandsperpacketcheck)])
    #     # print(111)
    #
    #     # self.decode_time = str(datetime.now() - tm_decode)
    #     # log.debug('sort')
    #     # sorted_binstr = BaseTools.sort_segment(bit_segments, self.index_length)
    #     # sorted_binstr = sorted_binstr[:self.seq_num]
    #     #
    #     # # del rscode
    #     # if self.codec_param.rs_num>0:
    #     #     validate_bit_segs, err_bit_segs = self._del_rscode(sorted_binstr)
    #     # else:
    #     #     validate_bit_segs, err_bit_segs = sorted_binstr, []
    #     # #repair and count missing segments
    #     # log.debug('repair')
    #     # validate_bit_segs = self._repair_segment(validate_bit_segs)
    #     #
    #     # log.debug('merge')
    #     # res_bit_str = SplitTools.merge(validate_bit_segs)
    #     # res_bit_str = res_bit_str[:self.codec_param.total_bit]
    #     #
    #     # log.debug('write')
    #     # self._bin_to_file(res_bit_str)
    #     #
    #     # self.run_time = str(datetime.now() - tm_run)
    #     # return self.output_file_path

    def file_binary_to_random(self,binary_sequence_ori):
        # 与圆周率的前500000位，进行异或操作，以实现随机化
        with open('./pi_str.txt', 'rb') as file:
            content = file.read()
            pi_binary_sequence = ''.join(format(byte, '08b') for byte in content)
        # print("pi_binary_sequence的长度：", len(pi_binary_sequence))
        str_1, str_2 = binary_sequence_ori, pi_binary_sequence
        len_str_1 = len(str_1)
        len_str_2 = len(str_2)
        # 使用列表推导式生成结果字符串,获取str_2中对应位置的字符，如果长度不够则循环使用
        binary_sequence_random = ''.join(str(int(str_1[i]) ^ int(str_2[i % len_str_2])) for i in range(len_str_1))
        # print("binary_sequence_ori的长度：", len(binary_sequence_ori))
        # print("binary_sequence_random的长度：", len(binary_sequence_random))
        return binary_sequence_random

    def common_decode_matrix(self):
        pass
        # # 矩阵形式解码：1.读取dna序列；2.将dna序列转换为矩阵形式的数据；3.对每个矩阵使用特定的解码方法进行解码；
        # #               4.rs纠错并删除（可替换）5.数据转换为bit并排序；6.恢复原文件
        #
        # self._parse_matrix_param()
        # self.strandIDbytes = 2
        # self.strandrunoutbytes = 2
        #
        # strandsperpacket = self.strandsperpacket
        # npackets = self.npackets
        # tm_run = datetime.now()
        # all_dna_sequences = self._get_base_line_list()
        # # npackets = int(np.ceil(len(all_dna_sequences) / strandsperpacket))
        # segmentfor_decode = np.zeros([len(all_dna_sequences), self.totstrandlen], dtype=np.uint8)
        # dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        # for i in range(len(all_dna_sequences)):
        #     thisseq = np.zeros([self.totstrandlen], dtype=uint8)
        #     for j in range(len(all_dna_sequences[i])):
        #         if j < self.totstrandlen:
        #             thisseq[j] = dict[all_dna_sequences[i][j]]
        #     segmentfor_decode[i] = thisseq
        #
        # # decode
        # start_time = datetime.now()
        # print(f"开始解码：")
        # outfilepath = self._decode(segmentfor_decode)
        # print(f"解码时间：{str(datetime.now() - start_time)}")
        # decode_numbers_ori = self.readmessplainlist(outfilepath)
        # decode_bytes = self.extractplaintext(decode_numbers_ori)
        # decode_bits = self.bytesTobits(decode_bytes)
        #
        # self.seq_num = len(decode_numbers_ori)
        # self.index_length = self.strandIDbytes*8
        # log.debug('sort')
        # sorted_bit_segments = BaseTools.sort_segment(decode_bits, self.index_length)
        # log.debug('repair')
        # validate_bit_segs = self._repair_segment(sorted_bit_segments)
        # log.debug('merge')
        # res_bit_str = SplitTools.merge(validate_bit_segs)
        #
        # res_bit_str = res_bit_str[:self.codec_param.total_bit]
        # res_bit_str = self.file_binary_to_random(res_bit_str)
        # log.debug('write')
        # self._bin_to_file(res_bit_str)
        #
        # self.judgeoutcomes(decode_bytes,decode_bits)
        # self.run_time = datetime.now() - tm_run
        # return self.output_file_path


        # sorted_binstr = []
        # for ipacket in range(npackets):
        #     thisbit_segments = sorted_bit_segments[
        #                        ipacket * strandsperpacket:(ipacket + 1) * strandsperpacket]
        #     sorted_binstr.append(thisbit_segments[:(strandsperpacket - self.strandsperpacketcheck)])
        # print(111)

        # self.decode_time = str(datetime.now() - tm_decode)
        # log.debug('sort')
        # sorted_binstr = BaseTools.sort_segment(bit_segments, self.index_length)
        # sorted_binstr = sorted_binstr[:self.seq_num]
        #
        # # del rscode
        # if self.codec_param.rs_num>0:
        #     validate_bit_segs, err_bit_segs = self._del_rscode(sorted_binstr)
        # else:
        #     validate_bit_segs, err_bit_segs = sorted_binstr, []
        # #repair and count missing segments
        # log.debug('repair')
        # validate_bit_segs = self._repair_segment(validate_bit_segs)
        #
        # log.debug('merge')
        # res_bit_str = SplitTools.merge(validate_bit_segs)
        # res_bit_str = res_bit_str[:self.codec_param.total_bit]
        #
        # log.debug('write')
        # self._bin_to_file(res_bit_str)
        #
        # self.run_time = str(datetime.now() - tm_run)
        # return self.output_file_path


