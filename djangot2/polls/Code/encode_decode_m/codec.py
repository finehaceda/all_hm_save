# -*- coding: utf-8 -*-
import copy
import csv
import logging
import math
import subprocess
import sys
import traceback
from datetime import datetime
from os import path, stat

import numpy as np
from numpy import uint8

from .PolarDna.polarmain import file_to_dna, dna_to_file
from .PolarDna.utils import data_handle
# from .yinyangcode import yinyangencode,yinyangdecode
from .yinyangcode_withphred import yinyangencode,yinyangdecode
from .DNAFountain.decode import Decode
from .DNAFountain.encode import Encode
from .abstract_codec import AbstractEncode, AbstractDecode
from .church import churchEncode, churchDecode
from .goldman import goldmanEncode, goldmanDecode
from .dnafountain import dnafountainEncode, dnafountainDecode
# from .hedges_encode_decode import hedgesEncode, hedgesDecode
from .tools import DecodeParameter, BaseTools, SplitTools
from ..utils import getbinfile

log = logging.getLogger('mylog')

"""
Reference:
* Church, G. M., et al. (2012). "Next-generation digital information storage in DNA." Science 337(6102): 1628.
* DOI: 10.1126/science.1226355
* https://pubmed.ncbi.nlm.nih.gov/22903519/
"""

class DNAFountainEncode(AbstractEncode):
    def __init__(self, input_file_path: str, output_dir: str, sequence_length: int, max_homopolymer=6,
                 rs_num=0, add_redundancy=False, add_primer=False, primer_length=20, redundancy=0.07,
                 gc_bias=0.2,c_dist=0.03,delta=0.5,crc_num=0):
        index_redundancy = 0  # for virtual segments in wukong
        seq_bit_to_base_ratio = 1
        # self.max_homopolymer = max_homopolymer
        self.redundancy = redundancy
        self.gc_bias = gc_bias
        self.crc_num = crc_num
        self.c_dist = c_dist
        self.delta = delta
        super().__init__(input_file_path, output_dir, 'DNAFountain', seq_bit_to_base_ratio=seq_bit_to_base_ratio,
                         index_redundancy=index_redundancy,
                         sequence_length=sequence_length, max_homopolymer=max_homopolymer, rs_num=rs_num,
                         add_redundancy=add_redundancy,
                         add_primer=add_primer, primer_length=primer_length)

    # def getgc(self,all_seqs):
    #     avggc = 0
    #     for dna_sequence in all_seqs:
    #         gc_count = dna_sequence.count('G') + dna_sequence.count('C')
    #         total_count = len(dna_sequence)
    #         avggc += gc_count / total_count
    #     return avggc/len(all_seqs)

    # def getgc(self,all_seqs):
    #     avggc = 0
    #     all_gc = []
    #     for dna_sequence in all_seqs:
    #         gc_count = dna_sequence.count('G') + dna_sequence.count('C')
    #         gc = gc_count / len(dna_sequence)
    #         avggc += gc
    #         all_gc.append(gc)
    #     with open(f'gcfiles/gc_{self.tool_name}.csv', mode='w', newline='') as file:
    #         writer = csv.writer(file)
    #         for g in all_gc:
    #             t = (g,)
    #             writer.writerow(t)
    #     return avggc / len(all_seqs)

    # 龙哥写的
    # homopolymer = 4
    # gc_bias = 0.2
    # redundancy = 0.09
    # c_dist = 0.03
    # delta = 0.5
    # # c_dist = 0.1  # 这个默认值，效果不好。上面的效果好
    # # delta = 0.05
    # header_size = 4  # 刚好对应seed有4 * 8 = 32 bit，对应16nt，与DNAFountain方法保持一致

    def _set_param_common(self,method='fountain'):
        param = "method:{},seq_len:{},gc_bias:{},max_homopolymer:{},rs_num:{},crc_num:{},redundancy:{},delta:{},c_dist:{},chunk_num:{}\n".format(
            method,self.codec_param.sequence_length,self.gc_bias, self.codec_param.max_homopolymer
            , self.codec_param.rs_num, self.crc_num, self.redundancy,self.delta,self.c_dist,self.chunk_num)
        return param

    def common_encode(self):
        tm_encode = datetime.now()
        encodefile = self.output_file_path
        # size = (self.codec_param.sequence_length-16-self.codec_param.rs_num*4)//4
        # 加入crc
        size = (self.codec_param.sequence_length-16-self.codec_param.rs_num*4-4*self.crc_num)//4

        # self.redundancy = 0.09,
        self.chunk_num = math.ceil((path.getsize(self.input_file_path)) / size)
        Encode(file_in=self.input_file_path, out=self.output_file_path,
               size=size,
               rs=self.codec_param.rs_num,  ## reedsolo code byte number
               alpha=self.redundancy,
               max_homopolymer = self.codec_param.max_homopolymer,
               gc = self.gc_bias,
               c_dist = self.c_dist,
               delta = self.delta,
               # max_homopolymer=self.codec_param.max_homopolymer,
               # gc=0.05,
               # delta=0.001,
               # c_dist=0.025,
               # alpha=0.1,
               ## ensure stop > chunk size
               stop=self.chunk_num * (1+self.redundancy),
               no_fasta=True,
               crc_num=self.crc_num).main()
        # encodeinfo={}

        self.encode_time = str(datetime.now() - tm_encode)
        with open(encodefile, 'r', encoding='utf-8') as f:
            old_content = f.readlines()
        param = self._set_param_common('fountain')
        dnaseqs = [old_content[i].strip('\n') for i in range(len(old_content))]
        with open(encodefile, 'w', encoding='utf-8') as f:
            f.write(param)
            # f.writelines(old_content)
            for i in range(len(dnaseqs)):
                f.write(f">seq{i}\n{dnaseqs[i]}\n")


        self.total_base = len(dnaseqs)  * len(dnaseqs[0])
        self.seq_num = len(dnaseqs)
        self.seq_len = len(dnaseqs[0])
        self._set_density()
        self.gc = self.getgc(dnaseqs)
        self.max_homopolymer = self.max_homopolymer_length(dnaseqs)
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print(f"total_bit:{self.total_bit},total_base:{self.total_base},dnanums:{len(dnaseqs)},dnaseqslen:{len(dnaseqs[0])},density:{self.density},real_gc_bias:{self.gc}")

    def _encode(self):
        pass

    # def _encode(self, bit_segments, bit_size):
    #
    #     base_segments = dnafountainEncode(bit_segments, redundancy=self.redundancy,homopolymer=self.homopolymer,bit_size=bit_size)
    #     return base_segments


class DNAFountainDecode(AbstractDecode):
    def __init__(self, input_file_path: str, orifile:str, output_dir: str, chunk_num:int, max_homopolymer=4, rs_num=0,
                 gc_bias=0.2, c_dist=0.03, delta=0.5, crc_num=0, phreds=None, decision='hard'):
        super().__init__(input_file_path, output_dir)
        self.output_file_path=input_file_path.rpartition('.')[0] + "_decode." + orifile.rpartition('.')[2]
        self.chunk_num=chunk_num
        self.orifile=orifile
        self.max_homopolymer=max_homopolymer
        self.rs_num=rs_num
        self.returncode = 0
        self.gc_bias = gc_bias
        self.c_dist = c_dist
        self.delta = delta
        self.crc_num = crc_num
        self.phreds = phreds
        self.decision = decision

    # def getbinfile(self,path):
    #     bin_to_str_list = list()
    #     for i in range(256):
    #         bin_to_str_list.append(bin(i)[2:].rjust(8, '0'))
    #     with open(path, 'rb') as f:
    #         bytesarr = f.read()
    #     str_list = [bin_to_str_list[i] for i in bytesarr]
    #     binstring = ''.join(str_list)
    #     binstring_bits = np.array([bit for bit in binstring], np.uint8)
    #     return binstring_bits

    def judgedrrick(self):
        ori_bits=getbinfile(self.orifile)
        decode_bits = getbinfile(self.output_file_path)

        if len(decode_bits) > len(ori_bits):
            badbits = np.count_nonzero(ori_bits - decode_bits[:ori_bits.size])
        else:
            print('??????????????????????len(decode_bits)<len(ori_bits)????????????????????????????')
            badbits = np.count_nonzero(ori_bits[:decode_bits.size] - decode_bits)

        # badbits = np.count_nonzero(ori_bits - decode_bits[:ori_bits.size])
        # badbits = np.count_nonzero(decode_bits - messplainbits)
        allbits = ori_bits.size
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print(f"badbits:{badbits},allbits:{allbits},bits recov:{1 - badbits / allbits}")
        self.returncode = 0

        infos = {
            'code': self.returncode,
            'badbits':badbits,
            'allbits': allbits,
            'bits_recov': 1 - badbits / allbits,
            'decodefile':self.output_file_path,
        }
        return infos

    def common_decode(self):
        tm_encode = datetime.now()
        infos = dict()
        print(f"------开始解码------")
        try:
            Decode(file_in=self.input_file_path, out=self.output_file_path,
                   header_size=4,  ## seed size
                   chunk_num=self.chunk_num,  ## source file chunk number
                   rs=self.rs_num,  ## reedsolo code byte number
                   # max_homopolymer=4,
                   # gc=0.2,
                   # c_dist=0.03,
                   # delta=0.5,

                   max_homopolymer=self.max_homopolymer,
                   gc=self.gc_bias,
                   c_dist=self.c_dist,
                   delta=self.delta,

                   # delta=0.001,
                   # c_dist=0.025,
                   # max_homopolymer=self.max_homopolymer,
                   # gc=0.05,
                   max_hamming=0).main(self.crc_num,self.decision,self.phreds)
                   # max_hamming=0).main(self.crc_num)
            print(f"------解码结束------")
            infos = self.judgedrrick()
        except SystemExit:
            infos = {'code': 1}
            print(f"------解码失败------")
            infos['decode_time'] = str(datetime.now() - tm_encode)
            return infos
        except ValueError:
            infos = {'code': 1}
            print(f"------解码成功，比对失败------")
            print(f"self.orifile:{self.orifile},self.output_file_path:{self.output_file_path}")
            infos['decode_time'] = str(datetime.now() - tm_encode)

            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(f"完整的堆栈跟踪信息:\n{''.join(traceback.format_exception(exc_type, exc_obj, exc_tb))}")
            return infos
        else:
            infos['decode_time'] = str(datetime.now() - tm_encode)
            print(f"infos:{infos}")
            return infos

    def _decode(self):
        pass


    # def _decode(self, base_line_list):
    #     bit_segments = dnafountainDecode(base_line_list,decode_packets=self.decode_packets,segment_length=self.sequence_length)
    #     return bit_segments


class DerrickEncode(AbstractEncode):
    def __init__(self, input_file_path:str, output_dir:str, sequence_length:int, max_homopolymer=6,
                 rs_num=0, add_redundancy=False, add_primer=False, primer_length=20,
                                 matrix_code=False,matrix_n=0,matrix_r=0):
        index_redundancy = 0 #for virtual segments in wukong
        seq_bit_to_base_ratio = 1

        if matrix_code:
            self.matrix_n = matrix_n
            self.matrix_k = matrix_n - matrix_r
        super().__init__(input_file_path, output_dir, 'derrick', seq_bit_to_base_ratio=seq_bit_to_base_ratio, index_redundancy=index_redundancy,
                         sequence_length=sequence_length, max_homopolymer=max_homopolymer, rs_num=rs_num, add_redundancy=add_redundancy,
                         add_primer=add_primer, primer_length=primer_length)
        self.codec_param.sequence_length = self.codec_param.sequence_length//4*4
    def readtxt(self,file_path):
        allseqs = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
        for i in range(len(lines)):
            allseqs.append(lines[i].strip('\n'))
        return allseqs

    def seqs_plus_indexs(self,dna_sequences):
        indexs = self.readtxt('./address.wjr.txt')
        indexs = indexs[:len(dna_sequences)]
        # 给序列添加index
        newdna_sequences = []
        for i in range(len(dna_sequences)):
            newdna_sequences.append(indexs[i] + dna_sequences[i])
        return newdna_sequences, indexs

    # def getgc(self,all_seqs):
    #     avggc = 0
    #     for dna_sequence in all_seqs:
    #         gc_count = dna_sequence.count('G') + dna_sequence.count('C')
    #         total_count = len(dna_sequence)
    #         avggc += gc_count / total_count
    #     return avggc/len(all_seqs)

    # def getgc(self,all_seqs):
    #     avggc = 0
    #     all_gc = []
    #     for dna_sequence in all_seqs:
    #         gc_count = dna_sequence.count('G') + dna_sequence.count('C')
    #         gc = gc_count / len(dna_sequence)
    #         avggc += gc
    #         all_gc.append(gc)
    #     with open(f'gcfiles/gc_{self.tool_name}.csv', mode='w', newline='') as file:
    #         writer = csv.writer(file)
    #         for g in all_gc:
    #             t = (g,)
    #             writer.writerow(t)
    #     return avggc / len(all_seqs)

    def _set_param_common(self,method='derrick'):
        param = "method:{},seq_len:{},matrix_n:{},matrix_r:{}\n".format(
            method,self.codec_param.sequence_length,self.matrix_n,self.matrix_n - self.matrix_k)
        return param

    def common_encode_matrix(self):
        tm_encode = datetime.now()
        encodefile = self.output_file_path
        # shell = f"derrick/derrick encode -i ./derrick/pi.txt -n {self.matrix_n} -k {self.matrix_k} -s {self.codec_param.sequence_length//4}" \
        #         f" {self.input_file_path} > {self.output_file_path}"
        shell = f"./derrick/derrick encode -i ./derrick/pi.txt -n {self.matrix_n} -k {self.matrix_k} -s {self.codec_param.sequence_length//4}" \
                f" {self.input_file_path} > {self.output_file_path}"
        print(f"shell encode:{shell}")
        result = subprocess.run(shell, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(result.stderr)

        self.encode_time = str(datetime.now() - tm_encode)
        with open(encodefile, 'r', encoding='utf-8') as f:
            old_content = f.readlines()
        dnaseqs = [old_content[i].strip('\n') for i in range(1,len(old_content),2)]
        dnaseqs,indexs = self.seqs_plus_indexs(dnaseqs)
        param = self._set_param_common('derrick')
        with open(encodefile, 'w', encoding='utf-8') as f:
            f.write(f"{param}")
            for i in range(len(dnaseqs)):
                f.write(f">seq{i}\n{dnaseqs[i]}\n")
                # f.write(f"{dnaseqs[i]}\n")
        mygc = self.getgc(dnaseqs)
        self.max_homopolymer = self.max_homopolymer_length(dnaseqs)

        self.gc=mygc
        self.total_base = len(dnaseqs)  * (len(dnaseqs[0]))
        self.seq_num = len(dnaseqs)
        self._set_density()

        print(f"total_bit:{self.total_bit},total_base:{self.total_base},dnanums:{len(dnaseqs)},density:{self.density},mygc:{mygc}")

    def _encode(self):
        pass


class DerrickDecode(AbstractDecode):
    def __init__(self, input_file_path:str, output_dir:str,inputfileforcompare='',matrix_code=False,matrix_n=0,matrix_r=0,sequence_length=260):
        super().__init__(input_file_path, output_dir)
        if matrix_code:
            self.matrix_n = matrix_n
            self.matrix_k = matrix_n - matrix_r
        self.sequence_length = sequence_length
        self.inputfileforcompare = inputfileforcompare
        self.output_file_path = self.output_dir + self.file_base_name + "_decode.jpg"

    def judgedrrick(self):
        infos={}
        try:
            bin_to_str_list = list()
            for i in range(256):
                bin_to_str_list.append(int(bin(i)[2:],2))
            with open(self.inputfileforcompare, 'rb') as f:
                bytesarr = f.read()
            # bytesarr = [bin_to_str_list[i] for i in bytesarr]
            bytesarr = np.array([bin_to_str_list[i] for i in bytesarr],dtype=np.uint8)
            with open(self.output_file_path, 'rb') as f:
                bytesarrout = f.read()
            bytesarrout = np.array([bin_to_str_list[i] for i in bytesarrout],dtype=np.uint8)

            badbytes = np.count_nonzero(bytesarrout - bytesarr)
            allbytes = len(bytesarr)
            print(f"badbytes:{badbytes},allbytes:{allbytes},bytes recov:{1-badbytes/allbytes}")

            messplainbits = self.bytesTobits([bytesarr])
            decode_bits = self.bytesTobits([bytesarrout])
            decode_bits1 = np.array([np.array([bit for bit in bits], np.uint8) for bits in decode_bits])
            messplainbits1 = np.array([np.array([bit for bit in bits], np.uint8) for bits in messplainbits])
            badbits = np.count_nonzero(decode_bits1 - messplainbits1)
            # badbits = np.count_nonzero(decode_bits - messplainbits)
            allbits = len(messplainbits) * len(messplainbits[0])
            print(f"badbits:{badbits},allbits:{allbits},bits recov:{1 - badbits / allbits}")
            infos = {
                'code': 0,
                'badbits': badbits,
                'allbits': allbits,
                'bits_recov': 1 - badbits / allbits,
                'decodefile': self.output_file_path,
            }
        except ValueError:
            infos = {'code':1}
        except FileNotFoundError:
            infos = {'code':1}
        except FileExistsError:
            infos = {'code':1}
        finally:
            return infos


    def common_decode_matrix(self):
        tm_decode = datetime.now()
        with open(self.input_file_path, 'r', encoding='utf-8') as f:
            old_content = f.readlines()
        dnaseqs = [old_content[i].strip('\n')[12:] for i in range(len(old_content))]
        # dnaseqs = [old_content[i].strip('\n') for i in range(len(old_content))]
        if len(dnaseqs) == 0:
            return {'code':1,'errorinfo':'无序列可解码！'}
        print(f"------derrick解码前处理------，去掉index，加上>seq\n需要解码的序列共有{len(dnaseqs)}条！！每条{len(dnaseqs[0])}nt")
        print(f"self.input_file_path:{self.input_file_path}")
        with open(self.input_file_path, 'w', encoding='utf-8') as f:
            for i in range(len(dnaseqs)):
                f.write(f">seq{i}\n{dnaseqs[i]}\n")
        shell = f"derrick/derrick decode -i ./derrick/pi.txt -n {self.matrix_n} -k {self.matrix_k}  -s {self.sequence_length//4}" \
                f"  {self.input_file_path} > {self.output_file_path}"
        print(f"------开始解码------")
        result = subprocess.run(shell, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(f"shell:{shell}\nresult.stderr:{result.stderr}\n------解码结束------\n")
        infos = self.judgedrrick()
        infos['decode_time'] = str(datetime.now() - tm_decode)
        return infos




    def _decode(self):
        pass


class PolarEncode(AbstractEncode):
    def __init__(self, input_file_path:str, output_dir:str, frozen_bits_len:int,):
        self.input_file_path = input_file_path
        self.tool_name = 'PolarCode'
        self.file_base_name,_ = self.get_file_name_extension(self.input_file_path)
        self.output_file_path = output_dir + self.file_base_name + "_{}.fasta".format(self.tool_name)
        self.frozen_bits_len = frozen_bits_len

    def get_file_name_extension(self,file_path_or_name):
        s = path.basename(file_path_or_name)
        file_name = s[:s.index('.'):] if '.' in s else s
        file_extension = s[s.index('.'):] if '.' in s else ''
        return file_name, file_extension

    def getgc(self,all_seqs):
        avggc = 0
        all_gc = []
        for dna_sequence in all_seqs:
            gc_count = dna_sequence.count('G') + dna_sequence.count('C')
            gc = gc_count / len(dna_sequence)
            avggc += gc
            all_gc.append(gc)
        with open(f'gcfiles/{self.tool_name}_{self.file_base_name}.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            for g in all_gc:
                t = (g,)
                writer.writerow(t)
        return avggc / len(all_seqs)


    def max_homopolymer_length(self,all_dna_sequence):
        max_length = 1  # 最短均聚物长度为1（单个核苷酸）
        current_length = 1  # 当前检查的均聚物长度
        all_homopolymer = []
        for dna_sequence in all_dna_sequence:
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
            all_homopolymer.append(max_length)
        with open(f'homopolymerfiles/{self.tool_name}_{self.file_base_name}.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            for g in all_homopolymer:
                t = (g,)
                writer.writerow(t)
        return max_length


    def _set_param_common(self,method='PolarCode'):
        param = "method:{},seq_len:{},homopolymer:{},frozen_bits_len:{}\n".format(
            method,260,4,self.frozen_bits_len)
        return param

    def common_encode(self):
        tm_encode = datetime.now()
        dna_sequences, matrices_ori, matrices_dna_ori, matrices_01_ori = file_to_dna(self.input_file_path,self.frozen_bits_len)
        self.encode_time = str(datetime.now() - tm_encode)

        param = self._set_param_common()
        with open(self.output_file_path, 'w') as f:
            f.write(param)
            # for i in range(len(dna_sequences)):
            #     f.write(dna_sequences[i] + '\n')
            for i in range(len(dna_sequences)):
                f.write(f">seq{i}\n{dna_sequences[i]}\n")

        mygc= self.getgc(dna_sequences)
        self.matrices_ori = matrices_ori
        self.matrices_dna_ori = matrices_dna_ori
        self.matrices_01_ori = matrices_01_ori
        self.gc = mygc
        self.max_homopolymer = self.max_homopolymer_length(dna_sequences)
        self.total_base = len(dna_sequences)  * (len(dna_sequences[0]))
        self.seq_num = len(dna_sequences)
        self.total_bit =  8*stat(self.input_file_path).st_size
        self.density = self.total_bit / self.total_base

        print(f"gcfile:gc_{self.tool_name}.csv")
        # print(f"total_bit:{self.total_bit},total_base:{self.total_base},density:{self.density},mygc:{mygc}")
        print(f"total_bit:{self.total_bit},total_base:{self.total_base},density:{self.density},dnanums:{len(dna_sequences)}")
        print(f"total_base:{self.total_base},mygc:{mygc}")

    def _encode(self):
        pass

class PolarDecode(AbstractDecode):
    def __init__(self, input_file_path: str, orifile: str, output_dir: str,matrices_ori,matrices_dna_ori,matrices_01_ori,frozen_bits_len=5):
        super().__init__(input_file_path, output_dir)
        self.orifile=orifile
        self.frozen_bits_len=int(frozen_bits_len)
        self.output_file_path = self.output_dir + self.file_base_name + "_decode.jpg"
        self.matrices_ori, self.matrices_dna_ori, self.matrices_01_ori=matrices_ori,matrices_dna_ori,matrices_01_ori

    def judgedrrick(self):
        infos = {}
        try:
            bin_to_str_list = list()
            for i in range(256):
                bin_to_str_list.append(int(bin(i)[2:], 2))
            with open(self.inputfileforcompare, 'rb') as f:
                bytesarr = f.read()
            # bytesarr = [bin_to_str_list[i] for i in bytesarr]
            bytesarr = np.array([bin_to_str_list[i] for i in bytesarr], dtype=np.uint8)
            with open(self.output_file_path, 'rb') as f:
                bytesarrout = f.read()
            bytesarrout = np.array([bin_to_str_list[i] for i in bytesarrout], dtype=np.uint8)

            badbytes = np.count_nonzero(bytesarrout - bytesarr)
            allbytes = len(bytesarr)
            print(f"badbytes:{badbytes},allbytes:{allbytes},bytes recov:{1 - badbytes / allbytes}")

            messplainbits = self.bytesTobits([bytesarr])
            decode_bits = self.bytesTobits([bytesarrout])
            decode_bits1 = np.array([np.array([bit for bit in bits], np.uint8) for bits in decode_bits])
            messplainbits1 = np.array([np.array([bit for bit in bits], np.uint8) for bits in messplainbits])
            badbits = np.count_nonzero(decode_bits1 - messplainbits1)
            # badbits = np.count_nonzero(decode_bits - messplainbits)
            allbits = len(messplainbits) * len(messplainbits[0])
            print(f"badbits:{badbits},allbits:{allbits},bits recov:{1 - badbits / allbits}")
            infos = {
                'code': 0,
                'badbits': badbits,
                'allbits': allbits,
                'bits_recov': 1 - badbits / allbits,
                'decodefile': self.output_file_path,
            }
        except ValueError:
            infos = {'code': 1}
        except FileNotFoundError:
            infos = {'code': 1}
        except FileExistsError:
            infos = {'code': 1}
        finally:
            return infos

    def readseqsandphreds(self):
        dna_sequences_clear, phred_scores = [],[]
        with open(self.input_file_path + '.phred', 'r') as f:
            lines = f.readlines()
        for i in range(len(lines)):
            # print(lines[i].rstrip().split(' '))
            phred_scores.append(list(map(float, lines[i].rstrip().split(' '))))
        with open(self.input_file_path, 'r') as f:
            lines = f.readlines()
        if not phred_scores:
            for i in range(len(lines)):
                dna_sequences_clear.append(lines[i].rstrip())
                phred_scores.append([0.99 for _ in range(len(lines[i].rstrip()))])
        else:
            print(f'there exist confidence!')
            for i in range(len(lines)):
                dna_sequences_clear.append(lines[i].rstrip())
        return dna_sequences_clear, phred_scores

    def common_decode(self):
        tm_encode = datetime.now()
        infos = dict()
        try:
            frozen_bits_len = self.frozen_bits_len
            dna_sequences_clear, phred_scores = self.readseqsandphreds()
            dna_seq_and_phred = data_handle.dna_seq_add_phred_scores(dna_sequences_clear, phred_scores)
            block_recover_ration,bit_recover,badbits,allbits = dna_to_file(dna_seq_and_phred, self.matrices_ori, self.matrices_dna_ori, self.matrices_01_ori,self.output_file_path, self.orifile, 1,frozen_bits_len)

        except FileNotFoundError:
            infos = {'code': 1, 'errorinfo': 'FileNotFoundError'}
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(f"完整的堆栈跟踪信息:\n{''.join(traceback.format_exception(exc_type, exc_obj, exc_tb))}")
        except IndexError:
            infos = {'code': 1, 'errorinfo': 'IndexError'}
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(f"完整的堆栈跟踪信息:\n{''.join(traceback.format_exception(exc_type, exc_obj, exc_tb))}")
        except ValueError:
            infos = {'code': 1, 'errorinfo': 'ValueError'}
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(f"完整的堆栈跟踪信息:\n{''.join(traceback.format_exception(exc_type, exc_obj, exc_tb))}")
        except Exception as e:

            infos = {'code': 1, 'errorinfo': {e}}
            print(f"异常类型: {type(e)}")
            print(f"异常信息: {e}")
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(f"完整的堆栈跟踪信息:\n{''.join(traceback.format_exception(exc_type, exc_obj, exc_tb))}")
        else:
            infos = {
                'code': 0,
                'badbits': badbits,
                'allbits': allbits,
                'bits_recov': bit_recover,
                'decodefile': self.output_file_path,
                # 'decode_time': run_time,
            }
        finally:
            run_time = datetime.now() - tm_encode
            infos['decode_time']=run_time
            print(f'解码共耗时：{run_time}')
            return infos

    def _decode(self):
        pass


class HedgesEncode(AbstractEncode):
    def __init__(self, input_file_path:str, output_dir:str, sequence_length:int, max_homopolymer=6,
                 rs_num=0, add_redundancy=False, add_primer=False, primer_length=20,
                                 matrix_code=False,matrix_n=0,matrix_r=0,coderatecode=0.5):
        index_redundancy = 0
        seq_bit_to_base_ratio = 1
        self.coderatecode = coderatecode
        if matrix_code:
            self.matrix_n = matrix_n
            self.matrix_r = matrix_r
        # if matrix_code:self.__matrix_init__(matrix_n,matrix_r)
        super().__init__(input_file_path, output_dir, 'hedges', seq_bit_to_base_ratio=seq_bit_to_base_ratio, index_redundancy=index_redundancy,
                         sequence_length=sequence_length, max_homopolymer=max_homopolymer, rs_num=rs_num, add_redundancy=add_redundancy,
                         add_primer=add_primer, primer_length=primer_length)

    def _set_param_common(self,method='hedges'):
        param = "method:{},seq_len:{},max_homopolymer:{},matrix_n:{},matrix_r:{},coderatecode:{},".format(
            method,self.codec_param.sequence_length,self.codec_param.max_homopolymer,self.matrix_n,self.matrix_r,self.coderatecode,)

        param += self._set_matrix_param_line()
        return param

    def common_encode_matrix(self):
        # # 矩阵形式编码
        # 1.读取文件；2.根据文件大小，rs码个数，序列长度（及码率）计算总共有多少个矩阵；3.填充矩阵（+rs）4.编码并保存序列
        tm_encode = datetime.now()
        self._check_params()
        self.__matrix_init__(self.matrix_n, self.matrix_r, self.coderatecode)
        self.norsway = False
        log.debug('read file')
        # bin_to_str_list = list()
        # for i in range(256):
        #     bin_to_str_list.append(int(bin(i)[2:],2))
        bin_to_str_list = list()
        for i in range(256):
            bin_to_str_list.append(bin(i)[2:].rjust(8, '0'))
        with open(self.input_file_path, 'rb') as f:
            bytesarr = f.read()
        # self.wizbytes = [bin_to_str_list[i] for i in bytesarr]
        str_list = [bin_to_str_list[i] for i in bytesarr]
        binstring = ''.join(str_list)
        binstring = self.file_binary_to_random(binstring)
        self.wizbytes = [int(binstring[i:i + 8], 2) for i in range(0, len(binstring), 8)]
        #长度不足则补0
        self.wizbytes.extend([0] * (self.messbytesperstrand - len(self.wizbytes) % self.messbytesperstrand))

        self.wizlen = len(self.wizbytes)
        self.npackets = int(math.ceil(float(self.wizlen) / self.messbytesperstrand / self.strandsperpacketmessage))
        self.bytesseqnum = int(math.ceil(float(self.wizlen) / self.messbytesperstrand))
        npackets=self.npackets
        # 填充字节数据到矩阵
        messplains = np.zeros([npackets * self.strandsperpacketmessage,self.bytesperstrand-self.strandrunoutbytes-self.strandIDbytes], dtype=uint8)
        segmentfor_encode = np.zeros([npackets * self.strandsperpacket,self.bytesperstrand], dtype=uint8)
        self.wizoffset = 0
        for ipacket in range(npackets):
            messpack, messplain = self.createmesspacket(ipacket)
            messplains[ipacket * self.strandsperpacketmessage:(ipacket + 1) * self.strandsperpacketmessage] = messplain
            segmentfor_encode[ipacket*self.strandsperpacket:(ipacket+1)*self.strandsperpacket] = messpack
        if self.norsway:
            for ipacket in range(npackets):
                segmentfor_encode[ipacket * self.strandsperpacket:(ipacket + 1) * self.strandsperpacket] = self.myprotectmesspacket(messpack)

        # 编码 + rs
        # self.writemessplain('messplains.txt', messplains)
        self.writemessplain('messbytesnors.txt', messplains[:self.bytesseqnum])
        outfilepath = self._encode(segmentfor_encode,self.coderatecode)

        self.encode_time = str(datetime.now() - tm_encode)
        # 计算 信息密度
        with open(outfilepath, 'r', encoding='utf-8') as f:
        # with open(outfilepath, 'r') as f:
            old_content = f.read()
        tseqs = old_content.split('\n')[1::2]
        self.total_base = npackets * self.strandsperpacket * self.totstrandlen
        self._set_density()
        dnaseqs = []
        for i in range(len(tseqs)):
            dnaseqs.append(tseqs[i])
        self.seq_num = len(dnaseqs)
        mygc = self.getgc(dnaseqs)
        self.gc=mygc
        self.max_homopolymer = self.max_homopolymer_length(dnaseqs)
        print(f"total_bit:{self.total_bit},total_base:{self.total_base},dnanums:{len(dnaseqs)},density:{self.density},mygc:{mygc}")
        # print(f"total_bit:{self.total_bit},total_base:{self.total_base},density:{self.density}")

        # 保存序列，首行添加编码的一些重要参数
        # param = self._set_matrix_param_line()
        param = self._set_param_common('hedges')
        log.debug('write')
        num=0
        with open(self.output_file_path, 'w') as f:
            f.write(param)
            # f.write(dnaseqs)
            # for i in range(len(dnaseqs)):
            #     f.write(dnaseqs[i]+"\n")
            for i in range(len(dnaseqs)):
                f.write(f">seq{i}\n{dnaseqs[i]}\n")
        # self.encode_time = str(datetime.now() - tm_encode)
        # return outfilepath

        print(f"实际编码的序列共有{self.seq_num}条")

    def _encode(self, bit_segments,coderatecode):
        # pass
        # activate_command = "conda run -n tensorflow_cdpm2 python encode_decode_m/hedges_encode_decode.py encode "

        def writemessplain(path, datas):
            with open(path, 'w') as file:
                for data in datas:
                    for d in data:
                        file.write(str(d)+' ')
                    file.write('\n')

        # outfilepath = self.output_file_path
        # outfilepath = 'hedges_encodefile.fasta'
        writemessplain('files/bit_segments.txt', bit_segments)
        activate_command = "conda run -n tensorflow_cdpm2 python ./polls/Code/encode_decode_m/hedges_encode_decode.py encode files/bit_segments.txt"
        activate_command += f" {self.output_file_path} {self.codec_param.sequence_length} {coderatecode} {self.max_hpoly_run} {self.npackets}"
        result = subprocess.Popen(activate_command, shell=True)
        result.wait()
        print(result.stderr)
        # base_segments = readseqs(encodefile)


        # base_segments = hedgesEncode(bit_segments, rep_num=self.codec_param.max_homopolymer)
        # base_segments = hedgesEncode(self.input_file_path, 'hedges_encode_test.fasta', 'hedges_messplainpath.txt')
        return self.output_file_path

class HedgesDecode(AbstractDecode):

    def __init__(self, input_file_path: str,orifile:str, output_dir: str):
        super().__init__(input_file_path, output_dir)
        self.orifile=orifile

    def writemessplain(self,path, datas):
        with open(path, 'w') as file:
            for data in datas:
                for d in data:
                    file.write(str(d) + ' ')
                file.write('\n')

    def getbinfile(self,path):
        bin_to_str_list = list()
        for i in range(256):
            bin_to_str_list.append(bin(i)[2:].rjust(8, '0'))
        with open(path, 'rb') as f:
            bytesarr = f.read()
        str_list = [bin_to_str_list[i] for i in bytesarr]
        binstring = ''.join(str_list)
        binstring_bits = np.array([bit for bit in binstring], np.uint8)
        return binstring_bits

    def judgeoutcomes(self,decode_bytes,decode_bits):
        messbytes = self.readmessplainlist('messbytesnors.txt')
        decode_bytes = np.array(decode_bytes[:len(messbytes)])
        badbytes = np.count_nonzero(decode_bytes - messbytes)
        allbytes = len(messbytes)*len(messbytes[0])
        print(f"badbytes:{badbytes},allbytes:{allbytes},bytes recov:{1-badbytes/allbytes}")
        # messplainbits = self.bytesTobits(messbytes)
        # decode_bits1 = np.array([np.array([bit for bit in bits],uint8) for bits in decode_bits[:len(messplainbits)]])
        # messplainbits1 = np.array([np.array([bit for bit in bits], uint8) for bits in messplainbits])
        # badbits = np.count_nonzero(decode_bits1 - messplainbits1)
        # # badbits = np.count_nonzero(decode_bits - messplainbits)
        # allbits = len(messplainbits)*len(messplainbits[0])
        # print(f"badbits:{badbits},allbits:{allbits},bits recov:{1-badbits/allbits}")
        ori_bits = self.getbinfile(self.orifile)
        decode_bits = self.getbinfile(self.output_file_path)
        print(f'ori_bits:{len(ori_bits)},decode_bits:{len(decode_bits)}')
        if len(decode_bits) >= len(ori_bits):
            badbits = np.count_nonzero(ori_bits - decode_bits[:ori_bits.size])
        else:
            print('??????????????????????len(decode_bits)<len(ori_bits)????????????????????????????')
            badbits = np.count_nonzero(ori_bits[:decode_bits.size] - decode_bits)
        allbits = ori_bits.size
        print(f"badbits:{badbits},allbits:{allbits},bits recov:{1 - badbits / allbits}")
        infos = {
            'code': 0,
            'badbits': badbits,
            'allbits': allbits,
            'bits_recov': 1 - badbits / allbits,
            'decodefile': self.output_file_path,
        }
        print(f'infos:{infos}')
        return infos

    def common_decode_matrix(self):
        # 矩阵形式解码：1.读取dna序列；2.将dna序列转换为矩阵形式的数据；3.对每个矩阵使用特定的解码方法进行解码；
        #               4.rs纠错并删除（可替换）5.数据转换为bit并排序；6.恢复原文件

        self._parse_matrix_param()
        self.strandIDbytes = 2
        self.strandrunoutbytes = 2

        strandsperpacket = self.strandsperpacket
        npackets = self.npackets
        tm_run = datetime.now()
        all_dna_sequences = self._get_base_line_list()
        # npackets = int(np.ceil(len(all_dna_sequences) / strandsperpacket))
        segmentfor_decode = np.zeros([len(all_dna_sequences), self.totstrandlen], dtype=np.uint8)
        dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        for i in range(len(all_dna_sequences)):
            thisseq = np.zeros([self.totstrandlen], dtype=uint8)
            for j in range(len(all_dna_sequences[i])):
                if j < self.totstrandlen:
                    thisseq[j] = dict[all_dna_sequences[i][j]]
            segmentfor_decode[i] = thisseq

        # decode
        start_time = datetime.now()
        print(f"开始解码：")
        outfilepath = self._decode(segmentfor_decode)
        print(f"解码时间：{str(datetime.now() - start_time)}")
        decode_numbers_ori = self.readmessplainlist(outfilepath)
        decode_bytes = self.extractplaintext(decode_numbers_ori)
        decode_bits = self.bytesTobits(decode_bytes)

        self.seq_num = len(decode_numbers_ori)
        self.index_length = self.strandIDbytes*8
        # log.debug('sort')
        # sorted_bit_segments = BaseTools.sort_segment(decode_bits, self.index_length)
        # log.debug('repair')
        # # validate_bit_segs = self._repair_segment(sorted_bit_segments)
        # validate_bit_segs = sorted_bit_segments
        # log.debug('merge')
        # res_bit_str = SplitTools.merge(validate_bit_segs)
        #
        # res_bit_str = res_bit_str[:self.codec_param.total_bit]
        # res_bit_str = self.file_binary_to_random(res_bit_str)
        # log.debug('write')
        # self._bin_to_file(res_bit_str)

        res_bit_str = ''.join(decode_bits)
        res_bit_str = res_bit_str[:self.codec_param.total_bit]
        log.debug('write')
        res_bit_str = self.file_binary_to_random(res_bit_str)
        self.binstring_to_file(res_bit_str,self.output_file_path)

        self.run_time = datetime.now() - tm_run
        infos = self.judgeoutcomes(decode_bytes,res_bit_str)

        infos['decode_time'] = self.run_time
        return infos

    def binstring_to_file(self,binstring, output_file_path):

        outstring = bytes(int(binstring[i:i + 8], 2) for i in range(0, len(binstring), 8))
        # 转换为 bytes 并写入文件
        with open(output_file_path, "wb") as f:
            f.write(outstring)

    def judgedrrick(self):
        ori_bits = self.getbinfile(self.orifile)
        decode_bits = self.getbinfile(self.output_file_path)

        if len(decode_bits) >= len(ori_bits):
            # for i in range(len(ori_bits)):
            #     if decode_bits[i]!=ori_bits[i]:
            #         print(f'hedges judge output file bit error position:{i}')
            badbits = np.count_nonzero(ori_bits - decode_bits[:ori_bits.size])
        else:
            print('??????????????????????len(decode_bits)<len(ori_bits)????????????????????????????')
            badbits = np.count_nonzero(ori_bits[:decode_bits.size] - decode_bits)
        # badbits = np.count_nonzero(decode_bits - messplainbits)
        allbits = ori_bits.size
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print(f"badbits:{badbits},allbits:{allbits},bits recov:{1 - badbits / allbits}")
        self.returncode = 1


    def _decode(self, segmentfor_decode):
        outfilepath = 'bit_segments_decode.txt'
        self.writemessplain('bit_segments_fordecode.fasta', segmentfor_decode)
        print("for each packet, these statistics are shown in two groups:")
        print("1.1 HEDGES decode failures(HEDGES链解码失败数量), 1.2 HEDGES bytes thus declared as erasures(HEDGES字节擦除数量)")
        print("1.3 R-S total errors detected in packet(rs观察到的字节错误), 1.4 max errors detected in a single decode")
        print("2.1 R-S reported as initially-uncorrected-but-recoverable total(rs报告最初未纠正但可恢复的字节错误), 2.2 same, but max in single decode")
        print("2.3 R-S total error codes; if zero, then R-S corrected all errors(剩下的rs错误) ")
        activate_command = "conda run -n tensorflow_cdpm2 python polls/Code/encode_decode_m/hedges_encode_decode.py decode bit_segments_fordecode.fasta"
        activate_command += f" {outfilepath} {self.totstrandlen} {self.coderatecode} {self.max_hpoly_run}  {self.npackets}"
        # activate_command += f" {outfilepath}"
        result = subprocess.Popen(activate_command, shell=True)
        result.wait()
        print(f"解码结果：{result.stderr},outfilepath:{outfilepath}")
        return outfilepath

class YinYangCodeEncode(AbstractEncode):
    # def __init__(self, yang_rule=None, yin_rule=None, virtual_nucleotide="A", max_iterations=100,
    #              max_ratio=0.8, faster=False, max_homopolymer=4, max_content=0.6, need_logs=False):
    def __init__(self, input_file_path: str, output_dir: str, sequence_length: int, max_homopolymer=6,
                 rs_num=0, add_redundancy=False, add_primer=False, primer_length=20,max_iterations=100,gc_bias=0.2,crcyyc=0):
        index_redundancy = 0
        seq_bit_to_base_ratio = 1
        self.max_iterations = max_iterations
        self.gc_bias=gc_bias
        self.crc_num=crcyyc
        super().__init__(input_file_path, output_dir, 'YinYangCode', seq_bit_to_base_ratio=seq_bit_to_base_ratio,
                         index_redundancy=index_redundancy,
                         sequence_length=sequence_length, max_homopolymer=max_homopolymer, rs_num=rs_num,
                         add_redundancy=add_redundancy,
                         add_primer=add_primer, primer_length=primer_length)

    def _set_param_line(self, left_primer="", right_primer=""):
        param = "totalBit:{},binSegLen:{},leftPrimer:{},rightPrimer:{},fileExtension:{},bRedundancy:{},RSNum:{}\n".format(
            self.total_bit, self.bin_split_len, left_primer, right_primer, self.file_extension,
            int(self.codec_param.add_redundancy), int(self.codec_param.rs_num))
        return param

    def _set_param_common(self,method='YYC'):
        param = "method:{},seq_len:{},gc_bias:{},max_homopolymer:{},rs_num:{},crc_num:{},max_iterations:{},index_length:{},total_count:{},".format(
            method, self.codec_param.sequence_length,self.gc_bias, self.codec_param.max_homopolymer,
            self.codec_param.rs_num,self.crc_num, self.max_iterations,self.index_length,self.total_count)
        param += self._set_param_line()
        return param

    def _encode(self, bit_segments,coderatecode):
        # max_homopolymer=0.5+self.codec_param.max_homopolymer/2
        # return yinyangencode(bit_segments,max_homopolymer,self.max_iterations)
        # return yinyangencode(bit_segments,self.codec_param.max_homopolymer,self.max_iterations)
        return yinyangencode(bit_segments,self.codec_param.max_homopolymer)

    def common_encode(self):
        """
        common encode flow
        """
        # 序列形式编码：1.读取文件；2.根据文件大小，rs码个数，序列长度计算总共有多少条序列及index长度；3.编码并保存文件

        self._check_params()
        tm_run = datetime.now()
        log.debug('read file')
        bin_str = self._file_to_bin()
        self.codec_param.crc_num = self.crc_num
        # split
        self.bin_split_len, self.index_length, self.rs_group, bit_segments = self._split_bin(bin_str)
        # print(f'self.codec_param.sequence_length')
        print(f'bit_segments:{len(bit_segments)},bit_segment_len:{len(bit_segments[0])}\nself.index_length:{self.index_length},self.rs:{self.codec_param.rs_num},crcnum:{self.crc_num}')
        # new_bit_segments = [None] * self.seq_num
        # for i in range(self.seq_num):
        #     new_bit_segments[i]=bit_segments[i][self.index_length:]
        # add
        # with open('messplains.txt', 'w') as file:
        #     for data in bit_segments:
        #         file.write(str(data) + '\n')
        with open('messplains.txt', 'w') as file:
            for data in bit_segments:
                file.write(str(data)+'\n')
        if self.codec_param.rs_num > 0:
            # bit_segments = self._add_rscode(new_bit_segments)
            bit_segments = self._add_rscode(bit_segments)
        self.decode_packets = len(bit_segments)
        self.total_count = len(bit_segments)
        # encode
        tm_encode = datetime.now()
        base_segments = self._encode(bit_segments, len(bin_str))
        self.seq_num = len(base_segments)
        self.encode_time = str(datetime.now() - tm_encode)
        print(f"编码时间为{str(datetime.now() - tm_encode)}")
        print(f"编码的序列共有{len(base_segments)}条，每条{len(base_segments[0])}nt")
        self.total_base = len(base_segments) * len(base_segments[0])
        self._set_density()
        mygc = self.getgc(base_segments)
        self.gc = mygc
        self.max_homopolymer = self.max_homopolymer_length(base_segments)
        print(
            f"!!!!!!!!!!total_bit:{self.total_bit},total_base:{self.total_base},dnaseqslen:{len(base_segments[0])},density:{self.density},mygc:{mygc}!!!!!!!!!!!!")

        param = self._set_param_common('YYC')
        log.debug('write')
        num = 0
        with open(self.output_file_path, 'w') as f:
            f.write(param)
            for i in range(len(base_segments)):
                if base_segments[i] is not None:
                    num+=1
                    f.write(f">seq{i}\n{str(''.join(base_segments[i]))}\n")
        print(f"实际编码的序列共有{num}条")
        # add primer (blast need fasta file to compare)
        # if self.codec_param.add_primer:
        #     self._add_primer()
        #     self.total_base += (len(base_segments) * 2 * self.codec_param.primer_length)

        self.run_time = str(datetime.now() - tm_run)
        self._set_density()
        # self.index_length = int(len(str(bin(len(bit_segments)))) - 2)
        # return self.output_file_path, self.index_length

class YinYangCodeDecode(AbstractDecode):
    def __init__(self, input_file_path: str, orifile: str, output_dir: str,index_length:int,total_count:int, crc_num=0, phreds=None, decision='hard'):
        super().__init__(input_file_path, output_dir)
        self.total_count=total_count
        # self.codeindex=codeindex
        self.index_length=index_length
        self.orifile=orifile
        self.crc_num = crc_num
        self.phreds = phreds
        self.decision = decision


    def judgedrrick(self):
        try:
            ori_bits=getbinfile(self.orifile)
            decode_bits = getbinfile(self.output_file_path)
            print(f'文件比对:\nori_bits:{self.orifile},decode_bits:{self.output_file_path}')
            if len(decode_bits) > len(ori_bits):
                badbits = np.count_nonzero(ori_bits - decode_bits[:ori_bits.size])
            else:
                print('??????????????????????len(decode_bits)<len(ori_bits)????????????????????????????')
                print(f'ori_bits.size:{ori_bits.size},decode_bits.size:{decode_bits.size}')
                badbits = np.count_nonzero(ori_bits[:decode_bits.size] - decode_bits)

            # badbits = np.count_nonzero(ori_bits - decode_bits[:ori_bits.size])
            # badbits = np.count_nonzero(ori_bits - decode_bits)
            # badbits = np.count_nonzero(decode_bits - messplainbits)
            allbits = ori_bits.size
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print(f"badbits:{badbits},allbits:{allbits},bits recov:{1 - badbits / allbits}")
            infos = {
                'code': 0,
                'badbits':badbits,
                'allbits': allbits,
                'bits_recov': 1 - badbits / allbits,
                'decodefile':self.output_file_path,
            }
        except ValueError:
            infos = {
                'code': 1,
            }
        finally:
            return infos

    def common_decode(self):
        """
        common decode flow
        """
        # 序列形式解码：1.读取dna序列；2.解码并排序；3.rs纠错并删除；4.恢复原文件
        tm_decode = datetime.now()
        self._parse_param()
        self.infos = dict()
        # self.ori_len = self.codec_param.bin_seg_len
        # tm_run = datetime.now()
        base_line_list = self._get_base_line_list()
        # print(f"==================================={len(base_line_list)}")
        # print((base_line_list[0]))
        print(f"------开始解码------")
        # bit_segments = self._decode(base_line_list)
        bit_segments,bit_phreds = self._decode(base_line_list)
        print(f"------解码结束------，共有序列：{len(base_line_list)}条，解码得到：{len(bit_segments)}行bits，解码时间为{str(datetime.now() - tm_decode)}")

        log.debug('sort')
        # sorted_binstr = bit_segments
        # sorted_binstr = BaseTools.sort_segment(bit_segments, self.index_length)
        if len(bit_phreds)>0:
            sorted_binstr,sorted_phreds = BaseTools.sort_segment_withphred(bit_segments,bit_phreds, self.index_length)
        else:
            sorted_binstr = BaseTools.sort_segment(bit_segments, self.index_length)
            sorted_phreds = bit_phreds
        sorted_binstr = sorted_binstr[:self.seq_num]

        if self.codec_param.rs_num>0:
            print(f"------开始删除rs码------")
            if self.crc_num >0:
                print(f'crc_num:{self.crc_num},开始去除crc及rs码')
                # validate_bit_segs, err_bit_segs = self._del_rscode(sorted_binstr,bit_phreds)
                validate_bit_segs, err_bit_segs, err_rate = self._del_rscode_crc(sorted_binstr,sorted_phreds)
                self.check_recover_ration(validate_bit_segs,err_bit_segs)
                print(f"原有{len(sorted_binstr)}条，经过rs解码后，剩余：{len(validate_bit_segs)}条")
            else:
                print(f'crc_num:{self.crc_num}')
                validate_bit_segs, err_bit_segs = self._del_rscode(sorted_binstr)
                self.check_recover_ration(validate_bit_segs,err_bit_segs)
            print(f"------删除rs码结束------")
        # if self.codec_param.rs_num > 0:
        #     print(f"------开始删除rs码------")
        #     validate_bit_segs, err_bit_segs = self._del_rscode(sorted_binstr)
        #     print(f"------删除rs码结束------")
        else:
            validate_bit_segs, err_bit_segs = sorted_binstr, []
        # self.infos['decode_time'] = str(datetime.now() - tm_decode)
        # self.infos['decodefile'] = self.output_file_path
        # self.infos['code'] = 0

        with open('messplains_decode.txt', 'w') as file:
            for data in validate_bit_segs:
                file.write(str(data)+'\n')
        # print(f"原有{len(sorted_binstr)}条，经过rs解码后，剩余：{len(validate_bit_segs)}条")
        # repair and count missing segments
        # log.debug('repair')
        # recov = self.check_recover_ration(validate_bit_segs, err_bit_segs)
        validate_bit_segs = self._repair_segment(validate_bit_segs)
        validate_bit_segs =self.validate_and_unify_bit_segs(validate_bit_segs,self.ori_len-self.index_length)
        self.infos['code'] = 0
        self.infos['decodefile'] = self.output_file_path


        log.debug('merge')
        res_bit_str = SplitTools.merge(validate_bit_segs)
        print(f"decode bits:{len(res_bit_str)},ori total_bit:{self.codec_param.total_bit}")
        res_bit_str = res_bit_str[:self.codec_param.total_bit]

        log.debug('write')
        self._bin_to_file(res_bit_str)
        # self.infos = self.judgedrrick()
        self.judgedrrick()

        self.infos['decode_time'] = str(datetime.now() - tm_decode)
        return self.infos

    # def _decode(self,dna_sequences):
    #     print(f"dna_sequences num:{len(dna_sequences)},self.total_count:{self.total_count},self.index_length:{self.index_length}")
    #     return yinyangdecode(dna_sequences,self.index_length,self.total_count)
    def validate_and_unify_bit_segs(self, bit_segments, target_length=None, padding='0'):

        # 统一长度
        unified_segments = []
        for seg in bit_segments:
            if len(seg) < target_length:
                # 填充不足的部分
                padded_seg = seg + padding * (target_length - len(seg))
                unified_segments.append(padded_seg)
            elif len(seg) > target_length:
                # 截断过长的部分
                unified_segments.append(seg[:target_length])
            else:
                unified_segments.append(seg)

        return unified_segments

    def _decode(self,dna_sequences):
        print(f"dna_sequences num:{len(dna_sequences)},self.total_count:{self.total_count},self.index_length:{self.index_length}")
        return yinyangdecode(dna_sequences,self.index_length,self.total_count,self.phreds)

    # def _decode(self, base_line_list):
    #     bit_segments = dnafountainDecode(base_line_list,decode_packets=self.decode_packets,segment_length=self.sequence_length)
    #     return bit_segments
