# -*- coding: utf-8 -*-
import logging
import math
from functools import cmp_to_key
from os import path

from tqdm import tqdm

from .ecc import ReedSolomon
from .utils import bin_to_str_list

log = logging.getLogger('mylog')


class CodecException(Exception):
    pass
    

class EncodeParameter:
    """
    parameters for all algorithms
    """

    def __init__(self, sequence_length=120, max_homopolymer=6, min_gc=0.3, max_gc=0.7,
                 primer_length=20, add_primer=False, rs_num=0, rs_matrix_num=0,
                 add_redundancy=False):
        self.sequence_length = sequence_length
        self.max_homopolymer = max_homopolymer
        self.min_gc = min_gc
        self.max_gc = max_gc
        self.add_primer = add_primer
        self.primer_length = primer_length
        self.rs_num = rs_num
        self.rs_matrix_num = rs_matrix_num
        self.add_redundancy = add_redundancy

class DecodeParameter:
    """
    parameters for all algorithms(in the first line of encoded file)
    """

    def __init__(self, total_bit=0, bin_seg_len=120, add_redundancy=False, rs_num=0,
                 left_primer_len=0, right_primer_len=0, file_extension=""):
        self.total_bit = total_bit
        self.bin_seg_len = bin_seg_len
        self.add_redundancy = add_redundancy
        self.rs_num = rs_num
        self.left_primer_len = left_primer_len
        self.right_primer_len = right_primer_len
        self.file_extension = file_extension


def file_binary_to_random(binary_sequence_ori):
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


class FileTools:
    """
    some methods for file
    """


    @staticmethod
    def file_to_bin(read_path):
        '''read file and return binary string'''
        with open(read_path, 'rb') as f:
            bytesarr = f.read()
        str_list = [bin_to_str_list[i] for i in bytesarr]
        binstring = ''.join(str_list)
        binstring = file_binary_to_random(binstring)
        return binstring

    @staticmethod
    def bin_to_file(binstring, write_path):
        binstring = file_binary_to_random(binstring)
        '''write binary data to file'''
        bytearr = bytearray([int(binstring[i:i + 8], 2) for i in range(0,len(binstring),8)])
        with open(write_path, 'wb') as f:
            f.write(bytearr)
            
    @staticmethod
    def get_file_name_extension(file_path_or_name):
        s = path.basename(file_path_or_name)
        file_name = s[:s.index('.'):] if '.' in s else s
        file_extension = s[s.index('.'):] if '.' in s else ''
        return file_name, file_extension
                


class SplitTools:
    """
    Split methods
    """

    @staticmethod
    def get_split_info(total, len_with_index, base=2):
        """
        Calculate the index length of each sequence given the total length and the length of each sequence
        @param total: total length of data
        @param len_with_index: sequence length
        @param base: e.g. binary:2, decimal:10
        @return:
        """
        if total <= 0 or len_with_index <= 0:
            raise Exception('input error, total:{}, len_with_index:{}'.format(total, len_with_index))
        seq_num = math.ceil(total / len_with_index)
        index_len = math.ceil(math.log(seq_num, base))
        data_len = len_with_index - index_len
        while data_len > 0:
            seq_num = math.ceil(total / data_len)
            if seq_num <= base ** index_len:
                return index_len, data_len, seq_num
            else:
                index_len += 1
                data_len -= 1
        raise Exception('no split result')

    @staticmethod
    def get_bin_split_length(ori_bin_length, bit_base_ratio, code_param: EncodeParameter, index_redundancy=0):
        """
        Get the length of binary seqment
        """
        # Estimate the binary length when splitting file data based on the initial sequence length
        # remove primer length
        if code_param.add_primer:
            tmp_dna_length = code_param.sequence_length - code_param.primer_length * 2
        else:
            tmp_dna_length = code_param.sequence_length
        # Calculate the binary length corresponding to the segmentation sequence (may include rs)
        seq_bin_length = math.ceil(tmp_dna_length * bit_base_ratio)
               
        # Remove RSCode length
        # rs_group = 1
        # if code_param.rs_num>=0:
        #     bytes_num = math.ceil(seq_bin_length / 8)
        #     rs_group = math.ceil(bytes_num / 255)
        #     seq_bin_length -= (code_param.rs_num*8*rs_group)

        if code_param.rs_num>=0 or code_param.crc_num>=0:
            bytes_num = math.ceil(seq_bin_length / 8)
            rs_group = math.ceil(bytes_num / 255)
            seq_bin_length = seq_bin_length- (code_param.rs_num*8*rs_group)-(code_param.crc_num*8*rs_group)

        # Considering that there is index redundancy, it is equivalent to the redundancy of the total length, which is subtracted from the data length
        true_seq_bin_length = seq_bin_length if index_redundancy == 0 else seq_bin_length - index_redundancy

        compute_bit = ori_bin_length if not code_param.add_redundancy else (ori_bin_length + int(ori_bin_length / 2))
        index_length, data_bin_length, seq_num = SplitTools.get_split_info(compute_bit, true_seq_bin_length, 2)
        index_length += index_redundancy
        
        # check
        if code_param.add_redundancy:
            while(1):
                tmp_index_len = index_length
                split_num = math.ceil(ori_bin_length / data_bin_length) + int(math.ceil(ori_bin_length / data_bin_length) / 2)
                index_length = math.ceil(math.log(split_num, 2)) + index_redundancy
                if index_length==tmp_index_len:
                    break
                else:
                    compute_bit = split_num * (data_bin_length-1)
                    index_length, data_bin_length, seq_num = SplitTools.get_split_info(compute_bit, true_seq_bin_length, 2)
                    index_length += index_redundancy
            
        log.debug('index_length: {}, data_bin_length: {}, seq_num: {}'.format(index_length, data_bin_length, seq_num))
        return index_length, data_bin_length, seq_num, rs_group

    @staticmethod
    def get_indexlen_and_segnum(total_bit, bin_seg_len, add_redundancy, index_redundancy=0):
        """
        for decode
        """
        tmp_num = math.ceil(total_bit / bin_seg_len)  # 向上取整，得到片段个数
        segments_num = tmp_num if not add_redundancy else (tmp_num + int(tmp_num / 2))  # 判断冗余
        index_len = math.ceil(math.log(segments_num, 2)) + index_redundancy
        
        log.debug("indexLen:{},segments_num:{}".format(index_len, segments_num))
        return index_len, segments_num

    @staticmethod
    def split(binstring, bin_split_length, add_redundancy=False, index_redundancy=0):
        """
        split bin string to bin list
        """
        bin_len = len(binstring)
        tmp_num = math.ceil(bin_len / bin_split_length)  # 向上取整，得到片段个数
        segments_num = tmp_num if not add_redundancy else (tmp_num + int(tmp_num / 2))  # 判断冗余

        index_length = math.ceil(math.log(segments_num, 2)) + index_redundancy
        
        log.debug('index:{}, data:{}'.format(index_length, bin_split_length))
        bit_segments = []
        tmp_length = 0
        pro_bar = tqdm(total=segments_num, desc="Spliting")
        for i in range(segments_num):
            bin_index = bin(i)[2:].zfill(index_length)  # index，转二进制，左补零
            if add_redundancy and i % 3 == 2:
                strAdd = ''.join([str(int(x) ^ int(y)) for x, y in
                                  zip(bit_segments[-1][index_length:], bit_segments[-2][index_length:])])
                bit_segments.append(bin_index + strAdd)
            elif tmp_length + bin_split_length <= len(binstring):  # 未到结尾
                strTemp = binstring[tmp_length:tmp_length + bin_split_length]
                bit_segments.append(bin_index + strTemp)
                tmp_length += bin_split_length
            else:  # end, the last paragraph is not long enough
                strTemp = binstring[tmp_length:].ljust(bin_split_length, '0')
                bit_segments.append(
                    bin_index + strTemp)  # if the length of last segments != bin_len , file zero on the right side of the sequence
                tmp_length += bin_split_length
            pro_bar.update()
        pro_bar.close()
        return bit_segments

    @staticmethod
    def repair_segment(index_length, bit_segments, split_num, is_redundancy):
        # print(index_length)
        # print(bit_segments)
        # print(split_num)
        # print(is_redundancy)
        dict_seg = dict() #exist index
        for bit_str in bit_segments:
            index = int(bit_str[:index_length], 2)
            if index < split_num:
                dict_seg[index] = bit_str[index_length:]

        repaired_indexs = []
        failed_indexs = []
        res_segments = list()
        if is_redundancy:  # try to repair
            for seg_index in range(split_num):
                if seg_index not in dict_seg.keys():
                    if seg_index % 3 == 0:
                        index1 = seg_index + 1
                        index2 = seg_index + 2
                    elif seg_index % 3 == 1:
                        index1 = seg_index - 1
                        index2 = seg_index + 1
                    else:
                        continue
                    if index1 in dict_seg.keys() and index2 in dict_seg.keys():
                        add_str = ''.join([str(int(x) ^ int(y)) for x, y in zip(dict_seg[index1], dict_seg[index2])])
                        dict_seg[seg_index] = add_str
                        log.info("repair segment {}".format(seg_index))
                        repaired_indexs.append(seg_index)
                    else:
                        log.warning("error to repair segment {}".format(seg_index))
                        failed_indexs.append(seg_index)
            for index in sorted(dict_seg.keys()):
                if index%3!=2: # delete redundancy and index
                    res_segments.append(dict_seg[index])
        else:
            failed_indexs = list(set(range(split_num)).difference(set(dict_seg.keys())))
            for index in sorted(dict_seg.keys()):
                res_segments.append(dict_seg[index])
        # for index in failed_indexs: log.warning('missing segment, index:{}'.format(index))

        repaired_rate = len(repaired_indexs) / split_num
        failed_rate = len(failed_indexs) / split_num
        return {"res_segments": res_segments,
                "repaired_indexs": repaired_indexs,
                "failed_indexs": failed_indexs,
                "repaired_rate": repaired_rate,
                "failed_rate": failed_rate}

    @staticmethod
    def merge(bit_segments):
        """
        merge bin list to bin string
        """
        res_bitstr = ''.join(bit_segments)
        return res_bitstr


class RsTools:
    @staticmethod
    def add_rs(bit_segments, check_bytes, rs_group):
        # add rscode
        rs = ReedSolomon(check_bytes)
        rs_res = rs.insert(bit_segments, group=rs_group)
        return rs_res

    @staticmethod
    def del_rs_crc(binsstr_list, ori_len, check_bytes,phreds=[],crccode=0):
        seq_bin_length = len(binsstr_list[0])
        bytes_num = math.ceil(seq_bin_length / 8)
        rs_group = math.ceil(bytes_num / 255)
        rs = ReedSolomon(check_bytes)
        if len(phreds) > 0:
            print(f'=====1，有使用质量值解码')
            bit_segments = rs.remove_crc(binsstr_list, ori_len, phreds,crccode, group=rs_group)
        else:
            print(f'=====2，未使用质量值解码')
            bit_segments = rs.remove_ori(binsstr_list, ori_len,crccode, group=rs_group)
        print(f"1:bit:{len(bit_segments['bit'])},err_index:{len(bit_segments['e_i'])},err_rate:{bit_segments['e_r']}")
        return {"validate_bit_seg": bit_segments['bit'], "err_index": bit_segments['e_i'],"err_rate": bit_segments['e_r']}
        # return [{"validate_bit_seg": bit_segments1['bit'], "err_index": bit_segments1['e_i'],"err_rate": bit_segments1['e_r']}]

    @staticmethod
    def del_rs(binsstr_list, ori_len, check_bytes):
        seq_bin_length = len(binsstr_list[0])
        bytes_num = math.ceil(seq_bin_length / 8)
        rs_group = math.ceil(bytes_num / 255)
        rs = ReedSolomon(check_bytes)
        bit_segments = rs.remove(binsstr_list, ori_len, group=rs_group)
        return {"validate_bit_seg": bit_segments['bit'], "err_index": bit_segments['e_i'],
                "err_rate": bit_segments['e_r']}

class BaseTools:
    """
    Tools for base
    """
    @staticmethod
    def creat_reverse_dict(ori_dict):
        res_dict = {}
        for i in ori_dict:
            res_dict[ori_dict[i]] = i
        return res_dict

    @staticmethod
    def get_repeat(dnastr):
        if len(dnastr) == 0:
            return 0
        count = 1
        maxnum = 1
        dna_len = len(dnastr)
        for i in range(1, dna_len):
            if dnastr[i] == dnastr[i - 1]:
                count += 1
            else:
                count = 1
            maxnum = max(count, maxnum)
        return maxnum
    
    @staticmethod
    def check_equal(dnastr):
        if len(dnastr)==0:
            return True
        a = dnastr[0]
        for x in dnastr:
            if x!=a:
                return False
        return True

    @staticmethod
    def get_gc(dnastr):
        return (dnastr.count("C") + dnastr.count("G")) / len(dnastr)

    @staticmethod
    def sort_segment(bit_segments, index_length):
        def comp_by_index(x, y):
            index_x = int(x[:index_length], 2)
            index_y = int(y[:index_length], 2)
            if index_x > index_y:
                return 1
            elif index_y > index_x:
                return -1
            else:
                return 0

        bit_segments = sorted(bit_segments, key=cmp_to_key(comp_by_index))
        return bit_segments

    @staticmethod
    def sort_segment_withphred(bit_segments,bit_phreds, index_length):
        def comp_by_index(x, y):
            index_x = int(x[:index_length], 2)
            index_y = int(y[:index_length], 2)
            if index_x > index_y:
                return 1
            elif index_y > index_x:
                return -1
            else:
                return 0

        # bit_segments = sorted(bit_segments, key=cmp_to_key(comp_by_index))

        # 先把 bit_segments 和 bit_phreds 一起组合成元组
        combined = list(zip(bit_segments, bit_phreds))

        # 对 combined 按照 bit_segments 部分的排序
        combined.sort(key=cmp_to_key(lambda x, y: comp_by_index(x[0], y[0])))

        # 排序后拆开元组，分别得到排序后的 bit_segments 和 bit_phreds
        sorted_bit_segments, sorted_bit_phreds = zip(*combined)

        # 返回排序后的结果
        return list(sorted_bit_segments), list(sorted_bit_phreds)
