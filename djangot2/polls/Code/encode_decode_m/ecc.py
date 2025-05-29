# -*- coding: utf-8 -*-
import logging
import math
from itertools import combinations

import numpy as np
from reedsolo import RSCodec, ReedSolomonError
from tqdm import tqdm

from polls.Code.encode_decode_m.utils import bin_to_str_list
from polls.Code.utils import crc16

log = logging.getLogger('mylog')

class ReedSolomon():
    def __init__(self, check_bytes:int):
        """
        reed-solomon code
        @param check_bytes: the number of rs code (bytes)
        """
        self.check_bytes = check_bytes
        self.tool = RSCodec(check_bytes)
        self.remainder = 0
        log.debug('ecc check_bytes:{}'.format(check_bytes))

    def insert_one(self, binstring, group=1):
        # If the binary length is not a multiple of 8, it will be left-padded with zeros
        self.remainder = 0
        if len(binstring)%8!=0:
            self.remainder = 8 - len(binstring) % 8
            binstring = '0'*self.remainder + binstring

        output_string = ''
        if group>=1 and type(group)==int:
            if group<=len(binstring)/8/255:
                group = 1
            group_len = math.ceil(len(binstring)/8/group)*8
            for i in range(group):
                last = group_len*(i+1)
                if last>=len(binstring): last = None
                group_string = binstring[group_len*i:last]
                byte_list = bytearray([int(group_string[i:i + 8], 2) for i in range(0, len(group_string), 8)])
                rs_list = self.tool.encode(byte_list)
                str_list = [bin_to_str_list[i] for i in rs_list]
                output_string += ''.join(str_list)
        else:
            raise Exception("Wrong rs group")
        return output_string[self.remainder:]

    
    def remove_one(self, binstring, group=1):
        self.remainder = 0
        # If the binary length is not a multiple of 8, it will be left-padded with zeros
        if len(binstring)%8!=0:
            self.remainder = 8 - len(binstring) % 8
            binstring = '0'*self.remainder + binstring

        data_len = len(binstring) - group*self.check_bytes*8
        output_string = ""
        if group>=1 and type(group)==int:
            if group<=data_len/8/255:
                group = 1
            group_len = math.ceil(data_len/8/group)*8 + self.check_bytes*8
            for i in range(group):
                last = group_len*(i+1)
                if last >= len(binstring): last = None
                group_string = binstring[i*group_len:last]
                byte_list = bytearray([int(group_string[i:i + 8], 2) for i in range(0, len(group_string), 8)])
                try:
                    decode_byte_list, full_list, err_pos = self.tool.decode(byte_list)
                    # if len(err_pos)!=0:
                        # print(err_pos)
                    str_list = [bin_to_str_list[i] for i in decode_byte_list]
                    output_string += ''.join(str_list)
                except ReedSolomonError:
                    # Irreparable
                    return {"data": binstring, "type": False}
                except IndexError:
                    # No data acquisition
                    return {"data": binstring, "type": False}
            return {"data": output_string, "type": True}
        else:
            raise Exception("Wrong rs group")
    
    
    def insert(self, segment_list, group=1):
        log.debug('start adding rs code.')
        res_list = list()
        if len(segment_list)==0:
            raise ValueError("Empty data.")
        # Insert rs codes into multiple sequences
        if type(segment_list)==list and type(segment_list[0]==str):
            pro_bar = tqdm(total=len(segment_list), desc="Add RSCode")
            for segment in segment_list:
                res = self.insert_one(segment, group=group)
                res_list.append(res)
                pro_bar.update()
            pro_bar.close()
        else:
            raise Exception("Input error")
        return res_list

    def remove_one_crc(self, binstring, binori='', phred=[], group=1):
        self.remainder = 0
        flag =0
        # If the binary length is not a multiple of 8, it will be left-padded with zeros
        if len(binstring)==120:
            flag = 1


        if len(binstring) % 8 != 0:
            self.remainder = 8 - len(binstring) % 8
            binstring = '0' * self.remainder + binstring
        tnum = group * self.check_bytes * 8 + self.crccode * 8
        k = 0.99
        #
        data_len = len(binstring) - group * self.check_bytes * 8

        output_string = ""
        if group >= 1 and type(group) == int:
            if group <= data_len / 8 / 255:
                group = 1
            group_len = math.ceil(data_len / 8 / group) * 8 + self.check_bytes * 8
            decoded_successfully = False
            for g in range(group):
                last = group_len * (g + 1)
                if last >= len(binstring): last = None
                group_string = binstring[g * group_len:last]
                byte_list = bytearray([int(group_string[i:i + 8], 2) for i in range(0, len(group_string), 8)])

                # start

                original_data_decoded = byte_list[:(-self.check_bytes-self.crccode)]
                # crc_received = byte_list[(-self.check_bytes-1)]
                crc_received2 = byte_list[-(self.check_bytes+self.crccode):-(self.check_bytes)]
                # 重新计算 CRC
                crc_recalculated = crc16(original_data_decoded)
                crc_received = (crc_received2[0] << 8) | crc_received2[1]
                # 校验 CRC
                if crc_received == crc_recalculated:
                    # data_corrected = original_data_decoded
                    # decoded_successfully = True
                    return {"data": binstring[:(len(binstring)-tnum)], "type": True}

                # self.possible_error_pos = [(i+self.remainder) // 8 for i, value in enumerate(phred) if
                #                            value < k and (i+self.remainder) // 8 < len(byte_list)]
                if flag == 1:
                    a = 1
                #     print(f'phred:{phred[:5]}\n{binori}\n{binstring[self.remainder:]}')
                # self.possible_error_pos1 = [{i:value} for i, value in enumerate(phred) if value < k]
                sorted_indices = np.argsort(phred)
                # sorted_data = [phred[i] for i in sorted_indices[:5]]

                self.possible_error_pos = [(value+self.remainder) // 8 for i, value in enumerate(sorted_indices[:5]) if (value+self.remainder) // 8 < len(byte_list)]

                all_combinations = [()]
                for ti in range(1, self.check_bytes + 1):
                    all_combinations += list(combinations(self.possible_error_pos, ti))
                for tci,comb in enumerate(all_combinations):
                    try:
                        erase_pos = list(comb)
                        # data_corrected = list(self.RSCodec.decode(data, erase_pos=erase_pos)[0])
                        decode_byte_list, full_list, err_pos = self.tool.decode(byte_list, erase_pos=erase_pos)
                        # if len(err_pos) != 0:
                        #     print(err_pos)

                        # original_data_decoded = decode_byte_list[:-1]
                        # crc_received = decode_byte_list[-1]
                        #250416改为2个byte的crc校验码
                        original_data_decoded = decode_byte_list[:-self.crccode]
                        crc_received2 = decode_byte_list[(-self.crccode):]
                        crc_received = (crc_received2[0] << 8) | crc_received2[1]

                        # 重新计算 CRC
                        crc_recalculated = crc16(original_data_decoded)
                        decoded_successfully = True
                        str_list = [bin_to_str_list[i] for i in decode_byte_list[:-self.crccode]]
                        output_string += ''.join(str_list)
                        # 校验 CRC
                        if crc_received == crc_recalculated:
                            # print(f'tci:{tci}')
                            # data_corrected = original_data_decoded
                            # decoded_successfully = True
                            return {"data": output_string[:(len(binstring)-tnum)], "type": True}

                    except ReedSolomonError:
                        sorted_indices = np.argsort(phred)
                        sorted_data = [phred[i] for i in sorted_indices]
                        # print(f'\nrs decode error,all_combinations:{self.possible_error_pos1}\nbinori:{binori}\n   bin:{binstring[self.remainder:]}')
                        # print(f'\nrs decode error,sorted_indices:{sorted_indices[:10]}\nsorted_data:\n{sorted_data[:10]}\n{binori}\n{binstring[self.remainder:]}')
                        # return -1, None
                        continue
                    except IndexError:
                        # No data acquisition
                        return {"data": binstring[:(len(binstring)-tnum)], "type": False}
                #end
            # if not decoded_successfully:
            # print(f'not decoded_successfully')
            return {"data": binstring[:(len(binstring)-tnum)], "type": False}
            # print(f'rs decoded_successfully , but crc detected failed\n')
            # return {"data": output_string[:(len(binstring)-tnum)], "type": True}
        else:
            raise Exception("Wrong rs group")


    def remove_crc(self, segment_list, oriLen, phreds, crccode=0, group=1):
        self.crccode=crccode
        log.debug('Check and  remove rs code.')
        if len(segment_list) == 0:
            raise ValueError("Empty data.")
        bit_segments = []
        error_bit_segments = []
        error_indices = []
        k = 0.9
        if type(segment_list) == list and type(segment_list[0]) == str:
            error_rate = 0
            pro_bar = tqdm(total=len(segment_list), desc="Del RSCode")

            with open('messplains.txt', 'r') as f:
                lines = f.readlines()
            binori = [l.rstrip('\n') for l in lines]
            for index, segment in enumerate(segment_list):
                if segment is not None:
                    output = self.remove_one_crc(segment,binori[index], phreds[index], group=group)
                    data, is_succeed = output.get("data"), output.get("type")
                    if is_succeed and len(data) >= oriLen:
                        bit_segments.append(data[len(data) - oriLen:])
                    else:
                        success = False
                        if len(segment) > oriLen:
                            # print(f'序列长于原序列，尝试去除')
                            phred = phreds[index]
                            sorted_indices = np.argsort(phred)
                            #net test 增加insert的错误去除功能
                            for indice in sorted_indices[:2]:
                                modified_data = self.introduce_error(segment, indice)
                                output = self.remove_one_crc(modified_data, binori[index], phreds[index], group=group)
                                data, is_succeed = output.get("data"), output.get("type")
                                if is_succeed and len(data) >= oriLen:
                                    # print(f'序列长于原序列，尝试去除，去除后成功解码，去除的位置是:{indice}')
                                    bit_segments.append(data[len(data) - oriLen:])
                                    success = True
                                    break
                        if not success:
                            # print(f'datalen:{len(data)},orilen:{oriLen},not succeed:{index}')
                            error_rate += 1
                            error_indices.append(index)
                            error_bit_segments.append(data)
                else:
                    error_rate += 1
                    error_indices.append(index)
                    error_bit_segments.append(None)
                pro_bar.update()
            pro_bar.close()
            error_rate /= len(segment_list)
        else:
            raise ValueError("Input error")
        return {"bit": bit_segments, "e_r": error_rate, "e_i": error_indices, "e_bit": error_bit_segments}

    def introduce_error(self,binary_data, i, error_type=0):

        data_list = list(binary_data)

        if 0 <= i < len(data_list):
            data_list.pop(i)
        return ''.join(data_list)

    def remove_ori(self, segment_list, oriLen,crccode=0, group=1):
        self.crccode=crccode
        log.debug('Check and  remove rs code.')
        if len(segment_list)==0:
            raise ValueError("Empty data.")
        bit_segments = []
        error_bit_segments = []
        error_indices = []
        if type(segment_list) == list and type(segment_list[0]) == str:
            error_rate = 0
            pro_bar = tqdm(total=len(segment_list), desc="Del RSCode")
            for index, segment in enumerate(segment_list):
                if segment is not None:
                    output = self.remove_one_ori(segment, group=group)
                    data, is_succeed = output.get("data"), output.get("type")
                    if is_succeed and len(data)>=oriLen:
                        bit_segments.append(data[len(data) - oriLen:])
                    else:
                        error_rate += 1
                        error_indices.append(index)
                        error_bit_segments.append(data)
                else:
                    error_rate += 1
                    error_indices.append(index)
                    error_bit_segments.append(None)
                pro_bar.update()
            pro_bar.close()
            error_rate /= len(segment_list)
        else:
            raise ValueError("Input error")
        return {"bit": bit_segments, "e_r": error_rate, "e_i": error_indices, "e_bit": error_bit_segments}

    def remove_one_ori(self, binstring, group=1):
        self.remainder = 0
        # If the binary length is not a multiple of 8, it will be left-padded with zeros
        if len(binstring) % 8 != 0:
            self.remainder = 8 - len(binstring) % 8
            binstring = '0' * self.remainder + binstring
        tnum = group * self.check_bytes * 8 + self.crccode*8
        data_len = len(binstring) - group * self.check_bytes * 8
        output_string = ""
        if group >= 1 and type(group) == int:
            if group <= data_len / 8 / 255:
                group = 1
            group_len = math.ceil(data_len / 8 / group) * 8 + self.check_bytes * 8
            for i in range(group):
                last = group_len * (i + 1)
                if last >= len(binstring): last = None
                group_string = binstring[i * group_len:last]
                byte_list = bytearray([int(group_string[i:i + 8], 2) for i in range(0, len(group_string), 8)])
                try:
                    decode_byte_list, full_list, err_pos = self.tool.decode(byte_list)

                    original_data_decoded = decode_byte_list[:-self.crccode]
                    crc_recalculated = crc16(original_data_decoded)
                    crc_received2 = decode_byte_list[(-self.crccode):]
                    crc_received = (crc_received2[0] << 8) | crc_received2[1]
                    if crc_received != crc_recalculated:
                        return {"data": binstring[:(len(binstring)-tnum)], "type": False}

                    # if len(err_pos) != 0:
                    #     print(err_pos)
                    str_list = [bin_to_str_list[i] for i in decode_byte_list]
                    output_string += ''.join(str_list)
                except ReedSolomonError:
                    # Irreparable
                    return {"data": binstring[:(len(binstring)-tnum)], "type": False}
                except IndexError:
                    # No data acquisition
                    return {"data": binstring[:(len(binstring)-tnum)], "type": False}
            return {"data": output_string[:(len(binstring)-tnum)], "type": True}
        else:
            raise Exception("Wrong rs group")


    def remove(self, segment_list, oriLen, group=1):
        log.debug('Check and  remove rs code.')
        if len(segment_list)==0:
            raise ValueError("Empty data.")
        bit_segments = []
        error_bit_segments = []
        error_indices = []
        if type(segment_list) == list and type(segment_list[0]) == str:
            error_rate = 0
            pro_bar = tqdm(total=len(segment_list), desc="Del RSCode")
            for index, segment in enumerate(segment_list):
                if segment is not None:
                    output = self.remove_one(segment, group=group)
                    data, is_succeed = output.get("data"), output.get("type")
                    if is_succeed and len(data)>=oriLen:
                        bit_segments.append(data[len(data) - oriLen:])
                    else:
                        error_rate += 1
                        error_indices.append(index)
                        error_bit_segments.append(data)
                else:
                    error_rate += 1
                    error_indices.append(index)
                    error_bit_segments.append(None)
                pro_bar.update()
            pro_bar.close()
            error_rate /= len(segment_list)
        else:
            raise ValueError("Input error")
        return {"bit": bit_segments, "e_r": error_rate, "e_i": error_indices, "e_bit": error_bit_segments}
