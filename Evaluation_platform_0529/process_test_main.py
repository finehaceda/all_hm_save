import csv
import os
import random




from simusequencing import test_ill_simu, test_nona_simu
from dpconsensus import test_consus
from encode_decode_m.codec import DerrickEncode, \
    DerrickDecode, HedgesEncode, HedgesDecode, DNAFountainEncode, DNAFountainDecode, YinYangCodeEncode, \
    YinYangCodeDecode, PolarCodeEncode, PolarCodeDecode
# from encode_decode import Church,DNAFountain
# from encode_decode_m.encode_decode import Church,DNAFountain
from encode_decode_m.encode_decode import DNAFountain

# file_input = os.getcwd() + "/testFiles/dna.jpg"
from simu_and_consus import cheatseqs, adddt4simu_advanced
from utils import readfile, writefile, sequence_length, readtxt, accdanseqs

# file_input = os.getcwd() + "/testFiles/66.jpg"
# file_input = os.getcwd() + "/testFiles/44.jpg"
# file_input = os.getcwd() + "/testFiles/r1_consus.fasta"
# file_input = os.getcwd() + "/testFiles/11.txt"
# file_input = os.getcwd() + "/testFiles/55.jpg"

# file_input = os.getcwd() + "/testFiles/33.jpg"
file_input = os.getcwd() + "/testFiles/00.png"
# file_input = os.getcwd() + "/testFiles/1.py"
# file_input = os.getcwd() + "/testFiles/5.jpg"
# file_input = os.getcwd() + "/testFiles/11.pdf"
# file_input = os.getcwd() + "/testFiles/1.zip"
# file_input = os.getcwd() + "/testFiles/1.jpg"

# file_input = os.getcwd() + "/testFiles/consus.fa"
# file_input = os.getcwd() + "/testFiles/0115.jpg"
# file_input = os.getcwd() + "/testFiles/11.jpg"
# file_input = os.getcwd() + "/testFiles/22.png"
# file_input = os.getcwd() + "/testFiles/10.jpg"
# file_input = os.getcwd() + "/testFiles/begin.mp3"
# file_input = os.getcwd() + "/testFiles/1.zip"
# file_input = os.getcwd() + "/testFiles/1.mp4"
# file_input = os.getcwd() + "/testFiles/10.zip"
# file_input = os.getcwd() + "/testFiles/address.wjr.txt"
output_dir = os.getcwd() + "/testFiles/testResult/"
# output_file_path = os.getcwd() + "/testFiles/22_encode.fasta"

# E   IndexError: index 162 is out of bounds for axis 0 with size 158        dt4只能模拟158长度的序列

###   ①加入了RS纠错码
    # ②整合了一下框架，使它具有通用性，可以调用不同的编解码算法，对同一个文件进行编码，以下参数一样。dt4最长只能模拟158长度的序列
    # ③三代模拟工具模拟，DeSP的代码

# 问题：dt4合成错误后单独的文件？
# index_length=16+1
index_length=20
# index_length=12
hedges_coderatecode=3
# coderates = array([NaN, 0.75, 0.6, 0.5, 1. / 3., 0.25, 1. / 6.])
# test parameter
# sequence_length = 130
copy_num=10
dnafountain_redundancy = 0.4
# seqdeep = 20
min_gc = 0.4
max_gc = 0.6
max_homopolymer = 4
rule_num = 1
rs_num = 1
# rs_matrix_num = 4
# add_redundancy = True
add_redundancy = False
#######################################
matrix_n = 255
matrix_r = 32
crccode = 2
# matrix_n = 512
# matrix_r = 32
# add_primer = True
matrix_code = True
add_primer = False
primer_length = 20



def test_dnafountain(copy_num,filename):

    # 无任何错误时，sequence_length = 280，1.jpg,编码的序列共有192条，每条156nt，redundancy设置为0.27  ,encodetype='dnafountain'
    encode_worker = DNAFountainEncode(input_file_path=file_input, output_dir=output_dir, sequence_length=sequence_length, max_homopolymer=max_homopolymer,
                              rs_num=rs_num,crc_num=crccode,add_redundancy=add_redundancy, add_primer=add_primer, primer_length=primer_length, redundancy=dnafountain_redundancy)
    encode_worker.common_encode()
    # encode_worker.chunk_num=286
    # encode_worker.size=30
    print(f'encode_worker.chunk_num:{encode_worker.chunk_num}')
    print(f'encode_worker.chunk_num:{encode_worker.size}')
    # 二代模拟
    # output_file_path_simu = test_ill_simu(newdna_sequences_path=encode_worker.output_file_path, index_length=index_length,copy_num=copy_num,encodetype='dnafountain',with_para=False)
    # # 三代模拟

    output_file_path_simu = test_nona_simu(newdna_sequences_path=encode_worker.output_file_path, index_length=index_length, copy_num=copy_num,encodetype='dnafountain',with_para=False)
    # aaa = [encode_worker.output_file_path,[]]
    errorrate = accdanseqs(output_file_path_simu, encode_worker.output_file_path, copy_num)
    # decode_worker = DNAFountainDecode(input_file_path=aaa,orifile=file_input,output_dir=output_dir, chunk_num=encode_worker.chunk_num,
    #                                   max_homopolymer=max_homopolymer,rs_num=rs_num)
    decode_worker = DNAFountainDecode(input_file_path=output_file_path_simu,orifile=file_input,output_dir=output_dir, chunk_num=encode_worker.chunk_num,
                                      size = encode_worker.size,max_homopolymer=max_homopolymer,rs_num=rs_num,crc_num=crccode)
    return decode_worker.common_decode1(copy_num,errorrate,filename)

filename = f'files/test.csv'
test_dnafountain(7,filename)

# # for copy_num in range(1):
# for copy_num in range(7,16):
#     # filename = f'files/files0514/fountainred=0.4/copy{copy_num}.csv'
# #     filename = f'files/files0514/fountainred=0.4/copy13_15.csv'
#     filename = f'files/files0522/fountainred=0.4/copy{copy_num}_new.csv'
#     # filename = f'files/files0522/fountainred=0.3/copy{copy_num}.csv'
#     # filename = f'files/files0522/fountainred=0.5/copy9-15_new.csv'
#     for i in range(100):
#         code = test_dnafountain(copy_num,filename)

def test_derrick(copy_num):
    # derrick 测试时确保序列长度为260nt
    encode_worker = DerrickEncode(input_file_path=file_input, output_dir=output_dir, sequence_length=sequence_length,max_homopolymer=max_homopolymer,
                                  rs_num=rs_num, add_redundancy=add_redundancy,add_primer=add_primer, primer_length=primer_length,
                                 matrix_code=matrix_code,matrix_n=matrix_n,matrix_r=matrix_r)
    encode_worker.common_encode_matrix()
    # 二代模拟
    # test_ill_simu(newdna_sequences=output_file_path, index_length=index_length,copy_num=copy_num)
    output_file_path_simu = test_ill_simu(newdna_sequences_path=encode_worker.output_file_path, index_length=12,encodetype='ali',copy_num=copy_num,with_para=False)
    # # 三代模拟
    # test_nona_simu(newdna_sequences=output_file_path, index_length=index_length, copy_num=copy_num)
    # output_file_path_simu = test_nona_simu(newdna_sequences_path=encode_worker.output_file_path, index_length=12,encodetype='ali', copy_num=copy_num,with_para=False)

    decode_worker = DerrickDecode(input_file_path=output_file_path_simu[0], output_dir=output_dir,inputfileforcompare=file_input,
                                  matrix_code=matrix_code,matrix_n=matrix_n,matrix_r=matrix_r)
    # decode_worker = DerrickDecode(input_file_path=encode_worker.output_file_path, output_dir=output_dir,inputfileforcompare=file_input,
    #                               matrix_code=matrix_code,matrix_n=matrix_n,matrix_r=matrix_r)
    decode_worker.common_decode_matrix()
    print(f"encode_path:{encode_worker.output_file_path},decode_path:{decode_worker.output_file_path}")

# test_derrick(copy_num)

def test_hedges(copy_num):
    # 注意修改 activate_command 命令的路径
    encode_worker = HedgesEncode(input_file_path=file_input, output_dir=output_dir, sequence_length=sequence_length, max_homopolymer=max_homopolymer,
                              rs_num=rs_num,add_redundancy=add_redundancy, add_primer=add_primer, primer_length=primer_length,
                                 matrix_code=matrix_code,matrix_n=matrix_n,matrix_r=matrix_r,coderatecode=hedges_coderatecode)
    encode_worker.common_encode_matrix()
    # 二代模拟
    output_file_path_simu = test_ill_simu(newdna_sequences_path=encode_worker.output_file_path, index_length=index_length,encodetype='hedges',copy_num=copy_num)
    # # 三代模拟
    # output_file_path_simu = test_nona_simu(newdna_sequences_path=encode_worker.output_file_path, index_length=index_length,encodetype='hedges', copy_num=copy_num)

    # decode_worker = HedgesDecode(input_file_path=encode_worker.output_file_path,orifile=file_input, output_dir=output_dir)
    decode_worker = HedgesDecode(input_file_path=output_file_path_simu[0],orifile=file_input, output_dir=output_dir)
    decode_worker.common_decode_matrix()
# test_hedges(10)

def testyinyang(csvpath,copy_num):
    encode_worker = YinYangCodeEncode(input_file_path=file_input, output_dir=output_dir, sequence_length=sequence_length, max_homopolymer=max_homopolymer,
                              rs_num=rs_num,add_redundancy=add_redundancy, add_primer=add_primer, primer_length=primer_length,crccode=crccode)
    # encode_worker.common_encode()
    # encode_worker.common_encode11(file_input)
    # # 二代模拟 +'.00_yyc_rs1
    output_file_path_simu = test_ill_simu(newdna_sequences_path=encode_worker.output_file_path, index_length=index_length,encodetype='noali',copy_num=copy_num)
    # # 三代模拟
    # output_file_path_simu = test_nona_simu(newdna_sequences_path=encode_worker.output_file_path, index_length=index_length,encodetype='ali', copy_num=copy_num)
    # #
    print(f'output_file_path_simu:{output_file_path_simu}')
    errorrate = accdanseqs(output_file_path_simu, encode_worker.output_file_path, copy_num,'')
    # errorrate = accdanseqs(output_file_path_simu, encode_worker.output_file_path, copy_num)
    # decode_worker = YinYangCodeDecode(input_file_path=encode_worker.output_file_path,orifile=file_input,output_dir=output_dir,index_length=encode_worker.index_length,
    #                                   codeindex=encode_worker.index_length,total_count=encode_worker.total_count)
    # decode_worker = YinYangCodeDecode(input_file_path=encode_worker.output_file_path,orifile=file_input,output_dir=output_dir,crccode=crccode)
    decode_worker = YinYangCodeDecode(input_file_path=output_file_path_simu,orifile=file_input, output_dir=output_dir
                                      ,modelpath=encode_worker.output_file_path,crccode=crccode)
    decode_worker.common_decode(csvpath,errorrate)
    print(f"encode_path:{encode_worker.output_file_path},decode_path:{decode_worker.output_file_path}")

# csv_data = ('method', f"badbits", f'allbits', f'success:', 'fail', 'block_err_rate','errorrate', f'bit_rev','block_rev')
# csvpath = f'files/yyc_copy10_0309_address.wjr.txt.csv'
# csvpath = f'files/1test.csv'
# testyinyang(csvpath,3)
# with open('files/yyc_copy12.csv', 'a', encoding='utf8', newline='') as f:
# for copy_num in range(3,4):
#     # csvpath = f'files/1test.csv'
#     # csvpath = f'files/yycfiles_0416/copy{copy_num}_1.csv'
#     # csvpath = f'files/yycfiles_0417_rs2/copy{copy_num}.csv'
#     # csvpath = f'files/yycfiles_0417_rs4_allcopy.csv'
#     csvpath = f'files/yycfiles_0512_rs8/copy{copy_num}.csv'
#     # csvpath = f'files/yycfiles_0512_rs8_copy{copy_num}.csv'
#     # with open(csvpath, 'a', encoding='utf8', newline='') as f:
#     #     writer = csv.writer(f)
#         # writer.writerow((''))
#         # writer.writerow((''))
#         # writer.writerow((f'copy={copy_num}'))
#         # writer.writerow(csv_data)
#     # # testyinyang()
#     for i in range(100):
#         testyinyang(csvpath,copy_num)

def testpolar():
    # 测试时确保序列长度为260nttestpolar
    encode_worker = PolarCodeEncode(input_file_path=file_input, output_dir=output_dir, frozen_bits_len=5)
    encode_worker.common_encode()
    # # 二代模拟 +'.00_yyc_rs1
    output_file_path_simu = test_ill_simu(newdna_sequences_path=encode_worker.output_file_path,
                                          index_length=index_length, encodetype='ali', copy_num=copy_num)
    # # 三代模拟
    # output_file_path_simu = test_nona_simu(newdna_sequences_path=encode_worker.output_file_path, index_length=index_length,encodetype='ali', copy_num=copy_num)

    decode_worker = PolarCodeDecode(input_file_path=output_file_path_simu[0],orifile=file_input,output_dir=output_dir,
                                    matrices_ori=encode_worker.matrices_ori, matrices_dna_ori=encode_worker.matrices_dna_ori, matrices_01_ori=encode_worker.matrices_01_ori)
    # decode_worker = YinYangCodeDecode(input_file_path=output_file_path_simu,orifile=file_input, output_dir=output_dir, index_length=encode_worker.index_length
    #                           , codeindex=encode_worker.index_length,modelpath=encode_worker.output_file_path)
    decode_worker.common_decode()
    print(f"encode_path:{encode_worker.output_file_path},decode_path:{decode_worker.output_file_path}")
# testpolar()

def testpolar_noerror():
    # 测试时确保序列长度为260nt
    encode_worker = PolarCodeEncode(input_file_path=file_input, output_dir=output_dir, frozen_bits_len=5)
    encode_worker.common_encode()
    decode_worker = PolarCodeDecode(input_file_path=encode_worker.output_file_path,orifile=file_input,output_dir=output_dir,
                                    matrices_ori=encode_worker.matrices_ori, matrices_dna_ori=encode_worker.matrices_dna_ori, matrices_01_ori=encode_worker.matrices_01_ori)
    # decode_worker = YinYangCodeDecode(input_file_path=output_file_path_simu,orifile=file_input, output_dir=output_dir, index_length=encode_worker.index_length
    #                           , codeindex=encode_worker.index_length,modelpath=encode_worker.output_file_path)
    decode_worker.common_decode()
    print(f"encode_path:{encode_worker.output_file_path},decode_path:{decode_worker.output_file_path}")
# testpolar_noerror()

# test_hedges()
# test_church()
# test_derrick()
# test_dnafountain()
# testyinyang()
# testpolar()



