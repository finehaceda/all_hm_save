import csv
import gzip
import random
import re
from datetime import datetime
from re import search
# 合成使用primers_0，合成后的序列为 primers_0[0] + ... + pri0pre20,其中pri0pre20是primers_0[1]的反向互补链
# 若序列过长，则为
import numpy as np

resultpath = './files/simu/'
primers_0 = ["ACACGACGCTCTTCCGATCT", "AGACGTGTGCTCTTCCGATCT"]
primers_2 = ["AATGATACGGCGACCACCGAGATCTACACTCTTTCCCTACACGACGCTCTTCCGATCT",
             "CAAGCAGAAGACGGCATACGAGATCGTGATGTGACTGGAGTTCAGACGTGTGCTCTTCCGATCT"]
# primers_0 = ["", ""]
# primers_2 = ["", ""]
pri0pre20 = "AGATCGGAAGAGCACACGTCT"
pri1pre20 = "AGATCGGAAGAGCGTCGTGT"
# pri0pre20 = "AGATCGGAAGAGCAC"
# pri1pre20 = "AGATCGGAAGAGCGT"
# primer_syn = "GTCACATCACGATCTCGTATGCCGTCTTCTGCTTG"
# primer_1reversed = "AGATCGGAAGAGCGTCGTGTAGGGAAAGAGTGTAGATCTCGGTGGTCGCCGTATCATT"
# primer_2reversed = "AGATCGGAAGAGCACACGTCTGAACTCCAGTCACATCACGATCTCGTATGCCGTCTTCTGCTTG"
primer_syn = ""
primer_1reversed = ""
primer_2reversed = ""
base_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
index_base = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
test_seqlen = 100
syn_to_badread = 20
# onethreadreads = 2000
onethreadreads = 120
# syn_to_badread = 10
# syn_to_badread = 30
# sequence_length = 124
# sequence_length = 130
# sequence_length = 248
# sequence_length = 488
# sequence_length = 152
# sequence_length = 495
sequence_length = 260


# sequence_length = 156

# myfile = './testFiles/33.jpg'
# resultpath = './testFiles/testResult/'
# encodefile = resultpath+'33_derrick_encode.fasta'
# messplainpath = resultpath+'messplain.txt'


class SimuInfo():
    # def __init__(self, channel:list , syncycles:int , synyield:int, pcrcycle:int, pcrpro:int, decay_year:int,
    #              decaylossrate: int, sample_ratio: int, syn_method: int, depth: int,):
    def __init__(self, inputfile_path: str, channel: list, synthesis_method='electrochemical', oligo_scale: int = 1,
                 sample_multiple: int = 100,
                 pcrcycle: int = 2, pcrpro: float = 0.8, decay_year: int = 2, decaylossrate: float = 0.3,
                 temperature: int = 20, humidity: float = 0.5,
                 sample_ratio: float = 0.005,
                 sequencing_method: str = 'paired-end', depth: int = 10, sequence_length: int = 120,
                 badparams: str = '', thread: int = 20):
        self.inputfile_path = inputfile_path
        self.channel = channel
        self.oligo_scale = int(oligo_scale)
        self.synthesis_method = synthesis_method
        self.sample_multiple = int(sample_multiple)
        self.pcrcycle = int(pcrcycle)
        self.pcrpro = float(pcrpro)
        self.decay_year = int(decay_year)
        self.decaylossrate = float(decaylossrate)
        self.temperature = int(temperature)
        self.humidity = float(humidity)
        self.sample_ratio = float(sample_ratio)
        self.sequencing_method = sequencing_method
        self.depth = int(depth)
        self.sequence_length = int(sequence_length)
        self.badparams = str(badparams)
        self.param = ''
        self.thread = int(thread)


initial_params = {
    "method": "fountain",
    # "method": "YYC",
    # "method": "derrick",
    # "method": "hedges",
    'seq_length': sequence_length,
    'mingc': 40,
    'maxgc': 60,
    'ecc': 1,
    'index_length': 12,
    'c_dist': '0.03',
    'delta': '0.5',
    'crc': '0',
    'rs_num': 4, 'redundancy_rate': 0.2, 'homopolymer': 5,
    'channel': [
        'synthesis', 'PCR', 'decay', 'sequencing', 'sampling'
    ],
    # 'channel':[
    #
    # ],
    # 'channel':[
    #     'synthesis','sequencing'
    # ],
    # 'channel':[
    #     'synthesis'
    # ],
    'gc': 0.2,
    'rebuild_method': 'SeqFormer',
    'oligo_scale': 1,
    'sample_multiple': 200,
    'pcrcycle': 20,
    'pcrpro': 0.9,
    'decay_year': 1,
    'decaylossrate': 0.5,
    'temperature': 20,
    'humidity': 0.5,
    'sequencing_method': 'paired-end',
    # 'sequencing_method':'Nanopone',
    'depth': 100,
    'thread_num': 20,
    'copy_number': 10,
    'reconstruct': 'no',
    'cluster_method': 'allbase',
    # 'cluster_method':'no',
    'confidence': 'no',
    'decision': 'hard',
    'sample_ratio': 200,
    'max_iterations': 100,
    'matrix_n': 255,
    'matrix_r': 32,
    'matrix_n_h': 255,
    'matrix_r_h': 32,
    # 'coderatecode':'3',
    'coderatecode': '1',
    'frozen_bits_len': 5,
    'synthesis_method': 'electrochemical',
}


def check(sequence, max_homopolymer, max_content):
    if max_homopolymer and not homopolymer(sequence, max_homopolymer):
        return False
    if max_content and not gc_content(sequence, max_content):
        return False

    return True


def homopolymer(sequence, max_homopolymer):
    homopolymers = "A{%d,}|C{%d,}|G{%d,}|T{%d,}" % tuple([1 + max_homopolymer] * 4)
    return False if search(homopolymers, sequence) else True

def crc8(self, data, polynomial=0x07, initial_value=0x00, final_xor_value=0x00):
    crc = initial_value
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = (crc << 1) ^ polynomial
            else:
                crc <<= 1
        crc &= 0xFF
    return crc ^ final_xor_value

def crc16(data: bytes, poly=0x11021, init_crc=0xFFFF):
    crc = init_crc
    for byte in data:
        crc ^= (byte << 8)
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ poly
            else:
                crc <<= 1
            crc &= 0xFFFF  # 保证 CRC 保持在 16 位范围内
    return crc


def gc_content(sequence, max_content):
    return (1 - max_content) <= (float(sequence.count("C") + sequence.count("G")) / float(len(sequence))) <= max_content


def readgzipfile(file_path):
    allseqs, allphreds = [], []
    with gzip.open(file_path, 'rt') as f:
        lines = f.readlines()
    for i in range(0, len(lines), 4):
        allseqs.append(lines[i + 1].strip())
        allphreds.append(lines[i + 3].strip())
        # for line in f:
        #     if line.startswith('A') or line.startswith('C') or line.startswith('G') or line.startswith('T'):
        #         allseqs.append(line.strip())
    return allseqs, allphreds


# 'GAAGATTCAGCACCATGAGCCTTAGATGATTGCACGACTTACGGTGGCCAGGGTGTCCCGTACTTGTTTGGGTCCTTACCGATGATCATTGTCATAGACT',
# 'TCCGAGCGTATTCTGAGTTAAAGGTACCGCATACAGCATGCTACCACGAAGAAAGGCGAGGATGCTTCTGCGCACTCAACGGAAGCGGTTTAACACACAT',
# 'AGCTGTAAACGGTCGATAATGATGTACATCTCGGTCATCGGACACCTTCCCACCAGAACAGGCCGCTATACATTACCCAAGCTCCTATTGGCAGCCTTGC',
# 'GCGGACAAAGTGCCAAACTCTACGGGAGCCTGTTGGGTCCTTAAACTGGGAGGAGTTCGTTTGGTCTTACGGCATAGACTACATCTGGTCTGGTGAAACA'


def readgfastqfile_withfirstline(file_path):
    allseqs, allphreds = [], []
    with open(file_path, 'rt') as f:
        lines = f.readlines()
    for i in range(1, len(lines), 4):
        allseqs.append(lines[i + 1].strip())
        allphreds.append(lines[i + 3].strip())
    return allseqs, allphreds


def readfile(file_path):
    allseqs = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    firstline = lines[0].strip('\n')
    for i in range(1, len(lines), 2):
        allseqs.append(lines[i + 1].strip('\n'))
    return allseqs, firstline


def readtxt(file_path):
    allseqs = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    for i in range(len(lines)):
        allseqs.append(lines[i].strip('\n'))
    return allseqs


def readsynfasta_(file_path):
    allinfos, allseqs = [], []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    for i in range(0, len(lines), 2):
        allinfos.append(lines[i])
        allseqs.append(lines[i + 1].strip('\n'))
    return allinfos, allseqs


def readsynfasta(file_path):
    allinfos, allseqs = [], []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    for i in range(0, len(lines), 2):
        nums = int(lines[i].strip('\n').split(':')[-1])
        # print(lines[i])
        for j in range(nums):
            allseqs.append(lines[i + 1].strip('\n'))
            allinfos.append(f">#{str(i).zfill(6)}_{str(j).zfill(2)}")
    return allinfos, allseqs


def readsynfasta_diff(file_path):
    allinfos, allseqs = [], []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    for i in range(0, len(lines), 2):
        allinfos.append(lines[i])
        allseqs.append(lines[i + 1].strip('\n'))
    return allinfos, allseqs


def writefile(file_path, seqs, firstline):
    with open(file_path, 'w') as f:
        f.write(f"{firstline}\n")
        for i in range(len(seqs)):
            f.write(f">seq{i}\n{seqs[i]}\n")


def saveseqs(file_path, seqs):
    with open(file_path, 'w') as f:
        for i in range(len(seqs)):
            f.write(f"{seqs[i]}\n")


def savefasta(file_path, seqs):
    with open(file_path, 'w') as f:
        for i in range(len(seqs)):
            f.write(f">seq{i}\n{seqs[i]}\n")


def savelistfasta(file_path, seqs):
    t = 0
    allseqs = []
    with open(file_path, 'w') as f:
        for i in range(len(seqs)):
            for j in range(len(seqs[i])):
                allseqs.append(seqs[i][j])
                f.write(f">seq{t}\n{seqs[i][j]}\n")
                t += 1
    return allseqs


def readsynseqs(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    allseqs = []
    for i in range(0, len(lines), 2):
        nums = int(lines[i].strip('\n').split(':')[-1])
        if nums > 2:
            allseqs.append(lines[i + 1].strip('\n'))
    return allseqs


def readseqs(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    allseqs = []
    for i in range(0, len(lines), 2):
        allseqs.append(lines[i + 1].strip('\n'))
    return allseqs


def readtestfastq(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    allseqs, allphreds = [], []
    # allseqs = []
    # minlen,maxlen = sequence_length*0.5,sequence_length*1.5
    try:
        for i in range(0, len(lines), 4):
            # if i+1 < len(lines) and minlen<len(lines[i+1])<maxlen:
            if i + 1 < len(lines):
                allseqs.append(lines[i + 1].strip('\n'))
                allphreds.append(lines[i + 3].strip('\n'))
    except:
        return allseqs[:len(allphreds)], allphreds
    return allseqs, allphreds

    # with open(filepath+'dt4dds_syn_manage.fasta','w') as f:
    #     for i in range(len(allseqs)):
    #         f.write(f">#seqs{i}\n{allseqs[i]}\n")
    # return allseqs,filepath+'dt4dds_syn_manage.fasta'
    # return allseqs


def readandsave_noline0(path, pathsave):
    allseqs = []
    with open(path, 'r') as f:
        lines = f.readlines()
    for i in range(2, len(lines), 2):
        allseqs.append(lines[i].strip('\n'))
    with open(pathsave, 'w') as f:
        f.writelines(lines[1:])
    return allseqs, lines[0]


def readandsave(path, pathsave):
    allseqs = []
    with open(path, 'r') as f:
        lines = f.readlines()
    param = ''
    if lines[0].startswith('method'):
        param = lines[0].rstrip()
    for i in range(1, len(lines), 2):
        allseqs.append(lines[i].strip('\n'))
    with open(pathsave, 'w') as f:
        f.writelines(lines)
    return allseqs, param

def readandsavefastq(path, pathsave):
    allseqs,allcon = [],[]
    with open(path, 'r') as f:
        lines = f.readlines()
    param = ''
    if lines[0].startswith('method'):
        param = lines[0].rstrip()
        lines = lines[1:]
    for i in range(0, len(lines), 4):
        allseqs.append(lines[i+1].strip('\n'))
        allcon.append(lines[i+3].strip('\n').split(' '))
        # allcon.append(lines[i+3].strip('\n').split(' ')[:-1])
    with open(pathsave, 'w') as f:
        f.writelines(lines)
    return allseqs,allcon, param

def read_unknown_andsave_hedges(path, pathsave, copy_num):
    print(f'hedges:copynum:{copy_num}')
    allseqs = []
    param = ''
    with open(path, 'r') as f:
        lines = f.readlines()
    if lines[0].startswith('method'):
        param = lines[0].rstrip()
    with open(path, 'r') as f:
        content = f.read()
    sequence_blocks = content.split('****')
    for block in sequence_blocks:
        # 分割拷贝行，过滤空行
        copies = [line.strip() for line in block.strip().split('\n') if line.strip()]
        # 仅保留前copy_num条拷贝
        limited_copies = copies[:copy_num]
        if limited_copies:  # 忽略空块
            allseqs.append(limited_copies)

    with open(pathsave, 'w') as f:
        for i in range(len(allseqs)):
            f.write(f">seq{i}\n{allseqs[i]}\n")
    print(f"共有{len(allseqs)}条序列")
    return allseqs, param


def read_unknown_andsave(path, pathsave):
    allseqs = []
    param = ''
    with open(path, 'r') as f:
        lines = f.readlines()
    if lines[0].startswith('method'):
        param = lines[0].rstrip()
    for i in range(len(lines)):
        if re.match(r"[ACGT]", lines[i][0]):
            allseqs.append(lines[i].strip('\n'))
        # if lines[i]=='\n': #derrick中，可能需要确保哪条序列是丢失的，才能解码成功
        #     allseqs.append('')
    with open(pathsave, 'w') as f:
        for i in range(len(allseqs)):
            f.write(f">seq{i}\n{allseqs[i]}\n")
    print(f"共有{len(allseqs)}条序列")
    return allseqs, param


def getRadomSeqsNoQua(dna_sequences, select_nums=5):
    # random.seed(27678)
    random.seed()
    new_dna_sequences = []
    allseqsnum = 0
    minnum = 1000
    for i in range(len(dna_sequences)):
        length = len(dna_sequences[i])
        allseqsnum += length
        if length < minnum: minnum = length
        if length >= select_nums:
            indexs = random.sample(range(length), select_nums)
        else:
            indexs = [i for i in range(length)]
        new_dna_sequences.append([dna_sequences[i][j] for j in indexs])
    # return np.array(new_dna_sequences),np.array(new_all_quas)
    print(f"\naverage seq num : {allseqsnum / len(dna_sequences)}")
    print(f"min seq num : {minnum}")
    return new_dna_sequences

def getrandomseqs(allseqs,select_nums=10):
    allseqsnum, minnum = 0,1000
    simuseqs_fixnum=[]
    for i in range(len(allseqs)):
        length = len(allseqs[i])
        allseqsnum += length
        if length < minnum: minnum = length
        if length > select_nums:
            indexs = random.sample(range(length), select_nums)
            simuseqs_fixnum.append([allseqs[i][j] for j in indexs if j < length])
        else:
            simuseqs_fixnum.append(allseqs[i])
    if select_nums!=1:
        print(f"\naverage seq num : {allseqsnum/len(simuseqs_fixnum)}")
        print(f"min seq num : {minnum}")
    return simuseqs_fixnum


def getrandomseqs_(allseqs,all_phreds,select_nums=10):
    allseqsnum, minnum = 0,1000
    simuseqs_fixnum,simphreds=[],[]
    for i in range(len(allseqs)):
        length = len(allseqs[i])
        allseqsnum += length
        if length < minnum: minnum = length
        if length > select_nums:
            indexs = random.sample(range(length), select_nums)
            simuseqs_fixnum.append([allseqs[i][j] for j in indexs if j < length])
            simphreds.append([all_phreds[i][j] for j in indexs if j < length])
        else:
            simuseqs_fixnum.append(allseqs[i])
            simphreds.append(all_phreds[i])
    if select_nums!=1:
        print(f"\naverage seq num : {allseqsnum/len(simuseqs_fixnum)}")
        print(f"min seq num : {minnum}")
    return simuseqs_fixnum,simphreds

def parse_dna_phred(file_path):
    entries = []
    current_dna = None
    all_phreds = []
    current_phreds = []

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # 跳过空行

            if re.match(r"[ACGT]", line[0]):
                # 保存上一个条目（如果有）
                if current_dna is not None:
                    all_phreds.append(current_phreds)
                # 开始新条目
                current_dna = line
                current_phreds = []
            elif line == '****':
                continue  # 跳过分隔符
            else:
                current_phreds.append(getphred_quality(line.strip('\n')))  # 收集PHRED行

        # 添加最后一个条目
        if current_dna is not None:
            all_phreds.append(current_phreds)

    return all_phreds


def getphred_quality(qualityScore):
    phred_qualitys = []
    # sss = seq[110:120]
    for index, i in enumerate(qualityScore):
        phred_quality = ord(i) - 33  # '@'的ASCII码是64，FASTQ使用的是Phred+33编码
        phred_qualitys.append(phred_quality)
    return phred_qualitys

def getoriandallseqs_nophred(path,copy_num):
    ori_dna_sequences, all_seqs = [], []
    with open(path, 'r') as f:
        lines = f.readlines()
    param = ''
    if lines[0].startswith('method'):
        param = lines[0].rstrip()
    maxl = len(lines)
    i = 0
    while i < maxl:
        ori_dna_sequences.append(lines[i].strip('\n'))
        i += 2
        seqs = []
        while i < maxl and re.match(r"[ACGT]", lines[i][0]):
            seqs.append(lines[i].rstrip())
            i += 1
        i += 2
        # if len(seqs)==0:
        #     ori_dna_sequences.pop()
        #     continue
        # print(i)
        all_seqs.append(seqs)
    all_seqs = getrandomseqs(all_seqs,copy_num)
    return ori_dna_sequences, all_seqs, param


def getoriandallseqs(path,cluster_phreds_path,copy_num):
    ori_dna_sequences, all_seqs,all_phreds = [], [], []
    with open(path, 'r') as f:
        lines = f.readlines()
    with open(cluster_phreds_path, 'r') as f:
        lines1 = f.readlines()
    param = ''
    if lines[0].startswith('method'):
        param = lines[0].rstrip()
    maxl = len(lines)
    i = 0
    while i < maxl:
        ori_dna_sequences.append(lines[i].strip('\n'))
        i += 2
        seqs = []
        phreds = []
        while i < maxl and re.match(r"[ACGT]", lines[i][0]):
            seqs.append(lines[i].rstrip())
            phreds.append(getphred_quality(lines1[i].strip('\n')))
            i += 1
        i += 2
        # if len(seqs)==0:
        #     ori_dna_sequences.pop()
        #     continue
        # print(i)
        all_seqs.append(seqs)
        all_phreds.append(phreds)
    all_seqs,all_phreds = getrandomseqs_(all_seqs,all_phreds,copy_num)
    return ori_dna_sequences, all_seqs, all_phreds,param


def write_dict_to_txt(dictionary, filename):
    # 打开文件，如果文件不存在会创建文件
    with open(filename, 'a') as file:
        for key, value in dictionary.items():
            # 将键值对写入文件，每行一个键值对，格式为 'key: value'
            if str(value).find(':') != -1:
                time_str = str(value)
                time_obj = datetime.strptime(time_str, "%H:%M:%S.%f")
                value = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second + time_obj.microsecond / 1e6
            file.write(f"{key}: {value}\n")
        file.write(f"\n")


def write_dict_to_csv(dictionary, filename):
    # 打开文件，如果文件不存在会创建文件
    values = [list(dictionary.keys()), []]
    for key, value in dictionary.items():
        # 将键值对写入文件，每行一个键值对，格式为 'key: value'
        if str(value).find(':') != -1:
            time_str = str(value)
            time_obj = datetime.strptime(time_str, "%H:%M:%S.%f")
            value = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second + time_obj.microsecond / 1e6
        values[1].append(value)
    if len(values) > 0:
        with open(filename + '.csv', 'a') as file:
            writer = csv.writer(file)
            writer.writerow(values[0])
            writer.writerow(values[1])
        # file.write(f"\n")


def getparamdict(param):
    paramdict = {}
    eparam = param.rstrip().split(',')
    for p in eparam:
        line = p.split(':')
        paramdict[line[0]] = line[1]
    return paramdict


def getbinfile(path):
    bin_to_str_list = list()
    for i in range(256):
        bin_to_str_list.append(bin(i)[2:].rjust(8, '0'))
    with open(path, 'rb') as f:
        bytesarr = f.read()
    str_list = [bin_to_str_list[i] for i in bytesarr]
    binstring = ''.join(str_list)
    binstring_bits = np.array([bit for bit in binstring], np.uint8)
    return binstring_bits
