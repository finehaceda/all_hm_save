import gzip
from re import search

resultpath = './testFiles/testResult/'
primers_0 = ["ACACGACGCTCTTCCGATCT", "AGACGTGTGCTCTTCCGATCT"]
primers_2 = ["AATGATACGGCGACCACCGAGATCTACACTCTTTCCCTACACGACGCTCTTCCGATCT", "CAAGCAGAAGACGGCATACGAGATCGTGATGTGACTGGAGTTCAGACGTGTGCTCTTCCGATCT"]
# primers_0 = ["", ""]
# primers_2 = ["", ""]
pri0pre20 = "AGATCGGAAGAGCACACGTC"
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
# sequence_length = 124
# sequence_length = 130
# sequence_length = 248
# sequence_length = 488
# sequence_length = 152
# sequence_length = 495
# sequence_length = 260
sequence_length = 156

bin_to_str_list = list()
for i in range(256):
    bin_to_str_list.append(bin(i)[2:].rjust(8, '0'))
# myfile = './testFiles/33.jpg'
# resultpath = './testFiles/testResult/'
# encodefile = resultpath+'33_derrick_encode.fasta'
# messplainpath = resultpath+'messplain.txt'
def check(sequence, max_homopolymer, max_content):
    if max_homopolymer and not homopolymer(sequence, max_homopolymer):
        return False
    if max_content and not gc_content(sequence, max_content):
        return False

    return True

def homopolymer(sequence, max_homopolymer):
    homopolymers = "A{%d,}|C{%d,}|G{%d,}|T{%d,}" % tuple([1 + max_homopolymer] * 4)
    return False if search(homopolymers, sequence) else True


def gc_content(sequence, max_content):
    return (1 - max_content) <= (float(sequence.count("C") + sequence.count("G")) / float(len(sequence))) <= max_content

def readgzipfile(file_path):
    allseqs,allphreds = [],[]
    with gzip.open(file_path, 'rt') as f:
        lines = f.readlines()
    for i in range(0,len(lines),4):
        allseqs.append(lines[i+1].strip())
        allphreds.append(lines[i+3].strip())
        # for line in f:
        #     if line.startswith('A') or line.startswith('C') or line.startswith('G') or line.startswith('T'):
        #         allseqs.append(line.strip())
    return allseqs,allphreds
 # 'GAAGATTCAGCACCATGAGCCTTAGATGATTGCACGACTTACGGTGGCCAGGGTGTCCCGTACTTGTTTGGGTCCTTACCGATGATCATTGTCATAGACT',
 # 'TCCGAGCGTATTCTGAGTTAAAGGTACCGCATACAGCATGCTACCACGAAGAAAGGCGAGGATGCTTCTGCGCACTCAACGGAAGCGGTTTAACACACAT',
 # 'AGCTGTAAACGGTCGATAATGATGTACATCTCGGTCATCGGACACCTTCCCACCAGAACAGGCCGCTATACATTACCCAAGCTCCTATTGGCAGCCTTGC',
 # 'GCGGACAAAGTGCCAAACTCTACGGGAGCCTGTTGGGTCCTTAAACTGGGAGGAGTTCGTTTGGTCTTACGGCATAGACTACATCTGGTCTGGTGAAACA'



def readfile(file_path):
    allseqs = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    firstline = lines[0].strip('\n')
    for i in range(1,len(lines),2):
        allseqs.append(lines[i+1].strip('\n'))
    return allseqs,firstline


def readtxt(file_path):
    allseqs = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    for i in range(len(lines)):
        allseqs.append(lines[i].strip('\n'))
    return allseqs



def writefile(file_path,seqs,firstline):
    with open(file_path, 'w') as f:
        f.write(f"{firstline}\n")
        for i in range(len(seqs)):
            f.write(f">seq{i}\n{seqs[i]}\n")

def saveseqs(file_path,seqs):
    with open(file_path, 'w') as f:
        for i in range(len(seqs)):
            f.write(f">seq{i}\n{seqs[i]}\n")




def readsynseqs(filepath):
    with open(filepath,'r') as f:
        lines = f.readlines()
    allseqs = []
    for i in range(0,len(lines),2):
        nums = int(lines[i].strip('\n').split(':')[-1])
        if nums>2:
            allseqs.append(lines[i+1].strip('\n'))
    return allseqs

def readseqs(filepath):
    with open(filepath,'r') as f:
        lines = f.readlines()
    allseqs = []
    for i in range(0,len(lines),2):
        allseqs.append(lines[i+1].strip('\n'))
    return allseqs

def readtestfastq(filepath):
    with open(filepath,'r') as f:
        lines = f.readlines()
    # allseqs,allphreds = [],[]
    allseqs = []
    # minlen,maxlen = sequence_length*0.5,sequence_length*1.5
    for i in range(0,len(lines),4):
        # if i+1 < len(lines) and minlen<len(lines[i+1])<maxlen:
        if i+1 < len(lines):
            allseqs.append(lines[i+1].strip('\n'))
        # allphreds.append(lines[i+3].strip('\n'))
    return allseqs,allseqs

    # with open(filepath+'dt4dds_syn_manage.fasta','w') as f:
    #     for i in range(len(allseqs)):
    #         f.write(f">#seqs{i}\n{allseqs[i]}\n")
    # return allseqs,filepath+'dt4dds_syn_manage.fasta'
    # return allseqs





