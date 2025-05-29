import csv
import gzip
from re import search

import Levenshtein

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
# sequence_length = 120
sequence_length = 152
# sequence_length = 260
# sequence_length = 156

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
        if nums>0:
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


def accdanseqs(output_file_path_simu, output_file_path, copy_num,encodetype='dnafountain'):
    print(f'output_file_path_simu:{output_file_path_simu}\noutput_file_path:{output_file_path}')
    output_file,output_filequas = output_file_path_simu[0],output_file_path_simu[1]
    all_consus, all_bsalign_quas, all_ori_seqs = getallconsus(output_file,output_filequas,output_file_path,encodetype)
    lens = len(all_consus)
    mismatchnums = []
    delnums = []
    insertnums = []
    mismatchnum0 = 0
    phred_scores = []
    delnum0 = 0
    insertnum0 = 0
    all_dis = []
    all_phred_score = []
    pre_seqs_morethan10 = []
    all_nums_gt0dis = 0
    all_dis_gl0 = 0
    all_dis_sum = 0
    all_nums = 0
    aliseq1, aliseq2 = [], []
    print(f'len(all_consus):{len(all_consus)},len(all_ori_seqs):{len(all_ori_seqs)}')
    for i in range(len(all_consus)):
        dis = Levenshtein.distance(all_consus[i], all_ori_seqs[i])
        phredscore = all_bsalign_quas[i]
        edit_ops = Levenshtein.editops(all_ori_seqs[i], all_consus[i])
        insertnum = sum(1 for op in edit_ops if op[0] == 'insert')
        delnum = sum(1 for op in edit_ops if op[0] == 'delete')
        mismatch = sum(1 for op in edit_ops if op[0] == 'replace')
        all_phred_score.append(phredscore)
        mismatchnums.append(mismatch)
        delnums.append(delnum)
        aliseq1.append(edit_ops)
        insertnums.append(insertnum)
        mismatchnum0 += int(mismatch)
        delnum0 += int(delnum)
        insertnum0 += int(insertnum)
        all_dis.append(dis)
        num = 0
        phred = ''
        for j in range(len(phredscore)):
            phred += str(j) + ':' + str(phredscore[j]) + ' '
        phred_scores.append(phred)
        all_nums += num
        if dis >= 1:
            all_dis_gl0 += 1
            all_dis_sum += dis
            all_nums_gt0dis += num
        if dis > 10:
            pre_seqs_morethan10.append(all_consus[i])

    # print(f"错误数量超过10的序列数量:{len(pre_seqs_morethan10)}")
    # print(f"mismatch数量:{mismatchnum0} del数量:{delnum0} insert数量:{insertnum0} ")
    print('发生错误的序列数量:' + str(all_dis_gl0) + ' 发生错误数量占所有数量： ' + str(all_dis_gl0 / lens))
    # print('发生错误的碱基数量 :' + str(all_dis_sum))
    # print('每条序列碱基数量：' + str(len(all_ori_seqs[0])))
    # print('平均编辑距离:' + str(all_dis_sum / lens))
    # if all_dis_gl0 != 0:
    #     print('when dis >= 1 平均编辑距离:' + str(all_dis_sum / all_dis_gl0))
    # print('recovery rate:' + str(1 - all_dis_sum / lens / len(all_ori_seqs[0])))

    # seq_len = len(all_ori_seqs[0])
    # data = [
    #     # ('', 'rec rate', 'edit error rate', 'error rate', 'success rate', 'indels', 'error base num', 'error seq num'),
    #     (f"dp{copy_num}", 1 - (all_dis_sum) / lens / seq_len, (all_dis_sum) / lens / seq_len, all_dis_gl0 / lens,
    #      1 - all_dis_gl0 / lens,
    #      f"{sum(mismatchnums)}:{sum(delnums)}:{sum(insertnums)}", f"{all_dis_gl0}", f"{all_dis_sum}"),
    # ]
    # print(data)
    # data = [
    #     (f"dp{copy_num}", 1 - (all_dis_sum) / lens / len(all_ori_seqs[0]),
    #      (all_dis_sum) / lens / len(all_ori_seqs[0]), all_dis_gl0 / lens, 1 - all_dis_gl0 / lens
    #      , sum(mismatchnums) / lens / len(all_ori_seqs[0]), sum(delnums) / lens / len(all_ori_seqs[0]),
    #      sum(insertnums) / lens / len(all_ori_seqs[0]),
    #      f"{sum(mismatchnums)}:{sum(delnums)}:{sum(insertnums)}", all_dis_gl0, all_dis_sum),
    # ]
    # with open('./files/bsalign_oriandpreseqs.fasta', 'w') as file:
    #     for i in range(len(all_ori_seqs)):
    #         file.write(f">oriseq{i}\n{all_ori_seqs[i]}\n>preseq{i} edit:{all_dis[i]} mis:{mismatchnums[i]}"
    #                    f" del:{delnums[i]} insert:{insertnums[i]}\n{all_consus[i]}\n"
    #             f">aliops{i}\n{aliseq1[i]}\n>phred{i}{phred_scores[i]}\n")

    # with open('./models/id20_seqcluster_fix/id20_bs_dp_phred_1219_lt0.csv', 'a', encoding='utf8', newline='') as f:
    #     writer = csv.writer(f)  # csv.writer()中可以传一个文件对象
    #     for line in data:  # 该data既可以是列表嵌套列表的数据类型也可以是列表嵌套元组的数据类型
    #         writer.writerow(line)

    # filepath='./files/test_250216.csv'
    # ratetest = [[0, 0.4], [0.4, 0.6], [0.6, 0.8], [0.8, 0.9], [0.9, 0.99], [0.99, 1]]
    # ratetest = [[0, 0.2], [0.2, 0.4], [0.4, 0.5], [0.5, 0.6], [0.6, 0.7], [0.7, 0.8], [0.8, 0.9], [0.9, 1]]
    # ratetest = [[0, 0.4], [0.4, 0.6], [0.6, 0.8], [0.8, 0.9], [0.9, 1]]
    # getalians(all_ori_seqs, all_consus, all_dis, all_phred_score, ratetest, filepath,'bsalign',copy_num)
    # getalians_for_acc(all_ori_seqs, all_consus, all_dis, all_phred_score, ratetest, filepath, '', all_dis_sum, 'bsalign', copy_num)
    return all_dis_gl0 / lens


def getallconsus(output_file,output_filequas,output_file_path,encodetype = 'dnafountain'):
    all_consus,all_bsalign_quas,all_ori_seqs = [],[],[]


    if encodetype == 'dnafountain':
        with open(output_file_path, 'r') as file:
            lines = file.readlines()
        for l in range(len(lines)):
            all_ori_seqs.append(lines[l].rstrip())
        with open(output_file, 'r') as file:
            lines = file.readlines()
        for l in range(len(lines)):
            all_consus.append(lines[l].rstrip())
    else:
        with open(output_file_path, 'r') as file:
            lines = file.readlines()
        for l in range(2,len(lines),2):
            all_ori_seqs.append(lines[l].rstrip())
        with open(output_file, 'r') as file:
            lines = file.readlines()
        for l in range(2,len(lines),2):
            all_consus.append(lines[l].rstrip())


    with open(output_filequas, 'r') as file:
        lines = file.readlines()
    for l in range(1, len(lines), 2):
        try:
            # print(f"l:{l}lines[l]:{lines[l]}")
            line = lines[l].rstrip().replace(']', '').replace('[', '').split(', ')
            all_bsalign_quas.append([float(p) for p in line])
        except:
            all_bsalign_quas.append([])
    return all_consus,all_bsalign_quas,all_ori_seqs

def getalians_for_acc(ori_seqs, pre_seqs, all_dis, pre_seqsphred,ratetest,filepath,revdata,errorbasenum,method = 'SeqTransformer',copy_num = ''):
    truebasenumsarr1 = [0]*len(ratetest)
    truebasenumsarr2 = [0]*len(ratetest)
    basenumsarr = [0]*len(ratetest)
    errorseq_basenumsarr = [0]*len(ratetest)
    seqnumsarr = [0]*len(ratetest)
    upnumsallerrorssarr = []
    for i in range(len(ratetest)):
        upnumsallerrorssarr.append({ 'insert': 0,'delete': 0, 'replace': 0, 'equal': 0})
    for i in range(len(ori_seqs)):
        seq1, seq2, dis, prewith_phred_pulsdel = ori_seqs[i], pre_seqs[i], all_dis[i], pre_seqsphred[i]
        if len(prewith_phred_pulsdel)==0:
            continue
        for r in range(len(ratetest)):
            ratel, rater = ratetest[r][0],ratetest[r][1]
            nums = len([i for i in range(len(prewith_phred_pulsdel)) if ratel < prewith_phred_pulsdel[i] <= rater])
            basenumsarr[r] += nums
            if nums > 0:
                seqnumsarr[r] += 1
            if dis != 0:
                errorseq_basenumsarr[r] += nums
        if dis == 0:
            continue
        edit_ops = Levenshtein.editops(seq1, seq2)
        visited = set()

        for op, i1, j1 in edit_ops:
            if j1 < len(prewith_phred_pulsdel):
                minphred = prewith_phred_pulsdel[j1]
                mini = j1
                if op == 'insert':
                    thisbase = seq2[j1]
                    fi = j1+1
                    while fi < len(prewith_phred_pulsdel) and seq2[fi] == thisbase:
                        if prewith_phred_pulsdel[fi] < minphred and fi not in visited:
                            minphred = prewith_phred_pulsdel[fi]
                            mini = fi
                        fi += 1
                    visited.add(mini)
                    # print(f"********************************find insert one i:{i}i1:{i1}j1:{j1}")
                elif op == 'replace':
                    visited.add(mini)
                elif op == 'delete':
                    thisbase = seq2[j1]
                    fi = j1-1
                    while fi >= 0 and seq2[fi] == thisbase:
                        if prewith_phred_pulsdel[fi] < minphred and fi not in visited:
                            minphred = prewith_phred_pulsdel[fi]
                        fi -= 1
                    # print(f"********************************find {minphred} thisbase:{thisbase} i:{i} i1:{i1} j1:{j1} fi:{fi+1}")
            else:
                minphred = prewith_phred_pulsdel[-1]
            for r in range(len(ratetest)):
                ratel, rater = ratetest[r][0], ratetest[r][1]
                if ratel < minphred <= rater:
                    upnumsallerrorssarr[r][op] += 1
                    if op != 'equal':
                        truebasenumsarr1[r] += 1
                        if op != 'delete':
                            truebasenumsarr2[r] += 1
    data = []
    # print(f"method, copy_num, ratel, rater, 总碱基数量, 真正错误碱基数量, indels， indel错误识别率,准确率(判断当前分布碱基是否可靠),识别率(当前分布碱基错误数量/当前分布总数量),灵敏度(占所有错误的比例)")
    print(
        f"method, copy_num, 总碱基数量, 真正错误碱基数量, ins/sub错误识别率, 准确率(判断当前分布碱基是否可靠), 识别率(当前分布碱基错误数量/当前分布总数量),灵敏度(占所有错误的比例),ratel-rater")

    for r in range(len(ratetest)):
        divbase1 = '0.0000'
        if basenumsarr[r] != 0:
            # divbase1 = 1 - truebasenumsarr1[r] / basenumsarr[r]
            divbase1 = round(truebasenumsarr1[r] / basenumsarr[r],4)
        divbase2 = '0.0000'
        if basenumsarr[r] != 0:
            divbase2 = round(truebasenumsarr2[r] / basenumsarr[r],4)
        read = 0
        if errorbasenum != 0:
            # print(f"truebasenumsarr1[r]:{truebasenumsarr1[r]}")
            # print(f"errorbasenum:{errorbasenum}")round(divbase1, 4)
            read = round(truebasenumsarr1[r] / errorbasenum,4)
        ratel, rater = ratetest[r][0], ratetest[r][1]
        # d = (method, copy_num, ratel, rater, seqnumsarr[r], basenumsarr[r], errorseq_basenumsarr[r], truebasenumsarr1[r], upnumsallerrorssarr[r], divbase1, divbase2)
        # d = (method, copy_num, ratel, rater, basenumsarr[r], truebasenumsarr1[r], upnumsallerrorssarr[r], divbase2,1-float(divbase1), divbase1, read,f"{ratel}-{rater}")
        d = (method, copy_num, basenumsarr[r], round(truebasenumsarr1[r], 4), divbase2,f"acc:{(1 - float(divbase1)):.4f}",
             f"rec:{divbase1}", f"Sensitivity:{read}",f"{ratel}-{rater}")

        data.append(d)
        if len(ratetest) < 10:
            print(d)
    # with open('./models/derrick_seqcluster_10000_b64/derrick_bs_dp_phred_1218.csv', 'a', encoding='utf8',newline='') as f:
        # with open('./models/derrick_seqcluster_1000_b8/test.csv', 'a', encoding='utf8', newline='') as f:
    with open(filepath, 'a', encoding='utf8', newline='') as f:
        writer = csv.writer(f)
        for line in revdata:
            writer.writerow(line)
        for line in data:
            writer.writerow(line)


