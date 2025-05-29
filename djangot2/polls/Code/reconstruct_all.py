import os
import re
import subprocess
from datetime import datetime

import Levenshtein

from polls.Code.plot_plt import saveseqsdistributed_fig
from polls.Code.reconstruct.dpconsensus import test_consus
from polls.Code.utils import getoriandallseqs_nophred, getrandomseqs, parse_dna_phred, getoriandallseqs

basedir = os.getcwd() + '/polls/Code/reconstruct'

def reconstruct_seq(method,confidence,param,cluster_file_path,copy_number=10,
                    dir='/home1/hongmei/00work_files/0000/0isecondwork/Evaluation_platform/djangot2/files/decode/'):
    print(f"--------开始重建reconstruct_seq--------\ncluster_file_path:{cluster_file_path}")
    cluster_seqs_path=cluster_file_path['cluster_seqs_path']
    cluster_phreds_path=cluster_file_path.get('cluster_phred_path','')
    starttime = datetime.now()
    # cluster_file_path=dir+'cluster.fasta'
    path=dir+'reconstruct.fastq'
    print(f"使用的序列重建方法为：{method}")
    all_consus,all_quas = [],[]
    #all_seqs_ori包含了测序序列为0的reads，all_seqs为不含测序序列为0的reads
    if cluster_phreds_path != '':
        ori_dna_sequences_ori, all_seqs_ori, all_quas, _ = getoriandallseqs(cluster_seqs_path, cluster_phreds_path, copy_number)
        print(f'序列重建-存在碱基置信度值可使用！')
    else:
        ori_dna_sequences_ori, all_seqs_ori, _ = getoriandallseqs_nophred(cluster_seqs_path, copy_number)
        # all_quas = parse_dna_phred(cluster_phreds_path)
    # saveseqsdistributed_fig(all_seqs)
    lasti = [i for i in range(len(all_seqs_ori)) if len(all_seqs_ori[i]) == 0]
    # last_oriseqs = [ori_dna_sequences[i] for i in lasti]
    all_seqs = [all_seqs_ori[i] for i in range(len(all_seqs_ori)) if i not in lasti]
    if len(all_quas) == 0:
        all_quas = all_seqs
    else:
        all_quas = [all_quas[i] for i in range(len(all_seqs_ori)) if i not in lasti]
    ori_dna_sequences = [ori_dna_sequences_ori[i] for i in range(len(ori_dna_sequences_ori)) if i not in lasti]
    # print(f"注意，这里有{len(last_oriseqs)}条序列丢失！")
    if copy_number==1:
        print(f"注意，这里仅使用1条序列！")
        all_consus = getrandomseqs(all_seqs, copy_number)
        all_consus = [all_consus[i][0] for i in range(len(all_consus))]
        all_quas = [[0.99 for _ in range(len(all_consus[i]))] for i in range(len(all_consus))]
    elif method == 'bsalign':
        for seqsi in range(len(all_seqs)):
            if len(all_seqs[seqsi])==0:
                all_consus.append('')
                all_quas.append([])
                continue
            _, _, qua, consusno_ = bsalign_alitest_1119(all_seqs[seqsi])

            all_consus.append(consusno_)
            all_quas.append(qua)
    elif method == 'SeqFormer':
        # 解码前纠错，得到共识序列,（使用我们的模型进行纠错）
        print(f'copy_number:{copy_number}')
        all_consus, all_quas,bs_consus = test_consus.getconsensus(ori_dna_sequences,all_seqs,all_quas,copy_number)
    else:
        all_consus = bmarun(method,cluster_file_path)
    all_consus, all_quas = getdatas(all_seqs_ori, all_quas, all_consus)
    if confidence == 'yes':
        save_seqs_with_dis(all_consus,ori_dna_sequences_ori,param,dir+'reconstruct.fasta')
        num,editerrorrate,seqerrorrate = save_seqs_and_confidence(all_consus,all_quas,ori_dna_sequences_ori,param,path)
    else:
        path = dir+'reconstruct.fasta'
        num,editerrorrate,seqerrorrate = save_seqs_with_dis(all_consus,ori_dna_sequences_ori,param,path)
    print(f"--------重建结束reconstruct_seq--------\n重建序列共有{len(all_consus)}条，重建序列与原序列总编辑距离为：{num}")
    return path,datetime.now()-starttime,editerrorrate,seqerrorrate

def getdatas(lastseqs,consensus_phreds,con_consensus_seqs):
    if len(lastseqs) != len(con_consensus_seqs):
        consensus_seqs = []
        consensus_p = []
        j = 0
        for i in range(len(lastseqs)):
            if len(lastseqs[i]) == 0:
                consensus_seqs.append('')
                consensus_p.append('')
            else:
                consensus_seqs.append(con_consensus_seqs[j])
                consensus_p.append(consensus_phreds[j])
                j += 1
    else:
        consensus_seqs = con_consensus_seqs
        consensus_p = consensus_phreds
    return consensus_seqs,consensus_p

def bmarun(m,filepath):
    with open(filepath,'r') as f:
        lines = f.readlines()
    print(f"文件{filepath}共有{len(filepath)}行")
    # shell = f'{basedir}/{m}/DNA {filepath} out > {basedir}/{m}/result.txt'
    shell = f'{basedir}/{m}/DNA {filepath} out > ./result.txt'
    result = subprocess.run(shell, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(f"shell语句为：{shell}\n使用{m}重建完成，result.stderr:{result.stderr}")
    # print(f"重建完成，result.stderr:{result.stderr}")
    # return process_file(f'{basedir}/{m}/result.txt')
    return process_file(f'./result.txt')

def process_file(path):
    allseqs=[]
    with open(path, 'r') as f:
        lines = f.readlines()
    for i in range(len(lines)):
        if lines[i].find('Total number of clusters') != -1:
            break
        if re.match(r"[ACGT]",lines[i][0]):
            allseqs.append(lines[i].rstrip())
    return allseqs

def bsalign_alitest_1119(cluster_seqs):
    num = len(cluster_seqs)
    # 0925 hm改，为了得到bsalign的质量值
    # save_seqs_remove_dis(cluster_seqs, './files/seqs.fasta')
    save_seqs(cluster_seqs, './files/seqs.fasta')
    shell = 'polls/Code/reconstruct/bsalign/bsalign poa files/seqs.fasta -o files/consus.txt -L > files/ali.ali'
    result = subprocess.run(shell, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # sleep(300)
    seqs,bsquas,aliconsus = read_alifilestest('files/ali.ali', num)
    # 将ASCII质量分数转换为整数
    bsquas = quality_scores_to_probabilities(bsquas)
    quasdict = []
    # quasno_ = []
    # with open('files/consus.txt', 'r') as file:
    #     lines = file.readlines()
    #     for line in lines:
    #         consus = line.strip('\n')
    # return seqs, consus, bsquas, consus
    consus = aliconsus
    # # if num <=3:
    #     return seqs,consus,bsquas,consus
    myseq = "" #包含了-的序列
    myseqno_ = ""
    for i in range(len(seqs[0])):
        dict = {'A':0,'C':0,'G':0,'T':0,'-':0}
        for j in range(num):
            dict[seqs[j][i].upper()]+=1
        max_key = max(dict, key=dict.get)
        myseq+=max_key
        quasdict.append(dict)
    # newquasdict = []
    for i in range(len(myseq)):
        flag = False
        if myseq[i] != '-':
            max_key = myseq[i]
            flag = True
        # if myseq[i] == '-' and quasdict[i][myseq[i]] <= 0.6:
        #     max_value = sorted(dict.values())[-2]
        #     max_key = next(key for key, value in dict.items() if value == max_value)
        #     flag = True
        if flag:
            myseqno_ += max_key
            # newquasdict.append(quasdict[i])
    # quasdict = newquasdict
    indexori,insertindexs,delindex = getfirstindex(consus,myseqno_)
    index = indexori
    newmyseqno_ = ""
    quasno_ = []
    indexd = 0
    try:
        for i in range(len(myseqno_)+len(delindex)):
            if i in insertindexs:
                continue
            elif i in delindex:
                newmyseqno_+=consus[index]
                # quasno_.append(0.35)
                # thisqua = 0.35
                thisqua = bsquas[index]
                indexd+=1
                index+=1
            else:
                thisqua = bsquas[index]
                if myseqno_[i-indexd] == consus[index]:
                    newmyseqno_+=consus[index]
                    # quasno_.append(quasdict[i-indexd][myseqno_[i-indexd]]/num)
                    # thisqua = quasdict[i-indexd][myseqno_[i-indexd]]/num
                    index+=1
                else:
                    newmyseqno_+=consus[index]
                    # quasno_.append(quasdict[i-indexd][consus[index]]/num)
                    # thisqua = quasdict[i-indexd][consus[index]]/num
                    index+=1
            # if thisqua > 0.99999:
            #     thisqua = 0.99
            quasno_.append(thisqua)
    except IndexError:
        # print(f"!!!!!!!!!!!!!!!!!!!!!!!!--------------------------bsalign-IndexError--------------------------!!!!!!!!!!!!!!!!!!!!!!!!")
        # print(myseqno_)
        if len(myseqno_) ==len(bsquas):
            # bsquas = ''.join(bsquas)
            return seqs,consus,bsquas,myseqno_
        else:
            quas = [0.99 for _ in range(len(myseqno_))]
            return seqs,consus,quas,myseqno_
    myseqno_ = newmyseqno_
    if len(myseqno_)!=len(quasno_):
        print("error!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # quasno_ = ''.join(quasno_)
    return seqs,consus,quasno_,myseqno_



def save_seqs(seq, path):
    with open(path, 'w') as file:
        for j, cus in enumerate(seq):
            file.write('>seq' + str(j) + '\n')
            file.write(str(cus) + '\n')


def save_seqs_with_dis(seq,ori_dna_sequences,param,path):
    editnum = 0
    num = 0
    with open(path, 'w') as file:
        # file.write(f"{param}")
        for j, cus in enumerate(seq):
            dis = Levenshtein.distance(ori_dna_sequences[j],cus)
            editnum += dis
            if dis > 0:
                num += 1
            file.write(f">seq{j}  dis:{dis}\n{cus}\n")
    # print(f"deep dpdis:{num}")
    editerrorrate = editnum/len(ori_dna_sequences)/len(ori_dna_sequences[0])
    seqerrorrate = num/len(ori_dna_sequences)
    return editnum,editerrorrate,seqerrorrate

def save_seqs_and_confidence(seq,phred,ori_dna_sequences,param,path):
    editnum = 0
    num = 0
    with open(path, 'w') as file:
        # file.write(f"{param}")
        for j, cus in enumerate(seq):
            dis = Levenshtein.distance(ori_dna_sequences[j],cus)
            editnum += dis
            if dis > 0:
                num += 1
            file.write(f"@seq_confidence{j}  dis:{dis}\n{cus}\n+\n{phred[j]}\n")
    editerrorrate = editnum/len(ori_dna_sequences)/len(ori_dna_sequences[0])
    seqerrorrate = num/len(ori_dna_sequences)
    return editnum,editerrorrate,seqerrorrate


def read_alifilestest(path,num):
    seq_inf = []
    with open(path, "r") as file:
        lines = file.readlines()
    for i in range(2,num+2):
        templine = lines[i].strip('\n').split(' ')[3].replace('.','-')
        seq_inf.append(templine)
    # lii = lines[num+7].strip('\n').split(' ')[3]
    qua = lines[num+7].strip('\n').split(' ')[1].split('\t')[2]
    consus = lines[num+6].strip('\n').split(' ')[1].split('\t')[2]
    return seq_inf,qua,consus

def quality_scores_to_probabilities(quality_scores):
    probabilities = []

    for score in quality_scores:
        # 将ASCII质量分数转换为整数
        Q = ord(score) - 33
        # 计算错误概率P
        P = 10 ** (-Q / 10)
        # 转换为0到1的形式
        probabilities.append(1 - P)

    return probabilities

def getdelinsert(lines):
    mismatchdelinsertindexs = []
    delinsertindexs = []
    upnums = 0
    line = lines[0].strip('\n').split('\t')
    mismatch, delnum, insertnum = line[-3], line[-2], line[-1]
    line = lines[2].strip('\n')
    upnumsindex = []
    # print(line)
    for i in range(len(line)):
        # a = line[i]
        # print(a)
        if line[i] == '-' or line[i] == '*':
            mismatchdelinsertindexs.append(i)
            if line[i] == '-':
                delinsertindexs.append(i)
    line = lines[3].strip('\n')
    delindex = [i for i in range(len(line)) if line[i] == '-']
    insertindexs = [x for x in delinsertindexs if x not in delindex]
    mismatchindexs = [x for x in mismatchdelinsertindexs if x not in delinsertindexs]
    # insertindexs = delinsertindexs - delindex
    for i in range(len(delindex)):
        if len(delindex) > i + 1:
            if delindex[i + 1] == delindex[i] + 1:
                continue
        for misi in range(len(mismatchdelinsertindexs)):
            if mismatchdelinsertindexs[misi] > delindex[i]:
                mismatchdelinsertindexs[misi] -= 1
        for misi in range(len(insertindexs)):
            if insertindexs[misi] > delindex[i]:
                insertindexs[misi] -= 1
        for misi in range(len(mismatchindexs)):
            if mismatchindexs[misi] > delindex[i]:
                mismatchindexs[misi] -= 1
    return insertindexs,delindex

def getfirstindex(seq1,seq2):
    with open('files/errors1.fasta', 'w') as file:
        # with open('seqs.fasta', 'w') as file:
        for j, cus in enumerate([seq1, seq2]):
            file.write('>' + str(j) + '\n')
            file.write(str(cus) + '\n')
    # shell = '../bsalign-master/bsalign align seqs.fasta > ali.ali'
    shell = 'polls/Code/reconstruct/bsalign/bsalign align files/errors1.fasta > files/alierrors1.ali'
    result = subprocess.run(shell, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    with open('files/alierrors1.ali','r') as file:
        lines = file.readlines()
    if len(lines)>0:
        line = lines[0].strip('\n').split('\t')
        insertindexs,delindex = getdelinsert(lines)
        return int(line[3]),insertindexs,delindex
    return 0

