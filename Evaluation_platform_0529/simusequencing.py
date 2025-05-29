#coding=utf-8
import subprocess
from datetime import datetime
import random

import Levenshtein
# from dpconsensus import test_consus
from Evaluation_platform.dpconsensus.config import dpconsensus_path
from Evaluation_platform.dpconsensus import test_consus
# from Evaluation_platform.simu_and_consus import  cheatseqs, adddt4simu_advanced_fornona, \
#     badreads_simuseqs, removeindexs,adddt4simu_advanced,getmergeseqs,getposseqs
# from Evaluation_platform.utils import  readseqs, saveseqs, readtxt
# from dpconsensus.config import dpconsensus_path
from hedges_master.hedges_config import strandIDbytes, leftprimer
from dt4dds import syn_simu, seq_simu, get_allsequences, syn_simu_advanced, seq_simu_advanced, clusterseqs, \
    mergesimuseqs


from simu_and_consus import cheatseqs, adddt4simu_advanced_fornona, \
    badreads_simuseqs, adddt4simu_advanced, getposseqs
from utils import readseqs, saveseqs, readtxt

myfile = './testFiles/33.jpg'
resultpath = './testFiles/testResult2/'
# encodefile = resultpath+'33_derrick_encode.fasta'
encodefile = resultpath+'33_hedges_encode.fasta'
messplainpath = resultpath+'messplain.txt'
encodefile_witherrors = resultpath+'33_derrick_encode_errors.fasta'
myfile_decode = resultpath+'33_derrick_decode.jpg'
# file_input = os.getcwd() + "/testFiles/33.jpg"
# # file_input = os.getcwd() + "/testFiles/th.jpg"
# # file_input = os.getcwd() + "/testFiles/22.png"
# output_dir = os.getcwd() + "/testFiles/testResult/"

#coding:gbk
# E   IndexError: index 162 is out of bounds for axis 0 with size 158        dt4只能模拟158长度的序列

###   ①加入了RS纠错码
    # ②整合了一下框架，使它具有通用性，可以调用不同的编解码算法，对同一个文件进行编码，以下参数一样。dt4最长只能模拟158长度的序列
    # ③三代模拟工具模拟，DeSP的代码

# 问题：dt4合成错误后单独的文件？

min_gc = 0.4
max_gc = 0.6
max_homopolymer = 3
rule_num = 1
rs_num = 4
add_redundancy = True
#######################################
# if True, Make sure blast is installed
# add_primer = True
add_primer = False
primer_length = 20

def clusterseqs_byindexs(simulated_seqs,indexforsearch,index_length):
    lastseqs = [[] for _ in range(len(indexforsearch))]
    # lastseqsphareds = [[] for _ in range(len(indexforsearch))]
    for i in range(len(simulated_seqs)):
        mindis, minindex = 10, -1
        for j in range(len(indexforsearch)):
            dis = Levenshtein.distance(simulated_seqs[i][:index_length], indexforsearch[j])
            if dis < mindis:
                mindis, minindex = dis, j
            if dis <= 0:
                break
        if dis<=1:
            lastseqs[minindex].append(simulated_seqs[i])
            # lastseqsphareds[minindex].append(simulated_phreds[i])
    return lastseqs

def seqs_plus_indexs(dna_sequences):
    indexs = readtxt('./testFiles/address.wjr.txt')
    indexs = indexs[:len(dna_sequences)]
    # 给序列添加index
    newdna_sequences = []
    for i in range(len(dna_sequences)):
        newdna_sequences.append(indexs[i] + dna_sequences[i])
    return newdna_sequences,indexs

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

def test_nona_simu(newdna_sequences_path,index_length,copy_num,encodetype='',with_para=True):
    # return newdna_sequences_path + '_nano.simuseqs'
    # return [newdna_sequences_path + '.simuseqs',newdna_sequences_path+'.phreds']
    # return [newdna_sequences_path + '0.2_12.simuseqs',newdna_sequences_path+'0.2_12.phreds']
    # return [newdna_sequences_path + '_nano.simuseqs',newdna_sequences_path+'_nano.phreds']
    # return [newdna_sequences_path + '_nano0.6100491055769905.simuseqs',newdna_sequences_path+'_nano0.6100491055769905.phreds']
    # 三代测序测试
    with open(newdna_sequences_path, 'r') as file:
        lines = file.readlines()
    if with_para:
        firstline = lines[0]
        lines = lines[1:]
    # newdna_sequences = [lines[i].strip('\n') for i in range(1,len(lines),2)]
    if encodetype =='dnafountain':
        newdna_sequences = [lines[i].strip('\n') for i in range(len(lines))]
    else:
        newdna_sequences = [lines[i].strip('\n') for i in range(1, len(lines), 2)]
    if encodetype =='derrick':
        newdna_sequences,indexs = seqs_plus_indexs(newdna_sequences)
    # 引入合成错误
    # all_seq = []
    #
    start_time = datetime.now()
    all_seq = adddt4simu_advanced_fornona(newdna_sequences, index_length)
    print(f"引入合成错误时间：{str(datetime.now() - start_time)}")
    print(f"合成后总共有{len(all_seq)}个类")

    # 引入三代测序错误，并进行聚类
    start_time = datetime.now()
    print(f"使用{index_length}个碱基进行聚类")
    print(f"开始badreads模拟...")
    # neworiseqs, nona_consensus_seqs,lastseqs = badreads_simuseqs(newdna_sequences, all_seq, index_length,False)
    neworiseqs, nona_consensus_seqs,lastseqs = badreads_simuseqs(newdna_sequences, all_seq, index_length)
    # # #
    print(f"聚类时间：{str(datetime.now() - start_time)}")

    # if encodetype=='ali':
    if True:
        # 解码前纠错，得到共识序列,（使用我们的模型进行纠错）
        start_time = datetime.now()
        modelpath=dpconsensus_path + '/modelnosf6k.pth'
        # modelpath = '/home2/hm/models/derrick250211/modelnosf6k_250307.pth'
        # modelpath='/home1/hongmei/00work_files/0000/0isecondwork/Evaluation_platform/djangot2/polls/Code/reconstruct/dpconsensus/files/modelbadread.pth'
        # con_consensus_seqs, consensus_phreds, all_consus,allbs_quas = test_consus.getconsensus(newdna_sequences, nona_consensus_seqs,
        #                                                                     nona_consensus_seqs,copy_num,modelpath)
        con_consensus_seqs, consensus_phreds, all_consus,allbs_quas = test_consus.getconsensus(neworiseqs, nona_consensus_seqs,
                                                                            nona_consensus_seqs,copy_num,modelpath)

        consensus_seqs,consensus_p=getdatas(lastseqs,consensus_phreds,con_consensus_seqs)
        # consensus_seqs,consensus_p=getdatas(lastseqs,allbs_quas,all_consus)

        print(f"使用模型计算consensus时间：{str(datetime.now() - start_time)}")
        cheatseqs('deep', consensus_seqs, newdna_sequences)
    else:
        # all_seq_clustersr = nona_consensus_seqs
        # if len(newdna_sequences) != len(all_seq_clustersr):
        #     print(
        #         f"attention!!! there is a cluster no one seq:dna_sequences:{len(newdna_sequences)},simu:{len(all_seq_clustersr)}")
        # for ss in [10,10,7,7]:
        #     copy_num = ss
        all_seq_clustersr = lastseqs


        con_consensus_seqs = []
        con_consensusforcompare = ['']*len(all_seq_clustersr)
        minclusternum,maxclusternum,meanclusternum=10000,0,0
        for i in range(len(all_seq_clustersr)):
            length = len(all_seq_clustersr[i])
            minclusternum = min(length,minclusternum)
            maxclusternum = max(length,maxclusternum)
            meanclusternum += length
            if length >= copy_num:
                indexs = random.sample(range(length), copy_num)
            else:
                indexs = [i for i in range(length)]
            for j in indexs:
                con_consensusforcompare[i] = all_seq_clustersr[i][j]
                con_consensus_seqs.append(all_seq_clustersr[i][j])
        consensus_seqs = con_consensus_seqs
        meanclusternum /= len(all_seq_clustersr)
        print(f"minclusternum:{minclusternum},maxclusternum:{maxclusternum},meanclusternum:{meanclusternum},")
        cheatseqs('one by one', con_consensusforcompare, newdna_sequences)
    # cheatseqs('bsalign', all_consus, newdna_sequences)
    # 移除index
    if encodetype =='derrick':
        for i in range(len(consensus_seqs)):
            consensus_seqs[i]=consensus_seqs[i][index_length:]
            consensus_p[i]=consensus_p[i][index_length:]
    with open(newdna_sequences_path + '_nano.simuseqs', 'w') as file:
        if with_para:
            file.write(firstline)
        if encodetype == 'dnafountain':
            for i in range(len(consensus_seqs)):
                file.write(f"{consensus_seqs[i]}\n")
        else:
            for i in range(len(consensus_seqs)):
                file.write(f">seq{i}\n{consensus_seqs[i]}\n")
    with open(newdna_sequences_path+'_nano.phreds', 'w') as file:
        for i in range(len(consensus_p)):
            file.write(f">p{i}\n{consensus_p[i]}\n")
    # return newdna_sequences_path + '_nano.simuseqs'
    return [newdna_sequences_path + '_nano.simuseqs',newdna_sequences_path+'_nano.phreds']
    # return consensus_seqs, consensus_phreds, all_consus


def getseqsandindexs(newdna_sequences_path,with_para,encodetype,index_length=0):
    with open(newdna_sequences_path, 'r') as file:
        lines = file.readlines()
    firstline=''
    if with_para:
        firstline = lines[0]
        lines = lines[1:]
    indexs = []
    if encodetype =='dnafountain':
        newdna_sequences = [lines[i].strip('\n') for i in range(len(lines))]
    else:
        newdna_sequences = [lines[i].strip('\n') for i in range(1, len(lines), 2)]
    # if index_length==0:
    #     return newdna_sequences,firstline,indexs
    # if encodetype =='derrick':
    #     newdna_sequences,indexs = seqs_plus_indexs(newdna_sequences)
    #     return newdna_sequences,'',indexs
    for i in range(len(newdna_sequences)):
        # indexs.append(dna_sequences[i][:3+strandIDbytes*8])
        indexs.append(newdna_sequences[i][:index_length])
    return newdna_sequences,firstline,indexs


def adddt4simu_advanced(dna_sequences,indexs,deep,encodetype='',simulate=True):
    start_time = datetime.now()
    if simulate:
        # 二、引入合成错误 dt4dds
        pool = syn_simu_advanced(dna_sequences)

        # 三、引入二代Illumina测序错误
        # if len(indexs)==0:
        #     seq_simu_advanced(pool,len(dna_sequences),deep)
        seq_simu_advanced(pool,len(dna_sequences),100)
        # if encodetype=='':
        #     seq_simu_advanced(pool,len(dna_sequences),40)
        # else:
        #     seq_simu_advanced(pool,len(dna_sequences),deep)
        print(f"\n引入二代合成测序错误时间：{str(datetime.now() - start_time)}")

        # 四、处理模拟测序序列
        # 1.处理文件，得到测序序列
        start_time = datetime.now()
        simulated_seqsr1, simulated_seqsr2 = get_allsequences()
        # saveseqs('./testFiles/simulated_seqsr1.fasta',simulated_seqsr1)
        # saveseqs('./testFiles/simulated_seqsr2.fasta',simulated_seqsr2)

        # 2.联合正向和反向测序序列 如果序列长度长于158+20，则可以通过正向反向找到重叠序列，然后进行merge得到一条序列
        if len(dna_sequences[0])>178:
            simulated_seqs = mergesimuseqs(simulated_seqsr1,simulated_seqsr2)
        else:
            simulated_seqs = simulated_seqsr1+simulated_seqsr2
        saveseqs('./testFiles/simulated_seqsr1r2.fasta',simulated_seqs)
        print(f"原有{len(simulated_seqsr1)}条测序序列，现有{len(simulated_seqs)}条测序序列")
    else:
        simulated_seqs = []
        with open('./testFiles/simulated_seqsr1r2.fasta','r') as f:
        # with open('./testFiles/simulated_seqsr1r2_00_yyc_rs1.fasta','r') as f:
        # with open('./testFiles/simulated_seqsr1r2_00_yyc_rs2.fasta','r') as f:
            lines = f.readlines()
        for i in range(1,len(lines),2):
            simulated_seqs.append(lines[i].rstrip('\n'))


    # 如果index_length==0，则indexs=[]，则不聚类
    if len(indexs)==0:
        return simulated_seqs
    # 3. 使用minhash聚类
    # all_seq_clustersr = clusterseqs_byminhash(simulated_seqs)
    # all_seq_clustersr2 = clusterseqs_byminhash(simulated_seqsr2)
    all_seq_clustersr = clusterseqs_byindexs(simulated_seqs,indexs,len(indexs[0]))
    print(f"读取正反向测序序列，使用index聚类时间：{str(datetime.now() - start_time)}")

    return all_seq_clustersr


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

#二代测序测试
# 源文件为main_test.py
def test_ill_simu(newdna_sequences_path,index_length,copy_num,encodetype='noali',with_para=True):
    # return [newdna_sequences_path + '.simuseqs', newdna_sequences_path + '.phreds']
    # 得到原序列及indexs
    newdna_sequences,firstline,indexs = getseqsandindexs(newdna_sequences_path,with_para,encodetype,index_length)
    # 引入二代合成及测序错误，并进行聚类得到每一类
    simulate = False
    # simulate = True
    # 模拟
    # all_seq_clustersr_ori = adddt4simu_advanced(newdna_sequences,indexs,copy_num,encodetype)
    # saveseqs('./illseqs_aftercluster.fasta',all_seq_clustersr_ori)
    # 不模拟
    all_seq_clustersr_ori = adddt4simu_advanced(newdna_sequences,indexs,copy_num,encodetype,simulate)

    # if encodetype=='ali':
    if True:
        start_time = datetime.now()
        if simulate:
            all_seq_clustersr = []
            print(f"attention!!! there is a cluster no one seq:")
            for i in range(len(all_seq_clustersr_ori)):
                if type(all_seq_clustersr_ori[i]) == list and len(all_seq_clustersr_ori[i]) > 0:
                    all_seq_clustersr.append(all_seq_clustersr_ori[i])
                else:
                    print(f"{i}")
            all_seq_clustersr = [all_seq_clustersr_ori[i] for i in range(len(all_seq_clustersr_ori)) if len(all_seq_clustersr_ori[i])>0]
            print(f"all_seq_clustersr1:{len(all_seq_clustersr)}")
        else:
            all_seq_clustersr = all_seq_clustersr_ori
        # 解码前纠错，得到共识序列,（使用我们的模型进行纠错）
        con_consensus_seqs, consensus_phreds,all_consus,allbs_quas = test_consus.getconsensus(newdna_sequences,all_seq_clustersr, all_seq_clustersr,copy_num)

        #使用deeplearning的consensus
        consensus_seqs,consensus_p=getdatas(all_seq_clustersr_ori,consensus_phreds,con_consensus_seqs)
        #使用bsalign的consensus
        # consensus_seqs, consensus_p = getdatas(all_seq_clustersr_ori, allbs_quas, all_consus)

        print(f"使用模型计算consensus时间：{str(datetime.now()-start_time)}")
        cheatseqs('deep', consensus_seqs, newdna_sequences)
    else:
        # # 在聚类序列里面选择一条

        all_seq_clustersr = [all_seq_clustersr_ori[i] for i in range(len(all_seq_clustersr_ori)) if len(all_seq_clustersr_ori[i]) > 0 ]
        if len(all_seq_clustersr_ori)!=len(all_seq_clustersr):
            consensus_seqs = []

            j=0
            for i in range(len(all_seq_clustersr_ori)):
                if len(all_seq_clustersr_ori[i])==0:
                    consensus_seqs.append([])
                else:
                    consensus_seqs.append(all_seq_clustersr[j])
                    j+=1
        else:
            consensus_seqs = all_seq_clustersr
        # if len(newdna_sequences) !=len(all_seq_clustersr):
        #     print(f"attention!!! there is a cluster no one seq:dna_sequences:{len(newdna_sequences)},simu:{len(all_seq_clustersr)}")
        all_seq_clustersr = consensus_seqs
        con_consensus_seqs = []
        con_consensusforcompare = ['']*len(all_seq_clustersr)
        for i in range(len(all_seq_clustersr)):
            length = len(all_seq_clustersr[i])
            if length >= copy_num:
                indexs = random.sample(range(length), copy_num)
            else:
                indexs = [i for i in range(length)]
            # con_consensusforcompare[i] = all_seq_clustersr[i][0]
            for j in indexs:
                con_consensusforcompare[i] = all_seq_clustersr[i][j]
                con_consensus_seqs.append(all_seq_clustersr[i][j])
        # all_seq_clustersr = []
        # print(f"attention!!! there is a cluster no one seq:")
        # for i in range(len(all_seq_clustersr_ori)):
        #     if type(all_seq_clustersr_ori[i]) == list and len(all_seq_clustersr_ori[i]) > 0:
        #         all_seq_clustersr.append(all_seq_clustersr_ori[i])
        #     else:
        #         print(f"{i}")
        # all_seq_clustersr = [all_seq_clustersr_ori[i] for i in range(len(all_seq_clustersr_ori)) if
        #                      len(all_seq_clustersr_ori[i]) > 0]
        # print(f"all_seq_clustersr1:{len(all_seq_clustersr)}")
        consensus_seqs = con_consensus_seqs
        consensus_p = consensus_seqs
        cheatseqs('', con_consensusforcompare, newdna_sequences)
    if encodetype =='derrick':
        newdna_sequences=[seq[index_length:] for seq in consensus_seqs]
        consensus_seqs=newdna_sequences


    with open(newdna_sequences_path+'.simuseqs', 'w') as file:
        if with_para:
            file.write(firstline)
        if encodetype == 'dnafountain':
            for i in range(len(consensus_seqs)):
                file.write(f"{consensus_seqs[i]}\n")
        else:
            for i in range(len(consensus_seqs)):
                file.write(f">seq{i}\n{consensus_seqs[i]}\n")
    with open(newdna_sequences_path+'.phreds', 'w') as file:
        for i in range(len(consensus_p)):
            file.write(f">p{i}\n{consensus_p[i]}\n")
    # return newdna_sequences_path+'.simuseqs' ,
    return [newdna_sequences_path+'.simuseqs',newdna_sequences_path+'.phreds']

