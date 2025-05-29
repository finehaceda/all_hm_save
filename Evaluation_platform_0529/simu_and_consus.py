import os
import subprocess
import threading
from datetime import datetime

from datasketch import MinHash, MinHashLSH
from joblib import delayed, Parallel

# from Evaluation_platform.dpconsensus.config import dpconsensus_path
# from Evaluation_platform.dt4dds import syn_simu, seq_simu, get_allsequences, syn_simu_advanced, seq_simu_advanced, \
#     clusterseqs, syn_simu_myadvanced
# from Evaluation_platform.dpconsensus import test_consus
# from Evaluation_platform.utils import pri0pre20, sequence_length, pri1pre20, primers_2, primer_syn, resultpath, \
#     readsynseqs, readtestfastq
from dt4dds import syn_simu, seq_simu, get_allsequences, syn_simu_advanced, seq_simu_advanced, clusterseqs, \
    mergesimuseqs
from utils import pri0pre20, sequence_length, pri1pre20, primers_2, resultpath, readsynseqs, readtestfastq, \
    saveseqs


def getphred_quality(qualityScore):
    return_phred_qualitys = []
    # sss = seq[110:120]
    for i in range(len(qualityScore)):
        phred_qualitys = []
        for qua in qualityScore[i]:
            phred_quality = ord(qua) - 33  # '@'的ASCII码是64，FASTQ使用的是Phred+33编码
            phred_qualitys.append(phred_quality)
        return_phred_qualitys.append(phred_qualitys)
    return return_phred_qualitys

def clusterseqs(simulated_seqs,indexforsearch,index_length,sequence_length):
    lastseqs = [[] for _ in range(len(indexforsearch))]
    nousedsimuseqs = 0
    for i in range(len(simulated_seqs)):
        # if sequence_length*0.8<len(simulated_seqs[i])<sequence_length*1.3:
        # if sequence_length*0.98<len(simulated_seqs[i])<sequence_length*1.05:
        # if sequence_length*0.9<len(simulated_seqs[i])<sequence_length*1.2:
        if sequence_length*0.85<len(simulated_seqs[i])<sequence_length*1.3:
            mindis, minindex = 10, -1
            for j in range(len(indexforsearch)):
                dis = Levenshtein.distance(simulated_seqs[i][:index_length], indexforsearch[j])
                if dis < mindis:
                    mindis, minindex = dis, j
                if dis <= 0:
                    break
            if mindis == 0:
                lastseqs[minindex].insert(0,simulated_seqs[i])
                continue

            rmindis,rminindex = 10,-1
            reversedseq_d = simulated_seqs[i][::-1]
            reversedseq=''
            dict = {'A': 'T', 'C': 'G', 'T': 'A', 'G': 'C'}
            for j in range(len(reversedseq_d)):
                reversedseq += dict[reversedseq_d[j]]
            for j in range(len(indexforsearch)):
                dis = Levenshtein.distance(reversedseq[:index_length], indexforsearch[j])
                if dis < rmindis:
                    rmindis, rminindex = dis, j
                if dis <= 0:
                    break
            if rmindis == 0:
                # lastseqs[rminindex].append(reversedseq)
                lastseqs[rminindex].insert(0,reversedseq)
                continue
            if mindis>=3 and rmindis>=3:
                nousedsimuseqs+=1
                continue
            if mindis < rmindis:
                lastseqs[minindex].append(simulated_seqs[i])
            else:
                lastseqs[rminindex].append(reversedseq)
    print(f"总共有模拟序列{len(simulated_seqs)}条，未使用的序列有:{nousedsimuseqs}条")
    return lastseqs


# def clusterseqs_byindexs(simulated_seqs,simulated_phreds,indexforsearch,index_length):
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


def clusterseqs_byminhash(dna_sequences):
    kmer_len = 9
    threshold = 0.30
    # threshold = 0.75
    num_perm = 128

    def compute_minhash(dna_seq, kmer_len, num_perm):
        minhash = MinHash(num_perm=num_perm)
        for i in range(0, len(dna_seq) - kmer_len + 1):
            kmer = dna_seq[i: i + kmer_len]
            minhash.update(kmer.encode('utf-8'))
        return minhash
    minhashes = Parallel(n_jobs=-2)(delayed(compute_minhash)(dna_sequences[i],kmer_len,num_perm) for i in range(len(dna_sequences)))
    lsh = MinHashLSH(threshold, num_perm=num_perm)
    for i, minhash in enumerate(minhashes):
        lsh.insert(i, minhash)
    clusters = []
    visited = set()
    for i, minhash in enumerate(minhashes):
        if i not in visited:
            cluster = set(lsh.query(minhash))
            clusters.append(cluster)
            visited.update(cluster)
    all_phreds_clusters = {}
    all_seq_clusters = {}
    seql = 0
    for seq_labels in clusters:
        all_seq_clusters[seql] = []
        all_phreds_clusters[seql] = []
        for seq_label in seq_labels:
            all_seq_clusters[seql].append(dna_sequences[seq_label])
        seql += 1
    return list(all_seq_clusters.values())





def getposseqs(consensus_seqs, indexforsearch):
    indexlen = len(indexforsearch[0])
    lastseqs = ["" for _ in range(len(indexforsearch))]
    for i in range(len(consensus_seqs)):
        mindis, minindex = 10, -1
        for j in range(len(indexforsearch)):
            # dis = Levenshtein.distance(consensus_seqs[i][:3+strandIDbytes*8], indexforsearch[j])
            dis = Levenshtein.distance(consensus_seqs[i][:indexlen], indexforsearch[j])
            if dis < mindis:
                mindis, minindex = dis, j
            if dis == 0:
                break
        # lastseqs[minindex] = consensus_seqs[i][12:]
        if lastseqs[minindex] == '':
            lastseqs[minindex] = consensus_seqs[i]
        else:
            print(f"i:{i}")
            print(f"有序列被覆盖！！\nindexforsearch:{indexforsearch[minindex]}\noriconsensus:{lastseqs[minindex]}\nconsensus_seqs:{consensus_seqs[i]}")
            lastseqs[minindex] = consensus_seqs[i]
    return lastseqs



def getdis(lastseqs, dna_sequences):
    alldis = 0
    alldisarr = []
    lossnum=0
    for i in range(len(lastseqs)):
        disgt100 = 0
        if len(lastseqs[i])>0:
            dis = Levenshtein.distance(lastseqs[i], dna_sequences[i])
            if dis<100:
                alldis += dis
                # if dis > 10:
                #     print(f"ori:{dna_sequences[i]} no.{i}\npre:{lastseqs[i]}")
            else:
                disgt100+=1
                # print(f"dis > 100 ,this seq error")
            alldisarr.append(dis)
        else:
            alldisarr.append('序列丢失')
            lossnum+=1
    print(f"lossnum:{lossnum},disgt100:{disgt100}")
    print(f"{alldisarr}")
    # if alldis>1000:
    #     print(f"lastseqs:\n{lastseqs}")
    #     print(f"dna_sequences:\n{dna_sequences}")
    # print(f"alldisarr:\n{alldisarr}")
    return alldis,alldisarr

def adddt4simu(dna_sequences,index_length):
    indexforsearch = []
    for i in range(len(dna_sequences)):
        indexforsearch.append(dna_sequences[i][:index_length])
    # indexforsearch = set(indexforsearchlist)
    # 二、引入合成错误 dt4dds
    pool = syn_simu(dna_sequences)
    seqs = pool._seqdict
    # 三、引入二代Illumina测序错误
    seq_simu(pool)
    # 四、聚类 1.处理文件，得到测序序列 2.联合正向和反向测序序列 3.使用minhash聚类
    simulated_seqs, simulated_phreds = get_allsequences()
    # all_seq_clusters, all_phreds_clusters = clusterseqs(simulated_seqs, simulated_phreds)
    all_seq_clusters, all_phreds_clusters = clusterseqs_byindexs(simulated_seqs, simulated_phreds, indexforsearch,
                                                                 index_length)
    return all_seq_clusters, all_phreds_clusters
    # # 五、解码前纠错，得到共识序列,（使用我们的模型进行纠错）
    # selectdna_sequences,consensus_seqs, consensus_phreds, bsalign_consus_ori = test_consus.getconsensus(
    #     all_seq_clusters, all_phreds_clusters)
    # # consensus_seqsori = [seq[0] for seq in all_seq_clusters if len(seq)>0]
    #
    # # 六、评测纠错结果 1.通过前10位作为index来找序列 2.计算 edit_distance
    # # 测试预测序列的结果 dp
    # # 归位
    # lastseqs = getposseqs(consensus_seqs, dna_sequences)
    # alldis,alldisarr = getdis(lastseqs, dna_sequences)
    # print(f"dpdis:{alldis}")
    #
    # with open(dpconsensus_path+'/oriandpreseqswith_.fasta','w') as file:
    #     for i in range(len(dna_sequences)):
    #         file.write(f">ori{i}\n{dna_sequences[i]}\n>preseq{i} dis:{alldisarr[i]}\n{lastseqs[i]}\nselectdna_sequences{i}:\n{selectdna_sequences[i]}\n")


    # # # # 测试预测序列的结果 dpori
    # lastseqs = getposseqs(consensus_seqsori, dna_sequences)
    # alldis,alldisarr  = getdis(lastseqs, dna_sequences)
    # print(f"dpdisori:{alldis}")
    # # 测试预测序列的结果 bsalign
    # lastseqs = getposseqs(bsalign_consus_ori, dna_sequences)
    # alldis,alldisarr  = getdis(lastseqs, dna_sequences)
    # print(f"bsaligndis:{alldis}")

    return consensus_seqs, consensus_phreds

def getsynseqs(seqpool,filepath):
    with open(filepath, 'w') as f:
        for index, (sequence, count) in enumerate(seqpool):
            string = '\n'.join([
                f"@Seq{str(index).zfill(9)}:{str(index + 1).zfill(9)}",
                str(sequence),
                "+",
                "F" * len(sequence),
                ''  # final newline
            ])
            # for i in range(count):
            #     string = '\n'.join([
            #         f"@Seq{str(index).zfill(9)}:{str(i + 1).zfill(9)}",
            #         str(sequence),
            #         "+",
            #         "F" * len(sequence),
            #         ''  # final newline
            #     ])
            f.writelines(string)

def readsynfasta(file_path):
    allinfos,allseqs = [],[]
    with open(file_path, 'r') as f:
        lines = f.readlines()
    for i in range(0,len(lines),2):
        allinfos.append(lines[i])
        allseqs.append(lines[i+1].strip('\n'))
    return allinfos,allseqs


def getsynseqs_forsequencing(pool,n_reads,path):
    cluster_seqpool = pool
    # sample oligos to get expected number of reads
    if cluster_seqpool.n_oligos >= n_reads:
        cluster_seqpool = cluster_seqpool.sample_by_counts(n_reads, remove_sampled_oligos=False)
    elif cluster_seqpool.n_oligos == 0:
        print("Unable to sequence, no sequence-able oligos found.")
    else:
        print(
            f"Only {cluster_seqpool.n_oligos} oligos available for total of {n_reads} reads. Continuing.")

    # 这里使用cluster_seqpool.to_list()会保存所有序列，且如果一条序列count为n，则保留n条，cluster_seqpool.save_as_fasta则会把序列和丰富度保存到文件中
    # cluster_seqpool.save_as_fasta("./testFiles/testResult/dt4dds_syn.fasta")
    cluster_seqpool.save_as_fasta(path)
    allinfos, allseqs_r1 = readsynfasta(path)
    print(f"合成后总共有{len(allseqs_r1)}条不同的序列")

    allseqs_r1 = cluster_seqpool.to_list()
    # true_allseqs_r1 = handle_synseqs_withprimer(allseqs_r1,sequence_length)
    # true_allseqs_r1 = allseqs_r1
    print(f"合成后总共有{len(allseqs_r1)}条用于测序的序列")
    # print(f"true_allseqs_r1:{len(true_allseqs_r1)}")
    # with open(path + '.pre', 'w') as f:
    #     # f.write(param)
    #     for i in range(len(allseqs_r1)):
    #         f.write(f">seq{i}\n{allseqs_r1[i]}\n")
    # cluster_seqpool.save_as_fasta(resultpath + "dt4dds_syn.fasta")
    # infos = {'path': path + '.pre'}
    return allseqs_r1

def adddt4simu_advanced_fornona(dna_sequences,index_length):
    # indexforsearch = []
    # for i in range(len(dna_sequences)):
    #     indexforsearch.append(dna_sequences[i][:index_length])
    # indexforsearch = set(indexforsearchlist)
    # 二、引入合成错误 dt4dds
    pool = syn_simu_advanced(dna_sequences)
    # 三、引入二代Illumina测序错误，目的是为了得到合成序列
    # seq_simu_advanced(pool,len(dna_sequences),100)
    # 四、聚类 1.处理文件，得到测序序列 2.联合正向和反向测序序列 3.使用minhash聚类
    # allseqs_withprimer = readsynseqs(resultpath+'dt4dds_syn.fasta')
    # allseqs_noprimer,filepath=get_badreads_phreds_test(allseqs_withprimer,resultpath)


    # 20250215改
    # 这里使用如果通过readsynseqs(resultpath+'dt4dds_syn.fasta')来读合成后的序列，那么仅读取单条不同的序列，而不是按照丰度读取。（方便测试）*100/*200
    # 真正模拟测序情况 ，可以省略readsynseqs(resultpath+'dt4dds_syn.fasta')，通过直接使用合成后的序列来测序，但是这样需要控制合成后采样的序列倍数。*10/*20/*30
    # TODO 如果badread三代测序，如果合成后的文件测序序列数量过多，会很慢。可以把大文件分为多个小文件，开多个线程测序
    # 已完成，根据序列数量开多个线程
    allseqs_withprimer = getsynseqs_forsequencing(pool,len(dna_sequences) * 20,resultpath+'dt4dds_syn.fasta')
    # allseqs_withprimer = readsynseqs(resultpath+'dt4dds_syn.fasta')
    allseqs_noprimer,filepath=get_badreads_phreds_test(allseqs_withprimer,resultpath)



    return allseqs_noprimer
    # # 五、解码前纠错，得到共识序列,（使用我们的模型进行纠错）
    # selectdna_sequences,consensus_seqs, consensus_phreds, bsalign_consus_ori = test_consus.getconsensus(
    #     all_seq_clusters, all_phreds_clusters)
    # # consensus_seqsori = [seq[0] for seq in all_seq_clusters if len(seq)>0]
    #
    # # 六、评测纠错结果 1.通过前10位作为index来找序列 2.计算 edit_distance
    # # 测试预测序列的结果 dp
    # # 归位
    # lastseqs = getposseqs(consensus_seqs, dna_sequences)
    # alldis,alldisarr = getdis(lastseqs, dna_sequences)
    # print(f"dpdis:{alldis}")
    #
    # with open(dpconsensus_path+'/oriandpreseqswith_.fasta','w') as file:
    #     for i in range(len(dna_sequences)):
    #         file.write(f">ori{i}\n{dna_sequences[i]}\n>preseq{i} dis:{alldisarr[i]}\n{lastseqs[i]}\nselectdna_sequences{i}:\n{selectdna_sequences[i]}\n")


    # # # # 测试预测序列的结果 dpori
    # lastseqs = getposseqs(consensus_seqsori, dna_sequences)
    # alldis,alldisarr  = getdis(lastseqs, dna_sequences)
    # print(f"dpdisori:{alldis}")
    # # 测试预测序列的结果 bsalign
    # lastseqs = getposseqs(bsalign_consus_ori, dna_sequences)
    # alldis,alldisarr  = getdis(lastseqs, dna_sequences)
    # print(f"bsaligndis:{alldis}")

    return consensus_seqs, consensus_phreds

def adddt4simu_advanced(dna_sequences,indexs):
    # indexforsearch = []
    # for i in range(len(dna_sequences)):
    #     indexforsearch.append(dna_sequences[i][:index_length])
    start_time = datetime.now()
    # 二、引入合成错误 dt4dds
    pool = syn_simu_advanced(dna_sequences)
    # 三、引入二代Illumina测序错误
    seq_simu_advanced(pool,len(dna_sequences))
    print(f"\n引入二代合成测序错误时间：{str(datetime.now() - start_time)}")
    # 四、聚类 1.处理文件，得到测序序列 2.联合正向和反向测序序列 3.使用minhash聚类
    start_time = datetime.now()
    simulated_seqsr1, simulated_seqsr2 = get_allsequences()
    # saveseqs('./testFiles/simulated_seqsr1.fasta',simulated_seqsr1)
    # saveseqs('./testFiles/simulated_seqsr2.fasta',simulated_seqsr2)
    #如果序列长度长于158+20，则可以正向反向找到重叠序列
    if len(dna_sequences[0])>178:
        simulated_seqs = mergesimuseqs(simulated_seqsr1,simulated_seqsr2)
    else:
        simulated_seqs=simulated_seqsr1+simulated_seqsr2
    saveseqs('./testFiles/simulated_seqsr1r2.fasta',simulated_seqs)


    print(f"原有{len(simulated_seqsr1)}条测序序列，现有{len(simulated_seqs)}条测序序列")
    # all_seq_clustersr = clusterseqs_byminhash(simulated_seqs)
    # all_seq_clustersr2 = clusterseqs_byminhash(simulated_seqsr2)
    all_seq_clustersr = clusterseqs_byindexs(simulated_seqs,indexs,len(indexs[0]))
    print(f"读取正反向测序序列，使用index聚类时间：{str(datetime.now() - start_time)}")

    return all_seq_clustersr

def getmergeseqs(consensus_seqs1, consensus_seqs2):
    mymergeseqs = []
    for i in range(len(consensus_seqs2)):
        frontnum = 30
        flag = False
        while(not flag and frontnum >= 15):
            if frontnum < 25:
                print(frontnum)
            frontnum -= 5
            head = consensus_seqs2[i][:frontnum]
            for j in range(len(consensus_seqs1)):
                index = consensus_seqs1[j].find(head)
                if index >= 0:
                    flag = True
                    mymergeseqs.append(consensus_seqs1[j][:index]+consensus_seqs2[i])
                    break
        if not flag:
            print(consensus_seqs2[i])
        # if frontnum <=5:
        #     print('!!!!!!!!!!!!not find!!!!!!!!!!!!!')
    return mymergeseqs

import Levenshtein

# print(dnas)
# examine the output dnas
# from Analysis.Analysis import inspect_distribution, examine_strand
#
# inspect_distribution(out_dnas, show=True)  # oligo number and error number distribution of the entire sequencing results
# examine_strand(out_dnas, index=index)

def cheatseqs(seq_way,consensus_seqs,dna_sequences,indexs = None):
    # 六、评测纠错结果 1.通过前10位作为index来找序列 2.计算 edit_distance
    # 测试预测序列的结果 dp
    # 归位
    if len(consensus_seqs)!=len(dna_sequences):
        print('序列数量不对！！！')
    # if indexs:
    #     consensus_seqs = getposseqs(consensus_seqs, indexs)
    alldis, alldisarr = getdis(consensus_seqs, dna_sequences)
    print(f"{seq_way} dpdis:{alldis}")


def get_badreads_phreds1(allseqs,allphreds):
    true_allseqs_r1,true_allphreds_r1 = [],[]
    true_allseqs_r2,true_allphreds_r2 = [],[]
    for i in range(len(allseqs)):
        index1 = allseqs[i].find(pri0pre20)
        index2 = allseqs[i].find(pri1pre20)
        if index1>=0:
            true_allseqs_r1.append(allseqs[i][:index1])
            true_allphreds_r1.append(allphreds[i][:index1])
        elif index2>=0:
            true_allseqs_r2.append(allseqs[i][:index2])
            true_allphreds_r2.append(allseqs[i][:index2])
    dict = {'A':'T','C':'G','T':'A','G':'C'}
    for i in range(len(true_allseqs_r2)):
        newseq = ""
        for j in range(len(true_allseqs_r2[i])-1,-1,-1):
            newseq += dict[true_allseqs_r2[i][j]]
        true_allseqs_r1.append(newseq)
        true_allphreds_r1.append(true_allphreds_r2[i][::-1])
    return true_allseqs_r1,true_allphreds_r1

def get_badreads_phreds_test(allseqs,filepath):
    true_allseqs_r1 = []
    for i in range(len(allseqs)):
        index0 = allseqs[i].find(primers_2[0][-20:])
        index1 = allseqs[i].find(pri0pre20)
        if index1>=0:
        #     seq = allseqs[i][index0+len(primers_2[0][-20:]):index1]
        #     if sequence_length -1 <=len(seq) <= sequence_length +1:
            true_allseqs_r1.append(allseqs[i][index0+len(primers_2[0][-20:]):index1])
                # true_allseqs_r1.append(seq)
    with open(filepath+'dt4dds_syn_manage.fasta','w') as f:
        for i in range(len(true_allseqs_r1)):
            f.write(f">#seqs{i}\n{true_allseqs_r1[i]}\n")
    return true_allseqs_r1,filepath+'dt4dds_syn_manage.fasta'
    # return true_allseqs_r1

def get_synfor_badreads(allseqs,filepath):
    true_allseqs_r1 = []
    for i in range(len(allseqs)):
        index0 = allseqs[i].find(primers_2[0])
        index1 = allseqs[i].find(pri0pre20)
        if index1>=0:
            true_allseqs_r1.append(allseqs[i][index0+len(primers_2[0]):index1])
    with open(filepath,'w') as f:
        for i in range(len(true_allseqs_r1)):
            f.write(f">#seqs{i}\n{true_allseqs_r1[i]}\n")
    return true_allseqs_r1,filepath+'dt4dds_syn_manage.fasta'

def get_badreads_for_cluster(allseqs,filepath):
    true_allseqs_r1 = []
    for i in range(len(allseqs)):
        index0 = allseqs[i].find('GTATTGCT')
        index1 = allseqs[i].find('GCAATACG')
        if index0>=0 and index1 >= sequence_length and (index1-index0) < sequence_length*1.3:
            true_allseqs_r1.append(allseqs[i][index0+len('CGTATTGCT'):index1])
        elif index0>=0 :
            start = index0+len('CGTATTGCT')
            true_allseqs_r1.append(allseqs[i][start:start+sequence_length+2])
        elif index1 >= sequence_length:
            true_allseqs_r1.append(allseqs[i][index1-sequence_length-2:index1])
    with open(filepath+'0604test.fasta','w') as f:
        for i in range(len(true_allseqs_r1)):
            f.write(f">#seqs{i}\n{true_allseqs_r1[i]}\n")
    return true_allseqs_r1,filepath+'0604test.fasta'
    # return true_allseqs_r1


def checkclusterout(neworiseqs, newlastseqs):
    n = len(neworiseqs)
    errorsnums,rightnums, rightdis = 0 ,0, 0
    errorsarrs = []
    for i in range(n):
        oriseq = neworiseqs[i]
        curerrnums = 0
        for j in range(len(newlastseqs[i])):
            dis = Levenshtein.distance(oriseq,newlastseqs[i][j])
            if dis > sequence_length * 0.2:
                errorsnums += 1
                curerrnums += 1
            else:
                rightdis+=dis
        rightnums += len(newlastseqs[i]) - curerrnums
        errorsarrs.append(curerrnums)
    errorslay = len([i for i in range(len(errorsarrs)) if errorsarrs[i]>0])
    # print(f"聚类错误情况：{errorsarrs},\n聚类错误数量：{errorsnums},聚类正确数量：{rightnums},聚类错误的类有：{errorslay}个,\n")
    print(f"聚类错误数量：{errorsnums},聚类正确数量：{rightnums},聚类错误的类有：{errorslay}个,\n")
          # f"平均聚类错误数量：{errorsnums/n},聚类错误的类中平均错误数量：{errorsnums/errorslay},模拟序列的错误率为：{rightdis/rightnums/sequence_length}")
          # f"平均聚类错误数量：{errorsnums/n},聚类错误的类中平均错误数量：{errorsnums/errorslay}")

def badreads_simuseqs(dna_sequences,synseqs,index_length,badsimu=True):
    start_time = datetime.now()
    indexforsearch = []
    for i in range(len(dna_sequences)):
        indexforsearch.append(dna_sequences[i][:index_length])
    # indexforsearch = set(indexforsearchlist)
    # # 二、引入合成错误 dt4dds
    # pool = syn_simu_advanced(dna_sequences)
    # getsynseqs(pool,"./testFiles/testResult/dt4dds_syn.fastq")
    # pool.save_as_fasta("./testFiles/testResult/dt4dds_syn.fasta")
    # pool = syn_simu(dna_sequences)
    # seqs = pool._seqdict.keys()
    # 三、引入二代Illumina测序错误
    # seq_simu_advanced(pool,len(dna_sequences))
    # del pool
    # allseqs = readsynseqs(resultpath+'dt4dds_syn.fasta')
    allseqs = synseqs
    # get_badreads_phreds(allseqs,resultpath)
    if badsimu:
        # 是否开多线程
        # threadon = False
        threadon = True
        if threadon:

            tempfiles ='/home2/hm/badreads_0isec/'
            # simulate_file_path = '/home2/hm/badreads_0isec/badreads_simutest.fastq'
            onethreadreads = 1000
            num = 0
            for i in range(0,len(allseqs),onethreadreads):
                file_index = i // onethreadreads + 1
                chunk = allseqs[i:i + onethreadreads]
                with open(tempfiles+f'dt4dds_syn_manage_{file_index}.fasta','w') as f:
                    for j in range(len(chunk)):
                        f.write(f">#seqs{i+j}\n{chunk[j]}\n")
                num += 1
            print(f'共有{num}个文件待模拟，每个文件{onethreadreads}条序列')
            simulate_file_path = badread_simulator_parallel(num, 20, tempfiles)
        else:
            simulate_file_path = resultpath + 'badreads_simutest.fastq'
            with open(resultpath+'dt4dds_syn_manage.fasta','w') as f:
                for i in range(len(allseqs)):
                    f.write(f">#seqs{i}\n{allseqs[i]}\n")
            # simulated_seqs,simupath = get_synfor_badreads(allseqs,resultpath+'dt4dds_syn_manage.fasta')
            # import edlib
            # shell = 'python3 Badread-main/badread-runner.py simulate '

            #原始
            shell = '/home1/hongmei/anaconda3/bin/python Badreadmain/badread-runner.py simulate --reference '\
                    +resultpath+'dt4dds_syn_manage.fasta --quantity 20x --start_adapter_seq "" --end_adapter_seq ""' \
                                '> '+simulate_file_path
            # 0.0102
            # shell = ('/home1/hongmei/anaconda3/bin/python Badreadmain/badread-runner.py simulate --reference '
            #          + resultpath + 'dt4dds_syn_manage.fasta --quantity ' + str(
            #     20) + 'x --start_adapter_seq "" --end_adapter_seq "" --glitches 0,0,0 --junk_reads 0 --random_reads 0 --chimeras 0 --identity 99,99.2,1'+
            #          ' > '+resultpath+'badreads_simutest.fastq')
            # 0.0200
            # shell = ('/home1/hongmei/anaconda3/bin/python Badreadmain/badread-runner.py simulate --reference '
            #          + resultpath + 'dt4dds_syn_manage.fasta --quantity ' + str(
            #     20) + 'x --start_adapter_seq "" --end_adapter_seq "" --glitches 0,0,0 --junk_reads 0 --random_reads 0 --chimeras 0 --identity 98,99,1'+
            #          ' > '+resultpath+'badreads_simutest.fastq')
            # 0.0304
            # shell = ('/home1/hongmei/anaconda3/bin/python Badreadmain/badread-runner.py simulate --reference '
            #          + resultpath + 'dt4dds_syn_manage.fasta --quantity ' + str(
            #     20) + 'x --start_adapter_seq "" --end_adapter_seq "" --glitches 0,0,0 --junk_reads 0 --random_reads 0 --chimeras 0 --identity 97,99,1'+
            #          ' > '+resultpath+'badreads_simutest.fastq')
            # 0.0402
            # shell = ('/home1/hongmei/anaconda3/bin/python Badreadmain/badread-runner.py simulate --reference '
            #          + resultpath + 'dt4dds_syn_manage.fasta --quantity ' + str(
            #     20) + 'x --start_adapter_seq "" --end_adapter_seq "" --glitches 0,0,0 --junk_reads 0 --random_reads 0 --chimeras 0 --identity 96,99,1.2'+
            #          ' > '+resultpath+'badreads_simutest.fastq')
            #  0.0472
            # shell = ('/home1/hongmei/anaconda3/bin/python Badreadmain/badread-runner.py simulate --reference '
            #          + resultpath + 'dt4dds_syn_manage.fasta --quantity ' + str(
            #     20) + 'x --start_adapter_seq "" --end_adapter_seq "" --glitches 0,0,0 --junk_reads 0 --random_reads 0 --chimeras 0 --identity 95,99,2'+
            #          ' > '+resultpath+'badreads_simutest.fastq')
            # 0.0496
            # shell = ('/home1/hongmei/anaconda3/bin/python Badreadmain/badread-runner.py simulate --reference '
            #          + resultpath + 'dt4dds_syn_manage.fasta --quantity ' + str(
            #     20) + 'x --start_adapter_seq "" --end_adapter_seq "" --glitches 0,0,0 --junk_reads 0 --random_reads 0 --chimeras 0 --identity 95,99,1.5'+
            #          ' > '+resultpath+'badreads_simutest.fastq')
            # 0.0600
            # shell = ('/home1/hongmei/anaconda3/bin/python Badreadmain/badread-runner.py simulate --reference '
            #          + resultpath + 'dt4dds_syn_manage.fasta --quantity ' + str(
            #     20) + 'x --start_adapter_seq "" --end_adapter_seq "" --glitches 0,0,0 --junk_reads 0 --random_reads 0 --chimeras 0 --identity 94,99,1.5'+
            #          ' > '+resultpath+'badreads_simutest.fastq')k
            print(f"shell:{shell}")
            result = subprocess.run(shell, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # print(f"badread result:{result.stderr}")
        print(f"引入测序错误时间：{str(datetime.now() - start_time)}")
        # Evaluation_platform.Badreadmain.badread.__main__.main(subparser_name='simulate',)

        # 四、聚类 1.处理文件，得到测序序列 2.联合正向和反向测序序列 3.使用index聚类
        start_time = datetime.now()
        #1.读取文件
        # allseqs, allphreds = readtestfastq(resultpath + 'badreads_simutest.fastq')
        allseqs, allphreds = readtestfastq(simulate_file_path)
        #2.根据adapter去除前尾部分
        # true_allseqs_r1, seqsforsimu = get_badreads_for_cluster(allseqs, resultpath)



        # 3.使用index聚类
        lastseqs = clusterseqs(allseqs, indexforsearch, index_length,len(dna_sequences[0]))
        saveseqs('./nanoseqs_aftercluster.fasta',lastseqs)
        # for i in range(len(dna_sequences)):
        #     if len(lastseqs[i]) == 0:
        #         print(f"类别{i}序列数量为0")
        neworiseqs = [dna_sequences[i] for i in range(len(dna_sequences)) if len(lastseqs[i])>0]

    else:
        lastseqs = readclusterfile('./nanoseqs_aftercluster.fasta')
        # lastseqs = readclusterfile('./nanoseqs_aftercluster_00_0.8_crc16_152nt_rs1.fasta')
        # lastseqs = readclusterfile('./nanoseqs_aftercluster_00_0.25_crc16_152nt_rs1.fasta')
        # lastseqs = readclusterfile('./nanoseqs_aftercluster_00_0.4_crc16_152nt_rs1.fasta')
        # lastseqs = readclusterfile('./nanoseqs_aftercluster_00_0.3.fasta')
        # lastseqs = readclusterfile('./nanoseqs_aftercluster_00_0.5.fasta')
        # lastseqs = readclusterfile('./nanoseqs_aftercluster_dnafountain.fasta')
        # lastseqs = readclusterfile('./nanoseqs_aftercluster_copycompare.fasta')
        neworiseqs = [dna_sequences[i] for i in range(len(dna_sequences)) if len(lastseqs[i]) > 0]
    print(f"attention!!! there is a cluster no one seq:\n")
    newlastseqs = []
    for i in range(len(lastseqs)):
        if len(lastseqs[i]) > 0:
            newlastseqs.append(lastseqs[i])
        else:
            print(f"{i} ")
            # print(f"attention!!! there is a cluster no one seq:\n{dna_sequences[i]}\n")

    # newlastseqs = [lastseqs[i] for i in range(len(lastseqs)) if len(lastseqs[i])>0]
    print(f"聚类后共有：{len(newlastseqs)}个类")
    print(f"聚类时间：{str(datetime.now() - start_time)}")
    # 4.查看聚类效果
    start_time = datetime.now()
    checkclusterout(neworiseqs,newlastseqs)
    print(f"查看聚类效果时间：{str(datetime.now() - start_time)}")
    return neworiseqs,newlastseqs,lastseqs

def badread_simulator_parallel(processes, depth,tempfiles='/home2/hm/badreads_0isec/'):
    threads = []
    one_thread_copy_number = depth
    print(f'one_thread_copy_number:{one_thread_copy_number}')
    # Start the subprocesses
    for p in range(1, processes+1):
        thread = threading.Thread(target=run_badread_simulate, args=(p, one_thread_copy_number,tempfiles))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    # Combine the output files and compress
    print(f'start badreads_simutest merge')
    with open(f'{tempfiles}badreads_simutest.fastq', 'wb') as outfile:
        subprocess.run(f'cat {tempfiles}temp_*.fastq', shell=True, stdout=outfile)

    # Clean up temporary files
    for p in range(1, processes + 1):
        os.remove(f"{tempfiles}temp_{p}.fastq")
        os.remove(f"{tempfiles}dt4dds_syn_manage_{p}.fasta")
    return f'{tempfiles}badreads_simutest.fastq'

def run_badread_simulate(process_num, depth,tempfiles):
    log_file = f"{tempfiles}badread_{process_num}.log"
    reads_file = f"{tempfiles}temp_{process_num}.fastq"
    fastq_file_path = f"{tempfiles}dt4dds_syn_manage_{process_num}.fasta"
    with open(log_file, 'w') as log, open(reads_file, 'w') as reads:
        # # subprocess.run([
        # #     'badread', 'simulate', '--reference', fastq_file_path, '--quantity', str(one_thread_copy_number)
        # #     + 'x --glitches 1000,3,3 --junk_reads 1 --random_reads 1 --chimeras 1 --identity 95,99,2.5'
        # # ], stderr=log, stdout=reads)
        # shell = ('badread simulate --reference ' + fastq_file_path + ' --quantity ' + str(
        #     one_thread_copy_number) + 'x --glitches 1000,3,3 --junk_reads 1 --random_reads 1 --chimeras 1 --identity 95,99,2.5')
        # subprocess.run(shell, shell=True, stderr=log, stdout=reads)
        # # subprocess.run([
        # #     'badread', 'simulate', '--reference', fastq_file_path, '--quantity', str(one_thread_copy_number)
        # #     + 'x --glitches 1000,3,3 --junk_reads 1 --random_reads 1 --chimeras 1 --identity 95,99,2.5 | gzip > ' + reads_file
        # # ], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        shell = f"/home1/hongmei/anaconda3/bin/python /home1/hongmei/00work_files/0000/0isecondwork/Evaluation_platform/Badreadmain/badread-runner.py simulate --reference " \
                f"{fastq_file_path} --quantity {depth}x  --start_adapter_seq '' --end_adapter_seq '' "
        # shell = ('/home1/hongmei/anaconda3/bin/python /home1/hongmei/00work_files/0000/0isecondwork/Evaluation_platform/Badreadmain/badread-runner.py'
        #          ' simulate --reference ' + fastq_file_path + ' --quantity ' + str(
        #     one_thread_copy_number) + 'x --glitches 1000,3,3 --junk_reads 1 --random_reads 1 --chimeras 1 --identity 95,99,2.5')
        subprocess.run(shell, shell=True, stderr=log, stdout=reads)

def readclusterfile(path):
    # lastseqs = [[] for _ in range(len(indexforsearch))]
    lastseqs = []
    with open(path,'r') as f:
        lines = f.readlines()
    for i in range(1,len(lines),2):
        if lines[i].startswith('[]'):
            print(f"{i // 2}丢失")
            lastseqs.append([])
            continue
        splitseqs = lines[i].rstrip().replace('[','').replace(']','').split(', ')
        lastseqs.append([s.rstrip('\'').lstrip('\'') for s in splitseqs])
    return lastseqs

def removeindexs(consensus_seqs,indexs,index_length):
    # indexs = set(indexs)
    real_seqs = []
    indexserrors = 0
    for i in range(len(consensus_seqs)):
        if consensus_seqs[i][:index_length] in indexs:
            real_seqs.append(consensus_seqs[i][index_length:])
        else:
            indexserrors+=1
            print(f"consusindex:{consensus_seqs[i][:index_length+3]} true:{indexs[i]}")
            real_seqs.append(consensus_seqs[i][index_length:])
    print(f"indexserrors:{indexserrors}")
    return real_seqs








