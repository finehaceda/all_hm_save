# 111:
# cd /home1/hongmei/00work_files/0000/0ifirstCompare/bwa_test/bwa
# ./bwa index /home1/hongmei/00work_files/0000/0isecondwork/Evaluation_platform/djangot2/files/index_DNAFountain.fasta
# ./bwa mem /home1/hongmei/00work_files/0000/0isecondwork/Evaluation_platform/djangot2/files/index_DNAFountain.fasta   /home1/hongmei/00work_files/0000/0isecondwork/Evaluation_platform/djangot2/files/simu/simulated_seqsr1r2.fasta  > /home1/hongmei/00work_files/0000/0isecondwork/Evaluation_platform/djangot2/files/decode/ali.sam
# cd /home1/hongmei/00work_files/0000/0isecondwork/Evaluation_platform/djangot2/files/decode
# cat ali.sam | samtools view -bS - > output.bam
# samtools sort output.bam -o output_sorted.bam
# samtools index output_sorted.bam
# samtools view output_sorted.bam | less -S
# samtools view output_sorted.bam | cut -f 3,4,5,6,10,11 > cluster.ali
import random
import re
import subprocess
import time
from datetime import datetime

import Levenshtein

from polls.Code.plot_plt import saveseqsdistributed_fig, saveseqsdistributed_data, saveedit_distributed_data
from polls.Code.utils import readseqs, readtestfastq, readfile, readgfastqfile_withfirstline, savefasta

dictm = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'A'}

def saveclusterfile(path,refseqs,simuseqs):
    with open(path,'w') as f:
        # alldis = []
        for i in range(len(refseqs)):
            dis = []
            f.write(f"{refseqs[i]}\n****\n")
            for j in range(len(simuseqs[i])):
                dis.append(Levenshtein.distance(simuseqs[i][j],refseqs[i]))
                f.write(f"{simuseqs[i][j]}\n")
            f.write(f"\n\n")
    with open(path+'withdis','w') as f:
        # alldis = []
        for i in range(len(refseqs)):
            dis = []
            f.write(f"{refseqs[i]}\n****\n")
            for j in range(len(simuseqs[i])):
                dis.append(Levenshtein.distance(simuseqs[i][j],refseqs[i]))
                f.write(f"{simuseqs[i][j]}\n")
            f.write(f"{dis}\n\n")
            # alldis.append(f"{dis}/{len(simuseqs[i])}")
        # print(f"every cluster sum dis/seqnum:{alldis}")
# def saveclusterfile(path,refseqs,simuseqs):
#     with open(path,'w') as f:
#         for i in range(len(refseqs)):
#             f.write(f"{refseqs[i]}\n****\n")
#             for j in range(len(simuseqs[i])):
#                 dis = Levenshtein.distance(simuseqs[i][j],refseqs[i])
#                 f.write(f"{simuseqs[i][j]} dis:{dis}\n")
#             f.write(f"\n\n")
# def cluster_by_index(simulated_seqs,indexforsearch,index_length,sequence_length):
def cluster_by_index(file_ref,file_seqs,index_length,save_path):
    # 使用index进行聚类，对于fasta文件，质量值也为序列，有phred的需要保存下来
    print(f"--------开始聚类cluster_by_index--------，聚类index_length:{index_length}")
    starttime = datetime.now()
    file_seqs_path = file_seqs + '.forcluster'
    file_ref_path = file_ref + '.forcluster'
    param,indexforsearch,simulated_seqs,simulated_quas,refseqs = getclusterseqs(file_ref,file_seqs,index_length)
    savefasta(file_seqs_path,simulated_seqs)
    savefasta(file_ref_path,indexforsearch)
    sequence_length = int(param.rstrip().split(',')[1].split(':')[1])
    lastseqs = [[] for _ in range(len(indexforsearch))]
    lastquas = [[] for _ in range(len(indexforsearch))]
    nousedsimuseqs = 0
    for i in range(len(simulated_seqs)):
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
                lastquas[minindex].insert(0,simulated_quas[i])
                continue

            rmindis,rminindex = 10,-1
            reversedseq_d = simulated_seqs[i][::-1]
            reversedseq=''
            dict = {'A': 'T', 'C': 'G', 'T': 'A', 'G': 'C'}
            for j in range(len(reversedseq_d)):
                reversedseq += dict[reversedseq_d[j]]
            rq = simulated_quas[i][::-1]
            for j in range(len(indexforsearch)):
                dis = Levenshtein.distance(reversedseq[:index_length], indexforsearch[j])
                if dis < rmindis:
                    rmindis, rminindex = dis, j
                if dis <= 0:
                    break
            if rmindis == 0:
                # lastseqs[rminindex].append(reversedseq)
                lastseqs[rminindex].insert(0,reversedseq)
                lastquas[rminindex].insert(0,rq)
                continue
            if mindis>=3 and rmindis>=3:
                nousedsimuseqs+=1
                continue
            if mindis < rmindis:
                lastseqs[minindex].append(simulated_seqs[i])
                lastquas[minindex].append(simulated_quas[i])
            else:
                lastseqs[rminindex].append(reversedseq)
                lastquas[rminindex].append(rq)

    #保存序列数量分布图
    # saveseqsdistributed_data(lastseqs)
    # saveseqsdistributed_fig(lastseqs)
    # saveedit_distributed_data(refseqs,lastseqs)
    saveclusterfile(save_path,refseqs,lastseqs)
    infos = {'cluster_seqs_path':save_path,'param':param}
    if str(file_seqs).endswith('fastq'):
        with open(save_path+'.phred','w') as f:
            for i in range(len(indexforsearch)):
                f.write(f"{indexforsearch[i]}\n****\n")
                for j in range(len(lastquas[i])):
                    f.write(f"{lastquas[i][j]}\n")
                f.write(f"\n\n")
        infos['cluster_phred_path']=save_path+'.phred'
    print(f"总共有模拟序列{len(simulated_seqs)}条，未使用的序列有:{nousedsimuseqs}条")

    infos['cluster_time'] = datetime.now()-starttime
    print(f"--------聚类结束cluster_by_index--------")
    return infos

def cluster_by_ref(file_ref,file_seqs,lens,save_path,dir='/home1/hongmei/00work_files/0000/0isecondwork/Evaluation_platform/djangot2/files/decode/'):
    print(f"--------开始聚类cluster_by_ref--------")
    starttime = datetime.now()
    # 使用ref进行聚类，不用区分fasta/fastq文件，都可以使用bwa进行聚类，在读取比对好后的ali文件不同，有phred的需要读取到来并保存下来
    file_seqs_path = file_seqs+'.forcluster'
    file_ref_path = file_ref+'.forcluster'
    param = getclusterfiles(file_ref,file_seqs,file_ref_path,file_seqs_path,lens)
    sequence_length = int(param.rstrip().split(',')[1].split(':')[1])
    commands = f"""
    cd /home1/hongmei/00work_files/0000/0ifirstCompare/bwa_test/bwa
    ./bwa index {file_ref_path}
    ./bwa mem {file_ref_path}   {file_seqs_path}  > {dir}ali.sam
    cd {dir}
    cat ali.sam | samtools view -bS - > output.bam
    samtools sort output.bam -o output_sorted.bam
    samtools index output_sorted.bam
    samtools view output_sorted.bam | cut -f 3,4,5,6,10,11 > cluster.ali
    """
    # print(f"commands:{commands}")
    result = subprocess.run(commands, shell=True, text=True, capture_output=True)
    # 输出结果
    # print(result.stdout)
    print(result.stderr)
    if str(file_seqs).endswith('fastq'):
        mdict=read_cluster_ali_phred(dir+'cluster.ali',sequence_length)
    else :
        mdict=read_cluster_ali_nophred(dir+'cluster.ali',sequence_length)
    simuseqs,simuquas = [],[]
    simuseqs_fixnum,simuquas_fixnum = [],[]
    refseqs = []
    with open(file_ref,'r') as f:
        lines = f.readlines()
    allseqsnum, minnum = 0,1000
    for i in range(2,len(lines),2):
        refseqs.append(lines[i].rstrip())
        seqs_quas = mdict.get('seq'+str(i//2-1),[[],[]])

        simuseqs.append(seqs_quas[0])
        simuquas.append(seqs_quas[1])
        # # select_nums=10
        # #250422修改 聚类后保存固定save_nums=10数量的序列
        # length = len(seqs_quas[0])
        # allseqsnum += length
        # if length < minnum: minnum = length
        # if length >= select_nums:
        #     indexs = random.sample(range(length), select_nums)
        # else:
        #     indexs = [j for j in range(length)]
        # # print(f'indexs:{indexs},len:{len(seqs_quas[0])},{len(seqs_quas[1])}')
        #
        # # print(f'1:{[j for j in indexs if j < len(seqs_quas[0])]}')
        # # print(f'2:{[j for j in indexs if j < len(seqs_quas[1])]}')
        # simuseqs_fixnum.append([seqs_quas[0][j] for j in indexs if j < len(seqs_quas[0])])
        # simuquas_fixnum.append([seqs_quas[1][j] for j in indexs if j < len(seqs_quas[1])])
    #保存序列数量分布图
    # saveseqsdistributed_data(simuseqs)
    # saveseqsdistributed_fig(simuseqs)
    # saveedit_distributed_data(refseqs,simuseqs)
    # saveclusterfile(dir+'cluster.fasta',refseqs,simuseqs_fixnum)
    # saveclusterfile(save_path,refseqs,simuseqs_fixnum)
    saveclusterfile(save_path,refseqs,simuseqs)
    infos = {'cluster_seqs_path':save_path,'param':param}
    if str(file_seqs).endswith('fastq'):
        with open(save_path+'.phred','w') as f:
            for i in range(len(refseqs)):
                f.write(f"{refseqs[i]}\n****\n")
                for j in range(len(simuseqs[i])):
                    f.write(f"{simuquas[i][j]}\n")
                f.write(f"\n\n")
        infos['cluster_phred_path']=save_path+'.phred'
    infos['cluster_time'] = datetime.now()-starttime
    print(f"--------聚类结束cluster_by_ref--------")
    # print(f"\naverage seq num : {allseqsnum/len(simuseqs_fixnum)}")
    # print(f"min seq num : {minnum}")
    return infos

#返回参考序列index和测序序列
def getclusterseqs(file_ref,file_seqs,lens):
    if str(file_seqs).endswith('fastq'):
        newsesq,newphreds = readgfastqfile_withfirstline(file_seqs)
    else:
        newsesq,_ = readfile(file_seqs)
        newphreds = newsesq
    newrefs=[]
    refall = []
    with open(file_ref,'r')as f:
        lines = f.readlines()
    for i in range(1,len(lines),2):
        refall.append(lines[i+1].rstrip())
        newrefs.append(lines[i+1].rstrip()[:lens])
    return lines[0],newrefs,newsesq,newphreds,refall

def getclusterfiles(file_ref,file_seqs,file_ref_path,file_seqs_path,lens,primer=''):
    newrefs = []
    newsesq = []
    #这里把参考序列分为原序列和反向互补链，测序序列去除第一行信息
    # dictm = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'A'}
    with open(file_ref,'r')as f:
        lines = f.readlines()
    param = lines[0]
    lines = lines[1:]
    for i in range(1,len(lines),2):
        thisseq = lines[i].rstrip()
        newrefs.append(thisseq)
        s = ''
        for j in range(len(thisseq) - 1, -1, -1):
            s += dictm[str(thisseq[j])]
        newrefs.append(s)
    if lens == -1:
        lens = len(newrefs[0])
    # print(f"len(newrefs):{len(newrefs)}")
    print(f"lens:{lens},file_ref_path:{file_ref_path},newrefs[i*2][:lens]:{newrefs[0][:lens]}")
    with open(file_ref_path,'w')as f:
        for i in range(len(newrefs)//2):
            f.write(f">seq{i}-0\n{newrefs[i*2][:lens]}\n")
            f.write(f">seq{i}-1\n{newrefs[i*2+1][:lens]}\n")
    # n=2
    # if str(file_seqs).endswith('fastq'):
    #     n = 4
    # with open(file_seqs,'r')as f:
    #     lines = f.readlines()[1:]
    # for i in range(1,len(lines),4):
    #     newsesq.append(lines[i].rstrip('\n'))
    # print(f"共有：{len(newsesq)}条待聚类的序列")
    # with open(file_seqs_path,'w')as f:
    #     for i in range(len(newsesq)):
    #         f.write(f">seq{i}\n{newsesq[i]}\n")
    with open(file_seqs,'r')as f:
        lines = f.readlines()[1:]
    with open(file_seqs_path,'w')as f:
        f.writelines(lines)
    # with open(file_seqs,'r')as f:
    #     lines = f.readlines()[1:]
    # for i in range(1,len(lines),4):
    #     newsesq.append(lines[i].rstrip('\n'))
    # with open(file_seqs_path,'w')as f:
    #     for i in range(len(newsesq)):
    #         f.write(f">seq{i}\n{newsesq[i]}\n")
    return param

def read_cluster_ali_phred(path,sequence_length):
    minlen = sequence_length*0.8
    with open(path,'r') as f:
        lines = f.readlines()
    mdict = {}
    for i in range(len(lines)):
        line = lines[i].rstrip().split('\t')
        # print(line)
        if not line[0].startswith('seq'):
            break
        name = line[0].split('-')[0]
        positive = line[0].split('-')[1]
        indels = line[3]
        left_number,right_number=getindels(indels,len(line[4]))
        if name not in mdict:
            mdict[name] = [[],[]]
        thisseq = line[4][left_number:right_number]
        if len(thisseq) < minlen:
            continue
        thisqua = line[5][left_number:right_number]
        if positive == '0':
            mdict[name][0].append(thisseq)
            mdict[name][1].append(thisqua)
        else:
            s,q = '',''
            for j in range(len(thisseq) - 1, -1, -1):
                s += dictm[str(thisseq[j])]
                q += str(thisqua[j])
            mdict[name][0].append(s)
            mdict[name][1].append(q)
    return mdict

def read_cluster_ali_nophred(path,sequence_length):
    minlen = sequence_length*0.8
    with open(path,'r') as f:
        lines = f.readlines()
    mdict = {}
    for i in range(len(lines)):
        line = lines[i].rstrip().split('\t')
        if not line[0].startswith('seq'):
            continue
        name = line[0].split('-')[0]
        positive = line[0].split('-')[1]
        indels,seq = line[3],line[4]
        left_number,right_number=getindels(indels,len(seq))
        if name not in mdict:
            mdict[name] = [[],[]]
        thisseq = seq[left_number:right_number]
        if len(thisseq) < minlen:
            continue

        # print(f"left_number:{left_number},right_number:{right_number},len:{len(thisseq)}")
        if positive == '0':
            mdict[name][0].append(thisseq)
        else:
            s = ''
            for j in range(len(thisseq) - 1, -1, -1):
                s += dictm[str(thisseq[j])]
            mdict[name][0].append(s)
    return mdict

def getindels(s,l):

    # 找到左边第一个数字（连续的数字）
    left_number = re.match(r'\d+S', s)
    if left_number:
        left_number = left_number.group().replace('S', '')
    else:
        left_number = None

    # 找到右边第一个数字（连续的数字，反向搜索）
    right_number = re.search(r'S\d+', s[::-1])
    if right_number:
        right_number = l-int(right_number.group()[::-1].replace('S', ''))  # 反转回来
    else:
        right_number = None
    if left_number == None:
        left_number = 0
    if right_number==None:
        right_number = l

    return int(left_number),int(right_number)