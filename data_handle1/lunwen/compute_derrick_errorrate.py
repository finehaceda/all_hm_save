import random
from datetime import datetime

import Levenshtein
import numpy as np


def getoriseqs(path):
    oridnaseqs = []
    dnaseqs = []
    dnaquas = []
    with open(path,'r') as file:
        lines = file.readlines()
    i = 0
    len1 = len(lines)
    while i < len1:
        # if len(oridnaseqs)>=10000:
        #     break
        if lines[i].startswith('>ori'):
            i += 1
            oridnaseqs.append(lines[i].strip('\n'))
            i+=1
            dnaseq = []
            dnaqua = []
            while i < len1 and not lines[i].startswith('>ori'):
                i+=1
                dnaseq.append(lines[i].strip('\n'))
                i+=2
                dnaqua.append(lines[i].strip('\n'))
                i+=1
            dnaseqs.append(dnaseq)
            dnaquas.append(dnaqua)
    return oridnaseqs, dnaseqs, dnaquas



def saveseqs(path,oriseqs,dnaseqs):
    with open(path,'w') as f:
        for i in range(len(oriseqs)):
            f.write(f"{oriseqs[i]}\n****\n")
            for j in range(len(dnaseqs[i])):
                f.write(f"{dnaseqs[i][j]}\n")
            f.write("\n\n")

def saveseqsphreds(path,oriseqs,dnaseqs,all_phreds):
    with open(path,'w') as f:
        for i in range(len(oriseqs)):
            f.write(f"{oriseqs[i]}\n****\n")
            for j in range(len(dnaseqs[i])):
                f.write(f"{dnaseqs[i][j]}\n")
                f.write(f"{all_phreds[i][j]}\n")
            f.write("\n\n")


def geDerrickTestFile11(seqspath):
    oridnaseqs, dnaseqs, dnaquas = getoriseqs(seqspath)
    oriseqs1,seqs1 = [],[]
    for i in range(len(oridnaseqs)):
        if len(dnaseqs[i])!= 0:
            oriseqs1.append(oridnaseqs[i])
            seqs1.append(dnaseqs[i])
    # print(len(oriseqs))
    # print(len(seqs))
    saveseqs("derrick_data_le5_no0.txt",oriseqs1,seqs1)

def geDerrickTestFile(seqspath):
    oridnaseqs, dnaseqs, dnaquas = getoriseqs(seqspath)
    ori_seqs,all_seqs,all_phreds  = [],[],[]
    for i in range(len(oridnaseqs)):
        if len(dnaseqs[i])!= 0:
            ori_seqs.append(oridnaseqs[i])
            all_seqs.append(dnaseqs[i])
            all_phreds.append(dnaseqs[i])
    print(len(ori_seqs))
    print(len(all_seqs))
    # saveseqs("derrick_data_le5_no0.txt",oriseqs1,seqs1)
    for i in range(5,16):
        new_seqs = [[]]*len(all_seqs)
        new_phreds = [[]]*len(all_phreds)
        for j in range(len(all_seqs)):
            random_numbers = random.sample(range(len(all_seqs[j])), min(i,len(all_seqs[j])))
            # for j in range(len(dnaseqs[i])):
            new_seqs[j] = [all_seqs[j][t] for t in random_numbers]
            new_phreds[j] = [all_phreds[j][t] for t in random_numbers]
        saveseqs(f'derrick_cluster/derrick_bma_data_le{i}_no0.txt',ori_seqs[10000:20000],new_seqs[10000:20000])
        saveseqsphreds(f'derrick_cluster/derrick_bma_data_le{i}_no0_phred.txt',ori_seqs[:20000],new_seqs[:20000],new_phreds[:20000])
        # saveseqsphreds(f'derrick_cluster/derrick_bma_data_le{i}_1_no0_phred.txt',ori_seqs[:15000],new_seqs[:15000],new_phreds[:15000])
        # saveseqsphreds(f'derrick_cluster/derrick_bma_data_le{i}_2_no0_phred.txt',ori_seqs[:10000]+ori_seqs[15000:20000],new_seqs[:10000]+new_seqs[15000:20000],new_phreds[:10000]+new_phreds[15000:20000])

def geDerrickTestFile_fix(seqspath):
    oridnaseqs, dnaseqs, dnaquas = getoriseqs(seqspath)
    ori_seqs,all_seqs,all_phreds  = [],[],[]
    for i in range(len(oridnaseqs)):
        if len(dnaseqs[i])!= 0:
            ori_seqs.append(oridnaseqs[i])
            all_seqs.append(dnaseqs[i])
            all_phreds.append(dnaquas[i])
    print(len(ori_seqs))
    print(len(all_seqs))
    # saveseqs("derrick_data_le5_no0.txt",oriseqs1,seqs1)
    for i in range(5,21):
        new_refs,new_seqs,new_phreds = [],[],[]
        for j in range(len(all_seqs)):
            if len(all_seqs[j])>=i:
                new_refs.append(ori_seqs[j])
                random_numbers = random.sample(range(len(all_seqs[j])), i)
                # for j in range(len(dnaseqs[i])):
                new_seqs.append([all_seqs[j][t] for t in random_numbers])
                new_phreds.append([all_phreds[j][t] for t in random_numbers])
        print(f"序列：{len(new_seqs)}条")
        # saveseqs(f'derrick_cluster_fix/derrick_bma_data_le{i}_no0.txt',new_refs[5000:15000],new_seqs[5000:15000])
        # saveseqsphreds(f'derrick_cluster_fix/derrick_bma_data_le{i}_no0_phred.txt',new_refs[10000:20000],new_seqs[10000:20000],new_phreds[10000:20000])
        saveseqsphreds(f'derrick_cluster_fix_all/derrick_bma_data_le{i}_no0_phred.txt',new_refs,new_seqs,new_phreds)
        # saveseqsphreds(f'derrick_cluster/derrick_bma_data_le{i}_1_no0_phred.txt',new_refs[:15000],new_seqs[:15000],new_phreds[:15000])
        # saveseqsphreds(f'derrick_cluster/derrick_bma_data_le{i}_2_no0_phred.txt',new_refs[:10000]+new_refs[15000:20000],new_seqs[:10000]+new_seqs[15000:20000],new_phreds[:10000]+new_phreds[15000:20000])

        # saveseqs(f'derrick_cluster_fix_1000/derrick_bma_data_le{i}_no0.txt',new_refs[10000:11000],new_seqs[10000:11000])
        # saveseqsphreds(f'derrick_cluster_fix_1000/derrick_bma_data_le{i}_no0_phred.txt',new_refs[:11000],new_seqs[:11000],new_phreds[:11000])

        # saveseqs(f'data_test/derrick_bma_data_le{i}_no0.txt',new_refs[11020:11030],new_seqs[11020:11030])
        # saveseqsphreds(f'data_test/derrick_bma_data_le{i}_no0_phred.txt',new_refs[:11000],new_seqs[:11000],new_phreds[:11000])

def remove2(dnaseqs, dnaquas):
    for i in range(len(dnaseqs)):
        for j in range(len(dnaseqs[i])):
            if str(dnaseqs[i][j]).rfind('2') != -1 or str(dnaseqs[i][j]).rfind('0') != -1 or str(dnaseqs[i][j]).rfind('1') != -1:
                dnaseqs[i][j]=dnaseqs[i][j][:len(dnaseqs[i][j])-1]
                dnaquas[i][j]=dnaquas[i][j][:len(dnaseqs[i][j])-1]
    return dnaseqs, dnaquas

def geDerrickTestFile_retest(seqspath,save_path):
    oridnaseqs, dnaseqs, dnaquas = getoriseqs(seqspath)
    ori_seqs,all_seqs,all_phreds  = [],[],[]
    errorrate = []
    dnaseqs, dnaquas = remove2(dnaseqs, dnaquas)
    length = len(oridnaseqs[0])
    for i in range(len(oridnaseqs)):
        if len(dnaseqs[i])!= 0:
            error = sum([Levenshtein.distance(oridnaseqs[i], dnaseqs[i][j]) for j in range(len(dnaseqs[i]))])/len(dnaseqs[i])
            errorrate.append(error)
            ori_seqs.append(oridnaseqs[i])
            all_seqs.append(dnaseqs[i])
            all_phreds.append(dnaquas[i])
    print(np.average(np.asarray(errorrate))/length)
    print(len(ori_seqs))
    print(len(all_seqs))
# geDerrickTestFile('/home1/hongmei/00work_files/1229/DNAtransformer/0122bsalign/getSequencefiles/seqsforphread260nodislimit.fasta')
# geDerrickTestFile('/home1/hongmei/00work_files/1229/DNAtransformer/0122bsalign/getSequencefiles/seqsforphread260.fasta')
# geDerrickTestFile_fix('/home1/hongmei/00work_files/1229/DNAtransformer/0122bsalign/getSequencefiles/seqsforphread260.fasta')
# geDerrickTestFile_retest('/home1/hongmei/00work_files/1229/DNAtransformer/0122bsalign/getSequencefiles/seqsforphread260.fasta')
geDerrickTestFile_retest('/home1/hongmei/00work_files/1229/DNAtransformer/0122bsalign/getSequencefiles/seqsforphread260.fasta',
                         '/home2/hm/datasets/derrick')

