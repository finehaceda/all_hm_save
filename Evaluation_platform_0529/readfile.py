
def saveseqs(seqs,path):
    with open(path, 'w') as file:
        for j in range(len(seqs)):
            file.write('>'+str(j) + '\n')
            file.write(str(seqs[j]).strip('\n') + '\n')

def getphred_quality(qualityScore):
    phred_qualitys = []
    # sss = seq[110:120]
    for index,i in enumerate(qualityScore):
        phred_quality = ord(i) - 33  # '@'的ASCII码是64，FASTQ使用的是Phred+33编码
        phred_qualitys.append(phred_quality)
    return phred_qualitys

def getAllphred_quality(path):
    all_ori_seqs = []
    allSeqsAndQuas = []
    all_seqs = []
    all_quas = []
    with open(path,'r') as file:
        lines = file.readlines()
    flag = 0
    seqs = []
    # for i in range(len(lines)//2):
    for i in range(len(lines)):
        if flag == 1:
            # all_ori_seqs.append(lines[i][:200].strip('\n'))
            all_ori_seqs.append(lines[i].strip('\n'))
            if len(seqs)>0:
                allSeqsAndQuas.append(seqs)
            seqs = []
            flag = 0
            continue
        if lines[i].startswith('>ori'):
            flag = 1
        elif not (lines[i].startswith('>seq') or  lines[i].startswith('>qua')):
            # seqs.append(lines[i][:200].strip('\n'))
            seqs.append(lines[i].strip('\n'))
    allSeqsAndQuas.append(seqs)
    for i in range(len(allSeqsAndQuas)):
        seqs_quas = allSeqsAndQuas[i]
        reads = seqs_quas[::2]
        quas = seqs_quas[1::2]
        quasScore = []
        for qua in quas:
            quasScore.append(getphred_quality(qua))
        all_seqs.append(reads)
        all_quas.append(quasScore)
    saveseqs(all_seqs,'seq260all_seqs.fasta')
    saveseqs(all_quas,'seq260all_quas.fasta')
    saveseqs(all_ori_seqs,'files/seq260all_ori_seqs.fasta')
    return all_ori_seqs,all_seqs,all_quas

# ori_dna_sequences原序列，all_seqs每条原序列所对应的多条测序序列（数量不同，但每个类都大于10），all_quas测序序列对应的质量值
ori_dna_sequences,all_seqs,all_quas = getAllphred_quality('seqsforphread260.fasta')