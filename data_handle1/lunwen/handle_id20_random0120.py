import random


# def saveseqs(path,oriseqs,dnaseqs,copy_num):
#     with open(path,'w') as f:
#         for i in range(len(oriseqs)):
#             f.write(f"{oriseqs[i]}\n****\n")
#             # num = random.randint(5,30)
#             random_numbers = random.sample(range(len(dnaseqs[i])), min(copy_num,len(dnaseqs[i])))
#             # for j in range(len(dnaseqs[i])):
#             for j in random_numbers:
#                 f.write(f"{dnaseqs[i][j]}\n")
#             f.write("\n\n")


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



# trellis_bma_data_le5_no0.txt
def generate(fastq_path, ref_path):
    ori_seqs = []
    all_seqs = []
    index = 0
    seqs = []
    nofastq_indexs = []
    com = '>seq-' + str(index) + '-'
    with open(fastq_path, 'r') as file:
        for line in file:
            if line.find(com) != -1:
                # if len(seqs) >= 5:
                #     continue
                # seqs.append(line.strip().split(' ')[-1])
                _,tseq,tqua = line.strip().split(' ')
                if 140 < len(tseq) < 160:
                    # seqs.append(tseq[25:-25])
                    seqs.append(tseq)
            else:
                if len(seqs) == 0:
                    # if len(seqs)<5:
                    nofastq_indexs.append(index)
                else:
                    all_seqs.append(seqs)
                    if len(all_seqs) >= 10000:
                        # seqs = []
                        break
                seqs = []
                index += 1
                com = '>seq-' + str(index) + '-'
    # if len(seqs) != 0:
    #     all_seqs.append(seqs)
    # else:
    #     nofastq_indexs.append(index)
    with open(ref_path, 'r') as file:
        for i,line in enumerate(file):
            if i%2 == 0 or i // 2 in nofastq_indexs:
                continue
            ori_seqs.append(line.strip('\n'))
            if len(ori_seqs)>=10000:
                break
        # lines = file.readlines()
    # for i in range(1, len(lines), 2):
    #     if i // 2 in nofastq_indexs:
    #         continue
    #     # ori_seqs.append(lines[i].strip('\n')[25:-25])
    #     ori_seqs.append(lines[i].strip('\n'))
    print(f"共有序列：{len(ori_seqs)}条，测序序列：{len(all_seqs)}条")
    for i in range(3,6):
        saveseqs(f'id20_bma_data_le{i}_no0.txt',ori_seqs,all_seqs,i)


# trellis_bma_data_le5_no0.txt
def generate_phred(fastq_path, ref_path,num=20000):
    ori_seqs = []
    all_seqs = []
    all_phreds = []
    index = 0
    seqs = []
    phreds = []
    nofastq_indexs = []
    com = '>seq-' + str(index) + '-'
    with open(fastq_path, 'r') as file:
        for line in file:
            if line.find(com) != -1:
                # if len(seqs) >= 5:
                #     continue
                # seqs.append(line.strip().split(' ')[-1])
                try:
                    splitline = line.strip().split(' ')
                    tseq, tqua = splitline[1],splitline[2]
                except IndexError:
                    continue
                if 146 < len(tseq) < 154:
                    seqs.append(tseq)
                    phreds.append(tqua)
                # seqs.append(tseq)
                # phreds.append(tqua)
            else:
                if len(seqs) == 0:
                    # if len(seqs)<5:
                    nofastq_indexs.append(index)
                else:
                    all_seqs.append(seqs)
                    all_phreds.append(phreds)
                    if len(all_seqs) >= num:
                        # seqs = []
                        break
                seqs = []
                phreds = []
                index += 1
                com = '>seq-' + str(index) + '-'
    # if len(seqs) != 0:
    #     all_seqs.append(seqs)
    # else:
    #     nofastq_indexs.append(index)
    with open(ref_path, 'r') as file:
        for i,line in enumerate(file):
            if i%2 == 0 or i // 2 in nofastq_indexs:
                continue
            ori_seqs.append(line.strip('\n'))
            if len(ori_seqs)>=num:
                break
        # lines = file.readlines()
    # for i in range(1, len(lines), 2):
    #     if i // 2 in nofastq_indexs:
    #         continue
    #     # ori_seqs.append(lines[i].strip('\n')[25:-25])
    #     ori_seqs.append(lines[i].strip('\n'))
    print(f"共有序列：{len(ori_seqs)}条，测序序列：{len(all_seqs)}条")
    for i in range(5,6):
        new_seqs = [[]]*len(all_seqs)
        new_phreds = [[]]*len(all_phreds)
        for j in range(len(all_seqs)):
            random_numbers = random.sample(range(len(all_seqs[j])), min(i, len(all_seqs[j])))
            # for j in range(len(dnaseqs[i])):
            new_seqs[j] = [all_seqs[j][t] for t in random_numbers]
            new_phreds[j] = [all_phreds[j][t] for t in random_numbers]
        saveseqs(f'/home2/hm/datasets/Randomaccess/no_fix_alldata/id20_bma_data_le{i}_no0.txt',ori_seqs[10000:],new_seqs[10000:])
        saveseqsphreds(f'/home2/hm/datasets/Randomaccess/no_fix_alldata/id20_bma_data_le{i}_no0_phred.txt',ori_seqs,new_seqs,new_phreds)
        # saveseqsphreds(f'id20_indexcluster/id20_bma_data_le{i}_1_no0_phred.txt',ori_seqs[:15000],new_seqs[:15000],new_phreds[:15000])
        # saveseqsphreds(f'id20_indexcluster/id20_bma_data_le{i}_2_no0_phred.txt',ori_seqs[:10000]+ori_seqs[15000:],new_seqs[:10000]+new_seqs[15000:],new_phreds[:10000]+new_phreds[15000:])


# trellis_bma_data_le5_no0.txt
def generate_phred_fix(fastq_path, ref_path,save_path,num=20000):
    ori_seqs = []
    all_seqs = []
    all_phreds = []
    index = 0
    seqs = []
    phreds = []
    nofastq_indexs = []
    com = '>seq-' + str(index) + '-'
    with open(fastq_path, 'r') as file:
        for line in file:
            if line.find(com) != -1:
                # if len(seqs) >= 5:
                #     continue
                # seqs.append(line.strip().split(' ')[-1])
                try:
                    splitline = line.strip().split(' ')
                    tseq, tqua = splitline[1],splitline[2]
                except IndexError:
                    continue
                if 146 < len(tseq) < 154:
                    seqs.append(tseq)
                    phreds.append(tqua)
                # seqs.append(tseq)
                # phreds.append(tqua)
            else:
                if len(seqs) == 0:
                    # if len(seqs)<5:
                    nofastq_indexs.append(index)
                else:
                    all_seqs.append(seqs)
                    all_phreds.append(phreds)
                    if len(all_seqs) >= num:
                        # seqs = []
                        break
                seqs = []
                phreds = []
                index += 1
                com = '>seq-' + str(index) + '-'
    # if len(seqs) != 0:
    #     all_seqs.append(seqs)
    # else:
    #     nofastq_indexs.append(index)
    with open(ref_path, 'r') as file:
        for i,line in enumerate(file):
            if i%2 == 0 or i // 2 in nofastq_indexs:
                continue
            ori_seqs.append(line.strip('\n'))
            if len(ori_seqs)>=num:
                break
        # lines = file.readlines()
    # for i in range(1, len(lines), 2):
    #     if i // 2 in nofastq_indexs:
    #         continue
    #     # ori_seqs.append(lines[i].strip('\n')[25:-25])
    #     ori_seqs.append(lines[i].strip('\n'))
    print(f"共有序列：{len(ori_seqs)}条，测序序列：{len(all_seqs)}条")
    for i in range(3,4):
        new_refs,new_seqs,new_phreds = [],[],[]
        # new_seqs = [[]]*len(all_seqs)
        # new_phreds = [[]]*len(all_phreds)
        for j in range(len(all_seqs)):
            if len(all_seqs[j])>=i:
                new_refs.append(ori_seqs[j])
                random_numbers = random.sample(range(len(all_seqs[j])), i)
                new_seqs.append([all_seqs[j][t] for t in random_numbers])
                new_phreds.append([all_phreds[j][t] for t in random_numbers])
            # elif len(all_seqs[j])>2:
            #     new_refs.append(ori_seqs[j])
            #     new_seqs.append(all_seqs[j])
            #     new_phreds.append(all_phreds[j])

        print(f"序列：{len(new_seqs)}条")
        # saveseqs(f'id20_indexcluster_fix/id20_bma_data_le{i}_no0.txt',new_refs[10000:20000],new_seqs[10000:20000])
        # saveseqsphreds(f'id20_indexcluster_fix/id20_bma_data_le{i}_no0_phred.txt',new_refs[:20000],new_seqs[:20000],new_phreds[:20000])
        # saveseqs(f'id20_seqcluster/id20_bma_data_le{i}_no0.txt',new_refs[10000:20000],new_seqs[10000:20000])
        # saveseqsphreds(f'id20_seqcluster/id20_bma_data_le{i}_no0_phred.txt',new_refs[:20000],new_seqs[:20000],new_phreds[:20000])
        # saveseqsphreds(f'id20_indexcluster/id20_bma_data_le{i}_1_no0_phred.txt',ori_seqs[:15000],new_seqs[:15000],new_phreds[:15000])
        # saveseqsphreds(f'id20_indexcluster/id20_bma_data_le{i}_2_no0_phred.txt',ori_seqs[:10000]+ori_seqs[15000:],new_seqs[:10000]+new_seqs[15000:],new_phreds[:10000]+new_phreds[15000:])

        saveseqs(f'{save_path}/data{i}.txt',new_refs[10000:],new_seqs[10000:])
        saveseqsphreds(f'{save_path}/data{i}_phred.txt',new_refs[:20000],new_seqs[:20000],new_phreds[:20000])

# Randomaccess
generate_phred_fix('/home2/hm/Randomaccess/pro_files1126/id20.simple.sameline.handle.noN.all.primer.fasta',
         '/home2/hm/Randomaccess/id20.refs.trs.txt','/home2/hm/datasets/Randomaccess/no_fix_alldata',50000)
# LDPC_Chandak
# generate_phred_fix('/home2/hm/LDPC_DNA_storage_data/ldpc_pro_files2/ldpc.simple.sameline.handle.noN.all.fasta',
#          '/home2/hm/LDPC_DNA_storage_data/ldpc_pro_files2/oligos_1.fa','/home2/hm/datasets/LDPC_Chandak',40000)
# derrick
# generate_phred_fix('/home2/hm/LDPC_DNA_storage_data/ldpc_pro_files2/ldpc.simple.sameline.handle.noN.all.fasta',
#          '/home2/hm/LDPC_DNA_storage_data/ldpc_pro_files2/oligos_1.fa','/home2/hm/datasets/LDPC_Chandak',40000)