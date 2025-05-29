import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime

import Levenshtein

from .dt4simu import get_allsequences, syn_simu_advanced, seq_simu_advanced, mergesimuseqs, logger, \
    get_allsequences_withprimer
from .utils import saveseqs, readseqs, pri0pre20, sequence_length, primers_0, readsynfasta, SimuInfo, readtestfastq, \
    readgzipfile, readsynfasta_diff, syn_to_badread, onethreadreads


# def adddt4simu_advanced(sequencespath,steps):
def adddt4simu_advanced(simuInfo:SimuInfo,filessave):
    param,dna_sequences = getseques(simuInfo.inputfile_path)
    sequence_length = int(param.rstrip().split(',')[1].split(':')[1])
    if param.rstrip().split(',')[0].split(':')[1] == 'derrick':
        sequence_length += 12
    # encode_method = param.rstrip().split(',')[0].split(':')[1]
    print(f"sequence_length1:{sequence_length}")
    simuInfo.sequence_length = sequence_length
    simuInfo.param = param
    start_time = datetime.now()
    # 二、模拟合成+decay+pcr+sampling错误 dt4dds
    pool = syn_simu_advanced(dna_sequences,simuInfo)
    prepath = './files/simu/simulated_noseqencing.fasta'
    pool.save_as_fasta(prepath,len(dna_sequences)*100)

    # # 使用cluster_seqpool.to_list() 会保存所有序列，且如果一条序列count为n，则保留n条，cluster_seqpool.save_as_fasta则会把序列和丰富度保存到文件中
    # allseqs = pool.to_list()
    # with open(prepath, 'w', encoding='utf-8') as f:
    #     for index, seq in enumerate(allseqs):
    #         f.write(f">#{str(index).zfill(6)}\n{str(seq)}\n")




    print(f"\n引入 syn/pcr/decay/sampling 错误时间：{str(datetime.now() - start_time)}")
    # 二、模拟测序错误  dt4dds + badreads
    infos = handle_simufiles(dna_sequences, simuInfo, pool, prepath, filessave)
    costtime = datetime.now() - start_time
    infos['time'] = costtime
    return infos

def getsynseqs_forsequencing(pool,n_reads,path,sequence_length):
    cluster_seqpool = pool
    # sample oligos to get expected number of reads
    if cluster_seqpool.n_oligos >= n_reads:
        cluster_seqpool = cluster_seqpool.sample_by_counts(n_reads, remove_sampled_oligos=False)
    elif cluster_seqpool.n_oligos == 0:
        logger.exception("Unable to sequence, no sequence-able oligos found.")
    else:
        logger.warning(
            f"Only {cluster_seqpool.n_oligos} oligos available for total of {n_reads} reads. Continuing.")

    # 这里使用cluster_seqpool.to_list()会保存所有序列，且如果一条序列count为n，则保留n条，cluster_seqpool.save_as_fasta则会把序列和丰富度保存到文件中
    # cluster_seqpool.save_as_fasta("./testFiles/testResult/dt4dds_syn.fasta")
    # 格式如下：>#000082, abundance: 4
    # AGCACACGTCTGAACTCCAGTCACATCACGATCTCGTATGCCGTCTTCTGCTTG
    cluster_seqpool.save_as_fasta(path + '.pre')
    # 按照abundance的数量读取reads
    # allinfos, allseqs_r1 = readsynfasta(path + '.pre')
    allinfos, allseqs_r1 = readsynfasta_diff(path + '.pre')
    print(f"合成后总共有{len(allseqs_r1)}条不同的序列")

    allseqs_r1 = cluster_seqpool.to_list()
    true_allseqs_r1 = handle_synseqs_withprimer(allseqs_r1,sequence_length)
    # true_allseqs_r1 = allseqs_r1
    print(f"合成后总共有{len(true_allseqs_r1)}条用于测序的序列")
    # print(f"true_allseqs_r1:{len(true_allseqs_r1)}")
    # with open(path + '.pre', 'w') as f:
    #     # f.write(param)
    #     for i in range(len(allseqs_r1)):
    #         f.write(f">seq{i}\n{allseqs_r1[i]}\n")

    # infos = {'path': path + '.pre'}
    return true_allseqs_r1

def handle_synseqs(allseqs_r1,sequence_length,positive=True):
    true_allseqs_r1 = []
    for i in range(len(allseqs_r1)):
        index_pre = allseqs_r1[i].find(primers_0[0])
        index = allseqs_r1[i].find(pri0pre20)
        tseq = allseqs_r1[i]
        # print(f"indexpre:{index_pre},index:{index}")
        if index_pre >= 0 and index >= 0:
            tseq = tseq[index_pre+len(primers_0[0]):index]
        elif index_pre >= 0:
            tseq = tseq[index_pre+len(primers_0[0]):index_pre+len(primers_0[0])+sequence_length+1]
        elif index >= 0:
            before = 0
            if index-sequence_length-1>0:
                before = index-sequence_length-1
            tseq = tseq[before:index]
        #注意，这个只能针对于正向序列
        elif positive:
            tseq = tseq[:sequence_length]
        true_allseqs_r1.append(tseq)
    return true_allseqs_r1

def handle_synseqs_phreds(allseqs_r1,allphreds_r1,sequence_length,positive=True):
    true_allseqs_r1 = []
    true_allphreds_r1 = []
    for i in range(len(allseqs_r1)):
        index_pre = allseqs_r1[i].find(primers_0[0])
        index = allseqs_r1[i].find(pri0pre20)
        tseq = allseqs_r1[i]
        tphred = allphreds_r1[i]
        # print(f"indexpre:{index_pre},index:{index}")
        if index_pre >= 0 and index >= 0:
            tseq = tseq[index_pre+len(primers_0[0]):index]
            tphred = tphred[index_pre+len(primers_0[0]):index]
        elif index_pre >= 0:
            tseq = tseq[index_pre+len(primers_0[0]):index_pre+len(primers_0[0])+sequence_length+1]
            tphred = tphred[index_pre+len(primers_0[0]):index_pre+len(primers_0[0])+sequence_length+1]
        elif index >= 0:
            before = 0
            if index-sequence_length-1>0:
                before = index-sequence_length-1
            tseq = tseq[before:index]
            tphred = tphred[before:index]
        #注意，这个只能针对于正向序列
        elif positive:
            tseq = tseq[:sequence_length]
            tphred = tphred[:sequence_length]
        true_allseqs_r1.append(tseq)
        true_allphreds_r1.append(tseq)
    return true_allseqs_r1,true_allphreds_r1

def handle_synseqs_withprimer(allseqs_r1,sequence_length,positive=True):
    true_allseqs_r1 = []
    for i in range(len(allseqs_r1)):
        index_pre = str(allseqs_r1[i]).find(primers_0[0])
        index = str(allseqs_r1[i]).rfind(pri0pre20)
        tseq = allseqs_r1[i]
        # print(f"indexpre:{index_pre},index:{index}")
        if index_pre >= 0 and index >= 0:
            # tseq = tseq[index_pre+len(primers_0[0]):index]
            tseq = tseq[index_pre:index+len(pri0pre20)]
        elif index_pre >= 0:
            tseq = tseq[index_pre:index_pre+len(primers_0[0])+sequence_length+len(pri0pre20)]
        elif index >= 0:
            before = 0
            if index-sequence_length-len(primers_0[0])>0:
                before = index-sequence_length-len(primers_0[0])
            tseq = tseq[before:index]
        #注意，这个只能针对于正向序列
        elif positive and len(tseq)<=158:
            tseq = tseq[:sequence_length]
        true_allseqs_r1.append(tseq)
    return true_allseqs_r1

def badsimu_before0423(allseqs,depth,badparams='',model='',resultpath='./files/'):
    start_time = datetime.now()
    with open(f"{resultpath}dt4dds_syn_manage.fasta",'w') as f:
        for i in range(len(allseqs)):
            f.write(f">#seqs{i}\n{allseqs[i]}\n")
    if badparams == '':
        if model == '':
            shell = f"/home1/hongmei/anaconda3/bin/python /home1/hongmei/00work_files/0000/0isecondwork/Evaluation_platform/Badreadmain/badread-runner.py simulate --reference "\
                    f"{resultpath}dt4dds_syn_manage.fasta --quantity {depth}x  --start_adapter_seq '' --end_adapter_seq '' " \
                                f"> {resultpath}badreads_simutest.fastq"
        else:
            shell = f"/home1/hongmei/anaconda3/bin/python /home1/hongmei/00work_files/0000/0isecondwork/Evaluation_platform/Badreadmain/badread-runner.py simulate --reference "\
                    f"{resultpath}dt4dds_syn_manage.fasta --quantity {depth}x --error_model {model} --qscore_model {model} --start_adapter_seq '' --end_adapter_seq '' " \
                                f"> {resultpath}badreads_simutest.fastq"
    else:
        print(f'使用输入参数成功！')
        shell=badparams
    print(f"shell:{shell}")
    result = subprocess.run(shell, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(f"--------------------------badreads simu....--------------------------------")
    print(f"引入测序错误时间：{str(datetime.now() - start_time)}")
    allseqs, allphreds = readtestfastq(resultpath + 'badreads_simutest.fastq')
    return allseqs, allphreds

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
        return fastq_file_path


def badread_simulator_parallel(processes, depth,tempfiles='/home2/hm/badreads/'):
    threads = []
    one_thread_copy_number = depth
    print(f'one_thread_copy_number:{one_thread_copy_number}')

    #线程池
    with ThreadPoolExecutor(max_workers=25) as executor:
        futures = []
        for p in range(1, processes+1):
            future = executor.submit(run_badread_simulate, p, one_thread_copy_number, tempfiles)
            futures.append(future)
        # 显式等待所有任务完成
        wait(futures)
    # # 多个线程
    # # Start the subprocesses
    # for p in range(1, processes + 1):
    #     thread = threading.Thread(target=run_badread_simulate, args=(p, one_thread_copy_number,tempfiles))
    #     threads.append(thread)
    #     thread.start()
    #
    # # Wait for all threads to complete
    # for thread in threads:
    #     thread.join()
    # Combine the output files and compress
    print(f'start badreads_simutest merge')
    with open(f'{tempfiles}badreads_simutest.fastq', 'wb') as outfile:
        subprocess.run(f'cat {tempfiles}temp_*.fastq', shell=True, stdout=outfile)

    # Clean up temporary files
    for p in range(1, processes + 1):
        os.remove(f"{tempfiles}dt4dds_syn_manage_{p}.fasta")
        os.remove(f"{tempfiles}temp_{p}.fastq")
        os.remove(f"{tempfiles}badread_{p}.log")
    return f'{tempfiles}badreads_simutest.fastq'

def badsimu(allseqs,depth,simuInfo,model='',resultpath='/home2/hm/badreads/'):
    start_time = datetime.now()
    # with open(f"{resultpath}dt4dds_syn_manage.fasta",'w') as f:
    #     for i in range(len(allseqs)):
    #         f.write(f">#seqs{i}\n{allseqs[i]}\n")

    tempfiles = resultpath
    # simulate_file_path = '/home2/hm/badreads_0isec/badreads_simutest.fastq'
    # onethreadreads = 2000
    num = 0
    for i in range(0, len(allseqs), onethreadreads):
        file_index = i // onethreadreads + 1
        chunk = allseqs[i:i + onethreadreads]
        with open(tempfiles + f'dt4dds_syn_manage_{file_index}.fasta', 'w') as f:
            for j in range(len(chunk)):
                f.write(f">#seqs{i + j}\n{chunk[j]}\n")
        num += 1
    print(f'共{num}个文件待测序')
    if simuInfo.badparams == '':
        if model == '':
            #开20个线程


            print(f"--------------------------badreads simu....--------------------------------")
            # badread_simulator_parallel(f'{resultpath}dt4dds_syn_manage.fasta',simuInfo.thread, num,tempfiles)
            badread_simulator_parallel(num, depth,tempfiles)
        #     shell = f"/home1/hongmei/anaconda3/bin/python /home1/hongmei/00work_files/0000/0isecondwork/Evaluation_platform/Badreadmain/badread-runner.py simulate --reference "\
        #             f"{resultpath}dt4dds_syn_manage.fasta --quantity {depth}x  --start_adapter_seq '' --end_adapter_seq '' " \
        #                         f"> {resultpath}badreads_simutest.fastq"
        # else:
        #     shell = f"/home1/hongmei/anaconda3/bin/python /home1/hongmei/00work_files/0000/0isecondwork/Evaluation_platform/Badreadmain/badread-runner.py simulate --reference "\
        #             f"{resultpath}dt4dds_syn_manage.fasta --quantity {depth}x --error_model {model} --qscore_model {model} --start_adapter_seq '' --end_adapter_seq '' " \
        #                         f"> {resultpath}badreads_simutest.fastq"
    # else:
    #     print(f'使用输入参数成功！')
    #     shell=badparams
    #     print(f"shell:{shell}")
    # result = subprocess.run(shell, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # print(f'result:{result.stderr}')
    print(f"引入测序错误时间：{str(datetime.now() - start_time)}")
    allseqs, allphreds = readtestfastq(resultpath + 'badreads_simutest.fastq')
    # allseqs, allphreds = readtestfastq(simulate_file_path)
    return allseqs, allphreds

def removeprimer(posseqs,posphreds,lp,rp,lenz=5,sequence_length=120):
    s,q = [],[]
    minlen = sequence_length - sequence_length*0.2
    for i in range(len(posseqs)):
        # print(f"seq:{posseqs[i]}")
        # print(f"lp:{lp},rp:{rp}")
        lennow = len(lp)
        while lennow > lenz:
            # print(f"lp[-lennow:]:{lp[-lennow:]},rp[:lennow]:{rp[:lennow]}")
            p00, p01 = posseqs[i].find(lp[-lennow:]), posseqs[i].rfind(rp[:lennow])
            thiseq,thiphred = '',''
            if p00 != -1 and p01 != -1:
                thiseq = posseqs[i][p00 + lennow:p01]
                thiphred = posphreds[i][p00 + lennow:p01]
            elif p00 != -1:
                # thiseq = posseqs[i][p00 + lennow:mright(posseqs[i],rp)]
                thiseq = posseqs[i][p00 + lennow:p00 + lennow + sequence_length]
                thiphred = posphreds[i][p00 + lennow:p00 + lennow + sequence_length]
            elif p01 != -1:
                # a,b = mleft(posseqs[i], lp)
                # thiseq = posseqs[i][a+b:p01]
                bi = 0
                if p01 - sequence_length > 0:
                    bi = p01 - sequence_length
                thiseq = posseqs[i][bi:p01]
                thiphred = posphreds[i][bi:p01]
            if thiseq != '':
                if len(thiseq) > minlen:
                    # if len(thiseq)!=len(thiphred):
                    #     print(f"!!!!!!!!!!!!!!!!!!!error!!!!!!!!!!thisseq:{thiseq}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    s.append(thiseq)
                    q.append(thiphred)
                break
            else:
                lennow -= 1
    return s,q

def handle_sequencing_seqs(allseqs, allphreds,sequence_length):
    # 序列为 primers_0[0] + ...+ primers_1[0]，反向互补链为primers_0[1] + ...+ primers_1[1]
    primers_0 = ["ACACGACGCTCTTCCGATCT", "AGACGTGTGCTCTTCCGATCT"]
    primers_1 = ["AGATCGGAAGAGCACACGTCT","AGATCGGAAGAGCGTCGTGT"]
    seqlen_withprimer = sequence_length+len(primers_0[0])+len(primers_0[1])
    mi,mx = seqlen_withprimer-seqlen_withprimer*0.2,seqlen_withprimer+seqlen_withprimer*0.2
    nallseqs,nallphreds = [],[]
    for i in range(len(allseqs)):
        if mi < len(allseqs[i]) < mx and len(allseqs[i]) == len(allphreds[i]):
            nallseqs.append(allseqs[i])
            nallphreds.append(allphreds[i])
    allseqs, allphreds = nallseqs,nallphreds
    print(f"待处理的测序序列共有：{len(allseqs)}条")
    # allseqs = [allseqs[s] for s in range(len(allseqs)) if mi < len(allseqs[s]) < mx ]
    # allphreds = [allphreds[s] for s in range(len(allphreds)) if mi < len(allphreds[s]) < mx ]
    lenz = 4
    # s1,s2=[],[]
    posseqs,zposseqs = [],[]
    posphreds,zposphreds = [],[]
    for i in range(len(allseqs)):
        seq,phred = allseqs[i],allphreds[i]
        ldis,rdis = Levenshtein.distance(primers_0[0][:8],seq[:8]) , Levenshtein.distance(primers_1[0][-8:],seq[-8:])
        zldis,zrdis = Levenshtein.distance(primers_0[1][:8],seq[:8]) , Levenshtein.distance(primers_1[1][-8:],seq[-8:])
        dis,zdis = ldis+rdis,zldis+zrdis
        if dis <= zdis:
            posseqs.append(seq)
            posphreds.append(phred)
        else:
            zposseqs.append(seq)
            zposphreds.append(phred)
    s1,q1 = removeprimer(posseqs,posphreds,primers_0[0],primers_1[0],lenz)
    s2,q2 = removeprimer(zposseqs,zposphreds,primers_0[1],primers_1[1],lenz)
    rs2,rq2 = [],[]
    dict = {'A': 'T', 'C': 'G', 'T': 'A', 'G': 'C'}
    for i in range(len(s2)):
        # print(f"s2:{s2[i]},q2:{q2[i]}")
        newseq,newphred = "",""
        for j in range(len(s2[i]) - 1, -1, -1):
            newseq += dict[s2[i][j]]
            newphred += q2[i][j]
        rs2.append(newseq)
        rq2.append(newphred)
    return s1,q1,rs2,rq2


def handle_simufiles(dna_sequences,simuInfo:SimuInfo,pool,prepath,path):
    # sequence_length = simuInfo.sequence_length
    print(f"simuInfo.channel:{simuInfo.channel},simuInfo.sequencing_method:{simuInfo.sequencing_method},simuInfo.param:{simuInfo.param}")
    if 'sequencing' in simuInfo.channel:
        #现在是二代测序不保留primer，三代测序保留primer
        if simuInfo.sequencing_method == 'paired-end':
            print(f"sequencing->paired-end...：")
            # 模拟测序
            seq_simu_advanced(pool, len(dna_sequences), simuInfo)
            # 1.处理文件，得到测序序列 如果长度长于158，则测序后的序列不会保留primer，primer都放在序列最后面，反向互补链放在primer最前面
            #如果长度小于158-20，会保留primer，如果长度大于158-20，不会保留primer

            simulated_seqsr1, simulated_seqsr2 = get_allsequences(simuInfo.sequence_length)
            # 2.联合正向和反向测序序列 如果序列长度长于158+20，则可以通过正向反向找到重叠序列，然后进行merge得到一条序列
            with open(path+'1', 'w') as f:
                f.write(simuInfo.param)
                for i in range(len(simulated_seqsr1)):
                    # f.write(f">seq{i}\n{simulated_seqs[i]}\n>phred{i}\n{simulated_phreds[i]}\n")
                    f.write(f">seq{i}\n{simulated_seqsr1[i]}\n")
            with open(path+'2', 'w') as f:
                f.write(simuInfo.param)
                for i in range(len(simulated_seqsr2)):
                    # f.write(f">seq{i}\n{simulated_seqs[i]}\n>phred{i}\n{simulated_phreds[i]}\n")
                    f.write(f">seq{i}\n{simulated_seqsr2[i]}\n")

            if len(dna_sequences[0]) > 178:
                print(f"merge...{simuInfo.sequence_length}")
                simulated_seqs = mergesimuseqs(simulated_seqsr1, simulated_seqsr2,simuInfo.sequence_length)
            else:
                simulated_seqs = simulated_seqsr1 + simulated_seqsr2
            # 0109 老师说保留primer，后面用于聚类
            # simulated_seqs, simulated_phreds = get_allsequences_withprimer()
            with open(path, 'w') as f:
                f.write(simuInfo.param)
                for i in range(len(simulated_seqs)):
                    # f.write(f">seq{i}\n{simulated_seqs[i]}\n>phred{i}\n{simulated_phreds[i]}\n")
                    f.write(f">seq{i}\n{simulated_seqs[i]}\n")

            # print(f"测序后共有{len(simulated_seqs)}条序列")
            print(f"原有{len(simulated_seqsr1)}条模拟测序序列，现有{len(simulated_seqs)}条模拟测序序列")
            infos = {'path': path}
            return infos
        if simuInfo.sequencing_method == 'single-end':
            print(f"sequencing->single-end...：")
            seq_simu_advanced(pool, len(dna_sequences), simuInfo)
            # simulated_seqsr1, simulated_seqsr2 = get_allsequences()
            allseqs_r1,_ = readgzipfile('R1.fq.gz')
            simulated_seqsr1 = handle_synseqs(allseqs_r1,simuInfo.sequence_length)
            print(f"测序后共有{len(simulated_seqsr1)}条序列")
            with open(path, 'w') as f:
                f.write(simuInfo.param)
                for i in range(len(simulated_seqsr1)):
                    # f.write(f">seq{i}\n{simulated_seqsr1[i]}\n>phred{i}\n{simulated_phredsr1[i]}\n")
                    f.write(f">seq{i}\n{simulated_seqsr1[i]}\n")

            infos = {'path': path}
            return infos
        if simuInfo.sequencing_method == 'Nanopone':
            print(f"sequencing->Nanopone-paired-end...：")
            # 处理合成序列，得到待测序序列
            n_reads = len(dna_sequences) * syn_to_badread
            print(f'len(dna_sequences):{len(dna_sequences)},n_reads:{n_reads}')
            true_allseqs_r1 = getsynseqs_forsequencing(pool,n_reads,path,simuInfo.sequence_length)
            # 使用badreads进行测序
            allseqs_, allphreds_ = badsimu(true_allseqs_r1,simuInfo.depth,simuInfo)
            # allseqs_, allphreds_ = badsimu(true_allseqs_r1,1,simuInfo.badparams)
            # 处理测序序列，去除primer
            simulated_seqsr1,q1, simulated_seqsr2,q2 = handle_sequencing_seqs(allseqs_, allphreds_,simuInfo.sequence_length)
            allseqs,allphreds = simulated_seqsr1+simulated_seqsr2,q1+q2
            print(f"使用badreads测序得到{len(allseqs_)}条模拟测序序列，处理后有{len(allseqs)}条模拟测序序列")
            # allseqs=allseqs_
            # allseqs=true_allseqs_r1

            #保留primer，for cluster
            # allseqs, allphreds = badsimu(true_allseqs_r1,simuInfo.depth)
            with open(path, 'w') as f:
                f.write(simuInfo.param)
                for i in range(len(allseqs)):
                    f.write(f"@seq{i}\n{allseqs[i]}\n+\n{allphreds[i]}\n")


            infos = {'path': path}
            # infos = {'path': path + '.pre'}
            return infos
        if simuInfo.sequencing_method == 'Pacbio':
            print(f"sequencing->Pacbio-paired-end...：")
            # 处理合成序列，得到待测序序列
            n_reads = len(dna_sequences) * simuInfo.depth
            true_allseqs_r1 = getsynseqs_forsequencing(pool,n_reads,path,simuInfo.sequence_length)
            # 使用badreads进行测序
            allseqs_, allphreds_ = badsimu(true_allseqs_r1,simuInfo.depth,simuInfo,'pacbio2021')
            # 处理测序序列，去除primer
            simulated_seqsr1,q1, simulated_seqsr2,q2 = handle_sequencing_seqs(allseqs_, allphreds_,simuInfo.sequence_length)
            allseqs,allphreds = simulated_seqsr1+simulated_seqsr2,q1+q2
            print(f"使用badreads测序得到{len(allseqs_)}条模拟测序序列，处理后有{len(allseqs)}条模拟测序序列")

            # allseqs, allphreds = badsimu(true_allseqs_r1,simuInfo.depth,'pacbio2021')
            with open(path, 'w') as f:
                f.write(simuInfo.param)
                for i in range(len(allseqs)):
                    # f.write(f">seq{i} {allseqs[i]}\n")
                    f.write(f"@seq{i}\n{allseqs[i]}\n+\n{allphreds[i]}\n")
            infos = {'path': path}
            return infos

    # when 'sequencing' not in simuInfo.channel 注意这里选择丰度>2的
    allinfos,allseqs_r1 = readsynfasta(prepath)
    true_allseqs_r1 = handle_synseqs(allseqs_r1,simuInfo.sequence_length)
    with open(path+'.fasta', 'w') as f:
        f.write(simuInfo.param)
        for i in range(len(true_allseqs_r1)):
            f.write(f"{allinfos[i]}\n{true_allseqs_r1[i]}\n")
    print(f"共有{len(true_allseqs_r1)}条模拟序列")
    # infos = {'path': path, 'costtime': costtime, }
    infos = {'path': path}
    # infos = {'path': path+ '.pre'}
    return infos

def getseques(filename):
    dnasequences = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    param = lines[0]
    for i in range(2,len(lines),2):
        dnasequences.append(lines[i].strip('\n'))
    return param,dnasequences