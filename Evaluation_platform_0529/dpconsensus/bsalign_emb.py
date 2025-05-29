import subprocess

import Levenshtein
# def readbsalignalignfile(path,allpredict,ratel,rater,dis,dna_sequence_pulsdel,prewith_phred_pulsdel):
import numpy as np


def readbsalignalignfile(path,consus,allpredict,ratel,rater,dis):
    with open(path,'r') as file:
        lines = file.readlines()
    if len(lines)>0:
        mismatchdelinsertindexs = []
        delinsertindexs = []
        upnums = 0
        line = lines[0].strip('\n').split('\t')
        mismatch,delnum,insertnum = line[-3],line[-2],line[-1]
        line = lines[2].strip('\n')
        upnumsindex = []
        # print(line)
        for i in range(len(line)):
            # a = line[i]
            # print(a)
            if line[i]=='-'or line[i] == '*':
                mismatchdelinsertindexs.append(i)
                if line[i]=='-':
                    delinsertindexs.append(i)
        line = lines[3].strip('\n')
        delindex = [i for i in range(len(line)) if line[i] =='-']
        insertindexs = [x for x in delinsertindexs if x not in delindex]
        mismatchindexs = [x for x in mismatchdelinsertindexs if x not in delinsertindexs]

        for i in range(len(delindex)):
            if len(delindex)>i+1:
                if delindex[i+1]==delindex[i]+1:
                    continue
            for misi in range(len(mismatchdelinsertindexs)):
                if mismatchdelinsertindexs[misi]>delindex[i]:
                    mismatchdelinsertindexs[misi] -= 1
            for misi in range(len(insertindexs)):
                if insertindexs[misi]>delindex[i]:
                    insertindexs[misi] -= 1
            for misi in range(len(mismatchindexs)):
                if mismatchindexs[misi]>delindex[i]:
                    mismatchindexs[misi] -= 1
        allpredictreal = [i for i in range(len(allpredict)) if ratel < allpredict[i] <= rater]
        mislen = len(mismatchdelinsertindexs)
        score = []
        upnumsallerrors = [0,0,0]

        for i in allpredictreal:
            if mislen>0:
                # insert_isexist = [j for j in insertindexs if abs(i-j)<10]
                # if len(insert_isexist)>0:
                #     for insertone in insert_isexist:
                #         if insertone <= i:
                #             iij=insertone+1
                #             while iij <=i and consus[iij] == consus[i]:
                #                 iij+=1
                #             if iij==i+1:
                #                 upnumsallerrors[0] += 1
                #                 upnums += 1
                if i in mismatchdelinsertindexs:
                    if i in insertindexs:
                        pass
                        minph = allpredict[i]
                        minphi = i
                        iij = i + 1
                        upnumsallerrors[0]+=1
                        while iij < len(allpredict) and consus[iij] == consus[i]:
                            if allpredict[iij] <= minph and allpredict[iij] <= ratel:
                                minph = allpredict[iij]
                                minphi = iij
                            iij += 1
                        upnums += 1
                        upnumsindex.append({minphi, minph})
                        if minphi != i:
                            # print(minph)
                            # score += f"phred: {minph}\n"
                            score.append(minph)
                            upnumsallerrors[0]-=1
                            upnums -= 1
                    else:
                        if i in mismatchindexs:
                            upnumsallerrors[2] += 1
                        else:
                            upnumsallerrors[1] += 1
                        upnums+=1
                        upnumsindex.append({i,allpredict[i]})

        # for i in allpredictreal:
        #     if mislen>0:
        #         if i in mismatchdelinsertindexs:
        #             upnums+=1
        # if mislen==0 and len(allpredictreal)>0 or mislen<dis:

        # if mislen<dis and ratel>=0.949:
        #     upnums+=dis-mislen
        #     upnumsallerrors[1] += dis-mislen
        #     delnum = int(delnum) + dis-mislen

        # print(mismatch,delnum,insertnum)
        return mismatch,delnum,insertnum,lines[1].strip('\n'),lines[3].strip('\n'),upnums,upnumsindex,score,upnumsallerrors
    return 0,0,0,'','',0,[],"",[]
    #     return mismatch,delnum,insertnum,lines[1].strip('\n'),lines[3].strip('\n'),upnums
    # return 0,0,0,'',''

# def readbsalignalignfile(path):
def readbsalignalignfile11(path):
    with open(path,'r') as file:
        lines = file.readlines()
    if len(lines)>0:
        line = lines[0].strip('\n').split('\t')
        mismatch,delnum,insertnum = line[-3],line[-2],line[-1]
        return mismatch,delnum,insertnum,lines[1].strip('\n'),lines[3].strip('\n')
    return 0,0,0,'',''


# def bsalign_alitest(seq1,seq2):
def bsalign_alitest(seq1,seq2,allpredict,ratel,rater,dis):
    with open('files/seqs.fasta', 'w') as file:
        for j, cus in enumerate([seq1,seq2]):
            file.write('>' + str(j) + '\n')
            file.write(str(cus) + '\n')
    shell = '../bsalign-master/bsalign align files/seqs.fasta > files/ali.ali'
    result = subprocess.run(shell, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # mismatch,delnum,insertnum,seq1,seq2 = readbsalignalignfile('files/ali.ali')
    if dis>=1 and dis <=10:
        mismatch,delnum,insertnum,seq1,seq2,upnums,upnumsindex,wrongin,upnumsallerrors = readbsalignalignfile(
            'files/ali.ali', seq2, allpredict, ratel, rater, dis)
        return mismatch, delnum, insertnum, seq1, seq2, upnums,upnumsindex,wrongin,upnumsallerrors
    else:
        # mismatch,delnum,insertnum,seq1,seq2 = readbsalignalignfile('ali.ali')
        return 0,0,0,seq1,seq2,0,[],[],[]
    # return mismatch,delnum,insertnum,seq1,seq2


def bsalign_alitest11(seq1,seq2):
# def bsalign_alitest(seq1,seq2,allpredict,ratel,rater,dis):
    with open('files/seqs.fasta', 'w') as file:
        for j, cus in enumerate([seq1,seq2]):
            file.write('>' + str(j) + '\n')
            file.write(str(cus) + '\n')
    shell = '../bsalign-master/bsalign align files/seqs.fasta > files/ali.ali'
    result = subprocess.run(shell, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    mismatch,delnum,insertnum,seq1,seq2 = readbsalignalignfile11('files/ali.ali')
    # if dis>1 and dis <=10:
    #     mismatch,delnum,insertnum,seq1,seq2,upnums = readbsalignalignfile('files/ali.ali',allpredict,ratel,rater,dis)
    #     return mismatch, delnum, insertnum, seq1, seq2, upnums
    # else:
    #     # mismatch,delnum,insertnum,seq1,seq2 = readbsalignalignfile('ali.ali')
    #     return 0,0,0,seq1,seq2,0
    return mismatch,delnum,insertnum,seq1,seq2

def count_bsalign_acc(all_consus,all_ori_seqs,all_bsalign_quas):
    rate = 0.9
    while rate < 0.9999:
    # while rate < 0.9999:
        ratel = rate
        # if rate >= 0.8999:
        #     if (rate >= 0.949):
        #         ratel = rate
        #         rater = rate + 0.1
        #     else:
        #         ratel = rate
        #         rater = rate + 0.05
        #
        #     # ratel = round(ratel, 2)
        #     # rater = round(rater, 2)
        # else:
        #     rater = rate + 0.1
        rater = rate + 0.1
        ratel = round(ratel, 2)
        rater = round(rater, 2)
        # ratel = rate
        # rater = rate+0.1
        rate = rater
        lens=len(all_consus)
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
        preupnums = 0
        all_nums_gt0dis = 0
        all_dis_gl0 = 0
        aliseq1,aliseq2  = [], []
        all_dis_sum = 0
        all_nums = 0
        print("ratel:"+str(ratel)+" rater:"+str(rater))
        upnumsallerrorss = [0,0,0]
        score = []
        afterphredupdatedis = 0
        for i in range(len(all_consus)):
            dis = Levenshtein.distance(all_consus[i], all_ori_seqs[i])
            # phredscore = getphred_acc(all_bsalign_quas[i])
            phredscore = all_bsalign_quas[i]
            if len(phredscore)!=len(all_consus[i]):
                print(str(all_consus[i]))
            # phredscore = all_bsalign_quas[i]
            # if(dis>10):
            #     print('bsalign')
            #     print(dis)
            #     print(all_consus[i])
            #     print(all_ori_seqs[i])
            mismatch, delnum, insertnum, seq1, seq2, upnums,upnumsindex,wrongin,upnumsallerrors =\
                bsalign_alitest(all_ori_seqs[i], all_consus[i],phredscore, ratel, rater, dis)
            errornums = int(mismatch)+int(delnum)+int(insertnum)
            score = np.append(score, wrongin)
            for j in range(len(upnumsallerrors)):
                upnumsallerrorss[j] += upnumsallerrors[j]
            all_phred_score.append(phredscore)
            mismatchnums.append(mismatch)
            aliseq1.append(seq1)
            aliseq2.append(seq2)
            delnums.append(delnum)
            insertnums.append(insertnum)
            mismatchnum0 += int(mismatch)
            # pre_seqs.append(all_consus[i])
            delnum0 += int(delnum)
            insertnum0 += int(insertnum)
            all_dis.append(dis)
            num = 0
            scorelt09 = []
            phred = ''
            for j in range(len(phredscore)):
                if ratel < phredscore[j] <= rater:
                    num += 1
                    # scorelt09.append({i, phredscore[j]})
                phred += str(j)+':'+str(phredscore[j]) + ' '
            phred_scores.append(phred)
            all_nums += num
            if dis >=1 and dis <=10:
                all_dis_gl0 += 1
                all_dis_sum += dis
                # all_dis_sum += errornums
                all_nums_gt0dis += num
            if dis>10:
                pre_seqs_morethan10.append(all_consus[i])
            preupnums += upnums
            afterphredupdatedis += dis - upnums
        print('预测碱基概率大于' + str(ratel) + '小于等于:' + str(rater) + '时'
              + '\n所有序列中，在此概率区间碱基数量总共有:' + str(all_nums)
              + ' 发生错误的序列中，在此概率区间的碱基数量有:' + str(all_nums_gt0dis)
              + ' 碱基概率在此区间，全部发生错误的碱基数量 :' + str(preupnums))
        # print(score)
        print(upnumsallerrorss)
        mydict = {'0':0,'1':0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0,'8':0,'9':0}
        for s in score:
            mydict[str(int(s*10))]+=1
        for key,value in mydict.items():
            if value>0:
                print(f"{key}:{value} ")
    print(pre_seqs_morethan10)
    print(f"mismatch数量:{mismatchnum0} del数量:{delnum0} insert数量:{insertnum0} ")
    print('发生错误的序列数量:' + str(all_dis_gl0) + ' 发生错误数量占所有数量： ' + str(all_dis_gl0 / lens))
    print('发生错误的碱基数量 :' + str(all_dis_sum))
    print('每条序列碱基数量：' + str(len(all_ori_seqs[0])))
    print('平均编辑距离:' + str(all_dis_sum / lens))
    print('when dis >= 1 平均编辑距离:' + str(all_dis_sum / all_dis_gl0))
    print('recovery rate:' + str(1 - all_dis_sum / lens / len(all_ori_seqs[0])))

    with open('./myfiles/bsalign_oriandpreseqs.fasta','w') as file:
        for i in range(len(all_ori_seqs)):
            # file.write(f">oriseq{i}\n{all_ori_seqs[i]}\n>preseq{i}   {all_dis[i]}\n{all_consus[i]}\n")
            # file.write(f">oriseq{i}\n{all_ori_seqs[i]}\n>preseq{i} edit:{all_dis[i]} mis:{mismatchnums[i]} del:{delnums[i]} insert:{insertnums[i]}\n{all_consus[i]}\n"
            #            f"phred{i}:{all_phred_score[i]}\n")
            file.write(f">oriseq{i}\n{all_ori_seqs[i]}\n>preseq{i} edit:{all_dis[i]} mis:{mismatchnums[i]}"
                       f" del:{delnums[i]} insert:{insertnums[i]}\n{all_consus[i]}\n"
                       f">aliseqs{i}\n{aliseq1[i]}\n{aliseq2[i]}\n>phredwith_scores{i}\n{phred_scores[i]}\n")

def count_bsalign_acc11(all_consus,all_ori_seqs):
    lens=len(all_consus)
    select_consus_len = len(all_ori_seqs[0])
    mismatchnums = []
    delnums = []
    insertnums = []
    all_dis = []
    all_dis_gl0 = 0
    all_dis_sum = 0
    for i in range(len(all_consus)):

        # te1,te2 = all_ori_seqs[i],all_consus[i]
        # dis = Levenshtein.distance(all_consus[i][:select_consus_len], all_ori_seqs[i])
        dis = Levenshtein.distance(all_consus[i], all_ori_seqs[i])
        mismatch, delnum, insertnum, seq1, seq2 = bsalign_alitest11(all_ori_seqs[i], all_consus[i])
        errornums = int(mismatch)+int(delnum)+int(insertnum)
        mismatchnums.append(mismatch)
        delnums.append(delnum)
        insertnums.append(insertnum)
        all_dis.append(dis)
        if dis >= 1 and dis < 10:
            # inputs_gl0.append(enc)
            all_dis_gl0 += 1
            # all_dis_sum += dis
            all_dis_sum += errornums
        # inputs.append(enc)
        # all_dis+=getEdit(model,enc_inputs[i],dec_outputs[i])
    # print('dis > 0 number:' + str(all_dis_gl0) + '     ' + str(all_dis_gl0 / lens))
    # print('average dis:' + str(all_dis_sum / lens))
    # print('when dis >= 1 average dis:' + str(all_dis_sum / all_dis_gl0))
    # print('recovery rate:' + str(1 - all_dis_sum / lens / len(all_ori_seqs[0])))


    print('发生错误的序列数量:' + str(all_dis_gl0) + ' 发生错误数量占所有数量： ' + str(all_dis_gl0 / lens))
    print('发生错误的碱基数量 :' + str(all_dis_sum))
    print('每条序列碱基数量：' + str(len(all_ori_seqs[0])))
    print('平均编辑距离:' + str(all_dis_sum / lens))
    print('when dis > 1 平均编辑距离:' + str(all_dis_sum / all_dis_gl0))
    print('recovery rate:' + str(1 - all_dis_sum / lens / len(all_ori_seqs[0])))

    with open('./myfiles/bsalign_oriandpreseqs.fasta','w') as file:
        for i in range(len(all_ori_seqs)):
            # file.write(f">oriseq{i}\n{all_ori_seqs[i]}\n>preseq{i}   {all_dis[i]}\n{all_consus[i]}\n")
            file.write(f">oriseq{i}\n{all_ori_seqs[i]}\n>preseq{i} edit:{all_dis[i]} mis:{mismatchnums[i]} del:{delnums[i]} insert:{insertnums[i]}\n"
                       f"{all_consus[i]}\n")
