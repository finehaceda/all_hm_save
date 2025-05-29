import subprocess

import numpy as np
import torch
from dpconsensus.config import tgt_vocab_size
from torch import nn, optim
from utils import sequence_length


# from Evaluation_platform.dpconsensus.config import tgt_vocab_size, dpconsensus_path
# from Evaluation_platform.utils import sequence_length

def convert_to_DNA(sequence):
    dna_sequence = ""
    for digit in sequence:
        digit = digit.item()
        if digit == 0:
            dna_sequence += 'A'
        elif digit == 1:
            dna_sequence += 'C'
        elif digit == 2:
            dna_sequence += 'G'
        elif digit == 3:
            dna_sequence += 'T'
        else:
            pass
            # 处理非法输入
            # print(f"Invalid digit: {digit}")
    return dna_sequence

def convert_to_DNA0(sequence):
    dna_sequence = ""
    for digit in sequence:
        digit = digit.item()
        if digit == 0:
            dna_sequence += 'A'
        elif digit == 1:
            dna_sequence += 'C'
        elif digit == 2:
            dna_sequence += 'G'
        elif digit == 3:
            dna_sequence += 'T'
        else:
            dna_sequence += '-'
            # 处理非法输入
            # print(f"Invalid digit: {digit}")
    return dna_sequence


def convert_to_DNA_(sequence,predictdata):
    new_predictdata=[]
    prewith_phred=[]
    i=0
    dna_sequence = ""
    for digit in sequence:
        digit = digit.item()
        if digit == 0:
            dna_sequence += 'A'
            new_predictdata.append(predictdata[i])
            prewith_phred.append(predictdata[i])
        elif digit == 1:
            dna_sequence += 'C'
            new_predictdata.append(predictdata[i])
            prewith_phred.append(predictdata[i])
        elif digit == 2:
            dna_sequence += 'G'
            new_predictdata.append(predictdata[i])
            prewith_phred.append(predictdata[i])
        elif digit == 3:
            dna_sequence += 'T'
            new_predictdata.append(predictdata[i])
            prewith_phred.append(predictdata[i])
        else:
            dna_sequence += '-'
            prewith_phred.append(predictdata[i])
        i+=1
            # 处理非法输入
            # print(f"Invalid digit: {digit}")
    return dna_sequence,prewith_phred,new_predictdata

def converttoDNAgetdel(sequence,predictdata,rate = 0.9):
    prewith_phred=[]
    i=0
    dna_sequence_pulsdel = ""
    tinydict = {'0': 'A', '1': 'C', '2': 'G', '3': 'T', '4': '-'}
    for digit in sequence:
        digit = str(digit.item())
        maxpre = np.max(predictdata[i])
        if digit=='4':
            if maxpre < rate:
                sorted_indices = np.argsort(predictdata[i])
                # 找到倒数第二个索引
                second_largest_index = sorted_indices[-2]
                # 找到第二大的数字
                second_largest_value = predictdata[i][second_largest_index]
                dna_sequence_pulsdel += tinydict[str(second_largest_index)]
                prewith_phred.append(second_largest_value)
            # elif maxpre < 0.9:
            #     dna_sequence_pulsdel += '-'
            #     prewith_phred.append(maxpre)

        else:
            dna_sequence_pulsdel += tinydict[digit]
            prewith_phred.append(maxpre)
        i+=1
    return dna_sequence_pulsdel,prewith_phred

def converttoDNAgetdel_tobase(sequence,predictdata,rate = 0.9):
    prewith_phred=[]
    i=0
    dna_sequence_pulsdel = ""
    tinydict = {'0': 'A', '1': 'C', '2': 'G', '3': 'T', '4': '-'}
    for digit in sequence:
        digit = str(digit.item())
        maxpre = np.max(predictdata[i])
        if digit=='4':
            if maxpre < rate:
                # dna_sequence_pulsdel += '-'
                # prewith_phred.append(maxpre)
                sorted_indices = np.argsort(predictdata[i])
                # 找到倒数第二个索引
                second_largest_index = sorted_indices[-2]
                # 找到第二大的数字
                second_largest_value = predictdata[i][second_largest_index]
                dna_sequence_pulsdel += tinydict[str(second_largest_index)]
                prewith_phred.append(second_largest_value)
        else:
            dna_sequence_pulsdel += tinydict[digit]
            prewith_phred.append(maxpre)
        i+=1
    return dna_sequence_pulsdel,prewith_phred

def min_max_normalize_last_dim(tensor):

    # 沿着最后一维计算最大值和最小值
    max_vals, _ = torch.max(tensor, dim=-1, keepdim=True)
    min_vals, _ = torch.min(tensor, dim=-1, keepdim=True)

    # 计算范围
    range_vals = max_vals - min_vals

    # 最大最小标准化
    normalized_tensor = (tensor - min_vals) / range_vals

    return normalized_tensor

# 定义一个标准化层
class Standardize(nn.Module):
    def __init__(self, feature_dim):
        super(Standardize, self).__init__()
        self.feature_dim = feature_dim

    def forward(self, x):
        # mean = x.mean(dim=-1, keepdim=True)
        # std = x.std(dim=-1, keepdim=True)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return (x - mean) / (std + 1e-6)

def getEdit(model,enc_inputs,dec_inputs):
    allpredict, _, _, _ = model(enc_inputs.view(1, enc_inputs.shape[0],-1).cuda(),dec_inputs.view(1, dec_inputs.shape[0],-1).cuda())
    # allpredict1, _, _, _ = model(dec_inputs.view(1, dec_inputs.shape[0],-1),dec_inputs.view(1, dec_inputs.shape[0],-1))
    allpredict = Standardize(tgt_vocab_size)(allpredict)
    # allpredict1 = Standardize(tgt_vocab_size)(allpredict1)
    # predictdata = (allpredict + allpredict1)
    predictdata = allpredict * 2
    predict = predictdata.data.max(1, keepdim=True)[1]

    # predictdata = Standardize(tgt_vocab_size)(predictdata)
    allpredicts = torch.nn.functional.softmax(predictdata, dim=-1)
    # allpredicts = min_max_normalize_last_dim(predictdata)
    # allpredicts = allpredicts / torch.sum(allpredicts, dim=-1, keepdim=True)
    pre = convert_to_DNA(predict)
    prewith_seq,prewith_phred,predictphred = convert_to_DNA_(predict,allpredicts.cpu().detach().numpy())
    # dna_sequence_pulsdel,prewith_phred_pulsdel = converttoDNAgetdel(predict,allpredicts.cpu().detach().numpy(),0)
    # ori_dna_sequence_pulsdel, ori_prewith_phred_pulsdel = dna_sequence_pulsdel,prewith_phred_pulsdel
    # dna_sequence_pulsdel,prewith_phred_pulsdel = converttoDNAgetdel_tobase(predict,allpredicts.cpu().detach().numpy(),0.8)
    dna_sequence_pulsdel,prewith_phred_pulsdel = converttoDNAgetdel(predict,allpredicts.cpu().detach().numpy(),0.4)
    # dna_sequence_pulsdel,prewith_phred_pulsdel = converttoDNAgetdel(predict,allpredicts.cpu().detach().numpy(),0)
    dna_sequence_pulsdel = dna_sequence_pulsdel.rstrip('-')
    prewith_phred_pulsdel = prewith_phred_pulsdel[:len(dna_sequence_pulsdel)]
    prewith_seq = prewith_seq.rstrip('-')
    prewith_phred = prewith_phred[:len(prewith_seq)]
    # dis = Levenshtein.distance(ori, dnatest_sequence)
    # print('ori seq'+str(dis))
    # dis = Levenshtein.distance(ori, dna_sequence_pulsdel)
    # dis = Levenshtein.distance(ori, pre[:len(ori)])
    # if dis>=5:
    #     print(pre)
    #     print(ori)
    # print('ori compare pre edit distance: '+str(dis))
    # return dis,enc_inputs.tolist()
    # return dis,enc_inputs.cpu().numpy()
    # return dis,ori,pre,prewith_seq,predictphred,prewith_phred,dna_sequence_pulsdel,prewith_phred_pulsdel
    return dna_sequence_pulsdel,prewith_phred_pulsdel,prewith_seq,prewith_phred
    # return dis,ori,pre,predict,allpredicts.cpu().detach().numpy()
    # return dis,ori,pre,predict,allpredicts

def getEdit_0215(model,enc_inputs,dec_inputs):
    allpredict, _, _, _ = model(enc_inputs.view(1, enc_inputs.shape[0],-1).cuda(),dec_inputs.view(1, dec_inputs.shape[0],-1).cuda())
    # allpredict1, _, _, _ = model(dec_inputs.view(1, dec_inputs.shape[0],-1),dec_inputs.view(1, dec_inputs.shape[0],-1))
    # allpredict = Standardize(tgt_vocab_size)(allpredict)
    # allpredict1 = Standardize(tgt_vocab_size)(allpredict1)
    # predictdata = (allpredict + allpredict1)
    predictdata = allpredict
    # predictdata = allpredict/0.2
    predict = predictdata.data.max(1, keepdim=True)[1]

    # predictdata = Standardize(tgt_vocab_size)(predictdata)
    allpredicts = torch.nn.functional.softmax(predictdata, dim=-1)
    # allpredicts = min_max_normalize_last_dim(predictdata)
    # allpredicts = allpredicts / torch.sum(allpredicts, dim=-1, keepdim=True)
    pre = convert_to_DNA(predict)
    prewith_seq,prewith_phred,predictphred = convert_to_DNA_(predict,allpredicts.cpu().detach().numpy())
    dna_sequence_pulsdel,prewith_phred_pulsdel = converttoDNAgetdel(predict,allpredicts.cpu().detach().numpy(),0)
    # dna_sequence_pulsdel,prewith_phred_pulsdel = converttoDNAgetdel(predict,allpredicts.cpu().detach().numpy(),0.6)
    # ori_dna_sequence_pulsdel, ori_prewith_phred_pulsdel = dna_sequence_pulsdel,prewith_phred_pulsdel
    # dna_sequence_pulsdel,prewith_phred_pulsdel = converttoDNAgetdel_tobase(predict,allpredicts.cpu().detach().numpy(),0.8)
    # dna_sequence_pulsdel,prewith_phred_pulsdel = converttoDNAgetdel(predict,allpredicts.cpu().detach().numpy(),0.4)
    # dna_sequence_pulsdel,prewith_phred_pulsdel = converttoDNAgetdel(predict,allpredicts.cpu().detach().numpy(),0.8)
    dna_sequence_pulsdel = dna_sequence_pulsdel.rstrip('-')
    prewith_phred_pulsdel = prewith_phred_pulsdel[:len(dna_sequence_pulsdel)]
    prewith_seq = prewith_seq.rstrip('-')
    prewith_phred = prewith_phred[:len(prewith_seq)]
    # dis = Levenshtein.distance(ori, dnatest_sequence)
    # print('ori seq'+str(dis))
    # dis = Levenshtein.distance(ori, dna_sequence_pulsdel)
    # dis = Levenshtein.distance(ori, pre[:len(ori)])
    # if dis>=5:
    #     print(pre)
    #     print(ori)
    # print('ori compare pre edit distance: '+str(dis))
    # return dis,enc_inputs.tolist()
    # return dis,enc_inputs.cpu().numpy()
    # return dis,ori,pre,prewith_seq,predictphred,prewith_phred,dna_sequence_pulsdel,prewith_phred_pulsdel
    return dna_sequence_pulsdel,prewith_phred_pulsdel,prewith_seq,prewith_phred
    # return dis,ori,pre,predict,allpredicts.cpu().detach().numpy()
    # return dis,ori,pre,predict,allpredicts

def getallEdit(model,enc_inputs,dec_inputs):
    allpredict, _, _, _ = model(enc_inputs,dec_inputs)
    # allpredict, _, _, _ = model(enc_inputs.view(1, enc_inputs.shape[0],-1),dec_inputs.view(1, dec_inputs.shape[0],-1))
    allpredict = Standardize(tgt_vocab_size)(allpredict)
    predictdata = allpredict
    predict = predictdata.data.max(1, keepdim=True)[1]
    allpredicts = torch.nn.functional.softmax(predictdata, dim=-1)
    pre = convert_to_DNA(predict)
    prewith_seq,prewith_phred,predictphred = convert_to_DNA_(predict,allpredicts.cpu().detach().numpy())
    dna_sequence_pulsdel,prewith_phred_pulsdel = converttoDNAgetdel(predict,allpredicts.cpu().detach().numpy(),0.5)
    dna_sequence_pulsdel = dna_sequence_pulsdel.rstrip('-')
    prewith_phred_pulsdel = prewith_phred_pulsdel[:len(dna_sequence_pulsdel)]
    return dna_sequence_pulsdel,prewith_phred_pulsdel,prewith_seq,prewith_phred

def readbsalignalignfile(path,allpredict,ratel,rater,dis,dna_sequence_pulsdel,prewith_phred_pulsdel):
    with open(path,'r') as file:
        lines = file.readlines()
    if len(lines)>0:
        preseqwith_indexs = []
        for i in range(len(dna_sequence_pulsdel)):
            if dna_sequence_pulsdel[i]=='-':
                preseqwith_indexs.append(i)
        mismatchdelinsertindexs = []
        delinsertindexs = []
        upnums = 0
        line = lines[0].strip('\n').split('\t')
        mismatch,delnum,insertnum,seqstart = line[-3],line[-2],line[-1],line[-8]
        flag = True
        line = lines[2].strip('\n')
        upnumsindex = []
        # print(line)
        for i in range(len(line)):
            # a = line[i]
            # print(a)
            if line[i]=='-'or line[i] == '*':
                mismatchdelinsertindexs.append(i)
                if  line[i]=='-':
                    delinsertindexs.append(i)
        line = lines[3].strip('\n')
        delindex = [i for i in range(len(line)) if line[i] =='-' and i != 0]
        insertindexs = [x for x in delinsertindexs if x not in delindex]
        mismatchindexs = [x for x in mismatchdelinsertindexs if x not in delinsertindexs]
        testnum = 1
        if int(delnum) >= 1 :
            if seqstart == '2' and dna_sequence_pulsdel[2]=='-':
                flag = False
                delnum,insertnum = int(delnum) - 1, int(insertnum) + 1
                insertindexs = [s+1 for s in insertindexs]
        for i in range(len(delindex)):
            if len(delindex)>i+1:
                if delindex[i+1]==delindex[i]+1:
                    testnum += 1
                    continue
            for misi in range(len(mismatchdelinsertindexs)):
                if mismatchdelinsertindexs[misi]>delindex[i]:
                    mismatchdelinsertindexs[misi] -= testnum
            for misi in range(len(insertindexs)):
                if insertindexs[misi]>delindex[i]:
                    insertindexs[misi] -= testnum
            for misi in range(len(mismatchindexs)):
                if mismatchindexs[misi]>delindex[i]:
                    mismatchindexs[misi] -= testnum
            testnum = 1
        allpredictreal = [i for i in range(len(prewith_phred_pulsdel)) if ratel < prewith_phred_pulsdel[i] <= rater]
        mislen = len(mismatchdelinsertindexs)
        score = []
        upnumsallerrors = [0,0,0,0]
        visited = []
        cansee = []
        for i in allpredictreal:
            if mislen>0:
                if i in visited:
                    continue
                if i in preseqwith_indexs and i not in insertindexs:
                    upnumsallerrors[1] += 1
                    upnums += 1
                    cansee.append(i)
                    upnumsindex.append({i, prewith_phred_pulsdel[i]})
                elif i in mismatchdelinsertindexs:
                    if i in insertindexs:
                        minph = prewith_phred_pulsdel[i]
                        minphi = i
                        iij = i + 1
                        upnumsallerrors[0]+=1
                        while dna_sequence_pulsdel[i] !='-' and iij < len(prewith_phred_pulsdel) and dna_sequence_pulsdel[iij] == dna_sequence_pulsdel[i]:
                            if prewith_phred_pulsdel[iij] <= minph and prewith_phred_pulsdel[iij] <= ratel:
                                minph = prewith_phred_pulsdel[iij]
                                minphi = iij
                            iij += 1
                        upnums += 1
                        upnumsindex.append({minphi, minph})
                        if minphi != i:
                            # score += f"phred: {minph}\n"
                            score.append(minph)
                            upnumsallerrors[0]-=1
                            upnums -= 1
                        # iij = i + 1
                        # while  iij < len(prewith_phred_pulsdel) and dna_sequence_pulsdel[iij] == dna_sequence_pulsdel[i]:
                        #     iij += 1
                        # while iij in visited:
                        #     iij+=1
                        # if iij in preseqwith_indexs:
                        # # if (iij-1 != i and iij in preseqwith_indexs) or iij in preseqwith_indexs:
                        #     visited.append(iij)
                    else:
                        if i in mismatchindexs:
                            upnumsallerrors[2] += 1
                            upnums+=1
                            upnumsindex.append({i,prewith_phred_pulsdel[i]})
                        elif flag:
                            upnumsallerrors[3] += 1
                            upnums += 1
                            upnumsindex.append({i, prewith_phred_pulsdel[i]})

        # if mislen<dis and ratel>=0.949:
        #     upnums+=dis-mislen
        #     upnumsallerrors[1] += dis-mislen
        #     delnum = int(delnum) + dis-mislen
        # print(score)
        # if len(mismatchdelinsertindexs)==0 and len(allpredictreal)>0:
        #     upnums+=len(allpredictreal)
        # if len(allpredictreal)>0 and mislen<dis:
        #     upnums+=dis-mislen
        # for i in allpredictreal:
        #     if i in mismatchdelinsertindexs:
        #         upnums+=1
        # print(mismatch,delnum,insertnum)

        # if upnums != dis:
        #     # if upnumsallerrors[0] == int(insertnum):
        #     #     upnumsallerrors = [int(insertnum), int(delnum), int(mismatch), 0]
        #     if int(insertnum) == len(preseqwith_indexs):
        #         upnums = int(insertnum)+int(delnum)+int(mismatch)+upnumsallerrors[3]
        #         # upnums = dis
        #         upnumsallerrors = [int(insertnum),int(delnum), int(mismatch),upnumsallerrors[3]]
        #     else:
        #         upnums = int(insertnum)+int(mismatch)+upnumsallerrors[3]
        #         # upnums = dis
        #         upnumsallerrors = [int(insertnum),len(preseqwith_indexs)-int(insertnum), int(mismatch)-(len(preseqwith_indexs)-int(insertnum)), upnumsallerrors[3]]
        return mismatch,delnum,insertnum,lines[1].strip('\n'),lines[3].strip('\n'),upnums,upnumsindex,score,upnumsallerrors
    return 0,0,0,'','',0,[],"",[]

def readbsalignalignfileori(path,allpredict,ratel,rater,dis,dna_sequence_pulsdel,prewith_phred_pulsdel):
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
                if  line[i]=='-':
                    delinsertindexs.append(i)
        line = lines[3].strip('\n')
        delindex = [i for i in range(len(line)) if line[i] =='-']
        insertindexs = [x for x in delinsertindexs if x not in delindex]
        mismatchindexs = [x for x in mismatchdelinsertindexs if x not in delinsertindexs]
        # insertindexs = delinsertindexs - delindex
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
        # for i in range(len(delindex)):
        #     if i!=0:
        #         delindex[i]=delindex[i]-1
        allpredictreal = [i for i in range(len(prewith_phred_pulsdel)) if ratel < prewith_phred_pulsdel[i] <= rater]
        mislen = len(mismatchdelinsertindexs)
        score = []
        upnumsallerrors = [0,0,0]
        for i in allpredictreal:
            if mislen>0:
                if i in mismatchdelinsertindexs:
                    if i in insertindexs:
                        minph = prewith_phred_pulsdel[i]
                        minphi = i
                        iij = i + 1
                        upnumsallerrors[0]+=1
                        while dna_sequence_pulsdel[i] != '-' and iij < len(prewith_phred_pulsdel) and dna_sequence_pulsdel[iij] == dna_sequence_pulsdel[i]:
                            if prewith_phred_pulsdel[iij] <= minph and prewith_phred_pulsdel[iij] <= ratel:
                                minph = prewith_phred_pulsdel[iij]
                                minphi = iij
                            iij += 1
                        upnums += 1
                        upnumsindex.append({minphi, minph})
                        if minphi != i:
                            # score += f"phred: {minph}\n"
                            score.append(minph)
                            upnumsallerrors[0]-=1
                            upnums -= 1
                        # print(f"ratel:{ratel} rater:{rater} phred: {minph}")
                    else:
                        if i in mismatchindexs:
                            upnumsallerrors[2] += 1
                        else:
                            upnumsallerrors[1] += 1
                        upnums+=1
                        upnumsindex.append({i,prewith_phred_pulsdel[i]})

        # if mislen<dis and ratel>=0.949:
        #     upnums+=dis-mislen
        #     upnumsallerrors[1] += dis-mislen
        #     delnum = int(delnum) + dis-mislen
        # print(score)
        # if len(mismatchdelinsertindexs)==0 and len(allpredictreal)>0:
        #     upnums+=len(allpredictreal)
        # if len(allpredictreal)>0 and mislen<dis:
        #     upnums+=dis-mislen
        # for i in allpredictreal:
        #     if i in mismatchdelinsertindexs:
        #         upnums+=1
        # print(mismatch,delnum,insertnum)
        return mismatch,delnum,insertnum,lines[1].strip('\n'),lines[3].strip('\n'),upnums,upnumsindex,score,upnumsallerrors
    return 0,0,0,'','',0,[],[],[]

def bsalign_alitest(seq1,seq2,allpredict,ratel,rater,dis,dna_sequence_pulsdel,prewith_phred_pulsdel,dir='files'):
    with open(dir+'/seqstest.fasta', 'w') as file:
        for j, cus in enumerate([seq1,dna_sequence_pulsdel]):
            file.write('>' + str(j) + '\n')
            file.write(str(cus) + '\n')
    shell = '../bsalign-master/bsalign align '+dir+'/seqstest.fasta >' +dir+'/alitest.ali'
    result = subprocess.run(shell, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if dis>=1:
        mismatch, delnum, insertnum, seq1, seq2, upnums, upnumsindex, wrongin, upnumsallerrors = readbsalignalignfileori(
            dir + '/alitest.ali', allpredict, ratel, rater, dis, dna_sequence_pulsdel, prewith_phred_pulsdel)
        # mismatch,delnum,insertnum,seq1,seq2,upnums,upnumsindex,wrongin,upnumsallerrors = readbsalignalignfile(dir+'/alitest.ali',allpredict,ratel,rater,dis,dna_sequence_pulsdel,prewith_phred_pulsdel)
        return mismatch, delnum, insertnum, seq1, seq2, upnums,upnumsindex,wrongin,upnumsallerrors
    else:
        # mismatch,delnum,insertnum,seq1,seq2 = readbsalignalignfile('ali.ali')
        return 0,0,0,seq1,dna_sequence_pulsdel,0,[],[],[]

def train(loader,model,model_path):
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99, weight_decay=0.001)
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)
    print("要加载的模型参数文件不存在！\n开始训练")
    losses = []
    for epoch in range(15):
        for enc_inputs, dec_inputs, dec_outputs, dec_outputsin in loader:
            enc_inputs, dec_inputs, dec_outputs = enc_inputs.cuda(), dec_inputs.cuda(), dec_outputs.cuda()
            # outputs1, enc_self_attns1, dec_self_attns1, dec_enc_attns1 = model(dec_inputs, dec_inputs)
            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
            # outputs = Standardize(tgt_vocab_size)(outputs)
            # outputs = min_max_normalize_last_dim(outputs)
            # outputs = outputs/torch.sum(outputs,dim=-1, keepdim=True)
            # outputs = torch.nn.functional.softmax(outputs, dim=-1)
            # outputs1 = torch.nn.functional.softmax(outputs1, dim=-1)
            # outputs1 = Standardize(tgt_vocab_size)(outputs1)
            # outputs1 = min_max_normalize_last_dim(outputs1)
            # outputs1 = outputs1/torch.sum(outputs1,dim=-1, keepdim=True)
            # loss = criterion(outputs, dec_outputs.view(-1))*2
            # loss1 = criterion(outputs1, dec_outputs.view(-1))
            # loss =  loss1
            loss = criterion(outputs, dec_outputs.view(-1))
            # loss1 = criterion(outputs1, dec_outputs.view(-1))
            # loss = loss + loss1
            # loss = 0.3*loss1 + 0.7*loss2
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
        losses.append(loss.item())
    torch.save(model.state_dict(),model_path)
    return losses


# def testnet(loader,model,lens,testseqs):
#     pre_seqs = []
#     phredwith_scores = []
#     for enc_inputs, dec_inputs, dec_outputs, max_indices in loader:
#         for i in range(len(enc_inputs)):
#             # pre, predict, allpredict, prewith_phred, dna_sequence_pulsdel, prewith_phred_pulsdel = getEdit(model,enc_inputs[i].cuda(),dec_inputs[i].cuda(),dec_outputs[i].cuda())
#             pre, predict, allpredict, prewith_phred, dna_sequence_pulsdel, prewith_phred_pulsdel,oriseqsandphred = getEdit(model,enc_inputs[i],dec_inputs[i],dec_outputs[i])
#             pre_seqs.append(dna_sequence_pulsdel)
#             phredwith_scores.append(prewith_phred_pulsdel)
#     if len(pre_seqs)>sequence_length+10:
#         pre_seqs=testseqs[]
#     with open(dpconsensus_path+'/oriandpreseqswith_.fasta','w') as file:
#         for i in range(len(pre_seqs)):
#             file.write(f">preseq{i}\n{pre_seqs[i]}\n>phredwith_scores{i}\n{phredwith_scores[i]}\n>oripreseq{i}\n{oriseqsandphred[0]}\n>oriphredwith_scores{i}\n{oriseqsandphred[1]}")
#     return pre_seqs,phredwith_scores

def testnet(model,enc_inputstest, dnatest_phreds,testseqs,testphreds):
    pre_seqs = []
    phred_scores = []
    phred_scoresstr = []
    prewith_seqs = []
    phredwith_scores = []
    # print(len(enc_inputstest))
    # dna_sequence_pulsdel, prewith_phred_pulsdel, prewith_seq, prewith_phred = getallEdit(model, enc_inputstest,enc_inputstest)
    for i in range(len(enc_inputstest)):
        # if i%100==0:print(i)
        # dna_sequence_pulsdel, prewith_phred_pulsdel,prewith_seq,prewith_phred = getEdit(model, enc_inputstest[i], dnatest_phreds[i])
        dna_sequence_pulsdel, prewith_phred_pulsdel,prewith_seq,prewith_phred = getEdit_0215(model, enc_inputstest[i], dnatest_phreds[i])
        if len(dna_sequence_pulsdel)>sequence_length*1.1:
            dna_sequence_pulsdel=testseqs[i][0]
            prewith_phred_pulsdel=''
        score = ''
        score1 = []
        for j in range(len(prewith_phred_pulsdel)):
            score+=str(j)+':'+str(np.max(prewith_phred_pulsdel[j]))+' '
            score1.append((np.max(prewith_phred_pulsdel[j])))
        phred_scoresstr.append(score)
        phred_scores.append(score1)
        score = ''
        for j in range(len(prewith_phred)):
            score+=str(j)+':'+str(np.max(prewith_phred[j]))+' '
        phredwith_scores.append(score)
        pre_seqs.append(dna_sequence_pulsdel)
        prewith_seqs.append(prewith_seq)
    return pre_seqs,phred_scores,prewith_seqs,phredwith_scores,phred_scoresstr
