import csv
import os
import re
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 2 为只输出 ERROR 级别日志
import subprocess
import sys
import numpy as np

import Levenshtein
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from embedding import trans_ori_hot, trans_seq_hot, trans_seq_hottest, trans_seq_hottest_nophred, trans_seq_hot_nophred
from MHA import Transformer, train, testnet, train_Adam, train_Adam_parallel,count_bsalign_acc_phred
from config import tgt_vocab_size, n_layers


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # Ensure immediate write to file or screen

    def flush(self):
        for f in self.files:
            f.flush()


# 定义一个标准化层
class Standardize(nn.Module):
    def __init__(self, feature_dim):
        super(Standardize, self).__init__()
        self.feature_dim = feature_dim

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return (x - mean) / (std + 1e-6)


def readbsalignalignfile11(path):
    with open(path, 'r') as file:
        lines = file.readlines()
    if len(lines) > 0:
        line = lines[0].strip('\n').split('\t')
        mismatch, delnum, insertnum = line[-3], line[-2], line[-1]
        return mismatch, delnum, insertnum, lines[1].strip('\n'), lines[3].strip('\n')
    return 0, 0, 0, '', ''


def bsalign_alitest11(seq1, seq2):
    # def bsalign_alitest(seq1,seq2,allpredict,ratel,rater,dis):
    with open('files/seqs.fasta', 'w') as file:
        for j, cus in enumerate([seq1, seq2]):
            file.write('>' + str(j) + '\n')
            file.write(str(cus) + '\n')
    shell = '../bsalign-master/bsalign align files/seqs.fasta > files/ali.ali'
    result = subprocess.run(shell, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    mismatch, delnum, insertnum, seq1, seq2 = readbsalignalignfile11('files/ali.ali')
    # if dis>1 and dis <=10:
    #     mismatch,delnum,insertnum,seq1,seq2,upnums = readbsalignalignfile('files/ali.ali',allpredict,ratel,rater,dis)
    #     return mismatch, delnum, insertnum, seq1, seq2, upnums
    # else:
    #     # mismatch,delnum,insertnum,seq1,seq2 = readbsalignalignfile('ali.ali')
    #     return 0,0,0,seq1,seq2,0
    return mismatch, delnum, insertnum, seq1, seq2


def count_bsalign_acc11_lod(all_consus, all_ori_seqs):
    lens = len(all_consus)
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
        errornums = int(mismatch) + int(delnum) + int(insertnum)
        # if dis - errornums>0:
        # delnum = dis-errornums+int(delnum)
        mismatchnums.append(mismatch)
        delnums.append(delnum)
        insertnums.append(insertnum)
        all_dis.append(dis)
        # if 1 <= dis <= 10:
        if 1 <= dis:
            # inputs_gl0.append(enc)
            all_dis_gl0 += 1
            all_dis_sum += dis
            # all_dis_sum += errornums

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

    with open('./myfiles/bsalign_oriandpreseqs.fasta', 'w') as file:
        for i in range(len(all_ori_seqs)):
            # file.write(f">oriseq{i}\n{all_ori_seqs[i]}\n>preseq{i}   {all_dis[i]}\n{all_consus[i]}\n")
            file.write(
                f">oriseq{i}\n{all_ori_seqs[i]}\n>preseq{i} edit:{all_dis[i]} mis:{mismatchnums[i]} del:{delnums[i]} insert:{insertnums[i]}\n{all_consus[i]}\n")

def count_bsalign_acc11(all_consus,all_ori_seqs,copy_num,data_time):
    start_time = datetime.now()
    lens=len(all_consus)
    select_consus_len = len(all_ori_seqs[0])
    mismatchnums = []
    delnums = []
    insertnums = []
    all_dis = []
    all_dis_gl0 = 0
    all_dis_sum = 0
    all_dis_summ10 = 0
    for i in range(len(all_consus)):

        # te1,te2 = all_ori_seqs[i],all_consus[i]
        # dis = Levenshtein.distance(all_consus[i][:select_consus_len], all_ori_seqs[i])
        dis = Levenshtein.distance(all_consus[i], all_ori_seqs[i])
        mismatch, delnum, insertnum, seq1, seq2 = bsalign_alitest11(all_ori_seqs[i], all_consus[i])
        mismatchnums.append(int(mismatch))
        delnums.append(int(delnum))
        insertnums.append(int(insertnum))
        all_dis.append(dis)
        # if dis >= 1 and dis < 10:
        if dis >= 1:
            # inputs_gl0.append(enc)
            all_dis_gl0 += 1
            if dis < 5:
                all_dis_sum += dis
            else:
                all_dis_summ10 += dis
                # print(f"{all_ori_seqs[i]} dis:{dis}\n{all_consus[i]}\n")
        # inputs.append(enc)
        # all_dis+=getEdit(model,enc_inputs[i],dec_outputs[i])
    # print('dis > 0 number:' + str(all_dis_gl0) + '     ' + str(all_dis_gl0 / lens))
    # print('average dis:' + str(all_dis_sum / lens))
    # print('when dis >= 1 average dis:' + str(all_dis_sum / all_dis_gl0))
    # print('recovery rate:' + str(1 - all_dis_sum / lens / len(all_ori_seqs[0])))

    tillnow = datetime.now()
    print(f"mismatch数量:{sum(mismatchnums)} del数量:{sum(delnums)} insert数量:{sum(insertnums)} ")
    print('发生错误的序列数量:' + str(all_dis_gl0) + ' 发生错误数量占所有数量： ' + str(all_dis_gl0 / lens))
    print('发生错误的碱基数量 :' + str(all_dis_sum) + ' >=5错误：'+str(all_dis_summ10) + ' 总错误：'+str(all_dis_summ10+all_dis_sum))
    print('每条序列碱基数量：' + str(len(all_ori_seqs[0])))
    print('总错误平均编辑距离:' + str((all_dis_summ10+all_dis_sum) / lens) + '<5平均编辑距离:' + str(all_dis_sum / lens))
    if all_dis_gl0>0:
        print('when dis > 1 平均编辑距离:' + str(all_dis_summ10+all_dis_sum / all_dis_gl0))
    print('recovery rate:' + str(1 - (all_dis_summ10+all_dis_sum) / lens / len(all_ori_seqs[0])))

    with open('./myfiles/bsalign_oriandpreseqs.fasta','w') as file:
        for i in range(len(all_ori_seqs)):
            # file.write(f">oriseq{i}\n{all_ori_seqs[i]}\n>preseq{i}   {all_dis[i]}\n{all_consus[i]}\n")
            file.write(f">oriseq{i}\n{all_ori_seqs[i]}\n>preseq{i} edit:{all_dis[i]} mis:{mismatchnums[i]} del:{delnums[i]} insert:{insertnums[i]}\n"
                       f"{all_consus[i]}\n")

    seq_len = len(all_ori_seqs[0])
    data = [
        # ('', 'rec rate', 'edit error rate', 'error rate', 'success rate', 'indels', 'error base num', 'error seq num'),
        (f'bsalign{copy_num}',1 - (all_dis_summ10+all_dis_sum) / lens / seq_len, (all_dis_summ10+all_dis_sum) / lens / seq_len,all_dis_gl0 / lens,1-all_dis_gl0 / lens,
         f"{sum(mismatchnums) }:{sum(delnums) }:{sum(insertnums) }", f"{all_dis_summ10+all_dis_sum}",  f"{all_dis_gl0}"),
    ]
    print(data)
    data = [
        # ('', 'rec rate', 'edit error rate', 'error rate', 'success rate', 'sub', 'del', 'insert', 'indels',
        #  'error seq num', 'error base num', 'time'),
        (f"bsalign{copy_num}",1 - (all_dis_summ10+all_dis_sum) / lens / len(all_ori_seqs[0]), (all_dis_summ10+all_dis_sum) / lens / len(all_ori_seqs[0]), all_dis_gl0 / lens, 1 - all_dis_gl0 / lens
         , sum(mismatchnums) / lens /  len(all_ori_seqs[0]), sum(delnums) / lens /  len(all_ori_seqs[0]), sum(insertnums) / lens /  len(all_ori_seqs[0]),
         f"{sum(mismatchnums) }:{sum(delnums) }:{sum(insertnums) }", all_dis_gl0, all_dis_summ10+all_dis_sum, tillnow - start_time+data_time),
    ]
    # with open('files/trellis_ldpc_1120.csv', 'a', encoding='utf8', newline='') as f:
    # with open('derrick_model/derrick20_bs_dp_3_10.csv', 'a', encoding='utf8', newline='') as f:

    return data


class MyDataSet(Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs, maxindexs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs
        self.maxindexs = maxindexs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx], self.maxindexs[idx]


def saveseqs(seqs, path):
    with open(path, 'w') as file:
        for j in range(len(seqs)):
            file.write('>' + str(j) + '\n')
            file.write(str(seqs[j]).strip('\n') + '\n')
    # print(j)



def saveseqs_forbma(path,oriseqs,dnaseqs):
    with open(path,'w') as f:
        for i in range(len(oriseqs)):
            f.write(f"{oriseqs[i]}\n****\n")
            for j in range(len(dnaseqs[i])):
                f.write(f"{dnaseqs[i][j]}\n")
            f.write("\n\n")

def getAllphred_quality_secondseqs(path):
    all_ori_seqs = []
    allSeqsAndQuas = []  # 存放变异的DNA序列
    all_seqs = []
    all_quas = []
    with open(path, 'r') as file:
        lines = file.readlines()
    flag = 0
    seqs = []
    for i in range(len(lines)):
        if flag == 1:
            if len(seqs) >= 5:  # 训练时，变异DNA条数不足5条的类会丢掉
                allSeqsAndQuas.append(seqs)
            elif len(all_ori_seqs) > 0:
                all_ori_seqs.pop()
            all_ori_seqs.append(lines[i].strip('\n'))
            seqs = []
            flag = 0
            continue
        if lines[i].startswith('>ori'):
            flag = 1
        elif not (lines[i].startswith('>seq') or lines[i].startswith('>qua')):
            # seqs.append(lines[i][:200].strip('\n'))
            seqs.append(lines[i].strip('\n'))
    if len(seqs) >= 5:
        allSeqsAndQuas.append(seqs)
    for i in range(len(allSeqsAndQuas)):
        # if len(allSeqsAndQuas[i]>5)
        # seqs_quas = allSeqsAndQuas[i]
        # reads = seqs_quas[::2]
        # quas = seqs_quas[1::2]
        # quasScore = []
        # for qua in quas:
        #     quasScore.append(getphred_quality(qua))
        all_seqs.append(allSeqsAndQuas[i])
        # all_quas.append(quasScore)
    # saveseqs(all_seqs, 'files/seq300all_seqs.fasta')
    # saveseqs(all_quas, 'files/seq300all_quas.fasta')
    # saveseqs(all_ori_seqs, 'files/seq300all_ori_seqs.fasta')
    return all_ori_seqs, all_seqs


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


def digitseqs_convert_to_DNA(digit_seqs):
    dna_sequences = []
    for digit_seq in digit_seqs:
        dna_sequence = convert_to_DNA(digit_seq)
        dna_sequences.append(dna_sequence)
    return dna_sequences


def train(rank, world_size, lr, batch_size, train_dataset, model_path):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12454'
    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # 定义设备
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # 创建模型并移动到GPU
    model = Transformer().to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)

    # 使用 DistributedSampler
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler)

    # 训练模型


    #
    # criterion = nn.CrossEntropyLoss()
    # # optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99, weight_decay=0.001)
    # optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)
    # print("要加载的模型参数文件不存在！\n开始训练")
    # losses = []
    # for epoch in range(30):
    #     for enc_inputs, dec_inputs, dec_outputs, dec_outputsin in train_loader:
    #         # enc_inputs, dec_inputs, dec_outputs = enc_inputs.cuda(), dec_inputs.cuda(), dec_outputs.cuda()
    #         enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(device), dec_inputs.to(device), dec_outputs.to(device)
    #         outputs1, enc_self_attns1, dec_self_attns1, dec_enc_attns1 = model(dec_inputs, dec_inputs)
    #         outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
    #         loss = criterion(outputs, dec_outputs.view(-1))
    #         loss1 = criterion(outputs1, dec_outputs.view(-1))
    #         loss = loss + loss1
    #         # loss = 0.3*loss1 + 0.7*loss2
    #         optimizer.zero_grad()
    #         loss.backward(retain_graph=True)
    #         optimizer.step()
    #     if rank == 0:
    #         print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
    #     losses.append(loss.item())
    # # torch.save(model.state_dict(), 'testzzz.pth')
    # if rank == 0:
    #     # torch.save(model.state_dict(), model_path)
    #     torch.save({'model_state_dict': model.module.state_dict(),
    #                 'optimizer_state_dict': optimizer.state_dict(),
    #             }, model_path)


    criterion = nn.CrossEntropyLoss()
    weight_decay = 1e-4
    if rank == 0:  # rank 好像是gpu的编号
        print("要加载的模型参数文件不存在！\n开始训练")
        print("使用Adam!!")
        print("lr = %f, weight_decay = %f" % (lr, weight_decay))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # 结合transformer_master的代码，这里还可以加上scheduler和warmup，以实现动态的调整学习率
    """
    # ReduceLROnPlateau 是一个学习率调度器，它根据验证集损失等监控指标在模型不再改善时动态降低学习率。
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 verbose=True,
                                                 factor=factor,
                                                 patience=patience)"""
    min_epoch_loss = 10.0
    for epoch in range(30):
        model.train()
        train_sampler.set_epoch(epoch)  # 每个 epoch 设置随机种子，确保数据顺序不同
        epoch_loss = 0.0
        seqs_correct, edit_dis_num = 0, 0
        seqs_all, base_all = 0, 0
        for enc_inputs, dec_inputs, dec_outputs, dec_outputsin in train_loader:  # enc_inputs 与 dec_inputs 是一样的选enc_inputs就行
            # outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs)
            # enc_inputs, dec_inputs, dec_outputs = enc_inputs.cuda(), dec_inputs.cuda(), dec_outputs.cuda()
            # outputs, _, _, _ = model(enc_inputs)
            enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(device), dec_inputs.to(device), dec_outputs.to(device)
            outputs, _, _, _ = model(enc_inputs,dec_inputs)
            # outputs, _, _, _ = model(enc_inputs)
            outputs1, _, _, _ = model(dec_inputs)

            # 根据最新的学习情况，下面不应该用softmax，因为交叉熵中有log_softmax的操作。故在训练时的输出不用。但在实际使用，做预测时，还是要softmax的。20240911
            # outputs = torch.nn.functional.softmax(outputs, dim=-1)  # 不用softmax恢复率会更高，红玫说
            loss = criterion(outputs, dec_outputs.view(-1))
            loss1 = criterion(outputs1, dec_outputs.view(-1))
            loss += loss1
            seqs_num = len(dec_outputs)  # dec_outputs就是原始DNA序列经过one-hot编码后的数据
            predictdata = torch.nn.functional.softmax(outputs, dim=-1)  # 先做softmax再做提取才对
            predict = predictdata.data.max(1, keepdim=True)[1]
            predict_digit_seqs = predict.reshape(seqs_num, -1)
            dna_seqs_pre = digitseqs_convert_to_DNA(predict_digit_seqs)
            dna_seqs_ori = digitseqs_convert_to_DNA(dec_outputs)
            # 统计块正确数量和碱基正确数量
            for i in range(len(dna_seqs_pre)):
                base_all += len(dna_seqs_pre[i])
                if dna_seqs_pre[i] == dna_seqs_ori[i]:
                    seqs_correct += 1
                else:  # 完全一样就不用算编辑距离了
                    edit_dis_num += Levenshtein.distance(dna_seqs_pre[i], dna_seqs_ori[i])
            seqs_all += len(dna_seqs_pre)

            optimizer.zero_grad()
            # loss.backward(retain_graph=True)   # 可以考虑换成loss.backward() ，GPT说可以减少内存占用
            loss.backward()
            # 先放着，后面再加
            """
            # 梯度裁剪：防止梯度爆炸，确保梯度值不超过设定的阈值 clip，eg: clip = 0.9 or 0.1 等等。
            # 如果梯度的范数大于 clip，那么所有的梯度将按比例缩小，使其范数等于 clip；如果梯度的范数小于或等于 clip，则不进行任何修改
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            """
            optimizer.step()

            epoch_loss += loss.item()
        avg_epoch_loss = epoch_loss / len(train_loader)
        if rank == 0:
            print(f'Rank {rank}, Epoch {epoch + 1}, Loss: {avg_epoch_loss:.6f}')
        if rank == 0 and epoch > 15 and avg_epoch_loss < min_epoch_loss:  # 只保存效果最好的
            # torch.save(model.state_dict(), model_path)
            min_epoch_loss = avg_epoch_loss
            seqs_recovery_rate = seqs_correct / seqs_all
            edit_dis_rate = edit_dis_num / base_all
            print(f'seqs_recovery_rate: {seqs_recovery_rate:.6f}, edit_dis_rate: {edit_dis_rate:.6f}')
            # print("找到更小的avg_epoch_loss，保存模型，min_epoch_loss = %.6f" % min_epoch_loss)
            torch.save({
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, model_path)
    # if rank == 0:
    #     seqs_recovery_rate = seqs_correct / seqs_all
    #     edit_dis_rate = edit_dis_num / base_all
    #     print(f'Rank {rank}, Epoch {epoch + 1}, Loss: {avg_epoch_loss:.6f}')
    #     print(
    #         f'Rank {rank}, Epoch {epoch + 1}, seqs_recovery_rate: {seqs_recovery_rate:.6f}, edit_dis_rate: {edit_dis_rate:.6f}')
    #     # torch.save(model.state_dict(), model_path)
    #     torch.save({
    #         'model_state_dict': model.module.state_dict(),
    #         'optimizer_state_dict': optimizer.state_dict(),
    #     }, model_path)
    # 还应该检查训练过程中，块错误率
    pass
    # # 清理进程组
    dist.destroy_process_group()


def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Transformer().to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)  # weights_only=True 加上试试
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def getphred_quality(qualityScore):
    phred_qualitys = []
    # sss = seq[110:120]
    for index, i in enumerate(qualityScore):
        phred_quality = ord(i) - 33  # '@'的ASCII码是64，FASTQ使用的是Phred+33编码
        phred_qualitys.append(phred_quality)
    return phred_qualitys


def getoriandallseqs(path):
    ori_dna_sequences, all_seqs, all_quas = [], [], []
    with open(path, 'r') as f:
        lines = f.readlines()
    maxl = len(lines)
    i = 0
    while i < maxl:
        ori_dna_sequences.append(lines[i].strip('\n'))
        i += 2
        seqs = []
        quas = []
        while i < maxl and re.match(r"[ACGT]", lines[i][0]):
            seqs.append(lines[i].strip('\n').strip('0'))
            i += 1
            quas.append(getphred_quality(lines[i].strip('\n')))
            i += 1
        i += 2
        if len(seqs) == 0:
            ori_dna_sequences.pop()
            continue
            # print(i)
        all_seqs.append(seqs)
        all_quas.append(quas)
    return ori_dna_sequences, all_seqs, all_quas

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
    # saveseqs(all_seqs,'files/seq300all_seqs.fasta')
    # saveseqs(all_quas,'files/seq300all_quas.fasta')
    # saveseqs(all_ori_seqs,'files/seq300all_ori_seqs.fasta')
    return all_ori_seqs,all_seqs,all_quas

if __name__ == "__main__":
    # Open the file in write mode
    output_file = open('output_screen.txt', 'w')

    # Create a Tee object that writes to both stdout and the file
    tee = Tee(sys.stdout, output_file)

    # Replace sys.stdout with the Tee object
    sys.stdout = tee

    for si in range(10, 11):
        # 1.读取训练和测试数据
        copy_num = si
        print("copy_num =", copy_num)

        # 数据准备
        # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        # train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        datanow = datetime.now()
        # ori_dna_sequences, all_seqs, all_quas = getAllphred_quality('/home3/wjl/Pycharm_service/data/seqsforphread300g30_dll_nano.fasta')
        # ori_dna_sequences, all_seqs, all_quas = getoriandallseqs(
        #     f'/home1/hongmei/00work_files/0000/0ifirstCompare/CompareTest/data_handle/derrick_cluster_fix/derrick_bma_data_le9_no0_phred.txt')
        # ori_dna_sequences, all_seqs, all_quas = getoriandallseqs(
        #     f'/home1/hongmei/00work_files/0000/0ifirstCompare/CompareTest/data_handle/derrick_cluster_fix/derrick_bma_data_le11_no0_phred.txt')
        # ori_dna_sequences, all_seqs, all_quas = getoriandallseqs(
        #     f'/home1/hongmei/00work_files/0000/0ifirstCompare/CompareTest/data_handle/id20_indexcluster_fix/id20_bma_data_le7_no0_phred.txt')
        # ori_dna_sequences, all_seqs, all_quas = getoriandallseqs(
        #     f'/home1/hongmei/00work_files/0000/0ifirstCompare/CompareTest/data_handle/derrick_cluster_fix_1000/derrick_bma_data_le{si}_no0_phred.txt')
        # ori_dna_sequences, all_seqs = getAllphred_quality_secondseqs('myfiles/merge_DNA_ori_error_0514_126739.fasta')
        # ori_dna_sequences, all_seqs, all_quas = getoriandallseqs(
        #     f'/home1/hongmei/00work_files/0000/0ifirstCompare/CompareTest/data_handle/derrick_cluster_fix/derrick_bma_data_le{si}_no0_phred.txt')

        # ori_dna_sequences, all_seqs, all_quas = getoriandallseqs(
        #     f'/home1/hongmei/00work_files/0000/0ifirstCompare/CompareTest/data_handle/derrick_cluster_fix_all/derrick_bma_data_le{si}_no0_phred.txt')
        ori_dna_sequences, all_seqs, all_quas = getoriandallseqs(f'./files/data{si}_phred.txt')
        # ori_dna_sequences, all_seqs, all_quas = getoriandallseqs(f'/home2/hm/datasets/derrick/data{si}_phred_rm02.txt')

        # ori_dna_sequences, all_seqs, all_quas = getoriandallseqs(
        #     f'/home1/hongmei/00work_files/0000/0ifirstCompare/CompareTest/data_handle/derrick_cluster_fix_1000/derrick_bma_data_le{si}_no0_phred.txt')
        # choose_seqsnums = 116739  # 训练集中DNA序列条数  # 这个分类比上面的要更合理
        # select_nums = 10000  # 验证集中DNA序列条数

        print("ori_dna_sequences = %d" % (len(ori_dna_sequences)))
        choose_seqsnums = int(len(ori_dna_sequences)*0.8) # 训练集中DNA序列条数
        select_nums = len(ori_dna_sequences)-choose_seqsnums  # 验证集中DNA序列条数
        # choose_seqsnums = 2073 # 训练集中DNA序列条数
        # select_nums = 2073  # 验证集中DNA序列条数
        choose_seqsnums = 100  # 训练集中DNA序列条数，测试用
        select_nums = 100  # 验证集中DNA序列条数

        print("choose_seqsnums = %d ; select_nums = %d" % (choose_seqsnums, select_nums))
        # testoriseqs = ori_dna_sequences[choose_seqsnums:choose_seqsnums + select_nums]
        # testseqs = all_seqs[choose_seqsnums:choose_seqsnums + select_nums]
        testoriseqs = ori_dna_sequences[choose_seqsnums:choose_seqsnums + select_nums]
        testseqs, testseqssquas = all_seqs[choose_seqsnums:choose_seqsnums + select_nums], all_quas[
                                                                                           choose_seqsnums:choose_seqsnums + select_nums]
        print(len(testoriseqs), len(testseqs))
        #
        # with open(f'/home2/hm/datasets/derrick/data{si}.txt', 'w') as f:
        #     for i in range(len(testoriseqs)):
        #         f.write(f"{testoriseqs[i]}\n****\n")
        #         for j in range(len(testseqs[i])):
        #             f.write(f"{testseqs[i][j]}\n")
        #         f.write("\n\n")
        # break


        # 2.对测试数据进行数据预处理
        oritest_seqs = trans_ori_hot(testoriseqs)
        # dnatest_sequences, all_consus = trans_seq_hottest_nophred(testseqs, testoriseqs, copy_num)
        dnatest_sequences, dnatest_sequences_phreds, all_consus, all_bsalign_quas = trans_seq_hottest(testseqs, testoriseqs,
                                                                                                      testseqssquas,
                                                                                                      copy_num)
        max_indices = np.argmax(dnatest_sequences, axis=-1)
        max_indices = torch.Tensor(max_indices)
        # enc_inputstest, dec_inputstest, dec_outputstest = torch.Tensor(dnatest_sequences), torch.Tensor(
        #     dnatest_sequences), torch.LongTensor(oritest_seqs)
        enc_inputstest, dec_inputstest, dec_outputstest = torch.Tensor(dnatest_sequences), torch.Tensor(
            dnatest_sequences_phreds), torch.LongTensor(oritest_seqs)

        enc_inputstest, dec_inputstest = Standardize(tgt_vocab_size)(enc_inputstest), Standardize(tgt_vocab_size)( dec_inputstest)

        handletime = datetime.now() - datanow
        ti = 0
        print(f"-----------------------------------bsalign--------------------------------\n")
        csv_data2 = count_bsalign_acc_phred(all_consus, testoriseqs,all_bsalign_quas, copy_num)
        for i in range(ti,ti+1):
            batch_size = 64
            print('第 %d 次实验开始！！' % i)
            # r = -2 * np.random.rand() - 3
            # lr = 10 ** r  # 学习率取对数坐标，在10^-4 到 10^-1 之间比较好
            # lr = 0.001
            lr = 0.00011546683198243592
            print("lr = %.8f" % lr)
            print("n_layers = %d" % n_layers)
            # model_path = './models/derrick/model_encinput_decinput_copy9_epoch50_loss_0.001864.pth'
            # model_path = './models/models_synth/model_0626_n_layers_2_305_l2_synth_adam_lr_0.0016587676466313169.pth'
            # model_path = f'./models/derrick/model_encinput_decinput_copy{copy_num}_epoch30_loss_.pth'
            # model_path = '/home3/wjl/Pycharm_service/deeplearning_align_24_0506/models/models_nano/model_1130_n_layers_2_308_copy_num_11_l2_nano_adam_lr_0.001.pth'
            # model_path = "./models/derrick/model_encinput_copy9_epoch30_loss.pth"
            # model_path = './models/derrick/model_decinput_copy9_epoch50_loss_0.001758.pth'
            # model_path = './models/models_synth/model_0626_n_layers_' + str(n_layers) + '_?_l2_synth_adam_lr_' + str(
            #     lr) + '.pth'
            # model_decinput_copy9_epoch50_loss_0.001758.pth
            # model_path = f'./models/derrick1000/model_10000train_enc_dec_copy{copy_num}_epoch20_seq{ti}_.pth'
            # model_path = f'./models/derrick_seqcluster_1000/model_10000train_encdecstdb8_copy{copy_num}_epoch30_seq{ti}_.pth'
            # model_path = f'./models/derrick_seqcluster_1000_b16/model_10000train_encdecstdb16_copy{copy_num}_epoch30_.pth'
            # model_path = f'./models/derrick_seqcluster_1000_b8/model_10000train_encdecstdb8_copy{copy_num}_epoch30_121703.pth'
            # model_path = f'/home2/hm/models/nano_model_Srinivasavardhan/model_{copy_num}.pth'
            # model_path = f'./models/derrick_seqcluster_1000_b16/testb32.path'
            # model_path = f'./models/derrick1000/model_10000train_enc_dec_copy9_epoch20_std_seq0_0.148.pth'
            # model_path = './models/derrick1000/model_10000train_enc_copy9_epoch20_seq_0.192_.pth'

            # 尝试训练模型测试derrick数据，效果不好
            # model_path = f'/home2/hm/models/derrick250211/model_{copy_num}.pth'
            # 尝试使用旧模型测试derrick数据
            model_path = f'../models/model_10000train_encdecstdb64_copy{copy_num}_epoch30_.pth'
            # model_path = f'/home2/hm/models/nano_model_old/derrick_seqcluster_10000_b64/model_10000train_encdecstdb64_copy19_epoch30_.pth'
            # model_path = f'/home2/hm/models/nano_model_old/derrick_seqcluster_10000_b64/model_10000train_encdecstdb64_copy19_epoch30_.pth'
            if os.path.exists(model_path):
                print("找到模型" + model_path)

            else:
                # 开始训练

                choose_seqsnums = 10000  # 训练集中DNA序列条数
                # 4.对训练数据进行数据预处理，训练模型
                trainoriseqs = ori_dna_sequences[:choose_seqsnums]
                # trainseqs = all_seqs[:choose_seqsnums]
                trainseqs, trainseqssquas = all_seqs[:choose_seqsnums], all_quas[:choose_seqsnums]
                # ori_seqs, dna_sequences = trans_seq_hot_nophred(trainseqs, trainoriseqs, copy_num)
                # enc_inputs, dec_inputs, dec_outputs = torch.Tensor(dna_sequences), torch.Tensor(dna_sequences), torch.LongTensor(ori_seqs)
                ori_seqs, dna_sequences, dna_sequences_phreds = trans_seq_hot(trainseqs, trainoriseqs, trainseqssquas,copy_num)
                enc_inputs, dec_inputs, dec_outputs = torch.Tensor(dna_sequences), torch.Tensor(
                    dna_sequences_phreds), torch.LongTensor(ori_seqs)
                enc_inputs, dec_inputs = Standardize(tgt_vocab_size)(enc_inputs), Standardize(tgt_vocab_size)(dec_inputs)
                train_dataset = MyDataSet(enc_inputs, dec_inputs, dec_outputs, dec_outputs)
                world_size = torch.cuda.device_count()
                mp.spawn(train, args=(world_size, lr, batch_size, train_dataset, model_path), nprocs=world_size,
                         join=True)  # 利用多GPU训练模型

            # 5.预测
            loadertest = DataLoader(MyDataSet(enc_inputstest, dec_inputstest, dec_outputstest, max_indices), 128,True)
            print('-----------------------------------deep learning--------------------------------')
            # 加载模型
            model = load_model(model_path)
            model.eval()  # 模型设置为评估模式
            print("模型参数已成功加载！" + model_path)
            # testnet(loadertest, model, len(enc_inputstest))
            csv_data1 = testnet(loadertest, model, len(enc_inputstest),handletime)

            print(f"oritest_seqs_num: {len(oritest_seqs)} copynum:{copy_num}")

            print('done')

        # Remember to close the file when done
        output_file.close()

        # Optionally, you can reset sys.stdout to its original value if needed
        sys.stdout = sys.__stdout__
        # count_bsalign_acc11(all_consus, testoriseqs)
        # csv_data2 = count_bsalign_acc11(all_consus, testoriseqs, copy_num, handletime)

        # with open('./models/derrick_seqcluster_10000_b64/derrick_bs_dp_phred_1220_remainerror.csv', 'a', encoding='utf8', newline='') as f:
        # # with open('./models/derrick/test.csv', 'a', encoding='utf8', newline='') as f:
        #     writer = csv.writer(f)  # csv.writer()中可以传一个文件对象
        #     for line in csv_data2:  # 该data既可以是列表嵌套列表的数据类型也可以是列表嵌套元组的数据类型
        #         writer.writerow(line)
        #     for line in csv_data1:  # 该data既可以是列表嵌套列表的数据类型也可以是列表嵌套元组的数据类型
        #         writer.writerow(line)