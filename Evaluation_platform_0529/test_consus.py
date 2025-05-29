import copy
import os

import Levenshtein
import torch
import torch.utils.data as Data
# from dpconsensus.MHA import Transformer
# from dpconsensus.config import tgt_vocab_size, dpconsensus_path
# from dpconsensus.embedding import trans_seq_hottest_nophred
# from dpconsensus.train_test import testnet
from Evaluation_platform.dpconsensus.embedding import trans_seq_hottest_nophred, trans_seq_hottest, \
    trans_seq_hot_nophred
from Evaluation_platform.dpconsensus.MHA import Transformer
from Evaluation_platform.dpconsensus.train_test import testnet, train
from Evaluation_platform.dpconsensus.config import tgt_vocab_size, dpconsensus_path
from torch import nn


class MyDataSet(Data.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs,maxindexs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs
        self.maxindexs = maxindexs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx], self.maxindexs[idx]

# 定义一个标准化层
class Standardize(nn.Module):
    def __init__(self, feature_dim):
        super(Standardize, self).__init__()
        self.feature_dim = feature_dim

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return (x - mean) / (std + 1e-6)

def load_model(model_path,n_layers):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Transformer(n_layers).cuda()
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)  # weights_only=True 加上试试
    print(f"checkpoint.keys():{checkpoint.keys()}")
    # print(f"checkpoint['model_state_dict'].keys():{checkpoint['model_state_dict'].keys}")

    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def getconsensus_ori(oriseqs,testseqs,testphreds,copy_num = 5,modelpath=dpconsensus_path+'/modelsecond.pth'):
    # 1.读取训练和测试数据
    torch.cuda.set_device(0)
    if (modelpath.find('modelsecond'))>=0:
        n_layers = 4
    else:
        n_layers = 2

    # print(len(testoriseqs),len(testseqs),len(testseqssquas))

    # dna_sequences,dnatest_sequences,dnatest_phreds,bs_consus_ori= trans_seq_hottest(testseqs,testphreds,copy_num)
    dnatest_sequences,all_consus,allbs_quas,all_consusori,selectseqs,bsalign_aliseqs = trans_seq_hottest_nophred(testseqs,copy_num)
    # max_indices = np.argmax(dnatest_sequences, axis=-1)
    # max_indices = torch.Tensor(max_indices)
    enc_inputstest, dec_inputstest = torch.Tensor(dnatest_sequences), torch.Tensor(dnatest_sequences)

    enc_inputstest, dec_inputstest = Standardize(tgt_vocab_size)(enc_inputstest), Standardize(tgt_vocab_size)(dec_inputstest)
    # model = Transformer(n_layers).cuda()
    # # model = Transformer()
    # # model = Transformer().cuda()
    # model.load_state_dict(torch.load(modelpath))
    model = load_model(modelpath, n_layers)

    print("模型参数已成功加载！")

    # loadertest = Data.DataLoader(MyDataSet(enc_inputstest, dec_inputstest, dec_inputstest, max_indices),1, True)
    print('-----------------------------------deep learning--------------------------------')
    pre_seqs,phred_scores,prewith_seqs,phredwith_scores  = testnet(model,enc_inputstest, dec_inputstest,testseqs,testphreds)
    # pre_seqs,phredwith_scores = testnet(loadertest,model,len(enc_inputstest),testseqs)
    with open(dpconsensus_path + '/oriandpreseqswith_.fasta', 'w') as file:
        for i in range(len(oriseqs)):
            dis = Levenshtein.distance(oriseqs[i],pre_seqs[i])
            # disbs = Levenshtein.distance(oriseqs[i],all_consus[i])
            # disbsori = Levenshtein.distance(oriseqs[i],all_consusori[i])
            # file.write(f">oriseq{i}\n{oriseqs[i]}\n>bs_oriseq{i} dis:{disbsori}\n{all_consusori[i]}\n>bs_seq{i} dis:{disbs}\n{all_consus[i]}\n>preseq{i} dis:{dis}\n{pre_seqs[i]}\n>prewith_seq{i}\n{prewith_seqs[i]}\n"
            #            f">phredwith_scores{i}\n{phredwith_scores[i]}\n>selectseqs{i}\n{selectseqs[i]}\n>bsalign_aliseqs{i}\n{bsalign_aliseqs[i]}\n")
            file.write(
                f">oriseq{i}\n{oriseqs[i]}  dis:{dis}\n{pre_seqs[i]}\n>phred_scores{i}\n{phred_scores[i]}\n")

            # file.write(f">oriseq{i}\n{oriseqs[i]}\n>preseq{i} dis:{dis}\n{pre_seqs[i]}\n>prewith_seq{i}\n{prewith_seqs[i]}\n>phredwith_scores{i}\n{phredwith_scores[i]}\n"
            #            f">bsalign_aliseqs{i}\n{bsalign_aliseqs[i]}\n>bs_seq{i} dis:{disbs}\n{all_consus[i]}\n")
    # with open(dpconsensus_path + '/cluseterseqs.fasta', 'w') as file:
    #     for i in range(len(oriseqs)):
    #         file.write(f">oriseq{i}\n{oriseqs[i]}\n>preseq{i}\n")
    #         for j in range(len(testseqs[i])):
    #             file.write(f"{testseqs[i][j]}\n")
    return pre_seqs,phred_scores,all_consus,allbs_quas


def getconsensus(oriseqs1,testseqs1,testphreds1,copy_num = 5,modelpath=dpconsensus_path+'/modelsecond.pth'):
    # 1.读取训练和测试数据
    torch.cuda.set_device(0)
    if (modelpath.find('modelsecond'))>=0:
        n_layers = 4
    else:
        n_layers = 6
        # n_layers = 6

    # modelpath = dpconsensus_path+'/modelbadread_250216_0.pth'
    #
    choose_seqsnums = 3000
    select_nums = 1000
    # oriseqs = copy.deepcopy(oriseqs1[choose_seqsnums:choose_seqsnums+select_nums])
    # testseqs = copy.deepcopy(testseqs1[choose_seqsnums:choose_seqsnums+select_nums])
    # print(f"oriseqs:{len(oriseqs)},testseqs:{len(testseqs)}")


    oriseqs = oriseqs1
    testseqs = testseqs1

    dnatest_sequences,all_consus,allbs_quas,all_consusori,selectseqs,bsalign_aliseqs = trans_seq_hottest_nophred(testseqs,copy_num)
    # max_indices = np.argmax(dnatest_sequences, axis=-1)
    # max_indices = torch.Tensor(max_indices)
    enc_inputstest, dec_inputstest = torch.Tensor(dnatest_sequences), torch.Tensor(dnatest_sequences)

    enc_inputstest, dec_inputstest = Standardize(tgt_vocab_size)(enc_inputstest), Standardize(tgt_vocab_size)(dec_inputstest)
    # model = Transformer(n_layers).cuda()
    # # model = Transformer()
    # model = Transformer().cuda()
    # model.load_state_dict(torch.load(modelpath))
    if os.path.exists(modelpath):
        print("找到模型" + modelpath)
        model = Transformer(n_layers).cuda()
        model.load_state_dict(torch.load(modelpath))
        # model = load_model(modelpath, n_layers)
    else:
        print("找不到模型" + modelpath)
        trainoriseqs = copy.deepcopy(testseqs1[:choose_seqsnums])
        trainseqs, trainseqssquas = copy.deepcopy(testseqs1[:choose_seqsnums]), copy.deepcopy(testseqs1[:choose_seqsnums])

        ori_seqs,dna_sequences = trans_seq_hot_nophred(trainseqs,trainoriseqs,copy_num)
        enc_inputs, dec_inputs, dec_outputs = torch.Tensor(dna_sequences), torch.Tensor(dna_sequences), torch.LongTensor(ori_seqs)

        enc_inputs, dec_inputs = Standardize(tgt_vocab_size)(enc_inputs), Standardize(tgt_vocab_size)(dec_inputs)
        loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs, dec_outputs), 8, True)
        model = Transformer(n_layers).cuda()
        losses = train(loader, model,modelpath)
        print(losses)
        model.load_state_dict(torch.load(modelpath))
    print(f"模型参数已成功加载！modelpath:**************{modelpath}****************")

    print('-----------------------------------deep learning--------------------------------')
    pre_seqs,phred_scores,prewith_seqs,phredwith_scores,phred_scoresstr  = testnet(model,enc_inputstest,
                                                                   dec_inputstest,testseqs,testseqs)
    with open(dpconsensus_path + '/oriandpreseqswith_.fasta', 'w') as file:
        for i in range(len(oriseqs)):
            dis = Levenshtein.distance(oriseqs[i],pre_seqs[i])
            file.write(
                f">oriseq{i}\n{oriseqs[i]}  dis:{dis}\n{pre_seqs[i]}\n>phred_scores{i}\n{phred_scoresstr[i]}\n")
    return pre_seqs,phred_scores,all_consus,allbs_quas


