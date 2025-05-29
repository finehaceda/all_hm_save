import csv
import difflib
import math
import os
import subprocess
from datetime import datetime

import Levenshtein
import keras
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torchvision.transforms import transforms

from config import *
import tensorflow as tf


# 4.3 定义位置信息
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pos_table = np.array(
            [[pos / np.power(10000, 2 * i / d_model) for i in range(d_model)] if pos != 0 else np.zeros(d_model) for pos
             in range(max_len)])
        pos_table[1:, 0::2] = np.sin(pos_table[1:, 0::2])  # 字嵌入维度为偶数时
        pos_table[1:, 1::2] = np.cos(pos_table[1:, 1::2])  # 字嵌入维度为奇数时
        self.pos_table = torch.FloatTensor(pos_table).cuda()  # 字嵌入维度为奇数时
        # self.pos_table = torch.FloatTensor(pos_table)                # enc_inputs: [seq_len, d_model]

    def forward(self, enc_inputs):  # enc_inputs: [batch_size, seq_len, d_model]
        # temp = self.pos_table[:enc_inputs.size(1), :]
        enc_inputs += self.pos_table[:enc_inputs.size(1), :]
        return self.dropout(enc_inputs.cuda())
        # return self.dropout(enc_inputs)


# 4.4 Mask掉停用词
def get_attn_pad_mask(seq_q, seq_k):  # seq_q: [batch_size, seq_len] ,seq_k: [batch_size, seq_len]
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # 判断 输入那些含有P(=0),用1标记 ,[batch_size, 1, len_k]
    return pad_attn_mask.expand(batch_size, len_q, len_k)


# 4.5 Decoder 输入 Mask
def get_attn_subsequence_mask(seq):  # seq: [batch_size, tgt_len]
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # 生成上三角矩阵,[batch_size, tgt_len, tgt_len]
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()  # [batch_size, tgt_len, tgt_len]
    return subsequence_mask


# 4.6 计算注意力信息、残差和归一化
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):  # Q: [batch_size, n_heads, len_q, d_k]
        # K: [batch_size, n_heads, len_k, d_k]
        # V: [batch_size, n_heads, len_v(=len_k), d_v]
        # attn_mask: [batch_size, n_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -1e9)  # 如果时停用词P就等于 0
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):  # input_Q: [batch_size, len_q, d_model]
        # input_K: [batch_size, len_k, d_model]
        # input_V: [batch_size, len_v(=len_k), d_model]
        # attn_mask: [batch_size, seq_len, seq_len]
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1,
                                                                           2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        # attn: [batch_size, n_heads, len_q, len_k]
        context = context.transpose(1, 2).reshape(batch_size, -1,
                                                  n_heads * d_v)
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(d_model).cuda()(output + residual), attn
        # return nn.LayerNorm(d_model)(output + residual), attn


# 4.7 前馈神经网络
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False))

    def forward(self, inputs):  # inputs: [batch_size, seq_len, d_model]
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model).cuda()(output + residual)  # [batch_size, seq_len, d_model]
        # return nn.LayerNorm(d_model)(output + residual)  # [batch_size, seq_len, d_model]


# 4.7 前馈神经网络
class PoswiseFeedForwardNet11(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet11, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False))

    def forward(self, inputs):  # inputs: [batch_size, seq_len, d_model]
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model).cuda()(output + residual)  # [batch_size, seq_len, d_model]
        # return nn.LayerNorm(d_model)(output + residual)  # [batch_size, seq_len, d_model]


# 4.8 单个encoder
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()  # 多头注意力机制
        self.pos_ffn = PoswiseFeedForwardNet()  # 前馈神经网络

    def forward(self, enc_inputs, enc_self_attn_mask):  # enc_inputs: [batch_size, src_len, d_model]
        # 输入3个enc_inputs分别与W_q、W_k、W_v相乘得到Q、K、V                          # enc_self_attn_mask: [batch_size, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                               # enc_outputs: [batch_size, src_len, d_model],
                                               enc_self_attn_mask)  # attn: [batch_size, n_heads, src_len, src_len]
        # enc_outputs = self.pos_ffn(enc_outputs)                                       # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # self.src_emb = nn.Embedding(src_vocab_size, d_model)
        # self.src_emb = nn.Linear(src_vocab_size, src_vocab_size*d_model)
        # self.src_emb = nn.Linear(1, d_model)                     # 把字转换字向量
        # self.pos_emb = PositionalEncoding(d_model)
        self.pos_ffn = PoswiseFeedForwardNet()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=2)

    def forward(self, enc_inputs, enc_inputs1=None):
        tensor = torch.randn(enc_inputs.shape[0], enc_inputs.shape[1]).cuda()
        enc_self_attn_mask = get_attn_pad_mask(tensor, tensor)  # enc_self_attn_mask: [batch_size, src_len, src_len]
        enc_self_attns = []
        # enc_outputs = enc_inputs
        enc_inputs = self.pos_ffn(enc_inputs)
        enc_outputs = self.conv1(enc_inputs.view(-1, 1, enc_inputs.shape[1], d_model))
        enc_outputs = enc_outputs.view(-1, enc_outputs.shape[2], d_model)

        # enc_outputss = self.pos_ffn(enc_inputs1)
        # enc_outputss0 = self.conv1(enc_inputs1.view(-1,1,enc_inputs1.shape[1],d_model))
        # enc_outputss1 = enc_outputss0.view(-1,enc_outputss0.shape[2],d_model)

        for layer in self.layers:
            # enc_outputs, enc_self_attn = layer(enc_inputs, enc_self_attn_mask)  # enc_outputs :   [batch_size, src_len, d_model],
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)

        enc_outputs = self.conv1(enc_outputs.view(-1, 1, enc_outputs.shape[1], d_model))
        enc_outputs = self.pos_ffn(enc_outputs.view(-1, enc_outputs.shape[2], d_model))
        # enc_outputs = enc_outputs.view(-1,enc_outputs.shape[2],d_model)
        return enc_outputs, enc_self_attns


class MyLSTM(nn.Module):
    def __init__(self, input_size=d_model, hidden_size=64, num_layers=2, output_size=decode_layer):
        super(MyLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        # Initialize cell state
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        # LSTM forward pass
        out, _ = self.bilstm(x, (h0, c0))

        # Output of the last LSTM cell
        out = self.fc(out)
        # out = self.fc(out[:, -1, :])
        return out


class FeedforwardNN(nn.Module):
    def __init__(self):
        super(FeedforwardNN, self).__init__()
        self.fc = nn.Linear(4, 32)  # 线性层将输入的每个特征映射到128维
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(32, 128)  # 输出层将128维特征映射回128维

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc_out(x)
        return x


def ConvTranspose(input_data):
    batch_size = 10
    input_channels = 4
    input_length = 128

    # 定义卷积上采样的参数
    out_channels = 64
    kernel_size = 3
    stride = 1
    padding = 1

    # 创建输入数据
    # input_data = torch.rand((batch_size, input_channels, input_length))

    # 定义卷积上采样层
    conv_transpose_layer = nn.ConvTranspose1d(in_channels=4, out_channels=64,
                                              kernel_size=3, stride=1, padding=1)

    # 执行卷积上采样操作
    output_data = conv_transpose_layer(input_data)
    return output_data.permute(0, 2, 1)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 定义卷积层
        self.conv1 = nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        # self.conv3 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)

        # 定义池化层
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        return x


class DataStandardizer:
    def __init__(self):
        self.mean = None
        self.std = None
        self.transform = None

    def fit(self, x):
        # 计算均值和标准差
        self.mean = torch.mean(x, dim=2)
        self.std = torch.std(x, dim=2)

    def normalize(self, x):
        # 标准化数据
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # 将数据转换为张量
            transforms.Normalize(self.mean, self.std)  # 标准化
        ])
        return self.transform(x)


# 定义一个MinMax标准化层
class MinMaxNormalize(nn.Module):
    def __init__(self, feature_dim):
        super(MinMaxNormalize, self).__init__()
        self.feature_dim = feature_dim

    def forward(self, x):
        min_val = x.min(dim=0, keepdim=True)[0]
        max_val = x.max(dim=0, keepdim=True)[0]
        # 防止分母为0
        x = (x - min_val) / (max_val - min_val + 1e-6)
        return x


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


# 4.12 Trasformer
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        # self.model = FeedforwardNN()
        self.projectionbegin = nn.Linear(tgt_vocab_size, d_model, bias=False)
        self.ConvTranspose = nn.ConvTranspose1d(in_channels=tgt_vocab_size, out_channels=d_model, kernel_size=3,
                                                stride=1, padding=1)
        self.ConvTranspose1 = nn.ConvTranspose1d(in_channels=tgt_vocab_size, out_channels=d_model_hidden, kernel_size=3,
                                                 stride=1, padding=1)
        self.ConvTranspose2 = nn.ConvTranspose1d(in_channels=d_model_hidden, out_channels=d_model, kernel_size=3,
                                                 stride=1, padding=1)
        self.pos_ffn11 = PoswiseFeedForwardNet11()
        self.CNN = CNN()
        self.Encoder = Encoder().cuda()
        self.Decoder = MyLSTM()
        self.projection = nn.Linear(decode_layer, tgt_vocab_size, bias=False)
        # self.projectionpre = nn.Linear(tgt_vocab_size, d_model, bias=False)
        self.normalize = MinMaxNormalize(feature_dim=d_model)
        self.standardize = Standardize(feature_dim=decode_layer)


        # self.arate = nn.Parameter(torch.tensor(1))
        # self.projection2 = nn.Linear(decode_layer, tgt_vocab_size, bias=False)

    def forward(self, enc_inputs, dec_inputs=None):
        # 第一版
        # enc_inputs = self.ConvTranspose(enc_inputs.permute(0, 2, 1)).permute(0, 2, 1) #128*4--->128*d_model
        # enc_outputs1, enc_self_attns1 = self.Encoder(enc_inputs1) #128*d_model--->128*d_model
        # dec_outputs1 = self.Decoder(enc_outputs1)  #128*d_model--->128*decode_layer
        # dec_logits1 = self.projection(dec_outputs1) #128*decode_layer--->128*tgt_vocab_size
        # 第二版
        enc_inputs = self.projectionbegin(enc_inputs)  # 128*4--->128*d_model

        enc_outputs, enc_self_attns = self.Encoder(enc_inputs)  # 128*d_model--->128*d_model

        dec_outputs = self.Decoder(enc_outputs)  # 128*d_model--->128*decode_layer
        dec_logits = self.projection(dec_outputs)  # 128*decode_layer--->128*tgt_vocab_size
        # dec_out = torch.stack((dec_logits, dec_inputs), dim=-1)  # [batch_size, 2]
        # dec_rate = self.fc(dec_out)
        # print(f"self.arate:{self.arate}")

        # enc_inputs = self.projectionbegin(dec_inputs) #128*4--->128*d_model
        #
        # enc_outputs, enc_self_attns = self.Encoder(enc_inputs) #128*d_model--->128*d_model
        #
        # dec_outputs = self.Decoder(enc_outputs)  #128*d_model--->128*decode_layer
        # dec_logits2 = self.projection(dec_outputs) #128*decode_layer--->128*tgt_vocab_size
        #
        # dec_logits =  self.arate * dec_logits1+(2-self.arate)*dec_logits2
        # 0.242
        # dec_logits = self.standardize(dec_logits)   # 这个是红玫加的,可以去掉
        #
        # 第三版
        # enc_inputs = self.projectionbegin(enc_inputs) #128*4--->128*d_model
        # dec_inputs = self.projectionbegin(dec_inputs) #128*4--->128*d_model
        # enc_outputs, enc_self_attns = self.Encoder(enc_inputs) #128*d_model--->128*d_model
        # dec_outputs, enc_self_attns = self.Encoder(dec_inputs) #128*d_model--->128*d_model
        # merge_outputs = nn.LayerNorm(d_model).cuda()(enc_outputs+dec_outputs)
        # #①0.24 0.9992
        # dec_outputs = self.Decoder(merge_outputs)  #128*d_model--->128*decode_layer
        # dec_logits = self.projection(dec_outputs) #128*decode_layer--->128*tgt_vocab_size
        # dec_logits = self.standardize(dec_logits)

        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, enc_self_attns, enc_self_attns
        # return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, enc_self_attns, enc_self_attns, self.arate.item()


# 4.13 定义网络
# model = Transformer().cuda()
#
# enc_inputs, dec_inputs, dec_outputs = make_data(sentences)
# loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 2, True)


def convert_to_DNA1234(sequence):
    dna_sequence = ""
    for digit in sequence:
        digit = digit.item()
        if digit == 1:
            dna_sequence += 'A'
        elif digit == 2:
            dna_sequence += 'C'
        elif digit == 3:
            dna_sequence += 'G'
        elif digit == 4:
            dna_sequence += 'T'
        else:
            # 处理非法输入
            print(f"Invalid digit: {digit}")
    return dna_sequence


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


def convert_to_DNA_(sequence, predictdata):
    new_predictdata = []
    prewith_phred = []
    i = 0
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
        i += 1
        # 处理非法输入
        # print(f"Invalid digit: {digit}")
    return dna_sequence, prewith_phred, new_predictdata


def converttoDNAgetdel11(sequence, predictdata, rate=0.9):
    prewith_phred = []
    i = 0
    dna_sequence_pulsdel = ""
    tinydict = {'0': 'A', '1': 'C', '2': 'G', '3': 'T', '4': '-'}
    for digit in sequence:
        digit = digit.item()
        maxpre = np.max(predictdata[i])
        if digit == 0:
            dna_sequence_pulsdel += 'A'
            prewith_phred.append(maxpre)
        elif digit == 1:
            dna_sequence_pulsdel += 'C'
            prewith_phred.append(maxpre)
        elif digit == 2:
            dna_sequence_pulsdel += 'G'
            prewith_phred.append(maxpre)
        elif digit == 3:
            dna_sequence_pulsdel += 'T'
            prewith_phred.append(maxpre)
        else:
            if maxpre < rate:
                dna_sequence_pulsdel += '-'
                prewith_phred.append(np.max(predictdata[i]))
        i += 1
    return dna_sequence_pulsdel, prewith_phred


def converttoDNAgetdel(sequence, predictdata, rate=0.9):
    prewith_phred = []
    i = 0
    dna_sequence_pulsdel = ""
    tinydict = {'0': 'A', '1': 'C', '2': 'G', '3': 'T', '4': '-'}
    for digit in sequence:
        digit = str(digit.item())
        maxpre = np.max(predictdata[i])
        if digit == '4':
            # pass
            # if maxpre < rate-0.2:
            if maxpre < rate:
                sorted_indices = np.argsort(predictdata[i])
                # 找到倒数第二个索引
                second_largest_index = sorted_indices[-2]
                # 找到第二大的数字
                second_largest_value = predictdata[i][second_largest_index]
                if second_largest_value > 0.3:
                    dna_sequence_pulsdel += tinydict[str(second_largest_index)]
                    prewith_phred.append(second_largest_value)
        # elif maxpre < rate:
        #     sorted_indices = np.argsort(predictdata[i])
        #     # 找到倒数第二个索引
        #     second_largest_index = sorted_indices[-2]
        #     # 找到第二大的数字
        #     if second_largest_index != 4:
        #         dna_sequence_pulsdel += tinydict[digit]
        #         prewith_phred.append(maxpre)
        else:
            dna_sequence_pulsdel += tinydict[digit]
            prewith_phred.append(maxpre)
        i += 1
    return dna_sequence_pulsdel, prewith_phred


def softmax0(your_array):
    normalized_array = (your_array) / (np.sum(your_array, axis=0))
    return normalized_array


def getEdit(model, enc_inputs, dec_inputs, dec_outputs):
    # device = next(model.parameters()).device
    ori = convert_to_DNA(dec_outputs)
    # predict_dec_input = mytest(model, enc_inputs.view(1, -1).cuda())
    # predict, _, _, _ = model(enc_inputs.view(1, -1).cuda(),predict_dec_input)
    # predict, _, _, _ = model(enc_inputs.view(1, -1).cuda(),enc_inputs.view(1, -1).cuda())
    with torch.no_grad():  # 在评估中，通常不用计算梯度
        allpredict, _, _, _ = model(enc_inputs.view(1, enc_inputs.shape[0], -1).cuda(),
                                    dec_inputs.view(1, dec_inputs.shape[0], -1).cuda())
        # allpredict = Standardize(tgt_vocab_size)(allpredict)
        allpredict1, _, _, _ = model(dec_inputs.view(1, dec_inputs.shape[0], -1).cuda(),
                                     dec_inputs.view(1, dec_inputs.shape[0], -1).cuda())
        allpredict = Standardize(tgt_vocab_size)(allpredict)
        allpredict1 = Standardize(tgt_vocab_size)(allpredict1)
        allpredict += allpredict1
        # allpredict = Standardize(tgt_vocab_size)(allpredict)
    # predict, _, _, _ = model(enc_inputs.cuda(),enc_inputs.cuda())
    # predict, _, _, _ = model(enc_inputs.view(1, -1),enc_inputs.view(1, -1))
    # predict = allpredict.data.max(1, keepdim=True)[1]
    allpredict = torch.nn.functional.softmax(allpredict, dim=-1)  # 先做softmax再做提取才对
    # predictdata = (allpredict + allpredict1)
    # predictdata = (allpredict + allpredict1)
    # predictdata = allpredict * 3
    predictdata = allpredict
    predict = predictdata.data.max(1, keepdim=True)[1]

    # allpredicts = min_max_normalize_last_dim(predictdata)   # 这两行换成softmax试试
    # allpredicts = allpredicts/torch.sum(allpredicts,dim=-1, keepdim=True)

    # allpredicts = torch.nn.functional.softmax(predictdata, dim=-1)
    # allpredicts1 = torch.nn.functional.softmax(allpredict1, dim=-1)
    # allpredicts2 = torch.nn.functional.softmax(predictdata, dim=-1)
    # allpredicts3 = torch.nn.functional.softmax(predictdata*2, dim=-1)
    # allpredicts = torch.nn.functional.softmax(predictdata*3, dim=-1)
    # predictdata = Standardize(tgt_vocab_size)(predictdata)
    # allpredicts = min_max_normalize_last_dim(predictdata)
    # allpredicts = allpredicts+allpredicts1
    # allpredicts = torch.nn.functional.softmax(allpredicts, dim=-1)

    # predictdata = Standardize(tgt_vocab_size)(predictdata)
    # allpredicts = torch.nn.functional.softmax(predictdata, dim=-1)

    # allpredicts2 = torch.nn.functional.softmax(predictdata, dim=-1)
    # allpredicts3 = min_max_normalize_last_dim(allpredicts2)
    # allpredicts = torch.exp(torch.nn.functional.log_softmax(predictdata, dim=-1))
    # outputs1 = allpredicts / torch.sum(allpredicts, dim=-1, keepdim=True)
    # allpredicts = allpredicts / torch.sum(allpredicts, dim=-1, keepdim=True)

    # allpredicts = torch.nn.functional.softmax(predictdata, dim=-1)
    # allpredicts = torch.nn.functional.softmax(allpredicts, dim=-1)
    # allpredict = allpredict.data.cpu().numpy()
    # allpredicts = []
    # for data in allpredict:
    #     # allpredicts.append(softmax0(data))
    #     allpredicts.append(torch.nn.functional.softmax(data, dim=1))
    # allpredict = softmax0(allpredict.data.cpu().numpy())
    pre = convert_to_DNA(predict)
    prewith_seq, prewith_phred, predictphred = convert_to_DNA_(predict, allpredict.cpu().detach().numpy())
    dna_sequence_pulsdel,prewith_phred_pulsdel = converttoDNAgetdel(predict,allpredict.cpu().detach().numpy(),0)
    # dna_sequence_pulsdel, prewith_phred_pulsdel = converttoDNAgetdel(predict, allpredict.cpu().detach().numpy(), 0.6)
    # print(enc_inputs)
    # print(dec_outputs)
    # print(predict.view(-1))
    # dis = Levenshtein.distance(ori, dnatest_sequence)
    # print('ori seq'+str(dis))
    dis = Levenshtein.distance(ori, dna_sequence_pulsdel)
    # dis = Levenshtein.distance(ori, pre[:len(ori)])
    # if dis>=5:
    #     print(pre)
    #     print(ori)
    # print('ori compare pre edit distance: '+str(dis))
    # return dis,enc_inputs.tolist()
    # return dis,enc_inputs.cpu().numpy()
    return dis, ori, pre, prewith_seq, prewith_phred, predictphred, dna_sequence_pulsdel, prewith_phred_pulsdel


# def getEdit(model,enc_inputs,dec_inputs,dec_outputs):
#     ori = convert_to_DNA(dec_outputs)
#     # predict_dec_input = mytest(model, enc_inputs.view(1, -1).cuda())
#     # predict, _, _, _ = model(enc_inputs.view(1, -1).cuda(),predict_dec_input)
#     # predict, _, _, _ = model(enc_inputs.view(1, -1).cuda(),enc_inputs.view(1, -1).cuda())
#     allpredict, _, _, _ = model(enc_inputs.view(1, enc_inputs.shape[0],-1).cuda(),dec_inputs.view(1, dec_inputs.shape[0],-1).cuda())
#     # allpredict1, _, _, _ = model(dec_inputs.view(1, dec_inputs.shape[0],-1).cuda(),dec_inputs.view(1, dec_inputs.shape[0],-1).cuda())
#     # predict, _, _, _ = model(enc_inputs.cuda(),enc_inputs.cuda())
#     # predict, _, _, _ = model(enc_inputs.view(1, -1),enc_inputs.view(1, -1))
#     # predict = allpredict.data.max(1, keepdim=True)[1]
#     # allpredicts = torch.nn.functional.softmax(allpredict, dim=1)
#     # predictdata = (allpredict + allpredict1)
#     # predictdata = (allpredict + allpredict1)
#     # predictdata = allpredict * 3
#     predictdata = allpredict
#     predict = predictdata.data.max(1, keepdim=True)[1]
#
#     # allpredicts = min_max_normalize_last_dim(predictdata)   # 这两行换成softmax试试
#     # allpredicts = allpredicts/torch.sum(allpredicts,dim=-1, keepdim=True)
#
#     allpredicts = torch.nn.functional.softmax(predictdata, dim=-1)
#     # allpredicts1 = torch.nn.functional.softmax(allpredict1, dim=-1)
#     # allpredicts2 = torch.nn.functional.softmax(predictdata, dim=-1)
#     # allpredicts3 = torch.nn.functional.softmax(predictdata*2, dim=-1)
#     # allpredicts = torch.nn.functional.softmax(predictdata*3, dim=-1)
#     # predictdata = Standardize(tgt_vocab_size)(predictdata)
#     # allpredicts = min_max_normalize_last_dim(predictdata)
#     # allpredicts = allpredicts+allpredicts1
#     # allpredicts = torch.nn.functional.softmax(allpredicts, dim=-1)
#
#
#     # predictdata = Standardize(tgt_vocab_size)(predictdata)
#     # allpredicts = torch.nn.functional.softmax(predictdata, dim=-1)
#
#     # allpredicts2 = torch.nn.functional.softmax(predictdata, dim=-1)
#     # allpredicts3 = min_max_normalize_last_dim(allpredicts2)
#     # allpredicts = torch.exp(torch.nn.functional.log_softmax(predictdata, dim=-1))
#     # outputs1 = allpredicts / torch.sum(allpredicts, dim=-1, keepdim=True)
#     # allpredicts = allpredicts / torch.sum(allpredicts, dim=-1, keepdim=True)
#
#     # allpredicts = torch.nn.functional.softmax(predictdata, dim=-1)
#     # allpredicts = torch.nn.functional.softmax(allpredicts, dim=-1)
#     # allpredict = allpredict.data.cpu().numpy()
#     # allpredicts = []
#     # for data in allpredict:
#     #     # allpredicts.append(softmax0(data))
#     #     allpredicts.append(torch.nn.functional.softmax(data, dim=1))
#     # allpredict = softmax0(allpredict.data.cpu().numpy())
#     pre = convert_to_DNA(predict)
#     prewith_seq,prewith_phred,predictphred = convert_to_DNA_(predict,allpredicts.cpu().detach().numpy())
#     dna_sequence_pulsdel,prewith_phred_pulsdel = converttoDNAgetdel(predict,allpredicts.cpu().detach().numpy(),0)
#     # dna_sequence_pulsdel,prewith_phred_pulsdel = converttoDNAgetdel(predict,allpredicts.cpu().detach().numpy(),0.5)
#     # print(enc_inputs)
#     # print(dec_outputs)
#     # print(predict.view(-1))
#     # dis = Levenshtein.distance(ori, dnatest_sequence)
#     # print('ori seq'+str(dis))
#     dis = Levenshtein.distance(ori, dna_sequence_pulsdel)
#     # dis = Levenshtein.distance(ori, pre[:len(ori)])
#     # if dis>=5:
#     #     print(pre)
#     #     print(ori)
#     # print('ori compare pre edit distance: '+str(dis))
#     # return dis,enc_inputs.tolist()
#     # return dis,enc_inputs.cpu().numpy()
#     return dis,ori,pre,prewith_seq,predictphred,prewith_phred,dna_sequence_pulsdel,prewith_phred_pulsdel
#     # return dis,ori,pre,predict,allpredicts.cpu().detach().numpy()
#     # return dis,ori,pre,predict,allpredicts

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # 避免数值不稳定性，减去最大值
    return exp_x / exp_x.sum(axis=1, keepdims=True)


def min_max_normalize_last_dim(tensor):
    # 沿着最后一维计算最大值和最小值
    max_vals, _ = torch.max(tensor, dim=-1, keepdim=True)
    min_vals, _ = torch.min(tensor, dim=-1, keepdim=True)

    # 计算范围
    range_vals = max_vals - min_vals

    # 最大最小标准化
    normalized_tensor = (tensor - min_vals) / range_vals

    return normalized_tensor


# 定义一个MinMax标准化层
class MinMaxNormalize(nn.Module):
    def __init__(self, feature_dim):
        super(MinMaxNormalize, self).__init__()
        self.feature_dim = feature_dim

    def forward(self, x):
        min_val = x.min(dim=0, keepdim=True)[0]
        max_val = x.max(dim=0, keepdim=True)[0]
        # 防止分母为0
        x = (x - min_val) / (max_val - min_val + 1e-6)
        return x


# 定义L1正则化函数
def l1_regularizer(weight, lambda_l1):
    return lambda_l1 * torch.norm(weight, 1)


# 定义L2正则化函数
def l2_regularizer(weight, lambda_l2):
    return lambda_l2 * torch.norm(weight, 2)


def train11(loader, model):
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99, weight_decay=0.001)
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)
    print("要加载的模型参数文件不存在！\n开始训练")
    losses = []
    for epoch in range(20):
        for enc_inputs, dec_inputs, dec_outputs, dec_outputsin in loader:
            enc_inputs, dec_inputs, dec_outputs = enc_inputs.cuda(), dec_inputs.cuda(), dec_outputs.cuda()
            # outputs1, enc_self_attns1, dec_self_attns1, dec_enc_attns1 = model(dec_inputs, dec_inputs)
            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
            # lambda_l1 = lambda_l2 = 0.01
            # l1_regularization = l1_regularizer(model.parameters(), lambda_l1)
            # l2_regularization = l2_regularizer(model.parameters(), lambda_l2)
            # outputs = min_max_normalize_last_dim(outputs)
            # outputscc = torch.exp(outputs)

            # outputs =  Standardize(tgt_vocab_size)(outputs)
            # outputs = torch.nn.functional.softmax(outputs, dim=1)

            # outputs = min_max_normalize_last_dim(outputs)
            # outputs = outputs/torch.sum(outputs,dim=-1, keepdim=True)
            # outputs1 = min_max_normalize_last_dim(outputs1)
            # outputs1 = outputs1/torch.sum(outputs1,dim=-1, keepdim=True)

            # outputs = torch.nn.functional.softmax(outputs, dim=-1)
            # outputs1 = torch.nn.functional.softmax(outputs1, dim=-1)
            # allpredicts22 = allpredicts2 / torch.sum(allpredicts2, dim=-1, keepdim=True)
            # outputsaa = torch.exp(torch.nn.functional.log_softmax(allpredicts1, dim=-1))
            # outputsbb = torch.exp(torch.nn.functional.log_softmax(allpredicts2, dim=-1))
            # outputs11 = torch.nn.functional.softmax(allpredicts1)
            # outputs22 = torch.nn.functional.softmax(allpredicts2)
            # outputs = outputs11/
            # outputs11bb = min_max_normalize_last_dim(outputs22)
            # outputs = torch.nn.functional.softmax(outputs, dim=-1)
            # outputs1 = torch.nn.functional.softmax(outputs1, dim=-1)
            # loss = criterion(outputs, dec_outputs.view(-1))*2
            # l1_penalty = 0.000001 * sum(p.abs().sum() for p in model.parameters())
            # loss += l1_penalty
            # loss2 = criterion(outputs1, dec_outputs.view(-1))
            # loss = criterion(outputs, dec_outputs.view(-1))*2
            loss = criterion(outputs, dec_outputs.view(-1))
            # loss2 = criterion(outputs1, dec_outputs.view(-1))
            # loss = loss + loss2
            # loss = 0.3*loss1 + 0.7*loss2
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
        losses.append(loss.item())
    torch.save(model.state_dict(), 'model2.pth')
    return losses


def train(loader, model, model_path):
    global loss
    criterion = nn.CrossEntropyLoss()
    lr, momentum, weight_decay = 1e-3, 0.99, 1e-4
    # print("lr = %f, momentum = %f, weight_decay = %f" % (lr, momentum, weight_decay))
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99, weight_decay=weight_decay)  # 加入L2正则化，防止过拟合
    # print("不加L2正则化！")
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    print("要加载的模型参数文件不存在！\n开始训练")
    losses = []
    for epoch in range(31):
        for enc_inputs, dec_inputs, dec_outputs, dec_outputsin in loader:
            enc_inputs, dec_inputs, dec_outputs = enc_inputs.cuda(), dec_inputs.cuda(), dec_outputs.cuda()
            # outputs1, enc_self_attns1, dec_self_attns1, dec_enc_attns1 = model(dec_inputs, dec_inputs)
            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
            # outputs = Standardize(tgt_vocab_size)(outputs)
            # outputs = min_max_normalize_last_dim(outputs)
            # outputs = outputs/torch.sum(outputs,dim=-1, keepdim=True)
            outputs = torch.nn.functional.softmax(outputs, dim=-1)  # softmax去掉会减少loss，但输出的概率没有那么准
            # outputs1 = torch.nn.functional.softmax(outputs1, dim=-1)
            # outputs1 = Standardize(tgt_vocab_size)(outputs1)
            # outputs1 = min_max_normalize_last_dim(outputs1)
            # outputs1 = outputs1/torch.sum(outputs1,dim=-1, keepdim=True)
            # loss = criterion(outputs, dec_outputs.view(-1))*2
            loss = criterion(outputs, dec_outputs.view(-1))
            # loss1 = criterion(outputs1, dec_outputs.view(-1))
            # loss = loss + loss1
            # loss = 0.3*loss1 + 0.7*loss2
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
        losses.append(loss.item())
    torch.save(model.state_dict(), model_path)
    return losses


def train_Adam(loader, model, model_path, lr):
    model.train()
    print("使用Adam!!")
    global loss
    criterion = nn.CrossEntropyLoss()  # 可以换CTC
    weight_decay = 1e-4
    print("lr = %f, weight_decay = %f" % (lr, weight_decay))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    print("要加载的模型参数文件不存在！\n开始训练")
    losses = []
    for epoch in range(10):
        for enc_inputs, dec_inputs, dec_outputs, dec_outputsin in loader:
            enc_inputs, dec_inputs, dec_outputs = enc_inputs.cuda(), dec_inputs.cuda(), dec_outputs.cuda()
            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs)
            outputs = torch.nn.functional.softmax(outputs, dim=-1)  # 不用softmax恢复率会更高，红玫说
            loss = criterion(outputs, dec_outputs.view(-1))
            optimizer.zero_grad()
            # loss.backward(retain_graph=True)
            loss.backward()
            optimizer.step()
        print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
        losses.append(loss.item())
    torch.save(model.state_dict(), model_path)
    return losses


def train_Adam_parallel(loader, model, model_path, lr):
    print("使用Adam!!")
    global loss
    criterion = nn.CrossEntropyLoss()  # 可以换CTC
    weight_decay = 1e-4
    print("lr = %f, weight_decay = %f" % (lr, weight_decay))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    print("要加载的模型参数文件不存在！\n开始训练")
    losses = []
    for epoch in range(15):
        model.train()
        for enc_inputs, dec_inputs, dec_outputs, dec_outputsin in loader:
            enc_inputs, dec_inputs, dec_outputs = enc_inputs.cuda(), dec_inputs.cuda(), dec_outputs.cuda()
            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
            outputs = torch.nn.functional.softmax(outputs, dim=-1)  # 不用softmax恢复率会更高，红玫说
            loss = criterion(outputs, dec_outputs.view(-1))
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
        losses.append(loss.item())
    torch.save(model.state_dict(), model_path)
    return losses


# def readbsalignalignfile11(path):
#     with open(path,'r') as file:
#         lines = file.readlines()
#     if len(lines)>0:
#         line = lines[0].strip('\n').split('\t')
#         mismatch,delnum,insertnum = line[-3],line[-2],line[-1]
#         # print(mismatch,delnum,insertnum)
#         return mismatch,delnum,insertnum,lines[1].strip('\n'),lines[3].strip('\n')
#     return 0,0,0,'',''

def readbsalignalignfile11(path, allpredict, rate=0.9):
    with open(path, 'r') as file:
        lines = file.readlines()
    if len(lines) > 0:
        mismatchdelinsertindexs = []
        upnums = 0
        line = lines[0].strip('\n').split('\t')
        mismatch, delnum, insertnum = line[-3], line[-2], line[-1]
        line = lines[2].strip('\n')
        # print(line)
        for i in range(len(line)):
            # a = line[i]
            # print(a)
            if line[i] == '-' or line[i] == '*':
                mismatchdelinsertindexs.append(i)
        allpredictreal = [i for i in range(len(allpredict)) if np.max(allpredict[i]) < rate]
        for i in allpredictreal:
            if i in mismatchdelinsertindexs:
                upnums += 1
        # print(mismatch,delnum,insertnum)
        return mismatch, delnum, insertnum, lines[1].strip('\n'), lines[3].strip('\n'), upnums
    return 0, 0, 0, '', '', 0


def readbsalignalignfile22(path, allpredict, ratel, rater, dis, dna_sequence_pulsdel, prewith_phred_pulsdel):
    with open(path, 'r') as file:
        lines = file.readlines()
    if len(lines) > 0:
        mismatchdelinsertindexs = []
        upnums = 0
        line = lines[0].strip('\n').split('\t')
        mismatch, delnum, insertnum = line[-3], line[-2], line[-1]
        line = lines[2].strip('\n')
        upnumsindex = []
        # print(line)
        for i in range(len(line)):
            # a = line[i]
            # print(a)
            if line[i] == '-' or line[i] == '*':
                mismatchdelinsertindexs.append(i)
        line = lines[3].strip('\n')
        delindex = [i for i in range(len(line)) if line[i] == '-']
        for i in range(len(delindex)):
            if len(delindex) > i + 1:
                if delindex[i + 1] == delindex[i] + 1:
                    continue
            for misi in range(len(mismatchdelinsertindexs)):
                if mismatchdelinsertindexs[misi] > delindex[i]:
                    mismatchdelinsertindexs[misi] -= 1
        allpredictreal = [i for i in range(len(allpredict)) if ratel < np.max(allpredict[i]) <= rater]
        mislen = len(mismatchdelinsertindexs)
        for i in allpredictreal:
            if mislen > 0:
                if i in mismatchdelinsertindexs:
                    upnums += 1
                    upnumsindex.append({i, np.max(allpredict[i])})
        # if len(mismatchdelinsertindexs)==0 and len(allpredictreal)>0:
        #     upnums+=len(allpredictreal)
        if len(allpredictreal) > 0 and mislen < dis:
            upnums += dis - mislen
        # for i in allpredictreal:
        #     if i in mismatchdelinsertindexs:
        #         upnums+=1
        # print(mismatch,delnum,insertnum)
        return mismatch, delnum, insertnum, lines[1].strip('\n'), lines[3].strip('\n'), upnums, upnumsindex
    return 0, 0, 0, '', '', 0, []


def readbsalignalignfile(path, allpredict, ratel, rater, dis, dna_sequence_pulsdel, prewith_phred_pulsdel):
    with open(path, 'r') as file:
        lines = file.readlines()
    if len(lines) > 0:
        mismatchdelinsertindexs = []
        delinsertindexs = []
        upnums = 0
        line = lines[0].strip('\n').split('\t')
        mismatch, delnum, insertnum = line[-3], line[-2], line[-1]
        line = lines[2].strip('\n')
        upnumsindex = []
        # print(line)
        for i in range(len(line)):
            # a = line[i]
            # print(a)
            if line[i] == '-' or line[i] == '*':
                mismatchdelinsertindexs.append(i)
                if line[i] == '-':
                    delinsertindexs.append(i)
        line = lines[3].strip('\n')
        delindex = [i for i in range(len(line)) if line[i] == '-']
        insertindexs = [x for x in delinsertindexs if x not in delindex]
        mismatchindexs = [x for x in mismatchdelinsertindexs if x not in delinsertindexs]
        # insertindexs = delinsertindexs - delindex
        for i in range(len(delindex)):
            if len(delindex) > i + 1:
                if delindex[i + 1] == delindex[i] + 1:
                    continue
            for misi in range(len(mismatchdelinsertindexs)):
                if mismatchdelinsertindexs[misi] > delindex[i]:
                    mismatchdelinsertindexs[misi] -= 1
            for misi in range(len(insertindexs)):
                if insertindexs[misi] > delindex[i]:
                    insertindexs[misi] -= 1
            for misi in range(len(mismatchindexs)):
                if mismatchindexs[misi] > delindex[i]:
                    mismatchindexs[misi] -= 1
        # for i in range(len(delindex)):
        #     if i!=0:
        #         delindex[i]=delindex[i]-1
        allpredictreal = [i for i in range(len(prewith_phred_pulsdel)) if ratel < prewith_phred_pulsdel[i] <= rater]
        mislen = len(mismatchdelinsertindexs)
        score = []
        upnumsallerrors = [0, 0, 0]
        for i in allpredictreal:
            if mislen > 0:
                if i in mismatchdelinsertindexs:
                    if i in insertindexs:
                        minph = prewith_phred_pulsdel[i]
                        minphi = i
                        iij = i + 1
                        upnumsallerrors[0] += 1
                        while iij < len(prewith_phred_pulsdel) and dna_sequence_pulsdel[iij] == dna_sequence_pulsdel[i]:
                            if prewith_phred_pulsdel[iij] <= minph:
                                minph = prewith_phred_pulsdel[iij]
                                minphi = iij
                            iij += 1
                        upnums += 1
                        upnumsindex.append({minphi, minph})
                        if minphi != i:
                            # score += f"phred: {minph}\n"
                            score.append(minph)
                            upnumsallerrors[0] -= 1
                            upnums -= 1
                        # print(f"ratel:{ratel} rater:{rater} phred: {minph}")
                    else:
                        if i in mismatchindexs:
                            upnumsallerrors[2] += 1
                        else:
                            upnumsallerrors[1] += 1
                        upnums += 1
                        upnumsindex.append({i, prewith_phred_pulsdel[i]})

        if mislen < dis and ratel >= 0.949:
            upnums += dis - mislen
            upnumsallerrors[1] += dis - mislen
            delnum = int(delnum) + dis - mislen
        # print(score)
        # if len(mismatchdelinsertindexs)==0 and len(allpredictreal)>0:
        #     upnums+=len(allpredictreal)
        # if len(allpredictreal)>0 and mislen<dis:
        #     upnums+=dis-mislen
        # for i in allpredictreal:
        #     if i in mismatchdelinsertindexs:
        #         upnums+=1
        # print(mismatch,delnum,insertnum)
        return mismatch, delnum, insertnum, lines[1].strip('\n'), lines[3].strip(
            '\n'), upnums, upnumsindex, score, upnumsallerrors
    return 0, 0, 0, '', '', 0, [], "", []


# def bsalign_alitest(seq1,seq2):
def bsalign_alitest11(seq1, seq2, allpredict, rate):
    with open('files/seqstest.fasta', 'w') as file:
        # with open('seqs.fasta', 'w') as file:
        for j, cus in enumerate([seq1, seq2]):
            file.write('>' + str(j) + '\n')
            file.write(str(cus) + '\n')
    # shell = '../bsalign-master/bsalign align seqs.fasta > ali.ali'
    shell = '/home1/hongmei/00work_files/1229/DNAtransformer/0122bsalign/bsalign-master/bsalign align files/seqstest.fasta > files/alitest.ali'
    result = subprocess.run(shell, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # mismatch,delnum,insertnum,seq1,seq2 = readbsalignalignfile('files/ali.ali')
    mismatch, delnum, insertnum, seq1, seq2, upnums = readbsalignalignfile('files/alitest.ali', allpredict, rate)
    # mismatch,delnum,insertnum,seq1,seq2 = readbsalignalignfile('ali.ali')

    return mismatch, delnum, insertnum, seq1, seq2, upnums
    # return mismatch,delnum,insertnum,seq1,seq2


def bsalign_alitest22(seq1, seq2, allpredict, ratel, rater, dis, dna_sequence_pulsdel, prewith_phred_pulsdel,
                      dir='files'):
    with open(dir + '/seqstest.fasta', 'w') as file:
        # with open('seqs.fasta', 'w') as file:
        for j, cus in enumerate([seq1, seq2]):
            file.write('>' + str(j) + '\n')
            file.write(str(cus) + '\n')
    shell = '../bsalign-master/bsalign align ' + dir + '/seqstest.fasta >' + dir + '/alitest.ali'
    result = subprocess.run(shell, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if dis >= 2:
        mismatch, delnum, insertnum, seq1, seq2, upnums, upnumsindex = readbsalignalignfile(dir + '/alitest.ali',
                                                                                            allpredict, ratel, rater,
                                                                                            dis, dna_sequence_pulsdel,
                                                                                            prewith_phred_pulsdel)
        return mismatch, delnum, insertnum, seq1, seq2, upnums, upnumsindex
    else:
        # mismatch,delnum,insertnum,seq1,seq2 = readbsalignalignfile('ali.ali')
        return 0, 0, 0, seq1, seq2, 0, []


def bsalign_alitest(seq1, seq2, allpredict, ratel, rater, dis, dna_sequence_pulsdel, prewith_phred_pulsdel,
                    dir='files'):
    with open(dir + '/seqstest.fasta', 'w') as file:
        for j, cus in enumerate([seq1, dna_sequence_pulsdel]):
            file.write('>' + str(j) + '\n')
            file.write(str(cus) + '\n')
    shell = '/home1/hongmei/00work_files/1229/DNAtransformer/0122bsalign/bsalign-master/bsalign align ' + dir + '/seqstest.fasta >' + dir + '/alitest.ali'
    result = subprocess.run(shell, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if dis >= 1:
        edit_ops = Levenshtein.editops(seq1, dna_sequence_pulsdel)
        insertnum = sum(1 for op in edit_ops if op[0] == 'insert')
        delnum = sum(1 for op in edit_ops if op[0] == 'delete')
        mismatch = sum(1 for op in edit_ops if op[0] == 'replace')
        mismatch, delnum, insertnum, seq1, seq2, upnums, upnumsindex, wrongin, upnumsallerrors = readbsalignalignfile(
            dir + '/alitest.ali', allpredict, ratel, rater, dis, dna_sequence_pulsdel, prewith_phred_pulsdel)
        return mismatch, delnum, insertnum, seq1, seq2, upnums, upnumsindex, wrongin, upnumsallerrors
    else:
        # mismatch,delnum,insertnum,seq1,seq2 = readbsalignalignfile('ali.ali')
        return 0, 0, 0, seq1, dna_sequence_pulsdel, 0, [], [], []


def compare_sequences(seq1, seq2):
    # 创建SequenceMatcher实例
    matcher = difflib.SequenceMatcher(None, seq1, seq2)

    insertions = 0
    deletions = 0
    substitutions = 0

    aligned_seq1 = []
    aligned_seq2 = []

    # 获取所有操作
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            aligned_seq1.append(seq1[i1:i2])
            aligned_seq2.append(seq2[j1:j2])
            substitutions += (i2 - i1)
        elif tag == 'insert':
            aligned_seq1.append('-' * (j2 - j1))  # 插入部分用'-'表示
            aligned_seq2.append(seq2[j1:j2])
            insertions += (j2 - j1)
        elif tag == 'delete':
            aligned_seq1.append(seq1[i1:i2])
            aligned_seq2.append('-' * (i2 - i1))  # 删除部分用'-'表示
            deletions += (i2 - i1)
        elif tag == 'equal':
            aligned_seq1.append(seq1[i1:i2])
            aligned_seq2.append(seq2[j1:j2])

    # 拼接为字符串
    aligned_seq1 = ''.join(aligned_seq1)
    aligned_seq2 = ''.join(aligned_seq2)

    return aligned_seq1, aligned_seq2, insertions, deletions, substitutions


def check_error_type(seq1, seq2, j):
    edit_ops = Levenshtein.editops(seq1, seq2)
    # insertnum = sum(1 for op in edit_ops if op[0] == 'insert')
    # delnum = sum(1 for op in edit_ops if op[0] == 'delete')
    # mismatch = sum(1 for op in edit_ops if op[0] == 'replace')
    jerror, num = "equal", 0
    for op, i1, j1 in edit_ops:
        if j1 == j:
            jerror = op
            num += 1
    if num == 0:
        num = 1
    return jerror, num
    # for si,ops in enumerate(matcher.get_opcodes()):
    #     op, i1, i2, j1, j2 = ops[0],ops[1],ops[2],ops[3],ops[4]
    #     if j1 <= j < j2:
    #         jerror = op
    #         if op == 'replace' or op == 'delete':
    #             jerror = op
    #         elif op == 'insert' and j1 <= j < j2:
    #             jerror = op
    #     if j1 <= j < j2 and op == 'equal':
    #         jerror = op
    #         jindex = si
    # return jerror,jindex


def getalians1218(ori_seqs, pre_seqs, all_dis, pre_seqsphred,method = 'SeqTransformer',copy_num = ''):
    # ratetest = [[0, 0.4], [0.4, 0.5], [0.5, 0.6], [0.6, 0.7], [0.7, 0.8], [0.8, 0.9], [0.9, 1]]
    # ratetest = [[0, 0.2], [0.2, 0.3], [0.3, 0.35], [0.35, 0.4], [0.4, 0.45], [0.45, 0.5], [0.5, 0.55],
    #             [0.55, 0.6], [0.6, 0.65], [0.65, 0.7], [0.7, 0.75], [0.75, 0.8], [0.8, 0.85], [0.85, 0.9], [0.9, 0.95], [0.95, 1]]
    # # ratetest = [[0, 0.6], [0.6, 1]]
    # # ratetest = [[0, 1]]
    # truebasenumsarr = [0,0]
    # upnumsallerrorssarr = [{ 'insert': 0,'delete': 0, 'replace': 0, 'equal': 0},{'insert': 0,'delete': 0, 'replace': 0, 'equal': 0}]
    ratetest = [[0, 0.5], [0.5, 0.7], [0.7, 0.9], [0.9, 1]]
    # ratetest = [[0, 0.6], [0.6, 1]]
    truebasenumsarr = [0,0,0,0]
    upnumsallerrorssarr = [{ 'insert': 0,'delete': 0, 'replace': 0, 'equal': 0},{'insert': 0,'delete': 0, 'replace': 0, 'equal': 0},
                           {'insert': 0,'delete': 0, 'replace': 0, 'equal': 0},{'insert': 0,'delete': 0, 'replace': 0, 'equal': 0}]
    k = 0
    for ratel, rater in ratetest:
        seqnums, basenums, errorseq_basenums, truebasenums = 0, 0, 0, 0
        truebasenums1,truebasenums2 = 0,0
        upnumsallerrorss = {'insert': 0,'delete': 0, 'replace': 0, 'equal': 0}
        # real = {'replace':0,'insert':0,'delete':0,'equal':0}
        for i in range(len(ori_seqs)):
            seq1, seq2, dis, prewith_phred_pulsdel = ori_seqs[i], pre_seqs[i], all_dis[i], pre_seqsphred[i]
            nums = len([i for i in range(len(prewith_phred_pulsdel)) if ratel < prewith_phred_pulsdel[i] <= rater])
            basenums += nums
            if nums > 0:
                seqnums += 1
            if dis == 0:
                continue
            indelss = {'insert': 0,'delete': 0, 'replace': 0, 'equal': 0}
            allpredictj = [i for i in range(len(prewith_phred_pulsdel)) if ratel < prewith_phred_pulsdel[i] <= rater]
            errorseq_basenums += len(allpredictj)

            edit_ops = Levenshtein.editops(seq1, seq2)
            for j in allpredictj:
                # basenums += 1
                jerror, num = check_error_type(seq1, seq2, j)
                # if jerror == 'equal':
                #     ji = j+1
                #     while ji < len(prewith_phred_pulsdel) and jerror == 'equal' and seq2[ji]==seq2[j]:
                #         jerror, num  = check_error_type(seq1, seq2, ji)
                #         ji += 1
                #     if jerror != 'equal' and ji < len(prewith_phred_pulsdel):
                #         for ir in range(len(ratetest)):
                #             if ratetest[ir][0] < prewith_phred_pulsdel[ji] <= ratetest[ir][1] :
                #                 # print("------------???_--------------------------------")
                #                 truebasenumsarr[ir] -= 1
                #                 upnumsallerrorssarr[ir][jerror] -= 1

                indelss[jerror] += num
            for op in edit_ops:
                if op[2] >= len(prewith_phred_pulsdel) and ratel < prewith_phred_pulsdel[-1] <= rater:
                    indelss[op[0]] += 1
            truebasenums1 += sum(value for key, value in indelss.items() if key != 'equal')
            truebasenums2 += sum(value for key, value in indelss.items() if key != 'equal' and key!='delete')
            truebasenumsarr[k] += sum(value for key, value in indelss.items() if key != 'equal' and key!='delete')
            for key, v in indelss.items():
                upnumsallerrorss[key] += v
                upnumsallerrorssarr[k][key] += v
        divbase1 = ''
        if basenums != 0 :
            divbase1 = 1-truebasenums1/basenums
        divbase2 = ''
        if basenums != 0 :
            divbase2 = 1-truebasenums2/basenums
        data = [
            # (method, copy_num, ratel, rater, seqnums, basenums, errorseq_basenums, truebasenums, upnumsallerrorss,divbase),
            ('', ratel, rater, seqnums, basenums, errorseq_basenums, truebasenumsarr[k], upnumsallerrorssarr[k],divbase1,divbase2),
        ]
        print(data)
        with open('./models/derrick_seqcluster_10000_b64/derrick_bs_dp_phred_1218.csv', 'a', encoding='utf8', newline='') as f:
        # with open('./models/derrick_seqcluster_1000_b8/test.csv', 'a', encoding='utf8', newline='') as f:
            writer = csv.writer(f)  # csv.writer()中可以传一个文件对象
            for line in data:  # 该data既可以是列表嵌套列表的数据类型也可以是列表嵌套元组的数据类型
                writer.writerow(line)
        k += 1

def getalians(ori_seqs, pre_seqs, all_dis, pre_seqsphred, ratetest, filepath, revdata, method='SeqTransformer',copy_num=''):
    # ratetest = [[0, 0.2],[0.2, 0.4], [0.4, 0.5], [0.5, 0.6], [0.6, 0.7], [0.7, 0.8], [0.8, 0.9], [0.9, 1]]
    # ratetest = [[0, 0.2], [0.2, 0.3], [0.3, 0.4], [0.35, 0.4], [0.4, 0.45], [0.45, 0.5], [0.5, 0.55],
    #             [0.55, 0.6], [0.6, 0.65], [0.65, 0.7], [0.7, 0.75], [0.75, 0.8], [0.8, 0.85], [0.85, 0.9], [0.9, 0.95], [0.95, 1]]
    # ratetest = [[0, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.45], [0.45, 0.5], [0.5, 0.55],
    #             [0.55, 0.6], [0.6, 0.65], [0.65, 0.7], [0.7, 0.75], [0.75, 0.8], [0.8, 0.85], [0.85, 0.9], [0.9, 0.95], [0.95, 1]]
    # # ratetest = [[0, 0.6], [0.6, 1]]
    # # ratetest = [[0, 1]]
    # truebasenumsarr = [0,0]
    # upnumsallerrorssarr = [{ 'insert': 0,'delete': 0, 'replace': 0, 'equal': 0},{'insert': 0,'delete': 0, 'replace': 0, 'equal': 0}]
    # ratetest = [[0, 0.5], [0.5, 0.7], [0.7, 0.9], [0.9, 1]]
    # ratetest = [[0, 0.6], [0.6, 1]]
    truebasenumsarr1 = [0]*len(ratetest)
    truebasenumsarr2 = [0]*len(ratetest)
    basenumsarr = [0]*len(ratetest)
    errorseq_basenumsarr = [0]*len(ratetest)
    seqnumsarr = [0]*len(ratetest)
    upnumsallerrorssarr = []
    for i in range(len(ratetest)):
        upnumsallerrorssarr.append({ 'insert': 0,'delete': 0, 'replace': 0, 'equal': 0})
    # upnumsallerrorssarr = [{ 'insert': 0,'delete': 0, 'replace': 0, 'equal': 0},{'insert': 0,'delete': 0, 'replace': 0, 'equal': 0},
    #                        {'insert': 0,'delete': 0, 'replace': 0, 'equal': 0},{'insert': 0,'delete': 0, 'replace': 0, 'equal': 0}]

    for i in range(len(ori_seqs)):
        seq1, seq2, dis, prewith_phred_pulsdel = ori_seqs[i], pre_seqs[i], all_dis[i], pre_seqsphred[i]
        for r in range(len(ratetest)):
            ratel, rater = ratetest[r][0],ratetest[r][1]
            nums = len([i for i in range(len(prewith_phred_pulsdel)) if ratel < prewith_phred_pulsdel[i] <= rater])
            basenumsarr[r] += nums
            if nums > 0:
                seqnumsarr[r] += 1
            if dis != 0:
                errorseq_basenumsarr[r] += nums
        if dis == 0:
            continue
        edit_ops = Levenshtein.editops(seq1, seq2)
        visited = set()

        for op, i1, j1 in edit_ops:
            if j1 < len(prewith_phred_pulsdel):
                minphred = prewith_phred_pulsdel[j1]
                mini = j1
                if op == 'insert':
                    thisbase = seq2[j1]
                    fi = j1+1
                    while fi < len(prewith_phred_pulsdel) and seq2[fi] == thisbase:
                        if prewith_phred_pulsdel[fi] < minphred and fi not in visited:
                            minphred = prewith_phred_pulsdel[fi]
                            mini = fi
                        fi += 1
                    visited.add(mini)
                elif op == 'replace':
                    visited.add(mini)
                elif op == 'delete':
                    thisbase = seq2[j1]
                    fi = j1-1
                    while fi >= 0 and seq2[fi] == thisbase:
                        if prewith_phred_pulsdel[fi] < minphred and fi not in visited:
                            minphred = prewith_phred_pulsdel[fi]
                            # mini = fi
                        fi -= 1
                    # visited.add(mini)
            else:
                minphred = prewith_phred_pulsdel[-1]
                # mini = len(prewith_phred_pulsdel)-1
            for r in range(len(ratetest)):
                ratel, rater = ratetest[r][0], ratetest[r][1]
                if ratel < minphred <= rater:
                    upnumsallerrorssarr[r][op] += 1
                    if op != 'equal':
                        truebasenumsarr1[r] += 1
                        if op != 'delete':
                            truebasenumsarr2[r] += 1
    data = []
    print(f"method, copy_num, ratel, rater, 总碱基数量, 真正错误碱基数量, indels， indel错误识别率,准确率, 当前分布错误识别率,占所有错误的灵敏度")
    for r in range(len(ratetest)):
        divbase1 = '0'
        if basenumsarr[r] != 0:
            divbase1 = 1 - truebasenumsarr1[r] / basenumsarr[r]
        divbase2 = ''
        if basenumsarr[r] != 0:
            divbase2 = 1 - truebasenumsarr2[r] / basenumsarr[r]
        read = '0'
        # if errorbasenum != 0:
        #     # print(f"truebasenumsarr1[r]:{truebasenumsarr1[r]}")
        #     # print(f"errorbasenum:{errorbasenum}")
        #     read = truebasenumsarr1[r] / errorbasenum
        ratel, rater = ratetest[r][0], ratetest[r][1]
        # d = (method, copy_num, ratel, rater, seqnumsarr[r], basenumsarr[r], errorseq_basenumsarr[r], truebasenumsarr1[r], upnumsallerrorssarr[r], divbase1, divbase2)
        d = (method, copy_num, ratel, rater, basenumsarr[r], truebasenumsarr1[r], upnumsallerrorssarr[r], divbase2,1-int(divbase1), divbase1, read)
        data.append(d)
        if len(ratetest) < 10:
            print(d)
        # print(d)
    # with open('./models/derrick_seqcluster_10000_b64/derrick_bs_dp_phred_1218.csv', 'a', encoding='utf8',newline='') as f:
        # with open('./models/derrick_seqcluster_1000_b8/test.csv', 'a', encoding='utf8', newline='') as f:
    with open(filepath, 'a', encoding='utf8', newline='') as f:
        writer = csv.writer(f)  # csv.writer()中可以传一个文件对象
        for line in revdata:  # 该data既可以是列表嵌套列表的数据类型也可以是列表嵌套元组的数据类型
            writer.writerow(line)
        for line in data:  # 该data既可以是列表嵌套列表的数据类型也可以是列表嵌套元组的数据类型
            writer.writerow(line)
    # k += 1

    #
    # for ratel, rater in ratetest:
    #     seqnums, basenums, errorseq_basenums, truebasenums = 0, 0, 0, 0
    #     truebasenums1,truebasenums2 = 0,0
    #     upnumsallerrorss = {'insert': 0,'delete': 0, 'replace': 0, 'equal': 0}
    #     # real = {'replace':0,'insert':0,'delete':0,'equal':0}


def testnet11(loader, model, lens):
    all_dis_sum = 0
    all_dis_gl0 = 0
    afterphredupdatedis = 0
    all_nums = 0
    # inputs = []
    all_dis = []
    ori_seqs = []
    pre_seqs = []
    phred_scores = []
    predict_seqs = []
    max_indicesnew = []
    max_indices_dis = []
    mismatchnums = []
    delnums = []
    insertnums = []
    scorelt09s = []
    preerrorsnum = 0
    preerrorerrornum = 0
    preerrortruenum = 0
    predisg0num = 0
    TP, FP, FN, TN = 0, 0, 0, 0
    for enc_inputs, dec_inputs, dec_outputs, max_indices in loader:
        for i in range(len(enc_inputs)):
            phred = ''
            # enc_inputs, dec_inputs, dec_outputs = enc_inputs.cuda(), dec_inputs.cuda(), dec_outputs.cuda()
            # dis, enc = getEdit(model,enc_inputs[i].cuda(),dec_outputs[i].cuda())
            rate = 0.9
            dis, ori, pre, predict, allpredict = getEdit(model, enc_inputs[i].cuda(), dec_inputs[i].cuda(),
                                                         dec_outputs[i].cuda())
            mismatch, delnum, insertnum, seq1, seq2, upnums = bsalign_alitest(ori, pre, allpredict, rate)
            # mismatch,delnum,insertnum,seq1,seq2 = bsalign_alitest(ori,pre)
            mismatchnums.append(mismatch)
            delnums.append(delnum)
            insertnums.append(insertnum)
            ori_seqs.append(ori)
            pre_seqs.append(pre)
            predict_seqs.append(predict)
            all_dis.append(dis)
            seqsss = convert_to_DNA0(max_indices[i])
            max_indicesnew.append(seqsss)
            scorelt09 = []
            # max_indices_dis.append(Levenshtein.distance(ori, seqsss))
            phred = ''
            num = 0
            for i in range(len(allpredict)):
                if np.max(allpredict[i]) < rate:
                    # print(f'sscore<0.9: {[sscore for sscore in allpredict[i] if sscore<0.9]}')
                    # print(["{:.3f}".format(num) for num in allpredict[i]])
                    num += 1
                    scorelt09.append({i, np.max(allpredict[i])})
                # phred += str("{:.3f}".format(np.max(allpredict[i])))+' '
                phred += str(np.max(allpredict[i])) + ' '
                # for j in range(4):
                #     phred += str(allpredict[i][j]) + ' '
                # phred += str(allpredict[i][4]) + ','
            phred_scores.append(phred)
            all_nums += num
            # phred_scores.append("{:.3f}".format(np.max(allpredict)) )
            # print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
            scorelt09s.append(scorelt09)
            # if scorelt09>0 and dis>0:
            #     print(f"when scorelt09>0 scorelt09:{scorelt09},dis:{dis}")
            # if  dis>0 and num>0:
            #     preerrortruenum+=1
            # print(f"-------------------------------------when dis>0 and num==0 dis:{dis}---------------------------------")
            if num > 0 and dis > 0:
                TP += 1
            elif num > 0 and dis == 0:
                FN += 1
            elif num == 0 and dis > 0:
                FP += 1
            elif num == 0 and dis == 0:
                TN += 1

            # if num>0 and dis>0:
            #     TP += 1
            # elif num==0 and dis>0:
            #     FN += 1
            # elif num>0 and dis==0:
            #     FP += 1
            # elif num==0 and dis==0:
            #     TN += 1
            # if num > 0:
            #     preerrorsnum+=1
            #     if dis>=1:
            #         predisg0num += 1
            # # preerrorsnum+=1
            # if  num>0 and dis==0:
            #     preerrorerrornum+=1
            # print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!when num>0 and dis==0 num:{num}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # if dis>0:
            #     predisg0num+=1
            # else:
            #     print(f"when scorelt09=0 dis=0")
            if dis >= 1:
                # inputs_gl0.append(enc)
                all_dis_gl0 += 1
            all_dis_sum += dis
            afterphredupdatedis += dis - upnums

            # inputs.append(enc)
            # all_dis+=getEdit(model,enc_inputs[i],dec_outputs[i])
    print('dis > 0 number:' + str(all_dis_gl0) + '     ' + str(all_dis_gl0 / lens))
    print('TP:' + str(TP) + ' FN:' + str(FN) + ' FP:' + str(FP) + ' TN:' + str(TN))
    # print('preerrorerrornum:'+str(preerrorerrornum)+' preerrorsnum:'+str(preerrorsnum)+' predisg0num:'+str(predisg0num))
    # print('preerrortrue rate:'+str(1-preerrorerrornum/preerrorsnum))
    # print('predis rate:'+str(preerrorsnum/all_dis_gl0))
    # print('predis rate:'+str(predisg0num/all_dis_gl0))
    # print('preerrorsnum:'+str(preerrortruenum)+'preerrortrue rate:'+str(preerrortruenum/preerrorsnum))
    print('phred < () nums :' + str(all_nums / lens))
    print('average dis:' + str(all_dis_sum / lens))
    print('recovery rate:' + str(1 - all_dis_sum / lens / len(ori_seqs[0])))
    print('after phred update average dis:' + str(afterphredupdatedis / lens))
    print('after phred update recovery rate:' + str(1 - afterphredupdatedis / lens / len(ori_seqs[0])))
    # print('scorelt09:' + scorelt09)
    # max_indicesnew = []
    # for seq in max_indices:
    #     max_indicesnew.append(convert_to_DNA(seq))
    with open('./myfiles/oriandpreseqs.fasta', 'w') as file:
        # with open('./oriandpreseqs.fasta','w') as file:
        for i in range(len(ori_seqs)):
            # file.write(f">oriseq{i}\n{ori_seqs[i]}\n>preseq{i}   {all_dis[i]}\n{predict_seqs[i]}\n{pre_seqs[i]}\n")
            # file.write(f">oriseq{i}\n{ori_seqs[i]}\n>preseq{i}   {all_dis[i]}\n{pre_seqs[i]}\n{phred_scores[i]}\n")
            # file.write(f">oriseq{i}\n{ori_seqs[i]}\n>preseq{i} edit:{all_dis[i]} mis:{mismatchnums[i]} del:{delnums[i]} insert:{insertnums[i]}\n{pre_seqs[i]}\n>predatamanage{i}  {max_indices_dis[i]}\n{max_indicesnew[i]}\n")
            file.write(
                f">oriseq{i}\n{ori_seqs[i]}\n>preseq{i} edit:{all_dis[i]} mis:{mismatchnums[i]} del:{delnums[i]} insert:{insertnums[i]}\n{pre_seqs[i]}\n>phred{i} scorelt09num:{scorelt09s[i]}\n{phred_scores[i]}\n")
            # f">predatamanage{i} \n{max_indicesnew[i]}\n")
    # print(phred_scores[0])
    # with open('./myfiles/score.fasta','w') as file:
    # # with open('./score.fasta','w') as file:
    #     for i in range(len(phred_scores)):
    #         # file.write(f">oriseq{i}\n{ori_seqs[i]}\n>preseq{i}   {all_dis[i]}\n{predict_seqs[i]}\n{pre_seqs[i]}\n")
    #         file.write(f">seq{i}\n{phred_scores[i]}\n")
    # file.write(f">oriseq{i}\n{ori_seqs[i]}\n>preseq{i} edit:{all_dis[i]} mis:{mismatchnums[i]} del:{delnums[i]} insert:{insertnums[i]}\n{pre_seqs[i]}\n>predatamanage{i}  {max_indices_dis[i]}\n{max_indicesnew[i]}\n")

    # with open('../myfiles/encinputs.fasta','w') as file:
    #     for  data in inputs:
    #         # ss = data.T
    #         file.write(str(data)+'\n')
    #         file.write('\n')
    #         # file.write(str(data.T)+'\n')


def testnet22(loader, model, lens):
    all_dis_sum = 0
    all_dis_gl0 = 0
    afterphredupdatedis = 0
    all_dis = []
    ori_seqs = []
    pre_seqs = []
    phred_scores = []
    predict_seqs = []
    max_indicesnew = []
    mismatchnums = []
    delnums = []
    insertnums = []
    scorelt09s = []
    preupnums = 0
    all_nums_gt0dis = 0
    all_nums = 0
    TP, FP, FN, TN = 0, 0, 0, 0
    for enc_inputs, dec_inputs, dec_outputs, max_indices in loader:
        for i in range(len(enc_inputs)):
            ratel = 0.55
            rater = 0.6
            dis, ori, pre, predict, allpredict = getEdit(model, enc_inputs[i].cuda(), dec_inputs[i].cuda(),
                                                         dec_outputs[i].cuda())
            mismatch, delnum, insertnum, seq1, seq2, upnums = bsalign_alitest(ori, pre, allpredict, ratel, rater, dis)
            mismatchnums.append(mismatch)
            delnums.append(delnum)
            insertnums.append(insertnum)
            ori_seqs.append(ori)
            pre_seqs.append(pre)
            predict_seqs.append(predict)
            all_dis.append(dis)
            seqsss = convert_to_DNA0(max_indices[i])
            max_indicesnew.append(seqsss)
            scorelt09 = []
            phred = ''
            num = 0
            for i in range(len(allpredict)):
                if ratel < np.max(allpredict[i]) <= rater:
                    num += 1
                    scorelt09.append({i, np.max(allpredict[i])})
                phred += str(np.max(allpredict[i])) + ' '
            phred_scores.append(phred)
            all_nums += num
            scorelt09s.append(scorelt09)
            if num > 0 and dis > 0:
                TP += 1
            elif num > 0 and dis == 0:
                FN += 1
            elif num == 0 and dis > 0:
                FP += 1
            elif num == 0 and dis == 0:
                TN += 1
            if dis >= 1:
                all_dis_gl0 += 1
                all_dis_sum += dis
                all_nums_gt0dis += num
            preupnums += upnums
            afterphredupdatedis += dis - upnums
    print('发生错误的序列数量:' + str(all_dis_gl0))
    print('发生错误的碱基数量 :' + str(all_dis_sum))
    print('每条序列碱基数量：' + str(len(ori_seqs[0])))
    print('预测碱基概率大于' + str(ratel) + '小于等于:' + str(rater) + '时，序列数量总共有:' + str(TP + FN) + '\n所有序列中，在此概率区间碱基数量总共有:' + str(
        all_nums)
          + ' 发生错误的序列中，在此概率区间的碱基数量有:' + str(all_nums_gt0dis)
          + ' 碱基概率在此区间，全部发生错误的碱基数量 :' + str(preupnums))
    print('dis > 0 number:' + str(all_dis_gl0) + '     ' + str(all_dis_gl0 / lens))
    print('average dis:' + str(all_dis_sum / lens))
    print('when dis >= 1 average dis:' + str(all_dis_sum / all_dis_gl0))
    print('recovery rate:' + str(1 - all_dis_sum / lens / len(ori_seqs[0])))

    with open('./myfiles/oriandpreseqs.fasta', 'w') as file:
        for i in range(len(ori_seqs)):
            file.write(
                f">oriseq{i}\n{ori_seqs[i]}\n>preseq{i} edit:{all_dis[i]} mis:{mismatchnums[i]} del:{delnums[i]} insert:{insertnums[i]}\n{pre_seqs[i]}\n>phred{i} scorelt09num:{scorelt09s[i]}\n{phred_scores[i]}\n")


def testnet_ori(loader, model, lens, datatime):
    rate = 0.0
    while rate < 0.5:
        # if rate >= 0.8999:
        #     if (rate >= 0.949):
        #         ratel = rate
        #         rater = rate + 0.1
        #     else:
        #         ratel = rate
        #         rater = rate + 0.05
        # else:
        #     rater = rate + 0.1
        ratel = rate
        rater = rate + 0.6
        # if rate >= 0.8999:
        #     # if (rate >= 0.9559):
        #     if (rate >= 0.999469):
        #         ratel = rate
        #         rater = rate + 0.1
        #     else:
        #         ratel = rate
        #         rater = rate + 0.056
        # else:
        #     rater = rate + 0.1
        # rater = rate+0.1
        all_dis_sum = 0
        all_dis_gl0 = 0
        afterphredupdatedis = 0
        all_dis = []
        ori_seqs = []
        pre_seqs = []
        aliseq1, aliseq2 = [], []
        phred_scores = []
        phredwith_scores = []
        predict_seqs = []
        max_indicesnew = []
        mismatchnums = []
        delnums = []
        insertnums = []
        mismatchnum0 = 0
        delnum0 = 0
        insertnum0 = 0
        allscorelt09s = []
        preupnums = 0
        all_nums_gt0dis = 0
        all_nums = 0
        TP, FP, FN, TN = 0, 0, 0, 0
        upnumsallerrorss = [0, 0, 0]
        # score = f"ratel:{ratel} rater:{rater}\n"
        score = []
        for enc_inputs, dec_inputs, dec_outputs, max_indices in loader:
            for i in range(len(enc_inputs)):
                # rater = 0.999
                # print(ratel,rater)
                dis, ori, pre, predict, allpredict, prewith_phred, dna_sequence_pulsdel, prewith_phred_pulsdel = getEdit(
                    model, enc_inputs[i].cuda(), dec_inputs[i].cuda(),
                    dec_outputs[i].cuda())
                # mismatch, delnum, insertnum, seq1, seq2, upnums,upnumsindex,wrongin,upnumsallerrors = bsalign_alitest(ori, pre, allpredict, ratel, rater, dis,dna_sequence_pulsdel,prewith_phred_pulsdel)
                mismatch, delnum, insertnum, seq1, seq2, upnums, upnumsindex, wrongin, upnumsallerrors = bsalign_alitest(
                    ori, pre, allpredict, ratel, rater, dis, dna_sequence_pulsdel, prewith_phred_pulsdel, 'tempfiles')
                # mismatch, delnum, insertnum, seq1, seq2, upnums,upnumsindex,wrongin ,upnumsallerrors = bsalign_alitest(ori, pre, allpredict, ratel, rater, dis,dna_sequence_pulsdel,prewith_phred_pulsdel,'testdir')

                # score+=wrongin
                errornums = int(mismatch) + int(delnum) + int(insertnum)
                # if dis - errornums>0:
                #     delnum += dis-errornums
                for j in range(len(upnumsallerrors)):
                    upnumsallerrorss[j] += upnumsallerrors[j]
                score = np.append(score, wrongin)
                aliseq1.append(seq1)
                aliseq2.append(seq2)
                mismatchnums.append(mismatch)
                delnums.append(delnum)
                insertnums.append(insertnum)
                mismatchnum0 += int(mismatch)
                delnum0 += int(delnum)
                insertnum0 += int(insertnum)
                ori_seqs.append(ori)
                # pre_seqs.append(pre)
                pre_seqs.append(dna_sequence_pulsdel)
                predict_seqs.append(predict)
                all_dis.append(dis)
                seqsss = convert_to_DNA0(max_indices[i])
                max_indicesnew.append(seqsss)
                # scorelt09 = []
                phred = ''
                num = 0
                for j in range(len(prewith_phred_pulsdel)):
                    if ratel < prewith_phred_pulsdel[j] <= rater:
                        num += 1
                        # scorelt09.append({i, np.max(allpredict[i])})
                    phred += str(j) + ':' + str(prewith_phred_pulsdel[j]) + ' '
                phred_scores.append(phred)
                phred = ''
                for j in range(len(prewith_phred)):
                    phred += str(j) + ':' + str(np.max(prewith_phred[j])) + ' '
                phredwith_scores.append(phred)

                all_nums += num
                # if num > 0 and dis > 0:
                #     TP += 1
                # elif num > 0 and dis == 0:
                #     FN += 1
                # elif num == 0 and dis > 0:
                #     FP += 1
                # elif num == 0 and dis == 0:
                #     TN += 1
                if dis >= 1:
                    all_dis_gl0 += 1
                    all_dis_sum += dis
                    # all_dis_sum += errornums
                    all_nums_gt0dis += num
                    # allscorelt09s.append(upnumsindex)
                # else:

                # if dis != int(mismatch)+int(delnum)+int(insertnum):
                #     print(f"dis:{dis}  mismatchnum0:{mismatch} delnum0:{delnum} insertnum0:{insertnum}")
                #     print(f"ori:{ori}\npre:{dna_sequence_pulsdel}")
                allscorelt09s.append(upnumsindex)
                preupnums += upnums
                afterphredupdatedis += dis - upnums

        print('预测碱基概率大于' + str(ratel) + '小于等于:' + str(rater) + '时，序列数量总共有:' + str(
            TP + FN) + '\n所有序列中，在此概率区间碱基数量总共有:' + str(all_nums)
              + ' 发生错误的序列中，在此概率区间的碱基数量有:' + str(all_nums_gt0dis)
              + ' 碱基概率在此区间，全部发生错误的碱基数量 :' + str(preupnums))
        print(upnumsallerrorss)
        # mydict = {'0':0,'1':0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0,'8':0,'9':0}
        # for s in score:
        #     mydict[str(int(s*10))]+=1
        # for key,value in mydict.items():
        #     if value>0:
        #         print(f"{key}:{value} ")
        # print(mydict)
        rate = rater
    print(f"mismatch数量:{mismatchnum0} del数量:{delnum0} insert数量:{insertnum0} ")
    print('发生错误的序列数量:' + str(all_dis_gl0) + ' 发生错误数量占所有数据： ' + str(all_dis_gl0 / lens))
    print('发生错误的碱基数量 :' + str(all_dis_sum))
    print('每条序列碱基数量：' + str(len(ori_seqs[0])))
    # print('dis > 1 number:' + str(all_dis_gl0) + ' 发生错误数量占所有数据： ' + str(all_dis_gl0 / lens))
    print('平均编辑距离:' + str(all_dis_sum / lens))
    print('when dis > 1 平均编辑距离:' + str(all_dis_sum / all_dis_gl0))
    print('recovery rate:' + str(1 - all_dis_sum / lens / len(ori_seqs[0])))

    with open('./myfiles/oriandpreseqs.fasta', 'w') as file:
        for i in range(len(ori_seqs)):
            # file.write(f">oriseq{i}\n{ori_seqs[i]}\n>preseq{i} edit:{all_dis[i]} mis:{mismatchnums[i]} del:{delnums[i]} insert:{insertnums[i]}\n{pre_seqs[i]}\n>phred{i} scorelt09num:{scorelt09s[i]}\n{phred_scores[i]}\n")
            file.write(
                f">oriseq{i}\n{ori_seqs[i]}\n>preseq{i} edit:{all_dis[i]} mis:{mismatchnums[i]} del:{delnums[i]} insert:{insertnums[i]}\n{pre_seqs[i]}\n"
                f">aliseqs{i}\n{aliseq1[i]}\n{aliseq2[i]}\n>phred{i}  scorelt09num:{allscorelt09s[i]}\n {phred_scores[i]}\n")
        # rate = rater

    with open('./myfiles/oriandpreseqswith_.fasta', 'w') as file:
        for i in range(len(ori_seqs)):
            file.write(
                f">preseq{i}\n{pre_seqs[i]}\n>prewith_seqs{i}\n{predict_seqs[i]}\n>phredwith_scores{i}\n{phredwith_scores[i]}\n")


# def getallphrederrors():
#

def testnet(loader, model, lens, datatime):
    start_time = datetime.now()
    mismatchnums = []
    delnums = []
    insertnums = []
    aliseq1, aliseq2 = [], []
    phred_scores = []  # 所有预测序列质量值拼起来的字符串
    phredwith_scores = []  # 所有带-的预测序列质量值拼起来的字符串
    ori_seqs = []  # 所有原序列
    all_dis = []  # 所有预测序列编辑距离
    pre_seqs = []  # 所有预测序列
    predict_seqs = []  # 所有带-的预测序列
    pre_seqsphred = []  # 所有预测序列质量值

    all_dis_sum = 0  # 编辑距离总和
    all_dis_gl0 = 0  # 编辑距离>0序列数量
    mismatchnum0 = 0
    delnum0 = 0
    insertnum0 = 0
    for enc_inputs, dec_inputs, dec_outputs, max_indices in loader:
        for i in range(len(enc_inputs)):
            dis, ori, pre, prewith_seq, prewith_phred, _, dna_sequence_pulsdel, prewith_phred_pulsdel = getEdit(
                model, enc_inputs[i].cuda(), dec_inputs[i].cuda(),
                dec_outputs[i].cuda())
            ori_seqs.append(ori)
            all_dis.append(dis)
            edit_ops = Levenshtein.editops(ori, dna_sequence_pulsdel)
            aliseq1.append(edit_ops)
            predict_seqs.append(prewith_seq)
            pre_seqs.append(dna_sequence_pulsdel)
            pre_seqsphred.append(prewith_phred_pulsdel)
            edit_ops = Levenshtein.editops(ori, dna_sequence_pulsdel)
            insertnum = sum(1 for op in edit_ops if op[0] == 'insert')
            delnum = sum(1 for op in edit_ops if op[0] == 'delete')
            mismatch = sum(1 for op in edit_ops if op[0] == 'replace')
            mismatchnums.append(mismatch)
            delnums.append(delnum)
            insertnums.append(insertnum)
            mismatchnum0 += int(mismatch)
            delnum0 += int(delnum)
            insertnum0 += int(insertnum)

            phred = ''
            for j in range(len(prewith_phred_pulsdel)):
                phred += str(j) + ':' + str(prewith_phred_pulsdel[j]) + ' '
            phred_scores.append(phred)
            phred = ''
            for j in range(len(prewith_phred)):
                phred += str(j) + ':' + str(np.max(prewith_phred[j])) + ' '
            phredwith_scores.append(phred)
            if dis >= 1:
                all_dis_gl0 += 1
                all_dis_sum += dis
    tillnow = datetime.now()

    print(f"mismatch数量:{mismatchnum0} del数量:{delnum0} insert数量:{insertnum0} ")
    print('发生错误的序列数量:' + str(all_dis_gl0) + ' 发生错误数量占所有数据： ' + str(all_dis_gl0 / lens))
    print('发生错误的碱基数量 :' + str(all_dis_sum))
    print('每条序列碱基数量：' + str(len(ori_seqs[0])))
    # print('dis > 1 number:' + str(all_dis_gl0) + ' 发生错误数量占所有数据： ' + str(all_dis_gl0 / lens))
    print('平均编辑距离:' + str(all_dis_sum / lens))
    if all_dis_gl0 != 0:
        print('when dis > 1 平均编辑距离:' + str(all_dis_sum / all_dis_gl0))
    print('recovery rate:' + str(1 - all_dis_sum / lens / len(ori_seqs[0])))

    with open('./myfiles/oriandpreseqs.fasta', 'w') as file:
        for i in range(len(ori_seqs)):
            # file.write(f">oriseq{i}\n{ori_seqs[i]}\n>preseq{i} edit:{all_dis[i]} mis:{mismatchnums[i]} del:{delnums[i]} insert:{insertnums[i]}\n{pre_seqs[i]}\n>phred{i} scorelt09num:{scorelt09s[i]}\n{phred_scores[i]}\n")
            # file.write(
            #     f">oriseq{i}\n{ori_seqs[i]}\n>preseq{i} edit:{all_dis[i]} mis:{mismatchnums[i]} del:{delnums[i]} insert:{insertnums[i]}\n{pre_seqs[i]}\n"
            #     f">aliseqs{i}\n{aliseq1[i]}\n{aliseq2[i]}\n>phred{i}{phred_scores[i]}\n")
            file.write(
                f">oriseq{i}\n{ori_seqs[i]}\n>preseq{i} edit:{all_dis[i]} mis:{mismatchnums[i]} del:{delnums[i]} insert:{insertnums[i]}\n{pre_seqs[i]}\n"
                f">aliops{i}\n{aliseq1[i]}\n>phred{i}{phred_scores[i]}\n")
        # rate = rater

    with open('./myfiles/oriandpreseqswith_.fasta', 'w') as file:
        for i in range(len(ori_seqs)):
            file.write(
                f">preseq{i}\n{pre_seqs[i]}\n>prewith_seqs{i}\n{predict_seqs[i]}\n>phredwith_scores{i}\n{phredwith_scores[i]}\n")

    seq_len = len(ori_seqs[0])
    data = [
        # ('', 'rec rate', 'edit error rate', 'error rate', 'success rate', 'indels', 'error base num', 'error seq num'),
        ('dp', 1 - all_dis_sum / lens / seq_len, all_dis_sum / lens / seq_len, all_dis_gl0 / lens,
         1 - all_dis_gl0 / lens,
         f"{insertnum0}:{delnum0}:{mismatchnum0}", f"{all_dis_sum}", f"{all_dis_gl0}"),
    ]
    print(data)
    data = [
        (f'dp', 1 - all_dis_sum / lens / seq_len, all_dis_sum / lens / seq_len, all_dis_gl0 / lens,
         1 - all_dis_gl0 / lens
         , mismatchnum0 / lens / seq_len, delnum0 / lens / seq_len, insertnum0 / lens / seq_len,
         f"{mismatchnum0}:{delnum0}:{insertnum0}", all_dis_gl0, all_dis_sum, tillnow - start_time + datatime),
    ]
    # with open('./models/derrick_seqcluster_10000_b64/derrick_bs_dp_phred_1219_lt0.csv', 'a', encoding='utf8', newline='') as f:
    #     writer = csv.writer(f)  # csv.writer()中可以传一个文件对象
    #     for line in data:  # 该data既可以是列表嵌套列表的数据类型也可以是列表嵌套元组的数据类型
    #         writer.writerow(line)
    # getalians(ori_seqs, pre_seqs, all_dis, pre_seqsphred)


    # filepath='./models/derrick_seqcluster_10000_b64/derrick_bs_dp_phred_1219_lt0_detail.csv'
    # ratetest = [[0, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.45], [0.45, 0.5], [0.5, 0.55],
    #             [0.55, 0.6], [0.6, 0.65], [0.65, 0.7], [0.7, 0.75], [0.75, 0.8], [0.8, 0.85], [0.85, 0.9], [0.9, 0.95], [0.95, 1]]
    # getalians(ori_seqs, pre_seqs, all_dis, pre_seqsphred,ratetest,filepath,data)

    # filepath='./models/derrick_seqcluster_10000_b64/derrick_bs_dp_phred_1220_lt0.csv'
    filepath='./myfiles/test.csv'
    ratetest = [[0, 0.2],[0.2, 0.4], [0.4, 0.5], [0.5, 0.6], [0.6, 0.7], [0.7, 0.8], [0.8, 0.9], [0.9, 1]]
    getalians(ori_seqs, pre_seqs, all_dis, pre_seqsphred,ratetest,filepath,data)


    return data


def testnet1216_ori(loader, model, lens, datatime):
    start_time = datetime.now()
    rate = 0.0
    # rates = [[0,0.4],[0.4,0.5],[0.5,0.6],[0.6,0.7],[0.7,0.8],[0.8,0.9],[0.9,1]]
    ratetest = [[0, 0.6], [0.6, 1]]
    for ri, [ratel, rater] in enumerate(ratetest):
        # if rate >= 0.8999:
        #     if (rate >= 0.949):
        #         ratel = rate
        #         rater = rate + 0.1
        #     else:
        #         ratel = rate
        #         rater = rate + 0.05
        # else:
        #     rater = rate + 0.1
        # ratel = rate
        # rater = rate + 0.6
        # if rate >= 0.8999:
        #     # if (rate >= 0.9559):
        #     if (rate >= 0.999469):
        #         ratel = rate
        #         rater = rate + 0.1
        #     else:
        #         ratel = rate
        #         rater = rate + 0.056
        # else:
        #     rater = rate + 0.1
        # rater = rate+0.1
        all_dis_sum = 0
        all_dis_gl0 = 0
        afterphredupdatedis = 0
        all_dis = []
        ori_seqs = []
        pre_seqs = []
        aliseq1, aliseq2 = [], []
        phred_scores = []
        phredwith_scores = []
        predict_seqs = []
        max_indicesnew = []
        mismatchnums = []
        delnums = []
        insertnums = []
        mismatchnum0 = 0
        delnum0 = 0
        insertnum0 = 0
        allscorelt09s = []
        preupnums = 0
        all_nums_gt0dis = 0
        all_nums = 0
        TP, FP, FN, TN = 0, 0, 0, 0
        # upnumsallerrorss = [0,0,0]
        upnumsallerrorss = {'replace': 0, 'insert': 0, 'delete': 0}
        # score = f"ratel:{ratel} rater:{rater}\n"
        score = []
        for enc_inputs, dec_inputs, dec_outputs, max_indices in loader:
            for i in range(len(enc_inputs)):
                # rater = 0.999
                # print(ratel,rater)
                dis, ori, pre, predict, allpredict, prewith_phred, dna_sequence_pulsdel, prewith_phred_pulsdel = getEdit(
                    model, enc_inputs[i].cuda(), dec_inputs[i].cuda(),
                    dec_outputs[i].cuda())
                mismatch, delnum, insertnum, seq1, seq2, upnums, upnumsindex, wrongin, upnumsallerrors = bsalign_alitest(
                    ori, pre, allpredict, ratel, rater, dis, dna_sequence_pulsdel, prewith_phred_pulsdel, 'tempfiles')
                if ri == len(ratetest) - 1:
                    edit_ops = Levenshtein.editops(ori, dna_sequence_pulsdel)
                    insertnum = sum(1 for op in edit_ops if op[0] == 'insert')
                    delnum = sum(1 for op in edit_ops if op[0] == 'delete')
                    mismatch = sum(1 for op in edit_ops if op[0] == 'replace')
                    mismatchnums.append(mismatch)
                    delnums.append(delnum)
                    insertnums.append(insertnum)
                    mismatchnum0 += int(mismatch)
                    delnum0 += int(delnum)
                    insertnum0 += int(insertnum)
                # 1216张红玫改，使用Levenshtein方法得到该质量值下插入删除替换错误的数量
                # indelss, seq1, seq2, upnums, upnumsindex = ggetaalianfs(ori,dna_sequence_pulsdel,dis,ratel,rater,prewith_phred_pulsdel)

                # score+=wrongin
                # errornums = int(mismatch)+int(delnum)+int(insertnum)
                # if dis - errornums>0:
                #     delnum += dis-errornums

                # for k,v in indelss.items():
                #     upnumsallerrorss[k] += indelss[k]
                # # score = np.append(score,wrongin)
                # aliseq1.append(seq1)
                # aliseq2.append(seq2)
                # mismatchnums.append(mismatch)
                # delnums.append(delnum)
                # insertnums.append(insertnum)
                # mismatchnum0+=int(mismatch)
                # delnum0+=int(delnum)
                # insertnum0+=int(insertnum)
                ori_seqs.append(ori)
                # pre_seqs.append(pre)
                pre_seqs.append(dna_sequence_pulsdel)
                predict_seqs.append(predict)
                all_dis.append(dis)
                seqsss = convert_to_DNA0(max_indices[i])
                max_indicesnew.append(seqsss)
                # scorelt09 = []
                phred = ''
                num = 0
                for j in range(len(prewith_phred_pulsdel)):
                    if ratel < prewith_phred_pulsdel[j] <= rater:
                        num += 1
                        # scorelt09.append({i, np.max(allpredict[i])})
                    phred += str(j) + ':' + str(prewith_phred_pulsdel[j]) + ' '
                phred_scores.append(phred)
                phred = ''
                for j in range(len(prewith_phred)):
                    phred += str(j) + ':' + str(np.max(prewith_phred[j])) + ' '
                phredwith_scores.append(phred)

                all_nums += num
                # if num > 0 and dis > 0:
                #     TP += 1
                # elif num > 0 and dis == 0:
                #     FN += 1
                # elif num == 0 and dis > 0:
                #     FP += 1
                # elif num == 0 and dis == 0:
                #     TN += 1
                if dis >= 1:
                    all_dis_gl0 += 1
                    all_dis_sum += dis
                    # all_dis_sum += errornums
                    all_nums_gt0dis += num
                    # allscorelt09s.append(upnumsindex)
                # else:

                # if dis != int(mismatch)+int(delnum)+int(insertnum):
                #     print(f"dis:{dis}  mismatchnum0:{mismatch} delnum0:{delnum} insertnum0:{insertnum}")
                #     print(f"ori:{ori}\npre:{dna_sequence_pulsdel}")
                allscorelt09s.append(upnumsindex)
                preupnums += upnums
                afterphredupdatedis += dis - upnums

        print('预测碱基概率大于' + str(ratel) + '小于等于:' + str(rater) + '时，序列数量总共有:' + str(
            TP + FN) + '\n所有序列中，在此概率区间碱基数量总共有:' + str(all_nums)
              + ' 发生错误的序列中，在此概率区间的碱基数量有:' + str(all_nums_gt0dis)
              + ' 碱基概率在此区间，全部发生错误的碱基数量 :' + str(preupnums))
        print(upnumsallerrorss)
        # print(f"errornums:{errornums}")
        # mydict = {'0':0,'1':0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0,'8':0,'9':0}
        # for s in score:
        #     mydict[str(int(s*10))]+=1
        # for key,value in mydict.items():
        #     if value>0:
        #         print(f"{key}:{value} ")
        # print(mydict)
        rate = rater
    tillnow = datetime.now()
    print(f"mismatch数量:{mismatchnum0} del数量:{delnum0} insert数量:{insertnum0} ")
    print('发生错误的序列数量:' + str(all_dis_gl0) + ' 发生错误数量占所有数据： ' + str(all_dis_gl0 / lens))
    print('发生错误的碱基数量 :' + str(all_dis_sum))
    print('每条序列碱基数量：' + str(len(ori_seqs[0])))
    # print('dis > 1 number:' + str(all_dis_gl0) + ' 发生错误数量占所有数据： ' + str(all_dis_gl0 / lens))
    print('平均编辑距离:' + str(all_dis_sum / lens))
    if all_dis_gl0 != 0:
        print('when dis > 1 平均编辑距离:' + str(all_dis_sum / all_dis_gl0))
    print('recovery rate:' + str(1 - all_dis_sum / lens / len(ori_seqs[0])))

    with open('./myfiles/oriandpreseqs.fasta', 'w') as file:
        for i in range(len(ori_seqs)):
            # file.write(f">oriseq{i}\n{ori_seqs[i]}\n>preseq{i} edit:{all_dis[i]} mis:{mismatchnums[i]} del:{delnums[i]} insert:{insertnums[i]}\n{pre_seqs[i]}\n>phred{i} scorelt09num:{scorelt09s[i]}\n{phred_scores[i]}\n")
            file.write(
                f">oriseq{i}\n{ori_seqs[i]}\n>preseq{i} edit:{all_dis[i]} mis:{mismatchnums[i]} del:{delnums[i]} insert:{insertnums[i]}\n{pre_seqs[i]}\n"
                f">aliseqs{i}\n{aliseq1[i]}\n{aliseq2[i]}\n>phred{i}  scorelt09num:{allscorelt09s[i]}\n {phred_scores[i]}\n")
        # rate = rater

    with open('./myfiles/oriandpreseqswith_.fasta', 'w') as file:
        for i in range(len(ori_seqs)):
            file.write(
                f">preseq{i}\n{pre_seqs[i]}\n>prewith_seqs{i}\n{predict_seqs[i]}\n>phredwith_scores{i}\n{phredwith_scores[i]}\n")

    seq_len = len(ori_seqs[0])
    data = [
        # ('', 'rec rate', 'edit error rate', 'error rate', 'success rate', 'indels', 'error base num', 'error seq num'),
        ('dp', 1 - all_dis_sum / lens / seq_len, all_dis_sum / lens / seq_len, all_dis_gl0 / lens,
         1 - all_dis_gl0 / lens,
         f"{mismatchnum0}:{delnum0}:{insertnum0}", f"{all_dis_sum}", f"{all_dis_gl0}"),
    ]
    print(data)
    data = [
        (f'dp', 1 - all_dis_sum / lens / seq_len, all_dis_sum / lens / seq_len, all_dis_gl0 / lens,
         1 - all_dis_gl0 / lens
         , mismatchnum0 / lens / seq_len, delnum0 / lens / seq_len, insertnum0 / lens / seq_len,
         f"{mismatchnum0}:{delnum0}:{insertnum0}", all_dis_gl0, all_dis_sum, tillnow - start_time + datatime),
    ]
    return data


def count_bsalign_acc11(all_consus, all_ori_seqs, copy_num, data_time):
    start_time = datetime.now()
    lens = len(all_consus)
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
    print('发生错误的碱基数量 :' + str(all_dis_sum) + ' >=5错误：' + str(all_dis_summ10) + ' 总错误：' + str(
        all_dis_summ10 + all_dis_sum))
    print('每条序列碱基数量：' + str(len(all_ori_seqs[0])))
    print('总错误平均编辑距离:' + str((all_dis_summ10 + all_dis_sum) / lens) + '<5平均编辑距离:' + str(all_dis_sum / lens))
    if all_dis_gl0 > 0:
        print('when dis > 1 平均编辑距离:' + str(all_dis_summ10 + all_dis_sum / all_dis_gl0))
    print('recovery rate:' + str(1 - (all_dis_summ10 + all_dis_sum) / lens / len(all_ori_seqs[0])))

    with open('./myfiles/bsalign_oriandpreseqs.fasta', 'w') as file:
        for i in range(len(all_ori_seqs)):
            # file.write(f">oriseq{i}\n{all_ori_seqs[i]}\n>preseq{i}   {all_dis[i]}\n{all_consus[i]}\n")
            file.write(
                f">oriseq{i}\n{all_ori_seqs[i]}\n>preseq{i} edit:{all_dis[i]} mis:{mismatchnums[i]} del:{delnums[i]} insert:{insertnums[i]}\n"
                f"{all_consus[i]}\n")

    seq_len = len(all_ori_seqs[0])
    data = [
        # ('', 'rec rate', 'edit error rate', 'error rate', 'success rate', 'indels', 'error base num', 'error seq num'),
        (f'bsalign{copy_num}', 1 - (all_dis_summ10 + all_dis_sum) / lens / seq_len,
         (all_dis_summ10 + all_dis_sum) / lens / seq_len, all_dis_gl0 / lens, 1 - all_dis_gl0 / lens,
         f"{sum(mismatchnums)}:{sum(delnums)}:{sum(insertnums)}", f"{all_dis_summ10 + all_dis_sum}", f"{all_dis_gl0}"),
    ]
    print(data)
    data = [
        # ('', 'rec rate', 'edit error rate', 'error rate', 'success rate', 'sub', 'del', 'insert', 'indels',
        #  'error seq num', 'error base num', 'time'),
        (f"bsalign{copy_num}", 1 - (all_dis_summ10 + all_dis_sum) / lens / len(all_ori_seqs[0]),
         (all_dis_summ10 + all_dis_sum) / lens / len(all_ori_seqs[0]), all_dis_gl0 / lens, 1 - all_dis_gl0 / lens
         , sum(mismatchnums) / lens / len(all_ori_seqs[0]), sum(delnums) / lens / len(all_ori_seqs[0]),
         sum(insertnums) / lens / len(all_ori_seqs[0]),
         f"{sum(mismatchnums)}:{sum(delnums)}:{sum(insertnums)}", all_dis_gl0, all_dis_summ10 + all_dis_sum,
         tillnow - start_time + data_time),
    ]
    # with open('files/trellis_ldpc_1120.csv', 'a', encoding='utf8', newline='') as f:
    # with open('derrick_model/derrick20_bs_dp_3_10.csv', 'a', encoding='utf8', newline='') as f:

    return data


def count_bsalign_acc_phred_1216ori(all_consus, all_ori_seqs, all_bsalign_quas, copy_num):
    lens = len(all_consus)
    rate = 0.0
    while rate < 0.99:
        # while rate < 0.9999:
        # if rate < 0.399:
        #     ratel = rate
        #     rater = rate + 0.1
        # else:
        #     ratel = rate
        #     rater = rate + 0.05
        # ratel = round(ratel, 2)
        # rater = round(rater, 2)
        ratel = rate
        rater = rate + 0.6
        rate = rater
        # lens=len(all_consus)
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
        aliseq1, aliseq2 = [], []
        all_dis_sum = 0
        all_nums = 0
        print("ratel:" + str(ratel) + " rater:" + str(rater))
        upnumsallerrorss = [0, 0, 0]
        score = []
        afterphredupdatedis = 0
        for i in range(len(all_consus)):
            dis = Levenshtein.distance(all_consus[i], all_ori_seqs[i])
            # phredscore = getphred_acc(all_bsalign_quas[i])
            phredscore = all_bsalign_quas[i]
            if len(phredscore) != len(all_consus[i]):
                print(str(all_consus[i]))
            # phredscore = all_bsalign_quas[i]
            # if(dis>10):
            #     print('bsalign')
            #     print(dis)
            #     print(all_consus[i])
            #     print(all_ori_seqs[i])
            # mismatch, delnum, insertnum, seq1, seq2, upnums,upnumsindex,wrongin,upnumsallerrors =bsalign_alitest(all_ori_seqs[i], all_consus[i],phredscore, ratel, rater, dis)
            mismatch, delnum, insertnum, seq1, seq2, upnums, upnumsindex, wrongin, upnumsallerrors = bsalign_alitest(
                all_ori_seqs[i], all_consus[i], phredscore, ratel, rater, dis, all_consus[i], all_bsalign_quas[i])
            edit_ops = Levenshtein.editops(all_ori_seqs[i], all_consus[i])
            insertnum = sum(1 for op in edit_ops if op[0] == 'insert')
            delnum = sum(1 for op in edit_ops if op[0] == 'delete')
            mismatch = sum(1 for op in edit_ops if op[0] == 'replace')
            mismatch, delnum, insertnum = int(mismatch), int(delnum), int(insertnum)
            # errornums = int(mismatch)+int(delnum)+int(insertnum)
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
                phred += str(j) + ':' + str(phredscore[j]) + ' '
            phred_scores.append(phred)
            all_nums += num
            if dis >= 1:
                all_dis_gl0 += 1
                all_dis_sum += dis
                # all_dis_sum += errornums
                all_nums_gt0dis += num
            if dis > 10:
                pre_seqs_morethan10.append(all_consus[i])
            preupnums += upnums
            afterphredupdatedis += dis - upnums
        print('预测碱基概率大于' + str(ratel) + '小于等于:' + str(rater) + '时'
              + '\n所有序列中，在此概率区间碱基数量总共有:' + str(all_nums)
              + ' 发生错误的序列中，在此概率区间的碱基数量有:' + str(all_nums_gt0dis)
              + ' 碱基概率在此区间，全部发生错误的碱基数量 :' + str(preupnums))
        # print(score)
        print(upnumsallerrorss)
        mydict = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0}
        for s in score:
            mydict[str(int(s * 10))] += 1
        for key, value in mydict.items():
            if value > 0:
                print(f"{key}:{value} ")
    print(f"错误数量超过10的序列数量:{pre_seqs_morethan10}")
    print(f"mismatch数量:{mismatchnum0} del数量:{delnum0} insert数量:{insertnum0} ")
    print('发生错误的序列数量:' + str(all_dis_gl0) + ' 发生错误数量占所有数量： ' + str(all_dis_gl0 / lens))
    print('发生错误的碱基数量 :' + str(all_dis_sum))
    print('每条序列碱基数量：' + str(len(all_ori_seqs[0])))
    print('平均编辑距离:' + str(all_dis_sum / lens))
    print('when dis >= 1 平均编辑距离:' + str(all_dis_sum / all_dis_gl0))
    print('recovery rate:' + str(1 - all_dis_sum / lens / len(all_ori_seqs[0])))

    with open('./myfiles/bsalign_oriandpreseqs.fasta', 'w') as file:
        for i in range(len(all_ori_seqs)):
            # file.write(f">oriseq{i}\n{all_ori_seqs[i]}\n>preseq{i}   {all_dis[i]}\n{all_consus[i]}\n")
            # file.write(f">oriseq{i}\n{all_ori_seqs[i]}\n>preseq{i} edit:{all_dis[i]} mis:{mismatchnums[i]} del:{delnums[i]} insert:{insertnums[i]}\n{all_consus[i]}\n"
            #            f"phred{i}:{all_phred_score[i]}\n")
            file.write(f">oriseq{i}\n{all_ori_seqs[i]}\n>preseq{i} edit:{all_dis[i]} mis:{mismatchnums[i]}"
                       f" del:{delnums[i]} insert:{insertnums[i]}\n{all_consus[i]}\n"
                       f">aliseqs{i}\n{aliseq1[i]}\n{aliseq2[i]}\n>phredwith_scores{i}\n{phred_scores[i]}\n")
    seq_len = len(all_ori_seqs[0])
    data = [
        # ('', 'rec rate', 'edit error rate', 'error rate', 'success rate', 'indels', 'error base num', 'error seq num'),
        (f"bsalign{copy_num}", 1 - (all_dis_sum) / lens / seq_len, (all_dis_sum) / lens / seq_len, all_dis_gl0 / lens,
         1 - all_dis_gl0 / lens,
         f"{sum(mismatchnums)}:{sum(delnums)}:{sum(insertnums)}", f"{all_dis_sum}", f"{all_dis_gl0}"),
    ]
    print(data)
    data = [
        # ('', 'rec rate', 'edit error rate', 'error rate', 'success rate', 'sub', 'del', 'insert', 'indels',
        #  'error seq num', 'error base num', 'time'),
        (f"bsalign{copy_num}", 1 - (all_dis_sum) / lens / len(all_ori_seqs[0]),
         (all_dis_sum) / lens / len(all_ori_seqs[0]), all_dis_gl0 / lens, 1 - all_dis_gl0 / lens
         , sum(mismatchnums) / lens / len(all_ori_seqs[0]), sum(delnums) / lens / len(all_ori_seqs[0]),
         sum(insertnums) / lens / len(all_ori_seqs[0]),
         f"{sum(mismatchnums)}:{sum(delnums)}:{sum(insertnums)}", all_dis_gl0, all_dis_sum),
    ]
    # with open('files/trellis_ldpc_1120.csv', 'a', encoding='utf8', newline='') as f:
    # with open('derrick_model/derrick20_bs_dp_3_10.csv', 'a', encoding='utf8', newline='') as f:

    return data

def count_bsalign_acc_phred(all_consus, all_ori_seqs, all_bsalign_quas, copy_num):
    lens = len(all_consus)
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
    aliseq1 = []
    all_nums_gt0dis = 0
    all_dis_gl0 = 0
    all_dis_sum = 0
    all_nums = 0
    for i in range(len(all_consus)):
        dis = Levenshtein.distance(all_consus[i], all_ori_seqs[i])
        phredscore = all_bsalign_quas[i]
        # if len(phredscore) != len(all_consus[i]):
        #     print(str(all_consus[i]))
        # mismatch, delnum, insertnum, seq1, seq2, upnums, upnumsindex, wrongin, upnumsallerrors = bsalign_alitest(
        #     all_ori_seqs[i], all_consus[i], phredscore, ratel, rater, dis, all_consus[i], all_bsalign_quas[i])
        edit_ops = Levenshtein.editops(all_ori_seqs[i], all_consus[i])
        insertnum = sum(1 for op in edit_ops if op[0] == 'insert')
        delnum = sum(1 for op in edit_ops if op[0] == 'delete')
        mismatch = sum(1 for op in edit_ops if op[0] == 'replace')

        aliseq1.append(edit_ops)
        all_phred_score.append(phredscore)
        mismatchnums.append(mismatch)
        delnums.append(delnum)
        insertnums.append(insertnum)
        mismatchnum0 += int(mismatch)
        delnum0 += int(delnum)
        insertnum0 += int(insertnum)
        all_dis.append(dis)
        num = 0
        phred = ''
        for j in range(len(phredscore)):
            phred += str(j) + ':' + str(phredscore[j]) + ' '
        phred_scores.append(phred)
        all_nums += num
        if dis >= 1:
            all_dis_gl0 += 1
            all_dis_sum += dis
            all_nums_gt0dis += num
        if dis > 10:
            pre_seqs_morethan10.append(all_consus[i])

    print(f"错误数量超过10的序列数量:{len(pre_seqs_morethan10)}")
    print(f"mismatch数量:{mismatchnum0} del数量:{delnum0} insert数量:{insertnum0} ")
    print('发生错误的序列数量:' + str(all_dis_gl0) + ' 发生错误数量占所有数量： ' + str(all_dis_gl0 / lens))
    print('发生错误的碱基数量 :' + str(all_dis_sum))
    print('每条序列碱基数量：' + str(len(all_ori_seqs[0])))
    print('平均编辑距离:' + str(all_dis_sum / lens))
    print('when dis >= 1 平均编辑距离:' + str(all_dis_sum / all_dis_gl0))
    print('recovery rate:' + str(1 - all_dis_sum / lens / len(all_ori_seqs[0])))
    with open('./myfiles/bsalign_oriandpreseqs.fasta', 'w') as file:
        for i in range(len(all_ori_seqs)):
            # file.write(f">oriseq{i}\n{all_ori_seqs[i]}\n>preseq{i}   {all_dis[i]}\n{all_consus[i]}\n")
            # file.write(f">oriseq{i}\n{all_ori_seqs[i]}\n>preseq{i} edit:{all_dis[i]} mis:{mismatchnums[i]} del:{delnums[i]} insert:{insertnums[i]}\n{all_consus[i]}\n"
            # #            f"phred{i}:{all_phred_score[i]}\n")
            # file.write(f">oriseq{i}\n{all_ori_seqs[i]}\n>preseq{i} edit:{all_dis[i]} mis:{mismatchnums[i]}"
            #            f" del:{delnums[i]} insert:{insertnums[i]}\n{all_consus[i]}\n"
            #            f">aliseqs{i}\n{aliseq1[i]}\n{aliseq2[i]}\n>phredwith_scores{i}\n{phred_scores[i]}\n")
            file.write(f">oriseq{i}\n{all_ori_seqs[i]}\n>preseq{i} edit:{all_dis[i]} mis:{mismatchnums[i]}"
                       f" del:{delnums[i]} insert:{insertnums[i]}\n{all_consus[i]}\n"
                       f">ali{i}\n{aliseq1[i]}\n>phredwith_scores{i}\n{phred_scores[i]}\n")
    seq_len = len(all_ori_seqs[0])
    data = [
        # ('', 'rec rate', 'edit error rate', 'error rate', 'success rate', 'indels', 'error base num', 'error seq num'),
        (f"bsalign{copy_num}", 1 - (all_dis_sum) / lens / seq_len, (all_dis_sum) / lens / seq_len, all_dis_gl0 / lens,
         1 - all_dis_gl0 / lens,
         f"{sum(insertnums)}:{sum(delnums)}:{sum(mismatchnums)}", f"{all_dis_sum}", f"{all_dis_gl0}"),
    ]
    print(data)
    data = [
        # ('', 'rec rate', 'edit error rate', 'error rate', 'success rate', 'sub', 'del', 'insert', 'indels',
        #  'error seq num', 'error base num', 'time'),
        (f"bsalign{copy_num}", 1 - (all_dis_sum) / lens / len(all_ori_seqs[0]),
         (all_dis_sum) / lens / len(all_ori_seqs[0]), all_dis_gl0 / lens, 1 - all_dis_gl0 / lens
         , sum(mismatchnums) / lens / len(all_ori_seqs[0]), sum(delnums) / lens / len(all_ori_seqs[0]),
         sum(insertnums) / lens / len(all_ori_seqs[0]),
         f"{sum(mismatchnums)}:{sum(delnums)}:{sum(insertnums)}", all_dis_gl0, all_dis_sum),
    ]
    # with open('./models/derrick_seqcluster_10000_b64/derrick_bs_dp_phred_1219.csv', 'a', encoding='utf8', newline='') as f:
    #     writer = csv.writer(f)  # csv.writer()中可以传一个文件对象
    #     for line in data:  # 该data既可以是列表嵌套列表的数据类型也可以是列表嵌套元组的数据类型
    #         writer.writerow(line)
    # getalians(all_ori_seqs, all_consus, all_dis, all_phred_score,'bsalign',copy_num)
    # with open('files/trellis_ldpc_1120.csv', 'a', encoding='utf8', newline='') as f:
    # with open('derrick_model/derrick20_bs_dp_3_10.csv', 'a', encoding='utf8', newline='') as f:
    # getnewcountbsalign(all_ori_seqs,all_consus,all_phred_score,all_dis,copy_num)


    # filepath='./models/derrick_seqcluster_10000_b64/derrick_bs_dp_phred_1219_lt0_detail.csv'
    # ratetest = [[0, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.45], [0.45, 0.5], [0.5, 0.55],
    #             [0.55, 0.6], [0.6, 0.65], [0.65, 0.7], [0.7, 0.75], [0.75, 0.8], [0.8, 0.85], [0.85, 0.9], [0.9, 0.95], [0.95, 1]]
    # getalians(all_ori_seqs, all_consus, all_dis, all_phred_score,ratetest,filepath,data,'bsalign',copy_num)

    # filepath='./models/derrick_seqcluster_10000_b64/derrick_bs_dp_phred_1220_lt0.csv'
    # ratetest = [[0, 0.2],[0.2, 0.4], [0.4, 0.5], [0.5, 0.6], [0.6, 0.7], [0.7, 0.8], [0.8, 0.9], [0.9, 1]]
    # getalians(all_ori_seqs, all_consus, all_dis, all_phred_score,ratetest,filepath,data,'bsalign',copy_num)

    return data

def getnewcountbsalign(all_ori_seqs,all_consus,all_phred_score,all_dis,copy_num):
    ratetest = [[0, 0.6], [0.6, 1]]
    for ratel, rater in ratetest:
        seqnums, basenums, errorseq_basenums, truebasenums = 0, 0, 0, 0
        upnumsallerrorss = {'insert': 0, 'delete': 0, 'replace': 0, 'equal': 0}
        for i in range(len(all_ori_seqs)):
            indelss = {'insert': 0, 'delete': 0, 'replace': 0, 'equal': 0}
            seq1, seq2, allpredict, ratel, rater, dis, dna_sequence_pulsdel, prewith_phred_pulsdel, dir = \
                all_ori_seqs[i], all_consus[i], '', ratel, rater, all_dis[i], all_consus[i], all_phred_score[i], 'files'
            with open(dir + '/seqstest.fasta', 'w') as file:
                for j, cus in enumerate([seq1, dna_sequence_pulsdel]):
                    file.write('>' + str(j) + '\n')
                    file.write(str(cus) + '\n')
            shell = '/home1/hongmei/00work_files/1229/DNAtransformer/0122bsalign/bsalign-master/bsalign align ' + dir + '/seqstest.fasta >' + dir + '/alitest.ali'
            result = subprocess.run(shell, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if dis >= 1:
                seqnums += 1
                mismatch, delnum, insertnum, seq1, seq2, upnums, upnumsindex, wrongin, upnumsallerrors = readbsalignalignfile(
                    dir + '/alitest.ali', allpredict, ratel, rater, dis, dna_sequence_pulsdel, prewith_phred_pulsdel)
                mismatch, delnum, insertnum = int(mismatch),int(delnum),int(insertnum)
                indelss['insert'] += insertnum
                indelss['delete'] += delnum
                indelss['replace'] += mismatch
            truebasenums += sum(value for key, value in indelss.items() if key != 'equal')
            for key, v in indelss.items():
                upnumsallerrorss[key] += v

        data = [
            ('', copy_num, ratel, rater, seqnums, basenums, errorseq_basenums, truebasenums, upnumsallerrorss),
            # ('', ratel, rater, seqnums, basenums, errorseq_basenums, truebasenumsarr[k], upnumsallerrorssarr[k]),
        ]
        print(data)


