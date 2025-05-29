import Levenshtein
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# from Evaluation_platform.dpconsensus.config import *
from dpconsensus.config import *
from torchvision.transforms import transforms


# 4.4 Mask掉停用词
def get_attn_pad_mask(seq_q, seq_k):                       # seq_q: [batch_size, seq_len] ,seq_k: [batch_size, seq_len]
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)          # 判断 输入那些含有P(=0),用1标记 ,[batch_size, 1, len_k]
    return pad_attn_mask.expand(batch_size, len_q, len_k)
# 4.5 Decoder 输入 Mask
def get_attn_subsequence_mask(seq):                               # seq: [batch_size, tgt_len]
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)          # 生成上三角矩阵,[batch_size, tgt_len, tgt_len]
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()  #  [batch_size, tgt_len, tgt_len]
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
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]
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
        self.enc_self_attn = MultiHeadAttention()                                     # 多头注意力机制
        self.pos_ffn = PoswiseFeedForwardNet()                                        # 前馈神经网络

    def forward(self, enc_inputs, enc_self_attn_mask):                                # enc_inputs: [batch_size, src_len, d_model]
        #输入3个enc_inputs分别与W_q、W_k、W_v相乘得到Q、K、V                          # enc_self_attn_mask: [batch_size, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,    # enc_outputs: [batch_size, src_len, d_model],
                                               enc_self_attn_mask)                    # attn: [batch_size, n_heads, src_len, src_len]
        # enc_outputs = self.pos_ffn(enc_outputs)                                       # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self,n_layers):
    # def __init__(self):
        super(Encoder, self).__init__()
        # self.src_emb = nn.Embedding(src_vocab_size, d_model)
        # self.src_emb = nn.Linear(src_vocab_size, src_vocab_size*d_model)
        # self.src_emb = nn.Linear(1, d_model)                     # 把字转换字向量
        # self.pos_emb = PositionalEncoding(d_model)
        self.pos_ffn = PoswiseFeedForwardNet()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=2)
    def forward(self, enc_inputs,enc_inputs1=None):
        tensor = torch.randn(enc_inputs.shape[0], enc_inputs.shape[1]).cuda()
        # tensor = torch.randn(enc_inputs.shape[0], enc_inputs.shape[1])
        enc_self_attn_mask = get_attn_pad_mask(tensor, tensor)           # enc_self_attn_mask: [batch_size, src_len, src_len]
        enc_self_attns = []
        # enc_outputs = enc_inputs
        enc_inputs = self.pos_ffn(enc_inputs)
        enc_outputs = self.conv1(enc_inputs.view(-1,1,enc_inputs.shape[1],d_model))
        enc_outputs = enc_outputs.view(-1,enc_outputs.shape[2],d_model)

        # enc_outputss = self.pos_ffn(enc_inputs1)
        # enc_outputss0 = self.conv1(enc_inputs1.view(-1,1,enc_inputs1.shape[1],d_model))
        # enc_outputss1 = enc_outputss0.view(-1,enc_outputss0.shape[2],d_model)

        for layer in self.layers:
            # enc_outputs, enc_self_attn = layer(enc_inputs, enc_self_attn_mask)  # enc_outputs :   [batch_size, src_len, d_model],
            enc_outputs, enc_self_attn = layer(enc_outputs,enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)

        enc_outputs = self.conv1(enc_outputs.view(-1,1,enc_outputs.shape[1],d_model))
        enc_outputs = self.pos_ffn(enc_outputs.view(-1,enc_outputs.shape[2],d_model))
        # enc_outputs = enc_outputs.view(-1,enc_outputs.shape[2],d_model)
        return enc_outputs, enc_self_attns


class MyLSTM(nn.Module):
    def __init__(self, input_size = d_model, hidden_size = 64, num_layers=2, output_size=decode_layer):
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
    def __init__(self,n_layers):
    # def __init__(self):
        super(Transformer, self).__init__()
        # self.projectionbegin = nn.Linear(tgt_vocab_size,d_model, bias=False)
        # self.Encoder = Encoder(n_layers).cuda()
        # self.Decoder = MyLSTM()
        # self.projection = nn.Linear(decode_layer, tgt_vocab_size, bias=False)

        # tow
        self.projectionbegin = nn.Linear(tgt_vocab_size, d_model, bias=False)
        self.ConvTranspose = nn.ConvTranspose1d(in_channels=tgt_vocab_size, out_channels=d_model, kernel_size=3,
                                                stride=1, padding=1)
        self.ConvTranspose1 = nn.ConvTranspose1d(in_channels=tgt_vocab_size, out_channels=d_model_hidden, kernel_size=3,
                                                 stride=1, padding=1)
        self.ConvTranspose2 = nn.ConvTranspose1d(in_channels=d_model_hidden, out_channels=d_model, kernel_size=3,
                                                 stride=1, padding=1)
        self.pos_ffn11 = PoswiseFeedForwardNet11()
        self.CNN = CNN()
        self.Encoder = Encoder(n_layers).cuda()
        self.Decoder = MyLSTM()
        self.projection = nn.Linear(decode_layer, tgt_vocab_size, bias=False)
        # self.projectionpre = nn.Linear(tgt_vocab_size, d_model, bias=False)
        self.normalize = MinMaxNormalize(feature_dim=d_model)
        self.standardize = Standardize(feature_dim=decode_layer)

    def forward(self, enc_inputs, dec_inputs=None):
        #第一版
        # enc_inputs = self.ConvTranspose(enc_inputs.permute(0, 2, 1)).permute(0, 2, 1) #128*4--->128*d_model
        # enc_outputs1, enc_self_attns1 = self.Encoder(enc_inputs1) #128*d_model--->128*d_model
        # dec_outputs1 = self.Decoder(enc_outputs1)  #128*d_model--->128*decode_layer
        # dec_logits1 = self.projection(dec_outputs1) #128*decode_layer--->128*tgt_vocab_size
        # 第二版
        enc_inputs = self.projectionbegin(enc_inputs) #128*4--->128*d_model

        enc_outputs, enc_self_attns = self.Encoder(enc_inputs) #128*d_model--->128*d_model

        dec_outputs = self.Decoder(enc_outputs)  #128*d_model--->128*decode_layer
        dec_logits = self.projection(dec_outputs) #128*decode_layer--->128*tgt_vocab_size
        #0.242
        # dec_logits = self.standardize(dec_logits)
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

def converttoDNAgetdel11(sequence,predictdata,rate = 0.9):
    prewith_phred=[]
    i=0
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
            if maxpre<rate:
                dna_sequence_pulsdel += '-'
                prewith_phred.append(np.max(predictdata[i]))
        i+=1
    return dna_sequence_pulsdel,prewith_phred

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
        else:
            dna_sequence_pulsdel += tinydict[digit]
            prewith_phred.append(maxpre)
        i+=1
    return dna_sequence_pulsdel,prewith_phred

def softmax0(your_array):
    normalized_array = (your_array) / (np.sum(your_array, axis=0))
    return normalized_array

def getEdit(model,enc_inputs,dec_inputs,dec_outputs):
    ori = convert_to_DNA(dec_outputs)
    # predict_dec_input = mytest(model, enc_inputs.view(1, -1).cuda())
    # predict, _, _, _ = model(enc_inputs.view(1, -1).cuda(),predict_dec_input)
    # predict, _, _, _ = model(enc_inputs.view(1, -1).cuda(),enc_inputs.view(1, -1).cuda())
    allpredict, _, _, _ = model(enc_inputs.view(1, enc_inputs.shape[0],-1).cuda(),dec_inputs.view(1, dec_inputs.shape[0],-1).cuda())
    allpredict1, _, _, _ = model(dec_inputs.view(1, dec_inputs.shape[0],-1).cuda(),dec_inputs.view(1, dec_inputs.shape[0],-1).cuda())
    # allpredict, _, _, _ = model(enc_inputs.view(1, enc_inputs.shape[0],-1),dec_inputs.view(1, dec_inputs.shape[0],-1))
    # allpredict1, _, _, _ = model(dec_inputs.view(1, dec_inputs.shape[0],-1),dec_inputs.view(1, dec_inputs.shape[0],-1))
    # predict, _, _, _ = model(enc_inputs.cuda(),enc_inputs.cuda())
    # predict, _, _, _ = model(enc_inputs.view(1, -1),enc_inputs.view(1, -1))
    # predict = allpredict.data.max(1, keepdim=True)[1]
    # allpredicts = torch.nn.functional.softmax(allpredict, dim=1)
    predictdata = (allpredict + allpredict1)
    # predictdata = (allpredict + allpredict1)
    # predictdata = allpredict*2
    # predictdata = allpredict
    predict = predictdata.data.max(1, keepdim=True)[1]

    # allpredicts = min_max_normalize_last_dim(predictdata)
    # predictdata = Standardize(tgt_vocab_size)(predictdata)
    predictdata11 = predictdata/6
    allpredicts = torch.nn.functional.softmax(predictdata11, dim=-1)
    # allpredicts = torch.nn.functional.softmax(allpredict, dim=-1)
    # allpredicts2 = torch.nn.functional.softmax(predictdata, dim=-1)
    # allpredicts3 = torch.nn.functional.softmax(predictdata*2, dim=-1)
    # allpredicts = torch.nn.functional.softmax(predictdata*3, dim=-1)
    # predictdata = Standardize(tgt_vocab_size)(predictdata)
    # allpredicts = min_max_normalize_last_dim(predictdata)
    # allpredicts = allpredicts+allpredicts1
    # allpredicts = torch.nn.functional.softmax(allpredicts, dim=-1)
    # allpredicts = allpredicts/torch.sum(allpredicts,dim=-1, keepdim=True)

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
    prewith_seq,prewith_phred,predictphred = convert_to_DNA_(predict,allpredicts.cpu().detach().numpy())
    # dna_sequence_pulsdel,prewith_phred_pulsdel = converttoDNAgetdel(predict,allpredicts.cpu().detach().numpy(),0)
    # dna_sequence_pulsdel,prewith_phred_pulsdel = converttoDNAgetdel(predict,allpredicts.cpu().detach().numpy(),0.5)
    dna_sequence_pulsdel,prewith_phred_pulsdel = converttoDNAgetdel(predict,allpredicts.cpu().detach().numpy(),0.6)
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
    return dis,ori,pre,prewith_seq,predictphred,prewith_phred,dna_sequence_pulsdel,prewith_phred_pulsdel
    # return dis,ori,pre,predict,allpredicts.cpu().detach().numpy()
    # return dis,ori,pre,predict,allpredicts

# def softmax(x):
#     exp_x = np.exp(x - np.max(x))  # 避免数值不稳定性，减去最大值
#     return exp_x / exp_x.sum(axis=1, keepdims=True)

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

def train11(loader,model):
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

def train(loader,model):
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99, weight_decay=0.001)
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)
    print("要加载的模型参数文件不存在！\n开始训练")
    losses = []
    for epoch in range(15):
        for enc_inputs, dec_inputs, dec_outputs, dec_outputsin in loader:
            enc_inputs, dec_inputs, dec_outputs = enc_inputs.cuda(), dec_inputs.cuda(), dec_outputs.cuda()
            outputs1, enc_self_attns1, dec_self_attns1, dec_enc_attns1 = model(dec_inputs, dec_inputs)
            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
            # outputs = Standardize(tgt_vocab_size)(outputs)
            # outputs = min_max_normalize_last_dim(outputs)
            # outputs = outputs/torch.sum(outputs,dim=-1, keepdim=True)
            outputs = torch.nn.functional.softmax(outputs, dim=-1)
            outputs1 = torch.nn.functional.softmax(outputs1, dim=-1)
            # outputs1 = Standardize(tgt_vocab_size)(outputs1)
            # outputs1 = min_max_normalize_last_dim(outputs1)
            # outputs1 = outputs1/torch.sum(outputs1,dim=-1, keepdim=True)
            # loss = criterion(outputs, dec_outputs.view(-1))*2
            # loss1 = criterion(outputs1, dec_outputs.view(-1))
            # loss =  loss1
            loss = criterion(outputs, dec_outputs.view(-1))
            loss1 = criterion(outputs1, dec_outputs.view(-1))
            loss = loss + loss1
            # loss = 0.3*loss1 + 0.7*loss2
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
        losses.append(loss.item())
    torch.save(model.state_dict(), 'modelsf6cmha.pth')
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




