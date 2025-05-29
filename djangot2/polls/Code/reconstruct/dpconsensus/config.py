import os

current_directory = os.getcwd()
# dpconsensus_path = current_directory+'/dpconsensus/files'
dpconsensus_path = current_directory+'/polls/Code/reconstruct/dpconsensus/files'
# d_model = 4  # 字 Embedding 的维度
d_model = 512 # 字 Embedding 的维度
# d_model = 128  # 字 Embedding 的维度
# d_model = 5  # 字 Embedding 的维度
d_model_hidden = 512  # 字 Embedding 的维度
d_ff = 1024     # 前向传播隐藏层维度
d_k = d_v = 64  # K(=Q), V的维度
# n_layers = 4  # 有多少个encoder和decoder
# n_layers = 6 # 有多少个encoder和decoder
n_heads = 8     # Multi-Head Attention设置为8
decode_layer = 64
# src_len = 130
# tgt_len = 128
# src_len = 128
# tgt_len = 128
# src_vocab_size = 4
# src_vocab_size = 128
tgt_vocab_size = 5
# src_idx2word = {src_vocab[key]: key for key in src_vocab}
# Encoder_input    Decoder_input        Decoder_output
# sentences = [['我 是 学 生 P', 'S I am a student', 'I am a student E'],  # S: 开始符号
#              ['我 喜 欢 学 习', 'S I like learning P', 'I like learning P E'],  # E: 结束符号
#              ['我 是 男 生 P', 'S I am a boy', 'I am a boy E']]  # P: 占位符号，如果当前句子不足固定长度用P占位

# src_vocab = {'P': 0, '我': 1, '是': 2, '学': 3, '生': 4, '喜': 5, '欢': 6, '习': 7, '男': 8}  # 词源字典  字：索引
# src_idx2word = {src_vocab[key]: key for key in src_vocab}
# src_vocab_size = len(src_vocab)  # 字典字的个数
# tgt_vocab = {'P': 0, 'S': 1, 'E': 2, 'I': 3, 'am': 4, 'a': 5, 'student': 6, 'like': 7, 'learning': 8, 'boy': 9}
# idx2word = {tgt_vocab[key]: key for key in tgt_vocab}  # 把目标字典转换成 索引：字的形式
# tgt_vocab_size = len(tgt_vocab)  # 目标字典尺寸
# src_len = len(sentences[0][0].split(" "))  # Encoder输入的最大长度
# tgt_len = len(sentences[0][1].split(" "))  # Decoder输入输出最大长度

# sentences = [['我 是 学 生 P', 'S I am a student', 'I am a student E'],  # S: 开始符号
#              ['我 喜 欢 学 习', 'S I like learning P', 'I like learning P E'],  # E: 结束符号
#              ['我 是 男 生 P', 'S I am a boy', 'I am a boy E']]  # P: 占位符号，如果当前句子不足固定长度用P占位
# src = torch.tensor([['我', '是', '学', '生', 'P'],
#                     ['我', '喜', '欢', '学', '习'],
#                     ['我', '是', '男', '生', 'P']])
# target = torch.tensor([['I', 'am', 'a' ,'student', 'E'],
#                     ['I', 'like' ,'learning', 'P', 'E'],
#                     ['I', 'am' ,'a', 'boy','P']])