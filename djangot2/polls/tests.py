import Levenshtein
from django.test import TestCase
allseqs = ['AGACGTGTGCTCTTCCCAATCTTTCGTGCTTTGGCATGAAAGTGTCGGAATGCTTGTTTCAGAAGAAGAGCCCAACCAAGCGCGACACCGAGATCACGCAAGAAGCCCCTAGCTACGTCTGGCCCGTTCCCGGTTGAAAGATGGAAGAGCGTCAGTCT',
           'AGACGTGTGCTCTTCCGATCTTTGGACGGGGTTGTTTCTATCACTCGTGCTTTCGTTGTGTTTTTGTGGTTTCTTGATGTAAGGTGTTGTGTTGTTGTGAGATCCAGCGCGTACATATGACGTACTTGTTGGACTTCAAAATCAGAAGAGCGTCGTGT',
           'ACACGACGCTGTTTCCGATCTTGAAGTCCAATTAATAGCAAAATTACTTCAGTAAACCCAGACGCTCTTACGGATATAGGACGCCATCTTGTGATATTCAATCGTCCCCTATGTCGAATAGTGAACCCACCATGCAACGAGATCGGAAGAGCACACGTC']

primers_0 = ["ACACGACGCTCTTCCGATCT", "AGACGTGTGCTCTTCCGATCT"]
primers_1 = ["AGATCGGAAGAGCACACGTCT", "AGATCGGAAGAGCGTCGTGT"]
sequence_length = 161
mi ,mx = sequence_length-sequence_length*0.2,sequence_length+sequence_length*0.2
allseqs = [s for s in allseqs if mi < len(s) < mx ]
lenz = 5
s1,s2=[],[]
def test1():
    for i in range(len(allseqs)):
        lennow = len(primers_0[0])
        positive = True
        while lennow > lenz:
            p00, p01 = allseqs[i].find(primers_0[0][-lennow:]), allseqs[i].rfind(primers_1[0][:lennow])
            thiseq = ''
            if p00 != -1 and p01 != -1:
                thiseq = allseqs[i][p00 + lennow:p01]
            elif p00 != -1:
                thiseq = allseqs[i][p00 + lennow:p00 + lennow + sequence_length]
            elif p01 != -1:
                bi = 0
                if p01 - sequence_length > 0:
                    bi = p01 - sequence_length
                thiseq = allseqs[i][bi:p01]
            else:
                # 尝试反向序列
                positive = False
                p10, p11 = allseqs[i].find(primers_0[1][-lennow:]), allseqs[i].rfind(primers_1[1][:lennow])
                if p10 != -1 and p11 != -1:
                    thiseq = allseqs[i][p00 + lennow:p11]
                elif p10 != -1:
                    thiseq = allseqs[i][p10 + lennow:p10 + lennow + sequence_length]
                elif p11 != -1:
                    bi = 0
                    if p11 - sequence_length > 0:
                        bi = p11 - sequence_length
                    thiseq = allseqs[i][bi:p11]
                else:
                    positive = True
                    lennow -= 2
            if thiseq != '':
                if positive:
                    s1.append(thiseq)
                else:
                    s2.append(thiseq)
                break

def ttt(posseqs,lp,rp,sequence_length=120):
    s = []
    for i in range(len(posseqs)):
        lennow = len(lp)
        while lennow > lenz:
            p00, p01 = posseqs[i].find(lp[-lennow:]), posseqs[i].rfind(rp[:lennow])
            thiseq = ''
            if p00 != -1 and p01 != -1:
                thiseq = posseqs[i][p00 + lennow:p01]
            elif p00 != -1:
                thiseq = posseqs[i][p00 + lennow:p00 + lennow + sequence_length]
            elif p01 != -1:
                bi = 0
                if p01 - sequence_length > 0:
                    bi = p01 - sequence_length
                thiseq = posseqs[i][bi:p01]
            if thiseq != '':
                s.append(thiseq)
                break
            else:
                lennow -= 1
    return s
def aa2():
    posseqs,zposseqs = [],[]
    s1seqs,s2seqs = [],[]
    for i in range(len(allseqs)):
        seq = allseqs[i]
        ldis,rdis = Levenshtein.distance(primers_0[0][:8],seq[:8]) , Levenshtein.distance(primers_1[0][-8:],seq[-8:])
        zldis,zrdis = Levenshtein.distance(primers_0[1][:8],seq[:8]) , Levenshtein.distance(primers_1[1][-8:],seq[-8:])
        dis,zdis = ldis+rdis,zldis+zrdis
        if dis <= zdis:
            posseqs.append(seq)
        else:
            zposseqs.append(seq)
    print(ttt(posseqs,primers_0[0],primers_1[0]))
    print(ttt(posseqs,primers_0[1],primers_1[1]))


aa2()
# print(f"s1:{s1}")
# print(f"s2:{s2}")