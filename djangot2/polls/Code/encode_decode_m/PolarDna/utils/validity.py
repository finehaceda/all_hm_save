import math


def check(sequence, max_homopolymer=math.inf, max_content=1, min_free_energy=None):
    # 考虑到加入块内地址可能导致均聚物长度过长，这里加入一个限制，不允许前3个碱基是一样的，即最多前2个碱基是一样的，250317
    if not homopolymer(sequence, max_homopolymer):
        return False
    if not cg_content(sequence, max_content):
        return False
    if not first_3nt_check(sequence):
        return False
    return True

def check_no_3nt_check(sequence, max_homopolymer=math.inf, max_content=1, min_free_energy=None):
    # 用于检查加入块地址和块内地址后的DNA序列，是否满足兼容性要求
    if not homopolymer(sequence, max_homopolymer):
        return False
    if not cg_content(sequence, max_content):
        return False
    return True

def first_3nt_check(sequence):
    # 检查前3个碱基是否一样，250317
    if sequence[0] == sequence[1] == sequence[2]:
        return False
    else:
        return True

def homopolymer(sequence, max_homopolymer):
    if max_homopolymer > len(sequence):
        return True
    missing_segments = ["A" * (1 + max_homopolymer), "C" * (1 + max_homopolymer), "G" * (1 + max_homopolymer),
                        "T" * (1 + max_homopolymer)]
    for missing_segment in missing_segments:
        # 下面使用in来确定是否有（max_homoploymer+1）长度为同源多聚物出现在DNA序列中
        # python处理字符串也太方便了
        if missing_segment in "".join(sequence):
            return False
    return True


def cg_content(motif, max_content):
    return (1 - max_content) <= float(motif.count("C") + motif.count("G")) / float(len(motif)) <= max_content

def check_num(sequence, max_homopolymer=math.inf, max_content=1, min_free_energy=None):
    max_homopolymer_f_num, gc_content_f_num = 0, 0
    if not homopolymer(sequence, max_homopolymer):
        max_homopolymer_f_num = 1
    if not cg_content(sequence, max_content):
        gc_content_f_num = 1
    return max_homopolymer_f_num, gc_content_f_num


def list_score(dna_sequence):
    homopolymer = 1
    while True:
        found = False
        for nucleotide in ["A", "C", "G", "T"]:
            if nucleotide * (1 + homopolymer) in dna_sequence:
                found = True
                break
        if found:
            homopolymer += 1
        else:
            break
    gc_bias = abs((dna_sequence.count("G") + dna_sequence.count("C")) / len(dna_sequence) - 0.5)
    h_score = (1.0 - (homopolymer - 1) / 5.0) / 2.0 if homopolymer < 10 else 0
    c_score = (1.0 - gc_bias / 0.3) / 2.0 if gc_bias < 0.3 else 0
    score = h_score + c_score
    return score
