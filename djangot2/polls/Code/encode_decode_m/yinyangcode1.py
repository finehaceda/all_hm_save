import math
import random

import numpy
from re import search
from .utils import index_base, base_index
max_iterations = 100
max_ratio = 0.8
max_content = 0.6
virtual_nucleotide="A"
yang_rule = [0, 1, 0, 1]
yin_rule = [[1, 1, 0, 0], [1, 0, 0, 1], [1, 1, 0, 0], [1, 1, 0, 0]]

# base_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
# index_base = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

def _bits_to_nucleotide(upper_bit, lower_bit, support_nucleotide):
    current_options = []
    for index in range(len(yang_rule)):
        if yang_rule[index] == upper_bit:
            current_options.append(index)

    if yin_rule[base_index.get(support_nucleotide)][current_options[0]] == lower_bit:
        return index_base[current_options[0]]
    else:
        return index_base[current_options[1]]


def gc_content(sequence, max_content):
    return (1 - max_content) <= (float(sequence.count("C") + sequence.count("G")) / float(len(sequence))) <= max_content

def homopolymer(sequence, max_homopolymer):
    homopolymers = "A{%d,}|C{%d,}|G{%d,}|T{%d,}" % tuple([1 + max_homopolymer] * 4)
    return False if search(homopolymers, sequence) else True

def check(sequence, max_homopolymer, max_content):
    if max_homopolymer and not homopolymer(sequence, max_homopolymer):
        return False
    if max_content and not gc_content(sequence, max_content):
        return False
    return True

def addition(fixed_bit_segment, total_count,index_length,max_homopolymer):
    times = 0
    # while True:
    while times < 100:
        times +=1
        # insert at least 2 interval.
        random_index = random.randint(total_count + 3, math.pow(2, index_length) - 1)
        random_segment = list(map(int, list(str(bin(random_index))[2:].zfill(index_length))))

        dna_sequence = [[], []]
        support_nucleotide_1 = virtual_nucleotide
        support_nucleotide_2 = virtual_nucleotide

        for bit_1, bit_2 in zip(fixed_bit_segment[: index_length], random_segment):
            current_nucleotide_1 = _bits_to_nucleotide(bit_1, bit_2, support_nucleotide_1)
            current_nucleotide_2 = _bits_to_nucleotide(bit_2, bit_1, support_nucleotide_2)
            dna_sequence[0].append(current_nucleotide_1)
            dna_sequence[1].append(current_nucleotide_2)
            support_nucleotide_1 = current_nucleotide_1
            support_nucleotide_2 = current_nucleotide_2

        work_flags = [True, True]
        for fixed_bit in fixed_bit_segment[index_length:]:
            current_nucleotide_1, current_nucleotide_2 = None, None
            for bit in [0, 1]:
                if work_flags[0] and current_nucleotide_1 is None:
                    current_nucleotide_1 = _bits_to_nucleotide(fixed_bit, bit, support_nucleotide_1)
                    if not check("".join(dna_sequence[0]) + current_nucleotide_1,
                                        max_homopolymer=max_homopolymer,
                                        max_content=max_content):
                        current_nucleotide_1 = None
                if work_flags[1] and current_nucleotide_2 is None:
                    current_nucleotide_2 = _bits_to_nucleotide(bit, fixed_bit, support_nucleotide_2)
                    if not check("".join(dna_sequence[1]) + current_nucleotide_2,
                                        max_homopolymer=max_homopolymer,
                                        max_content=max_content):
                        current_nucleotide_2 = None

            if current_nucleotide_1 is None:
                work_flags[0] = False
                dna_sequence[0] = None
            else:
                dna_sequence[0].append(current_nucleotide_1)
                support_nucleotide_1 = current_nucleotide_1

            if current_nucleotide_2 is None:
                work_flags[1] = False
                dna_sequence[1] = None
            else:
                dna_sequence[1].append(current_nucleotide_2)
                support_nucleotide_2 = current_nucleotide_2

        for potential_dna_sequence in dna_sequence:
            if potential_dna_sequence is not None and check("".join(potential_dna_sequence),
                                                                   max_homopolymer=max_homopolymer,
                                                                   max_content=max_content):
                return potential_dna_sequence

# def normal_encode(bit_segments,max_homopolymer,max_iterations):
def normal_encode(bit_segments,max_homopolymer):
    dna_sequences = []
    bad_data = []
    index_length = int(len(str(bin(len(bit_segments)))) - 2)
    total_count = len(bit_segments)
    for row in range(len(bit_segments)):
        if numpy.sum(bit_segments[row]) > len(bit_segments[row]) * max_ratio \
                or numpy.sum(bit_segments[row]) < len(bit_segments[row]) * (1 - max_ratio):
            bad_data.append(row)

    if len(bit_segments) < len(bad_data) * 5:
        print("There may be a large number of sequences that are difficult for synthesis or sequencing. "
              + "We recommend you to re-select the rule or take a new run.")
    if len(bad_data) == 0 and len(bit_segments) == 0:
        return []
    elif len(bad_data) == 0:
        good_data, band_data = [], []
        for row in range(len(bit_segments)):
            good_data.append(bit_segments[row])
    elif len(bad_data) == len(bit_segments):
        good_data, bad_data = [], []
        for row in range(len(bit_segments)):
            bad_data.append(bit_segments[row])
    else:
        x, y = [], []
        for row in range(len(bit_segments)):
            if row in bad_data:
                y.append(bit_segments[row])
            else:
                x.append(bit_segments[row])
        good_data, bad_data = x, y


    while len(good_data) + len(bad_data) > 0:
        if(len(good_data) + len(bad_data)==2):
            print(111)
        if len(good_data) > 0 and len(bad_data) > 0:
            fixed_bit_segment, is_finish, state = good_data.pop(), False, True
        elif len(good_data) > 0:
            fixed_bit_segment, is_finish, state = good_data.pop(), False, False
        elif len(bad_data) > 0:
            fixed_bit_segment, is_finish, state = bad_data.pop(), False, True
        else:
            raise ValueError("Wrong pairing for Yin-Yang Code!")

        for pair_time in range(max_iterations):
            if state:
                if len(bad_data) > 0:
                    selected_index = random.randint(0, len(bad_data) - 1)
                    selected_bit_segment = bad_data[selected_index]
                else:
                    break
            else:
                if len(good_data) > 0:
                    selected_index = random.randint(0, len(good_data) - 1)
                    selected_bit_segment = good_data[selected_index]
                else:
                    break

            dna_sequence = [[], []]
            support_nucleotide_1 = virtual_nucleotide
            support_nucleotide_2 = virtual_nucleotide
            for bit_1, bit_2 in zip(fixed_bit_segment, selected_bit_segment):
                current_nucleotide_1 = _bits_to_nucleotide(bit_1, bit_2, support_nucleotide_1)
                current_nucleotide_2 = _bits_to_nucleotide(bit_2, bit_1, support_nucleotide_2)
                dna_sequence[0].append(current_nucleotide_1)
                dna_sequence[1].append(current_nucleotide_2)
                support_nucleotide_1 = current_nucleotide_1
                support_nucleotide_2 = current_nucleotide_2

            if check("".join(dna_sequence[0]),
                            max_homopolymer=max_homopolymer, max_content=max_content):
                is_finish = True
                dna_sequences.append(dna_sequence[0])
                if state:
                    del bad_data[selected_index]
                else:
                    del good_data[selected_index]
                break
            elif check("".join(dna_sequence[1]),
                              max_homopolymer=max_homopolymer, max_content=max_content):
                is_finish = True
                dna_sequences.append(dna_sequence[1])
                if state:
                    del bad_data[selected_index]
                else:
                    del good_data[selected_index]
                break

        # additional information
        if not is_finish:
            dna_sequences.append(addition(fixed_bit_segment, total_count,index_length,max_homopolymer))


    return dna_sequences

def faster_encode(bit_segments,max_homopolymer):
    dna_sequences = []

    index_length = int(len(str(bin(len(bit_segments)))) - 2)
    total_count = len(bit_segments)
    while len(bit_segments) > 0:
        fixed_bit_segment, is_finish = bit_segments.pop(), False
        for pair_time in range(max_iterations):
            if len(bit_segments) > 0:
                selected_index = random.randint(0, len(bit_segments) - 1)
                selected_bit_segment = bit_segments[selected_index]

                dna_sequence = [[], []]
                support_nucleotide_1 = virtual_nucleotide
                support_nucleotide_2 = virtual_nucleotide
                for bit_1, bit_2 in zip(fixed_bit_segment, selected_bit_segment):
                    current_nucleotide_1 = _bits_to_nucleotide(bit_1, bit_2, support_nucleotide_1)
                    current_nucleotide_2 = _bits_to_nucleotide(bit_2, bit_1, support_nucleotide_2)
                    dna_sequence[0].append(current_nucleotide_1)
                    dna_sequence[1].append(current_nucleotide_2)
                    support_nucleotide_1 = current_nucleotide_1
                    support_nucleotide_2 = current_nucleotide_2

                if check("".join(dna_sequence[0]),
                                max_homopolymer=max_homopolymer, max_content=max_content):
                    is_finish = True
                    dna_sequences.append(dna_sequence[0])
                    del bit_segments[selected_index]
                    break
                elif check("".join(dna_sequence[1]),
                                  max_homopolymer=max_homopolymer, max_content=max_content):
                    is_finish = True
                    dna_sequences.append(dna_sequence[1])
                    del bit_segments[selected_index]
                    break

        # additional information
        if not is_finish:
            dna_sequences.append(addition(fixed_bit_segment, total_count,index_length,max_homopolymer))


    return dna_sequences









def decode(dna_sequences,index_length,total_count):
    bit_segments = []

    for sequence_index, dna_sequence in enumerate(dna_sequences):
        upper_bit_segment, lower_bit_segment = [], []

        support_nucleotide = virtual_nucleotide
        for current_nucleotide in dna_sequence:
            upper_bit = yang_rule[base_index[current_nucleotide]]
            lower_bit = yin_rule[base_index[support_nucleotide]][base_index[current_nucleotide]]
            upper_bit_segment.append(upper_bit)
            lower_bit_segment.append(lower_bit)
            support_nucleotide = current_nucleotide

        bit_segments.append(upper_bit_segment)
        bit_segments.append(lower_bit_segment)


    remain_bit_segments = []
    for bit_segment in bit_segments:
        segment_index = int("".join(list(map(str, bit_segment[:index_length]))), 2)
        if segment_index < total_count:
            remain_bit_segments.append(bit_segment)

    return remain_bit_segments



# def yinyangencode(bit_segments,max_homopolymer,max_iterations):
#     bit_segments = [[int(bit) for bit in bit_seq] for bit_seq in bit_segments]
#     # return faster_encode(bit_segments,max_homopolymer)
#     return normal_encode(bit_segments,max_homopolymer,max_iterations)


def yinyangencode(bit_segments,max_homopolymer):
    bit_segments = [[int(bit) for bit in bit_seq] for bit_seq in bit_segments]
    # return faster_encode(bit_segments,max_homopolymer)
    return normal_encode(bit_segments,max_homopolymer)



def yinyangdecode(dna_sequences,index_length,total_count):
    bit_segments = decode(dna_sequences, index_length, total_count)
    bit_segments = [''.join(map(str, numbers)) for numbers in bit_segments]
    return bit_segments
    # return normal_encode(bit_segments,max_homopolymer)

