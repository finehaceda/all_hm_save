import math
import random
import sys
import numpy

from re import search
from collections import defaultdict
from datetime import datetime

base_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
index_base = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

class LFSR(object):

    def __init__(self):
        pass

    @staticmethod
    def lfsr(state, mask):
        result = state
        nbits = mask.bit_length() - 1
        while True:
            result = result << 1
            xor = result >> nbits
            if xor != 0:
                result ^= mask
            yield result

    @staticmethod
    def lfsr32p():
        return 0b100000000000000000000000011000101

    @staticmethod
    def lfsr32s():
        return 0b001010101

    def lfsr_s_p(self):
        return self.lfsr(self.lfsr32s(), self.lfsr32p())

class PRNG(object):

    def __init__(self, number, delta, c):
        self.number = number
        self.delta = delta
        self.c = c
        self.value = self.c * math.log(self.number / self.delta) * math.sqrt(self.number)
        self.cdf, self.degree = self.gen_rsd_cdf(number, self.value, self.delta)

    def get_src_blocks_wrap(self, seed):
        random.seed(seed)
        p = random.random()
        d = self._sample_degree(p)
        return random.sample(range(int(self.number)), d)

    @staticmethod
    def gen_rsd_cdf(number, value, delta):
        pivot = int(math.floor(number / value))
        value_1 = [value / number * 1 / d for d in range(1, pivot)]
        value_2 = [value / number * math.log(value / delta)]
        value_3 = [0 for _ in range(pivot, number)]
        tau = value_1 + value_2 + value_3
        rho = [1.0 / number] + [1.0 / (d * (d - 1)) for d in range(2, number + 1)]
        degree = sum(rho) + sum(tau)
        mu = [(rho[d] + tau[d]) / degree for d in range(number)]
        cdf = numpy.cumsum(mu)
        return cdf, degree

    def _sample_degree(self, p):
        index = None
        for index, value in enumerate(self.cdf):
            if value > p:
                return index + 1
        return index + 1


class Droplet(object):

    def __init__(self):
        self.seed = None
        self.payload = None
        self.chuck_indices = None

    def get_dna(self, seed, prng, bit_segments, header_size):
        self.seed = seed
        self.payload = None
        self.chuck_indices = prng.get_src_blocks_wrap(seed)

        for chuck_index in self.chuck_indices:
            if self.payload is None:
                self.payload = bit_segments[chuck_index]
            else:
                self.payload = list(map(self.xor, self.payload, bit_segments[chuck_index]))

        bit_list = self._get_seed_list(header_size) + self.payload

        dna_sequence = []
        for index in range(0, len(bit_list), 2):
            dna_sequence.append(index_base.get(bit_list[index] * 2 + bit_list[index + 1]))

        return dna_sequence

    def init_binaries(self, prng, dna_sequence, header_size):
        # recover the bit segment
        bit_segment = []
        for base in dna_sequence:
            index = base_index.get(base)
            bit_segment.append(int(index / 2))
            bit_segment.append(index % 2)

        self.seed = self.get_seed(bit_segment[:header_size * 8])
        self.payload = bit_segment[header_size * 8:]
        self.chuck_indices = prng.get_src_blocks_wrap(self.seed)

    def update_binaries(self, chunk_index, bit_segments):
        self.payload = list(map(self.xor, self.payload, bit_segments[chunk_index]))
        # subtract (ie. xor) the value of the solved segment from the droplet.
        self.chuck_indices.remove(chunk_index)

    def _get_seed_list(self, header_size):
        seed_list = [0 for _ in range(header_size * 8)]
        temp_seed = self.seed
        for index in range(len(seed_list)):
            seed_list[index] = temp_seed % 2
            temp_seed = int((temp_seed - seed_list[index]) / 2)
        return seed_list

    @staticmethod
    def get_seed(seed_list):
        seed = 0
        for value in seed_list[::-1]:
            seed = seed * 2 + value

        return seed

    @staticmethod
    def xor(value_1, value_2):
        return value_1 ^ value_2

def check(sequence, max_homopolymer, max_content):
    if max_homopolymer and not homopolymerf(sequence, max_homopolymer):
        return False
    if max_content and not gc_content(sequence, max_content):
        return False

    return True

def gc_content(sequence, max_content):
    return (1 - max_content) <= (float(sequence.count("C") + sequence.count("G")) / float(len(sequence))) <= max_content

def homopolymerf(sequence, max_homopolymer):
    homopolymers = "A{%d,}|C{%d,}|G{%d,}|T{%d,}" % tuple([1 + max_homopolymer] * 4)
    return False if search(homopolymers, sequence) else True

def update_droplets(droplet, bit_segments, done_segments, chunk_to_droplets):
    for chunk_index in (set(droplet.chuck_indices) & done_segments):
        droplet.update_binaries(chunk_index, bit_segments)
        # cut the edge between droplet and input segment.
        chunk_to_droplets[chunk_index].discard(droplet)

    if len(droplet.chuck_indices) == 1:
        # the droplet has only one input segment
        lone_chunk = droplet.chuck_indices.pop()
        # assign the droplet value to the input segment (=entry[0][0])
        bit_segments[lone_chunk] = droplet.payload
        # add the lone_chunk to a data structure of done segments.
        done_segments.add(lone_chunk)
        # cut the edge between the input segment and the droplet
        chunk_to_droplets[lone_chunk].discard(droplet)
        # update other droplets
        for other_droplet in chunk_to_droplets[lone_chunk].copy():
            update_droplets(other_droplet, bit_segments, done_segments, chunk_to_droplets)

def dnafountainEncode(bit_segments,redundancy,homopolymer,bit_size):
    header_size = 4
    # c_dist = 0.1
    # delta = 0.05
    # gc_bias = 0.2
    c_dist = 0.025
    delta = 0.001
    gc_bias = 0.05
    start_time = datetime.now()

    for segment_index, bit_segment in enumerate(bit_segments):
        bit_segments[segment_index] = [int(bit) for bit in bit_segment]
        if len(bit_segment) % 2 != 0:
            bit_segments[segment_index] = [0] + bit_segments[segment_index]

    decode_packets = len(bit_segments)

    dna_sequences = []
    final_count = math.ceil(len(bit_segments) * (1 + redundancy))

    # things related to random number generator, starting an lfsr with a certain state and a polynomial for 32bits.
    lfsr = LFSR().lfsr_s_p()
    # create the solition distribution object
    prng = PRNG(number=decode_packets, delta=delta, c=c_dist)
    used_seeds = dict()
    chuck_recorder = []
    while len(dna_sequences) < final_count:
        seed = next(lfsr)
        if seed in used_seeds:
            continue

        # initialize droplet and trans-code to DNA.
        droplet = Droplet()
        dna_sequence = droplet.get_dna(seed, prng, bit_segments, header_size)

        # check validity.
        if check("".join(dna_sequence),
                 max_homopolymer=homopolymer, max_content=0.5 + gc_bias):
            dna_sequences.append(dna_sequence)
            chuck_recorder.append(droplet.chuck_indices)

    # pre-check the decoding process in the encoding process
    # if need_pre_check:
    #     try:
    #         visited_indices = [0] * decode_packets
    #         for chuck_indices in chuck_recorder:
    #             for chuck_index in chuck_indices:
    #                 visited_indices[chuck_index] += 1
    #         if 0 in visited_indices:
    #             no_visit_indices = []
    #             for index, visited in enumerate(visited_indices):
    #                 if visited == 0:
    #                     no_visit_indices.append(index)
    #             raise ValueError("bit segment " + str(no_visit_indices) + " are not been encoded!")
    #         if need_logs:
    #             print("Pre-check the decoding process.")
    #         dnafountainDecode(dna_sequences)
    #     except ValueError:
    #         raise ValueError("Based on the pre decoding operation, "
    #                          "it is found that the encoded data does not meet the full rank condition."
    #                          "Please increase \"redundancy\" or use compression to "
    #                          "change the original digital data.")
    # else:
    #     if need_logs:
    #         print("We recommend that you test whether it can be decoded before starting the wet experiment.")

    encoding_runtime = (datetime.now() - start_time).total_seconds()

    nucleotide_count = 0
    for dna_sequence in dna_sequences:
        nucleotide_count += len(dna_sequence)

    information_density = bit_size / nucleotide_count
    print()
    print(f"information_density: {information_density}, encoding_runtime: {encoding_runtime}")

    str_dna_sequences = ['']*len(dna_sequences)
    for i in range(len(dna_sequences)):
        str = ''
        for j in range(len(dna_sequences[i])):
            str += dna_sequences[i][j]
        str_dna_sequences[i] = str
    return str_dna_sequences



def dnafountainDecode(dna_sequences,decode_packets,segment_length):
    start_time = datetime.now()
    header_size = 4
    # c_dist = 0.1
    # delta = 0.05
    c_dist = 0.025
    delta = 0.001
    # creating the solition distribution object
    prng = PRNG(number=decode_packets, delta=delta, c=c_dist)

    bit_segments = [None] * decode_packets
    done_segments = set()
    chunk_to_droplets = defaultdict(set)

    for dna_sequence in dna_sequences:
        droplet = Droplet()
        droplet.init_binaries(prng, dna_sequence, header_size)

        for chunk_num in droplet.chuck_indices:
            chunk_to_droplets[chunk_num].add(droplet)

        update_droplets(droplet, bit_segments, done_segments, chunk_to_droplets)

        if len(done_segments) == decode_packets:
            break
    print(f"decode_packets:{decode_packets},done_segments:{len(done_segments)}")
    if None in bit_segments or decode_packets - len(done_segments) > 0:
        raise ValueError("Couldn't decode the whole file, because some bit segments are not recovered!")
    decoding_runtime = (datetime.now() - start_time).total_seconds()

    for segment_index, bit_segment in enumerate(bit_segments):
        if len(bit_segment) != segment_length:
            bit_segments[segment_index] = bit_segment[: segment_length]
    print(f"decoding_runtime: {decoding_runtime}")
    #
    # str_dna_sequences = ['']*len(dna_sequences)
    # for i in range(len(dna_sequences)):
    #     str = ''
    #     for j in range(len(dna_sequences[i])):
    #         str += dna_sequences[i][j]
    #     str_dna_sequences[i] = str

    str_bit_segments = [None]*len(bit_segments)
    for i in range(len(bit_segments)):
        strs = ''
        for j in range(len(bit_segments[i])):
            strs += str(bit_segments[i][j])
        str_bit_segments[i] = strs

    return str_bit_segments
    # return bit_segments












