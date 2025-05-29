import math
import random
import sys
from collections import defaultdict
from datetime import datetime

import numpy
from utils import check, index_base, base_index


class Church():
    # def __init__(self, input_file_path:str, output_dir:str, sequence_length:int, max_homopolymer=6,
    #              rs_num=0, add_redundancy=False, add_primer=False, primer_length=20):
    #     index_redundancy = 0 #for virtual segments in wukong
    #     seq_bit_to_base_ratio = 1
    #     super().__init__(input_file_path, output_dir, 'church', seq_bit_to_base_ratio=seq_bit_to_base_ratio, index_redundancy=index_redundancy,
    #                      sequence_length=sequence_length, max_homopolymer=max_homopolymer, rs_num=rs_num, add_redundancy=add_redundancy,
    #                      add_primer=add_primer, primer_length=primer_length)

    def __init__(self, need_logs=False):
        self.need_logs = need_logs
        self.carbon_options = [["A", "C"], ["G", "T"]]

        if self.need_logs:
            print("create Church et al. successfully")


    def encode(self,bit_segments,bit_size):
        start_time = datetime.now()
        self.bit_size = bit_size
        self.segment_length = len(bit_segments[0])
        dna_sequences = []

        for segment_index, bit_segment in enumerate(bit_segments):
            dna_sequence = []
            for bit in bit_segment:
                options, window = self.carbon_options[bit], dna_sequence[-3:]
                if len(window) == 3 and len(set(window)) == 1:
                    for option in options:
                        if option != window[0]:
                            dna_sequence.append(option)
                            break
                else:
                    dna_sequence.append(random.choice(options))

            dna_sequences.append(dna_sequence)
        encoding_runtime = (datetime.now() - start_time).total_seconds()
        if self.need_logs:
            self.monitor.output(segment_index + 1, len(bit_segments))

        nucleotide_count = 0
        for dna_sequence in dna_sequences:
            nucleotide_count += len(dna_sequence)

        information_density = bit_size / nucleotide_count
        print()
        print(f"information_density: {information_density}, encoding_runtime: {encoding_runtime}")


        return dna_sequences

    def _decode(self, dna_sequences,segment_length):
        start_time = datetime.now()
        self.segment_length = segment_length
        bit_segments = []

        for sequence_index, dna_sequence in enumerate(dna_sequences):
            bit_segment = []
            for nucleotide in dna_sequence:
                for option_index, carbon_option in enumerate(self.carbon_options):
                    if nucleotide in carbon_option:
                        bit_segment.append(option_index)

            bit_segments.append(bit_segment)

            if self.need_logs:
                self.monitor.output(sequence_index + 1, len(dna_sequences))

        for segment_index, bit_segment in enumerate(bit_segments):
            if len(bit_segment) != self.segment_length:
                bit_segments[segment_index] = bit_segment[: self.segment_length]
        decoding_runtime = (datetime.now() - start_time).total_seconds()
        print(f"decoding_runtime: {decoding_runtime}")
        return bit_segments

class DNAFountain11(object):

    def __init__(self, homopolymer=4, gc_bias=0.2, redundancy=0.07, header_size=4,
                 c_dist=0.1, delta=0.05, recursion_depth=10000000, decode_packets=None, need_pre_check=False,
                 need_logs=False):
        self.need_logs = need_logs
        self.homopolymer = homopolymer
        self.gc_bias = gc_bias
        self.redundancy = redundancy
        self.header_size = header_size
        self.c_dist = c_dist
        self.delta = delta
        self.recursion_depth = recursion_depth
        self.need_pre_check = need_pre_check
        self.prng = None
        self.decode_packets = decode_packets
        sys.setrecursionlimit(self.recursion_depth)

        if self.need_logs:
            print("Create DNA Fountain successfully")


    def encode(self,bit_segments,bit_size):
        start_time = datetime.now()
        self.bit_size = bit_size
        self.segment_length = len(bit_segments[0])


        for segment_index, bit_segment in enumerate(bit_segments):
            if len(bit_segment) % 2 != 0:
                bit_segments[segment_index] = [0] + bit_segment

        self.decode_packets = len(bit_segments)

        dna_sequences = []
        final_count = math.ceil(len(bit_segments) * (1 + self.redundancy))

        # things related to random number generator, starting an lfsr with a certain state and a polynomial for 32bits.
        lfsr = DNAFountain.LFSR().lfsr_s_p()
        # create the solition distribution object
        self.prng = DNAFountain.PRNG(number=self.decode_packets, delta=self.delta, c=self.c_dist)
        used_seeds = dict()
        chuck_recorder = []
        while len(dna_sequences) < final_count:
            seed = next(lfsr)
            if seed in used_seeds:
                continue

            # initialize droplet and trans-code to DNA.
            droplet = DNAFountain.Droplet()
            dna_sequence = droplet.get_dna(seed, self.prng, bit_segments, self.header_size)

            # check validity.
            if check("".join(dna_sequence),
                            max_homopolymer=self.homopolymer, max_content=0.5 + self.gc_bias):
                dna_sequences.append(dna_sequence)
                chuck_recorder.append(droplet.chuck_indices)

            if self.need_logs:
                self.monitor.output(len(dna_sequences), final_count)

        # pre-check the decoding process in the encoding process
        if self.need_pre_check:
            try:
                visited_indices = [0] * self.decode_packets
                for chuck_indices in chuck_recorder:
                    for chuck_index in chuck_indices:
                        visited_indices[chuck_index] += 1
                if 0 in visited_indices:
                    no_visit_indices = []
                    for index, visited in enumerate(visited_indices):
                        if visited == 0:
                            no_visit_indices.append(index)
                    raise ValueError("bit segment " + str(no_visit_indices) + " are not been encoded!")
                if self.need_logs:
                    print("Pre-check the decoding process.")
                self.decode(dna_sequences)
            except ValueError:
                raise ValueError("Based on the pre decoding operation, "
                                 "it is found that the encoded data does not meet the full rank condition."
                                 "Please increase \"redundancy\" or use compression to "
                                 "change the original digital data.")
        else:
            if self.need_logs:
                print("We recommend that you test whether it can be decoded before starting the wet experiment.")

        encoding_runtime = (datetime.now() - start_time).total_seconds()

        nucleotide_count = 0
        for dna_sequence in dna_sequences:
            nucleotide_count += len(dna_sequence)

        information_density = bit_size / nucleotide_count
        print()
        print(f"information_density: {information_density}, encoding_runtime: {encoding_runtime}")


        return dna_sequences

    def decode(self, dna_sequences, segment_length):
        start_time = datetime.now()
        self.segment_length = segment_length

        # creating the solition distribution object
        self.prng = DNAFountain.PRNG(number=self.decode_packets, delta=self.delta, c=self.c_dist)

        bit_segments = [None] * self.decode_packets
        done_segments = set()
        chunk_to_droplets = defaultdict(set)

        for dna_sequence in dna_sequences:
            droplet = DNAFountain.Droplet()
            droplet.init_binaries(self.prng, dna_sequence, self.header_size)

            for chunk_num in droplet.chuck_indices:
                chunk_to_droplets[chunk_num].add(droplet)

            self.update_droplets(droplet, bit_segments, done_segments, chunk_to_droplets)

            if self.need_logs:
                self.monitor.output(len(done_segments), self.decode_packets)

            if len(done_segments) == self.decode_packets:
                break

        if None in bit_segments or self.decode_packets - len(done_segments) > 0:
            raise ValueError("Couldn't decode the whole file, because some bit segments are not recovered!")
        decoding_runtime = (datetime.now() - start_time).total_seconds()

        for segment_index, bit_segment in enumerate(bit_segments):
            if len(bit_segment) != self.segment_length:
                bit_segments[segment_index] = bit_segment[: self.segment_length]
        print(f"decoding_runtime: {decoding_runtime}")
        return bit_segments

    def update_droplets(self, droplet, bit_segments, done_segments, chunk_to_droplets):
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
                self.update_droplets(other_droplet, bit_segments, done_segments, chunk_to_droplets)

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

class DNAFountain(object):

    def __init__(self, homopolymer=4, gc_bias=0.2, redundancy=0.07, header_size=4,
                 c_dist=0.1, delta=0.05, recursion_depth=10000000, decode_packets=None, need_pre_check=False,
                 need_logs=False):
        self.need_logs = need_logs
        self.homopolymer = homopolymer
        self.gc_bias = gc_bias
        self.redundancy = redundancy
        self.header_size = header_size
        self.c_dist = c_dist
        self.delta = delta
        self.recursion_depth = recursion_depth
        self.need_pre_check = need_pre_check
        self.prng = None
        self.decode_packets = decode_packets
        sys.setrecursionlimit(self.recursion_depth)

        if self.need_logs:
            print("Create DNA Fountain successfully")


    def encode(self,bit_segments,bit_size):
        start_time = datetime.now()
        self.bit_size = bit_size
        self.segment_length = len(bit_segments[0])


        for segment_index, bit_segment in enumerate(bit_segments):
            if len(bit_segment) % 2 != 0:
                # bit_segment = [int(bit) for bit in bit_segment]
                bit_segments[segment_index] = [0] + bit_segment

        self.decode_packets = len(bit_segments)

        dna_sequences = []
        final_count = math.ceil(len(bit_segments) * (1 + self.redundancy))

        # things related to random number generator, starting an lfsr with a certain state and a polynomial for 32bits.
        lfsr = DNAFountain.LFSR().lfsr_s_p()
        # create the solition distribution object
        self.prng = DNAFountain.PRNG(number=self.decode_packets, delta=self.delta, c=self.c_dist)
        used_seeds = dict()
        chuck_recorder = []
        while len(dna_sequences) < final_count:
            seed = next(lfsr)
            if seed in used_seeds:
                continue

            # initialize droplet and trans-code to DNA.
            droplet = DNAFountain.Droplet()
            dna_sequence = droplet.get_dna(seed, self.prng, bit_segments, self.header_size)

            # check validity.
            if check("".join(dna_sequence),
                            max_homopolymer=self.homopolymer, max_content=0.5 + self.gc_bias):
                dna_sequences.append(dna_sequence)
                chuck_recorder.append(droplet.chuck_indices)

            if self.need_logs:
                self.monitor.output(len(dna_sequences), final_count)

        # pre-check the decoding process in the encoding process
        if self.need_pre_check:
            try:
                visited_indices = [0] * self.decode_packets
                for chuck_indices in chuck_recorder:
                    for chuck_index in chuck_indices:
                        visited_indices[chuck_index] += 1
                if 0 in visited_indices:
                    no_visit_indices = []
                    for index, visited in enumerate(visited_indices):
                        if visited == 0:
                            no_visit_indices.append(index)
                    raise ValueError("bit segment " + str(no_visit_indices) + " are not been encoded!")
                if self.need_logs:
                    print("Pre-check the decoding process.")
                self.decode(dna_sequences)
            except ValueError:
                raise ValueError("Based on the pre decoding operation, "
                                 "it is found that the encoded data does not meet the full rank condition."
                                 "Please increase \"redundancy\" or use compression to "
                                 "change the original digital data.")
        else:
            if self.need_logs:
                print("We recommend that you test whether it can be decoded before starting the wet experiment.")

        encoding_runtime = (datetime.now() - start_time).total_seconds()

        nucleotide_count = 0
        for dna_sequence in dna_sequences:
            nucleotide_count += len(dna_sequence)

        information_density = bit_size / nucleotide_count
        print()
        print(f"information_density: {information_density}, encoding_runtime: {encoding_runtime}")


        return dna_sequences

    def decode(self, dna_sequences, segment_length):
        start_time = datetime.now()
        self.segment_length = segment_length

        # creating the solition distribution object
        self.prng = DNAFountain.PRNG(number=self.decode_packets, delta=self.delta, c=self.c_dist)

        bit_segments = [None] * self.decode_packets
        done_segments = set()
        chunk_to_droplets = defaultdict(set)

        for dna_sequence in dna_sequences:
            droplet = DNAFountain.Droplet()
            droplet.init_binaries(self.prng, dna_sequence, self.header_size)

            for chunk_num in droplet.chuck_indices:
                chunk_to_droplets[chunk_num].add(droplet)

            self.update_droplets(droplet, bit_segments, done_segments, chunk_to_droplets)

            if self.need_logs:
                self.monitor.output(len(done_segments), self.decode_packets)

            if len(done_segments) == self.decode_packets:
                break

        if None in bit_segments or self.decode_packets - len(done_segments) > 0:
            raise ValueError("Couldn't decode the whole file, because some bit segments are not recovered!")
        decoding_runtime = (datetime.now() - start_time).total_seconds()

        for segment_index, bit_segment in enumerate(bit_segments):
            if len(bit_segment) != self.segment_length:
                bit_segments[segment_index] = bit_segment[: self.segment_length]
        print(f"decoding_runtime: {decoding_runtime}")
        return bit_segments

    def update_droplets(self, droplet, bit_segments, done_segments, chunk_to_droplets):
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
                self.update_droplets(other_droplet, bit_segments, done_segments, chunk_to_droplets)

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

