from datasketch import MinHash, MinHashLSH
import logging

from datasketch import MinHash, MinHashLSH
from dt4ddsmaster import dt4dds
from joblib import delayed, Parallel

# from Evaluation_platform.encode_decode_m.config import test_seqlen, sequence_length
# from Evaluation_platform.utils import primers_0,primers_2,pri0pre20,pri1pre20
# from Evaluation_platform.utils import readgzipfile,test_seqlen, sequence_length
from utils import readgzipfile, sequence_length, primers_0, primers_2, pri0pre20, pri1pre20

logger = logging.getLogger(__name__)

dt4dds.default_logging()
dt4dds.config.enable_multiprocessing = True
# dt4dds.config.n_processes = 2
# dt4dds.config.enable_multiprocessing = False
# dt4dds.config.n_processes = 1
dt4dds.config.show_progressbars = True

# primers_0 = ["ACACGACGCTCTTCCGATCT", "AGACGTGTGCTCTTCCGATCT"]
# primers_2 = ["AATGATACGGCGACCACCGAGATCTACACTCTTTCCCTACACGACGCTCTTCCGATCT", "CAAGCAGAAGACGGCATACGAGATCGTGATGTGACTGGAGTTCAGACGTGTGCTCTTCCGATCT"]
# pri0pre20 = "AGATCGGAAGAGCACACGTC"
# pri1pre20 = "AGATCGGAAGAGCGTCGTGT"
def syn_simu(seq_list):
    # define primer sequences for PCR
    # "'R1.fq':'AGACGTGTGCTCTTCCGATCT'的互补序列'AGATCGGAAGAGCACACGTCT'前的序列"
    # "'R2.fq':'ACACGACGCTCTTCCGATCT'的互补序列'AGATCGGAAGAGCGTCGTGT'前的序列"

    #
    # Synthesis
    #

    # read the sequences from the provided example file
    # seq_list = dt4dds.tools.txt_to_seqlist("demos/design_sequences.txt")

    # create a synthesis instance and process the list of design sequences
    array_synthesis = dt4dds.processes.ArraySynthesis()
    array_synthesis.process(seq_list)

    # sample 100k oligos
    pool = array_synthesis.sample_by_counts(100000)
    pool = dt4dds.generators.attach_primers_to_pool(pool, *primers_0)
    pool.volume = 5


    #
    # PCR
    #

    # specify primers and total number of cycles, then process the effect of PCR
    pcr = dt4dds.processes.PCR(primers=primers_2, n_cycles=30)
    pool = pcr.process(pool)
    # seqs = pool._seqdict
    return pool

def syn_simu_myadvanced(seq_list):
    dt4dds.properties.set_property_settings(
        dt4dds.settings.defaults.SequenceProperties(
            efficiency_distribution='normal',
            efficiency_params={'loc': 1.0, 'scale': 0.0051},
        )
    )
    #
    # Synthesis
    #

    # create a synthesis instance and process the list of design sequences
    array_synthesis = dt4dds.processes.ArraySynthesis()
    array_synthesis.process(seq_list)

    # sample 100k oligos
    pool = array_synthesis.sample_by_counts(100000)
    pool = dt4dds.generators.attach_primers_to_pool(pool, *primers_0)
    pool.volume = 5


    #
    # PCR
    #

    # specify primers and total number of cycles, then process the effect of PCR
    pcr = dt4dds.processes.PCR(primers=primers_2, n_cycles=30)
    pool = pcr.process(pool)
    return pool

def syn_simu_advanced(seq_list):
    # def _sequence(self, seqpool):
    #
    #     cluster_seqpool = seqpool
    #
    #     # sample oligos to get expected number of reads
    #     if cluster_seqpool.n_oligos >= self.settings.n_reads:
    #         cluster_seqpool = cluster_seqpool.sample_by_counts(self.settings.n_reads, remove_sampled_oligos=False)
    #     elif cluster_seqpool.n_oligos == 0:
    #         logger.exception("Unable to sequence, no sequence-able oligos found.")
    #     else:
    #         logger.warning(f"Only {cluster_seqpool.n_oligos} oligos available for total of {self.settings.n_reads} reads. Continuing.")
    #
    #     cluster_seqpool.save_as_fasta("./testFiles/testResult/dt4dds_syn.fasta")

    dt4dds.properties.set_property_settings(
        dt4dds.settings.defaults.SequenceProperties(
            efficiency_distribution='normal',
            efficiency_params={'loc': 1.0, 'scale': 0.0051},
        )
    )

    #
    # Electrochemical synthesis with specified coverage bias
    #

    # read the sequences from the provided example file
    # seq_list = dt4dds.tools.txt_to_seqlist("design_sequences.txt")
    n_seqs = len(seq_list)

    # set up the synthesis by using defaults for electrochemical synthesis
    synthesis_settings = dt4dds.settings.defaults.ArraySynthesis_Twist()
    # settings can be customized further when passing to the process instance
    array_synthesis = dt4dds.processes.ArraySynthesis(synthesis_settings(
        oligo_distribution_type='lognormal',
        oligo_distribution_params={'mean': 0, 'sigma': 0.30},
    ))
    array_synthesis = dt4dds.processes.ArraySynthesis()
    array_synthesis.process(seq_list)

    # sample with mean coverage of 200
    pool = array_synthesis.sample_by_counts(100 * n_seqs)
    # pool = array_synthesis.sample_by_counts(100*n_seqs)
    pool = dt4dds.generators.attach_primers_to_pool(pool, *primers_0)
    pool.volume = 1


    #
    # Aging for one half-live
    #
    aging_settings = dt4dds.settings.defaults.Aging()
    aging = dt4dds.processes.Aging(aging_settings(
        fixed_decay_ratio=0.5,
    ))
    pool = aging.process(pool)
    pool.volume = 1

    #
    # PCR with High Fidelity polymerase for 30 cycles at mean efficiency of 95%
    #
    pcr_settings = dt4dds.settings.defaults.PCR_HiFi()
    pcr = dt4dds.processes.PCR(pcr_settings(
        primers=primers_2,
        template_volume=1,
        volume=20,
        efficiency_mean=0.95,
        n_cycles=30,
    ))
    pool = pcr.process(pool)
    # _sequence(pool)
    # seqs = pool._seqdict
    return pool

def seq_simu(pool):
    # illumina_sequencing.Sequencing(pool)
    #
    # Sequencing-by-synthesis
    #

    # specify current directory as output directory to save the sequencing data
    sbs_sequencing = dt4dds.processes.SBSSequencing(output_directory=".")
    sbs_sequencing.process(pool)

def seq_simu_advanced(pool,n_seqs,deep=100):
    #
    # Sequencing-by-synthesis with paired reads and sequencing coverage of 25
    #
    synthesis_settings = dt4dds.settings.defaults.iSeq100()
    sbs_sequencing = dt4dds.processes.SBSSequencing(synthesis_settings(
        output_directory=".",
        # n_reads=int(40 * n_seqs),
        n_reads=int(deep * n_seqs),
        read_length=158,
        # read_length=175,
        read_mode='paired-end',
    ))
    sbs_sequencing.process(pool)
#
# def get_allsequences1():
#     allseqs_r1,_ = readgzipfile('R1.fq.gz')
#     allseqs_r2,_ = readgzipfile('R2.fq.gz')
#     true_allseqs_r1,true_allseqs_r2 = [],[]
#     for i in range(len(allseqs_r1)):
#         index = allseqs_r1[i].find(pri0pre20)
#         if index < 0:
#             true_allseqs_r1.append(allseqs_r1[i][:sequence_length+2])
#         else:
#             true_allseqs_r1.append(allseqs_r1[i][:index])
#         index = allseqs_r2[i].find(pri1pre20)
#         if index < 0:
#             true_allseqs_r2.append(allseqs_r2[i][:sequence_length+2])
#         else:
#             true_allseqs_r2.append(allseqs_r2[i][:index])
#     reversed_true_allseqs_r2 = []
#     dict = {'A':'T','C':'G','T':'A','G':'C'}
#     for i in range(len(true_allseqs_r2)):
#         newseq = ""
#         for j in range(len(true_allseqs_r2[i])-1,-1,-1):
#             newseq += dict[true_allseqs_r2[i][j]]
#         reversed_true_allseqs_r2.append(newseq)
#     return true_allseqs_r1,reversed_true_allseqs_r2

def get_allsequences():
    allseqs_r1,_ = readgzipfile('R1.fq.gz')
    allseqs_r2,_ = readgzipfile('R2.fq.gz')
    true_allseqs_r1,true_allseqs_r2 = [],[]
    for i in range(len(allseqs_r1)):
        index = allseqs_r1[i].find(pri0pre20)
        if index < 0:
            # true_allseqs_r1.append(allseqs_r1[i][:sequence_length+2])
            true_allseqs_r1.append(allseqs_r1[i][:sequence_length])
        else:
            true_allseqs_r1.append(allseqs_r1[i][:index])
        index = allseqs_r2[i].find(pri1pre20)
        if index < 0:
            true_allseqs_r2.append(allseqs_r2[i][:sequence_length])
        else:
            true_allseqs_r2.append(allseqs_r2[i][:index])
    reversed_true_allseqs_r2 = []
    dict = {'A':'T','C':'G','T':'A','G':'C'}
    for i in range(len(true_allseqs_r2)):
        newseq = ""
        for j in range(len(true_allseqs_r2[i])-1,-1,-1):
            newseq += dict[true_allseqs_r2[i][j]]
        reversed_true_allseqs_r2.append(newseq)
    return true_allseqs_r1,reversed_true_allseqs_r2

def mergesimuseqs(simulated_seqsr1,simulated_seqsr2):
    simulated_seqs=[]
    for i in range(len(simulated_seqsr1)):
        frontnum = 20
        # flag = False
        while (frontnum >= 15):
            # if frontnum < 25:print(frontnum)
            frontnum -= 5
            head = simulated_seqsr2[i][:frontnum]
            index1 = sequence_length-158-5
            seq111 = simulated_seqsr1[i][index1:]
            if seq111.find(head) >= 0:
                index1 += seq111.find(head)
            else:
                index1 = -1
            index = index1
            # index = sequence_length-158-1
            # index += simulated_seqsr1[i][index:].find(head)
            # index = simulated_seqsr1[i].find(head)
            if index >= 0:
                simulated_seqs.append(simulated_seqsr1[i][:index] + simulated_seqsr2[i])
                break
        # if not flag:
        #     print(f"{i} not find simulated_seqsr2's fronthead")
        #     simulated_seqs.append(simulated_seqsr1[i][:sequence_length-len(simulated_seqsr2[i])] + simulated_seqsr2[i])
    return simulated_seqs


def get_illsimuseqs():
    allseqs_r1,_ = readgzipfile('R1.fq.gz')
    allseqs_r2,_ = readgzipfile('R2.fq.gz')
    true_allseqs_r1,true_allseqs_r2 = [],[]
    for i in range(len(allseqs_r1)):
        index = allseqs_r1[i].find(pri0pre20)
        if index < 0:
            true_allseqs_r1.append(allseqs_r1[i])
        else:
            true_allseqs_r1.append(allseqs_r1[i][:index])
        index = allseqs_r2[i].find(pri1pre20)
        if index < 0:
            true_allseqs_r2.append(allseqs_r2[i])
        else:
            true_allseqs_r2.append(allseqs_r2[i][:index])
    reversed_true_allseqs_r2 = []
    dict = {'A':'T','C':'G','T':'A','G':'C'}
    for i in range(len(true_allseqs_r2)):
        newseq = ""
        for j in range(len(true_allseqs_r2[i])-1,-1,-1):
            newseq += dict[true_allseqs_r2[i][j]]
        reversed_true_allseqs_r2.append(newseq)

    return true_allseqs_r1,reversed_true_allseqs_r2




def clusterseqs(dna_sequences,dna_sequences_phreds):
    kmer_len = 9
    threshold = 0.30
    num_perm = 128

    def getphred_quality(qualityScore):
        phred_qualitys = []
        # sss = seq[110:120]
        for index, i in enumerate(qualityScore):
            phred_quality = ord(i) - 33  # '@'的ASCII码是64，FASTQ使用的是Phred+33编码
            phred_qualitys.append(phred_quality)
        return phred_qualitys

    def compute_minhash(dna_seq, kmer_len, num_perm):
        minhash = MinHash(num_perm=num_perm)
        for i in range(0, len(dna_seq) - kmer_len + 1):
            kmer = dna_seq[i: i + kmer_len]
            minhash.update(kmer.encode('utf-8'))
        return minhash
    minhashes = Parallel(n_jobs=-2)(delayed(compute_minhash)(dna_sequences[i],kmer_len,num_perm) for i in range(len(dna_sequences)))
    lsh = MinHashLSH(threshold, num_perm=num_perm)
    for i, minhash in enumerate(minhashes):
        lsh.insert(i, minhash)
    clusters = []
    visited = set()
    for i, minhash in enumerate(minhashes):
        if i not in visited:
            cluster = set(lsh.query(minhash))
            clusters.append(cluster)
            visited.update(cluster)
    all_phreds_clusters = {}
    all_seq_clusters = {}
    seql = 0
    for seq_labels in clusters:
        all_seq_clusters[seql] = []
        all_phreds_clusters[seql] = []
        for seq_label in seq_labels:
            all_seq_clusters[seql].append(dna_sequences[seq_label])
            all_phreds_clusters[seql].append(getphred_quality(dna_sequences_phreds[seq_label]))
        seql += 1
    return list(all_seq_clusters.values()),list(all_phreds_clusters.values())


# lists = clusterseqs(get_allsequences())
# print(111)












