import logging
from . import dt4dds
from joblib import delayed, Parallel

from .utils import readgzipfile, sequence_length, primers_0, primers_2, pri0pre20, pri1pre20, SimuInfo

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





def syn_simu_advanced(seq_list, simuInfo:SimuInfo):
    n_seqs = len(seq_list)
    dt4dds.properties.set_property_settings(
        dt4dds.settings.defaults.SequenceProperties(
            efficiency_distribution='normal',
            efficiency_params={'loc': 1.0, 'scale': 0.0051},
        )
    )
    print(f"synthesis...：")
    synthesis_settings = dt4dds.settings.SynthesisSettings()
    array_synthesis = dt4dds.processes.ArraySynthesis(synthesis_settings(
        per_oligo_scale=simuInfo.oligo_scale,
    ))

    array_synthesis.process(seq_list)
    pool = array_synthesis.sample_by_counts(simuInfo.sample_multiple * n_seqs)
    pool = dt4dds.generators.attach_primers_to_pool(pool, *primers_0)
    pool.volume = 1
    # pool.save_as_fasta("./1")

    if 'decay' in simuInfo.channel:
        #
        # Aging for one half-live
        #
        print(f"decay...：")
        aging_settings = dt4dds.settings.defaults.Aging()
        aging = dt4dds.processes.Aging(aging_settings(
            fixed_decay_ratio=simuInfo.decaylossrate,
            arrhenius_t=simuInfo.decay_year,
            arrhenius_T=simuInfo.temperature,
            arrhenius_RH=simuInfo.humidity,
        ))
        pool = aging.process(pool)
        pool.volume = 1


    if 'PCR' in simuInfo.channel:
        #
        # PCR with High Fidelity polymerase for 30 cycles at mean efficiency of 95%
        #
        print(f"PCR...：")
        pcr_settings = dt4dds.settings.defaults.PCR_HiFi()
        pcr = dt4dds.processes.PCR(pcr_settings(
            primers=primers_2,
            template_volume=1,
            volume=20,
            n_cycles=simuInfo.pcrcycle,
            efficiency_mean=simuInfo.pcrpro
        ))
        pool = pcr.process(pool)

    if 'sampling' in simuInfo.channel:
        print(f"sampling...：")
        # sample with mean coverage of 200
        pool = pool.sample_by_counts(simuInfo.sample_ratio * n_seqs)
        # pool = array_synthesis.sample_by_counts(100*n_seqs)
        # pool = dt4dds.generators.attach_primers_to_pool(pool, *primers_0)

        pool.volume = 1
    return pool

def seq_simu_advanced(pool,n_seqs,simuInfo:SimuInfo):
    #
    # Sequencing-by-synthesis with paired reads and sequencing coverage of 25
    #
    synthesis_settings = dt4dds.settings.defaults.iSeq100()
    sbs_sequencing = dt4dds.processes.SBSSequencing(synthesis_settings(
        output_directory=".",
        # n_reads=int(40 * n_seqs),
        n_reads=int(simuInfo.depth * n_seqs),
        # read_length=min(simuInfo.sequence_length,137),
        read_length=158,
        # read_length=175,
        read_mode=simuInfo.sequencing_method,
    ))
    sbs_sequencing.process(pool)

def get_allsequences_withprimer():
    allseqs_r1,allphreds_r1 = readgzipfile('R1.fq.gz')
    allseqs_r2,allphreds_r2 = readgzipfile('R2.fq.gz')
    true_allseqs_r1,true_allphreds_r1 = [],[]
    true_allseqs_r2,true_allphreds_r2 = [],[]
    # indexs0,indexs1 = [],[]
    primerlen1 = len(pri0pre20)
    primerlen2 = len(pri1pre20)
    for i in range(len(allseqs_r1)):
        index = allseqs_r1[i].find(pri0pre20)
        # indexs0.append(index)
        if index < 0:
            true_allseqs_r1.append(allseqs_r1[i][:sequence_length+primerlen1])
            true_allphreds_r1.append(allphreds_r1[i][:sequence_length+primerlen1])
        else:
            true_allseqs_r1.append(allseqs_r1[i][:index+primerlen1])
            true_allphreds_r1.append(allphreds_r1[i][:index+primerlen1])
        index = allseqs_r2[i].find(pri1pre20)
        # indexs1.append(index)
        if index < 0:
            true_allseqs_r2.append(allseqs_r2[i][:sequence_length+primerlen2])
            true_allphreds_r2.append(allphreds_r2[i][:sequence_length+primerlen2])
        else:
            true_allseqs_r2.append(allseqs_r2[i][:index+primerlen2])
            true_allphreds_r2.append(allphreds_r2[i][:index+primerlen2])
    reversed_true_allseqs_r2 = []
    reversed_true_allphreds_r2 = []
    dict = {'A':'T','C':'G','T':'A','G':'C'}
    for i in range(len(true_allseqs_r2)):
        newseq = ""
        for j in range(len(true_allseqs_r2[i])-1,-1,-1):
            newseq += dict[true_allseqs_r2[i][j]]
        reversed_true_allseqs_r2.append(newseq)
        reversed_true_allphreds_r2.append(true_allphreds_r2[i][::-1])
    return true_allseqs_r1+reversed_true_allseqs_r2, true_allphreds_r1+reversed_true_allphreds_r2
    # print('done')

def get_allsequences(sequence_length=120):
    allseqs_r1,allphreds_r1 = readgzipfile('R1.fq.gz')
    allseqs_r2,allphreds_r2 = readgzipfile('R2.fq.gz')
    true_allseqs_r1,true_allphreds_r1 = [],[]
    true_allseqs_r2,true_allphreds_r2 = [],[]
    for i in range(len(allseqs_r1)):
        index = allseqs_r1[i].find(pri0pre20)
        if index < 0:
            # true_allseqs_r1.append(allseqs_r1[i][:sequence_length+2])
            true_allseqs_r1.append(allseqs_r1[i][:sequence_length])
            true_allphreds_r1.append(allphreds_r1[i][:sequence_length])
        else:
            true_allseqs_r1.append(allseqs_r1[i][:index])
            true_allphreds_r1.append(allphreds_r1[i][:index])
        index = allseqs_r2[i].find(pri1pre20)
        if index < 0:
            true_allseqs_r2.append(allseqs_r2[i][:sequence_length])
            true_allphreds_r2.append(allphreds_r2[i][:sequence_length])
        else:
            true_allseqs_r2.append(allseqs_r2[i][:index])
            true_allphreds_r2.append(allphreds_r2[i][:index])
    reversed_true_allseqs_r2 = []
    reversed_true_allphreds_r2 = []
    dict = {'A':'T','C':'G','T':'A','G':'C'}
    for i in range(len(true_allseqs_r2)):
        newseq = ""
        for j in range(len(true_allseqs_r2[i])-1,-1,-1):
            newseq += dict[true_allseqs_r2[i][j]]
        reversed_true_allseqs_r2.append(newseq)
        reversed_true_allphreds_r2.append(true_allphreds_r2[i][::-1])
    return true_allseqs_r1,reversed_true_allseqs_r2,true_allphreds_r1,reversed_true_allphreds_r2

def mergesimuseqs(simulated_seqsr1,simulated_seqsr2,sequence_length):
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
            # print(f"seq111:{seq111},head:{head}")
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










