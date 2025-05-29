import Levenshtein

from Evaluation_platform.DeSPmain.Model.Model import Synthesizer, Decayer, PCRer, Sampler, Sequencer, DNA_Channel_Model
from Evaluation_platform.DeSPmain.Model.config import *

arg = DEFAULT_PASSER
arg.seq_depth = 20
arg.sam_ratio = 0.01
arg.seq_TM = TM_NNP
# arg.seq_TM = TM_NGS
# construct a channel by linking modules
Modules = [
    ('synthesizing', Synthesizer(arg)),
    ('decaying', Decayer(arg)),
    ('pcring', PCRer(arg=arg)),
    ('sampling', Sampler(arg=arg)),
    ('sequencing', Sequencer(arg))
]
Model = DNA_Channel_Model(Modules)

with open('lena.dna') as f:
    dnas = f.readlines()
in_dnas = [dna.split('\n')[0] for dna in dnas]
out_dnas = Model(in_dnas)
oris = []
dnas = []
dnasdis = []
for itemi in out_dnas:
    oris.append(itemi['ori'])
    thislist = itemi['re']
    dnaslist = []
    dnasseqsdis = []
    for j in range(len(thislist)):
        dnaslist.append(thislist[j][2])
        dnasseqsdis.append(len(thislist[j][1]))
    dnas.append(dnaslist)
    dnasdis.append(dnasseqsdis)
for i in range(len(oris)):
    diss = 0
    for j in range(len(dnas[i])):
        diss += Levenshtein.distance(dnas[i][j], oris[i])
    print()
# print(dnas)
# examine the output dnas
# from Analysis.Analysis import inspect_distribution, examine_strand
#
# inspect_distribution(out_dnas, show=True)  # oligo number and error number distribution of the entire sequencing results
# examine_strand(out_dnas, index=index)
