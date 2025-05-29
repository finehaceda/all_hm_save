
from numpy import array,NaN

# coderates = array([NaN, 0.75, 0.6, 0.5, 1./3., 0.25, 1./6.]) # table of coderates 1..6
coderates = array([NaN, 0.75, 0.6, 0.5, 1./3., 0.25, 1./6.]) # table of coderates 1..6

# user-settable parameters for this test
coderatecode = 2 # test this coderate in coderates table above
npackets = 1 # number of packets (of 255 strands each) to generate and test
totstrandlen = 262 # total length of DNA strand
strandIDbytes = 2 # ID bytes each strand for packet and sequence number
strandrunoutbytes = 2 # confirming bytes end of each strand (see paper)
hlimit = 1000000 # maximum size of decode heap, see paper
# leftprimer = "TCGAAGTCAGCGTGTATTGTATG"
# rightprimer = "TAGTGAGTGCGATTAAGCGTGTT" # for direct right appending (no revcomp)
leftprimer = "T"
rightprimer = "T" # for direct right appending (no revcomp)
# leftprimer = ""
# rightprimer = "" # for direct right appending (no revcomp)

# this test generates substitution, deletion, and insertion errors
# sub,del,ins rates to simulate (as multiple of our observed values):
(srate,drate,irate) = 1.5 * array([0.0238, 0.0082, 0.0039])

# set parameters for DNA constrants (normally not changed, except for no constraint)
max_hpoly_run = 4 # max homopolymer length allowed (0 for no constraint)
GC_window = 12 # window for GC count (0 for no constraint)
max_GC = 8 # max GC allowed in window (0 for no constraint)
min_GC = GC_window - max_GC


# not normally user settable because assumed by Reed-Solomon outer code:
strandsperpacket = 255  # for RS(255,32)
strandsperpacketcheck = 32  # for RS(255,32)

# compute some derived parameters and set parameters in NRpyDNAcode module
leftlen = len(leftprimer)
rightlen = len(rightprimer)
strandlen = totstrandlen - leftlen - rightlen
strandsperpacketmessage = strandsperpacket - strandsperpacketcheck