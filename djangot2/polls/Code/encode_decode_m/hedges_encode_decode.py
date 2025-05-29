import subprocess
import sys
from numpy import *
import numpy as np
# from hedges_master.hedges_config import *
# from hedges_config import *
import NRpyDNAcode as code
import NRpyRS as RS


# (NSALT, MAXSEQ, NSTAK, HLIMIT) = code.getparams() # get settable code parameters
# code.setparams(8*strandIDbytes, MAXSEQ, NSTAK, hlimit) # change NSALT and HLIMIT
# bytesperstrand = int(strandlen*coderates[coderatecode]/4.)
# messbytesperstrand = bytesperstrand - strandIDbytes - strandrunoutbytes # payload bytes per strand
# messbytesperpacket = strandsperpacket * messbytesperstrand # payload bytes per packet of 255 strands
#
# code.setcoderate(coderatecode,leftprimer,rightprimer) # set code rate with left and right primers
# code.setdnaconstraints(GC_window, max_GC, min_GC, max_hpoly_run) # set DNA constraints (see paper)


def readmessplain(path):
    returnarr = []
    with open(path, 'r') as file:
        lines = file.readlines()
    for i in range(len(lines)):
        line = lines[i].strip(' \n').split(' ')
        # returnarr.append(np.array(line,dtype=np.uint8))
        returnarr.append(line)
    return np.array(returnarr,dtype=np.uint8)

def getdnaseqs(dnapack):
    dna_sequences = []
    dict = {0:'A',1:'C',2:'G',3:'T'}
    for i in range(len(dnapack)):
        thisseq = ''
        for j in range(len(dnapack[i])):
            thisseq += dict[dnapack[i][j]]
        dna_sequences.append(thisseq)
    return dna_sequences

def saveseqs(file_path,seqs):
    with open(file_path, 'w') as f:
        for i in range(len(seqs)):
            # f.write(f">seq{i}\n{seqs[i]}\n")
            f.write('>seq%d' % i + '\n')
            f.write(str(seqs[i]) + '\n')

def saveseqs0123(file_path,seqs):
    with open(file_path, 'w') as f:
        for i in range(len(seqs)):
            # f.write(f">seq{i}\n{seqs[i]}\n")
            # f.write('>seq%d' % i + '\n')
            for j in range(len(seqs[i])):
                f.write(str(seqs[i][j])+' ')
            f.write('\n')

def protectmesspacket(packetin) : # fills in the RS check strands
    packet = packetin.copy()
    regin = zeros(strandsperpacket,dtype=uint8)
    for j in range(messbytesperstrand) :
        for i in range(strandsperpacket) :
            regin[i] = packet[i,((j+i)% messbytesperstrand)+strandIDbytes]
        regout = RS.rsencode(regin)
        for i in range(strandsperpacket):
            packet[i,((j+i)% messbytesperstrand)+strandIDbytes] = regout[i]
    return packet

def correctmesspacket(packetin,epacket) :
    # error correction of the outer RS code from a HEDGES decoded packet and erasure mask
    packet = packetin.copy()
    regin = zeros(strandsperpacket,dtype=uint8)
    erase = zeros(strandsperpacket,dtype=uint8)
    tot_detect = 0
    tot_uncorrect = 0
    max_detect = 0
    max_uncorrect = 0
    toterrcodes = 0
    for j in range(messbytesperstrand) :
        for i in range(strandsperpacket) :
            regin[i] = packet[i,((j+i)% messbytesperstrand)+strandIDbytes]
            erase[i] = epacket[i,((j+i)% messbytesperstrand)+strandIDbytes]
        locations = array(argwhere(erase),dtype=int32)
        (decoded, errs_detected, errs_corrected, errcode, ok) = RS.rsdecode(regin,locations)
        tot_detect += errs_detected
        tot_uncorrect += max(0,(errs_detected-errs_corrected))
        max_detect = max(max_detect,errs_detected)
        max_uncorrect = max(max_uncorrect,max(0,(errs_detected-errs_corrected)))
        toterrcodes += (0 if errcode==0 else 1)
        for i in range(strandsperpacket) :
            packet[i,((j+i)% messbytesperstrand)+strandIDbytes] = decoded[i]
    return (packet,tot_detect,tot_uncorrect,max_detect,max_uncorrect,toterrcodes)

def hedgesEncode(orimpacket,outfilepath,totstrandlen,strandsperpacket):
    # HEDGES encode a message packet into strands of DNA
    npackets = int(ceil(len(orimpacket) / strandsperpacket))
    mpacket = np.zeros([len(orimpacket), bytesperstrand], dtype=np.uint8)
    for ipacket in range(npackets):
        mpacket[ipacket*strandsperpacket:(ipacket+1)*strandsperpacket] = protectmesspacket(orimpacket[ipacket*strandsperpacket:(ipacket+1)*strandsperpacket])


    filler = array([0,2,1,3,0,3,2,1,2,0,3,1,3,1,2,0,2,3,1,0,3,2,1,0,1,3],dtype=np.uint8)
    dpacket = np.zeros([len(mpacket),totstrandlen],dtype=np.uint8)
    for i in range(len(mpacket)) :
        dna = code.encode(mpacket[i,:])
        if len(dna) < totstrandlen : # need filler after message and before right primer
            # dnaleft = dna[:-1]
            # dnaright = dna[-1:]
            dnaleft = dna
            dnaright = np.zeros(0)
            dna = np.concatenate((dnaleft,filler[:totstrandlen-len(dna)],dnaright))
            #n.b. this can violate the output constraints (very slightly at end of strand)
        dpacket[i,:len(dna)] = dna
    all_dna_sequences = getdnaseqs(dpacket)
    saveseqs(outfilepath, all_dna_sequences)
    saveseqs0123(outfilepath+'.0123.txt', dpacket)

    # print(all_dna_sequences[0])
    # print(all_dna_sequences[1])
    return 'hedges_encodefile.fasta'


def writemessplain(path, datas):
    with open(path, 'w') as file:
        for data in datas:
            for dd in data:
                for d in dd:
                    file.write(str(d) + ' ')
                file.write('\n')

# def extractplaintext(cpacket) :
#     # extract plaintext from a corrected packet
#     plaintext = zeros(strandsperpacketmessage*messbytesperstrand,dtype=uint8)
#     for i in range(strandsperpacketmessage) :
#         plaintext[i*messbytesperstrand:(i+1)*messbytesperstrand] = (
#             cpacket[i,strandIDbytes:strandIDbytes+messbytesperstrand])
#     return plaintext

def hedgesDecode11(dnapacket,outfilepath,strandsperpacket) :
    # HEDGES decode strands of DNA (assumed ordered by packet and ID number) to a packet
    baddecodes = 0
    erasures = 0
    npackets = int(ceil(len(dnapacket) / strandsperpacket))
    allmesscheck=[]
    # "for each packet, these statistics are shown in two groups:"
    # "1.1 HEDGES decode failures, 1.2 HEDGES bytes thus declared as erasures"
    # "1.3 R-S total errors detected in packet, 1.4 max errors detected in a single decode"
    # "2.1 R-S reported as initially-uncorrected-but-recoverable total, 2.2 same, but max in single decode"
    # "2.3 R-S total error codes; if zero, then R-S corrected all errors"
    # "2.4 Actual number of byte errors when compared to known plaintext input"
    for ipacket in range(npackets) :
        mythispacket = dnapacket[ipacket*strandsperpacket:(ipacket+1)*strandsperpacket]
        mpacket = np.zeros([strandsperpacket, bytesperstrand], dtype=np.uint8)
        epacket = np.ones([strandsperpacket, bytesperstrand], dtype=np.uint8)
        for i in range(strandsperpacket):
            (errcode, mess, _, _, _, _) = code.decode(mythispacket[i,:],8*bytesperstrand)
            if errcode > 0 :
                baddecodes += 1
                erasures += max(0,messbytesperstrand-len(mess))
            lenmin = min(len(mess),bytesperstrand)
            mpacket[i,:lenmin] = mess[:lenmin]
            epacket[i,:lenmin] = 0
        (cpacket, tot_detect, tot_uncorrect, max_detect, max_uncorrect, toterrcodes) = correctmesspacket(mpacket, epacket)
        print ("%3d: (%3d %3d %3d %3d) (%3d %3d %3d)" % (ipacket, baddecodes, erasures,
                                                             tot_detect, max_detect, tot_uncorrect, max_uncorrect,toterrcodes))
        # messcheck = extractplaintext(cpacket)
        # badbytes = count_nonzero(messplain-messcheck)
        allmesscheck.append(cpacket)
    writemessplain(outfilepath, allmesscheck)
    # print(allmesscheck)
    # print (mpacket,epacket,baddecodes,erasures)
    # return (mpacket,epacket,baddecodes,erasures)


def hedgesDecode(dnapacket,outfilepath,strandsperpacket,npackets) :
    # HEDGES decode strands of DNA (assumed ordered by packet and ID number) to a packet
    # baddecodes = 0
    # erasures = 0
    # npackets = int(ceil(len(dnapacket) / strandsperpacket))
    allmesscheck=[]
    # "for each packet, these statistics are shown in two groups:"
    # "1.1 HEDGES decode failures, 1.2 HEDGES bytes thus declared as erasures"
    # "1.3 R-S total errors detected in packet, 1.4 max errors detected in a single decode"
    # "2.1 R-S reported as initially-uncorrected-but-recoverable total, 2.2 same, but max in single decode"
    # "2.3 R-S total error codes; if zero, then R-S corrected all errors"
    # "2.4 Actual number of byte errors when compared to known plaintext input"
    visited = np.full(strandsperpacket*npackets, False, dtype=bool)
    mpacket = np.zeros([strandsperpacket*npackets, bytesperstrand], dtype=np.uint8)
    epacket = np.ones([strandsperpacket*npackets, bytesperstrand], dtype=np.uint8)
    baddecodes = np.zeros([npackets], dtype=np.uint16)
    erasures = np.zeros([npackets], dtype=np.uint16)
    # allerasures = [[]]*4
    # maxerasures = 0
    for i in range(len(dnapacket)):
        if i%strandsperpacket == 0:print("have done %d strands" % (i+1))
        (errcode, mess, _, _, _, _) = code.decode(dnapacket[i, :], 8 * bytesperstrand)
        if errcode > 0:
            err = max(0, messbytesperstrand - len(mess))
            # maxerasures = max(maxerasures, err)
            if err >= 20 or mess[0]>=npackets or mess[1]>=strandsperpacket or visited[mess[0]*strandsperpacket+mess[1]]:
                continue
            baddecodes[mess[0]] += 1
            erasures[mess[0]] += err
            # print ("baddecodes:%d , erasures:%d" % (baddecodes,erasures))
        packetindex = mess[0]*strandsperpacket+mess[1]
        if not visited[packetindex]:
            lenmin = min(len(mess), bytesperstrand)
            mpacket[packetindex, :lenmin] = mess[:lenmin]
            epacket[packetindex, :lenmin] = 0
            visited[packetindex] = True
        # allerasures[mess[0]].append(err)

    for ipacket in range(npackets):
        smpacket = mpacket[ipacket*strandsperpacket:(ipacket+1)*strandsperpacket]
        sepacket = epacket[ipacket*strandsperpacket:(ipacket+1)*strandsperpacket]
        (cpacket, tot_detect, tot_uncorrect, max_detect, max_uncorrect, toterrcodes) = correctmesspacket(smpacket, sepacket)
        print ("%3d: (%3d %3d %3d %3d) (%3d %3d %3d)" % (ipacket, baddecodes[ipacket], erasures[ipacket],
                                                             tot_detect, max_detect, tot_uncorrect, max_uncorrect,toterrcodes))
        # print "\n"
        # print allerasures[ipacket]
        allmesscheck.append(cpacket)

    # for ipacket in range(npackets) :
    #     mythispacket = dnapacket[ipacket*strandsperpacket:(ipacket+1)*strandsperpacket]
    #     mpacket = np.zeros([strandsperpacket, bytesperstrand], dtype=np.uint8)
    #     epacket = np.ones([strandsperpacket, bytesperstrand], dtype=np.uint8)
    #     for i in range(strandsperpacket):
    #         (errcode, mess, _, _, _, _) = code.decode(mythispacket[i,:],8*bytesperstrand)
    #         if errcode > 0 :
    #             baddecodes += 1
    #             erasures += max(0,messbytesperstrand-len(mess))
    #         lenmin = min(len(mess),bytesperstrand)
    #         mpacket[i,:lenmin] = mess[:lenmin]
    #         epacket[i,:lenmin] = 0
    #     (cpacket, tot_detect, tot_uncorrect, max_detect, max_uncorrect, toterrcodes) = correctmesspacket(mpacket, epacket)
    #     print ("%3d: (%3d %3d %3d %3d) (%3d %3d %3d)" % (ipacket, baddecodes, erasures,
    #                                                          tot_detect, max_detect, tot_uncorrect, max_uncorrect,toterrcodes))
    #     # messcheck = extractplaintext(cpacket)
    #     # badbytes = count_nonzero(messplain-messcheck)
    #     allmesscheck.append(cpacket)
    writemessplain(outfilepath, allmesscheck)
    # print(allmesscheck)
    # print (mpacket,epacket,baddecodes,erasures)
    # return (mpacket,epacket,baddecodes,erasures)



if __name__ == "__main__":
    way,segments_path,outfilepath,strandlen,coderatecode,max_hpoly_run ,npackets= sys.argv[1:]
    # way, segments_path,outfilepath,strandlen,coderatecode,max_hpoly_run  = 'encode','bit_segments.txt','bit_segments_decode.fasta',260,2,3
    # way, segments_path,outfilepath,strandlen,coderatecode,max_hpoly_run  = 'decode','22_hedges.fasta','22_hedges_decode.txt',260,2,3
    # way, segments_path,outfilepath,strandlen,coderatecode,max_hpoly_run  = 'decode','hedges_datas.txt','hedges_decode_ints.txt'
    # way, segments_path,outfilepath,strandlen,coderatecode,max_hpoly_run,npackets  = 'decode','bit_segments_decode.fasta','22_hedges_decode.txt',260,2,3,1020
    # way, segments_path,outfilepath,strandlen,coderatecode,max_hpoly_run,npackets  = 'decode','22_hedges.fasta.0123.txt','22_hedges_decode.txt',260,2,3,4
    # print(sys.argv[1:])
    strandIDbytes = 2
    strandsperpacket = 255
    # strandsperpacket = 512
    hlimit = 1000000
    strandrunoutbytes = 2
    coderates = array([NaN, 0.75, 0.6, 0.5, 1. / 3., 0.25, 1. / 6.])
    (NSALT, MAXSEQ, NSTAK, HLIMIT) = code.getparams()
    code.setparams(8 * strandIDbytes, MAXSEQ, NSTAK, hlimit)
    # leftprimer = "T"
    # rightprimer = "T"
    leftprimer = ""
    rightprimer = ""
    GC_window = 12
    max_GC = 8
    min_GC = GC_window - max_GC
    # print(strandlen)
    # print(coderatecode)
    npackets = int(npackets)
    bytesperstrand = int(int(strandlen) * coderates[int(coderatecode)] / 4.)
    messbytesperstrand = bytesperstrand - strandIDbytes - strandrunoutbytes
    # messbytesperpacket = strandsperpacket * messbytesperstrand

    code.setcoderate(int(coderatecode), leftprimer, rightprimer)
    code.setdnaconstraints(GC_window, max_GC, min_GC, int(max_hpoly_run))
    # way, bit_segments_path = '','bit_segments.txt'
    if way == 'encode':
        bit_segments = readmessplain(segments_path)
        # print(bit_segments[0])
        print(len(bit_segments))
        hedgesEncode(bit_segments,outfilepath,int(strandlen),strandsperpacket)
    elif way == 'decode':

        bit_segments = readmessplain(segments_path)
        # print(bit_segments[0])
        print(len(bit_segments))
        # print "2.4 Actual number of byte errors when compared to known plaintext input"
        hedgesDecode(bit_segments,outfilepath,strandsperpacket,npackets)
