import numpy as np
import matplotlib.pyplot as plt

#a few parameters
k = 8                       # number of information bits
N = 16                      # code length

#define computational modules
def full_adder(a,b,c):
    s = (a ^ b) ^ c
    c = (a & b) | (c & (a ^ b))
    return s,c

def add_bool(a,b):
    if len(a) != len(b):
        raise ValueError('arrays with different length')
    k = len(a)
    s = np.zeros(k,dtype=bool)
    c = False
    for i in reversed(range(0,k)):
        s[i], c = full_adder(a[i],b[i],c)
    if c:
        warnings.warn("Addition overflow!")
    return s

def inc_bool(a):
    k = len(a)
    increment = np.hstack((np.zeros(k-1,dtype=bool), np.ones(1,dtype=bool)))
    a = add_bool(a,increment)
    return a

def bitrevorder(x):
    m = np.amax(x)
    n = np.ceil(np.log2(m)).astype(int)
    for i in range(0,len(x)):
        x[i] = int('{:0{n}b}'.format(x[i],n=n)[::-1],2)
    return x


def polar_design_awgn(N, k, design_snr_dB):
    S = 10 ** (design_snr_dB / 10)
    z0 = np.zeros(N)

    z0[0] = np.exp(-S)
    for j in range(1, int(np.log2(N)) + 1):
        u = 2 ** j
        for t in range(0, int(u / 2)):
            T = z0[t]
            z0[t] = 2 * T - T ** 2  # upper channel
            z0[int(u / 2) + t] = T ** 2  # lower channel

    # sort into increasing order
    idx = np.argsort(z0)

    # select k best channels
    idx = np.sort(bitrevorder(idx[0:k]))

    A = np.zeros(N, dtype=bool)
    A[idx] = True
    return A


def polar_transform_iter(u):
    N = len(u)
    n = 1
    x = np.copy(u)
    stages = np.log2(N).astype(int)
    for s in range(0, stages):
        i = 0
        while i < N:
            for j in range(0, n):
                idx = i + j
                x[idx] = x[idx] ^ x[idx + n]  # ?? x[idx+n] = x[idx] ^ x[idx+n]
            i = i + 2 * n  # i: is the size of block of partition
        n = 2 * n  # n: is the step of addition of two nodes
        return x

# Create all possible information words
d = np.zeros((2**k,k),dtype=bool)
for i in range(1,2**k):
    d[i]= inc_bool(d[i-1])
np.savetxt('polar_code_stages/possible_information_word.txt', d, fmt='%.2f', delimiter = '\n')

# Create sets of all possible codewords (codebook)
A = polar_design_awgn(N, k, design_snr_dB=0)  # logical vector indicating the nonfrozen bit locations
np.savetxt('polar_code_stages/frozen bits.txt', A, fmt='%.1i', delimiter = '  ')
print(A)
x = np.zeros((2**k, N),dtype=bool)
u = np.zeros((2**k, N),dtype=bool)
u[:,A] = d
np.savetxt('polar_code_stages/information_word_in_position.txt', u, fmt='%.1i', delimiter = '  ')

for i in range(0,2**k):
    x[i] = polar_transform_iter(u[i])
print(x[8])
np.savetxt('polar_code_stages/encoded_codeword.txt', x, fmt='%.1i', delimiter = '  ')