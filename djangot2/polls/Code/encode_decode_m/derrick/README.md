# Users' Guide
_derrick_ is a software program to designed for encoding and decoding arbitrary binary information to and from DNA sequences for DNA digital data storage. The encoding process involves converting digital files into DNA sequences, including randomization, adding CRC64 and RS codes, and then translating them into DNA sequences; decoding utilizes soft decision decoding strategies to correct errors and successfully decode the DNA sequences back to the original files.

# Installation
```sh
git clone https://github.com/wushigang2/derrick.git
cd derrick
make
```

# General usage
```sh
derrick
```
```sh
commands:
 encode      encode a file.
 decode      decode a file.
```

## encode: convert the file into DNA sequences
```sh
derrick encode
```
```sh
usage: derrick encode [options] <input file>
 -i <string> the pi.txt file for randomization, [NULL]
 -n <int>    n of rs(n,k), n = 2^N - 1, N = 8/10/12/14/16, [255]
 -k <int>    k of rs(n,k), [235]
 -s <int>    the number of rs blocks within a matrix, [62]
 -v          verbose.
 for example: use rs(1023,991) to encode a raw file, each block has 100 rs.
 derrick encode -i pi.txt -n 1023 -k 991 -s 100 <original file>
```

## decode
```sh
derrick decode
```
```sh
usage: derrick decode [options] <input file>
 -i <string> the pi.txt file, [NULL]
 -n <int>    n of rs(n,k), n = 2^N - 1, N = 8/10/12/14/16, [255]
 -k <int>    k of rs(n,k), [235]
 -s <int>    the number of rs blocks within a matrix, [62]
 -M <int>    score for match, [2]
 -X <int>    penalty for mismatch, [-6]
 -O <int>    penalty for gap open, [-3]
 -E <int>    penalty for gap extension, [-2]
 -c <int>    the maximum size of subset to change errors, when Derrick soft decision decoding searches subset iteratively from the candidate error sets [4]
 -d <int>     the maximum size of subset to delete errors, [4]
 -m <string> sequencing mode: illumina/pacbio/nanopore, [pacbio/nanopore]
 -j <int>    jump mode of collision, [0]
             0: jump to last exceed.
             2: jump to first continuous exceed.
 -u <int>    search upper range of candidate, [255]
 -l <int>    search lower range of candidate, [32]
 -r <int>    search raise range of candidate, [0]
 -t <int>    seconds for decoding a block timeout, [6000]
 -a <int>    the number of permitted backtracks, [180000]
 -R <string> ref file, [NULL]
 -b <int>    the serial number of block within a matrix to begin decoding, [0]
 -e <int>    the serial number of block within a matrix to end decoding, [0]
 -f <int>    apply shifting algorithm or not, [1]
 -v          verbose.
 for example: use rs(1023,991) to decode an encoded file, each block has 100 rs.
 derrick decode -i pi.txt -n 1023 -k 991 -s 100 <sequencing file>
```
## Original files for _in silico_ tests 
All  raw data is available at https://github.com/wushigang2/derrick/tree/main/data_files_insilico/

# Contact
Shigang Wu wushigang@caas.cn
