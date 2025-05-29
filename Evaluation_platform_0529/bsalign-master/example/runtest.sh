FA=real.txt
# parameter -M 2 -X 2 -O 4 -E 2 -Q 0 -P 0

echo Running Poa.bsalign
../bsalign poa -o $FA.consensus.fasta -L $FA > $FA.consensus.bsalign

