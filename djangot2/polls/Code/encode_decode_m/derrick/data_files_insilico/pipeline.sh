# encode
perl merge.pl -o merge.txt -i playfun.mp4 -i pgzf -i group.jpg -i wtdbg2 -i wtdbg-cns -i wtpoa-cns > count.txt
derrick encode -i pi.txt -n 255 -k 211 -s 62 merge.txt > ref.211.fa
gzip ref.211.fa
# decode
gunzip ref.211.fa.gz
derrick decode -i pi.txt -n 255 -k 211 -s 62 ref.211.fa > merge.txt
perl split.pl -i1 count.txt -i2 merge.txt -o1 playfun.mp4 -o2 pgzf -o3 group.jpg -o4 wtdbg2 -o5 wtdbg-cns -o6 wtpoa-cns
