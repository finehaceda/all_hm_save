#include "derrick.h"

int usage_encode()
{
	fprintf(stdout,
        "usage: derrick encode [options] <input file>\n"
	" -i <string> pi file, [NULL]\n"
        " -n <int>    n of rs(n,k), n = 2^N - 1, N = 8/10/12/14/16, [255]\n"
        " -k <int>    k of rs(n,k), [235]\n"
	" -s <int>    number of symbol(or call rs) per segment(or call matrix), [62]\n"
        " -v          verbose\n"
	"\n"
        "for example: use rs(1023,991) to encode a file\n"
        "derrick encode -i pi.txt -n 1023 -k 991 <input file>\n"
        );
        return 1;
}

int main_encode(int argc, char **argv)
{
	FILE *fp;
        struct ENCODEFILE *ef;
	char *pifn;
	int c, n, k, nuss, ver, hrm, nblocks, i, j;
        pifn = NULL;
        n = 255;
        k = 235;
        nuss = 62;
        ver = 0;
        while((c = getopt(argc, argv, "hi:n:k:s:v")) != -1)
        {
                switch(c)
                {
			case 'i': 
				pifn = optarg; 
				break;
                        case 'n':
                                n = atoi(optarg);
                                break;
                        case 'k':
                                k = atoi(optarg);
				break;
			case 's':
                                nuss = atoi(optarg);
                                break;
                        case 'v':
                                ver++;
                                break;
                        default:
                                return usage_encode();
                }
        }
	hrm = log(n + 1) / log(4);
        if(optind < argc)
        {
                fp = fopen(argv[argc - 1], "r");
        }
        else
        {
                return usage_encode();
        }
	clock_t start, finish;
        double time;
        fprintf(stderr, "begin, encoding...\n");
        start = clock();
	i = 0;
	while((c = fgetc(fp)) != -1)
        {
                i = i + 8;
        }
	fclose(fp);
	i = i + 32;
	j = k * nuss * hrm * 2 - 64;
	if(i % j != 0)
	{
		i = i + (j - (i % j));
	}
	nblocks = i / j;
	ef = init_ef(n, k, nuss, ver, hrm, nblocks);
	fp = fopen(argv[argc - 1], "r");
	i = 0;
        while((c = fgetc(fp)) != -1)
        {
		intseq_to_u1iseq(8, &c, 1, ef->bitseq + i, 1);
		if(ef->ver >= 1)
		{
			fprintf(stderr, "%d ", c);
			for(j = 0; j < 8; j++)
			{
				fprintf(stderr, "%u", ef->bitseq[i + j]);
			}
			fprintf(stderr, "\n");
		}
		i = i + 8;
        }
	fclose(fp);
	j = 0;
	for(i = i; i < ef->nblocks * ef->bitlen - 32; i++)
	{
		ef->bitseq[i] = 0;
		if(ef->ver >= 1)
		{
			fprintf(stderr, "%u\n", ef->bitseq[i]);
		}
		j++;
	}
	intseq_to_u1iseq(32, &j, 1, ef->bitseq + i, 1);
	if(ef->ver >= 1)
	{
		fprintf(stderr, "%d ", j);
		for(j = 0; j < 32; j++)
		{
			fprintf(stderr, "%u", ef->bitseq[i + j]);
		}
		fprintf(stderr, "\n");
	}
	if(ef->ver >= 1)
	{
		for(i = 0; i < ef->nblocks * ef->bitlen; i++)
		{
			fprintf(stderr, "%u", ef->bitseq[i]);
		}
		fprintf(stderr, "\n");
	}
	yihuo_pi(ef->bitseq, ef->nblocks * ef->bitlen, pifn);
	if(ef->ver >= 1)
        {
                for(i = 0; i < ef->nblocks * ef->bitlen; i++)
                {
                        fprintf(stderr, "%u", ef->bitseq[i]);
                }
                fprintf(stderr, "\n");
        }
	add_crc64(ef);
	if(ef->ver >= 1)
	{
		for(i = 0; i < ef->nblocks; i++)
        	{
        	        for(j = 0; j < ef->bitlen; j++)
        	        {
				fprintf(stderr, "%d %d\n", ef->bitseq[i * ef->bitlen + j], ef->qseq[i * ef->qlen + j]);
        	        }
			for(j = j; j < ef->qlen; j++)
                        {
                                fprintf(stderr, "- %d\n", ef->qseq[i * ef->qlen + j]);
                        }
        	}
	}
	qseq_to_rseq(ef);
	if(ef->ver >= 1)
	{
		int k1, k2;
		for(i = 0; i < ef->nblocks; i++)
		{
			for(j = 0; j < ef->qlen; j++)
			{
				fprintf(stderr, "%d", ef->qseq[i * ef->qlen + j]);
			}
			fprintf(stderr, "\n");
			for(k2 = 0; k2 < ef->rlen; k2++)
			{
				for(k1 = 0; k1 < ef->k; k1++)
				{
					fprintf(stderr, "%d", ef->rseq[(i * ef->k + k1) * ef->rlen + k2]);
				}
				fprintf(stderr, "\n");
			}
		}
	}
	for(i = 0; i < ef->nblocks; i++)
	{
		for(j = 0; j < ef->k; j++)
		{
			u1iseq_to_u1iseq(1, ef->rseq + (i * ef->k + j) * ef->rlen, 2, ef->tseq + (i * ef->n + j) * ef->tlen, ef->tlen);
			if(ef->ver >= 1)
			{
				int l;
				for(l = 0; l < ef->rlen; l++)
				{
					fprintf(stderr, "%d", ef->rseq[(i * ef->k + j) * ef->rlen + l]);
				}
				fprintf(stderr, "\n");
				for(l = 0; l < ef->tlen; l++)
                                {
                                        fprintf(stderr, "%d", ef->tseq[(i * ef->n + j) * ef->tlen + l]);
                                }
                                fprintf(stderr, "\n");
			}
		}
	}
	add_rs(ef);
	if(ef->ver >= 1)
	{
		for(i = 0; i < ef->nblocks * ef->n; i++)
		{
			for(j = 0; j < ef->tlen; j++)
			{
				fprintf(stderr, "%d", ef->tseq[i * ef->tlen + j]);
			}
			fprintf(stderr, "\n");
		}
	}
	for(i = 0; i < ef->nblocks * ef->n; i++)
	{
		j = i + 1;
		fprintf(stdout, ">contig%d\n", j);
		for(j = 0; j < ef->tlen; j++)
		{
			switch(ef->tseq[i * ef->tlen + j])
                	{
                	        case 1:
                	                fprintf(stdout, "C");
                	                break;
                	        case 2:
                	                fprintf(stdout, "G");
                	                break;
                	        case 3:
                	                fprintf(stdout, "T");
                	                break;
                        	default:
                                	fprintf(stdout, "A");
					break;
                	}
		}
		fprintf(stdout, "\n");
	}
	finish = clock();
        time = (finish - start) / CLOCKS_PER_SEC;
        fprintf(stderr, "end, %d blocks %.3f sec.\n", ef->nblocks, time);
	free_ef(ef);
        return 0;
}

int usage_decode()
{
        fprintf(stdout,
        "usage: derrick decode [options] <input file>\n"
	" -i <string> pi file, [NULL]\n"
        " -n <int>    n of rs(n,k), n = 2^N - 1, N = 8/10/12/14/16, [255]\n"
        " -k <int>    k of rs(n,k), [235]\n"
	" -s <int>    number of symbol(or call rs) per setment(or call matrix), [62]\n"
	" -M <int>    score for match, [2]\n"
	" -X <int>    penalty for mismatch, [-6]\n"
	" -O <int>    penalty for gap open, [-3]\n"
	" -E <int>    penalty for gap extension, [-2]\n"
	" -c <int>    max change number of rs soft decision, [4]\n"
        " -d <int>    max delete number of rs soft decision, [4]\n"
	" -m <string> sequencing mode: illumina/pacbio/nanopore, [pacbio/nanopore]\n"
	" -j <int>    jump mode of collision, [0]\n"
        "             0: jump to last exceed.\n"
	"             2: jump to first continuous exceed.\n"
	" -u <int>    search upper range of candidate, [255]\n"
	" -l <int>    search lower range of candidate, [32]\n"
	" -r <int>    search raise range of candidate, [0]\n"
	" -t <int>    how many seconds a block timeout, [6000]\n"
	" -a <int>    the number of allowed backtracks, [180000]\n"
	" -R <string> ref file, [NULL]\n"
        " -b <int>    beg block, [0]\n"
        " -e <int>    end block, [0]\n"
	" -f <int>    shift or not, [1]\n"
        " -v          verbose\n"
	"\n"
        "for example: use rs(1023,991) to decode an encoded file\n"
        "derrick decode -i pi.txt -n 1023 -k 991 <input file>\n"
        );
        return 1;
}

int main_decode(int argc, char **argv)
{
	if(_DEBUG_LOG_)
	{
	}
	FileReader *fr;
        BioSequence *seq;
	struct DECODEFILE *df;
	struct ARTICLE *article;
	char *pifn, *str, **reffn;
	int c, n, k, nuss, mat, mis, gapo, gape, maxchange, maxdelete, mode, jump, upper, lower, raise, timeout, allowed, begblock, endblock, shift_or_not, ver, hrm, nblocks, i, j;
	pifn = NULL;
	n = 255;
	k = 235;
	nuss = 62;
	mat = 2;
	mis = -6;
	gapo = -3;
	gape = -2;
	maxchange = 4;
	maxdelete = 4;
	mode = 3;
	jump = 0;
	upper = 32;
	lower = 32;
	raise = 0;
	timeout = 6000;
	allowed = 180000;
	reffn = calloc(1, sizeof(char *));
	reffn[0] = NULL;
	begblock = 0;
	endblock = 0;
	shift_or_not = 1;
	ver = 0;
	while((c = getopt(argc, argv, "hi:n:k:s:M:X:O:E:c:d:m:j:u:l:r:t:a:R:b:e:f:v")) != -1)
	{
                switch(c)
		{
			case 'i':
                                pifn = optarg;
                                break;
                        case 'n': 
				n = atoi(optarg);
				break;
                        case 'k': 
				k = atoi(optarg); 
				break;
			case 's':
                                nuss = atoi(optarg);
                                break;
			case 'M':
                                mat = atoi(optarg);
                                break;
			case 'X':
                                mis = atoi(optarg);
                                break;
			case 'O':
                                gapo = atoi(optarg);
                                break;
			case 'E':
                                gape = atoi(optarg);
                                break;
			case 'c':
                                maxchange = atoi(optarg);
                                break;
			case 'd':
                                maxdelete = atoi(optarg);
                                break;
			case 'm':
                                str = optarg;
                                if(strcasecmp(str, "illumina") == 0) mode = 2;
                                break;
			case 'j':
                                jump = atoi(optarg);
                                break;
			case 'u':
                                upper = atoi(optarg);
                                break;
			case 'l':
                                lower = atoi(optarg);
                                break;
			case 'r':
                                raise = atoi(optarg);
                                break;
			case 't':
                                timeout = atoi(optarg);
                                break;
			case 'a':
				allowed = atoi(optarg);
				break;
			case 'R':
                                reffn[0] = optarg;
                                break;
			case 'b':
                                begblock = atoi(optarg);
                                break;
			case 'e':
                                endblock = atoi(optarg);
                                break;
			case 'f':
                                shift_or_not = atoi(optarg);
                                break;
                        case 'v': 
				ver++; 
				break;
                        default: 
				return usage_decode();
		}
	}
	hrm = log(n + 1) / log(4);
	if(optind < argc)
	{
        	fr = open_all_filereader(1, argv + argc - 1, 0);
	}
	else
        {
                return usage_decode();
        }
	seq = init_biosequence();
	i = 0;
	while(readseq_filereader(fr, seq))
        {
		i++;
	}
	nblocks = i / n;
	free_biosequence(seq);
        close_filereader(fr);
	df = init_df(n, k, nuss, mat, mis, gapo, gape, maxchange, maxdelete, mode, jump, upper, lower, raise, timeout, allowed, begblock, endblock, ver, hrm, nblocks);
	article = init_article(shift_or_not);
	fr = open_all_filereader(1, argv + argc - 1, 0);
        seq = init_biosequence();
	i = 0;
	while(readseq_filereader(fr, seq))
	{
		for(j = 0; j < seq->seq->size && j < df->qlen; j++)
		{
			if(seq->seq->string[j] == 67 || seq->seq->string[j] == 99)
                        {
                        	df->qseq[i * df->qlen + j] = 1;
                        }
                        else if(seq->seq->string[j] == 71 || seq->seq->string[j] == 103)
                        {
                        	df->qseq[i * df->qlen + j] = 2;
                        }
                        else if(seq->seq->string[j] == 84 || seq->seq->string[j] == 116)
                        {
                        	df->qseq[i * df->qlen + j] = 3;
                        }
                        else
                        {
                        	df->qseq[i * df->qlen + j] = 0;
                        }
			df->qlts[i * df->qltl + j] = seq->qlt->string[j];
		}
		i++;
	}
        free_biosequence(seq);
        close_filereader(fr);
	if(reffn[0])
	{
		df->hasref = 1;
		fr = open_all_filereader(1, reffn, 0);
		seq = init_biosequence();
		i = 0;
		while(readseq_filereader(fr, seq))
		{
			for(j = 0; j < seq->seq->size && j < df->Rlen; j++)
			{
				if(seq->seq->string[j] == 67 || seq->seq->string[j] == 99)
				{
					df->Rseq[i * df->Rlen + j] = 1;
				}
				else if(seq->seq->string[j] == 71 || seq->seq->string[j] == 103)
				{
					df->Rseq[i * df->Rlen + j] = 2;
				}
				else if(seq->seq->string[j] == 84 || seq->seq->string[j] == 116)
				{
					df->Rseq[i * df->Rlen + j] = 3;
				}
				else
				{
					df->Rseq[i * df->Rlen + j] = 0;
				}
				//df->qlts[i * df->qltl + j] = seq->qlt->string[j];
			}
			i++;
		}
		free_biosequence(seq);
		close_filereader(fr);
		free(reffn);
	}
	clock_t start, finish;
	double time;
	for(i = df->begblock - 1; i < df->endblock; i++)
	{
		article->crc64_times = 0;
		article->shift_times = 0;
		fprintf(stderr, "begin, %d/%d block decoding...\n", i + 1, df->nblocks);
		start = clock();
		time = 0;
		df->jump = jump;
		if(decode_block(df, article, i) == 0)
		{
			finish = clock();
			time = (finish - start) / CLOCKS_PER_SEC;
			fprintf(stderr, "end, success %.3f sec, backtrack times %d/%d, crc64 times %d, shift times %d.\n", time, df->diff, df->number, article->crc64_times, article->shift_times);
		}
		else
		{
			df->jump = df->jump == 0 ? 2 : 0;
			if(decode_block(df, article, i) == 0)
			{
				finish = clock();
				time = (finish - start) / CLOCKS_PER_SEC;
				fprintf(stderr, "end, success %.3f sec, backtrack times %d/%d, crc64 times %d, shift times %d.\n", time, df->diff, df->number, article->crc64_times, article->shift_times);
			}
			else
			{
                        	finish = clock();
				time = (finish - start) / CLOCKS_PER_SEC;
                        	fprintf(stderr, "end, failure %.3f sec, backtrack times %d/%d, crc64 times %d, shift times %d.\n", time, df->diff, df->number, article->crc64_times, article->shift_times);
			}
		}
	}
	if(df->begblock == 1 && df->endblock == df->nblocks)
	{
		struct ENCODEFILE *ef;
		ef = init_ef(df->n, df->k, df->nuss, df->ver, df->hrm, df->nblocks);
		for(i = 0; i < ef->nblocks; i++)
		{
			for(j = 0; j < ef->k; j++)
			{
				u1iseq_to_u1iseq(2, df->tseq + (i * df->n + j) * df->tlen, 1, ef->rseq + (i * ef->k + j) * ef->rlen, df->tlen);
			}
		}
		rseq_to_qseq(ef);
		del_crc64(ef);
		yihuo_pi(ef->bitseq, ef->nblocks * ef->bitlen, pifn);
		u1iseq_to_intseq(1, ef->bitseq + ef->nblocks * ef->bitlen - 32, 32, &j, 1);
		if(j >= ef->bitlen)
		{
			j = ef->bitlen - 32;
		}
		for(i = 0; i < ef->nblocks * ef->bitlen - j - 32; i = i + 8)
		{
			u1iseq_to_intseq(1, ef->bitseq + i, 8, &c, 1);
			fprintf(stdout, "%c", c);
		}
		free_ef(ef);
	}
	else
	{
		for(i = (df->begblock - 1) * df->n; i < df->endblock * df->n; i++)
		{
			fprintf(stdout, ">decode_seq\n");
			for(j = 0; j < df->tlen; j++)
			{
				if(df->tseq[i * df->tlen + j] == 1)
				{
					fprintf(stdout, "C");
				}
				else if(df->tseq[i * df->tlen + j] == 2)
				{
                                        fprintf(stdout, "G");
				}
				else if(df->tseq[i * df->tlen + j] == 3)
                                {
                                        fprintf(stdout, "T");
                                }
				else
				{
                                        fprintf(stdout, "A");
				}
			}
			fprintf(stdout, "\n");
		}
	}
	free_df(df);
	free_article(article);
	return 0;
}

int usage()
{
        fprintf(stdout,
        "program: derrick\n"
        "version: %s\n"
        "author : Shigang Wu <wushigang@caas.cn>\n"
        "usage  : derrick <cmd> [options]\n"
        "\n"
        "commands:\n"
        " encode      use rs to encode a file\n"
        " decode      use rs to decode an encoded file\n"
        , TOSTR(VERSION)
        );
        return 1;
}

int main(int argc, char **argv)
{
	if(argc < 2)
        {
                return usage();
        }
	if(strcasecmp("encode", argv[1]) == 0) return main_encode(argc - 1, argv + 1);
	if(strcasecmp("decode", argv[1]) == 0) return main_decode(argc - 1, argv + 1);
	fprintf(stderr, " -- unknown command '%s' -- \n", argv[1]);
	return 1;
}
