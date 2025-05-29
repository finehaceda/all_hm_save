#include <math.h>
#include "bsalign/filereader.h"
#include "char.h"
#include "int.h"
#include "crc64.c"

struct ARTICLE
{
	int crc64_times, shift_times, shift_or_not;
};

struct ARTICLE * init_article(int shift_or_not)
{
        struct ARTICLE *article;
        article = (struct ARTICLE *)malloc(sizeof(struct ARTICLE));
	article->shift_or_not = shift_or_not;
	return article;
}

void free_article(struct ARTICLE *article)
{
	free(article);
}

void u8iseq_to_u1iseq(int len1, u8i *seq1, int len2, u1i *seq2, int len)
{
        int i, j, p;
        if(len1 < len2)
        {
                p = (u8i)pow(2, len1);
                for(i = 0; i < len; i++)
                {
                        seq2[i] = 0;
                        for(j = 0; j < len2 / len1; j++)
                        {
                                seq2[i] = seq2[i] + seq1[i * (len2 / len1) + j] * (u8i)pow(p, (len2 / len1) - j - 1);
                        }
                }
        }
        else
        {
                p = (u8i)pow(2, len2);
                u8i tmp1, tmp2;
                for(i = 0; i < len; i++)
                {
                        tmp1 = seq1[i];
                        for(j = 0; j < len1 / len2; j++)
                        {
                                tmp2 = tmp1 % (u8i)pow(p, (len1 / len2) - j - 1);
                                tmp2 = tmp1 - tmp2;
                                seq2[i * (len1 / len2) + j] = tmp2 / (u8i)pow(p, (len1 / len2) - j - 1);
                                tmp1 = tmp1 % (u8i)pow(p, (len1 / len2) - j - 1);
                        }
                }
        }
}

void u4iseq_to_u1iseq(int len1, u4i *seq1, int len2, u1i *seq2, int len)
{
	int i, j, p;
	if(len1 < len2)
	{
		p = (u8i)pow(2, len1);
		for(i = 0; i < len; i++)
                {
                        seq2[i] = 0;
                        for(j = 0; j < len2 / len1; j++)
                        {
                                seq2[i] = seq2[i] + seq1[i * (len2 / len1) + j] * (u8i)pow(p, (len2 / len1) - j - 1);
                        }
                }
	}
	else
	{
		p = (u8i)pow(2, len2);
		u4i tmp1, tmp2;
                for(i = 0; i < len; i++)
                {
                        tmp1 = seq1[i];
                        for(j = 0; j < len1 / len2; j++)
                        {
                                tmp2 = tmp1 % (u8i)pow(p, (len1 / len2) - j - 1);
                                tmp2 = tmp1 - tmp2;
                                seq2[i * (len1 / len2) + j] = tmp2 / (u8i)pow(p, (len1 / len2) - j - 1);
                                tmp1 = tmp1 % (u8i)pow(p, (len1 / len2) - j - 1);
                        }
                }
	}
}

void u1iseq_to_u1iseq(int len1, u1i *seq1, int len2, u1i *seq2, int len)
{
        int i, j, p;
        if(len1 < len2)
        {
                p = (u8i)pow(2, len1);
                for(i = 0; i < len; i++)
                {
                        seq2[i] = 0;
                        for(j = 0; j < len2 / len1; j++)
                        {
                                seq2[i] = seq2[i] + seq1[i * (len2 / len1) + j] * (u8i)pow(p, (len2 / len1) - j - 1);
                        }
                }
        }
        else
        {
                p = (u8i)pow(2, len2);
                u1i tmp1, tmp2;
                for(i = 0; i < len; i++)
                {
                        tmp1 = seq1[i];
                        for(j = 0; j < len1 / len2; j++)
                        {
                                tmp2 = tmp1 % (u8i)pow(p, (len1 / len2) - j - 1);
                                tmp2 = tmp1 - tmp2;
                                seq2[i * (len1 / len2) + j] = tmp2 / (u8i)pow(p, (len1 / len2) - j - 1);
                                tmp1 = tmp1 % (u8i)pow(p, (len1 / len2) - j - 1);
                        }
                }
        }
}

void u1iseq_to_u4iseq(int len1, u1i *seq1, int len2, u4i *seq2, int len)
{
        int i, j, p;
        if(len1 < len2)
        {
                p = (u8i)pow(2, len1);
                for(i = 0; i < len; i++)
                {
                        seq2[i] = 0;
                        for(j = 0; j < len2 / len1; j++)
                        {
                                seq2[i] = seq2[i] + seq1[i * (len2 / len1) + j] * (u8i)pow(p, (len2 / len1) - j - 1);
                        }
                }
        }
        else
        {
                p = (u8i)pow(2, len2);
                u1i tmp1, tmp2;
                for(i = 0; i < len; i++)
                {
                        tmp1 = seq1[i];
                        for(j = 0; j < len1 / len2; j++)
                        {
                                tmp2 = tmp1 % (u8i)pow(p, (len1 / len2) - j - 1);
                                tmp2 = tmp1 - tmp2;
                                seq2[i * (len1 / len2) + j] = tmp2 / (u8i)pow(p, (len1 / len2) - j - 1);
                                tmp1 = tmp1 % (u8i)pow(p, (len1 / len2) - j - 1);
                        }
                }
        }
}

void u1iseq_to_u8iseq(int len1, u1i *seq1, int len2, u8i *seq2, int len)
{
        int i, j, p;
        if(len1 < len2)
        {
                p = (u8i)pow(2, len1);
                for(i = 0; i < len; i++)
                {
                        seq2[i] = 0;
                        for(j = 0; j < len2 / len1; j++)
                        {
                                seq2[i] = seq2[i] + seq1[i * (len2 / len1) + j] * (u8i)pow(p, (len2 / len1) - j - 1);
                        }
                }
        }
        else
        {
                p = (u8i)pow(2, len2);
                u1i tmp1, tmp2;
                for(i = 0; i < len; i++)
                {
                        tmp1 = seq1[i];
                        for(j = 0; j < len1 / len2; j++)
                        {
                                tmp2 = tmp1 % (u8i)pow(p, (len1 / len2) - j - 1);
                                tmp2 = tmp1 - tmp2;
                                seq2[i * (len1 / len2) + j] = tmp2 / (u8i)pow(p, (len1 / len2) - j - 1);
                                tmp1 = tmp1 % (u8i)pow(p, (len1 / len2) - j - 1);
                        }
                }
        }
}

void intseq_to_u1iseq(int len1, int *seq1, int len2, u1i *seq2, int len)
{
        int i, j, p;
        if(len1 < len2)
        {
                p = (u8i)pow(2, len1);
                for(i = 0; i < len; i++)
                {
                        seq2[i] = 0;
                        for(j = 0; j < len2 / len1; j++)
                        {
                                seq2[i] = seq2[i] + seq1[i * (len2 / len1) + j] * (u8i)pow(p, (len2 / len1) - j - 1);
                        }
                }
        }
        else
        {
                p = (u8i)pow(2, len2);
                int tmp1, tmp2;
                for(i = 0; i < len; i++)
                {
                        tmp1 = seq1[i];
                        for(j = 0; j < len1 / len2; j++)
                        {
                                tmp2 = tmp1 % (u8i)pow(p, (len1 / len2) - j - 1);
                                tmp2 = tmp1 - tmp2;
                                seq2[i * (len1 / len2) + j] = tmp2 / (u8i)pow(p, (len1 / len2) - j - 1);
                                tmp1 = tmp1 % (u8i)pow(p, (len1 / len2) - j - 1);
                        }
                }
        }
}

void u1iseq_to_intseq(int len1, u1i *seq1, int len2, int *seq2, int len)
{
        int i, j, p;
        if(len1 < len2)
        {
                p = (u8i)pow(2, len1);
                for(i = 0; i < len; i++)
                {
                        seq2[i] = 0;
                        for(j = 0; j < len2 / len1; j++)
                        {
                                seq2[i] = seq2[i] + seq1[i * (len2 / len1) + j] * (u8i)pow(p, (len2 / len1) - j - 1);
                        }
                }
        }
        else
        {
                p = (u8i)pow(2, len2);
                u1i tmp1, tmp2;
                for(i = 0; i < len; i++)
                {
                        tmp1 = seq1[i];
                        for(j = 0; j < len1 / len2; j++)
                        {
                                tmp2 = tmp1 % (u8i)pow(p, (len1 / len2) - j - 1);
                                tmp2 = tmp1 - tmp2;
                                seq2[i * (len1 / len2) + j] = tmp2 / (u8i)pow(p, (len1 / len2) - j - 1);
                                tmp1 = tmp1 % (u8i)pow(p, (len1 / len2) - j - 1);
                        }
                }
        }
}

void yihuo_pi(u1i *seq, int len, char *pifn)
{
        FILE *fp;
        int c, i, j;
        u1i *buffer;
        int size;
        fp = fopen(pifn, "r");
        size = 0;
        while((c = fgetc(fp)) != -1)
        {
                size = size + 2;
        }
        fclose(fp);
        buffer = (u1i *)calloc(size, sizeof(u1i));
        fp = fopen(pifn, "r");
        size = 0;
        while((c = fgetc(fp)) != -1)
        {
		if(c == 67)
		{
			c = 1;
		}
		else if(c == 71)
		{
			c = 2;
		}
		else if(c == 84)
		{
			c = 3;
		}
		else
		{
			c = 0;
		}
                intseq_to_u1iseq(2, &c, 1, buffer + size, 1);
                size = size + 2;
        }
        fclose(fp);
        for(i = 0; i < len; i++)
        {
                j = i % size;
                seq[i] = seq[i] ^ buffer[j];
        }
        free(buffer);
}

struct ENCODEFILE
{
        int n, k, nuss, ver, hrm, nblocks;
        u1i *bitseq, *qseq, *rseq, *tseq;
        int  bitlen,  qlen,  rlen,  tlen;
	struct rs1 *rs1;
        struct rs2 *rs2;
        DTYPE1 *block1;
        DTYPE2 *block2;
};

struct ENCODEFILE * init_ef(int n, int k, int nuss, int ver, int hrm, int nblocks)
{
	struct ENCODEFILE *ef;
        ef = (struct ENCODEFILE *)malloc(sizeof(struct ENCODEFILE));
	ef->n = n;
        ef->k = k;
        ef->nuss = nuss;
        ef->ver = ver;
        ef->hrm = hrm;
        ef->nblocks = nblocks;
	ef->tlen = ef->nuss * ef->hrm;
	ef->rlen = ef->tlen * 2;
	ef->qlen = ef->k * ef->rlen;
	ef->bitlen = ef->qlen - 64;
	ef->bitseq = (u1i *)calloc(ef->nblocks * ef->bitlen, sizeof(u1i));
        ef->qseq = (u1i *)calloc(ef->nblocks * ef->qlen, sizeof(u1i));
	ef->rseq = (u1i *)calloc(ef->nblocks * ef->k * ef->rlen, sizeof(u1i));
	ef->tseq = (u1i *)calloc(ef->nblocks * ef->n * ef->tlen, sizeof(u1i));
	if(ef->n <= 255)
        {
                {
                        ef->rs1 = (struct rs1 *)init_rs_char(8, 0x11d, 1, 1, ef->n - ef->k, 0);
                }
                ef->block1 = (DTYPE1 *)calloc(ef->rs1->nn, sizeof(DTYPE1));
                ef->rs2 = (struct rs2 *)init_rs_int(10, 0x409, 1, 1, 32, 0);
                ef->block2 = (DTYPE2 *)calloc(ef->rs2->nn, sizeof(DTYPE2));
        }
        else
        {
                if(ef->n == 65535)
                {
                        ef->rs2 = (struct rs2 *)init_rs_int(16, 0x1100b, 1, 1, ef->n - ef->k, 0);
                }
                else if(ef->n == 16383)
                {
                        ef->rs2 = (struct rs2 *)init_rs_int(14, 0x4443, 1, 1, ef->n - ef->k, 0);
                }
                else if(ef->n == 4095)
                {
                        ef->rs2 = (struct rs2 *)init_rs_int(12, 0x1053, 1, 1, ef->n - ef->k, 0);
                }
                else
                {
                        ef->rs2 = (struct rs2 *)init_rs_int(10, 0x409, 1, 1, ef->n - ef->k, 0);
                }
                ef->block2 = (DTYPE2 *)calloc(ef->rs2->nn, sizeof(DTYPE2));
                ef->rs1 = (struct rs1 *)init_rs_char(8, 0x11d, 1, 1, 32, 0);
                ef->block1 = (DTYPE1 *)calloc(ef->rs1->nn, sizeof(DTYPE1));
        }
	return ef;
}

void free_ef(struct ENCODEFILE *ef)
{
        free(ef->bitseq);
        free(ef->qseq);
	free(ef->rseq);
	free(ef->tseq);
	free(ef->rs1);
        free(ef->rs2);
        free(ef->block1);
        free(ef->block2);
        free(ef);
}

void add_crc64(struct ENCODEFILE *ef)
{
        int i, j;
        u1i *buffer;
        int size;
        u8i crc;
        size = ef->bitlen / 2;
        buffer = (u1i *)calloc(size, sizeof(u1i));
        for(i = 0; i < ef->nblocks; i++)
        {
                for(j = 0; j < ef->bitlen; j++)
                {
                        ef->qseq[i * ef->qlen + j] = ef->bitseq[i * ef->bitlen + j];
                }
                u1iseq_to_u1iseq(1, ef->qseq + i * ef->qlen, 2, buffer, size);
                crc = crc64(0, buffer, size);
                u8iseq_to_u1iseq(64, &crc, 1, ef->qseq + i * ef->qlen + ef->bitlen, 1);
        }
        free(buffer);
}

void del_crc64(struct ENCODEFILE *ef)
{
        int i, j;
        for(i = 0; i < ef->nblocks; i++)
        {
                for(j = 0; j < ef->bitlen; j++)
                {
			ef->bitseq[i * ef->bitlen + j] = ef->qseq[i * ef->qlen + j];
                }
        }
}

void qseq_to_rseq(struct ENCODEFILE *ef)
{
	int i, j, k1, k2;
	i = 0;
	for(j = 0; j < ef->nblocks; j++)
	{
		for(k2 = 0; k2 < ef->rlen; k2++)
		{
			for(k1 = 0; k1 < ef->k; k1++)
			{
				ef->rseq[(j * ef->k + k1) * ef->rlen + k2] = ef->qseq[i];
				i++;
			}
		}
	}
}

void rseq_to_qseq(struct ENCODEFILE *ef)
{
	int i, j, k1, k2;
	i = 0;
        for(j = 0; j < ef->nblocks; j++)
        {
                for(k2 = 0; k2 < ef->rlen; k2++)
                {
                        for(k1 = 0; k1 < ef->k; k1++)
                        {
				ef->qseq[i] = ef->rseq[(j * ef->k + k1) * ef->rlen + k2];
                                i++;
                        }
                }
        }
}

void add_rs(struct ENCODEFILE *ef)
{
	int i, j, k;
	if(ef->n <= 255)
	{
		for(i = 0; i < ef->nblocks; i++)
		{
			for(j = 0; j < ef->nuss; j++)
			{
				for(k = 0; k < ef->k; k++)
				{
					u1iseq_to_u1iseq(2, ef->tseq + (i * ef->n + k) * ef->tlen + (j * ef->hrm), ef->hrm * 2, ef->block1 + k, 1);
				}
				ENCODE_RS1(ef->rs1, &ef->block1[0], &ef->block1[ef->rs1->nn - ef->rs1->nroots]);
				for(k = ef->k; k < ef->n; k++)
				{
					u1iseq_to_u1iseq(ef->hrm * 2, ef->block1 + k, 2, ef->tseq + (i * ef->n + k) * ef->tlen + (j * ef->hrm), 1);
				}
				
			}
		}
	}
	else
	{
		for(i = 0; i < ef->nblocks; i++)
                {
                        for(j = 0; j < ef->nuss; j++)
                        {
                                for(k = 0; k < ef->k; k++)
                                {
                                        u1iseq_to_intseq(2, ef->tseq + (i * ef->n + k) * ef->tlen + (j * ef->hrm), ef->hrm * 2, ef->block2 + k, 1);
                                }
                                ENCODE_RS2(ef->rs2, &ef->block2[0], &ef->block2[ef->rs2->nn - ef->rs2->nroots]);
                                for(k = ef->k; k < ef->n; k++)
                                {
                                        intseq_to_u1iseq(ef->hrm * 2, ef->block2 + k, 2, ef->tseq + (i * ef->n + k) * ef->tlen + (j * ef->hrm), 1);
                                }

                        }
                }
	}
}

struct DECODEFILE
{
        int n, k, nuss, mat, mis, gapo, gape, maxchange, maxdelete, mode, jump, upper, lower, raise, timeout, allowed, hasref, useref, begblock, endblock, ver, hrm, nblocks;
	clock_t start, finish;
	double time;
	int number;
	int diff;
	int last;
        u1i *tseq, *qseq, *qlts, *Rseq;
	int  tlen,  qlen,  qltl,  Rlen;
	struct rs1 *rs1;
        struct rs2 *rs2;
        DTYPE1 *block1;
        DTYPE2 *block2;
        int *derrlocs1, *derrlocs2, erasures, *derrors;
	int **mtx, **ugs, **lgs, *idx;
	int datsize, *datkey, *datvalue;
        int qltsize, *qltkey, *qltvalue;
	int h4psize, *h4pkey, *h4pvalue;
	int h3psize, *h3pkey, *h3pvalue;
        int hpsize,  *hpkey,  *hpvalue;
        int errsize, *errkey, *errvalue, *errplus;
        int *level, *sr;
	int *changec, *changen, *changer;
	int *deletec, *deleten, *deleter;
	u1i *buffer1, *buffer2, *buffer3;
};

struct DECODEFILE * init_df(int n, int k, int nuss, int mat, int mis, int gapo, int gape, int maxchange, int maxdelete, int mode, int jump, int upper, int lower, int raise, int timeout, int allowed, int begblock, int endblock, int ver, int hrm, int nblocks)
{
	struct DECODEFILE *df;
	int i, j, k1, k2;
	int size1, size2, size3;
        df = (struct DECODEFILE *)malloc(sizeof(struct DECODEFILE));
	df->n = n;
	df->k = k;
	df->nuss = nuss;
	df->mat = mat;
        df->mis = mis;
        df->gapo = gapo;
        df->gape = gape;
	if(maxchange > 0)
        {
                df->maxchange = maxchange;
        }
        else
        {
                df->maxchange = 0;
        }
        if(maxdelete > 0)
        {
                df->maxdelete = maxdelete;
        }
        else
        {
                df->maxdelete = 0;
        }
	if(df->maxdelete > df->n - df->k)
	{
		df->maxdelete = df->n - df->k;
	}
	if(df->maxchange < df->maxdelete)
	{
		df->maxchange = df->maxdelete;
	}
	df->mode = mode;
	df->jump = jump;
	df->upper = upper;
	df->lower = lower;
	df->raise = raise;
	df->timeout = timeout;
	df->allowed = allowed;
	df->last = df->nuss;
	df->hasref = 0;
	df->begblock = begblock;
        df->endblock = endblock;
	df->ver = ver;
	df->hrm = hrm;
	df->nblocks = nblocks;
	if(df->begblock < 1 || df->endblock < 1 || df->begblock > df->nblocks || df->endblock > df->nblocks || df->begblock > df->endblock)
	{
		df->begblock = 1;
		df->endblock = df->nblocks;
	}
        df->tlen = df->nuss * df->hrm;
	df->qlen = df->tlen * 2;
        df->qltl = df->qlen;
	df->Rlen = df->tlen;
        df->tseq = (u1i *)calloc(df->nblocks * df->n * df->tlen, sizeof(u1i));
	df->qseq = (u1i *)calloc(df->nblocks * df->n * df->qlen, sizeof(u1i));
        df->qlts = (u1i *)calloc(df->nblocks * df->n * df->qltl, sizeof(u1i));
        df->Rseq = (u1i *)calloc(df->nblocks * df->n * df->Rlen, sizeof(u1i));
        for(i = 0; i < df->nblocks * df->n; i++)
        {
		for(j = 0; j < df->tlen; j++)
                {
                        df->tseq[i * df->tlen + j] = 0;
                }
		for(j = 0; j < df->qlen; j++)
		{
			df->qseq[i * df->qlen + j] = 0;
		}
		for(j = 0; j < df->qltl; j++)
                {
                        df->qlts[i * df->qltl + j] = 0;
                }
		for(j = 0; j < df->Rlen; j++)
                {
                        df->Rseq[i * df->Rlen + j] = 0;
                }
        }
	if(df->n <= 255)
        {
		{
			df->rs1 = (struct rs1 *)init_rs_char(8, 0x11d, 1, 1, df->n - df->k, 0);
		}
                df->block1 = (DTYPE1 *)calloc(df->rs1->nn, sizeof(DTYPE1));
                df->derrlocs1 = (int *)calloc(df->rs1->nroots, sizeof(int));
                df->rs2 = (struct rs2 *)init_rs_int(10, 0x409, 1, 1, 32, 0);
                df->block2 = (DTYPE2 *)calloc(df->rs2->nn, sizeof(DTYPE2));
                df->derrlocs2 = (int *)calloc(df->rs2->nroots, sizeof(int));
        }
	else
        {
		if(df->n == 65535)
                {
                        df->rs2 = (struct rs2 *)init_rs_int(16, 0x1100b, 1, 1, df->n - df->k, 0);
                }
                else if(df->n == 16383)
                {
                        df->rs2 = (struct rs2 *)init_rs_int(14, 0x4443, 1, 1, df->n - df->k, 0);
                }
		else if(df->n == 4095)
                {
                        df->rs2 = (struct rs2 *)init_rs_int(12, 0x1053, 1, 1, df->n - df->k, 0);
                }
		else
		{
			df->rs2 = (struct rs2 *)init_rs_int(10, 0x409, 1, 1, df->n - df->k, 0);
		}
                df->block2 = (DTYPE2 *)calloc(df->rs2->nn, sizeof(DTYPE2));
                df->derrlocs2 = (int *)calloc(df->rs2->nroots, sizeof(int));
                df->rs1 = (struct rs1 *)init_rs_char(8, 0x11d, 1, 1, 32, 0);
                df->block1 = (DTYPE1 *)calloc(df->rs1->nn, sizeof(DTYPE1));
                df->derrlocs1 = (int *)calloc(df->rs1->nroots, sizeof(int));
        }
	df->derrors = (int *)calloc(df->nuss, sizeof(int));
	df->mtx = (int **)calloc(df->n, sizeof(int *));
        df->ugs = (int **)calloc(df->n, sizeof(int *));
        df->lgs = (int **)calloc(df->n, sizeof(int *));
	for(i = 0; i < df->n; i++)
	{
		df->mtx[i] = (int *)calloc((df->tlen + 1) * (df->qlen + 1), sizeof(int));
                df->ugs[i] = (int *)calloc((df->tlen + 1) * (df->qlen + 1), sizeof(int));
                df->lgs[i] = (int *)calloc((df->tlen + 1) * (df->qlen + 1), sizeof(int));
	}
	df->idx = (int *)calloc(df->n * df->nuss, sizeof(int));
	for(i = 0; i < df->n; i++)
	{
		k1 = 0;
		k2 = 0;
		j = k1 * (df->qlen + 1) + k2;
		df->mtx[i][j] = 0;
		k2 = 0;
		for(k1 = 1; k1 < df->tlen + 1; k1++)
		{
			j = k1 * (df->qlen + 1) + k2;
			df->mtx[i][j] = df->gapo + k1 * df->gape;
			df->lgs[i][j] = df->mtx[i][j] + df->gapo;
		}
		k1 = 0;
		for(k2 = 1; k2 < df->qlen + 1; k2++)
                {
			j = k1 * (df->qlen + 1) + k2;
                        df->mtx[i][j] = df->gapo + k2 * df->gape;
			df->ugs[i][j] = df->mtx[i][j] + df->gapo;
                }
		df->idx[i * df->nuss] = 0;
	}
	df->datkey = (int *)calloc(df->n, sizeof(int));
        df->qltkey = (int *)calloc(df->n, sizeof(int));
        df->h4pkey = (int *)calloc(df->n, sizeof(int));
        df->h3pkey = (int *)calloc(df->n, sizeof(int));
        df->hpkey = (int *)calloc(df->n, sizeof(int));
        df->errkey = (int *)calloc(df->n, sizeof(int));
        df->datvalue = (int *)calloc(df->n, sizeof(int));
        df->qltvalue = (int *)calloc(df->n, sizeof(int));
        df->h4pvalue = (int *)calloc(df->n, sizeof(int));
        df->h3pvalue = (int *)calloc(df->n, sizeof(int));
        df->hpvalue = (int *)calloc(df->n, sizeof(int));
        df->errvalue = (int *)calloc(df->n, sizeof(int));
	df->errplus = (int *)calloc(df->n * df->nuss, sizeof(int));
	for(i = 0; i < df->n; i++)
        {
                df->errplus[i * df->nuss] = 0;
        }
        df->level = (int *)calloc(df->nuss, sizeof(int));
        df->sr = (int *)calloc(df->nuss, sizeof(int));
	df->changec = (int *)calloc(df->nuss * df->n, sizeof(int));
        df->changen = (int *)calloc(df->nuss, sizeof(int));
        df->changer = (int *)calloc(df->nuss, sizeof(int));
        df->deletec = (int *)calloc(df->nuss * df->n, sizeof(int));
        df->deleten = (int *)calloc(df->nuss, sizeof(int));
        df->deleter = (int *)calloc(df->nuss, sizeof(int));
        size1 = df->k * df->tlen * 2;
        size2 = size1;
        size3 = size2 / 2;
        df->buffer1 = (u1i *)calloc(size1, sizeof(u1i));
        df->buffer2 = (u1i *)calloc(size2, sizeof(u1i));
        df->buffer3 = (u1i *)calloc(size3, sizeof(u1i));
	return df;
}

void free_df(struct DECODEFILE *df)
{
	int i;
        free(df->tseq);
        free(df->qseq);
        free(df->qlts);
        free(df->Rseq);
	free(df->rs1);
        free(df->rs2);
        free(df->block1);
        free(df->block2);
        free(df->derrlocs1);
        free(df->derrlocs2);
	free(df->derrors);
	for(i = 0; i < df->n; i++)
	{
		free(df->mtx[i]);
                free(df->ugs[i]);
                free(df->lgs[i]);
	}
	free(df->mtx);
        free(df->ugs);
        free(df->lgs);
	free(df->idx);
        free(df->datkey);
        free(df->qltkey);
        free(df->h4pkey);
        free(df->h3pkey);
        free(df->hpkey);
        free(df->errkey);
        free(df->datvalue);
        free(df->qltvalue);
        free(df->h4pvalue);
        free(df->h3pvalue);
        free(df->hpvalue);
        free(df->errvalue);
	free(df->errplus);
        free(df->level);
        free(df->sr);
	free(df->changec);
        free(df->changen);
        free(df->changer);
        free(df->deletec);
        free(df->deleten);
        free(df->deleter);
	free(df->buffer1);
        free(df->buffer2);
        free(df->buffer3);
        free(df);
}

int get_df_idx(struct DECODEFILE *df, int rid, int sid)
{
	int df_idx, max_score, cbg, ced, i, j, ii, ij, i1j, ij1, i1j1;
	cbg = sid * df->hrm + 1;
	ced = cbg + df->hrm;
	for(i = cbg; i < ced; i++)
	{
		for(j = 1; j < df->qlen + 1; j++)
		{
			ii = rid % df->n;
			ij = i * (df->qlen + 1) + j;
			i1j1 = (i - 1) * (df->qlen + 1) + (j - 1);
			i1j = (i - 1) * (df->qlen + 1) + j;
			ij1 = i * (df->qlen + 1) + (j - 1);
			if(df->tseq[rid * df->tlen + (i - 1)] == df->qseq[rid * df->qlen + (j - 1)])
                	{
                	        df->mtx[ii][ij] = df->mtx[ii][i1j1] + df->mat;
                	}
                	else
                	{
                	        df->mtx[ii][ij] = df->mtx[ii][i1j1] + df->mis;
                	}
                	if(df->mtx[ii][ij] < df->ugs[ii][i1j] + df->gape)
                	{
                	        df->mtx[ii][ij] = df->ugs[ii][i1j] + df->gape;
                	}
                	if(df->mtx[ii][ij] < df->lgs[ii][ij1] + df->gape)
                	{
                	        df->mtx[ii][ij] = df->lgs[ii][ij1] + df->gape;
                	}
                	if(df->ugs[ii][i1j] + df->gape < df->mtx[ii][ij] + df->gapo)
                	{
                	        df->ugs[ii][ij] = df->mtx[ii][ij] + df->gapo;
                	}
                	else
                	{
                	        df->ugs[ii][ij] = df->ugs[ii][i1j] + df->gape;
                	}
                	if(df->lgs[ii][ij1] + df->gape < df->mtx[ii][ij] + df->gapo)
                	{
                	        df->lgs[ii][ij] = df->mtx[ii][ij] + df->gapo;
                	}
                	else
                	{
                	        df->lgs[ii][ij] = df->lgs[ii][ij1] + df->gape;
                	}
		}
	}
	i = ced - 1;
        {
		j = 0;
		ii = rid % df->n;
		ij = i * (df->qlen + 1) + j;
		df_idx = j;
		max_score = df->mtx[ii][ij];
                for(j = 1; j < df->qlen + 1; j++)
                {
			ii = rid % df->n;
			ij = i * (df->qlen + 1) + j;
			if(max_score <= df->mtx[ii][ij])
			{
				df_idx = j;
				max_score = df->mtx[ii][ij];
			}
		}
	}
	return df_idx;
}

void get_df_dat(struct DECODEFILE *df, int bid, int sid)
{
	int i, j;
	for(i = 0; i < df->n; i++)
        {
		df->datkey[i] = i;
		j = (bid * df->n + i) * df->qlen + df->idx[i * df->nuss + sid];
                u1iseq_to_intseq(2, df->qseq + j, df->hrm * 2, df->datvalue + i, 1);
        }
	df->datsize = df->n;
}

int is_hp(u1i *seq, int beg, int len, int id)
{
	int i;
	if(beg < id)
	{
		return 1;
	}
	for(i = beg + 1; i < beg + len; i++)
	{
		if(seq[beg] != seq[i])
		{
			return 1;
		}
	}
	return 0;
}

int is_change(u1i *seq, int beg, int len, int id)
{
	int i;
	for(i = id; i < beg + len; i++)
	{
		if(seq[i] != seq[i + 1])
		{
			return 0;
		}
	}
	return 1;
}

int get_change(u1i *seq, int beg, int len, int id)
{
	int change, i;
	u1i *buffer;
	int size;
	buffer = (u1i *)calloc(len, sizeof(u1i));
	size = 0;
	for(i = beg; i < id; i++)
	{
		buffer[size] = seq[i];
		size++;
	}
	for(i = id; i < beg + len; i++)
	{
		buffer[size] = seq[i + 1];
                size++;
	}
	u1iseq_to_intseq(2, buffer, len * 2, &change, 1);
	free(buffer);
	return change;
}

void maopaosort(int *keybuffer, int *valuebuffer, int size, int flag)
{
        int i, j, tmp;
        if(flag == 0)
        {
                for(i = 1; i <= size - 1; i++)
                {
                        for(j = size - 1; j >= i; j--)
                        {
                                if(valuebuffer[j - 1] > valuebuffer[j])
                                {
                                        tmp = valuebuffer[j - 1];
                                        valuebuffer[j - 1] = valuebuffer[j];
                                        valuebuffer[j] = tmp;
                                        tmp = keybuffer[j - 1];
                                        keybuffer[j - 1] = keybuffer[j];
                                        keybuffer[j] = tmp;
                                }
                        }
                }
        }
        else
        {
                for(i = 1; i <= size - 1; i++)
                {
                        for(j = size - 1; j >= i; j--)
                        {
                                if(valuebuffer[j - 1] < valuebuffer[j])
                                {
                                        tmp = valuebuffer[j - 1];
                                        valuebuffer[j - 1] = valuebuffer[j];
                                        valuebuffer[j] = tmp;
                                        tmp = keybuffer[j - 1];
                                        keybuffer[j - 1] = keybuffer[j];
                                        keybuffer[j] = tmp;
                                }
                        }
                }
        }
}

void get_df_hp(struct DECODEFILE *df, int bid, int sid)
{
	int i, j, begidx, endidx, minidx, minqlt;
	for(i = 0; i < df->n; i++)
	{
		df->qltkey[i] = i;
		begidx = (bid * df->n + i) * df->qlen + df->idx[i * df->nuss + sid];
		endidx = begidx + df->hrm;
		minidx = begidx;
		if(is_change(df->qseq, begidx, df->hrm, begidx) == 0)
		{
			minqlt = df->qlts[minidx];
		}
		else
		{
			minqlt = 83;
		}
		for(j = begidx + 1; j < endidx; j++)
		{
			if(is_change(df->qseq, begidx, df->hrm, j) == 0 && minqlt > df->qlts[j])
			{
				minidx = j;
				minqlt = df->qlts[j];
			}
		}
		df->qltvalue[i] = minqlt;
		df->hpvalue[i] = get_change(df->qseq, begidx, df->hrm, minidx);
	}
	maopaosort(df->qltkey, df->qltvalue, df->n, 0);
	for(i = 0; i < df->n; i++)
	{
		df->qltvalue[i] = df->hpvalue[df->qltkey[i]];
	}
	for(i = 0; i < df->n; i++)
	{
		df->hpkey[i] = df->qltkey[i];
		df->hpvalue[i] = df->qltvalue[i];
	}
	df->hpsize = df->sr[sid];
	if(df->ver >= 1 && df->level[sid] == 1)
	{
		fprintf(stderr, "%d/%d | candidate |", sid, df->nuss);
		for(i = 0; i < df->hpsize; i++)
		{
			fprintf(stderr, " %d,%d", df->hpkey[i], df->hpvalue[i]);
		}
		fprintf(stderr, "\n");
	}
}

void get_df_qlt(struct DECODEFILE *df, int bid, int sid)
{
        int i, j, k;
        for(i = 0; i < df->n; i++)
        {
                df->qltkey[i] = i;
                j = (bid * df->n + i) * df->qltl + df->idx[i * df->nuss + sid];
                df->qltvalue[i] = df->qlts[j];
		k = j + df->hrm;
                for(j = j + 1; j < k; j++)
                {
                        if(df->qltvalue[i] > df->qlts[j])
                        {
                                df->qltvalue[i] = df->qlts[j];
                        }
                }
        }
        maopaosort(df->qltkey, df->qltvalue, df->n, 0);
        df->qltsize = df->sr[sid];
	if(df->ver >= 1 && df->level[sid] == 1)
	{
		fprintf(stderr, "%d/%d | candidate |", sid, df->nuss);
		for(i = 0; i < df->qltsize; i++)
		{
			fprintf(stderr, " %d,%d", df->qltkey[i], df->qltvalue[i]);
		}
		fprintf(stderr, "\n");
	}
}

int zuhe(int *c, int n, int r)
{
	if(n < r)
	{
		return 1;
	}
        int flag, i, cnt;
        flag = 1;
        for(i = n - 1; i >= n - r; i--)
        {
                flag = flag * c[i];
        }
        if(flag == 1)
        {
                return 1;
        }
        else
        {
                for(i = 0; i < n; i++)
                {
                        if(c[i] == 1)
                        {
                                break;
                        }
                }
                cnt = 0;
                for(i = i; i < n; i++)
                {
                        if(c[i] == 0)
                        {
                                c[i] = 1;
                                cnt--;
                                for(i = 0; i < cnt; i++)
                                {
                                        c[i] = 1;
                                }
                                break;
                        }
                        else
                        {
                                c[i] = 0;
                                cnt++;
                        }
                }
                return 0;
        }
}

int get_df_derrors(struct DECODEFILE *df, struct ARTICLE *article, int bid, int sid, int jiu_or_diu)
{
	int df_derrors, i, j;
	df_derrors = 0;
	if(df->ver >= 1)
	{
		fprintf(stderr, "%d/%d | ", sid, df->nuss);
		if(df->mode == 2)
		{
			if(jiu_or_diu == 0)
			{
				fprintf(stderr, "ref |");
			}
			else if(jiu_or_diu == 1)
			{
				fprintf(stderr, "hard |");
			}
			else
			{
				fprintf(stderr, "delete=%d/%d |", df->deleter[sid], df->deleten[sid]);
			}
		}
		else
		{
			if(jiu_or_diu == 0)
			{
                                fprintf(stderr, "ref |");
			}
			else if(jiu_or_diu == 1)
                        {
                                fprintf(stderr, "hard |");
                        }
			else if(jiu_or_diu == 2)
			{
                        	fprintf(stderr, "delete=%d/%d |", df->changer[sid], df->changen[sid]);
			}
			else
			{
                                fprintf(stderr, "change=%d/%d |", df->changer[sid], df->changen[sid]);
			}
		}
	}
	if(df->n <= 255)
	{
		for(i = 0; i < df->n; i++)
		{
			j = (bid * df->n + i) * df->tlen + (sid * df->hrm);
			u1iseq_to_u1iseq(df->hrm * 2, df->block1 + i, 2, df->tseq + j, 1);
			if(sid + 1 < df->nuss)
			{
				df->errplus[i * df->nuss + (sid + 1)] = df->errplus[i * df->nuss + sid];
				if(df->block1[i] != (DTYPE1)df->datvalue[i])
				{
					df->errplus[i * df->nuss + (sid + 1)]++;
					df_derrors++;
					if(df->ver >= 1)
					{
						fprintf(stderr, " %d,%d", i, df->block1[i]);
					}
				}
				if(article->shift_or_not == 1)
				{
					df->idx[i * df->nuss + (sid + 1)] = get_df_idx(df, bid * df->n + i, sid);
				}
				else
				{
					df->idx[i * df->nuss + (sid + 1)] = df->idx[i * df->nuss + sid] + 4;
				}
				if(df->idx[i * df->nuss + (sid + 1)] != df->idx[i * df->nuss + sid] + 4)
				{
					article->shift_times++;
				}
			}
			else
			{
				if(df->block1[i] != (DTYPE1)df->datvalue[i])
				{
					df_derrors++;
					if(df->ver >= 1)
                                        {
                                                fprintf(stderr, " %d,%d", i, df->block1[i]);
                                        }
				}
			}
		}
	}
	else
	{
		for(i = 0; i < df->n; i++)
		{
			j = (bid * df->n + i) * df->tlen + (sid * df->hrm);
			intseq_to_u1iseq(df->hrm * 2, df->block2 + i, 2, df->tseq + j, 1);
			if(sid + 1 < df->nuss)
			{
				df->errplus[i * df->nuss + (sid + 1)] = df->errplus[i * df->nuss + sid];
				if(df->block2[i] != (DTYPE2)df->datvalue[i])
				{
					df->errplus[i * df->nuss + (sid + 1)]++;
                                        df_derrors++;
					if(df->ver >= 1)
                                        {
                                                fprintf(stderr, " %d,%d", i, df->block2[i]);
                                        }
				}
				if(article->shift_or_not == 1)
				{
					df->idx[i * df->nuss + (sid + 1)] = get_df_idx(df, bid * df->n + i, sid);
				}
				else
				{
					df->idx[i * df->nuss + (sid + 1)] = df->idx[i * df->nuss + sid] + 4;
				}
				if(df->idx[i * df->nuss + (sid + 1)] != df->idx[i * df->nuss + sid] + 4)
                                {
                                        article->shift_times++;
                                }
			}
			else
			{
				if(df->block2[i] != (DTYPE2)df->datvalue[i])
                                {
                                        df_derrors++;
					if(df->ver >= 1)
                                        {
                                                fprintf(stderr, " %d,%d", i, df->block2[i]);
                                        }
                                }
			}
		}
	}
	if(df->ver >= 1)
        {
                fprintf(stderr, "\n");
        }
	return df_derrors;
}

void init_direct(struct DECODEFILE *df, int sid)
{
	df->level[sid] = 1;
	df->derrors[sid] = -1;
}

void init_change_hp(struct DECODEFILE *df, int sid)
{
	int i;
	df->level[sid] = 3;
	df->changen[sid] = df->sr[sid];
        df->changer[sid] = 1;
        for(i = 0; i < df->changer[sid]; i++)
        {
        	df->changec[sid * df->n + i] = 1;
        }
        for(i = df->changer[sid]; i < df->changen[sid]; i++)
        {
        	df->changec[sid * df->n + i] = 0;
        }
}

void init_delete_qlt(struct DECODEFILE *df, int sid)
{
        int i;
	df->level[sid] = 2;
	df->deleten[sid] = df->sr[sid];
        df->deleter[sid] = 2;
	for(i = 0; i < df->deleter[sid]; i++)
	{
		df->deletec[sid * df->n + i] = 1;
	}
	for(i = df->deleter[sid]; i < df->deleten[sid]; i++)
	{
		df->deletec[sid * df->n + i] = 0;
	}
}

int direct_else(struct DECODEFILE *df, int sid)
{
	df->sr[sid] = df->lower;
	if(df->mode == 2)
	{
                init_delete_qlt(df, sid);
	}
	else
	{
                init_change_hp(df, sid);
	}
        return 1;
}

int direct(struct DECODEFILE *df, struct ARTICLE *article, int bid, int sid)
{
	int i;
	while(1)
	{
		if(df->n <= 255)
        	{
        	        for(i = 0; i < df->n; i++)
        	        {
				df->block1[i] = df->datvalue[i];
        	        }
        	        df->erasures = 0;
        	        df->derrors[sid] = DECODE_RS1(df->rs1, df->block1, df->derrlocs1, df->erasures);
        	}
        	else
        	{
        	        for(i = 0; i < df->n; i++)
        	        {
				df->block2[i] = df->datvalue[i];
        	        }
        	        df->erasures = 0;
        	        df->derrors[sid] = DECODE_RS2(df->rs2, df->block2, df->derrlocs2, df->erasures);
        	}
		if(df->derrors[sid] != -1)
		{
			df->derrors[sid] = get_df_derrors(df, article, bid, sid, 1);
			if(df->derrors[sid] > (df->n - df->k) / 2)
			{
				df->derrors[sid] = -1;
			}
		}
		if(df->derrors[sid] != -1)
		{
			return 0;
		}
		else
		{
			if(direct_else(df, sid) == 1)
			{
				return 1;
			}
		}
		df->finish = clock();
		df->time = (df->finish - df->start) / CLOCKS_PER_SEC;
		if(df->time > df->timeout)
		{
			return 1;
		}
	}
}

int change_hp_else(struct DECODEFILE *df, int sid)
{
	int i;
	if(zuhe(df->changec + sid * df->n, df->changen[sid], df->changer[sid]) == 1)
	{
		df->changer[sid] = df->changer[sid] + 1;
		if(df->changer[sid] <= df->changen[sid] && df->changer[sid] <= df->maxchange)
		{
			for(i = 0; i < df->changer[sid]; i++)
			{
				df->changec[sid * df->n + i] = 1;
			}
			for(i = df->changer[sid]; i < df->changen[sid]; i++)
			{
				df->changec[sid * df->n + i] = 0;
			}
		}
		else
		{
			df->sr[sid] = df->sr[sid] + df->raise;
                        if(df->raise != 0 && df->sr[sid] <= df->upper)
                        {
                                init_change_hp(df, sid);
                        }
                        else
                        {
                                init_direct(df, sid);
                        }
			return 1;
		}
	}
	return 0;
}

int change_hp(struct DECODEFILE *df, struct ARTICLE *article, int bid, int sid)
{
	int i, j, k;
	while(1)
	{
		if(df->n <= 255)
        	{
        	        for(i = 0; i < df->n; i++)
        	        {
        	                df->block1[i] = df->datvalue[i];
        	        }
			j = 0;
			for(i = 0; i < df->changen[sid]; i++)
                        {
                                if(df->changec[sid * df->n + i] == 1)
                                {
                                        df->block1[df->hpkey[i]] = df->hpvalue[i];
					j++;
                                }
                        }
			df->erasures = 0;
        	        df->derrors[sid] = DECODE_RS1(df->rs1, df->block1, df->derrlocs1, df->erasures);
        	}
        	else
        	{
        	        for(i = 0; i < df->n; i++)
        	        {
        	                df->block2[i] = df->datvalue[i];
        	        }
			j = 0;
			for(i = 0; i < df->changen[sid]; i++)
                        {
                                if(df->changec[sid * df->n + i] == 1)
                                {
                                        df->block2[df->hpkey[i]] = df->hpvalue[i];
					j++;
                                }
                        }
			df->erasures = 0;
        	        df->derrors[sid] = DECODE_RS2(df->rs2, df->block2, df->derrlocs2, df->erasures);
        	}
		k = 3;
		if(df->derrors[sid] != -1)
		{
			df->derrors[sid] = get_df_derrors(df, article, bid, sid, k);
			if(df->derrors[sid] - df->changer[sid] != (df->n - df->k) / 2)
			{
				df->derrors[sid] = -1;
			}
			else
			{
				for(i = 0; i < df->changen[sid]; i++)
				{
					if(df->changec[sid * df->n + i] == 1 && df->n <= 255)
					{
						if(df->block1[df->hpkey[i]] != df->hpvalue[i])
						{
							df->derrors[sid] = -1;
						}
					}
					if(df->changec[sid * df->n + i] == 1 && df->n > 255)
                                        {
                                                if(df->block2[df->hpkey[i]] != df->hpvalue[i])
                                                {
                                                        df->derrors[sid] = -1;
                                                }
                                        }
				}
			}
		}
		if(df->derrors[sid] == -1 && j >= 2 && j <= df->maxdelete && j % 2 == 0)
                {
			k = 2;
                        if(df->n <= 255)
                        {
                                for(i = 0; i < df->n; i++)
                                {
                                        df->block1[i] = df->datvalue[i];
                                }
                                df->erasures = 0;
                                for(i = 0; i < df->changen[sid]; i++)
                                {
                                        if(df->changec[sid * df->n + i] == 1)
                                        {
                                                df->derrlocs1[df->erasures] = df->hpkey[i];
                                                df->erasures++;
                                        }
                                }
                                df->derrors[sid] = DECODE_RS1(df->rs1, df->block1, df->derrlocs1, df->erasures);
                        }
                        else
                        {
                                for(i = 0; i < df->n; i++)
                                {
                                        df->block2[i] = df->datvalue[i];
                                }
                                df->erasures = 0;
                                for(i = 0; i < df->changen[sid]; i++)
                                {
                                        if(df->changec[sid * df->n + i] == 1)
                                        {
                                                df->derrlocs2[df->erasures] = df->hpkey[i];
                                                df->erasures++;
                                        }
                                }
                                df->derrors[sid] = DECODE_RS2(df->rs2, df->block2, df->derrlocs2, df->erasures);
			}
		}
		if(k == 2 && df->derrors[sid] != -1)
                {
                        df->derrors[sid] = get_df_derrors(df, article, bid, sid, k);
                        if(df->derrors[sid] - df->changer[sid] / 2 != (df->n - df->k) / 2)
                        {
                                df->derrors[sid] = -1;
                        }
                        else
                        {
                                for(i = 0; i < df->changen[sid]; i++)
                                {
                                        if(df->changec[sid * df->n + i] == 1 && df->n <= 255)
                                        {
                                                if(df->block1[df->hpkey[i]] == df->datvalue[df->hpkey[i]])
                                                {
                                                        df->derrors[sid] = -1;
                                                }
                                        }
                                        if(df->changec[sid * df->n + i] == 1 && df->n > 255)
                                        {
                                                if(df->block2[df->hpkey[i]] == df->datvalue[df->hpkey[i]])
                                                {
                                                        df->derrors[sid] = -1;
                                                }
                                        }
                                }
                        }
                }
		if(df->derrors[sid] != -1)
		{
			change_hp_else(df, sid);
			return 0;
		}
		else
		{
			if(change_hp_else(df, sid) == 1)
			{
				return 1;
			}
		}
		df->finish = clock();
                df->time = (df->finish - df->start) / CLOCKS_PER_SEC;
                if(df->time > df->timeout)
                {
                        return 1;
                }
	}
}

int delete_qlt_else(struct DECODEFILE *df, int sid)
{
	int i;
	if(zuhe(df->deletec + sid * df->n, df->deleten[sid], df->deleter[sid]) == 1)
	{
		df->deleter[sid] = df->deleter[sid] + 2;
		if(df->deleter[sid] <= df->deleten[sid] && df->deleter[sid] <= df->maxdelete)
		{
			for(i = 0; i < df->deleter[sid]; i++)
			{
				df->deletec[sid * df->n + i] = 1;
			}
			for(i = df->deleter[sid]; i < df->deleten[sid]; i++)
			{
				df->deletec[sid * df->n + i] = 0;
			}
		}
		else
		{
			df->sr[sid] = df->sr[sid] + df->raise;
			if(df->raise != 0 && df->sr[sid] <= df->upper)
			{
				init_delete_qlt(df, sid);
			}
			else
			{
                        	init_direct(df, sid);
			}
                        return 1;
		}
	}
        return 0;
}

int delete_qlt(struct DECODEFILE *df, struct ARTICLE *article, int bid, int sid)
{
        int i;
        while(1)
        {
                if(df->n <= 255)
                {
                        for(i = 0; i < df->n; i++)
                        {
                                df->block1[i] = df->datvalue[i];
                        }
                        df->erasures = 0;
                        for(i = 0; i < df->deleten[sid]; i++)
                        {
                                if(df->deletec[sid * df->n + i] == 1)
                                {
                                        df->derrlocs1[df->erasures] = df->qltkey[i];
                                        df->erasures++;
                                }
                        }
                        df->derrors[sid] = DECODE_RS1(df->rs1, df->block1, df->derrlocs1, df->erasures);
                }
                else
                {
                        for(i = 0; i < df->n; i++)
                        {
                                df->block2[i] = df->datvalue[i];
                        }
                        df->erasures = 0;
                        for(i = 0; i < df->deleten[sid]; i++)
                        {
                                if(df->deletec[sid * df->n + i] == 1)
                                {
                                        df->derrlocs2[df->erasures] = df->qltkey[i];
                                        df->erasures++;
                                }
                        }
                        df->derrors[sid] = DECODE_RS2(df->rs2, df->block2, df->derrlocs2, df->erasures);
                }
		if(df->derrors[sid] != -1)
		{
			df->derrors[sid] = get_df_derrors(df, article, bid, sid, 2);
			if(df->derrors[sid] - df->deleter[sid] / 2 != (df->n - df->k) / 2)
			{
				df->derrors[sid] = -1;
			}
			else
			{
				for(i = 0; i < df->deleten[sid]; i++)
				{
					if(df->deletec[sid * df->n + i] == 1 && df->n <= 255)
					{
						if(df->block1[df->qltkey[i]] == df->datvalue[df->qltkey[i]])
						{
							df->derrors[sid] = -1;
						}
					}
					if(df->deletec[sid * df->n + i] == 1 && df->n > 255)
                                        {
                                                if(df->block2[df->qltkey[i]] == df->datvalue[df->qltkey[i]])
                                                {
                                                        df->derrors[sid] = -1;
                                                }
                                        }
				}
			}
		}
                if(df->derrors[sid] != -1)
                {
			delete_qlt_else(df, sid);
			return 0;
                }
                else
                {
                        if(delete_qlt_else(df, sid) == 1)
                        {
                                return 1;
                        }
                }
		df->finish = clock();
                df->time = (df->finish - df->start) / CLOCKS_PER_SEC;
                if(df->time > df->timeout)
                {
                        return 1;
                }
        }
}

int decode_rs(struct DECODEFILE *df, struct ARTICLE *article, int bid, int sid)
{
	if(df->level[sid] == 1)
	{
		get_df_dat(df, bid, sid);
		if(df->ver >= 1 && df->mode == 2)
		{
			df->sr[sid] = df->lower;
			get_df_qlt(df, bid, sid);
		}
		if(df->ver >= 1 && df->mode == 3)
		{
			df->sr[sid] = df->lower;
			get_df_hp(df, bid, sid);
		}
		if(direct(df, article, bid, sid) == 0)
		{
			return 0;
		}
		else
		{
			if(df->time > df->timeout)
			{
				return 1;
			}
		}
	}
	while(df->level[sid] == 2)
	{
		get_df_dat(df, bid, sid);
		get_df_qlt(df, bid, sid);
		if(delete_qlt(df, article, bid, sid) == 0)
		{
			return 0;
		}
		else
		{
			if(df->time > df->timeout)
			{
				return 1;
			}
		}
	}
	while(df->level[sid] == 3)
        {
                get_df_dat(df, bid, sid);
                get_df_hp(df, bid, sid);
                if(change_hp(df, article, bid, sid) == 0)
                {
                        return 0;
                }
                else
                {
                        if(df->time > df->timeout)
                        {
                                return 1;
                        }
                }
        }
	return 1;
}

int crc64_check(struct DECODEFILE *df, int bid)
{
	int i, k1, k2;
	int size1, size2, size3;
	u8i crc1, crc2;
	size1 = df->k * df->tlen * 2;
	size2 = size1;
	size3 = size2 / 2;
	u1iseq_to_u1iseq(2, df->tseq + bid * df->n * df->tlen, 1, df->buffer1, df->k * df->tlen);
	i = 0;
	for(k2 = 0; k2 < df->tlen * 2; k2++)
	{
		for(k1 = 0; k1 < df->k; k1++)
		{
			df->buffer2[i] = df->buffer1[k1 * (df->tlen * 2) + k2];
			i++;
		}
	}
	u1iseq_to_u1iseq(1, df->buffer2, 2, df->buffer3, size3);
	crc1 = crc64(0, df->buffer3, size3 - 32);
	u1iseq_to_u8iseq(2, df->buffer3 + size3 - 32, 64, &crc2, 1);
	if(crc1 != crc2)
	{
		return 1;
	}
	return 0;
}

int decode_ref(struct DECODEFILE *df, struct ARTICLE *article, int bid)
{
	int sid, i, j, k, flag;
	flag = 0;
	for(sid = 0; sid < df->nuss; sid++)
	{
		for(i = 0; i < df->n; i++)
                {
                        j = (bid * df->n + i) * df->tlen + (sid * df->hrm);
			for(k = 0; k < df->hrm; k++)
			{
				if(df->tseq[j + k] != df->Rseq[j + k])
				{
					flag = 1;
					break;
				}
			}
			if(flag == 1)
			{
				break;
			}
		}
		if(flag == 1)
		{
			break;
		}
	}
	get_df_dat(df, bid, sid);
	if(df->n <= 255)
        {
		for(i = 0; i < df->n; i++)
		{
			j = (bid * df->n + i) * df->Rlen + (sid * df->hrm);
			u1iseq_to_u1iseq(2, df->Rseq + j, df->hrm * 2, df->block1 + i, 1);
		}
	}
	else
	{
		for(i = 0; i < df->n; i++)
                {
                        j = (bid * df->n + i) * df->Rlen + (sid * df->hrm);
			u1iseq_to_intseq(2, df->Rseq + j, df->hrm * 2, df->block2 + i, 1);
                }
	}
	if(df->ver >= 1)
        {
                //fprintf(stderr, "decode_ref_and_");
        }
	df->derrors[sid] = get_df_derrors(df, article, bid, sid, 0);
	for(i = 0; i <= sid; i++)
	{
		df->derrors[i] = 0;
	}
	df->level[sid] = 1;
	return sid;
}

int decode_block(struct DECODEFILE *df, struct ARTICLE *article, int bid)
{
	df->start = clock();
	df->time = 0;
	df->number = 0;
	df->diff = 0;
	df->useref = 0;
	int i, j;
	for(i = bid * df->n; i < (bid + 1) * df->n; i++)
        {
                for(j = 0; j < df->tlen; j++)
                {
                        df->tseq[i * df->tlen + j] = 0;
                }
        }
	for(i = 0; i < df->nuss; i++)
	{
		df->level[i] = 1;
	}
	for(i = 0; i < df->nuss; i++)
	{
		if(decode_rs(df, article, bid, i) != 0)
                {
			if(df->time > df->timeout || df->number > df->allowed)
			{
				if(df->hasref == 1)
				{
					i = decode_ref(df, article, bid);
					for(j = i + 1; j < df->nuss; j++)
					{
						df->level[j] = 1;
					}
					df->start = clock();
					df->time = 0;
					df->number = 0;
					df->useref = 1;
				}
				else
				{
					return 1;
				}
			}
			else
			{
				for(j = 0; j < i - 1; j++)
				{
					if(df->derrors[j] > (df->n - df->k) / 2 && df->derrors[j + 1] > (df->n - df->k) / 2)
					{
						break;
					}
				}
				if(df->jump == 2 && j < i - 1)
				{
					i = j - 1;
					if(df->ver >= 1)
					{
						//fprintf(stderr, "rs error then recall rs:%d\n", j);
					}
					df->number++;
					if(df->last != j)
					{
						df->last = j;
						df->diff++;
					}
					for(j = j + 1; j < df->nuss; j++)
					{
						df->level[j] = 1;
					}
				}
				else
				{
					for(j = i - 1; j >= 0; j--)
					{
						if(df->derrors[j] > (df->n - df->k) / 2)
						{
							break;
						}
					}
					if(j == -1)
					{
						if(df->hasref == 1)
						{
							i = decode_ref(df, article, bid);
							for(j = i + 1; j < df->nuss; j++)
                                        		{
                                        		        df->level[j] = 1;
                                        		}
							df->start = clock();
							df->time = 0;
							df->number = 0;
							df->useref = 1;
						}
						else
						{
							//fprintf(stderr, "rs error\n");
							return 1;
						}
					}
					else
					{
						i = j - 1;
						if(df->ver >= 1)
						{
							//fprintf(stderr, "rs error then recall rs:%d\n", j);
						}
						df->number++;
						if(df->last != j)
						{
							df->last = j;
							df->diff++;
						}
					}
				}
			}
                }
		if(i == df->nuss - 1)
		{
			if(crc64_check(df, bid) != 0)
        		{
	                        for(j = 0; j < i; j++)
				{
					if(df->derrors[j] > (df->n - df->k) / 2 && df->derrors[j + 1] > (df->n - df->k) / 2)
                                	{
                                	        break;
                                	}
				}
				if(df->jump == 2 && j < i)
				{
					i = j - 1;
					if(df->ver >= 1)
					{
						fprintf(stderr, "crc64 error then recall rs:%d\n", j);
					}
					df->number++;
					if(df->last != j)
					{
						df->last = j;
						df->diff++;
					}
					article->crc64_times++;
					for(j = j + 1; j < df->nuss; j++)
                                        {
                                                df->level[j] = 1;
                                        }
				}
				else
				{
					for(j = i; j >= 0; j--)
                                	{
                                	        if(df->derrors[j] > (df->n - df->k) / 2)
                                	        {
                                	                break;
                                	        }
                                	}
					if(j == -1)
                                	{
						if(df->hasref == 1)
						{
							i = decode_ref(df, article, bid);
							for(j = i + 1; j < df->nuss; j++)
                                        		{
                                        		        df->level[j] = 1;
                                        		}
							df->start = clock();
		                                        df->time = 0;
							df->number = 0;
							df->useref = 1;
						}
						else
						{
                                	        	//fprintf(stderr, "crc64 error\n");
                                	        	return 1;
						}
                                	}
					else
					{
						i = j - 1;
						if(df->ver >= 1)
						{
							fprintf(stderr, "crc64 error then recall rs:%d\n", j);
						}
						df->number++;
						if(df->last != j)
						{
							df->last = j;
							df->diff++;
						}
						article->crc64_times++;
					}
				}
        		}
		}
	}
	if(df->useref == 1)
	{
		return 1;
	}
	else
	{
		return 0;
	}
}
