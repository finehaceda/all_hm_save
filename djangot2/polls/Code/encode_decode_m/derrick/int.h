/* Include file to configure the RS codec for integer symbols
 *
 * Copyright 2002, Phil Karn, KA9Q
 * May be used under the terms of the GNU General Public License (GPL)
 */
#define DTYPE2 int

/* Reed-Solomon codec control block */
struct rs2 {
  int mm;              /* Bits per symbol */
  int nn;              /* Symbols per block (= (1<<mm)-1) */
  DTYPE2 *alpha_to;     /* log lookup table */
  DTYPE2 *index_of;     /* Antilog lookup table */
  DTYPE2 *genpoly;      /* Generator polynomial */
  int nroots;     /* Number of generator roots = number of parity symbols */
  int fcr;        /* Firs2t consecutive root, index form */
  int prim;       /* Primitive element, index form */
  int iprim;      /* prim-th root of 1, index form */
  int pad;        /* Padding bytes in shortened block */
};

static inline int modnn2(struct rs2 *rs2,int x){
  while (x >= rs2->nn) {
    x -= rs2->nn;
    x = (x >> rs2->mm) + (x & rs2->nn);
  }
  return x;
}
#define MODNN22(x) modnn2(rs2,x)

#define MM2 (rs2->mm)
#define NN2 (rs2->nn)
#define ALPHA_TO2 (rs2->alpha_to) 
#define INDEX_OF2 (rs2->index_of)
#define GENPOLY2 (rs2->genpoly)
#define NROOTS2 (rs2->nroots)
#define FCR2 (rs2->fcr)
#define PRIM2 (rs2->prim)
#define IPRIM22 (rs2->iprim)
#define PAD2 (rs2->pad)
#define A02 (NN2)

#define ENCODE_RS2 encode_rs_int
#define DECODE_RS2 decode_rs_int
#define INIT_RS2 init_rs_int
#define FREE_RS2 free_rs2_int

void ENCODE_RS2(void *p,DTYPE2 *data,DTYPE2 *parity);
int DECODE_RS2(void *p,DTYPE2 *data,int *eras_pos,int no_eras);
void *INIT_RS2(int symsize,int gfpoly,int fcr,
		   int prim,int nroots,int pad);
void FREE_RS2(void *p);




