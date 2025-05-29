/* Include file to configure the RS codec for character symbols
 *
 * Copyright 2002, Phil Karn, KA9Q
 * May be used under the terms of the GNU General Public License (GPL)
 */
#define DTYPE1 unsigned char

/* Reed-Solomon codec control block */
struct rs1 {
  int mm;              /* Bits per symbol */
  int nn;              /* Symbols per block (= (1<<mm)-1) */
  DTYPE1 *alpha_to;     /* log lookup table */
  DTYPE1 *index_of;     /* Antilog lookup table */
  DTYPE1 *genpoly;      /* Generator polynomial */
  int nroots;     /* Number of generator roots = number of parity symbols */
  int fcr;        /* Firs1t consecutive root, index form */
  int prim;       /* Primitive element, index form */
  int iprim;      /* prim-th root of 1, index form */
  int pad;        /* Padding bytes in shortened block */
};

static inline int modnn1(struct rs1 *rs1,int x){
  while (x >= rs1->nn) {
    x -= rs1->nn;
    x = (x >> rs1->mm) + (x & rs1->nn);
  }
  return x;
}
#define MODNN11(x) modnn1(rs1,x)

#define MM1 (rs1->mm)
#define NN1 (rs1->nn)
#define ALPHA_TO1 (rs1->alpha_to) 
#define INDEX_OF1 (rs1->index_of)
#define GENPOLY1 (rs1->genpoly)
#define NROOTS1 (rs1->nroots)
#define FCR1 (rs1->fcr)
#define PRIM1 (rs1->prim)
#define IPRIM11 (rs1->iprim)
#define PAD1 (rs1->pad)
#define A01 (NN1)

#define ENCODE_RS1 encode_rs_char
#define DECODE_RS1 decode_rs_char
#define INIT_RS1 init_rs_char
#define FREE_RS1 free_rs1_char

void ENCODE_RS1(void *p,DTYPE1 *data,DTYPE1 *parity);
int DECODE_RS1(void *p,DTYPE1 *data,int *eras_pos,int no_eras);
void *INIT_RS1(int symsize,int gfpoly,int fcr,
		    int prim,int nroots,int pad);
void FREE_RS1(void *p);





