/* The engine's compact layouts and state (shared by model.c, which builds / caches them, and
   step.c, which runs them). See docs/engine.md for the design and the exactness argument. */
#ifndef QD_INTERNAL_H
#define QD_INTERNAL_H

#include <stdint.h>
#include <string.h>

#include "qd_engine.h"

#define QD_LAYOUT_VERSION 5
#define QD_MAX_KS 8       /* conv kernel */
#define QD_MAX_THREADS 16

/* Sparse weights are 32-bit entries: the bf16 weight in the high half (so `e & 0xFFFF0000` is
   the weight as a float) and a 16-bit index in the low half (the output row of a CSC column,
   or the input column of a CSR row). */
#define W_OF(e) u2f((e) & 0xFFFF0000u)
#define I_OF(e) ((e) & 0xFFFFu)

static inline float u2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static inline uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
/* round to the nearest bf16, ties to even (as float). With FEAT_BF16, BFCVT does it in one
   instruction: it equals the integer form on every finite float (checked exhaustively on an
   M5 Max, all 2^32 bit patterns; docs/engine.md), FPCR's default rounding, no flush to zero. */
#if defined(__ARM_FEATURE_BF16)
#include <arm_neon.h>
static inline float bf16r(float x) { return vcvtah_f32_bf16(vcvth_bf16_f32(x)); }
#else
static inline float bf16r(float x) {
    uint32_t u = f2u(x);
    return u2f((u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u);
}
#endif

/* An MLP unit whose up row and down column have one entry each (the circuit's units: a gate
   over a few inputs times one input, added to one residual dim). */
typedef struct {
    uint16_t iu, id; /* up input dim, down output dim */
    float wu, wd;
} unit1_t;

/* A blocked-CSC projection's cached prefix. The residual dims below pre are constant over
   a run (the ballast), so the outputs' partial sums over them, the first terms of each output's
   sum in increasing input order, are the same at every step: base. acc holds base between
   steps; a step adds the other inputs' terms (marking the blocks it touches), reads, and puts
   base back in the touched blocks. The prefix's (dim, value) pairs are compared every step;
   on a mismatch base is recomputed. */
typedef struct {
    float *base, *acc;       /* [nb * 16] */
    uint64_t *live0;         /* [nbw] blocks with an output that is live at base */
    int32_t n_live0;         /* (their count) */
    int32_t nb, nbw;         /* blocks, bitmap words */
    int32_t pre;             /* the residual dims [0, pre) are constant at this sublayer */
    int32_t ok, pn;          /* base is valid for the pn cached pairs */
    int32_t *pcol;           /* [pre] */
    uint32_t *pval;          /* [pre] (bits) */
} prefix_t;

/* Interval bounds on groups of eager channels (step.c, docs/engine.md). A group is 8 blocks
   (128 channels) of one region: the single-tap blocks [0, rb) or the ring blocks [rb, nb). A
   channel's projection is base + the terms of the inputs beyond the prefix. In each input row
   a group's channels fall into three weight classes (w > 0, w < 0, w = 0); base_j is written
   as r_j plus one offset per row, the offset of channel j's class there (fitted whenever base
   changes; any offsets give a valid bound, and the circuit's address detectors, whose biases
   are sums over their pattern bits, are fitted exactly), so per row and group the class
   offsets plus the class's extreme weights times the input bound every channel's share, and
   one pass over the nonzero inputs bounds every group. A group whose conv pre-activations are
   all provably <= -89 (silu exactly -0) is skipped: no projection, no conv; a ring group's
   projection at this position is deferred (pend) and computed only if a later position whose
   conv reads it cannot prove the group dead. */
typedef struct {
    int32_t on, ng, ng0, ngp;   /* groups: [0, ng0) single-tap, [ng0, ng) ring; then the regions */
    int32_t rq;                 /* the two regions as groups: rq (single-tap) and rq + 1 (ring);
                                   a dead region's groups need no test of their own */
    int32_t rows, ngw;          /* table rows; words of a group bitmap */
    int32_t *colid;             /* [H] an input dim's table row, -1: no weight into qk (up to
                                   H rows: wider than int16 past 32,768 dims) */
    uint32_t *gptr;             /* [rows][ng + 1] where each group's blocks start in the row's
                                   column (blocked CSC indices) */
    int8_t *alias;              /* [rows][3][ngp] the class standing in for each class (itself
                                   when present in the group, else a present one) */
    float *cwx, *cwn;           /* [rows][3][ngp] per class: its extreme weights (of the alias) */
    float *wabs;                /* [rows][ngp] max |w| */
    float *ghi, *glo;           /* [rows][3][ngp] class offsets (alias) minus their max / min
                                   (qk_fresh) */
    float *tmin, *tmax;         /* [KS][ngp] conv tap ranges per group */
    float *blo, *bhi, *babs;    /* [ngp] (qk_fresh) the bounds with every input 0; max |base| + the
                                   offsets' magnitude (rounding error) */
    float *slo, *shi;           /* [KS][ngp] the projection ranges of each ring slot */
    float *lo, *hi, *mag;       /* [ngp] scratch */
    uint64_t *gdead;            /* [ngw] the groups proved dead this step */
    uint8_t *rr_ok;             /* [KS] the ring groups' ranges at that slot are in slo / shi
                                   (else only the ring region's: qk_bound) */
    uint64_t *pend;             /* [KS][ngw] ring groups whose projection at that slot is
                                   deferred */
    int32_t nlive;              /* blocks not proved dead this step */
} qkbound_t;

/* One value head that can reach the output. */
typedef struct {
    int32_t h, kh;         /* value head and the key head it reads (h / (nv / nk)) */
    int32_t a_row, b_row;  /* rows of the layer's pool (-1: an all-zero row) */
    float dt_bias, A_log;
    int32_t v0, nvc;       /* its lazy value channels: [v0, v0 + nvc) */
    int32_t o0, no;        /* its output dims (out_proj column and z row both nonzero) */
} vhead_t;

/* A blocked CSC matrix over the residual: per input dim, the 16-row blocks of outputs it
   touches, each with 16 bf16 weights (the zeros inside a block add exactly nothing). The
   circuit's columns are dense over runs of rows, so blocks are nearly full. */
typedef struct {
    uint32_t *ptr; /* [H + 1] */
    uint16_t *blk; /* [nblk] block b: rows 16 b .. 16 b + 15 */
    uint16_t *w;   /* [nblk][16] */
    /* derived at open: thread t scatters the blocks [split[t][c], split[t + 1][c]) of column c
       (split[0] = ptr, split[nth] = ptr + 1); its blocks are [bsplit[t], bsplit[t + 1]) */
    uint32_t *split[QD_MAX_THREADS + 1];
    int32_t bsplit[QD_MAX_THREADS + 1];
    int32_t nact;  /* threads [0, nact) own blocks */
} bcsc_t;

typedef struct {
    /* --- token mixer --------------------------------------------------------------------- */
    int32_t attn;            /* some value head reaches the output */
    int32_t nqk, nqk_pad;    /* eager channels: the queries and keys of the used key heads */
    int32_t nqk1;            /* channels [0, nqk1) read only the current tap (no ring) */
    int32_t nact;            /* active value heads */
    int32_t nvch, nod, npool;
    int32_t na_cols, na_out; /* residual dims the mixer reads / writes */
    float *g_in;             /* [Hp] 1 + w */
    bcsc_t qk;               /* the residual -> eager channels */
    float *qk_tap;           /* [KS][nqk_pad], oldest tap first */
    uint32_t *qk_id;         /* [nqk_pad] kind (0 q, 1 k) << 31 | key head << 16 | dim */
    int32_t *a_cols, *a_out;
    vhead_t *vh;             /* [nact], increasing h */
    int32_t *vch_dim, *vch_row; /* [nvch] value dim, pool row */
    float *vch_tap;          /* [nvch][KS] */
    int32_t *od_dim, *od_zrow; /* [nod] */
    uint32_t *od_ptr, *od_ent; /* [nod + 1] CSC: out_proj column -> residual dims */
    float *norm_w;           /* [dv] */
    uint32_t *pool_ptr, *pool_ent; /* CSR rows: value channels, z, a, b */
    /* --- MLP ----------------------------------------------------------------------------- */
    int32_t mlp, nu, nu_pad;
    int32_t nm_cols, nm_out;
    float *g_post;           /* [Hp] */
    bcsc_t gate;             /* the residual -> units (all of them: liveness needs the gate) */
    uint32_t *up_ptr, *up_ent; /* per unit, CSR (only live units read it) */
    int32_t *m_cols, *m_out;
    uint32_t *dn_ptr, *dn_ent; /* per unit: residual dims */
    /* --- derived at open (not cached) ---------------------------------------------------- */
    uint64_t *a_colb, *m_colb; /* [Hp / 64] bitmaps of a_cols / m_cols */
    float *decay_c;          /* [nact] the decay of heads with a_row < 0 (a constant) */
    int32_t *kh_v;           /* [nk + 1] key head kh's value heads: vh[kh_v[kh], kh_v[kh + 1]) */
    uint8_t *kh_decay;       /* [nk] some value head of the key head decays (or may) */
    float *vbound;           /* [nact] |v| <= vbound max |xn| (+ rounding): per value channel
                                the sum of |taps| times the sum of its row's |w|, the largest */
    unit1_t *u1;             /* [nu] when every unit has one down entry, else NULL (the up
                                fields: the first up entry) */
    uint64_t *u1_blk;        /* [nbw] (with u1) the blocks whose units all have one up entry */
    uint64_t *u1_uni;        /* [nbw] (with u1) of those, the whole blocks whose 16 units read
                                one up input */
    int32_t *u1_iu;          /* [nu_pad] (with u1) the up input dims of one-entry units, as
                                arrays (0 elsewhere) */
    float *u1_wu;            /* [nu_pad] (with u1) their up weights (0 elsewhere) */
    /* nonzero |w| ranges (lo, hi) of the weights each kind of value multiplies: the attention's
       normed input (qk, pool), its gated output (out_proj), the MLP's normed input (gate, up),
       its activations (down) */
    float wr[4][2];
    prefix_t gp;             /* the gate's cached prefix */
    prefix_t qp;             /* the eager channels' cached prefix (live0: at every tap) */
    uint8_t *pool_pre;       /* [npool] the pool row reads only dims below qp.pre */
    float *pool_c;           /* [npool] those rows' dots for the cached prefix */
    uint64_t *ring_tb;       /* [KS][qp.nbw] per ring slot: the blocks whose projections there
                                are not bf16(base) */
    qkbound_t qb;            /* the eager channels' group bounds */
    /* the z rows (gates) of the output dims, deduplicated (a RAM layer's 4,096 share 4): each
       distinct row's silu(bf16(z)) is computed once per step (zu_pos: the step it holds) */
    int32_t nzu;
    int32_t *od_zu;          /* [nod] the distinct row of each output dim's z */
    int32_t *zu_row;         /* [nzu] its pool row */
    float *zu_val;           /* [nzu] */
    int64_t *zu_pos;         /* [nzu] */
    int32_t *vz_ptr, *vz_id; /* [nact + 1], the distinct z rows of each value head */
    int32_t rb;              /* the first block with ring channels */
    /* --- state --------------------------------------------------------------------------- */
    float *S;                /* [nv][dk][dv] */
    uint64_t *rows_nz;       /* [nv][dk / 64]: rows that may be nonzero */
    float *qk_hist;          /* [KS][nqk_pad] the eager channels' projections, a ring */
    float *xn_ring;          /* [KS][Hp] the normed inputs, for the lazy value channels */
    int32_t *ring_nz;        /* [KS][Hp] the nonzero dims of each ring slot */
    int32_t ring_n[QD_MAX_KS];
    float xn_hi[QD_MAX_KS];  /* the largest |xn| of each slot */
} layer_t;

typedef struct qd_pool qd_pool;

/* a nonzero output of a parallel stage: an eager channel's value, a live unit's activation
   (thread-local lists, concatenated in thread order = channel / unit order) */
typedef struct {
    uint32_t id;
    float v;
} item_t;

struct qd_engine {
    /* shapes */
    int32_t H, Hp, nl, V, nk, nv, dk, dv, KS, r;
    float eps, qscale;
    layer_t *L;
    int32_t last_needed;     /* the last layer with a token mixer: forced steps stop there */
    /* embedding: the nonzero columns of every row */
    int32_t necol;
    int32_t *ecol;
    uint16_t *etab;          /* [V][necol] bf16 */
    /* final norm and tied head */
    float *gain;             /* [Hp] */
    int32_t ncode;
    int32_t *code;           /* dims with a nonzero final gain */
    uint16_t *ecode;         /* [V][ncode] bf16: the head restricted to the code dims */
    int32_t sign_ok;
    int32_t *sign_tab;       /* [2^ncode] the token with each sign code, -1 for none */
    int32_t pre;             /* the first MLP's prefix bound (info) */
    /* allocations (freed at close) */
    void **allocs;
    int32_t nallocs, capallocs;
    /* run state */
    int64_t pos;
    float *x, *dx;           /* [Hp] the residual and a sublayer's output */
    uint64_t *xb;            /* [Hp / 64] the nonzero dims of x */
    uint64_t *dxb;           /* [Hp / 64] the dims of dx a sublayer touched */
    float inv;               /* 1 / rms(x), valid while inv_ok (x unchanged since) */
    int32_t inv_ok;
    float *xm;               /* [Hp] the MLP's normed input */
    int32_t *mnz;            /* [Hp] the MLP's nonzero normed inputs */
    int32_t mnnz;
    float *y;                /* [max(dv, ncode) + pad] */
    /* the nonzero queries / keys of the current layer, per key head */
    int32_t *qn, *kn;        /* [nk] counts */
    int32_t *qd, *kd;        /* [nk][dk] dims */
    float *qv, *kv;          /* [nk][dk] values */
    /* options */
    int32_t flags;           /* QD_OPEN_* */
    int32_t exact_head, profile;
    float *capture;
    /* threads */
    int32_t nth;
    qd_pool *pool;
    item_t *items[QD_MAX_THREADS]; /* per thread: the nonzero outputs of its share */
    int32_t nitems[QD_MAX_THREADS];
    /* the MLP's activations: per unit (read where amask has the unit's bit), per block the
       mask of its nonzero ones, per thread the blocks with one, in increasing order */
    float *act;
    uint16_t *amask;
    int32_t *mblk[QD_MAX_THREADS];
    int32_t nmblk[QD_MAX_THREADS];
    uint64_t *tb[QD_MAX_THREADS];  /* per thread: the blocks its scatter touched */
    uint64_t tslow[QD_MAX_THREADS], tmac[QD_MAX_THREADS], tblk[QD_MAX_THREADS];
    float tlo[QD_MAX_THREADS], thi[QD_MAX_THREADS];
    /* stats / info */
    qd_engine_stats st;
    qd_engine_info info;
};

/* model.c */
void *qd_alloc(qd_engine *e, size_t n); /* zeroed, 64-byte aligned, freed at close */

/* step.c */
void qd_derive(qd_engine *e); /* the derived fields above, for e->nth threads */
void qd_state_alloc(qd_engine *e);
void qd_state_reset(qd_engine *e);
int qd_pool_start(qd_engine *e);
void qd_pool_stop(qd_engine *e);

#endif
