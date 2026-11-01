/* One step of the forward pass over the compact layouts (internal.h), and the worker pool.

   Everything is computed in the precision and order the reference defines (qwendoom/engine.py
   RefEngine, itself what Qwen3_5ForCausalLM computes in bf16): fp32 arithmetic, bf16 rounding
   where the bf16 model rounds, each projection output summed in increasing input order, norms
   in the 16-lane order of sum16. Only work whose result is exactly zero (or exactly known) is
   skipped: zero inputs and weights, units whose silu is exactly 0, heads with no key or query,
   residual dims that are 0, partial sums over constant inputs that are the same as at the last
   step (prefix_t), channel groups whose bound proves every pre-activation <= -89 (qkbound_t;
   a dead ring group's projection is deferred until a position that reads it is not dead),
   value heads whose gates are all 0, key writes with beta = 0 (write_noop). Repeated values
   are computed once: 1 / rms(x) while x is unchanged, a z row shared by output dims.
   docs/engine.md has the whole argument.

   Fused multiply-adds are used only for bf16 x bf16 products, which are exact in fp32 while
   they stay in its normal range (16 significant bits): then fma(w, x, s) = s + w * x, the
   reference's multiply-then-add. The range guard checks that per sublayer (stats.range_risk).
   Everything else (conv taps, the delta rule, norms) multiplies and adds separately, as torch
   does; -ffp-contract=off keeps the compiler from fusing them. */
#include <arm_neon.h>
#include <mach/mach_time.h>
#include <math.h>
#include <pthread.h>
#include <pthread/qos.h>
#include <stdatomic.h>
#include <stdlib.h>

#include "internal.h"

/* --- elementwise functions with their exact ranges ------------------------------------------ */

/* silu(x) = x / (1 + exp(-x)) in fp32. For x >= 17, exp(-x) < 2^-24 and 1 + exp(-x) rounds to
   1; for x <= -89, exp(-x) overflows to inf; silu(+-0) = +-0. In those ranges every libm (and
   numpy, and torch) agrees; elsewhere the result may differ in the last bit between libraries,
   so those calls are counted (stats.slow_transcendental; the circuit never makes any). */
static inline float silu_s(float x, uint64_t *slow) {
    if (x >= 17.0f) return x;
    if (x <= -89.0f) return -0.0f;
    if (x == 0.0f) return x;
    ++*slow;
    return x / (1.0f + expf(-x));
}

static inline float sigmoid_s(float x, uint64_t *slow) {
    if (x >= 17.0f) return 1.0f;
    if (x <= -89.0f) return 0.0f;
    if (x == 0.0f) return 0.5f;
    ++*slow;
    return 1.0f / (1.0f + expf(-x));
}

/* exp(g), g = -exp(A_log) * softplus(a + dt_bias) (torch's softplus: threshold 20) */
static float decay_s(float a, float dt_bias, float A_log, uint64_t *slow) {
    const float xa = a + dt_bias;
    /* xa = 0, A = 1: exp(-log1p(1)) = exp(-fl(ln 2)) is 0.03 ulp below 0.5: 0.5 for any exp
       with an error below 0.47 ulp (the circuit's halving heads) */
    if (xa == 0.0f && A_log == 0.0f) return 0.5f;
    float sp;
    if (xa > 20.0f) sp = xa;
    else if (xa < -104.0f) sp = 0.0f; /* exp(xa) underflows to 0, log1p(0) = 0 */
    else {
        ++*slow;
        sp = log1pf(expf(xa));
    }
    float A = 1.0f; /* exp(0) */
    if (A_log != 0.0f) {
        ++*slow;
        A = expf(A_log);
    }
    const float g = -A * sp;
    if (g == 0.0f) return 1.0f;
    if (g <= -104.0f) return 0.0f; /* below half the smallest subnormal */
    ++*slow;
    return expf(g);
}

/* bf16r of 4 lanes (BFCVTN + shift with FEAT_BF16; see bf16r) */
static inline float32x4_t bf16r4(float32x4_t x) {
#if defined(__ARM_FEATURE_BF16)
    return vreinterpretq_f32_u32(vshll_n_u16(vreinterpret_u16_bf16(vcvt_bf16_f32(x)), 16));
#else
    uint32x4_t u = vreinterpretq_u32_f32(x);
    uint32x4_t lsb = vandq_u32(vshrq_n_u32(u, 16), vdupq_n_u32(1));
    u = vaddq_u32(u, vaddq_u32(vdupq_n_u32(0x7FFF), lsb));
    return vreinterpretq_f32_u32(vandq_u32(u, vdupq_n_u32(0xFFFF0000u)));
#endif
}

/* lanes with silu(x) possibly nonzero: x > -89 and x != +-0 */
static inline uint32x4_t live4(float32x4_t x) {
    return vandq_u32(vcgtq_f32(x, vdupq_n_f32(-89.0f)),
                     vtstq_u32(vreinterpretq_u32_f32(x), vdupq_n_u32(0x7FFFFFFFu)));
}

/* --- reductions -------------------------------------------------------------------------------- */

static inline float tree16(const float l[16]) {
    float c[4];
    for (int j = 0; j < 4; j++) c[j] = (l[j] + l[j + 8]) + (l[j + 4] + l[j + 12]);
    return (c[0] + c[1]) + (c[2] + c[3]);
}

/* sum16 (engine.py) of x[i]^2 over n (a multiple of 16) dense values */
static float sumsq16(const float *x, int n) {
    float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0;
    for (int i = 0; i < n; i += 16) {
        float32x4_t v0 = vld1q_f32(x + i), v1 = vld1q_f32(x + i + 4);
        float32x4_t v2 = vld1q_f32(x + i + 8), v3 = vld1q_f32(x + i + 12);
        a0 = vaddq_f32(a0, vmulq_f32(v0, v0));
        a1 = vaddq_f32(a1, vmulq_f32(v1, v1));
        a2 = vaddq_f32(a2, vmulq_f32(v2, v2));
        a3 = vaddq_f32(a3, vmulq_f32(v3, v3));
    }
    float32x4_t c = vaddq_f32(vaddq_f32(a0, a2), vaddq_f32(a1, a3));
    return (vgetq_lane_f32(c, 0) + vgetq_lane_f32(c, 1)) + (vgetq_lane_f32(c, 2) + vgetq_lane_f32(c, 3));
}

/* sum16 of v[i]^2 over sparse entries with increasing dims: the zero dims add nothing (a lane
   sum is >= +0, and +0 + +0 = +0), so each lane adds the same values in the same order. */
static float sumsq16_sparse(const int32_t *dim, const float *v, int n) {
    float l[16] = {0};
    for (int i = 0; i < n; i++) l[dim[i] & 15] += v[i] * v[i];
    return tree16(l);
}

/* 1 / rms(x) over the whole residual, from its nonzero dims (xb); kept until x changes (a
   sublayer often adds nothing) */
static float x_inv_rms(qd_engine *e) {
    if (e->inv_ok) return e->inv;
    /* sum16 over the 64-dim words holding a nonzero dim (as sumsq16; the others add +0) */
    float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0;
    for (int w = 0; w < e->Hp / 64; w++) {
        if (!e->xb[w]) continue;
        const float *x = e->x + 64 * w;
        for (int i = 0; i < 64; i += 16) {
            const float32x4_t v0 = vld1q_f32(x + i), v1 = vld1q_f32(x + i + 4);
            const float32x4_t v2 = vld1q_f32(x + i + 8), v3 = vld1q_f32(x + i + 12);
            a0 = vaddq_f32(a0, vmulq_f32(v0, v0));
            a1 = vaddq_f32(a1, vmulq_f32(v1, v1));
            a2 = vaddq_f32(a2, vmulq_f32(v2, v2));
            a3 = vaddq_f32(a3, vmulq_f32(v3, v3));
        }
    }
    const float32x4_t c = vaddq_f32(vaddq_f32(a0, a2), vaddq_f32(a1, a3));
    const float ss = (vgetq_lane_f32(c, 0) + vgetq_lane_f32(c, 1)) + (vgetq_lane_f32(c, 2) + vgetq_lane_f32(c, 3));
    e->inv = 1.0f / sqrtf(ss / (float)e->H + e->eps);
    e->inv_ok = 1;
    return e->inv;
}

/* --- the range guard ----------------------------------------------------------------------------- */

enum { WR_ATTN_IN, WR_ATTN_OUT, WR_MLP_IN, WR_MLP_OUT };

/* products of the nonzero |w| of range k and |v| in [lo, hi] are exact in fp32 (normal) */
static inline int range_ok(const layer_t *l, int k, float lo, float hi) {
    return hi == 0.0f ||
           ((double)lo * l->wr[k][0] >= 0x1p-126 && (double)hi * l->wr[k][1] < 0x1p127);
}

/* --- sparse projections -------------------------------------------------------------------------- */

/* RMSNorm with the 1 + w gain, then bf16, at the nonzero residual dims the sublayer reads
   (colb): xn[c] for those, their nonzero results' dims to nz (increasing). The other dims of
   xn must already be 0. Returns the count; lo / hi get the range of |xn|. */
static int norm_cols(qd_engine *e, const float *g, const uint64_t *colb, float *xn, int32_t *nz,
                     float *lo, float *hi) {
    const float inv = x_inv_rms(e);
    const float *x = e->x;
    float l = INFINITY, h = 0.0f;
    int n = 0;
    for (int w = 0; w < e->Hp / 64; w++)
        for (uint64_t m = e->xb[w] & colb[w]; m; m &= m - 1) {
            const int c = 64 * w + __builtin_ctzll(m);
            const float v = bf16r(x[c] * inv * g[c]);
            if (v == 0.0f) continue;
            xn[c] = v;
            nz[n++] = c;
            const float a = fabsf(v);
            l = fminf(l, a);
            h = fmaxf(h, a);
        }
    *lo = l;
    *hi = h;
    return n;
}

/* acc[16 b + i] += w[i] * xn[c] over the nonzero inputs c (increasing) and the column's blocks
   [s0[c], s1[c]): each output sums its terms in increasing input order, as a row-major dot
   product does; a zero weight inside a block adds +-0 to a sum that is never -0, which changes
   nothing. tb (or NULL) gets the blocks touched (a column's blocks are increasing: a word's
   bits gather in a register). Returns the blocks visited. */
static inline __attribute__((always_inline)) uint64_t
bscatter(const bcsc_t *m, const uint32_t *s0, const uint32_t *s1, const float *xn, const int32_t *nz,
         int n, float *acc, uint64_t *tb) {
    uint64_t blocks = 0;
    for (int k = 0; k < n; k++) {
        const int c = nz[k];
        const float32x4_t xv = vdupq_n_f32(xn[c]);
        const uint32_t j1 = s1[c];
        blocks += j1 - s0[c];
        uint64_t bits = 0;
        uint32_t cw = 0;
        for (uint32_t j = s0[c]; j < j1; j++) {
            const uint32_t b = m->blk[j];
            if (tb) {
                if (b >> 6 != cw) {
                    tb[cw] |= bits;
                    cw = b >> 6;
                    bits = 0;
                }
                bits |= 1ull << (b & 63);
            }
            float *a = acc + 16 * b;
            const uint16x8_t w0 = vld1q_u16(m->w + (size_t)j * 16), w1 = vld1q_u16(m->w + (size_t)j * 16 + 8);
            const float32x4_t f0 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(w0), 16));
            const float32x4_t f1 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(w0), 16));
            const float32x4_t f2 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(w1), 16));
            const float32x4_t f3 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(w1), 16));
            vst1q_f32(a, vfmaq_f32(vld1q_f32(a), f0, xv));
            vst1q_f32(a + 4, vfmaq_f32(vld1q_f32(a + 4), f1, xv));
            vst1q_f32(a + 8, vfmaq_f32(vld1q_f32(a + 8), f2, xv));
            vst1q_f32(a + 12, vfmaq_f32(vld1q_f32(a + 12), f3, xv));
        }
        if (tb) tb[cw] |= bits;
    }
    return blocks;
}

/* the 16-bit mask of the live lanes of one block (silu(bf16(acc)) possibly nonzero); g gets
   the bf16 values */
static inline uint32_t block_live(const float *a, float g[16]) {
    static const uint32_t bit[4] = {1, 2, 4, 8};
    const uint32x4_t bv = vld1q_u32(bit);
    uint32_t mask = 0;
    for (int q = 0; q < 4; q++) {
        const float32x4_t v = bf16r4(vld1q_f32(a + 4 * q));
        vst1q_f32(g + 4 * q, v);
        mask |= vaddvq_u32(vandq_u32(live4(v), bv)) << (4 * q);
    }
    return mask;
}

/* The leading entries of nz below p->pre (the constant prefix): if their (dim, value) pairs
   differ from the cached ones, base and acc are recomputed from them and *fresh is set (the
   caller then recomputes live0). Returns their count. */
static int prefix_same(const prefix_t *p, const float *xn, const int32_t *nz, int nnz, int *k) {
    int n = 0;
    while (n < nnz && nz[n] < p->pre) n++;
    *k = n;
    int same = p->ok && n == p->pn;
    for (int i = 0; i < n && same; i++) same = p->pcol[i] == nz[i] && p->pval[i] == f2u(xn[nz[i]]);
    return same;
}

static int prefix_sync(qd_engine *e, prefix_t *p, const bcsc_t *m, const float *xn,
                       const int32_t *nz, int nnz, int *fresh) {
    int k;
    const int same = prefix_same(p, xn, nz, nnz, &k);
    *fresh = !same;
    if (same) return k;
    e->st.prefix_misses++;
    memset(p->base, 0, (size_t)p->nb * 64);
    bscatter(m, m->ptr, m->ptr + 1, xn, nz, k, p->base, NULL);
    memcpy(p->acc, p->base, (size_t)p->nb * 64);
    for (int i = 0; i < k; i++) {
        p->pcol[i] = nz[i];
        p->pval[i] = f2u(xn[nz[i]]);
    }
    p->pn = k;
    p->ok = 1;
    return k;
}

/* put base back into the blocks of tb[w0, w1) */
static void restore_blocks(prefix_t *p, const uint64_t *tb, int w0, int w1) {
    for (int w = w0; w < w1; w++)
        for (uint64_t bm = tb[w]; bm; bm &= bm - 1) {
            const int b = 64 * w + __builtin_ctzll(bm);
            memcpy(p->acc + 16 * b, p->base + 16 * b, 64);
        }
}

/* the bitmap word w of the blocks [b0, b1) */
static inline uint64_t range_word(int w, int b0, int b1) {
    const int lo = b0 - 64 * w, hi = b1 - 64 * w; /* the bits [lo, hi) of this word */
    if (hi <= 0 || lo >= 64 || hi <= lo) return 0;
    uint64_t m = ~0ull;
    if (lo > 0) m &= ~0ull << lo;
    if (hi < 64) m &= (1ull << hi) - 1;
    return m;
}

/* One CSR row (increasing columns) against xn (zero outside its nonzero dims). */
static inline float row_dot(const uint32_t *ent, uint32_t a, uint32_t b, const float *xn) {
    float s = 0.0f;
    for (uint32_t j = a; j < b; j++) s = fmaf(W_OF(ent[j]), xn[I_OF(ent[j])], s);
    return s;
}

/* a pool row (z, a, b) at the current position: rows over the constant prefix only are the
   same at every step of a cached prefix (qk_fresh computes them) */
static inline float pool_dot(const layer_t *l, int row, const float *xn) {
    if (row < 0) return 0.0f;
    if (l->pool_pre[row]) return l->pool_c[row];
    return row_dot(l->pool_ent, l->pool_ptr[row], l->pool_ptr[row + 1], xn);
}

/* silu(bf16(z)) of distinct z row u at this step (a value head's gate on an output dim) */
static inline float zgate(qd_engine *e, layer_t *l, int u, const float *xn) {
    if (l->zu_pos[u] != e->pos) {
        l->zu_val[u] = silu_s(bf16r(pool_dot(l, l->zu_row[u], xn)), &e->st.slow_transcendental);
        l->zu_pos[u] = e->pos;
    }
    return l->zu_val[u];
}

/* x = bf16(x + bf16(dx)) and dx = 0 over the dims the sublayer touched (dxb; elsewhere dx is
   0 and x + +-0 = x); x's nonzero bitmap follows */
static void residual(qd_engine *e) {
    float *x = e->x, *dx = e->dx;
    uint64_t *xb = e->xb, *db = e->dxb;
    int same = 1;
    for (int w = 0; w < e->Hp / 64; w++) {
        uint64_t set = 0, clr = 0;
        for (uint64_t m = db[w]; m; m &= m - 1) {
            const int j = __builtin_ctzll(m), i = 64 * w + j;
            const float v = bf16r(x[i] + bf16r(dx[i]));
            dx[i] = 0.0f;
            same &= v == x[i]; /* (as values: a zero's sign never reaches the norm) */
            x[i] = v;
            const uint64_t b = 1ull << j, nzm = -(uint64_t)(v != 0.0f);
            set |= b & nzm;
            clr |= b & ~nzm;
        }
        xb[w] = (xb[w] | set) & ~clr;
        db[w] = 0;
    }
    e->inv_ok &= same;
}

/* --- timing ------------------------------------------------------------------------------------ */

static double g_tick_ns;

static inline uint64_t tick(const qd_engine *e) { return e->profile ? mach_absolute_time() : 0; }

static inline void lap(qd_engine *e, int stage, uint64_t *t) {
    if (!e->profile) return;
    uint64_t n = mach_absolute_time();
    e->st.ns[stage] += (double)(n - *t) * g_tick_ns;
    *t = n;
}

/* --- the thread pool ----------------------------------------------------------------------------- */

typedef void (*job_fn)(qd_engine *e, int t, void *arg);

struct qd_pool {
    qd_engine *e;
    pthread_t th[QD_MAX_THREADS];
    int nstarted;
    job_fn fn;
    void *arg;
    int nact; /* threads [0, nact) run the job */
    _Alignas(128) _Atomic uint32_t gen;
    _Alignas(128) _Atomic int32_t pending;
    _Alignas(128) _Atomic int32_t sleepers;
    _Atomic int32_t stop;
    pthread_mutex_t mu;
    pthread_cond_t cv;
};

typedef struct {
    qd_pool *p;
    int t;
} worker_arg;

/* how long an idle worker spins before it sleeps on the condition variable */
#define SPIN_NS 2000000.0

static void *worker(void *va) {
    worker_arg *wa = va;
    qd_pool *p = wa->p;
    const int t = wa->t;
    free(wa);
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    uint32_t seen = atomic_load(&p->gen);
    for (;;) {
        uint64_t t0 = 0;
        for (uint32_t spins = 0;; spins++) {
            if (atomic_load_explicit(&p->gen, memory_order_acquire) != seen) break;
            if ((spins & 1023) == 1023) {
                const uint64_t now = mach_absolute_time();
                if (!t0) t0 = now;
                else if ((double)(now - t0) * g_tick_ns > SPIN_NS) {
                    pthread_mutex_lock(&p->mu);
                    atomic_fetch_add(&p->sleepers, 1);
                    while (atomic_load(&p->gen) == seen) pthread_cond_wait(&p->cv, &p->mu);
                    atomic_fetch_sub(&p->sleepers, 1);
                    pthread_mutex_unlock(&p->mu);
                    break;
                }
            }
#if defined(__aarch64__)
            __asm__ volatile("yield");
#endif
        }
        seen = atomic_load(&p->gen);
        if (atomic_load(&p->stop)) break;
        /* (every worker acknowledges every job, taking part or not: the caller then never
           rewrites fn / arg / nact while a worker may still read them) */
        if (t < p->nact) p->fn(p->e, t, p->arg);
        atomic_fetch_sub_explicit(&p->pending, 1, memory_order_release);
    }
    return NULL;
}

/* run fn(e, t, arg) for the threads t < nact (the caller is thread 0) and wait for them; the
   per-thread outputs of the others are zeroed */
static void par(qd_engine *e, job_fn fn, void *arg, int nact) {
    for (int t = nact; t < e->nth; t++) {
        e->nitems[t] = e->nmblk[t] = 0;
        e->tblk[t] = e->tslow[t] = e->tmac[t] = 0;
        e->tlo[t] = INFINITY;
        e->thi[t] = 0.0f;
    }
    qd_pool *p = e->pool;
    if (!p || nact <= 1) {
        fn(e, 0, arg);
        return;
    }
    p->fn = fn;
    p->arg = arg;
    p->nact = nact;
    atomic_store_explicit(&p->pending, e->nth - 1, memory_order_relaxed);
    atomic_fetch_add(&p->gen, 1); /* (seq_cst: ordered before the sleepers load) */
    if (atomic_load(&p->sleepers)) {
        pthread_mutex_lock(&p->mu);
        pthread_cond_broadcast(&p->cv);
        pthread_mutex_unlock(&p->mu);
    }
    fn(e, 0, arg);
    uint64_t t = tick(e);
    while (atomic_load_explicit(&p->pending, memory_order_acquire) > 0) {
#if defined(__aarch64__)
        __asm__ volatile("yield");
#endif
    }
    lap(e, QD_ST_SYNC, &t);
}

static void empty_job(qd_engine *e, int t, void *arg) { (void)e, (void)t, (void)arg; }

double qd_engine_bench_sync(qd_engine *e, int n) {
    par(e, empty_job, NULL, e->nth); /* (wake the workers) */
    const uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < n; i++) par(e, empty_job, NULL, e->nth);
    return (double)(mach_absolute_time() - t0) * g_tick_ns / (n > 0 ? n : 1);
}

int qd_pool_start(qd_engine *e) {
    if (e->nth <= 1) return 0;
    qd_pool *p = calloc(1, sizeof *p);
    p->e = e;
    pthread_mutex_init(&p->mu, NULL);
    pthread_cond_init(&p->cv, NULL);
    e->pool = p;
    for (int t = 1; t < e->nth; t++) {
        worker_arg *wa = malloc(sizeof *wa);
        wa->p = p;
        wa->t = t;
        if (pthread_create(&p->th[t], NULL, worker, wa)) {
            free(wa);
            return -1;
        }
        p->nstarted = t;
    }
    return 0;
}

void qd_pool_stop(qd_engine *e) {
    qd_pool *p = e->pool;
    if (!p) return;
    atomic_store(&p->stop, 1);
    atomic_fetch_add(&p->gen, 1);
    pthread_mutex_lock(&p->mu);
    pthread_cond_broadcast(&p->cv);
    pthread_mutex_unlock(&p->mu);
    for (int t = 1; t <= p->nstarted; t++) pthread_join(p->th[t], NULL);
    pthread_mutex_destroy(&p->mu);
    pthread_cond_destroy(&p->cv);
    free(p);
    e->pool = NULL;
}

/* --- the token mixer --------------------------------------------------------------------------- */

typedef struct {
    layer_t *l;
    int slot;            /* the ring slot of this position */
    int pre;             /* nz[0, pre) is the cached prefix */
    const float *xn;
    const int32_t *nz;
    int nnz;
    int nact;            /* the threads running the job: thread t takes the ranges [v0, v1) */
} sub_arg;

/* Thread t of nact covers the precomputed thread ranges [v0, v1) of m (contiguous: its outputs
   stay in channel / unit order) */
static inline void job_ranges(const bcsc_t *m, int t, int nact, int *v0, int *v1) {
    *v0 = t * m->nact / nact;
    *v1 = (t + 1) * m->nact / nact;
}

/* The threads worth waking for a projection: its block visits (the nonzero inputs beyond the
   prefix) plus the blocks live at base, about PAR_GRAIN of them per thread; a small job runs
   on the caller alone, with no round trip through the pool. */
#define PAR_GRAIN 128
static int job_threads(const bcsc_t *m, const prefix_t *p, const int32_t *nz, int k0, int n) {
    if (m->nact <= 1) return 1;
    uint32_t w = (uint32_t)p->n_live0;
    for (int i = k0; i < n; i++) w += m->ptr[nz[i] + 1] - m->ptr[nz[i]];
    const int k = 1 + (int)(w / PAR_GRAIN);
    return k < m->nact ? k : m->nact;
}

/* job_threads for the eager channels: with the group bounds, the blocks not proved dead */
static int qk_threads(const layer_t *l, const int32_t *nz, int k0, int n) {
    const bcsc_t *m = &l->qk;
    if (!l->qb.on) return job_threads(m, &l->qp, nz, k0, n);
    if (m->nact <= 1 || !l->qb.nlive) return 1;
    uint64_t v = 0;
    for (int i = k0; i < n; i++) v += m->ptr[nz[i] + 1] - m->ptr[nz[i]];
    const uint64_t w = l->qb.nlive + v * l->qb.nlive / l->qp.nb;
    const int k = 1 + (int)(w / PAR_GRAIN);
    return k < m->nact ? k : m->nact;
}

/* The conv pre-activations (bf16) of the 4 channels j..j+3 (all single-tap, j < m1, or all
   on the ring): oldest tap first, multiply then add, as torch's conv does. */
static inline float32x4_t conv4(const layer_t *l, int KS, int np, int m1, int j, float32x4_t c,
                                const float *const *hs) {
    if (j < m1) return bf16r4(vmulq_f32(vld1q_f32(l->qk_tap + (size_t)(KS - 1) * np + j), c));
    float32x4_t s = vdupq_n_f32(0);
    for (int k = 0; k < KS; k++)
        s = vaddq_f32(s, vmulq_f32(vld1q_f32(l->qk_tap + (size_t)k * np + j), vld1q_f32(hs[k] + j)));
    return bf16r4(s);
}

/* the largest t x h over t in [t0, t1], h in [h0, h1] */
static inline double imul_hi(double t0, double t1, double h0, double h1) {
    double m = t0 * h0;
    m = fmax(m, t0 * h1);
    m = fmax(m, t1 * h0);
    return fmax(m, t1 * h1);
}

/* whether every channel of group g has a conv pre-activation <= -89 when its projection at
   tap k lies in [lo[k], hi[k]]: pre = bf16(sum_k fl(tap_k bf16(proj_k))). A bf16 rounding moves
   a value by at most 2^-9 of it (2^-8 is used), the KS products and sums by gamma_KS. */
static int group_dead(const qkbound_t *B, int g, int KS, const double *lo, const double *hi) {
    double s = 0.0, mag = 0.0;
    for (int k = 0; k < KS; k++) {
        const double t0 = B->tmin[k * B->ngp + g], t1 = B->tmax[k * B->ngp + g];
        if (t0 == 0.0 && t1 == 0.0) continue;
        const double h0 = lo[k] - fabs(lo[k]) * 0x1p-8, h1 = hi[k] + fabs(hi[k]) * 0x1p-8;
        const double m = imul_hi(t0, t1, h0, h1);
        s += m;
        mag += fmax(fmax(fabs(t0), fabs(t1)) * fmax(fabs(h0), fabs(h1)), fabs(m));
    }
    s += (double)(KS + 2) * 0x1p-23 * mag;
    s += fabs(s) * 0x1p-8;
    return s <= -89.0;
}

/* the blocks [b0, b1) of eager-channel group g (qkbound_t) */
static inline void group_blocks(const layer_t *l, int g, int *b0, int *b1) {
    const qkbound_t *B = &l->qb;
    if (g < B->ng0) {
        *b0 = 8 * g;
        *b1 = *b0 + 8 < l->rb ? *b0 + 8 : l->rb;
    } else {
        *b0 = l->rb + 8 * (g - B->ng0);
        *b1 = *b0 + 8 < l->qp.nb ? *b0 + 8 : l->qp.nb;
    }
}

static inline int block_group(const layer_t *l, int b) {
    return b < l->rb ? b / 8 : l->qb.ng0 + (b - l->rb) / 8;
}

/* the bound's group of eager channel j (level 0) or its region (level 1) */
static inline int chan_index(const layer_t *l, int j, int level) {
    const int g = block_group(l, j >> 4);
    return level ? l->qb.rq + (g >= l->qb.ng0) : g;
}

/* adds the terms of the inputs nz[k0, n) to the bounds of the groups [g0, g1) (multiples of
   4); returns the inputs with a table row */
static int bound_terms(qkbound_t *B, const float *xn, const int32_t *nz, int k0, int n, int g0,
                       int g1) {
    const int ngp = B->ngp;
    int cnt = 0;
    for (int i = k0; i < n; i++) {
        const int c = nz[i], r = B->colid[c];
        if (r < 0) continue;
        cnt++;
        const float32x4_t x = vdupq_n_f32(xn[c]), ax = vabsq_f32(x);
        const size_t o = (size_t)r * 3 * ngp;
        const float *wx = B->cwx + o, *wn = B->cwn + o, *gh = B->ghi + o, *gl = B->glo + o;
        const float *wa = B->wabs + (size_t)r * ngp;
        for (int g = g0; g < g1; g += 4) {
            float32x4_t h = vdupq_n_f32(-INFINITY), lw = vdupq_n_f32(INFINITY);
            for (int k = 0; k < 3; k++) {
                const float32x4_t a = vmulq_f32(vld1q_f32(wx + k * ngp + g), x);
                const float32x4_t b = vmulq_f32(vld1q_f32(wn + k * ngp + g), x);
                h = vmaxq_f32(h, vaddq_f32(vld1q_f32(gh + k * ngp + g), vmaxq_f32(a, b)));
                lw = vminq_f32(lw, vaddq_f32(vld1q_f32(gl + k * ngp + g), vminq_f32(a, b)));
            }
            vst1q_f32(B->lo + g, vaddq_f32(vld1q_f32(B->lo + g), lw));
            vst1q_f32(B->hi + g, vaddq_f32(vld1q_f32(B->hi + g), h));
            vst1q_f32(B->mag + g, vaddq_f32(vld1q_f32(B->mag + g), vmulq_f32(vld1q_f32(wa + g), ax)));
        }
    }
    return cnt;
}

/* a ring group's (or region's) projection range at this slot, from its bound */
static void ring_range(qkbound_t *B, int g, int slot, double ef) {
    const double err = ef * B->mag[g];
    B->slo[slot * B->ngp + g] = (float)(B->lo[g] - err); /* (rounded outwards in index_dead) */
    B->shi[slot * B->ngp + g] = (float)(B->hi[g] + err);
}

/* whether group (or region) g is proved dead at this position: single-tap channels from the
   current projection range, ring channels from every slot's (ring_range first) */
static int index_dead(const qkbound_t *B, int g, int ring, int slot, int KS, double ef) {
    const int ngp = B->ngp;
    double lo4[QD_MAX_KS], hi4[QD_MAX_KS];
    if (!ring) {
        const double err = ef * B->mag[g];
        for (int k = 0; k < KS; k++) lo4[k] = hi4[k] = 0.0;
        lo4[KS - 1] = B->lo[g] - err;
        hi4[KS - 1] = B->hi[g] + err;
    } else {
        for (int k = 0; k < KS; k++) {
            const int sk = (slot + k + 1) % KS;
            lo4[k] = B->slo[sk * ngp + g];
            hi4[k] = B->shi[sk * ngp + g];
            lo4[k] -= fabs(lo4[k]) * 0x1p-20; /* (the float roundings of the stored range) */
            hi4[k] += fabs(hi4[k]) * 0x1p-20;
        }
    }
    return group_dead(B, g, KS, lo4, hi4);
}

static void set_range(uint64_t *bm, int a, int b) {
    for (int w = a >> 6; w < (b + 63) >> 6; w++) bm[w] |= range_word(w, a, b);
}

/* This position's group bounds (qkbound_t): every channel j of group g has its projection
   fl(base_j + the terms of nz[k0, n)) in [lo_g, hi_g]. With base_j = r_j + sum over the rows of
   the offset of j's class there, the exact sum is r_j + sum over the rows of (offset + term),
   and per row the maximum (minimum) over the group's classes of offset + the class's extreme
   weights times the input bounds that share (rows whose input is 0 add their largest (smallest)
   offset, in blo / bhi). fp32 rounds the engine's n - k0 additions by at most gamma_n (|base| +
   sum |terms|), and this float computation of the bound (whose terms also hold offsets, at most
   the magnitude in babs twice) by at most 3 (n + 2) u of the same magnitudes: (2 n + 8) 2^-21
   is used, u = 2^-24. The regions first: a dead region's groups are all dead and get no bound
   of their own (a ring group's range at this slot is then computed later, from the slot's
   normed inputs, if a position that reads the slot needs it: rr_ok). Then B->gdead: the groups
   proved dead, B->nlive: the blocks of the others. */
static void qk_bound(qd_engine *e, layer_t *l, int slot, const float *xn, const int32_t *nz,
                     int k0, int n) {
    qkbound_t *B = &l->qb;
    const int ngp = B->ngp, KS = e->KS, rq = B->rq, ng0 = B->ng0, ng = B->ng, ring = ng > ng0;
    memcpy(B->lo, B->blo, (size_t)ngp * 4);
    memcpy(B->hi, B->bhi, (size_t)ngp * 4);
    memcpy(B->mag, B->babs, (size_t)ngp * 4);
    const int cnt = bound_terms(B, xn, nz, k0, n, rq, rq + 4);
    const double ef = (double)(2 * cnt + 8) * 0x1p-21;
    if (ring) ring_range(B, rq + 1, slot, ef);
    const int dead_s = ng0 > 0 && index_dead(B, rq, 0, slot, KS, ef);
    const int dead_r = ring && index_dead(B, rq + 1, 1, slot, KS, ef);
    if (ring && !dead_r) { /* the per-group tests read every slot's group ranges */
        for (int s = 0; s < KS; s++) {
            if (s == slot || B->rr_ok[s]) continue;
            memcpy(B->lo, B->blo, (size_t)ngp * 4);
            memcpy(B->hi, B->bhi, (size_t)ngp * 4);
            memcpy(B->mag, B->babs, (size_t)ngp * 4);
            const int32_t *nzs = l->ring_nz + (size_t)s * e->Hp;
            const int ns = l->ring_n[s];
            int ks = 0;
            while (ks < ns && nzs[ks] < l->qp.pre) ks++;
            const int cs = bound_terms(B, l->xn_ring + (size_t)s * e->Hp, nzs, ks, ns, ng0 & ~3, rq);
            const double es = (double)(2 * cs + 8) * 0x1p-21;
            for (int g = ng0; g < ng; g++) ring_range(B, g, s, es);
            B->rr_ok[s] = 1;
        }
        memcpy(B->lo, B->blo, (size_t)rq * 4);
        memcpy(B->hi, B->bhi, (size_t)rq * 4);
        memcpy(B->mag, B->babs, (size_t)rq * 4);
    }
    if (!dead_s || (ring && !dead_r))
        bound_terms(B, xn, nz, k0, n, dead_s ? ng0 & ~3 : 0, ring && !dead_r ? rq : (ng0 + 3) & ~3);
    memset(B->gdead, 0, (size_t)B->ngw * 8);
    B->nlive = l->qp.nb;
    if (ring) {
        B->rr_ok[slot] = !dead_r;
        if (!dead_r)
            for (int g = ng0; g < ng; g++) ring_range(B, g, slot, ef);
    }
    for (int g = 0; g < ng; g++) {
        if (g < ng0 ? dead_s : dead_r) continue;
        if (!index_dead(B, g, g >= ng0, slot, KS, ef)) continue;
        e->st.bound_skips++;
        int b0, b1;
        group_blocks(l, g, &b0, &b1);
        B->gdead[g >> 6] |= 1ull << (g & 63);
        B->nlive -= b1 - b0;
    }
    if (dead_s) {
        e->st.bound_skips += ng0;
        set_range(B->gdead, 0, ng0);
        B->nlive -= l->rb;
    }
    if (dead_r) {
        e->st.bound_skips += ng - ng0;
        set_range(B->gdead, ng0, ng);
        B->nlive -= l->qp.nb - l->rb;
    }
}

/* The bound's offsets for qk's new base (qkbound_t), for the groups and the regions: per row
   beyond the prefix and group, each class's offset is the mean of its channels' bases minus the
   group's mean (0 for a class the group lacks or a row with one class: such offsets are equal
   and cancel); r_j = base_j - the offsets of j's classes (in double: exact for the circuit's
   values, and any rounding is far below the bound's error term). */
static void qk_fit(qd_engine *e, layer_t *l) {
    qkbound_t *B = &l->qb;
    const prefix_t *p = &l->qp;
    const bcsc_t *m = &l->qk;
    const int ngp = B->ngp, R = B->rows, nqk = l->nqk, H = e->H;
    const size_t n3 = (size_t)R * 3 * ngp;
    double *sum = calloc(n3, 8), *gam = calloc(n3, 8), *gsum = calloc(ngp, 8);
    double *gm = calloc(ngp, 8), *rsum = calloc(ngp, 8), *rj = calloc(nqk, 8);
    double *rlo = malloc(ngp * 8), *rhi = malloc(ngp * 8), *ra = calloc(ngp, 8), *ba = calloc(ngp, 8);
    int32_t *cnt = calloc(n3, 4), *gcnt = calloc(ngp, 4);
    for (int lv = 0; lv < 2; lv++) {
        for (int j = 0; j < nqk; j++) {
            const int g = chan_index(l, j, lv);
            gsum[g] += p->base[j];
            gcnt[g]++;
        }
        for (int c = p->pre; c < H; c++) {
            const int r = B->colid[c];
            if (r < 0) continue;
            for (uint32_t q = m->ptr[c]; q < m->ptr[c + 1]; q++)
                for (int i = 0; i < 16; i++) {
                    const int j = 16 * m->blk[q] + i;
                    const float w = u2f((uint32_t)m->w[(size_t)q * 16 + i] << 16);
                    if (j >= nqk || w == 0.0f) continue;
                    const size_t o = ((size_t)r * 3 + (w > 0.0f ? 0 : 1)) * ngp + chan_index(l, j, lv);
                    sum[o] += p->base[j];
                    cnt[o]++;
                }
        }
    }
    for (int g = 0; g < ngp; g++) gm[g] = gcnt[g] ? gsum[g] / gcnt[g] : 0.0;
    for (int c = p->pre; c < H; c++) {
        const int r = B->colid[c];
        if (r < 0) continue;
        for (int g = 0; g < ngp; g++) {
            if (!gcnt[g]) continue;
            const size_t o = (size_t)r * 3 * ngp + g;
            const int n0 = gcnt[g] - cnt[o] - cnt[o + ngp];
            const double s0 = gsum[g] - sum[o] - sum[o + ngp];
            double gk[3];
            for (int k = 0; k < 3; k++) {
                const int nk = k < 2 ? cnt[o + k * ngp] : n0;
                const double sk = k < 2 ? sum[o + k * ngp] : s0;
                gk[k] = nk ? (double)(float)(sk / nk - gm[g]) : 0.0;
            }
            for (int k = 0; k < 3; k++) gam[o + k * ngp] = gk[B->alias[o + k * ngp]];
            rsum[g] += gam[o + 2 * ngp];
        }
    }
    for (int g = 0; g < ngp; g++) {
        rlo[g] = INFINITY;
        rhi[g] = -INFINITY;
    }
    for (int lv = 0; lv < 2; lv++) {
        for (int j = 0; j < nqk; j++) rj[j] = (double)p->base[j] - rsum[chan_index(l, j, lv)];
        for (int c = p->pre; c < H; c++) {
            const int r = B->colid[c];
            if (r < 0) continue;
            for (uint32_t q = m->ptr[c]; q < m->ptr[c + 1]; q++)
                for (int i = 0; i < 16; i++) {
                    const int j = 16 * m->blk[q] + i;
                    const float w = u2f((uint32_t)m->w[(size_t)q * 16 + i] << 16);
                    if (j >= nqk || w == 0.0f) continue;
                    const size_t o = (size_t)r * 3 * ngp + chan_index(l, j, lv);
                    rj[j] -= gam[o + (w > 0.0f ? 0 : 1) * ngp] - gam[o + 2 * ngp];
                }
        }
        for (int j = 0; j < nqk; j++) {
            const int g = chan_index(l, j, lv);
            rlo[g] = fmin(rlo[g], rj[j]);
            rhi[g] = fmax(rhi[g], rj[j]);
            ra[g] = fmax(ra[g], fabs(rj[j]));
            ba[g] = fmax(ba[g], fabs((double)p->base[j]));
        }
    }
    for (int g = 0; g < ngp; g++) {
        if (!gcnt[g]) continue;
        double glo = 0.0, ghi = 0.0, ga = 0.0;
        for (int r = 0; r < R; r++) {
            const size_t o = (size_t)r * 3 * ngp + g;
            double mn = gam[o], mx = gam[o];
            for (int k = 1; k < 3; k++) {
                mn = fmin(mn, gam[o + k * ngp]);
                mx = fmax(mx, gam[o + k * ngp]);
            }
            for (int k = 0; k < 3; k++) {
                B->ghi[o + k * ngp] = (float)(gam[o + k * ngp] - mx);
                B->glo[o + k * ngp] = (float)(gam[o + k * ngp] - mn);
            }
            glo += mn;
            ghi += mx;
            ga += fmax(fabs(mn), fabs(mx));
        }
        B->blo[g] = (float)(rlo[g] + glo);
        B->bhi[g] = (float)(rhi[g] + ghi);
        B->babs[g] = (float)(ba[g] + ra[g] + ga);
    }
    double *tmp[] = {sum, gam, gsum, gm, rsum, rj, rlo, rhi, ra, ba};
    for (size_t i = 0; i < sizeof tmp / sizeof *tmp; i++) free(tmp[i]);
    free(cnt);
    free(gcnt);
}

/* qk's base changed: live0 = the blocks with a live channel when every input beyond the prefix
   is 0 at every tap; every ring slot is marked as holding arbitrary projections; the bound's
   offsets */
static void qk_fresh(qd_engine *e, layer_t *l, const float *xn) {
    prefix_t *p = &l->qp;
    for (int r = 0; r < l->npool; r++)
        if (l->pool_pre[r]) l->pool_c[r] = row_dot(l->pool_ent, l->pool_ptr[r], l->pool_ptr[r + 1], xn);
    const int KS = e->KS, np = l->nqk_pad, m1 = l->nqk1 & ~3;
    memset(p->live0, 0, (size_t)p->nbw * 8);
    for (int j = 0; j < np; j += 4) {
        const float32x4_t c = bf16r4(vld1q_f32(p->base + j));
        float32x4_t pre;
        if (j < m1) {
            pre = bf16r4(vmulq_f32(vld1q_f32(l->qk_tap + (size_t)(KS - 1) * np + j), c));
        } else {
            float32x4_t s = vdupq_n_f32(0);
            for (int k = 0; k < KS; k++)
                s = vaddq_f32(s, vmulq_f32(vld1q_f32(l->qk_tap + (size_t)k * np + j), c));
            pre = bf16r4(s);
        }
        uint32_t lv[4];
        vst1q_u32(lv, live4(pre));
        for (int i = 0; i < 4; i++)
            if (lv[i] && j + i < l->nqk) p->live0[(j + i) >> 10] |= 1ull << (((j + i) >> 4) & 63);
    }
    memset(l->ring_tb, 0xFF, (size_t)KS * p->nbw * 8);
    p->n_live0 = 0;
    for (int w = 0; w < p->nbw; w++) p->n_live0 += __builtin_popcountll(p->live0[w]);
    if (l->qb.on) qk_fit(e, l);
}

/* The deferred projections of ring group g at ring slot s (the position whose normed inputs
   are xn_ring[s]) into qk_hist: base plus the terms of that position's inputs beyond the
   prefix, in increasing input order, as the scatter would have computed them then (base is
   the same: qk_flush runs before it changes). */
static void qk_materialize(qd_engine *e, layer_t *l, int s, int g) {
    const prefix_t *p = &l->qp;
    const bcsc_t *m = &l->qk;
    int b0, b1;
    group_blocks(l, g, &b0, &b1);
    float a[128];
    memcpy(a, p->base + 16 * b0, (size_t)(b1 - b0) * 64);
    const float *xn = l->xn_ring + (size_t)s * e->Hp;
    const int32_t *nz = l->ring_nz + (size_t)s * e->Hp;
    const int n = l->ring_n[s];
    int k = 0;
    while (k < n && nz[k] < p->pre) k++;
    uint64_t blocks = 0;
    const qkbound_t *B = &l->qb;
    for (; k < n; k++) {
        const int c = nz[k], r = B->colid[c];
        if (r < 0) continue;
        const float32x4_t xv = vdupq_n_f32(xn[c]);
        const uint32_t *gp = B->gptr + (size_t)r * (B->ng + 1) + g;
        for (uint32_t j = gp[0]; j < gp[1]; j++) {
            float *ab = a + 16 * (m->blk[j] - b0);
            const uint16_t *w = m->w + (size_t)j * 16;
            for (int q = 0; q < 16; q += 4) {
                const float32x4_t f = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w + q), 16));
                vst1q_f32(ab + q, vfmaq_f32(vld1q_f32(ab + q), f, xv));
            }
            blocks++;
        }
    }
    e->st.mac[QD_ST_QK] += 16 * blocks;
    float *h = l->qk_hist + (size_t)s * l->nqk_pad + 16 * b0;
    for (int q = 0; q < 16 * (b1 - b0); q += 4) vst1q_f32(h + q, bf16r4(vld1q_f32(a + q)));
}

/* qk's base is about to change: compute every deferred projection (not the current slot's:
   that position has left the window); a slot without its ring groups' ranges takes the ring
   region's (a looser bound on the same projections) */
static void qk_flush(qd_engine *e, layer_t *l, int slot) {
    qkbound_t *B = &l->qb;
    const int ngp = B->ngp;
    for (int s = 0; s < e->KS; s++) {
        if (B->rr_ok[s]) continue;
        for (int g = B->ng0; g < B->ng; g++) {
            B->slo[s * ngp + g] = B->slo[s * ngp + B->rq + 1];
            B->shi[s * ngp + g] = B->shi[s * ngp + B->rq + 1];
        }
        B->rr_ok[s] = 1;
    }
    for (int s = 0; s < e->KS; s++) {
        uint64_t *pd = B->pend + (size_t)s * B->ngw;
        for (int w = 0; w < B->ngw; w++) {
            for (uint64_t m = s == slot ? 0 : pd[w]; m; m &= m - 1)
                qk_materialize(e, l, s, 64 * w + __builtin_ctzll(m));
            pd[w] = 0;
        }
    }
}

/* After the bound: this position's projections of the dead ring groups are deferred; a live
   ring group gets its deferred projections at the other slots computed (its conv reads them) */
static void qk_defer(qd_engine *e, layer_t *l, int slot) {
    qkbound_t *B = &l->qb;
    uint64_t *cur = B->pend + (size_t)slot * B->ngw;
    for (int w = 0; w < B->ngw; w++) {
        const uint64_t ring = range_word(w, B->ng0, B->ng);
        cur[w] = B->gdead[w] & ring;
        for (uint64_t m = ring & ~B->gdead[w]; m; m &= m - 1) {
            const int g = 64 * w + __builtin_ctzll(m);
            for (int s = 0; s < e->KS; s++) {
                uint64_t *pd = B->pend + (size_t)s * B->ngw + w;
                if (s == slot || !(*pd >> (g & 63) & 1)) continue;
                qk_materialize(e, l, s, g);
                *pd &= ~(1ull << (g & 63));
            }
        }
    }
}

/* the conv and silu of block b's channels (its projections in acc, the earlier ones in hs):
   the nonzero results to out in channel order; returns the new count */
static inline int conv_block(const qd_engine *e, const layer_t *l, int b, const float *acc,
                             const float *const *hs, item_t *out, int n, uint64_t *slow) {
    const int KS = e->KS, np = l->nqk_pad, m1 = l->nqk1 & ~3, j = 16 * b;
    float pv[16];
    uint32_t lv[16];
    uint32x4_t any = vdupq_n_u32(0);
    for (int q = 0; q < 16; q += 4) {
        const float32x4_t c = bf16r4(vld1q_f32(acc + j + q));
        const float32x4_t pre = conv4(l, KS, np, m1, j + q, c, hs);
        const uint32x4_t live = live4(pre);
        vst1q_f32(pv + q, pre);
        vst1q_u32(lv + q, live);
        any = vorrq_u32(any, live);
    }
    if (!vmaxvq_u32(any)) return n;
    for (int i = 0; i < 16; i++) {
        if (!lv[i] || j + i >= l->nqk) continue;
        const float sv = bf16r(silu_s(pv[i], slow));
        if (sv != 0.0f) out[n++] = (item_t){l->qk_id[j + i], sv};
    }
    return n;
}

/* Thread t's blocks [b0, b1) of eager channels with the group bounds: only the groups not
   proved dead (qkbound_t), each scattered on its own (a group's blocks in an input's column
   are contiguous: gptr), then its ring blocks' projections into this slot, its conv and silu
   (its projections at the other slots are in qk_hist: qk_defer), base back into acc. */
static void qk_live_job(qd_engine *e, int t, const sub_arg *a, int b0, int b1) {
    layer_t *l = a->l;
    prefix_t *p = &l->qp;
    const qkbound_t *B = &l->qb;
    const bcsc_t *m = &l->qk;
    const int KS = e->KS, np = l->nqk_pad, G1 = B->ng + 1;
    float *acc = p->acc, *cur = l->qk_hist + (size_t)a->slot * np;
    const float *hs[QD_MAX_KS];
    for (int k = 0; k < KS; k++) hs[k] = l->qk_hist + (size_t)((a->slot + k + 1) % KS) * np;
    item_t *out = e->items[t];
    int n = 0;
    uint64_t slow = 0, blocks = 0, tp = t == 0 ? tick(e) : 0;
    const int g1 = b1 > b0 ? block_group(l, b1 - 1) + 1 : 0;
    for (int g = b1 > b0 ? block_group(l, b0) : 0; g < g1; g++) {
        if (B->gdead[g >> 6] >> (g & 63) & 1) continue;
        int gb0, gb1;
        group_blocks(l, g, &gb0, &gb1);
        gb0 = gb0 > b0 ? gb0 : b0;
        gb1 = gb1 < b1 ? gb1 : b1;
        for (int i = a->pre; i < a->nnz; i++) {
            const int c = a->nz[i], r = B->colid[c];
            if (r < 0) continue;
            const float32x4_t xv = vdupq_n_f32(a->xn[c]);
            for (uint32_t j = B->gptr[(size_t)r * G1 + g]; j < B->gptr[(size_t)r * G1 + g + 1]; j++) {
                const int bk = m->blk[j];
                if (bk < gb0 || bk >= gb1) continue; /* (a group split between threads) */
                float *ab = acc + 16 * bk;
                const uint16_t *w = m->w + (size_t)j * 16;
                for (int q = 0; q < 16; q += 4) {
                    const float32x4_t f = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w + q), 16));
                    vst1q_f32(ab + q, vfmaq_f32(vld1q_f32(ab + q), f, xv));
                }
                blocks++;
            }
        }
        if (t == 0) lap(e, QD_ST_QK, &tp);
        for (int bk = gb0 > l->rb ? gb0 : l->rb; bk < gb1; bk++) /* this position's projections */
            for (int q = 0; q < 16; q += 4)
                vst1q_f32(cur + 16 * bk + q, bf16r4(vld1q_f32(acc + 16 * bk + q)));
        for (int bk = gb0; bk < gb1; bk++) n = conv_block(e, l, bk, acc, hs, out, n, &slow);
        memcpy(acc + 16 * gb0, p->base + 16 * gb0, (size_t)(gb1 - gb0) * 64);
        if (t == 0) lap(e, QD_ST_CONV, &tp);
    }
    e->tblk[t] = blocks;
    e->nitems[t] = n;
    e->tslow[t] = slow;
}

/* Thread t's blocks of eager channels: the projection's terms beyond the cached prefix, the
   ring slot's blocks (bf16 of the projection: data where touched, base elsewhere), then for
   the blocks that can hold a live channel (touched now, holding data at some tap, or live at
   base) the conv and silu; the nonzero results go to its item list in channel order. With the
   group bounds: qk_live_job. */
static void qk_job(qd_engine *e, int t, void *va) {
    const sub_arg *a = va;
    layer_t *l = a->l;
    prefix_t *p = &l->qp;
    const int KS = e->KS, np = l->nqk_pad, m1 = l->nqk1 & ~3, rb = m1 >> 4;
    int v0, v1;
    job_ranges(&l->qk, t, a->nact, &v0, &v1);
    const int b0 = l->qk.bsplit[v0], b1 = l->qk.bsplit[v1];
    if (l->qb.on) {
        qk_live_job(e, t, a, b0, b1);
        return;
    }
    const int w0 = b0 >> 6, w1 = (b1 + 63) >> 6, nbw = p->nbw;
    uint64_t *tb = e->tb[t];
    const float *acc = p->acc;
    uint64_t tp = t == 0 ? tick(e) : 0;
    memset(tb + w0, 0, (size_t)(w1 - w0) * 8);
    e->tblk[t] = bscatter(&l->qk, l->qk.split[v0], l->qk.split[v1], a->xn, a->nz + a->pre,
                          a->nnz - a->pre, p->acc, tb);
    if (t == 0) lap(e, QD_ST_QK, &tp);
    float *cur = l->qk_hist + (size_t)a->slot * np;
    const float *hs[QD_MAX_KS];
    for (int k = 0; k < KS; k++) hs[k] = l->qk_hist + (size_t)((a->slot + k + 1) % KS) * np;
    uint64_t *rt = l->ring_tb + (size_t)a->slot * nbw;
    item_t *out = e->items[t];
    int n = 0;
    uint64_t slow = 0;
    for (int w = w0; w < w1; w++) {
        const uint64_t range = range_word(w, b0, b1);
        const uint64_t ring = range & range_word(w, rb, p->nb);
        const uint64_t now = tb[w] & ring;
        for (uint64_t bm = rt[w] & ring & ~now; bm; bm &= bm - 1) { /* back to base */
            const int j = 16 * (64 * w + __builtin_ctzll(bm));
            for (int q = 0; q < 16; q += 4) vst1q_f32(cur + j + q, bf16r4(vld1q_f32(p->base + j + q)));
        }
        for (uint64_t bm = now; bm; bm &= bm - 1) { /* this position's projections */
            const int j = 16 * (64 * w + __builtin_ctzll(bm));
            for (int q = 0; q < 16; q += 4) vst1q_f32(cur + j + q, bf16r4(vld1q_f32(acc + j + q)));
        }
        rt[w] = (rt[w] & ~ring) | now;
        uint64_t held = 0;
        for (int k = 0; k < KS; k++) held |= l->ring_tb[(size_t)k * nbw + w];
        for (uint64_t bm = (tb[w] | p->live0[w] | (held & ring)) & range; bm; bm &= bm - 1)
            n = conv_block(e, l, 64 * w + __builtin_ctzll(bm), acc, hs, out, n, &slow);
    }
    restore_blocks(p, tb, w0, w1);
    if (t == 0) lap(e, QD_ST_CONV, &tp);
    e->nitems[t] = n;
    e->tslow[t] = slow;
}

/* insertion sort of a head's (dim, value) list: a head whose channels sit in both the 1-tap
   and the ring region arrives out of order */
static void sort_dims(int32_t *d, float *v, int n) {
    for (int i = 1; i < n; i++) {
        const int32_t dk = d[i];
        const float vk = v[i];
        int j = i - 1;
        for (; j >= 0 && d[j] > dk; j--) {
            d[j + 1] = d[j];
            v[j + 1] = v[j];
        }
        d[j + 1] = dk;
        v[j + 1] = vk;
    }
}

/* A write with beta = 0, S += k (v - kv) 0, changes no row while v - kv is finite (then every
   added term is +-0): |kv| < 2^126 (NaN fails) and |v| < 2^101, from |v| <= 1.1 vbound max |xn|
   + 0.3 over the slots it reads (the row sums, the bf16 roundings and the taps; |silu(x)| <=
   max(|x|, 0.28)). Then the value need not be computed. */
static int write_noop(const qd_engine *e, const layer_t *l, int i, const float *kvm) {
    uint32x4_t big = vdupq_n_u32(0);
    for (int c = 0; c < e->dv; c += 4)
        big = vorrq_u32(big, vmvnq_u32(vcltq_f32(vabsq_f32(vld1q_f32(kvm + c)), vdupq_n_f32(0x1p126f))));
    float mx = 0.0f;
    for (int k = 0; k < e->KS; k++) mx = fmaxf(mx, l->xn_hi[k]);
    return !vmaxvq_u32(big) && (double)l->vbound[i] * mx < 0x1p100;
}

static void attn_block(qd_engine *e, layer_t *l, int slot) {
    const int Hp = e->Hp, KS = e->KS, dk = e->dk, dv = e->dv, nk = e->nk;
    uint64_t t = tick(e);
    float *xn = l->xn_ring + (size_t)slot * Hp;
    int32_t *rnz = l->ring_nz + (size_t)slot * Hp;
    for (int k = 0; k < l->ring_n[slot]; k++) xn[rnz[k]] = 0.0f;
    float lo, hi;
    const int nnz = norm_cols(e, l->g_in, l->a_colb, xn, rnz, &lo, &hi);
    l->ring_n[slot] = nnz;
    l->xn_hi[slot] = hi;
    if (!range_ok(l, WR_ATTN_IN, lo, hi)) e->st.range_risk++;
    lap(e, QD_ST_NORM, &t);

    int fresh, k;
    if (l->qb.on && !prefix_same(&l->qp, xn, rnz, nnz, &k)) qk_flush(e, l, slot);
    k = prefix_sync(e, &l->qp, &l->qk, xn, rnz, nnz, &fresh);
    if (fresh) qk_fresh(e, l, xn);
    if (l->qb.on) {
        qk_bound(e, l, slot, xn, rnz, k, nnz);
        qk_defer(e, l, slot);
    }
    lap(e, QD_ST_NORM, &t);
    sub_arg sa = {l, slot, k, xn, rnz, nnz, qk_threads(l, rnz, k, nnz)};
    par(e, qk_job, &sa, sa.nact);
    t = tick(e);
    /* the nonzero queries / keys, per key head, in dim order */
    memset(e->qn, 0, nk * 4);
    memset(e->kn, 0, nk * 4);
    uint32_t unsorted = 0;
    for (int th = 0; th < e->nth; th++) {
        e->st.mac[QD_ST_QK] += 16 * e->tblk[th];
        e->st.slow_transcendental += e->tslow[th];
        const item_t *it = e->items[th];
        for (int i = 0; i < e->nitems[th]; i++) {
            const uint32_t id = it[i].id;
            const int kh = id >> 16 & 0x7FFF, d = id & 0xFFFF;
            int32_t *cnt = id >> 31 ? e->kn : e->qn;
            int32_t *dd = (id >> 31 ? e->kd : e->qd) + kh * dk;
            float *vv = (id >> 31 ? e->kv : e->qv) + kh * dk;
            if (cnt[kh] && dd[cnt[kh] - 1] > d) unsorted = 1;
            dd[cnt[kh]] = d;
            vv[cnt[kh]++] = it[i].v;
        }
    }
    if (unsorted)
        for (int kh = 0; kh < nk; kh++) {
            sort_dims(e->qd + kh * dk, e->qv + kh * dk, e->qn[kh]);
            sort_dims(e->kd + kh * dk, e->kv + kh * dk, e->kn[kh]);
        }
    lap(e, QD_ST_CONV, &t);

    /* the value heads */
    float *dx = e->dx, *y = e->y;
    float kn[256], vv[256], kvm[256], dl[256], o[256];
    float ylo = INFINITY, yhi = 0.0f;
    /* (in increasing h, as before; a value head whose decay is 1 does nothing without a key or
       a query, so only the key heads with one, or with such a decay, are visited) */
    for (int i = 0, kh0 = 0; kh0 < nk; kh0++) {
        if (!e->kn[kh0] && !e->qn[kh0] && !l->kh_decay[kh0]) continue;
        for (i = l->kh_v[kh0]; i < l->kh_v[kh0 + 1]; i++) {
        const vhead_t *v = &l->vh[i];
        const int h = v->h, kh = v->kh;
        const int nkk = e->kn[kh], nqq = e->qn[kh];
        float *S = l->S + (size_t)h * dk * dv;
        uint64_t *rz = l->rows_nz + (size_t)h * (dk / 64);
        const float decay = v->a_row < 0 ? l->decay_c[i]
                                         : decay_s(bf16r(pool_dot(l, v->a_row, xn)), v->dt_bias,
                                                   v->A_log, &e->st.slow_transcendental);
        if (decay != 1.0f) {
            const float32x4_t dc = vdupq_n_f32(decay);
            for (int w = 0; w < dk / 64; w++)
                for (uint64_t m = rz[w]; m; m &= m - 1) {
                    float *row = S + (size_t)(w * 64 + __builtin_ctzll(m)) * dv;
                    for (int c = 0; c < dv; c += 4) vst1q_f32(row + c, vmulq_f32(vld1q_f32(row + c), dc));
                }
            if (decay == 0.0f) memset(rz, 0, (dk / 64) * 8);
        }
        if (nkk) { /* the delta rule over the key's nonzero rows */
            e->st.key_writes++;
            const float b = bf16r(pool_dot(l, v->b_row, xn));
            const float beta = bf16r(sigmoid_s(b, &e->st.slow_transcendental));
            const int32_t *kd = e->kd + kh * dk;
            const float *kv = e->kv + kh * dk;
            const float ik = 1.0f / sqrtf(sumsq16_sparse(kd, kv, nkk) + 1e-6f);
            for (int j = 0; j < nkk; j++) kn[j] = kv[j] * ik;
            memset(kvm, 0, dv * 4);
            for (int j = 0; j < nkk; j++) {
                if (kn[j] == 0.0f) continue;
                const float *row = S + (size_t)kd[j] * dv;
                const float32x4_t kj = vdupq_n_f32(kn[j]);
                for (int c = 0; c < dv; c += 4)
                    vst1q_f32(kvm + c, vaddq_f32(vld1q_f32(kvm + c), vmulq_f32(vld1q_f32(row + c), kj)));
            }
            if (beta == 0.0f && write_noop(e, l, i, kvm)) goto read;
            /* the value: lazily, from the normed inputs of the last KS positions */
            uint64_t tv = tick(e);
            memset(vv, 0, dv * 4);
            for (int j = v->v0; j < v->v0 + v->nvc; j++) {
                const uint32_t a0 = l->pool_ptr[l->vch_row[j]], a1 = l->pool_ptr[l->vch_row[j] + 1];
                float cs = 0.0f;
                for (int k = 0; k < KS; k++) {
                    const float tw = l->vch_tap[(size_t)j * KS + k];
                    if (tw == 0.0f) continue; /* (+0 * finite adds +-0 to a sum) */
                    const float *xk = l->xn_ring + (size_t)((slot + k + 1) % KS) * Hp;
                    cs = cs + tw * bf16r(row_dot(l->pool_ent, a0, a1, xk));
                    e->st.mac[QD_ST_VLAZY] += a1 - a0;
                }
                vv[l->vch_dim[j]] = bf16r(silu_s(bf16r(cs), &e->st.slow_transcendental));
            }
            if (e->profile) { /* (a stage of its own: taken out of delta below) */
                const double d = (double)(mach_absolute_time() - tv) * g_tick_ns;
                e->st.ns[QD_ST_VLAZY] += d;
                e->st.ns[QD_ST_DELTA] -= d;
            }
            const float32x4_t bt = vdupq_n_f32(beta);
            for (int c = 0; c < dv; c += 4)
                vst1q_f32(dl + c, vmulq_f32(vsubq_f32(vld1q_f32(vv + c), vld1q_f32(kvm + c)), bt));
            for (int j = 0; j < nkk; j++) {
                if (kn[j] == 0.0f) continue;
                float *row = S + (size_t)kd[j] * dv;
                const float32x4_t kj = vdupq_n_f32(kn[j]);
                for (int c = 0; c < dv; c += 4)
                    vst1q_f32(row + c, vaddq_f32(vld1q_f32(row + c), vmulq_f32(kj, vld1q_f32(dl + c))));
                rz[kd[j] / 64] |= 1ull << (kd[j] % 64);
            }
        }
    read:
        if (!nqq) continue; /* q = 0: o = 0, and the norm of 0 is 0 */
        /* the read over the query's nonzero rows, then the gated RMSNorm */
        e->st.query_reads++;
        const int32_t *qd = e->qd + kh * dk;
        const float *qv = e->qv + kh * dk;
        const float iq = 1.0f / sqrtf(sumsq16_sparse(qd, qv, nqq) + 1e-6f);
        memset(o, 0, dv * 4);
        for (int j = 0; j < nqq; j++) {
            const float qj = (qv[j] * iq) / e->qscale;
            if (qj == 0.0f) continue;
            const float *row = S + (size_t)qd[j] * dv;
            const float32x4_t q4 = vdupq_n_f32(qj);
            for (int c = 0; c < dv; c += 4)
                vst1q_f32(o + c, vaddq_f32(vld1q_f32(o + c), vmulq_f32(vld1q_f32(row + c), q4)));
        }
        uint32x4_t any = vdupq_n_u32(0), fin = vdupq_n_u32(~0u);
        for (int c = 0; c < dv; c += 4) {
            const float32x4_t r = bf16r4(vld1q_f32(o + c));
            vst1q_f32(o + c, r);
            any = vorrq_u32(any, vtstq_u32(vreinterpretq_u32_f32(r), vdupq_n_u32(0x7FFFFFFFu)));
            fin = vandq_u32(fin, vcltq_f32(vabsq_f32(r), vdupq_n_f32(INFINITY)));
        }
        if (!vmaxvq_u32(any)) continue;
        /* every gate silu(z) of the head 0: with o finite, y is (|o / rms(o)| <= sqrt(dv)), so
           every y z is +-0 and the head adds nothing */
        int gated = 0;
        for (int u = l->vz_ptr[i]; u < l->vz_ptr[i + 1] && !gated; u++)
            gated = zgate(e, l, l->vz_id[u], xn) != 0.0f;
        if (!gated && vminvq_u32(fin)) continue;
        const float inv = 1.0f / sqrtf(sumsq16(o, dv) / (float)dv + e->eps);
        const float32x4_t iv = vdupq_n_f32(inv);
        for (int c = 0; c < dv; c += 4)
            vst1q_f32(y + c, bf16r4(vmulq_f32(vld1q_f32(l->norm_w + c),
                                              bf16r4(vmulq_f32(vld1q_f32(o + c), iv)))));
        for (int j = v->o0; j < v->o0 + v->no; j++) {
            const int d = l->od_dim[j];
            if (y[d] == 0.0f) continue;
            const float yz = bf16r(y[d] * zgate(e, l, l->od_zu[j], xn));
            if (yz == 0.0f) continue;
            ylo = fminf(ylo, fabsf(yz));
            yhi = fmaxf(yhi, fabsf(yz));
            for (uint32_t k = l->od_ptr[j]; k < l->od_ptr[j + 1]; k++) {
                const uint32_t en = l->od_ent[k], i = I_OF(en);
                dx[i] = fmaf(W_OF(en), yz, dx[i]);
                e->dxb[i >> 6] |= 1ull << (i & 63);
            }
            e->st.mac[QD_ST_OUT] += l->od_ptr[j + 1] - l->od_ptr[j];
        }
        }
    }
    if (!range_ok(l, WR_ATTN_OUT, ylo, yhi)) e->st.range_risk++;
    lap(e, QD_ST_DELTA, &t);
    residual(e);
    lap(e, QD_ST_RESID, &t);
}

/* --- the MLP ----------------------------------------------------------------------------------- */

/* the 16-bit mask of the nonzero lanes of a block of activations */
static inline uint32_t nonzero16(const float *a) {
    static const uint32_t bit[4] = {1, 2, 4, 8};
    const uint32x4_t bv = vld1q_u32(bit), m = vdupq_n_u32(0x7FFFFFFFu);
    uint32_t mask = 0;
    for (int q = 0; q < 4; q++) {
        const uint32x4_t nz = vtstq_u32(vreinterpretq_u32_f32(vld1q_f32(a + 4 * q)), m);
        mask |= vaddvq_u32(vandq_u32(nz, bv)) << (4 * q);
    }
    return mask;
}

/* Thread t's blocks of units: the gate's terms beyond the cached prefix (prefix_t), then the
   units of the blocks it touched or that are live at base, in increasing order: for the live
   ones (silu(gate) != 0) the up projection and the activation, into e->act; a block with a
   nonzero activation goes to its block list (e->mblk) with the mask of those (e->amask). Then
   base goes back into the touched blocks. */
static void mlp_job(qd_engine *e, int t, void *va) {
    const sub_arg *a = va;
    const layer_t *l = a->l;
    const prefix_t *p = &l->gp;
    int v0, v1;
    job_ranges(&l->gate, t, a->nact, &v0, &v1);
    const int b0 = l->gate.bsplit[v0], b1 = l->gate.bsplit[v1];
    const int w0 = b0 >> 6, w1 = (b1 + 63) >> 6;
    uint64_t *tb = e->tb[t];
    float *acc = p->acc;
    const float *xn = a->xn;
    uint64_t tp = t == 0 ? tick(e) : 0;
    memset(tb + w0, 0, (size_t)(w1 - w0) * 8);
    e->tblk[t] = bscatter(&l->gate, l->gate.split[v0], l->gate.split[v1], xn, a->nz + a->pre,
                          a->nnz - a->pre, acc, tb);
    if (t == 0) lap(e, QD_ST_GATE, &tp);
    float *act = e->act;
    uint16_t *am = e->amask;
    int32_t *mb = e->mblk[t];
    int n = 0, nmb = 0;
    uint64_t slow = 0, up_mac = 0;
    /* |act| range as bits: lo1 = min(bits - 1) (0 wraps around), hi = max(bits) */
    uint32_t lo1 = 0xFFFFFFFFu, hib = 0;
    float g[16];
    const unit1_t *u1 = l->u1;
    uint32x4_t lov = vdupq_n_u32(0xFFFFFFFFu), hiv = vdupq_n_u32(0);
    for (int w = w0; w < w1; w++)
        for (uint64_t bm = (tb[w] | p->live0[w]) & range_word(w, b0, b1); bm; bm &= bm - 1) {
            const int b = 64 * w + __builtin_ctzll(bm);
            if (u1 && (l->u1_blk[w] >> (b & 63) & 1)) {
                /* all 16 units at once: silu(g) = g for g >= 17 and +-0 for g <= -89 or 0
                   (silu(g) * up is then +-0: no item); a block with a unit in between takes
                   the scalar path below */
                const float *ab = acc + 16 * b;
                const int32_t *iu = l->u1_iu + 16 * b;
                const int uni = l->u1_uni[w] >> (b & 63) & 1;
                uint32x4_t slowm = vdupq_n_u32(0), anym = vdupq_n_u32(0);
                float32x4_t gv4[4];
                for (int q = 0; q < 4; q++) {
                    gv4[q] = bf16r4(vld1q_f32(ab + 4 * q));
                    anym = vorrq_u32(anym, live4(gv4[q]));
                }
                if (!vmaxvq_u32(anym)) continue; /* (about half the candidate blocks) */
                float *av = act + 16 * b;
                for (int q = 0; q < 16; q += 4) {
                    const float32x4_t gq = gv4[q / 4];
                    const uint32x4_t ge = vcgeq_f32(gq, vdupq_n_f32(17.0f));
                    slowm = vorrq_u32(slowm, vbicq_u32(live4(gq), ge));
                    const float32x4_t sq = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(gq), ge));
                    float32x4_t xq;
                    if (uni) {
                        xq = vld1q_dup_f32(xn + iu[0]);
                    } else {
                        xq = vdupq_n_f32(0);
                        xq = vld1q_lane_f32(xn + iu[q], xq, 0);
                        xq = vld1q_lane_f32(xn + iu[q + 1], xq, 1);
                        xq = vld1q_lane_f32(xn + iu[q + 2], xq, 2);
                        xq = vld1q_lane_f32(xn + iu[q + 3], xq, 3);
                    }
                    const float32x4_t upq = bf16r4(vmulq_f32(vld1q_f32(l->u1_wu + 16 * b + q), xq));
                    const float32x4_t aq = bf16r4(vmulq_f32(sq, upq));
                    vst1q_f32(av + q, aq);
                    const uint32x4_t ab4 = vandq_u32(vreinterpretq_u32_f32(aq), vdupq_n_u32(0x7FFFFFFFu));
                    lov = vminq_u32(lov, vsubq_u32(ab4, vdupq_n_u32(1)));
                    hiv = vmaxq_u32(hiv, ab4);
                }
                if (!vmaxvq_u32(slowm)) {
                    const uint32_t mask = nonzero16(av);
                    if (mask) {
                        am[b] = (uint16_t)mask;
                        mb[nmb++] = b;
                        n += __builtin_popcount(mask);
                    }
                    up_mac += 16;
                    continue;
                }
            }
            uint32_t mask = 0;
            for (uint32_t lm = block_live(acc + 16 * b, g); lm; lm &= lm - 1) {
                const int lane = __builtin_ctz(lm), u = 16 * b + lane;
                const float gv = g[lane];
                /* (g is bf16 already: silu(g) = g for g >= 17) */
                const float s = gv >= 17.0f ? gv : bf16r(silu_s(gv, &slow));
                float up;
                if (u1 && l->up_ptr[u + 1] - l->up_ptr[u] == 1) {
                    up = u1[u].wu * xn[u1[u].iu];
                    up_mac++;
                } else {
                    const uint32_t r0 = l->up_ptr[u], r1 = l->up_ptr[u + 1];
                    up = row_dot(l->up_ent, r0, r1, xn);
                    up_mac += r1 - r0;
                }
                /* (branch-free: about a third of the live units multiply a zero up value) */
                const float av = bf16r(s * bf16r(up));
                const uint32_t ab = f2u(av) & 0x7FFFFFFFu;
                act[u] = av;
                mask |= (uint32_t)(ab != 0) << lane;
                lo1 = lo1 < ab - 1 ? lo1 : ab - 1;
                hib = hib > ab ? hib : ab;
            }
            if (mask) {
                am[b] = (uint16_t)mask;
                mb[nmb++] = b;
                n += __builtin_popcount(mask);
            }
        }
    lo1 = lo1 < vminvq_u32(lov) ? lo1 : vminvq_u32(lov);
    hib = hib > vmaxvq_u32(hiv) ? hib : vmaxvq_u32(hiv);
    const float lo = lo1 == 0xFFFFFFFFu ? INFINITY : u2f(lo1 + 1), hi = u2f(hib);
    restore_blocks((prefix_t *)p, tb, w0, w1);
    if (t == 0) lap(e, QD_ST_UP, &tp);
    e->nitems[t] = n;
    e->nmblk[t] = nmb;
    e->tslow[t] = slow;
    e->tmac[t] = up_mac;
    e->tlo[t] = lo;
    e->thi[t] = hi;
}

static void mlp_block(qd_engine *e, layer_t *l) {
    uint64_t t = tick(e);
    float *xm = e->xm;
    for (int k = 0; k < e->mnnz; k++) xm[e->mnz[k]] = 0.0f;
    float lo, hi;
    e->mnnz = norm_cols(e, l->g_post, l->m_colb, xm, e->mnz, &lo, &hi);
    if (!range_ok(l, WR_MLP_IN, lo, hi)) e->st.range_risk++;
    int fresh;
    const int k = prefix_sync(e, &l->gp, &l->gate, xm, e->mnz, e->mnnz, &fresh);
    if (fresh) { /* live0: the blocks with a unit that is live at base */
        prefix_t *p = &l->gp;
        float g[16];
        memset(p->live0, 0, (size_t)p->nbw * 8);
        p->n_live0 = 0;
        for (int b = 0; b < p->nb; b++)
            if (block_live(p->base + 16 * b, g)) {
                p->live0[b >> 6] |= 1ull << (b & 63);
                p->n_live0++;
            }
    }
    lap(e, QD_ST_NORM, &t);
    sub_arg sa = {l, 0, k, xm, e->mnz, e->mnnz, job_threads(&l->gate, &l->gp, e->mnz, k, e->mnnz)};
    par(e, mlp_job, &sa, sa.nact);
    t = tick(e);
    /* down: the units with a nonzero activation in increasing order (the thread block lists in
       thread order); a run of units writing one dim accumulates in a register (the same
       additions, in order) */
    float *dx = e->dx;
    const float *act = e->act;
    uint64_t *db = e->dxb;
    uint64_t dn = 0;
    int ci = -1;
    float cv = 0.0f;
    for (int th = 0; th < e->nth; th++) {
        e->st.mac[QD_ST_GATE] += 16 * e->tblk[th];
        e->st.mac[QD_ST_UP] += e->tmac[th];
        e->st.slow_transcendental += e->tslow[th];
        e->st.active_units += e->nitems[th];
        if (!range_ok(l, WR_MLP_OUT, e->tlo[th], e->thi[th])) e->st.range_risk++;
        const int32_t *mb = e->mblk[th];
        const int nmb = e->nmblk[th];
        if (l->u1) {
            for (int k = 0; k < nmb; k++)
                for (uint32_t m = e->amask[mb[k]]; m; m &= m - 1) {
                    const int u = 16 * mb[k] + __builtin_ctz(m);
                    const unit1_t *r = &l->u1[u];
                    if (r->id != ci) {
                        if (ci >= 0) dx[ci] = cv;
                        ci = r->id;
                        cv = dx[ci];
                        db[ci >> 6] |= 1ull << (ci & 63);
                    }
                    cv = fmaf(r->wd, act[u], cv);
                }
            dn += e->nitems[th];
        } else {
            for (int k = 0; k < nmb; k++)
                for (uint32_t m = e->amask[mb[k]]; m; m &= m - 1) {
                    const int u = 16 * mb[k] + __builtin_ctz(m);
                    for (uint32_t q = l->dn_ptr[u]; q < l->dn_ptr[u + 1]; q++) {
                        const uint32_t en = l->dn_ent[q];
                        if ((int)I_OF(en) != ci) {
                            if (ci >= 0) dx[ci] = cv;
                            ci = I_OF(en);
                            cv = dx[ci];
                            db[ci >> 6] |= 1ull << (ci & 63);
                        }
                        cv = fmaf(W_OF(en), act[u], cv);
                    }
                    dn += l->dn_ptr[u + 1] - l->dn_ptr[u];
                }
        }
    }
    if (ci >= 0) dx[ci] = cv;
    e->st.mac[QD_ST_DOWN] += dn;
    lap(e, QD_ST_DOWN, &t);
    residual(e);
    lap(e, QD_ST_RESID, &t);
}

/* --- the head ---------------------------------------------------------------------------------- */

static int head(qd_engine *e) {
    uint64_t t = tick(e);
    const int n = e->ncode;
    const float inv = x_inv_rms(e);
    float *y = e->y;
    for (int i = 0; i < n; i++) y[i] = bf16r(e->x[e->code[i]] * inv * e->gain[e->code[i]]);
    if (!e->exact_head && e->sign_ok) {
        /* every token row is 0 or +-A on the code dims: with |y| = m on all of them, a row
           scores A m (n - 2 d), d = its sign disagreements, so y's own signs win */
        const float m = fabsf(y[0]);
        int ok = m != 0.0f;
        uint32_t bits = 0;
        for (int i = 0; i < n && ok; i++) {
            ok = fabsf(y[i]) == m;
            bits |= (uint32_t)(y[i] > 0) << i;
        }
        if (ok && e->sign_tab[bits] >= 0) {
            e->st.head_sign++;
            lap(e, QD_ST_HEAD, &t);
            return e->sign_tab[bits];
        }
    }
    /* the full head over the code dims: logits = bf16(E y), the first maximum */
    e->st.head_exact++;
    float best = -INFINITY;
    int arg = 0;
    for (int tk = 0; tk < e->V; tk++) {
        const uint16_t *row = e->ecode + (size_t)tk * n;
        float s = 0.0f;
        for (int i = 0; i < n; i++) s = s + u2f((uint32_t)row[i] << 16) * y[i];
        s = bf16r(s);
        if (s > best) {
            best = s;
            arg = tk;
        }
    }
    lap(e, QD_ST_HEAD, &t);
    return arg;
}

/* --- steps ------------------------------------------------------------------------------------- */

static int forward(qd_engine *e, int tok, int want_head) {
    uint64_t t = tick(e);
    const int slot = (int)(e->pos % e->KS);
    e->pos++;
    if ((e->flags & QD_OPEN_REFRESH) && e->pos % 7 == 0)
        for (int L = 0; L < e->nl; L++) e->L[L].qp.ok = e->L[L].gp.ok = 0;
    for (int w = 0; w < e->Hp / 64; w++) {
        for (uint64_t m = e->xb[w]; m; m &= m - 1) e->x[64 * w + __builtin_ctzll(m)] = 0.0f;
        e->xb[w] = 0;
    }
    e->inv_ok = 0;
    if (tok >= 0 && tok < e->V) {
        const uint16_t *row = e->etab + (size_t)tok * e->necol;
        for (int j = 0; j < e->necol; j++)
            if (row[j] & 0x7FFF) {
                const int i = e->ecol[j];
                e->x[i] = u2f((uint32_t)row[j] << 16);
                e->xb[i >> 6] |= 1ull << (i & 63);
            }
    }
    lap(e, QD_ST_EMBED, &t);
    const int full = want_head || e->capture;
    for (int L = 0; L < e->nl; L++) {
        layer_t *l = &e->L[L];
        if (!full && L > e->last_needed) break;
        if (l->attn) attn_block(e, l, slot);
        if (!full && L == e->last_needed) break; /* its MLP only feeds the head */
        if (l->mlp) mlp_block(e, l);
        if (e->capture) memcpy(e->capture + (size_t)L * e->H, e->x, e->H * 4);
    }
    e->st.steps++;
    if (!want_head) {
        e->st.forced++;
        return -1;
    }
    return head(e);
}

int qd_engine_step(qd_engine *e, int token) { return forward(e, token, 1); }

void qd_engine_step_forced(qd_engine *e, int token) { forward(e, token, 0); }

int qd_engine_feed(qd_engine *e, const int *tokens, int n) {
    for (int i = 0; i + 1 < n; i++) forward(e, tokens[i], 0);
    return n > 0 ? forward(e, tokens[n - 1], 1) : -1;
}

int qd_engine_generate(qd_engine *e, int token, int n, int stop, int *out) {
    int i = 0;
    while (i < n) {
        token = forward(e, token, 1);
        out[i++] = token;
        if (token == stop) break;
    }
    return i;
}

int qd_engine_generate_until(qd_engine *e, int token, int n, const int *stops, int n_stops,
                             int *out) {
    int i = 0;
    while (i < n) {
        token = forward(e, token, 1);
        out[i++] = token;
        for (int k = 0; k < n_stops; k++)
            if (token == stops[k]) return i;
    }
    return i;
}

int qd_engine_check(qd_engine *e, const int *toks, int n, const uint8_t *skip) {
    for (int i = 0; i + 1 < n; i++) {
        if (skip && skip[i + 1]) {
            forward(e, toks[i], 0);
            continue;
        }
        if (forward(e, toks[i], 1) != toks[i + 1]) return i;
    }
    return -1;
}

/* --- derived layouts, state ------------------------------------------------------------------- */

static void wrange(float r[2], const uint16_t *w, size_t n) {
    for (size_t i = 0; i < n; i++) {
        const float a = fabsf(u2f((uint32_t)w[i] << 16));
        if (a == 0.0f) continue;
        r[0] = fminf(r[0], a);
        r[1] = fmaxf(r[1], a);
    }
}

static void wrange_ent(float r[2], const uint32_t *ent, size_t n) {
    for (size_t i = 0; i < n; i++) {
        const float a = fabsf(W_OF(ent[i]));
        if (a == 0.0f) continue;
        r[0] = fminf(r[0], a);
        r[1] = fmaxf(r[1], a);
    }
}

/* thread splits of a blocked CSC: contiguous block ranges with about equal block counts, on
   64-block boundaries (each thread owns whole words of the block bitmaps, whole lines of acc) */
static void split_bcsc(qd_engine *e, bcsc_t *m, int nblocks) {
    const int H = e->H, nth = e->nth;
    const uint32_t nb = m->ptr[H];
    uint32_t *cnt = calloc(nblocks + 1, 4);
    for (uint32_t j = 0; j < nb; j++) cnt[m->blk[j] + 1]++;
    for (int b = 0; b < nblocks; b++) cnt[b + 1] += cnt[b];
    m->bsplit[0] = 0;
    for (int t = 1; t < nth; t++) {
        const uint64_t want = (uint64_t)nb * t / nth;
        int b = m->bsplit[t - 1];
        while (b < nblocks && cnt[b] < want) b++;
        b = (b + 32) & ~63; /* (whole words of the block bitmaps per thread) */
        if (b > nblocks) b = nblocks;
        m->bsplit[t] = b < m->bsplit[t - 1] ? m->bsplit[t - 1] : b;
    }
    m->bsplit[nth] = nblocks;
    free(cnt);
    m->nact = 1; /* the threads that own blocks */
    for (int t = 1; t < nth; t++)
        if (m->bsplit[t] < nblocks) m->nact = t + 1;
    m->split[0] = m->ptr;
    m->split[nth] = m->ptr + 1;
    for (int t = 1; t < nth; t++) {
        m->split[t] = qd_alloc(e, (size_t)H * 4);
        for (int c = 0; c < H; c++) {
            uint32_t j = m->ptr[c];
            while (j < m->ptr[c + 1] && m->blk[j] < m->bsplit[t]) j++;
            m->split[t][c] = j;
        }
    }
}

/* The constant prefix bounds: per sublayer (attention, MLP of each layer, in order), the
   first residual dim that an earlier sublayer writes or whose embedding value differs between
   tokens (rows that are all zero, like PAD, aside: a step after one of those just recomputes
   the bases). pre[2 L] for layer L's attention, pre[2 L + 1] for its MLP. */
static void constant_prefix(qd_engine *e, int32_t *pre) {
    uint8_t *var = calloc(e->Hp, 1);
    const int n = e->necol;
    uint16_t *v0 = malloc((size_t)n * 2);
    int have = 0;
    for (int64_t t = 0; t < e->V; t++) {
        const uint16_t *row = e->etab + (size_t)t * n;
        int any = 0;
        for (int j = 0; j < n && !any; j++) any = (row[j] & 0x7FFF) != 0;
        if (!any) continue;
        if (!have) {
            memcpy(v0, row, (size_t)n * 2);
            have = 1;
            for (int j = 0; j < n; j++)
                if (!(row[j] & 0x7FFF)) var[e->ecol[j]] = 1;
            continue;
        }
        for (int j = 0; j < n; j++)
            if (row[j] != v0[j]) var[e->ecol[j]] = 1;
    }
    free(v0);
    for (int L = 0; L < e->nl; L++) {
        const layer_t *l = &e->L[L];
        for (int s = 0; s < 2; s++) {
            int32_t p = 0;
            while (p < e->H && !var[p]) p++;
            pre[2 * L + s] = p;
            if (s == 0)
                for (int k = 0; l->attn && k < l->na_out; k++) var[l->a_out[k]] = 1;
            else
                for (int k = 0; l->mlp && k < l->nm_out; k++) var[l->m_out[k]] = 1;
        }
    }
    free(var);
}

static void prefix_alloc(qd_engine *e, prefix_t *p, int nb, int32_t pre) {
    p->pre = pre;
    p->nb = nb;
    p->nbw = (nb + 63) / 64;
    p->base = qd_alloc(e, (size_t)nb * 64);
    p->acc = qd_alloc(e, (size_t)nb * 64);
    p->live0 = qd_alloc(e, (size_t)p->nbw * 8);
    p->pcol = qd_alloc(e, (size_t)(pre + 1) * 4);
    p->pval = qd_alloc(e, (size_t)(pre + 1) * 4);
    p->ok = 0;
}

/* The group bound tables of a layer's eager channels (qkbound_t). Off with QD_OPEN_GENERIC. */
static void qk_bounds(qd_engine *e, layer_t *l) {
    qkbound_t *B = &l->qb;
    const int H = e->H, KS = e->KS, np = l->nqk_pad, nb = l->qp.nb;
    l->rb = (l->nqk1 & ~3) >> 4;
    B->on = nb > 0 && !(e->flags & QD_OPEN_GENERIC);
    if (!B->on) return;
    B->ng0 = (l->rb + 7) / 8;
    B->ng = B->ng0 + (nb - l->rb + 7) / 8;
    B->rq = (B->ng + 3) & ~3;
    B->ngp = B->rq + 4;
    B->ngw = (B->ng + 63) / 64;
    const int ngp = B->ngp, ng = B->ng;
    B->colid = qd_alloc(e, (size_t)H * 4);
    int rows = 0;
    for (int c = 0; c < H; c++) B->colid[c] = l->qk.ptr[c + 1] > l->qk.ptr[c] ? rows++ : -1;
    B->rows = rows;
    const size_t n3 = (size_t)rows * 3 * ngp;
    B->alias = qd_alloc(e, n3);
    B->cwx = qd_alloc(e, n3 * 4);
    B->cwn = qd_alloc(e, n3 * 4);
    B->ghi = qd_alloc(e, n3 * 4);
    B->glo = qd_alloc(e, n3 * 4);
    B->wabs = qd_alloc(e, (size_t)rows * ngp * 4);
    B->gptr = qd_alloc(e, (size_t)rows * (ng + 1) * 4);
    /* the classes w > 0, w < 0 present per row and group (or region), their extreme weights;
       w = 0 is present where some of the group's channels have no weight in the row */
    int32_t *cnt = calloc(n3, 4), *gch = calloc(ngp, 4);
    for (size_t i = 0; i < n3; i++) {
        B->cwx[i] = -INFINITY;
        B->cwn[i] = INFINITY;
    }
    B->tmin = qd_alloc(e, (size_t)KS * ngp * 4);
    B->tmax = qd_alloc(e, (size_t)KS * ngp * 4);
    for (size_t i = 0; i < (size_t)KS * ngp; i++) {
        B->tmin[i] = INFINITY;
        B->tmax[i] = -INFINITY;
    }
    for (int lv = 0; lv < 2; lv++)
        for (int j = 0; j < l->nqk; j++) {
            const int g = chan_index(l, j, lv);
            gch[g]++;
            for (int k = 0; k < KS; k++) {
                B->tmin[k * ngp + g] = fminf(B->tmin[k * ngp + g], l->qk_tap[(size_t)k * np + j]);
                B->tmax[k * ngp + g] = fmaxf(B->tmax[k * ngp + g], l->qk_tap[(size_t)k * np + j]);
            }
        }
    for (int g = 0; g < ngp; g++)
        for (int k = 0; k < KS && !gch[g]; k++) B->tmin[k * ngp + g] = B->tmax[k * ngp + g] = 0.0f;
    for (int c = 0; c < H; c++) {
        const int r = B->colid[c];
        if (r < 0) continue; /* (no blocks) */
        for (int lv = 0; lv < 2; lv++)
            for (uint32_t q = l->qk.ptr[c]; q < l->qk.ptr[c + 1]; q++)
                for (int i = 0; i < 16; i++) {
                    const int j = 16 * l->qk.blk[q] + i;
                    const float w = u2f((uint32_t)l->qk.w[(size_t)q * 16 + i] << 16);
                    if (j >= l->nqk || w == 0.0f) continue;
                    const int g = chan_index(l, j, lv);
                    const size_t o = ((size_t)r * 3 + (w > 0.0f ? 0 : 1)) * ngp + g;
                    cnt[o]++;
                    B->cwx[o] = fmaxf(B->cwx[o], w);
                    B->cwn[o] = fminf(B->cwn[o], w);
                    B->wabs[(size_t)r * ngp + g] = fmaxf(B->wabs[(size_t)r * ngp + g], fabsf(w));
                }
        uint32_t q = l->qk.ptr[c];
        for (int g = 0; g <= ng; g++) {
            int b0 = l->qp.nb, b1;
            if (g < ng) group_blocks(l, g, &b0, &b1);
            while (q < l->qk.ptr[c + 1] && l->qk.blk[q] < b0) q++;
            B->gptr[(size_t)r * (ng + 1) + g] = q;
        }
    }
    for (int r = 0; r < rows; r++)
        for (int g = 0; g < ngp; g++) {
            const size_t o = (size_t)r * 3 * ngp + g;
            const int has[3] = {cnt[o] > 0, cnt[o + ngp] > 0, gch[g] - cnt[o] - cnt[o + ngp] > 0};
            B->cwx[o + 2 * ngp] = B->cwn[o + 2 * ngp] = 0.0f;
            const int first = has[0] ? 0 : has[1] ? 1 : 2;
            for (int k = 0; k < 3; k++) B->alias[o + k * ngp] = (int8_t)(has[k] ? k : first);
            for (int k = 0; k < 3; k++) {
                B->cwx[o + k * ngp] = B->cwx[o + B->alias[o + k * ngp] * ngp];
                B->cwn[o + k * ngp] = B->cwn[o + B->alias[o + k * ngp] * ngp];
            }
        }
    free(cnt);
    free(gch);
    B->blo = qd_alloc(e, (size_t)ngp * 4);
    B->bhi = qd_alloc(e, (size_t)ngp * 4);
    B->babs = qd_alloc(e, (size_t)ngp * 4);
    B->slo = qd_alloc(e, (size_t)KS * ngp * 4);
    B->shi = qd_alloc(e, (size_t)KS * ngp * 4);
    B->lo = qd_alloc(e, (size_t)ngp * 4);
    B->hi = qd_alloc(e, (size_t)ngp * 4);
    B->mag = qd_alloc(e, (size_t)ngp * 4);
    B->rr_ok = qd_alloc(e, (size_t)KS);
    memset(B->rr_ok, 1, (size_t)KS);
    B->gdead = qd_alloc(e, (size_t)B->ngw * 8);
    B->pend = qd_alloc(e, (size_t)KS * B->ngw * 8);
}

/* The distinct z rows of a layer's output dims (as CSR rows: the same entries give the same
   z), and each value head's list of them. */
static void z_rows(qd_engine *e, layer_t *l) {
    const int nod = l->nod;
    int cap = 64;
    while (cap < 2 * nod) cap *= 2;
    int32_t *tab = malloc((size_t)cap * 4);
    for (int i = 0; i < cap; i++) tab[i] = -1;
    l->od_zu = qd_alloc(e, (size_t)nod * 4 + 4);
    int32_t *rows = malloc((size_t)nod * 4 + 4);
    l->nzu = 0;
    for (int j = 0; j < nod; j++) {
        const int r = l->od_zrow[j];
        const uint32_t a = l->pool_ptr[r], b = l->pool_ptr[r + 1];
        uint32_t h = 2166136261u;
        for (uint32_t k = a; k < b; k++) h = (h ^ l->pool_ent[k]) * 16777619u;
        int s = h & (cap - 1);
        for (;; s = (s + 1) & (cap - 1)) {
            if (tab[s] < 0) {
                tab[s] = l->nzu;
                rows[l->nzu++] = r;
                break;
            }
            const int r2 = rows[tab[s]];
            const uint32_t a2 = l->pool_ptr[r2], b2 = l->pool_ptr[r2 + 1];
            if (b - a == b2 - a2 && !memcmp(l->pool_ent + a, l->pool_ent + a2, (size_t)(b - a) * 4)) break;
        }
        l->od_zu[j] = tab[s];
    }
    l->zu_row = qd_alloc(e, (size_t)l->nzu * 4 + 4);
    memcpy(l->zu_row, rows, (size_t)l->nzu * 4);
    l->zu_val = qd_alloc(e, (size_t)l->nzu * 4 + 4);
    l->zu_pos = qd_alloc(e, (size_t)l->nzu * 8 + 8);
    int32_t *seen = malloc((size_t)l->nzu * 4 + 4);
    for (int u = 0; u < l->nzu; u++) seen[u] = -1;
    l->vz_ptr = qd_alloc(e, ((size_t)l->nact + 1) * 4);
    l->vz_id = qd_alloc(e, (size_t)nod * 4 + 4);
    int n = 0;
    for (int i = 0; i < l->nact; i++) {
        l->vz_ptr[i] = n;
        for (int j = l->vh[i].o0; j < l->vh[i].o0 + l->vh[i].no; j++)
            if (seen[l->od_zu[j]] != i) {
                seen[l->od_zu[j]] = i;
                l->vz_id[n++] = l->od_zu[j];
            }
    }
    l->vz_ptr[l->nact] = n;
    free(seen);
    free(rows);
    free(tab);
}

static uint64_t *colmap(qd_engine *e, const int32_t *cols, int n) {
    uint64_t *b = qd_alloc(e, (size_t)e->Hp / 8);
    for (int k = 0; k < n; k++) b[cols[k] >> 6] |= 1ull << (cols[k] & 63);
    return b;
}

void qd_derive(qd_engine *e) {
    if (g_tick_ns == 0) {
        mach_timebase_info_data_t tb;
        mach_timebase_info(&tb);
        g_tick_ns = (double)tb.numer / tb.denom;
    }
    int32_t *pre = calloc(2 * e->nl, 4);
    if (!(e->flags & QD_OPEN_GENERIC)) constant_prefix(e, pre);
    e->pre = pre[1];
    uint64_t slow = 0;
    int maxnbw = 1;
    for (int L = 0; L < e->nl; L++) {
        layer_t *l = &e->L[L];
        for (int k = 0; k < 4; k++) {
            l->wr[k][0] = INFINITY;
            l->wr[k][1] = 0.0f;
        }
        if (l->attn) {
            l->a_colb = colmap(e, l->a_cols, l->na_cols);
            l->decay_c = qd_alloc(e, (size_t)l->nact * 4);
            for (int i = 0; i < l->nact; i++)
                l->decay_c[i] = decay_s(0.0f, l->vh[i].dt_bias, l->vh[i].A_log, &slow);
            l->vbound = qd_alloc(e, (size_t)l->nact * 4 + 4);
            for (int i = 0; i < l->nact; i++)
                for (int j = l->vh[i].v0; j < l->vh[i].v0 + l->vh[i].nvc; j++) {
                    double t = 0.0, w = 0.0;
                    for (int k = 0; k < e->KS; k++) t += fabs(l->vch_tap[(size_t)j * e->KS + k]);
                    for (uint32_t q = l->pool_ptr[l->vch_row[j]]; q < l->pool_ptr[l->vch_row[j] + 1]; q++)
                        w += fabs(W_OF(l->pool_ent[q]));
                    l->vbound[i] = fmaxf(l->vbound[i], (float)(t * w * 1.01));
                }
            l->kh_v = qd_alloc(e, ((size_t)e->nk + 1) * 4);
            l->kh_decay = qd_alloc(e, (size_t)e->nk);
            for (int kh = 0, i = 0; kh <= e->nk; kh++) {
                while (i < l->nact && l->vh[i].kh < kh) i++;
                l->kh_v[kh] = i;
            }
            for (int i = 0; i < l->nact; i++)
                l->kh_decay[l->vh[i].kh] |= l->vh[i].a_row >= 0 || l->decay_c[i] != 1.0f;
            split_bcsc(e, &l->qk, l->nqk_pad / 16);
            prefix_alloc(e, &l->qp, l->nqk_pad / 16, pre[2 * L]);
            qk_bounds(e, l);
            l->pool_pre = qd_alloc(e, (size_t)l->npool + 1);
            l->pool_c = qd_alloc(e, ((size_t)l->npool + 1) * 4);
            for (int r = 0; r < l->npool; r++) {
                const uint32_t a = l->pool_ptr[r], b = l->pool_ptr[r + 1];
                l->pool_pre[r] = b > a && (int)I_OF(l->pool_ent[b - 1]) < l->qp.pre;
            }
            if (l->qp.nbw > maxnbw) maxnbw = l->qp.nbw;
            l->ring_tb = qd_alloc(e, (size_t)e->KS * l->qp.nbw * 8);
            z_rows(e, l);
            wrange(l->wr[WR_ATTN_IN], l->qk.w, (size_t)l->qk.ptr[e->H] * 16);
            wrange_ent(l->wr[WR_ATTN_IN], l->pool_ent, l->pool_ptr[l->npool]);
            wrange_ent(l->wr[WR_ATTN_OUT], l->od_ent, l->od_ptr[l->nod]);
        }
        if (l->mlp) {
            l->m_colb = colmap(e, l->m_cols, l->nm_cols);
            split_bcsc(e, &l->gate, l->nu_pad / 16);
            wrange(l->wr[WR_MLP_IN], l->gate.w, (size_t)l->gate.ptr[e->H] * 16);
            wrange_ent(l->wr[WR_MLP_IN], l->up_ent, l->up_ptr[l->nu]);
            wrange_ent(l->wr[WR_MLP_OUT], l->dn_ent, l->dn_ptr[l->nu]);
            prefix_alloc(e, &l->gp, l->nu_pad / 16, pre[2 * L + 1]);
            if (l->gp.nbw > maxnbw) maxnbw = l->gp.nbw;
            int one = !(e->flags & QD_OPEN_GENERIC);
            for (int u = 0; u < l->nu && one; u++) one = l->dn_ptr[u + 1] - l->dn_ptr[u] == 1;
            if (one) {
                l->u1 = qd_alloc(e, (size_t)l->nu * sizeof(unit1_t));
                l->u1_iu = qd_alloc(e, (size_t)l->nu_pad * 4);
                l->u1_wu = qd_alloc(e, (size_t)l->nu_pad * 4);
                l->u1_blk = qd_alloc(e, (size_t)l->gp.nbw * 8);
                l->u1_uni = qd_alloc(e, (size_t)l->gp.nbw * 8);
                for (int b = 0; b < l->nu_pad / 16; b++) l->u1_blk[b >> 6] |= 1ull << (b & 63);
                for (int u = 0; u < l->nu; u++) {
                    const uint32_t eu = l->up_ent[l->up_ptr[u]], ed = l->dn_ent[l->dn_ptr[u]];
                    l->u1[u] = (unit1_t){(uint16_t)I_OF(eu), (uint16_t)I_OF(ed), W_OF(eu), W_OF(ed)};
                    if (l->up_ptr[u + 1] - l->up_ptr[u] != 1) {
                        l->u1_blk[u >> 10] &= ~(1ull << ((u >> 4) & 63));
                        continue;
                    }
                    l->u1_iu[u] = (int32_t)I_OF(eu);
                    l->u1_wu[u] = W_OF(eu);
                }
                for (int b = 0; 16 * b + 16 <= l->nu; b++) { /* (whole blocks) */
                    int same = l->u1_blk[b >> 6] >> (b & 63) & 1;
                    for (int i = 1; i < 16 && same; i++) same = l->u1_iu[16 * b + i] == l->u1_iu[16 * b];
                    if (same) l->u1_uni[b >> 6] |= 1ull << (b & 63);
                }
            }
        }
    }
    free(pre);
    for (int t = 0; t < e->nth; t++) e->tb[t] = qd_alloc(e, (size_t)maxnbw * 8);
    e->st.slow_transcendental += slow;
}

void qd_state_alloc(qd_engine *e) {
    int maxw = 16;
    for (int L = 0; L < e->nl; L++) {
        layer_t *l = &e->L[L];
        if (l->attn) {
            l->S = qd_alloc(e, (size_t)e->nv * e->dk * e->dv * 4);
            l->rows_nz = qd_alloc(e, (size_t)e->nv * (e->dk / 64) * 8);
            l->qk_hist = qd_alloc(e, (size_t)e->KS * l->nqk_pad * 4);
            l->xn_ring = qd_alloc(e, (size_t)e->KS * e->Hp * 4);
            l->ring_nz = qd_alloc(e, (size_t)e->KS * e->Hp * 4);
            if (l->nqk_pad > maxw) maxw = l->nqk_pad;
        }
        if (l->mlp && l->nu_pad > maxw) maxw = l->nu_pad;
    }
    e->x = qd_alloc(e, e->Hp * 4);
    e->dx = qd_alloc(e, e->Hp * 4);
    e->xb = qd_alloc(e, e->Hp / 8);
    e->dxb = qd_alloc(e, e->Hp / 8);
    e->xm = qd_alloc(e, e->Hp * 4);
    e->mnz = qd_alloc(e, e->Hp * 4);
    e->y = qd_alloc(e, (size_t)(e->dv + e->ncode + 16) * 4);
    e->qn = qd_alloc(e, e->nk * 4);
    e->kn = qd_alloc(e, e->nk * 4);
    e->qd = qd_alloc(e, (size_t)e->nk * e->dk * 4);
    e->kd = qd_alloc(e, (size_t)e->nk * e->dk * 4);
    e->qv = qd_alloc(e, (size_t)e->nk * e->dk * 4);
    e->kv = qd_alloc(e, (size_t)e->nk * e->dk * 4);
    for (int t = 0; t < e->nth; t++) e->items[t] = qd_alloc(e, (size_t)maxw * sizeof(item_t));
    int maxu = 16;
    for (int L = 0; L < e->nl; L++)
        if (e->L[L].mlp && e->L[L].nu_pad > maxu) maxu = e->L[L].nu_pad;
    e->act = qd_alloc(e, (size_t)maxu * 4);
    e->amask = qd_alloc(e, (size_t)maxu / 16 * 2);
    for (int t = 0; t < e->nth; t++) e->mblk[t] = qd_alloc(e, (size_t)maxu / 16 * 4);
}

void qd_state_reset(qd_engine *e) {
    for (int L = 0; L < e->nl; L++) {
        layer_t *l = &e->L[L];
        l->gp.ok = 0; /* (recomputed at the next step) */
        if (!l->attn) continue;
        l->qp.ok = 0;
        if (l->qb.on) { /* (the ring is all 0) */
            memset(l->qb.slo, 0, (size_t)e->KS * l->qb.ngp * 4);
            memset(l->qb.shi, 0, (size_t)e->KS * l->qb.ngp * 4);
            memset(l->qb.pend, 0, (size_t)e->KS * l->qb.ngw * 8);
            memset(l->qb.rr_ok, 1, (size_t)e->KS);
        }
        memset(l->zu_pos, 0, (size_t)l->nzu * 8); /* (steps count from 1) */
        memset(l->S, 0, (size_t)e->nv * e->dk * e->dv * 4);
        memset(l->rows_nz, 0, (size_t)e->nv * (e->dk / 64) * 8);
        memset(l->qk_hist, 0, (size_t)e->KS * l->nqk_pad * 4);
        memset(l->xn_ring, 0, (size_t)e->KS * e->Hp * 4);
        memset(l->ring_n, 0, sizeof l->ring_n);
        memset(l->xn_hi, 0, sizeof l->xn_hi);
    }
    memset(e->x, 0, e->Hp * 4);
    memset(e->dx, 0, e->Hp * 4);
    memset(e->xb, 0, e->Hp / 8);
    memset(e->dxb, 0, e->Hp / 8);
    memset(e->xm, 0, e->Hp * 4);
    e->mnnz = 0;
    e->pos = 0;
    e->inv_ok = 0;
}
