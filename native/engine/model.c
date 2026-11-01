/* Opening a checkpoint: read config.json, map model.safetensors, discover the nonzero structure
   and build the compact layouts (internal.h), or load them from the cache keyed by the file's
   sha256. Also the small API calls (close, reset, rows, options, info, stats). */
#include <CommonCrypto/CommonDigest.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <libgen.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "internal.h"
#include "json.h"

static _Thread_local char g_err[512];

const char *qd_engine_last_error(void) { return g_err; }

static void fail(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_err, sizeof g_err, fmt, ap);
    va_end(ap);
}

void *qd_alloc(qd_engine *e, size_t n) {
    void *p = NULL;
    if (posix_memalign(&p, 64, n ? n : 64)) return NULL;
    memset(p, 0, n ? n : 64);
    if (e->nallocs == e->capallocs) {
        e->capallocs = e->capallocs ? 2 * e->capallocs : 256;
        e->allocs = realloc(e->allocs, e->capallocs * sizeof(void *));
    }
    e->allocs[e->nallocs++] = p;
    return p;
}

/* --- safetensors ---------------------------------------------------------------------------- */

typedef struct {
    const char *path;
    int fd;
    const uint8_t *map;
    size_t size, base;
    jv *hdr;
} st_file;

/* A regular file opened for reading, and its stat; -1 (and the error) if it cannot be. */
static int open_file(const char *path, struct stat *sb) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return fail("cannot open %s: %s", path, strerror(errno)), -1;
    if (fstat(fd, sb)) fail("cannot stat %s: %s", path, strerror(errno));
    else if (!S_ISREG(sb->st_mode)) fail("cannot read %s: not a regular file", path);
    else return fd;
    close(fd);
    return -1;
}

static int st_open(st_file *f, const char *path) {
    f->path = path;
    struct stat sb;
    if ((f->fd = open_file(path, &sb)) < 0) return -1;
    f->size = sb.st_size;
    if (f->size < 8) return fail("%s: truncated (%zu bytes)", path, f->size), -1;
    f->map = mmap(NULL, f->size, PROT_READ, MAP_PRIVATE, f->fd, 0);
    if (f->map == MAP_FAILED) {
        f->map = NULL;
        return fail("cannot map %s: %s", path, strerror(errno)), -1;
    }
    uint64_t n;
    memcpy(&n, f->map, 8);
    /* (safetensors caps the header at 100 MB: a larger length is not one) */
    if (n > f->size - 8 || n > 100 << 20)
        return fail("%s: truncated or not safetensors (a %llu-byte header in %zu bytes)", path,
                    (unsigned long long)n, f->size), -1;
    f->base = 8 + n;
    /* (parsed from a NUL-terminated copy: a malformed header's last literal or number must
       not be read past its end, which may be the end of the mapping) */
    char *h = malloc(n + 1);
    memcpy(h, f->map + 8, n);
    h[n] = 0;
    f->hdr = json_parse(h, n);
    free(h);
    if (!f->hdr || f->hdr->type != J_OBJ) return fail("bad safetensors header in %s", path), -1;
    return 0;
}

static void st_close(st_file *f) {
    if (f->hdr) json_free(f->hdr);
    if (f->map) munmap((void *)f->map, f->size);
    if (f->fd >= 0) close(f->fd);
}

/* A bf16 tensor of shape [d0, d1, d2] (-1: any size; missing dims are 1), the shape in shape,
   its data inside the file. NULL (and the error) if absent or not that. */
static const uint16_t *st_get(st_file *f, const char *name, int64_t shape[3], int64_t d0,
                              int64_t d1, int64_t d2) {
    jv *t = json_get(f->hdr, name);
    if (!t) return fail("%s: missing tensor %s", f->path, name), NULL;
    jv *dt = json_get(t, "dtype"), *sh = json_get(t, "shape"), *off = json_get(t, "data_offsets");
    if (!dt || dt->type != J_STR || strcmp(dt->str, "BF16"))
        return fail("%s: %s is not BF16", f->path, name), NULL;
    int ok = sh && sh->type == J_ARR && sh->n <= 3 && off && off->type == J_ARR && off->n == 2 &&
             off->items[0].type == J_NUM && off->items[1].type == J_NUM;
    int64_t cnt = 1;
    for (int i = 0; i < 3 && ok; i++) {
        double d = i < sh->n ? sh->items[i].num : 1;
        ok = (i >= sh->n || sh->items[i].type == J_NUM) && d >= 0 && d <= 1e15 && d == floor(d);
        shape[i] = ok ? (int64_t)d : 0;
        ok = ok && !__builtin_mul_overflow(cnt, shape[i], &cnt);
    }
    if (!ok) return fail("%s: %s has a bad shape or data_offsets", f->path, name), NULL;
    if ((d0 >= 0 && shape[0] != d0) || (d1 >= 0 && shape[1] != d1) || (d2 >= 0 && shape[2] != d2))
        return fail("%s: %s has shape [%lld, %lld, %lld], expected [%lld, %lld, %lld] (-1: any)",
                    f->path, name, (long long)shape[0], (long long)shape[1], (long long)shape[2],
                    (long long)d0, (long long)d1, (long long)d2), NULL;
    /* the data: [begin, end) of the bytes after the header, 2 per element */
    const double b = off->items[0].num, end = off->items[1].num, room = (double)(f->size - f->base);
    if (!(b >= 0 && b <= end && b == floor(b) && end == floor(end)) || end - b != 2.0 * cnt)
        return fail("%s: %s's data_offsets do not hold its shape", f->path, name), NULL;
    if (end > room)
        return fail("%s: %s ends past the end of the file (truncated?)", f->path, name), NULL;
    return (const uint16_t *)(f->map + f->base + (size_t)b);
}

static inline float bf(uint16_t b) { return u2f((uint32_t)b << 16); }
static inline int nzb(uint16_t b) { return (b & 0x7FFF) != 0; } /* -0 counts as zero */

static int row_nz(const uint16_t *w, int64_t n) {
    for (int64_t i = 0; i < n; i++)
        if (nzb(w[i])) return 1;
    return 0;
}

/* --- building ------------------------------------------------------------------------------- */

/* A growable array of u32 / i32 / f32 values. */
typedef struct { uint32_t *a; int64_t n, cap; } vec_t;

static void vpush(vec_t *v, uint32_t x) {
    if (v->n == v->cap) {
        v->cap = v->cap ? 2 * v->cap : 1024;
        v->a = realloc(v->a, v->cap * 4);
    }
    v->a[v->n++] = x;
}

static void *vfreeze(qd_engine *e, vec_t *v) {
    void *p = qd_alloc(e, v->n * 4);
    if (v->n) memcpy(p, v->a, v->n * 4);
    free(v->a);
    memset(v, 0, sizeof *v);
    return p;
}

/* Append matrix row w[0..C) as a CSR row: entries in increasing column order. */
static void csr_row(vec_t *ent, const uint16_t *w, int64_t C) {
    for (int64_t c = 0; c < C; c++)
        if (nzb(w[c])) vpush(ent, (uint32_t)w[c] << 16 | (uint32_t)c);
}

/* Blocked CSC of the rows `rows` (local row j, block j / 16) of an [R, C] matrix. */
static void build_bcsc(qd_engine *e, const uint16_t *W, int64_t C, const int32_t *rows, int n,
                       bcsc_t *m, int64_t *nnz) {
    const int nb = (n + 15) / 16;
    uint32_t *ptr = qd_alloc(e, (C + 1) * 4);
    uint8_t *hit = malloc(C);
    for (int pass = 0; pass < 2; pass++) {
        uint32_t *fill = pass ? calloc(C, 4) : NULL;
        for (int b = 0; b < nb; b++) {
            memset(hit, 0, C);
            for (int j = 16 * b; j < 16 * b + 16 && j < n; j++) {
                const uint16_t *w = W + (int64_t)rows[j] * C;
                for (int64_t c = 0; c < C; c++) hit[c] |= nzb(w[c]);
            }
            for (int64_t c = 0; c < C; c++) {
                if (!hit[c]) continue;
                if (!pass) {
                    ptr[c + 1]++;
                    continue;
                }
                uint32_t k = ptr[c] + fill[c]++;
                m->blk[k] = (uint16_t)b;
                for (int i = 0; i < 16; i++) {
                    int j = 16 * b + i;
                    uint16_t v = j < n ? W[(int64_t)rows[j] * C + c] : 0;
                    m->w[(size_t)k * 16 + i] = nzb(v) ? v : 0;
                    *nnz += nzb(v);
                }
            }
        }
        if (!pass) {
            for (int64_t c = 0; c < C; c++) ptr[c + 1] += ptr[c];
            m->blk = qd_alloc(e, (size_t)ptr[C] * 2);
            m->w = qd_alloc(e, (size_t)ptr[C] * 32);
        }
        free(fill);
    }
    free(hit);
    m->ptr = ptr;
}

/* The sorted union of the input dims of some blocked CSC matrices and CSR rows. */
static int32_t *used_cols(qd_engine *e, const bcsc_t *a, const bcsc_t *b, const uint32_t *pptr,
                          const uint32_t *pent, int npool, int32_t *n) {
    uint8_t *u = calloc(e->H, 1);
    for (int c = 0; c < e->H; c++) {
        if (a && a->ptr[c + 1] > a->ptr[c]) u[c] = 1;
        if (b && b->ptr[c + 1] > b->ptr[c]) u[c] = 1;
    }
    for (int r = 0; r < npool; r++)
        for (uint32_t k = pptr[r]; k < pptr[r + 1]; k++) u[I_OF(pent[k])] = 1;
    int32_t *out = qd_alloc(e, e->H * 4);
    *n = 0;
    for (int c = 0; c < e->H; c++)
        if (u[c]) out[(*n)++] = c;
    free(u);
    return out;
}

/* The sorted residual dims written by CSC entries. */
static int32_t *out_dims(qd_engine *e, const uint32_t *ptr, const uint32_t *ent, int n, int32_t *cnt) {
    uint8_t *u = calloc(e->H, 1);
    for (uint32_t k = 0; k < ptr[n]; k++) u[I_OF(ent[k])] = 1;
    int32_t *out = qd_alloc(e, e->H * 4);
    *cnt = 0;
    for (int c = 0; c < e->H; c++)
        if (u[c]) out[(*cnt)++] = c;
    free(u);
    return out;
}

static int build_layer(qd_engine *e, st_file *f, const char *pre, int L, int full_attn,
                       int64_t *nnz) {
    layer_t *ly = &e->L[L];
    const int H = e->H, Hp = e->Hp, nk = e->nk, nv = e->nv, dk = e->dk, dv = e->dv, KS = e->KS;
    char nm[256];
    int64_t sh[3];
#define T(name, d0, d1, d2) \
    (snprintf(nm, sizeof nm, "%slayers.%d.%s", pre, L, name), st_get(f, nm, sh, d0, d1, d2))
    const uint16_t *w;
    ly->g_in = qd_alloc(e, Hp * 4);
    ly->g_post = qd_alloc(e, Hp * 4);
    if (!(w = T("input_layernorm.weight", H, 1, 1))) return -1;
    for (int i = 0; i < H; i++) ly->g_in[i] = 1.0f + bf(w[i]);
    if (!(w = T("post_attention_layernorm.weight", H, 1, 1))) return -1;
    for (int i = 0; i < H; i++) ly->g_post[i] = 1.0f + bf(w[i]);

    if (full_attn) {
        /* supported only as all zero: then o_proj makes the token mixer add exactly 0 */
        if (!(w = T("self_attn.o_proj.weight", -1, -1, -1))) return -1;
        if (row_nz(w, sh[0] * sh[1] * sh[2]))
            return fail("layer %d: full attention with nonzero weights is unsupported", L), -1;
    } else {
        const int64_t C = 2LL * nk * dk + (int64_t)nv * dv, VD = (int64_t)nv * dv;
        const uint16_t *qkv = T("linear_attn.in_proj_qkv.weight", C, H, 1);
        const uint16_t *conv = T("linear_attn.conv1d.weight", C, 1, KS);
        const uint16_t *zw = T("linear_attn.in_proj_z.weight", VD, H, 1);
        const uint16_t *aw = T("linear_attn.in_proj_a.weight", nv, H, 1);
        const uint16_t *bw = T("linear_attn.in_proj_b.weight", nv, H, 1);
        const uint16_t *ow = T("linear_attn.out_proj.weight", H, VD, 1);
        const uint16_t *dtb = T("linear_attn.dt_bias", nv, 1, 1);
        const uint16_t *alog = T("linear_attn.A_log", nv, 1, 1);
        const uint16_t *nw = T("linear_attn.norm.weight", dv, 1, 1);
        if (!qkv || !conv || !zw || !aw || !bw || !ow || !dtb || !alog || !nw) return -1;
        ly->norm_w = qd_alloc(e, dv * 4);
        for (int i = 0; i < dv; i++) ly->norm_w[i] = bf(nw[i]);
        /* out_proj columns (value head h, dim d) -> residual dims; z rows */
        uint32_t *ocnt = calloc(VD + 1, 4);
        for (int i = 0; i < H; i++)
            for (int64_t c = 0; c < VD; c++)
                if (nzb(ow[i * VD + c])) ocnt[c]++;
        uint8_t *znz = calloc(VD, 1);
        for (int64_t c = 0; c < VD; c++) znz[c] = row_nz(zw + c * H, H);
        uint8_t *chnz = calloc(C, 1);
        for (int64_t c = 0; c < C; c++)
            chnz[c] = row_nz(qkv + c * H, H) && row_nz(conv + c * KS, KS);
        /* active value heads: some output dim with both an out_proj column and a z row */
        int32_t *act = calloc(nv, 4);
        int nact = 0;
        for (int h = 0; h < nv; h++) {
            int ok = 0;
            for (int d = 0; d < dv && !ok; d++) ok = ocnt[(int64_t)h * dv + d] && znz[(int64_t)h * dv + d];
            if (ok) act[nact++] = h;
        }
        ly->nact = nact;
        ly->attn = nact > 0;
        if (ly->attn) {
            /* eager channels: queries and keys of the key heads the active value heads read;
               those that read only the current tap first */
            uint8_t *khu = calloc(nk, 1);
            for (int i = 0; i < nact; i++) khu[act[i] / e->r] = 1;
            vec_t chs = {0};
            for (int one = 1; one >= 0; one--)
                for (int kind = 0; kind < 2; kind++)
                    for (int kh = 0; kh < nk; kh++)
                        for (int d = 0; d < dk; d++) {
                            int64_t c = (int64_t)kind * nk * dk + (int64_t)kh * dk + d;
                            if (!khu[kh] || !chnz[c] || row_nz(conv + c * KS, KS - 1) == one) continue;
                            vpush(&chs, (uint32_t)c);
                            ly->nqk1 += one;
                        }
            free(khu);
            ly->nqk = (int32_t)chs.n;
            ly->nqk_pad = (ly->nqk + 15) & ~15;
            ly->qk_id = qd_alloc(e, ly->nqk_pad * 4);
            ly->qk_tap = qd_alloc(e, (size_t)KS * ly->nqk_pad * 4);
            for (int j = 0; j < ly->nqk_pad; j++) ly->qk_id[j] = 0xFFFFFFFFu;
            for (int j = 0; j < ly->nqk; j++) {
                uint32_t c = chs.a[j], kind = c >= (uint32_t)(nk * dk);
                uint32_t rel = c - kind * nk * dk;
                ly->qk_id[j] = kind << 31 | (rel / dk) << 16 | (rel % dk);
                for (int t = 0; t < KS; t++) ly->qk_tap[(size_t)t * ly->nqk_pad + j] = bf(conv[c * KS + t]);
            }
            build_bcsc(e, qkv, H, (int32_t *)chs.a, ly->nqk, &ly->qk, nnz);
            free(chs.a);
            /* the pool of lazily computed rows, and the value heads */
            vec_t pptr = {0}, pent = {0};
            vpush(&pptr, 0);
#define POOL_ROW(wrow) (csr_row(&pent, (wrow), H), vpush(&pptr, (uint32_t)pent.n), (int32_t)pptr.n - 2)
            ly->vh = qd_alloc(e, nact * sizeof(vhead_t));
            vec_t vdim = {0}, vrow = {0}, vtap = {0}, odim = {0}, ozrow = {0};
            for (int i = 0; i < nact; i++) {
                vhead_t *v = &ly->vh[i];
                int h = act[i];
                v->h = h;
                v->kh = h / e->r;
                v->a_row = row_nz(aw + (int64_t)h * H, H) ? POOL_ROW(aw + (int64_t)h * H) : -1;
                v->b_row = row_nz(bw + (int64_t)h * H, H) ? POOL_ROW(bw + (int64_t)h * H) : -1;
                v->dt_bias = bf(dtb[h]);
                v->A_log = bf(alog[h]);
                v->v0 = (int32_t)vdim.n;
                for (int d = 0; d < dv; d++) {
                    int64_t c = 2LL * nk * dk + (int64_t)h * dv + d;
                    if (!chnz[c]) continue;
                    vpush(&vdim, d);
                    vpush(&vrow, (uint32_t)POOL_ROW(qkv + c * H));
                    for (int t = 0; t < KS; t++) vpush(&vtap, f2u(bf(conv[c * KS + t])));
                }
                v->nvc = (int32_t)vdim.n - v->v0;
                v->o0 = (int32_t)odim.n;
                for (int d = 0; d < dv; d++) {
                    int64_t c = (int64_t)h * dv + d;
                    if (!ocnt[c] || !znz[c]) continue;
                    vpush(&odim, d);
                    vpush(&ozrow, (uint32_t)POOL_ROW(zw + c * H));
                }
                v->no = (int32_t)odim.n - v->o0;
            }
#undef POOL_ROW
            ly->nvch = (int32_t)vdim.n;
            ly->nod = (int32_t)odim.n;
            ly->vch_dim = vfreeze(e, &vdim);
            ly->vch_row = vfreeze(e, &vrow);
            ly->vch_tap = vfreeze(e, &vtap);
            ly->od_dim = vfreeze(e, &odim);
            ly->od_zrow = vfreeze(e, &ozrow);
            ly->npool = (int32_t)pptr.n - 1;
            *nnz += pent.n;
            ly->pool_ptr = vfreeze(e, &pptr);
            ly->pool_ent = vfreeze(e, &pent);
            /* out_proj columns of the output dims: CSC entries (weight, residual dim) */
            int32_t *colj = malloc(VD * 4);
            for (int64_t c = 0; c < VD; c++) colj[c] = -1;
            for (int i = 0; i < nact; i++)
                for (int j = ly->vh[i].o0; j < ly->vh[i].o0 + ly->vh[i].no; j++)
                    colj[(int64_t)ly->vh[i].h * dv + ly->od_dim[j]] = j;
            ly->od_ptr = qd_alloc(e, (ly->nod + 1) * 4);
            for (int64_t c = 0; c < VD; c++)
                if (colj[c] >= 0) ly->od_ptr[colj[c] + 1] = ocnt[c];
            for (int j = 0; j < ly->nod; j++) ly->od_ptr[j + 1] += ly->od_ptr[j];
            ly->od_ent = qd_alloc(e, (size_t)ly->od_ptr[ly->nod] * 4);
            uint32_t *fill = calloc(ly->nod + 1, 4);
            for (int i = 0; i < H; i++)
                for (int64_t c = 0; c < VD; c++) {
                    int j = colj[c];
                    if (j >= 0 && nzb(ow[i * VD + c]))
                        ly->od_ent[ly->od_ptr[j] + fill[j]++] = (uint32_t)ow[i * VD + c] << 16 | i;
                }
            *nnz += ly->od_ptr[ly->nod];
            free(fill);
            free(colj);
            ly->a_cols = used_cols(e, &ly->qk, NULL, ly->pool_ptr, ly->pool_ent, ly->npool, &ly->na_cols);
            ly->a_out = out_dims(e, ly->od_ptr, ly->od_ent, ly->nod, &ly->na_out);
        }
        free(ocnt);
        free(znz);
        free(chnz);
        free(act);
    }
    /* MLP: units whose gate row, up row and down column are all nonzero */
    const uint16_t *gw = T("mlp.gate_proj.weight", -1, H, 1);
    if (!gw) return -1;
    const int64_t I = sh[0];
    const uint16_t *uw = T("mlp.up_proj.weight", I, H, 1), *dw = T("mlp.down_proj.weight", H, I, 1);
    if (!uw || !dw) return -1;
    uint8_t *dnz = calloc(I, 1);
    for (int i = 0; i < H; i++)
        for (int64_t u = 0; u < I; u++)
            if (nzb(dw[i * I + u])) dnz[u] = 1;
    vec_t units = {0};
    for (int64_t u = 0; u < I; u++)
        if (dnz[u] && row_nz(gw + u * H, H) && row_nz(uw + u * H, H)) vpush(&units, (uint32_t)u);
    free(dnz);
    ly->nu = (int32_t)units.n;
    ly->nu_pad = (ly->nu + 15) & ~15;
    ly->mlp = ly->nu > 0;
    if (ly->mlp) {
        if (ly->nu > 16 * 65536) return fail("layer %d: too many MLP units", L), -1;
        build_bcsc(e, gw, H, (int32_t *)units.a, ly->nu, &ly->gate, nnz);
        /* up rows: CSR (few nonzeros each; only the units whose gate is live read them) */
        vec_t uptr = {0}, uent = {0};
        vpush(&uptr, 0);
        for (int j = 0; j < ly->nu; j++) {
            csr_row(&uent, uw + (int64_t)units.a[j] * H, H);
            vpush(&uptr, (uint32_t)uent.n);
        }
        *nnz += uent.n;
        ly->up_ptr = vfreeze(e, &uptr);
        ly->up_ent = vfreeze(e, &uent);
        /* down columns, scanned row-major (entries per unit in increasing residual dim) */
        int32_t *uj = malloc(I * 4);
        for (int64_t u = 0; u < I; u++) uj[u] = -1;
        for (int j = 0; j < ly->nu; j++) uj[units.a[j]] = j;
        ly->dn_ptr = qd_alloc(e, (ly->nu + 1) * 4);
        for (int i = 0; i < H; i++)
            for (int64_t u = 0; u < I; u++)
                if (uj[u] >= 0 && nzb(dw[i * I + u])) ly->dn_ptr[uj[u] + 1]++;
        for (int j = 0; j < ly->nu; j++) ly->dn_ptr[j + 1] += ly->dn_ptr[j];
        ly->dn_ent = qd_alloc(e, (size_t)ly->dn_ptr[ly->nu] * 4);
        uint32_t *fill = calloc(ly->nu, 4);
        for (int i = 0; i < H; i++)
            for (int64_t u = 0; u < I; u++)
                if (uj[u] >= 0 && nzb(dw[i * I + u]))
                    ly->dn_ent[ly->dn_ptr[uj[u]] + fill[uj[u]]++] = (uint32_t)dw[i * I + u] << 16 | i;
        *nnz += ly->dn_ptr[ly->nu];
        free(fill);
        free(uj);
        ly->m_cols = used_cols(e, &ly->gate, NULL, ly->up_ptr, ly->up_ent, ly->nu, &ly->nm_cols);
        ly->m_out = out_dims(e, ly->dn_ptr, ly->dn_ent, ly->nu, &ly->nm_out);
    }
    free(units.a);
#undef T
    return 0;
}

/* The tied (or separate) head restricted to the code dims, and the sign table: the head's argmax
   is the row whose code signs equal y's when every row is all zero or all +-A on the code dims
   and |y| is the same on all of them (step.c checks the latter per step). */
static void build_head(qd_engine *e, const uint16_t *E) {
    const int64_t H = e->H, V = e->V, n = e->ncode;
    e->ecode = qd_alloc(e, (size_t)V * n * 2);
    uint32_t amp = 0;
    int ok = n > 0 && n <= 24;
    for (int64_t t = 0; t < V; t++) {
        int zeros = 0;
        for (int i = 0; i < n; i++) {
            uint16_t b = E[t * H + e->code[i]];
            e->ecode[t * n + i] = b;
            if (!nzb(b)) zeros++;
            else if (!amp) amp = b & 0x7FFF;
            else if ((b & 0x7FFF) != amp) ok = 0;
        }
        if (zeros && zeros != n) ok = 0;
    }
    e->sign_ok = ok && amp;
    if (!e->sign_ok) return;
    e->sign_tab = qd_alloc(e, ((size_t)1 << n) * 4);
    for (size_t i = 0; i < (size_t)1 << n; i++) e->sign_tab[i] = -1;
    for (int64_t t = V - 1; t >= 0; t--) { /* (duplicates: the first token wins, as argmax) */
        if (!nzb(e->ecode[t * n])) continue;
        uint32_t bits = 0;
        for (int i = 0; i < n; i++) bits |= (uint32_t)!(e->ecode[t * n + i] >> 15) << i;
        e->sign_tab[bits] = (int32_t)t;
    }
}

static int build(qd_engine *e, st_file *f, jv *cfg, int64_t *nnz) {
    /* the tensor prefix: whatever precedes "embed_tokens.weight" */
    char pre[128] = "";
    for (int i = 0; i < f->hdr->n; i++) {
        const char *k = f->hdr->keys[i], *p = strstr(k, "embed_tokens.weight");
        if (p && !p[19]) {
            snprintf(pre, sizeof pre, "%.*s", (int)(p - k), k);
            break;
        }
    }
    char nm[256];
    int64_t sh[3];
    snprintf(nm, sizeof nm, "%sembed_tokens.weight", pre);
    const uint16_t *E = st_get(f, nm, sh, e->V, e->H, 1);
    if (!E) return -1;
    jv *types = json_get(cfg, "layer_types");
    for (int L = 0; L < e->nl; L++) {
        const char *t = types->items[L].str;
        int full = !strcmp(t, "full_attention");
        if (!full && strcmp(t, "linear_attention")) return fail("layer type %s", t), -1;
        if (build_layer(e, f, pre, L, full, nnz)) return -1;
    }
    /* the embedding's nonzero columns */
    const int64_t V = e->V, H = e->H;
    uint8_t *cnz = calloc(H, 1);
    for (int64_t t = 0; t < V; t++)
        for (int64_t i = 0; i < H; i++)
            if (nzb(E[t * H + i])) cnz[i] = 1;
    for (int i = 0; i < H; i++) e->necol += cnz[i];
    e->ecol = qd_alloc(e, e->necol * 4);
    for (int i = 0, j = 0; i < H; i++)
        if (cnz[i]) e->ecol[j++] = i;
    free(cnz);
    e->etab = qd_alloc(e, (size_t)V * e->necol * 2);
    for (int64_t t = 0; t < V; t++)
        for (int j = 0; j < e->necol; j++) e->etab[t * e->necol + j] = E[t * H + e->ecol[j]];
    /* final norm and head */
    snprintf(nm, sizeof nm, "%snorm.weight", pre);
    const uint16_t *nw = st_get(f, nm, sh, H, 1, 1);
    if (!nw) return -1;
    e->gain = qd_alloc(e, e->Hp * 4);
    for (int i = 0; i < H; i++) e->gain[i] = 1.0f + bf(nw[i]);
    for (int i = 0; i < H; i++) e->ncode += e->gain[i] != 0;
    e->code = qd_alloc(e, e->ncode * 4);
    for (int i = 0, j = 0; i < H; i++)
        if (e->gain[i] != 0) e->code[j++] = i;
    const uint16_t *head = E;
    if (!json_num(cfg, "tie_word_embeddings", 0)) {
        head = st_get(f, "lm_head.weight", sh, V, H, 1);
        if (!head) return -1;
    }
    build_head(e, head);
    return 0;
}

/* --- the cache --------------------------------------------------------------------------------- */

typedef struct {
    FILE *fp;
    int load; /* 0: save */
    int err;
    qd_engine *e;
    size_t left; /* (load) the bytes of the file not read yet */
} ser_t;

static void ser_raw(ser_t *s, void *p, size_t n) {
    if (s->err || !n) return;
    if ((s->load ? fread(p, 1, n, s->fp) : fwrite(p, 1, n, s->fp)) != n) s->err = 1;
    if (s->load && !s->err) s->left -= n;
}
#define SER_I32(s, x) ser_raw(s, &(x), 4)
#define SER_F32(s, x) ser_raw(s, &(x), 4)
static void ser_arr(ser_t *s, void **p, size_t n) {
    uint64_t m = n;
    ser_raw(s, &m, 8);
    if (s->err) return;
    /* (a length the rest of the file cannot hold is never allocated) */
    if (m != n || (s->load && (n > s->left || !(*p = qd_alloc(s->e, n))))) {
        s->err = 1;
        return;
    }
    ser_raw(s, *p, n);
}
#define SER_ARR(s, p, n) ser_arr(s, (void **)&(p), (size_t)(n))

static void ser_bcsc(ser_t *s, qd_engine *e, bcsc_t *m) {
    SER_ARR(s, m->ptr, (e->H + 1) * 4);
    if (s->err) return;
    SER_ARR(s, m->blk, (size_t)m->ptr[e->H] * 2);
    SER_ARR(s, m->w, (size_t)m->ptr[e->H] * 32);
}

static void ser_model(ser_t *s, qd_engine *e) {
    const int Hp = e->Hp, KS = e->KS;
    for (int L = 0; L < e->nl; L++) {
        layer_t *l = &e->L[L];
        int32_t *ints[] = {&l->attn, &l->nqk, &l->nqk_pad, &l->nqk1, &l->nact, &l->nvch, &l->nod,
                           &l->npool, &l->na_cols, &l->na_out, &l->mlp, &l->nu, &l->nu_pad,
                           &l->nm_cols, &l->nm_out};
        for (size_t i = 0; i < sizeof ints / sizeof *ints; i++) ser_raw(s, ints[i], 4);
        SER_ARR(s, l->g_in, Hp * 4);
        SER_ARR(s, l->g_post, Hp * 4);
        if (l->attn) {
            ser_bcsc(s, e, &l->qk);
            SER_ARR(s, l->qk_tap, (size_t)KS * l->nqk_pad * 4);
            SER_ARR(s, l->qk_id, l->nqk_pad * 4);
            SER_ARR(s, l->a_cols, e->H * 4);
            SER_ARR(s, l->a_out, e->H * 4);
            SER_ARR(s, l->vh, l->nact * sizeof(vhead_t));
            SER_ARR(s, l->vch_dim, l->nvch * 4);
            SER_ARR(s, l->vch_row, l->nvch * 4);
            SER_ARR(s, l->vch_tap, (size_t)l->nvch * KS * 4);
            SER_ARR(s, l->od_dim, l->nod * 4);
            SER_ARR(s, l->od_zrow, l->nod * 4);
            SER_ARR(s, l->od_ptr, (l->nod + 1) * 4);
            if (s->err) return;
            SER_ARR(s, l->od_ent, (size_t)l->od_ptr[l->nod] * 4);
            SER_ARR(s, l->norm_w, e->dv * 4);
            SER_ARR(s, l->pool_ptr, (l->npool + 1) * 4);
            if (s->err) return;
            SER_ARR(s, l->pool_ent, (size_t)l->pool_ptr[l->npool] * 4);
        }
        if (l->mlp) {
            ser_bcsc(s, e, &l->gate);
            SER_ARR(s, l->up_ptr, (l->nu + 1) * 4);
            if (s->err) return;
            SER_ARR(s, l->up_ent, (size_t)l->up_ptr[l->nu] * 4);
            SER_ARR(s, l->m_cols, e->H * 4);
            SER_ARR(s, l->m_out, e->H * 4);
            SER_ARR(s, l->dn_ptr, (l->nu + 1) * 4);
            if (s->err) return;
            SER_ARR(s, l->dn_ent, (size_t)l->dn_ptr[l->nu] * 4);
        }
        if (s->err) return;
    }
    SER_I32(s, e->necol);
    SER_ARR(s, e->ecol, e->necol * 4);
    SER_ARR(s, e->etab, (size_t)e->V * e->necol * 2);
    SER_ARR(s, e->gain, Hp * 4);
    SER_I32(s, e->ncode);
    SER_ARR(s, e->code, e->ncode * 4);
    SER_ARR(s, e->ecode, (size_t)e->V * e->ncode * 2);
    SER_I32(s, e->sign_ok);
    if (e->sign_ok) SER_ARR(s, e->sign_tab, ((size_t)1 << e->ncode) * 4);
    ser_raw(s, &e->info.nnz_compact, 8);
}

static const char MAGIC[8] = "QDENGC\0\0";

/* The header pins the shapes (so a cache built for other shapes is never misread). */
static void ser_header(ser_t *s, qd_engine *e) {
    char m[8];
    memcpy(m, MAGIC, 8);
    ser_raw(s, m, 8);
    if (s->load && memcmp(m, MAGIC, 8)) s->err = 1;
    int32_t v[11] = {QD_LAYOUT_VERSION, e->H, e->Hp, e->nl, e->V, e->nk, e->nv, e->dk, e->dv, e->KS,
                     (int32_t)sizeof(vhead_t)};
    int32_t w[11];
    memcpy(w, v, sizeof v);
    ser_raw(s, w, sizeof w);
    if (s->load && memcmp(v, w, sizeof v)) s->err = 1;
}

static void hex(const uint8_t *d, char *out) {
    for (int i = 0; i < 32; i++) snprintf(out + 2 * i, 3, "%02x", d[i]);
}

static int is_sha(const char *s) {
    for (int i = 0; i < 64; i++)
        if (!((s[i] >= '0' && s[i] <= '9') || (s[i] >= 'a' && s[i] <= 'f'))) return 0;
    return 1;
}

/* sha256 of the file, memoized in <cache>/index.tsv by (path, size, mtime, inode). -1 (and the
   error) if it cannot be read, memoized or not. */
static int file_sha(const char *cache, const char *path, char out[65]) {
    struct stat sb;
    int fd = open_file(path, &sb);
    if (fd < 0) return -1;
    char real[PATH_MAX];
    if (!realpath(path, real)) snprintf(real, sizeof real, "%s", path);
    char key[PATH_MAX + 128];
    snprintf(key, sizeof key, "%s\t%lld\t%ld.%09ld\t%llu", real, (long long)sb.st_size,
             (long)sb.st_mtimespec.tv_sec, sb.st_mtimespec.tv_nsec, (unsigned long long)sb.st_ino);
    char idx[PATH_MAX];
    if (cache) {
        snprintf(idx, sizeof idx, "%s/index.tsv", cache);
        FILE *fp = fopen(idx, "r");
        if (fp) {
            char line[PATH_MAX + 256];
            size_t kl = strlen(key);
            /* (the memoized value names the cache file: 64 hex digits or ignored) */
            while (fgets(line, sizeof line, fp))
                if (!strncmp(line, key, kl) && line[kl] == '\t' && strlen(line + kl + 1) >= 64 &&
                    is_sha(line + kl + 1)) {
                    memcpy(out, line + kl + 1, 64);
                    out[64] = 0;
                    fclose(fp);
                    close(fd);
                    return 0;
                }
            fclose(fp);
        }
    }
    CC_SHA256_CTX c;
    CC_SHA256_Init(&c);
    const size_t n = sb.st_size;
    const uint8_t *m = n ? mmap(NULL, n, PROT_READ, MAP_PRIVATE, fd, 0) : NULL;
    if (m == MAP_FAILED) {
        fail("cannot map %s: %s", path, strerror(errno));
        close(fd);
        return -1;
    }
    for (size_t o = 0; o < n; o += 1 << 26) CC_SHA256_Update(&c, m + o, (CC_LONG)(n - o < (1u << 26) ? n - o : 1u << 26));
    if (m) munmap((void *)m, n);
    close(fd);
    uint8_t d[32];
    CC_SHA256_Final(d, &c);
    hex(d, out);
    if (cache) {
        FILE *fp = fopen(idx, "a");
        if (fp) {
            fprintf(fp, "%s\t%s\n", key, out);
            fclose(fp);
        }
    }
    return 0;
}

static void default_cache(char *out, size_t n) {
    const char *env = getenv("QD_ENGINE_CACHE");
    if (env) {
        snprintf(out, n, "%s", env);
        return;
    }
    Dl_info di;
    if (dladdr((void *)qd_engine_open, &di) && di.dli_fname) {
        char tmp[PATH_MAX];
        snprintf(tmp, sizeof tmp, "%s", di.dli_fname);
        snprintf(out, n, "%s/engine-cache", dirname(tmp));
    } else {
        snprintf(out, n, "build/engine-cache");
    }
}

/* --- open / close ------------------------------------------------------------------------------ */

/* The whole file, NUL-terminated; NULL (and the error) if it cannot be read. */
static char *slurp(const char *path, size_t *len) {
    struct stat sb;
    int fd = open_file(path, &sb);
    if (fd < 0) return NULL;
    size_t n = sb.st_size;
    char *b = malloc(n + 1);
    ssize_t k = 0;
    for (*len = 0; *len < n && (k = read(fd, b + *len, n - *len)) > 0;) *len += k;
    if (k >= 0) b[*len] = 0;
    else {
        fail("cannot read %s: %s", path, strerror(errno));
        free(b);
        b = NULL;
    }
    close(fd);
    return b;
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

qd_engine *qd_engine_open(const char *weights_dir, int n_threads) {
    return qd_engine_open_ex(weights_dir, n_threads, NULL, 0);
}

qd_engine *qd_engine_open_ex(const char *dir, int n_threads, const char *cache_dir, int flags) {
    double t0 = now_s();
    g_err[0] = 0;
    char path[PATH_MAX];
    snprintf(path, sizeof path, "%s/config.json", dir);
    size_t clen;
    char *ctext = slurp(path, &clen);
    if (!ctext) return NULL;
    jv *root = json_parse(ctext, clen);
    free(ctext);
    if (!root) return fail("bad %s", path), NULL;
    jv *cfg = json_get(root, "text_config") ? json_get(root, "text_config") : root;
    qd_engine *e = calloc(1, sizeof *e);
    qd_engine *ret = NULL;
    st_file f = {.fd = -1};
    e->flags = flags;
    /* the shapes: positive integers (a default when absent) */
    const struct { const char *key; int32_t *v; int dflt; } shapes[] = {
        {"hidden_size", &e->H, 0}, {"num_hidden_layers", &e->nl, 0}, {"vocab_size", &e->V, 0},
        {"linear_num_key_heads", &e->nk, 0}, {"linear_num_value_heads", &e->nv, 0},
        {"linear_key_head_dim", &e->dk, 0}, {"linear_value_head_dim", &e->dv, 0},
        {"linear_conv_kernel_dim", &e->KS, 4}};
    for (size_t i = 0; i < sizeof shapes / sizeof *shapes; i++) {
        double v = json_num(cfg, shapes[i].key, shapes[i].dflt);
        if (!(v >= 1 && v <= 1 << 30) || v != floor(v)) {
            fail("%s: %s is %s", path, shapes[i].key,
                 json_get(cfg, shapes[i].key) ? "not a positive integer" : "missing");
            goto out;
        }
        *shapes[i].v = (int32_t)v;
    }
    e->Hp = (e->H + 63) & ~63; /* (whole 64-bit words of the nonzero bitmaps) */
    e->eps = (float)json_num(cfg, "rms_norm_eps", 1e-6);
    e->qscale = (float)sqrt((double)e->dk); /* torch divides by the Python float, cast to fp32 */
    jv *types = json_get(cfg, "layer_types");
    int ok = types && types->type == J_ARR && types->n == e->nl;
    for (int L = 0; ok && L < e->nl; L++) ok = types->items[L].type == J_STR;
    if (!ok) {
        fail("%s: layer_types is not a list of num_hidden_layers names", path);
        goto out;
    }
    /* (the 16-bit indices: a residual dim in a sparse entry or unit record, an eager channel's
       16-row block (2 nk dk channels at most) and dim in qk_id, whose key head has 15 bits) */
    if (e->nv % e->nk || e->dk % 64 || e->dv % 16 || e->dk > 256 || e->dv > 256 ||
        e->KS > QD_MAX_KS || e->H > 65536 || (int64_t)e->nk * e->dk > 8 * 65536) {
        fail("unsupported shapes (nv %% nk, dk %% 64, dv %% 16, head dims <= 256, conv <= 8, "
             "hidden <= 65536, key heads x key dim <= 524288)");
        goto out;
    }
    e->r = e->nv / e->nk;
    e->L = qd_alloc(e, e->nl * sizeof(layer_t));
    snprintf(path, sizeof path, "%s/model.safetensors", dir);
    char cache[PATH_MAX] = "";
    if (!(flags & QD_OPEN_NO_CACHE)) {
        if (cache_dir) snprintf(cache, sizeof cache, "%s", cache_dir);
        else default_cache(cache, sizeof cache);
        if (!strcmp(cache, "off") || !cache[0]) cache[0] = 0;
        else mkdir(cache, 0755);
    }
    if (file_sha(cache[0] ? cache : NULL, path, e->info.sha256)) goto out;
    if (flags & QD_OPEN_VERBOSE)
        fprintf(stderr, "qd_engine: %s sha256 %s (%.2f s)\n", path, e->info.sha256, now_s() - t0);
    char cpath[PATH_MAX + 128] = "";
    if (cache[0]) {
        snprintf(cpath, sizeof cpath, "%s/%s-v%d.qdc", cache, e->info.sha256, QD_LAYOUT_VERSION);
        FILE *fp = fopen(cpath, "rb");
        if (fp) {
            /* (not a file, truncated, or another version or shapes: stale, rebuilt below) */
            struct stat cs;
            ser_t s = {fp, 1, 0, e, !fstat(fileno(fp), &cs) && S_ISREG(cs.st_mode) ? cs.st_size : 0};
            ser_header(&s, e);
            if (!s.err) ser_model(&s, e);
            fclose(fp);
            if (!s.err) e->info.from_cache = 1;
            else if (flags & QD_OPEN_VERBOSE) fprintf(stderr, "qd_engine: stale cache %s\n", cpath);
        }
    }
    if (!e->info.from_cache) {
        /* (a failed cache read may have left partial arrays: start from fresh layouts) */
        for (int i = 0; i < e->nallocs; i++) free(e->allocs[i]);
        e->nallocs = 0;
        e->L = qd_alloc(e, e->nl * sizeof(layer_t));
        e->necol = e->ncode = 0;
        if (st_open(&f, path)) goto out;
        int64_t nnz = 0;
        if (build(e, &f, cfg, &nnz)) goto out;
        e->info.nnz_compact = nnz;
        if (cpath[0]) {
            char tmp[PATH_MAX + 160];
            snprintf(tmp, sizeof tmp, "%s.%d.tmp", cpath, (int)getpid());
            FILE *fp = fopen(tmp, "wb");
            if (fp) {
                ser_t s = {fp, 0, 0, e, 0};
                ser_header(&s, e);
                ser_model(&s, e);
                if (fclose(fp)) s.err = 1; /* (the buffered tail not written) */
                if (s.err || rename(tmp, cpath)) unlink(tmp);
            }
        }
    }
    e->last_needed = -1;
    for (int L = 0; L < e->nl; L++)
        if (e->L[L].attn) e->last_needed = L;
    const char *envt = getenv("QD_THREADS");
    e->nth = n_threads > 0 ? n_threads : envt ? atoi(envt) : 1;
    if (e->nth < 1) e->nth = 1;
    if (e->nth > QD_MAX_THREADS) e->nth = QD_MAX_THREADS;
    qd_derive(e);
    qd_state_alloc(e);
    if (qd_pool_start(e)) {
        fail("cannot start worker threads");
        goto out;
    }
    qd_engine_reset(e);
    qd_engine_info *in = &e->info;
    in->hidden = e->H;
    in->n_layers = e->nl;
    in->vocab = e->V;
    in->key_heads = e->nk;
    in->value_heads = e->nv;
    in->key_dim = e->dk;
    in->value_dim = e->dv;
    in->conv_kernel = e->KS;
    in->n_threads = e->nth;
    in->n_code = e->ncode;
    in->sign_head = e->sign_ok;
    in->last_needed_layer = e->last_needed;
    in->prefix_dims = e->pre;
    in->open_seconds = now_s() - t0;
    if (flags & QD_OPEN_VERBOSE) {
        fprintf(stderr, "qd_engine: open %.2f s (%s), %lld compact nonzeros, %d threads\n",
                in->open_seconds, in->from_cache ? "cache" : "built", (long long)in->nnz_compact,
                e->nth);
        for (int L = 0; L < e->nl; L++) {
            const layer_t *l = &e->L[L];
            fprintf(stderr, "  layer %2d: qk %d (%d 1-tap, %u blocks) heads %d vch %d od %d pool %d "
                    "cols %d out %d | units %d (%u blocks) cols %d out %d\n", L, l->nqk, l->nqk1,
                    l->attn ? l->qk.ptr[e->H] : 0, l->nact, l->nvch, l->nod, l->npool, l->na_cols,
                    l->na_out, l->nu, l->mlp ? l->gate.ptr[e->H] : 0, l->nm_cols, l->nm_out);
            fprintf(stderr, "            |w| attn in [%g, %g] out [%g, %g] mlp in [%g, %g] out [%g, %g]\n",
                    l->wr[0][0], l->wr[0][1], l->wr[1][0], l->wr[1][1], l->wr[2][0], l->wr[2][1],
                    l->wr[3][0], l->wr[3][1]);
        }
    }
    ret = e;
out:
    st_close(&f);
    json_free(root);
    if (!ret) qd_engine_close(e);
    return ret;
}

void qd_engine_close(qd_engine *e) {
    if (!e) return;
    qd_pool_stop(e);
    for (int i = 0; i < e->nallocs; i++) free(e->allocs[i]);
    free(e->allocs);
    free(e);
}

void qd_engine_reset(qd_engine *e) { qd_state_reset(e); }

static float *state_row(qd_engine *e, int layer, int vh, int row) {
    if (layer < 0 || layer >= e->nl || vh < 0 || vh >= e->nv || row < 0 || row >= e->dk) return NULL;
    layer_t *l = &e->L[layer];
    if (!l->S) return NULL;
    return l->S + ((size_t)vh * e->dk + row) * e->dv;
}

void qd_engine_set_row(qd_engine *e, int layer, int vh, int row, const float *vec) {
    float *s = state_row(e, layer, vh, row);
    if (!s) return;
    memcpy(s, vec, e->dv * 4);
    int nz = 0;
    for (int i = 0; i < e->dv; i++) nz |= vec[i] != 0;
    uint64_t *m = e->L[layer].rows_nz + (size_t)vh * (e->dk / 64) + row / 64;
    if (nz) *m |= 1ull << (row % 64);
    else *m &= ~(1ull << (row % 64));
}

void qd_engine_set_rows(qd_engine *e, int layer, int vh, int n, const int *rows, const float *vecs) {
    for (int i = 0; i < n; i++) qd_engine_set_row(e, layer, vh, rows[i], vecs + (size_t)i * e->dv);
}

void qd_engine_get_row(qd_engine *e, int layer, int vh, int row, float *out) {
    float *s = state_row(e, layer, vh, row);
    if (s) memcpy(out, s, e->dv * 4);
    else memset(out, 0, e->dv * 4);
}

void qd_engine_set_exact_head(qd_engine *e, int on) { e->exact_head = on; }
void qd_engine_capture(qd_engine *e, float *buf) { e->capture = buf; }
void qd_engine_set_profile(qd_engine *e, int on) { e->profile = on; }
void qd_engine_get_info(qd_engine *e, qd_engine_info *info) { *info = e->info; }

void qd_engine_get_stats(qd_engine *e, qd_engine_stats *s, int reset) {
    *s = e->st;
    if (reset) memset(&e->st, 0, sizeof e->st);
}

const char *qd_engine_stage_name(int stage) {
    static const char *names[QD_N_STAGES] = {"embed", "norm", "qk_proj", "conv", "delta",
                                             "v_lazy", "out_proj", "gate", "up", "down",
                                             "residual", "head", "sync"};
    return stage >= 0 && stage < QD_N_STAGES ? names[stage] : "?";
}
