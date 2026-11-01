/* The qwendoom host computer (see qd_host.h): ELF loading, the decoded-instruction caches, the
   host ecall services, dirty tracking and pokes, the native sim context, and the C API.

   Memory is one anonymous mapping of the whole 80 MB address space (unused gaps never touched,
   so never committed); qd_mem's four pointers point into it, and a guest address is an offset.
   The instruction semantics follow chipformer.rv.ref (Qwenputer, MIT (c) 2026 Divyansh Agrawal),
   whose RV32 machine and trace records the sim context's token count reproduces. */
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "qd_host.h"

#define QD_SPACE      (QD_HOST_BASE + QD_HOST_SIZE)  /* the mapping covers [0, QD_SPACE) */
#define QD_NULL_GUARD 0x100u
#define QD_MISA       0x40001100u                    /* RV32, I and M (chipformer.rv.spec) */
#define HOST_ARGV_TOP (QD_HOST_BASE + QD_HOST_SIZE)  /* argv strings, then the host stack below */
#define HOST_STACK_TOP (HOST_ARGV_TOP - 4096)
#define SIM_BUDGET    200000000ull                   /* a sim tick that does not yield is a bug */
#define MAX_FDS       32
#define OUT_CAP       (1 << 16)
#define KEYQ          256

#define QD_EV_YIELD    100  /* internal: the sim yielded */
#define QD_EV_CONTINUE 101  /* internal: the host ecall was served, keep running */

enum {
    U_ILLEGAL, U_LUI, U_AUIPC, U_JAL, U_JALR, U_BEQ, U_BNE, U_BLT, U_BGE, U_BLTU, U_BGEU,
    U_LB, U_LH, U_LW, U_LBU, U_LHU, U_SB, U_SH, U_SW,
    U_ADDI, U_SLTI, U_SLTIU, U_XORI, U_ORI, U_ANDI, U_SLLI, U_SRLI, U_SRAI,
    U_ADD, U_SUB, U_SLL, U_SLT, U_SLTU, U_XOR, U_SRL, U_SRA, U_OR, U_AND,
    U_MUL, U_MULH, U_MULHSU, U_MULHU, U_DIV, U_DIVU, U_REM, U_REMU, U_SH1ADD, U_SH2ADD, U_SH3ADD,
    U_FENCE, U_ECALL, U_EBREAK, U_CSR_TIME, U_CSR_ZERO, U_CSR_MISA, U_BADJUMP, U_END, U_NUM
};

static const char *const UOP_NAMES[U_NUM] = {
    "ILLEGAL", "LUI", "AUIPC", "JAL", "JALR", "BEQ", "BNE", "BLT", "BGE", "BLTU", "BGEU",
    "LB", "LH", "LW", "LBU", "LHU", "SB", "SH", "SW",
    "ADDI", "SLTI", "SLTIU", "XORI", "ORI", "ANDI", "SLLI", "SRLI", "SRAI",
    "ADD", "SUB", "SLL", "SLT", "SLTU", "XOR", "SRL", "SRA", "OR", "AND",
    "MUL", "MULH", "MULHSU", "MULHU", "DIV", "DIVU", "REM", "REMU", "SH1ADD", "SH2ADD", "SH3ADD",
    "FENCE", "ECALL", "EBREAK", "CSR_TIME", "CSR_ZERO", "CSR_MISA", "BADJUMP", "END"
};

/* a decoded instruction: handler, immediate (branch / JAL: the target as an instruction delta;
   AUIPC: the absolute value), register slots (x0 as a destination becomes the scratch x32) */
typedef struct {
    const void *op;
    int32_t imm;
    uint8_t rd, rs1, rs2, uop;
} dinst;

typedef struct {
    uint32_t x[33];  /* x0..x31 and the scratch destination of x0 writes */
    uint32_t pc;
    uint64_t icount;
    /* sim context only */
    uint64_t tokens;
    uint32_t n63;
    const uint32_t *inq;
    uint32_t inq_len, inq_pos;
    qd_sim_stats stats;
} cpu_t;

typedef struct {
    uint32_t addr, size;
    const char *name;
} sym_t;

enum { ST_RUNNING, ST_WAIT_SIM, ST_EXITED, ST_ERROR };

struct qd_host {
    uint8_t *m;  /* the address space */
    qd_mem mem;

    /* decoded code: ROM for both contexts, the host text for the host context */
    dinst *rom_host, *rom_sim, *text_host;
    uint32_t text_base, text_size;
    const void *const *labels_host, *const *labels_sim;

    cpu_t host, sim;
    int state, exit_code;
    char err[1024], desc[256];

    /* symbols */
    sym_t *syms;
    int nsyms;
    char *strtab;
    uint32_t sym_sim_main, sym_gp;

    /* dirty tracking: RAM words the host wrote, and the RAM the sim last saw */
    uint64_t dirty[QD_RAM_SIZE / 4 / 64];
    uint8_t *shadow;
    uint32_t *pokes, *inq_buf;
    int npokes_words;
    int need_resync;
    int in_level_load;

    /* frames, keys, files, output */
    uint32_t frame_screen, frame_pal;
    int have_frame;
    uint16_t keyq[KEYQ];
    int key_head, key_tail;
    char wad_dir[1024], cfg_dir[1024];
    int fds[MAX_FDS];
    char out[OUT_CAP];
    int out_len, echo;

    uint64_t ctr[QD_CTR_N];
};

static inline uint32_t ld32(const uint8_t *p) { uint32_t v; memcpy(&v, p, 4); return v; }
static inline uint16_t ld16(const uint8_t *p) { uint16_t v; memcpy(&v, p, 2); return v; }
static inline int16_t ld16s(const uint8_t *p) { int16_t v; memcpy(&v, p, 2); return v; }
static inline void st32(uint8_t *p, uint32_t v) { memcpy(p, &v, 4); }
static inline void st16(uint8_t *p, uint16_t v) { memcpy(p, &v, 2); }

/* the size of a value record (chipformer.rv.spec.value_record): F0 | V F1 | V V F2 */
static inline int vlen(uint32_t v)
{
    int32_t s = (int32_t)v;
    if ((uint32_t)(s + (1 << 14)) < (1u << 15))
        return 1;
    if ((uint32_t)(s + (1 << 29)) < (1u << 30))
        return 2;
    return 3;
}

static int nibbles_used(uint32_t v)
{
    int n = 0;
    for (int i = 0; i < 8; i++)
        n += (v >> 4 * i) != 0;
    return n;
}

/* k0 of a DIV-class record (chipformer.rv.ref.div_start): quotient digits k0..7 are emitted */
static int div_start(int uop, uint32_t a, uint32_t b)
{
    int sb = (uop == U_DIV || uop == U_REM) && (b >> 31);
    int da = nibbles_used(a), db = nibbles_used(sb ? ~b : b);
    if (b == 0)
        return 0;
    int k = 7 - da + db;
    return k < 0 ? 0 : k > 8 ? 8 : k;
}

static uint32_t rv_div(int uop, uint32_t a, uint32_t b)
{
    int32_t sa = (int32_t)a, sb = (int32_t)b;
    switch (uop) {
    case U_DIV:
        if (b == 0) return 0xFFFFFFFFu;
        if (sa == INT32_MIN && sb == -1) return a;
        return (uint32_t)(sa / sb);
    case U_DIVU:
        return b ? a / b : 0xFFFFFFFFu;
    case U_REM:
        if (b == 0) return a;
        if (sa == INT32_MIN && sb == -1) return 0;
        return (uint32_t)(sa % sb);
    default:
        return b ? a % b : a;
    }
}

/* --- errors, output, symbols ---------------------------------------------------------------- */

int qd_host_symbolize(qd_host *h, uint32_t addr, char *buf, int n)
{
    int lo = 0, hi = h->nsyms - 1, best = -1;
    while (lo <= hi) {  /* the last symbol at or below addr */
        int mid = (lo + hi) / 2;
        if (h->syms[mid].addr <= addr) {
            best = mid;
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    if (best >= 0 && addr - h->syms[best].addr < (h->syms[best].size ? h->syms[best].size : 1))
        return snprintf(buf, n, "%s+%#x", h->syms[best].name, addr - h->syms[best].addr);
    return snprintf(buf, n, "%#x", addr);
}

static const char *region_name(uint32_t a)
{
    if (a < QD_RAM_SIZE) return "RAM";
    if (a - QD_DROM_BASE < QD_DROM_SIZE) return "DROM";
    if (a - QD_ROM_BASE < QD_ROM_SIZE) return "ROM";
    if (a - QD_HOST_BASE < QD_HOST_SIZE) return "HOST";
    return "unmapped";
}

/* "0x... (symbol+off, REGION)" for an error message (one per message: a per-host buffer) */
static const char *describe(qd_host *h, uint32_t a)
{
    char sym[200];
    qd_host_symbolize(h, a, sym, sizeof sym);
    snprintf(h->desc, sizeof h->desc, "%#x (%s, %s)", a, sym, region_name(a));
    return h->desc;
}

static void fail(qd_host *h, uint32_t pc, const char *fmt, ...)
{
    char msg[512], where[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    qd_host_symbolize(h, pc, where, sizeof where);
    snprintf(h->err, sizeof h->err, "%s at pc %#x (%s, %s)", msg, pc, where, region_name(pc));
    h->state = ST_ERROR;
}

static void out_bytes(qd_host *h, const char *p, size_t n)
{
    if (h->echo)
        fwrite(p, 1, n, stderr);
    for (size_t i = 0; i < n; i++) {
        if (h->out_len == OUT_CAP) {  /* keep the newest half */
            memmove(h->out, h->out + OUT_CAP / 2, OUT_CAP / 2);
            h->out_len = OUT_CAP / 2;
        }
        h->out[h->out_len++] = p[i];
    }
}

static void sim_putchar(qd_host *h, int c)
{
    char ch = (char)c;
    out_bytes(h, &ch, 1);
}

/* a guest C string (bounded), "" outside the address space */
static const char *gstr(qd_host *h, uint32_t a, char *buf, size_t n)
{
    size_t i = 0;
    for (; i + 1 < n && a + i < QD_SPACE && h->m[a + i]; i++)
        buf[i] = (char)h->m[a + i];
    buf[i] = 0;
    return buf;
}

/* format I_Error's message from guest memory: fmt and a va_list (a pointer to the int-sized
   arguments the callee spilled to its stack) */
static void guest_format(qd_host *h, uint32_t fmt, uint32_t ap, char *out, size_t cap)
{
    char f[512], spec[32], tmp[512];
    size_t o = 0;
    gstr(h, fmt, f, sizeof f);
    out[0] = 0;
    for (const char *p = f; *p && o + 1 < cap;) {
        if (*p != '%') {
            out[o++] = *p++;
            out[o] = 0;
            continue;
        }
        size_t k = 0;
        spec[k++] = *p++;
        while (*p && strchr("-+ #0123456789.lhz", *p) && k < sizeof spec - 2)
            spec[k++] = *p++;
        char conv = *p ? *p++ : 0;
        spec[k++] = conv;
        spec[k] = 0;
        uint32_t arg = ap + 4 < QD_SPACE ? ld32(h->m + ap) : 0;
        if (conv == '%') {
            snprintf(tmp, sizeof tmp, "%%");
        } else if (conv == 's') {
            char s[256];
            snprintf(tmp, sizeof tmp, spec, gstr(h, arg, s, sizeof s));
            ap += 4;
        } else if (conv && strchr("diuxXoc", conv)) {
            /* drop length modifiers: every argument is a 32-bit word here */
            char sp2[32];
            size_t j = 0;
            for (size_t i = 0; spec[i]; i++)
                if (!strchr("lhz", spec[i]))
                    sp2[j++] = spec[i];
            sp2[j] = 0;
            snprintf(tmp, sizeof tmp, sp2, arg);
            ap += 4;
        } else {
            snprintf(tmp, sizeof tmp, "%s", spec);
        }
        size_t t = strlen(tmp);
        if (o + t >= cap)
            t = cap - 1 - o;
        memcpy(out + o, tmp, t);
        o += t;
        out[o] = 0;
    }
}

static void guest_exit_error(qd_host *h, int sim, uint32_t code, uint32_t fmt, uint32_t ap)
{
    char msg[768];
    if (fmt)
        guest_format(h, fmt, ap, msg, sizeof msg);
    else
        snprintf(msg, sizeof msg, "exit(%d)", (int32_t)code);
    char where[256];
    cpu_t *c = sim ? &h->sim : &h->host;
    qd_host_symbolize(h, c->pc, where, sizeof where);
    snprintf(h->err, sizeof h->err, "%s I_Error: %s (pc %#x, %s)", sim ? "sim" : "host", msg, c->pc,
             where);
    h->state = ST_ERROR;
}

/* --- decoding -------------------------------------------------------------------------------- */

static int32_t sx(uint32_t v, int bits) { return (int32_t)(v << (32 - bits)) >> (32 - bits); }

/* decode the instruction word w at pc into d (uop, slots, immediate as the ISA defines it) */
static void decode(uint32_t w, dinst *d)
{
    uint32_t opc = w & 0x7F, rd = w >> 7 & 31, f3 = w >> 12 & 7, rs1 = w >> 15 & 31,
             rs2 = w >> 20 & 31, f7 = w >> 25;
    int32_t i_imm = sx(w >> 20, 12);
    int32_t s_imm = sx((w >> 25) << 5 | (w >> 7 & 31), 12);
    int32_t b_imm = sx((w >> 31 & 1) << 12 | (w >> 7 & 1) << 11 | (w >> 25 & 63) << 5 | (w >> 8 & 15) << 1, 13);
    int32_t j_imm = sx((w >> 31 & 1) << 20 | (w >> 12 & 0xFF) << 12 | (w >> 20 & 1) << 11 | (w >> 21 & 0x3FF) << 1, 21);
    memset(d, 0, sizeof *d);
    d->uop = U_ILLEGAL;
    d->rd = rd;
    d->rs1 = rs1;
    d->rs2 = rs2;
    switch (opc) {
    case 0x37: d->uop = U_LUI; d->imm = (int32_t)(w & 0xFFFFF000u); return;
    case 0x17: d->uop = U_AUIPC; d->imm = (int32_t)(w & 0xFFFFF000u); return;
    case 0x6F: d->uop = U_JAL; d->imm = j_imm; return;
    case 0x67: if (f3 == 0) { d->uop = U_JALR; d->imm = i_imm; } return;
    case 0x63: {
        static const uint8_t br[8] = {U_BEQ, U_BNE, U_ILLEGAL, U_ILLEGAL, U_BLT, U_BGE, U_BLTU, U_BGEU};
        d->uop = br[f3]; d->imm = b_imm; return;
    }
    case 0x03: {
        static const uint8_t ld[8] = {U_LB, U_LH, U_LW, U_ILLEGAL, U_LBU, U_LHU, U_ILLEGAL, U_ILLEGAL};
        d->uop = ld[f3]; d->imm = i_imm; return;
    }
    case 0x23: {  /* stores keep the data register in the rd slot, as the circuit's ROM word does */
        static const uint8_t stv[8] = {U_SB, U_SH, U_SW, U_ILLEGAL, U_ILLEGAL, U_ILLEGAL, U_ILLEGAL, U_ILLEGAL};
        d->uop = stv[f3]; d->imm = s_imm; d->rd = rs2; return;
    }
    case 0x13:
        d->imm = i_imm;
        switch (f3) {
        case 0: d->uop = U_ADDI; return;
        case 2: d->uop = U_SLTI; return;
        case 3: d->uop = U_SLTIU; return;
        case 4: d->uop = U_XORI; return;
        case 6: d->uop = U_ORI; return;
        case 7: d->uop = U_ANDI; return;
        case 1: if (f7 == 0) { d->uop = U_SLLI; d->imm = rs2; } return;
        case 5:
            if (f7 == 0) { d->uop = U_SRLI; d->imm = rs2; }
            else if (f7 == 0x20) { d->uop = U_SRAI; d->imm = rs2; }
            return;
        }
        return;
    case 0x33:
        if (f7 == 0) {
            static const uint8_t r[8] = {U_ADD, U_SLL, U_SLT, U_SLTU, U_XOR, U_SRL, U_OR, U_AND};
            d->uop = r[f3];
        } else if (f7 == 0x20 && (f3 == 0 || f3 == 5)) {
            d->uop = f3 ? U_SRA : U_SUB;
        } else if (f7 == 1) {
            static const uint8_t m[8] = {U_MUL, U_MULH, U_MULHSU, U_MULHU, U_DIV, U_DIVU, U_REM, U_REMU};
            d->uop = m[f3];
        } else if (f7 == 0x10 && (f3 == 2 || f3 == 4 || f3 == 6)) {
            d->uop = f3 == 2 ? U_SH1ADD : f3 == 4 ? U_SH2ADD : U_SH3ADD;
        }
        return;
    case 0x0F:
        d->uop = U_FENCE;  /* FENCE (FENCE.I, f3 = 1, is not allowed) */
        if (f3 != 0) d->uop = U_ILLEGAL;
        return;
    case 0x73:
        if (w == 0x00000073u) { d->uop = U_ECALL; return; }
        if (w == 0x00100073u) { d->uop = U_EBREAK; return; }
        /* pure CSR reads (csrrs / csrrc / csrrsi / csrrci with rs1 = 0) of the stub CSRs */
        if ((f3 == 2 || f3 == 3 || f3 == 6 || f3 == 7) && rs1 == 0) {
            uint32_t csr = w >> 20;
            if (csr == QD_CSR_INPUT) d->uop = U_CSR_TIME;
            else if (csr == 0xC00 || csr == 0xC02 || csr == 0xC80 || csr == 0xC81 || csr == 0xC82 || csr == 0xF14)
                d->uop = U_CSR_ZERO;
            else if (csr == 0x301) d->uop = U_CSR_MISA;
        }
        return;
    }
}

static int writes_rd(int u)
{
    return !(u == U_BEQ || u == U_BNE || u == U_BLT || u == U_BGE || u == U_BLTU || u == U_BGEU
             || u == U_SB || u == U_SH || u == U_SW || u == U_FENCE || u == U_ECALL || u == U_EBREAK
             || u == U_ILLEGAL || u == U_BADJUMP || u == U_END);
}

/* decode [base, base + 4 * n) into out (n + 1 entries: an END sentinel last) for one context */
static void decode_region(qd_host *h, dinst *out, uint32_t base, uint32_t n, const void *const *labels)
{
    for (uint32_t i = 0; i < n; i++) {
        dinst *d = &out[i];
        uint32_t pc = base + 4 * i;
        decode(ld32(h->m + pc), d);
        if (d->uop == U_AUIPC)
            d->imm = (int32_t)(pc + (uint32_t)d->imm);
        if (d->uop == U_JAL || (d->uop >= U_BEQ && d->uop <= U_BGEU)) {
            uint32_t t = pc + (uint32_t)d->imm;
            if ((t & 3) || t - base >= 4 * n)
                d->uop = U_BADJUMP;
            else
                d->imm /= 4;
        }
        if (writes_rd(d->uop) && d->rd == 0)
            d->rd = 32;
        d->op = labels[d->uop];
    }
    memset(&out[n], 0, sizeof(dinst));
    out[n].uop = U_END;
    out[n].op = labels[U_END];
}

/* the decoded instruction at t for a context (sim: ROM only), or 0 */
static inline int locate(qd_host *h, int sim, uint32_t t, const dinst **ip, const dinst **dbase,
                         uint32_t *pbase)
{
    if (t & 3)
        return 0;
    if (t - QD_ROM_BASE < QD_ROM_SIZE) {
        *dbase = sim ? h->rom_sim : h->rom_host;
        *pbase = QD_ROM_BASE;
    } else if (!sim && t - h->text_base < h->text_size) {
        *dbase = h->text_host;
        *pbase = h->text_base;
    } else {
        return 0;
    }
    *ip = *dbase + (t - *pbase) / 4;
    return 1;
}

/* --- host memory checks ---------------------------------------------------------------------- */

static int host_load_ok(qd_host *h, uint32_t a)
{
    if (a < QD_RAM_SIZE) {
        if (a < QD_NULL_GUARD)
            h->ctr[QD_CTR_HOST_NULL_ACCESSES]++;
        return 1;
    }
    return a - QD_DROM_BASE < QD_DROM_SIZE + QD_ROM_SIZE;
}

static int host_store_slow(qd_host *h, uint32_t a, int n)
{
    if (a < QD_RAM_SIZE) {
        if (a < QD_NULL_GUARD)
            h->ctr[QD_CTR_HOST_NULL_ACCESSES]++;
        uint32_t w0 = a >> 2, w1 = (a + n - 1) >> 2;
        if (w1 >= QD_RAM_SIZE / 4)
            return 0;
        h->dirty[w0 >> 6] |= 1ull << (w0 & 63);
        h->dirty[w1 >> 6] |= 1ull << (w1 & 63);
        h->ctr[QD_CTR_RAM_WRITES]++;
        return 1;
    }
    if (a - QD_DROM_BASE < QD_DROM_SIZE && a + n <= QD_DROM_BASE + QD_DROM_SIZE) {
        h->ctr[QD_CTR_DROM_WRITES]++;
        if (!h->in_level_load && h->ctr[QD_CTR_LEVELS])  /* (startup tables come before) */
            h->ctr[QD_CTR_DROM_WRITES_MIDLEVEL]++;
        h->need_resync = 1;
        return 1;
    }
    return 0;  /* ROM (code) and unmapped addresses */
}

/* a host write of n bytes at a (an ecall filling guest memory) */
static void mark_host_writes(qd_host *h, uint32_t a, uint32_t n)
{
    uint32_t end = a + n;
    if (a < QD_RAM_SIZE)
        for (uint32_t w = a & ~3u; w < end && w < QD_RAM_SIZE; w += 4)
            host_store_slow(h, w, 1);
    if (end > QD_DROM_BASE && a < QD_DROM_BASE + QD_DROM_SIZE)
        host_store_slow(h, a > QD_DROM_BASE ? a : QD_DROM_BASE, 1);
}

/* --- host ecalls -------------------------------------------------------------------------------- */

/* guest path -> host path under dir: relative, at most 128 bytes, no empty, "." or ".."
   components (a leading "./" is dropped; a trailing "/" is allowed) */
static int host_path(const char *dir, const char *g, char *out, size_t n)
{
    while (g[0] == '.' && g[1] == '/')
        g += 2;
    if (!dir[0] || !g[0] || g[0] == '/' || strlen(g) > 128)
        return 0;
    for (const char *p = g; *p;) {
        const char *e = strchr(p, '/');
        size_t len = e ? (size_t)(e - p) : strlen(p);
        if (!len || (len == 1 && p[0] == '.') || (len == 2 && p[0] == '.' && p[1] == '.'))
            return 0;
        p += len + (e != NULL);
    }
    snprintf(out, n, "%s/%s", dir, g);
    return 1;
}

static int host_ecall(qd_host *h, cpu_t *c)
{
    uint32_t *x = c->x, a7 = x[17], a0 = x[10], a1 = x[11], a2 = x[12];
    char path[1100], g[256], g2[256];
#define RET(v) do { x[10] = (uint32_t)(v); return QD_EV_CONTINUE; } while (0)
#define BUF_OK(a, n) ((a) < QD_SPACE && (n) <= QD_SPACE - (a))
    switch (a7) {
    case QD_EC_PUTCHAR:
        sim_putchar(h, a0 & 0xFF);
        RET(a0);
    case QD_EC_WRITE:
        if (!BUF_OK(a1, a2))
            RET(-1);
        out_bytes(h, (const char *)h->m + a1, a2);
        RET(a2);
    case QD_EC_EXIT:
        if (a1) {
            guest_exit_error(h, 0, a0, a1, a2);
            return QD_EV_ERROR;
        }
        h->exit_code = (int32_t)a0;
        h->state = ST_EXITED;
        return QD_EV_EXIT;
    case QD_EC_OPEN: {
        int slot = -1, fd = -1;
        for (int i = 3; i < MAX_FDS; i++)
            if (h->fds[i] < 0) { slot = i; break; }
        gstr(h, a0, g, sizeof g);
        if (slot < 0)
            RET(-1);
        if (a1 == QD_O_WRITE) {
            if (host_path(h->cfg_dir, g, path, sizeof path))
                fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        } else {
            if (host_path(h->cfg_dir, g, path, sizeof path))
                fd = open(path, O_RDONLY);
            if (fd < 0 && host_path(h->wad_dir, g, path, sizeof path))
                fd = open(path, O_RDONLY);
        }
        if (fd >= 0) {
            struct stat sb;
            if (fstat(fd, &sb) || !S_ISREG(sb.st_mode)) {
                close(fd);
                fd = -1;
            }
        }
        if (fd < 0)
            RET(-1);
        h->fds[slot] = fd;
        RET(slot);
    }
    case QD_EC_CLOSE:
        if (a0 >= 3 && a0 < MAX_FDS && h->fds[a0] >= 0) {
            close(h->fds[a0]);
            h->fds[a0] = -1;
            RET(0);
        }
        RET(-1);
    case QD_EC_READ:
    case QD_EC_FWRITE: {
        if (a0 < 3 || a0 >= MAX_FDS || h->fds[a0] < 0 || !BUF_OK(a1, a2))
            RET(-1);
        if (a7 == QD_EC_READ) {
            /* the read lands in guest memory like a host store: never in ROM or the null guard */
            if ((a1 < QD_ROM_BASE + QD_ROM_SIZE && a1 + a2 > QD_ROM_BASE) || a1 < QD_NULL_GUARD)
                RET(-1);
            mark_host_writes(h, a1, a2);
            ssize_t r = read(h->fds[a0], h->m + a1, a2);
            RET(r < 0 ? -1 : r);
        }
        ssize_t r = write(h->fds[a0], h->m + a1, a2);
        RET(r < 0 ? -1 : r);
    }
    case QD_EC_SEEK:
        if (a0 < 3 || a0 >= MAX_FDS || h->fds[a0] < 0)
            RET(-1);
        RET(lseek(h->fds[a0], (int32_t)a1, a2 == 0 ? SEEK_SET : a2 == 1 ? SEEK_CUR : SEEK_END));
    case QD_EC_REMOVE:
        gstr(h, a0, g, sizeof g);
        RET(host_path(h->cfg_dir, g, path, sizeof path) ? unlink(path) : -1);
    case QD_EC_RENAME: {
        char path2[1100];
        gstr(h, a0, g, sizeof g);
        gstr(h, a1, g2, sizeof g2);
        if (!host_path(h->cfg_dir, g, path, sizeof path) || !host_path(h->cfg_dir, g2, path2, sizeof path2))
            RET(-1);
        RET(rename(path, path2));
    }
    case QD_EC_MKDIR:
        gstr(h, a0, g, sizeof g);
        RET(host_path(h->cfg_dir, g, path, sizeof path) && (!mkdir(path, 0755) || errno == EEXIST)
            ? 0 : -1);
    case QD_EC_FRAME:
        if (!BUF_OK(a0, 64000) || !BUF_OK(a1, 768)) {
            fail(h, c->pc, "frame buffer outside memory");
            return QD_EV_ERROR;
        }
        h->frame_screen = a0;
        h->frame_pal = a1;
        h->have_frame = 1;
        h->ctr[QD_CTR_FRAMES]++;
        x[10] = 0;
        return QD_EV_FRAME;
    case QD_EC_SIM_TICK:
        h->state = ST_WAIT_SIM;
        h->ctr[QD_CTR_SIM_TICKS]++;
        x[10] = 0;
        return QD_EV_SIM_TICK;
    case QD_EC_GETKEY:
        if (h->key_head == h->key_tail)
            RET(-1);
        RET(h->keyq[h->key_tail++ % KEYQ]);
    case QD_EC_LEVEL:  /* a0 = 1: P_SetupLevel starts; 0: it is done */
        h->need_resync = 1;
        h->in_level_load = a0 == 1;
        if (a0 == 0)
            h->ctr[QD_CTR_LEVELS]++;
        RET(0);
    case QD_EC_TITLE:
        RET(0);
    }
    fail(h, c->pc, "unknown host ecall a7=%#x", a7);
    return QD_EV_ERROR;
#undef RET
#undef BUF_OK
}

/* --- the interpreters ------------------------------------------------------------------------- */

#define QD_SIM 0
#include "qd_interp.inc"
#undef QD_SIM
#define QD_SIM 1
#include "qd_interp.inc"
#undef QD_SIM

/* --- ELF loading ------------------------------------------------------------------------------- */

static int sym_cmp(const void *a, const void *b)
{
    const sym_t *x = a, *y = b;
    return x->addr < y->addr ? -1 : x->addr > y->addr;
}

static int load_elf(qd_host *h, const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "qd_host: cannot open %s: %s\n", path, strerror(errno));
        return 0;
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *b = malloc(n);
    if (!b || fread(b, 1, n, f) != (size_t)n) {
        fclose(f);
        free(b);
        return 0;
    }
    fclose(f);
    int ok = 0;
    if (n < 52 || memcmp(b, "\x7f" "ELF", 4) || b[4] != 1 || ld16(b + 18) != 243) {
        fprintf(stderr, "qd_host: %s is not an RV32 ELF\n", path);
        goto out;
    }
    uint32_t entry = ld32(b + 24), phoff = ld32(b + 28), shoff = ld32(b + 32);
    uint16_t phentsize = ld16(b + 42), phnum = ld16(b + 44), shentsize = ld16(b + 46),
             shnum = ld16(b + 48);
    h->host.pc = entry;
    for (int i = 0; i < phnum; i++) {
        const uint8_t *p = b + phoff + i * phentsize;
        uint32_t type = ld32(p), off = ld32(p + 4), va = ld32(p + 8), fsz = ld32(p + 16),
                 msz = ld32(p + 20), flags = ld32(p + 24);
        if (type != 1 || !msz)
            continue;
        const char *r = region_name(va);
        if (strcmp(r, region_name(va + msz - 1)) || !strcmp(r, "unmapped")) {
            fprintf(stderr, "qd_host: segment %#x+%#x is not inside one region\n", va, msz);
            goto out;
        }
        memcpy(h->m + va, b + off, fsz);
        memset(h->m + va + fsz, 0, msz - fsz);
        if ((flags & 1) && va >= QD_HOST_BASE) {  /* executable host segment: the host text */
            if (h->text_size) {
                fprintf(stderr, "qd_host: more than one executable HOST segment\n");
                goto out;
            }
            h->text_base = va;
            h->text_size = (msz + 3) & ~3u;
        }
    }
    /* symbols: functions and objects with sizes, sorted by address */
    for (int i = 0; i < shnum; i++) {
        const uint8_t *s = b + shoff + i * shentsize;
        if (ld32(s + 4) != 2)  /* SHT_SYMTAB */
            continue;
        const uint8_t *strs = b + shoff + ld32(s + 24) * shentsize;
        uint32_t soff = ld32(s + 16), ssz = ld32(s + 20), stroff = ld32(strs + 16), strsz = ld32(strs + 20);
        h->strtab = malloc(strsz);
        memcpy(h->strtab, b + stroff, strsz);
        int cnt = ssz / 16;
        h->syms = calloc(cnt, sizeof(sym_t));
        for (int k = 0; k < cnt; k++) {
            const uint8_t *e = b + soff + 16 * k;
            uint32_t name = ld32(e), value = ld32(e + 4), size = ld32(e + 8);
            uint8_t info = e[12];
            uint16_t shndx = ld16(e + 14);
            const char *nm = h->strtab + name;
            if (!strcmp(nm, "sim_main"))
                h->sym_sim_main = value;
            if (!strcmp(nm, "__global_pointer$"))
                h->sym_gp = value;
            int type = info & 15;
            if (shndx == 0 || !nm[0] || (type != 1 && type != 2 && type != 0) || nm[0] == '$'
                || !strncmp(nm, ".L", 2))
                continue;
            h->syms[h->nsyms++] = (sym_t){value, size, nm};
        }
        qsort(h->syms, h->nsyms, sizeof(sym_t), sym_cmp);
    }
    if (!h->text_size || !h->sym_sim_main || !h->sym_gp) {
        fprintf(stderr, "qd_host: %s lacks host text, sim_main or __global_pointer$\n", path);
        goto out;
    }
    ok = 1;
out:
    free(b);
    return ok;
}

/* --- API ------------------------------------------------------------------------------------- */

qd_host *qd_host_new(const char *elf_path, const char *wad_path, int argc, const char *const *argv)
{
    qd_host *h = calloc(1, sizeof *h);
    if (!h)
        return NULL;
    h->m = mmap(NULL, QD_SPACE + 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (h->m == MAP_FAILED) {
        free(h);
        return NULL;
    }
    h->mem = (qd_mem){h->m + QD_RAM_BASE, h->m + QD_DROM_BASE, h->m + QD_ROM_BASE, h->m + QD_HOST_BASE};
    for (int i = 0; i < MAX_FDS; i++)
        h->fds[i] = -1;
    h->shadow = calloc(1, QD_RAM_SIZE);
    h->pokes = malloc(sizeof(uint32_t) * (2 * QD_RAM_SIZE / 4 + 1));
    h->inq_buf = malloc(sizeof(uint32_t) * (2 * QD_RAM_SIZE / 4 + 1));
    if (!h->shadow || !h->pokes || !h->inq_buf || !load_elf(h, elf_path)) {
        qd_host_free(h);
        return NULL;
    }
    run_host(h, NULL, 0, &h->labels_host);
    run_sim(h, NULL, 0, &h->labels_sim);
    uint32_t nrom = QD_ROM_SIZE / 4, ntext = h->text_size / 4;
    h->rom_host = malloc(sizeof(dinst) * (nrom + 1));
    h->rom_sim = malloc(sizeof(dinst) * (nrom + 1));
    h->text_host = malloc(sizeof(dinst) * (ntext + 1));
    decode_region(h, h->rom_host, QD_ROM_BASE, nrom, h->labels_host);
    decode_region(h, h->rom_sim, QD_ROM_BASE, nrom, h->labels_sim);
    decode_region(h, h->text_host, h->text_base, ntext, h->labels_host);

    /* the WAD's directory is the guest's file system (reads); argv strings at the top of HOST */
    const char *slash = strrchr(wad_path, '/');
    if (slash)
        snprintf(h->wad_dir, sizeof h->wad_dir, "%.*s", (int)(slash - wad_path), wad_path);
    else
        snprintf(h->wad_dir, sizeof h->wad_dir, ".");
    const char *wad_name = slash ? slash + 1 : wad_path;
    int gargc = 3 + argc;
    const char **gargv = calloc(gargc, sizeof(char *));
    gargv[0] = "doom";
    gargv[1] = "-iwad";
    gargv[2] = wad_name;
    for (int i = 0; i < argc; i++)
        gargv[3 + i] = argv[i];
    uint32_t sp = HOST_ARGV_TOP, ptrs[64];
    for (int i = gargc - 1; i >= 0 && i < 64; i--) {
        size_t len = strlen(gargv[i]) + 1;
        sp -= len;
        memcpy(h->m + sp, gargv[i], len);
        ptrs[i] = sp;
    }
    sp &= ~15u;
    sp -= 4 * (gargc + 1);
    for (int i = 0; i < gargc; i++)
        st32(h->m + sp + 4 * i, ptrs[i]);
    st32(h->m + sp + 4 * gargc, 0);
    free(gargv);
    if (sp < HOST_STACK_TOP) {
        fprintf(stderr, "qd_host: argv too long\n");
        qd_host_free(h);
        return NULL;
    }
    h->host.x[2] = HOST_STACK_TOP;
    h->host.x[10] = gargc;
    h->host.x[11] = sp;
    h->state = ST_RUNNING;
    qd_host_resync(h);  /* the sim context starts at its entry */
    h->ctr[QD_CTR_RESYNCS] = 0;
    h->need_resync = 1;  /* the first snapshot is taken at the first SIM_TICK */
    return h;
}

void qd_host_free(qd_host *h)
{
    if (!h)
        return;
    for (int i = 0; i < MAX_FDS; i++)
        if (h->fds[i] >= 0)
            close(h->fds[i]);
    if (h->m && h->m != MAP_FAILED)
        munmap(h->m, QD_SPACE + 4096);
    free(h->rom_host);
    free(h->rom_sim);
    free(h->text_host);
    free(h->syms);
    free(h->strtab);
    free(h->shadow);
    free(h->pokes);
    free(h->inq_buf);
    free(h);
}

void qd_host_set_config_dir(qd_host *h, const char *dir)
{
    snprintf(h->cfg_dir, sizeof h->cfg_dir, "%s", dir ? dir : "");
}

void qd_host_set_echo(qd_host *h, int echo) { h->echo = echo; }

int qd_host_run(qd_host *h, uint64_t max_instr)
{
    if (h->state == ST_WAIT_SIM)
        return QD_EV_SIM_TICK;
    if (h->state == ST_EXITED)
        return QD_EV_EXIT;
    if (h->state != ST_RUNNING)
        return QD_EV_ERROR;
    return run_host(h, &h->host, h->host.icount + max_instr, NULL);
}

qd_mem *qd_host_mem(qd_host *h) { return &h->mem; }
const char *qd_host_error(qd_host *h) { return h->err; }
int qd_host_exit_code(qd_host *h) { return h->exit_code; }

int qd_host_frame(qd_host *h, uint8_t *idx, uint8_t *pal)
{
    if (!h->have_frame)
        return -1;
    if (idx)
        memcpy(idx, h->m + h->frame_screen, 64000);
    if (pal)
        memcpy(pal, h->m + h->frame_pal, 768);
    return 0;
}

void qd_host_key(qd_host *h, int doom_key, int down)
{
    if (h->key_head - h->key_tail >= KEYQ)
        return;  /* full: drop */
    h->keyq[h->key_head++ % KEYQ] = (uint16_t)((down ? 1 : 0) << 8 | (doom_key & 0xFF));
}

int qd_host_action_key(const char *a)
{
    /* doomgeneric's doomkeys.h and default bindings (m_controls.c) */
    static const struct { const char *name; int key; } map[] = {
        {"left", 0xac}, {"right", 0xae}, {"forward", 0xad}, {"back", 0xaf},
        {"strafeleft", 0xa0}, {"straferight", 0xa1}, {"fire", 0xa3}, {"use", 0xa2},
        {"run", 0x80 + 0x36}, {"menu", 27}, {"enter", 13},
        {"weapon1", '1'}, {"weapon2", '2'}, {"weapon3", '3'},
    };
    for (size_t i = 0; i < sizeof map / sizeof map[0]; i++)
        if (!strcmp(a, map[i].name))
            return map[i].key;
    return -1;
}

/* collect the changed dirty words into h->pokes (without clearing anything) */
static int collect_pokes(qd_host *h)
{
    int n = 0;
    for (int i = 0; i < (int)(QD_RAM_SIZE / 4 / 64); i++) {
        uint64_t bits = h->dirty[i];
        while (bits) {
            int b = __builtin_ctzll(bits);
            bits &= bits - 1;
            uint32_t a = (uint32_t)(i * 64 + b) * 4;
            uint32_t v = ld32(h->m + a), old = ld32(h->shadow + a);
            if (v != old && a != 0) {
                h->pokes[n++] = a;
                h->pokes[n++] = v;
            }
        }
    }
    h->pokes[n++] = 0;
    return n;
}

int qd_host_pending_pokes(qd_host *h)
{
    return (collect_pokes(h) - 1) / 2;
}

int qd_host_take_pokes(qd_host *h, uint32_t *out, int max)
{
    int n = collect_pokes(h);
    if (out && max < n)
        return -n;
    if (out)
        memcpy(out, h->pokes, sizeof(uint32_t) * n);
    for (int i = 0; i + 1 < n; i += 2)  /* the sim will have these values */
        st32(h->shadow + h->pokes[i], h->pokes[i + 1]);
    memset(h->dirty, 0, sizeof h->dirty);
    h->npokes_words = n;
    memcpy(h->inq_buf, h->pokes, sizeof(uint32_t) * n);
    h->sim.inq = h->inq_buf;
    h->sim.inq_len = n;
    h->sim.inq_pos = 0;
    h->ctr[QD_CTR_POKES] += (n - 1) / 2;
    return n;
}

int qd_host_need_resync(qd_host *h) { return h->need_resync; }

void qd_host_sim_entry(qd_host *h, uint32_t *pc, uint32_t *sp, uint32_t *gp)
{
    if (pc) *pc = h->sym_sim_main;
    if (sp) *sp = QD_SIM_STACK_TOP;
    if (gp) *gp = h->sym_gp;
}

void qd_host_resync(qd_host *h)
{
    memset(&h->sim.x, 0, sizeof h->sim.x);
    h->sim.pc = h->sym_sim_main;
    h->sim.x[2] = QD_SIM_STACK_TOP;
    h->sim.x[3] = h->sym_gp;
    h->sim.n63 = 0;
    h->sim.inq = NULL;
    h->sim.inq_len = h->sim.inq_pos = 0;
    memcpy(h->shadow, h->m, QD_RAM_SIZE);
    memset(h->dirty, 0, sizeof h->dirty);
    h->need_resync = 0;
    h->ctr[QD_CTR_RESYNCS]++;
}

int64_t qd_host_sim_tick_native(qd_host *h, qd_sim_stats *stats)
{
    if (h->state != ST_WAIT_SIM) {
        snprintf(h->err, sizeof h->err, "sim tick requested while the host is not waiting for one");
        return -1;
    }
    if (h->need_resync) {
        snprintf(h->err, sizeof h->err, "sim tick with a resync pending (call qd_host_resync)");
        return -1;
    }
    if (!h->sim.inq)  /* take_pokes was not called: feed the pokes anyway */
        qd_host_take_pokes(h, NULL, 0);
    cpu_t *c = &h->sim;
    memset(&c->stats, 0, sizeof c->stats);
    c->stats.min_sp = c->x[2];
    uint64_t i0 = c->icount, t0 = c->tokens;
    int ev = run_sim(h, c, c->icount + SIM_BUDGET, NULL);
    c->stats.instructions = c->icount - i0;
    c->stats.tokens = c->tokens - t0;
    c->inq = NULL;
    if (stats)
        *stats = c->stats;
    h->ctr[QD_CTR_SIM_TOKENS] += c->stats.tokens;
    if (ev == QD_EV_YIELD)
        return (int64_t)c->stats.instructions;
    if (ev == QD_EV_BUDGET)
        fail(h, c->pc, "the sim did not yield within %llu instructions", SIM_BUDGET);
    h->state = ST_ERROR;
    return -1;
}

int qd_host_complete_sim_tick(qd_host *h)
{
    if (h->state != ST_WAIT_SIM)
        return -1;
    memcpy(h->shadow, h->m, QD_RAM_SIZE);  /* RAM as the sim leaves it */
    h->state = ST_RUNNING;
    return 0;
}

uint64_t qd_host_instructions(qd_host *h) { return h->host.icount; }
uint64_t qd_host_sim_instructions(qd_host *h) { return h->sim.icount; }

int qd_host_counters(qd_host *h, uint64_t *out, int n)
{
    for (int i = 0; i < n && i < QD_CTR_N; i++)
        out[i] = h->ctr[i];
    return QD_CTR_N;
}

uint32_t qd_host_symbol(qd_host *h, const char *name)
{
    for (int i = 0; i < h->nsyms; i++)
        if (!strcmp(h->syms[i].name, name))
            return h->syms[i].addr;
    return 0;
}

const char *qd_host_uop_name(int uop)
{
    return uop >= 0 && uop < U_NUM ? UOP_NAMES[uop] : "";
}

int qd_host_output(qd_host *h, char *buf, int n)
{
    int k = h->out_len < n ? h->out_len : n;
    memcpy(buf, h->out, k);
    memmove(h->out, h->out + k, h->out_len - k);
    h->out_len -= k;
    return k;
}
