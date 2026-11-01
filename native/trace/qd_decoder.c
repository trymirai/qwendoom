/* qd_dec: the trace decoder (the mirror). It reads the model's tokens one at a time and keeps the
   machine state they imply: the implied PC (anchor + n), the register file (written from value
   records' payloads and J links, never computed), and RAM (stores applied to the shared qd_mem).
   It knows the instruction at the PC (from the ROM), so it checks that every token is legal at
   its position: the record grammar for that instruction always, and in strict mode also every
   token it can derive from its own registers and memory without computing an rd value. */
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#include "qd_isa.h"

enum { /* the phase: what the next token may be */
    PH_START, /* the first token of a record */
    PH_VAL,   /* inside a value record (nv chunks so far) */
    PH_SBH,   /* the merged word of SB / SH (nv chunks so far), then SS0 / SS1 */
    PH_QD,    /* DIV digits: QD with index k */
    PH_HEX,   /* a fault's PC digits (hex so far) */
    PH_FHALT, /* the HALT that ends a fault record */
    PH_DONE,  /* the stream ended */
};

enum { /* the next record's class */
    P_R, P_INST, P_BADPC, P_BADN,
};

#define FAM(f) (1u << QD_FAM_##f)
#define VALUE_START (FAM(V) | FAM(F0))

struct qd_dec {
    qd_mem mem;
    qd_map map;
    qd_icache ic;
    int strict;
    uint32_t x[32];
    uint32_t anchor, n;
    /* the record in progress */
    int ph, plan;
    const qd_inst *e;
    uint32_t pc, ea;
    uint32_t shape;      /* first token: the families the grammar allows */
    uint32_t allow;      /* ... and (strict) the registers */
    int sem_err;         /* the error for a family in shape but not in allow */
    uint32_t fault_mask; /* allowed FLT causes */
    int32_t want_j, want_byte, want_bank, want_k0; /* -1: not checked */
    uint32_t vc[3];
    int nv, k, hex, fcause;
    int fresh; /* nothing fed since set_context: a GO (the run's first token) is accepted */
    /* results */
    int err, halted, fault_cause;
    char msg[400];
    uint32_t exit_code, fault_pc, last_byte, last_store;
    uint64_t tokens, records, instrs, stores;
};

static void prepare(qd_dec *d);

qd_dec *qd_dec_new(const qd_mem *mem, const qd_map *map, int strict) {
    if (map->rom_size / 4 > 65536) return NULL;
    qd_dec *d = calloc(1, sizeof *d);
    if (!d) return NULL;
    d->mem = *mem;
    d->map = *map;
    d->strict = strict;
    if (!qd_icache_init(&d->ic, mem->rom, map)) {
        free(d);
        return NULL;
    }
    qd_dec_set_context(d, 0, 0, NULL);
    return d;
}

void qd_dec_free(qd_dec *d) {
    if (!d) return;
    qd_icache_free(&d->ic);
    free(d);
}

void qd_dec_set_context(qd_dec *d, uint32_t anchor, uint32_t n, const uint32_t *regs) {
    d->anchor = anchor;
    d->n = n;
    if (regs) memcpy(d->x, regs, sizeof d->x);
    else memset(d->x, 0, sizeof d->x);
    d->x[0] = 0;
    d->err = d->halted = 0;
    d->fault_cause = -1;
    d->msg[0] = 0;
    d->exit_code = d->fault_pc = d->last_byte = 0;
    d->last_store = UINT32_MAX;
    d->tokens = d->records = d->instrs = d->stores = 0;
    d->fresh = 1;
    prepare(d);
}

void qd_dec_get_context(const qd_dec *d, uint32_t *anchor, uint32_t *n, uint32_t *regs) {
    if (anchor) *anchor = d->anchor;
    if (n) *n = d->n;
    if (regs) memcpy(regs, d->x, sizeof d->x);
}

uint32_t qd_dec_pc(const qd_dec *d) { return d->map.rom_base + 4 * (d->anchor + d->n); }
int qd_dec_wants_input(const qd_dec *d) {
    return !d->err && d->ph == PH_START && d->plan == P_INST && d->e->op == OP_INPUT;
}
int qd_dec_at_boundary(const qd_dec *d) { return d->ph == PH_START; }
int qd_dec_error(const qd_dec *d) { return d->err; }
const char *qd_dec_error_text(const qd_dec *d) { return d->msg; }
int qd_dec_halted(const qd_dec *d) { return d->halted; }
uint32_t qd_dec_exit_code(const qd_dec *d) { return d->exit_code; }
uint32_t qd_dec_last_byte(const qd_dec *d) { return d->last_byte; }
uint32_t qd_dec_last_store(const qd_dec *d) { return d->last_store; }

void qd_dec_fault(const qd_dec *d, int *cause, uint32_t *pc) {
    if (cause) *cause = d->fault_cause;
    if (pc) *pc = d->fault_pc;
}

void qd_dec_counts(const qd_dec *d, uint64_t *tokens, uint64_t *records, uint64_t *instructions,
                   uint64_t *stores) {
    if (tokens) *tokens = d->tokens;
    if (records) *records = d->records;
    if (instructions) *instructions = d->instrs;
    if (stores) *stores = d->stores;
}

/* Set up the next record from the context: the instruction at the PC, the families its record
   may start with (shape: the grammar alone), and in strict mode what the registers decide (allow,
   want_*). */
static void prepare(qd_dec *d) {
    d->ph = d->halted ? PH_DONE : PH_START;
    d->nv = 0;
    d->e = NULL;
    d->pc = qd_dec_pc(d);
    d->fault_mask = 0;
    d->want_j = d->want_byte = d->want_bank = d->want_k0 = -1;
    d->sem_err = QD_DEC_E_SHAPE;
    if (d->n > QD_N_MAX) { d->plan = P_BADN; return; }
    if (d->n == QD_N_MAX) {
        d->plan = P_R;
        d->shape = d->allow = FAM(R);
        return;
    }
    uint32_t wi = d->anchor + d->n;
    if (wi >= d->ic.words) { d->plan = P_BADPC; return; }
    const qd_inst *e = d->e = qd_fetch(&d->ic, wi);
    d->plan = P_INST;
    uint32_t a = d->x[e->rs1], b = d->x[e->rs2];
    /* strict: what the registers decide: the fault the record must be (>= 0), none (-2),
       a branch not taken (-1); -3: not known (lenient) */
    int strict = d->strict, fault = -3;
    switch (e->op) {
    case OP_LUI: case OP_ADDI: case OP_SLTI: case OP_SLTIU: case OP_XORI: case OP_ORI:
    case OP_ANDI: case OP_SLLI: case OP_SRLI: case OP_SRAI: case OP_ADD: case OP_SUB: case OP_SLL:
    case OP_SLT: case OP_SLTU: case OP_XOR: case OP_SRL: case OP_SRA: case OP_OR: case OP_AND:
    case OP_MUL: case OP_MULH: case OP_MULHSU: case OP_MULHU: case OP_SH1ADD: case OP_SH2ADD:
    case OP_SH3ADD:
        d->shape = d->allow = e->rd ? VALUE_START : FAM(NT);
        return;
    case OP_INPUT: /* rd != x0: the host-injected value record */
        d->shape = d->allow = VALUE_START;
        return;
    case OP_DIV: case OP_DIVU: case OP_REM: case OP_REMU:
        d->shape = d->allow = e->rd ? FAM(QS) : FAM(NT);
        if (strict && e->rd) d->want_k0 = qd_div_start(e->op, a, b);
        return;
    case OP_FENCE:
        d->shape = d->allow = FAM(NT);
        return;
    case OP_LB: case OP_LH: case OP_LW: case OP_LBU: case OP_LHU: {
        uint32_t ea = d->ea = a + e->imm, sz = (uint32_t)qd_access_size(e->op);
        d->shape = (e->rd ? VALUE_START : FAM(NT)) | FAM(FLT);
        d->fault_mask = 1u << QD_FAULT_MISALIGNED_LOAD | 1u << QD_FAULT_LOAD_OUT_OF_RANGE;
        if (strict)
            fault = (ea & (sz - 1)) ? QD_FAULT_MISALIGNED_LOAD
                    : (qd_in(ea, d->map.ram_base, d->map.ram_size)
                       || qd_in(ea, d->map.drom_base, d->map.drom_size)) ? -2
                    : QD_FAULT_LOAD_OUT_OF_RANGE;
        break;
    }
    case OP_SB: case OP_SH: case OP_SW: {
        uint32_t ea = d->ea = a + e->imm, sz = (uint32_t)qd_access_size(e->op);
        d->shape = (e->op == OP_SW ? FAM(ST0) | FAM(ST1) : FAM(V)) | FAM(FLT);
        d->fault_mask = 1u << QD_FAULT_MISALIGNED_STORE | 1u << QD_FAULT_STORE_OUT_OF_RANGE;
        d->sem_err = QD_DEC_E_STORE;
        if (strict) {
            fault = (ea & (sz - 1)) ? QD_FAULT_MISALIGNED_STORE
                    : qd_in(ea, d->map.ram_base, d->map.ram_size) ? -2
                    : QD_FAULT_STORE_OUT_OF_RANGE;
            if (fault == -2) d->want_bank = (int32_t)((ea - d->map.ram_base) >> 6 & 1);
        }
        break;
    }
    case OP_BEQ: case OP_BNE: case OP_BLT: case OP_BGE: case OP_BLTU: case OP_BGEU:
    case OP_JAL: case OP_JALR: {
        int branch = e->op != OP_JAL && e->op != OP_JALR;
        uint32_t t = e->op == OP_JALR ? (a + e->imm) & ~1u : e->imm;
        int bad = (t & 3) || !qd_in(t, d->map.rom_base, d->map.rom_size);
        d->shape = FAM(J) | FAM(FLT) | (branch ? FAM(NT) : 0);
        d->fault_mask = 1u << QD_FAULT_BAD_JUMP;
        d->sem_err = QD_DEC_E_JUMP;
        if (e->op != OP_JALR || strict) { /* a static target is known without registers */
            if (bad) d->shape &= ~FAM(J);
            else d->shape &= ~FAM(FLT), d->want_j = (int32_t)((t - d->map.rom_base) >> 2);
        }
        if (strict)
            fault = branch && !qd_taken(e->op, a, b) ? -1 : bad ? QD_FAULT_BAD_JUMP : -2;
        if (!(d->shape & FAM(FLT))) d->fault_mask = 0;
        d->allow = d->shape;
        if (fault == -1) d->allow = FAM(NT), d->fault_mask = 0;      /* not taken */
        else if (fault == -2) d->allow &= ~FAM(NT);                 /* taken, good target */
        else if (fault >= 0) d->allow = FAM(FLT);                   /* taken, bad target */
        return;
    }
    case OP_ECALL:
        d->shape = FAM(BYTE) | FAM(HALT) | FAM(FLT);
        d->fault_mask = 1u << QD_FAULT_ECALL;
        d->sem_err = QD_DEC_E_ECALL;
        d->allow = d->shape;
        if (strict) {
            if (d->x[17] == 1) d->allow = FAM(BYTE), d->want_byte = (int32_t)(d->x[10] & 0x1FF);
            else if (d->x[17] == 93) d->allow = FAM(HALT);
            else d->allow = FAM(FLT);
            if (d->allow != FAM(FLT)) d->fault_mask = 0;
        }
        return;
    case OP_EBREAK:
        d->shape = d->allow = FAM(FLT);
        d->fault_mask = 1u << QD_FAULT_EBREAK;
        return;
    default: /* illegal */
        d->shape = d->allow = FAM(FLT);
        d->fault_mask = 1u << QD_FAULT_ILLEGAL;
        return;
    }
    /* loads and stores: strict knows whether the access faults (fault >= 0) or not (-2) */
    d->allow = d->shape;
    if (fault >= 0) d->allow = FAM(FLT), d->fault_mask = 1u << fault;
    else if (fault == -2) d->allow &= ~FAM(FLT), d->fault_mask = 0;
}

/* Record an error (sticky) with the token's context; returns the code. */
static int fail(qd_dec *d, int code, int32_t tok, const char *fmt, ...) {
    char tb[32], what[200];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(what, sizeof what, fmt, ap);
    va_end(ap);
    d->err = code;
    if (d->plan == P_INST)
        snprintf(d->msg, sizeof d->msg,
                 "token %llu %s, record %llu at pc 0x%08x (%s, word 0x%08x, anchor %u n %u): %s",
                 (unsigned long long)d->tokens, qd_tok_name(tok, tb),
                 (unsigned long long)d->records, d->pc, qd_op_names[d->e->op], d->e->raw,
                 d->anchor, d->n, what);
    else
        snprintf(d->msg, sizeof d->msg,
                 "token %llu %s, record %llu at pc 0x%08x (anchor %u n %u): %s",
                 (unsigned long long)d->tokens, qd_tok_name(tok, tb),
                 (unsigned long long)d->records, d->pc, d->anchor, d->n, what);
    return code;
}

static void fams_text(uint32_t mask, char *buf, size_t len) {
    buf[0] = 0;
    for (int f = 0; f < QD_N_FAMS; f++)
        if (mask >> f & 1) {
            size_t l = strlen(buf);
            snprintf(buf + l, len - l, "%s%s", l ? "/" : "", qd_fam_names[f]);
        }
}

/* The record committed: the next one starts. */
static inline int commit(qd_dec *d, int ev) {
    d->records++;
    if (d->plan == P_INST) d->instrs++;
    prepare(d);
    return ev;
}

/* A value record's word arrived (the rd write of a value / input / DIV record). */
static inline int commit_value(qd_dec *d, int32_t tok, uint32_t v) {
    if (qd_value_size(v) != d->nv + 1)
        return fail(d, QD_DEC_E_VALUE, tok, "value 0x%08x in %d tokens (canonical: %d)", v,
                    d->nv + 1, qd_value_size(v));
    d->x[d->e->rd] = v;
    d->n++;
    return commit(d, QD_DEC_COMMIT);
}

static int apply_store(qd_dec *d, int32_t tok, int bank, uint32_t word, int merged) {
    const qd_inst *e = d->e;
    uint32_t ea = d->ea, sz = (uint32_t)qd_access_size(e->op);
    if ((ea & (sz - 1)) || !qd_in(ea, d->map.ram_base, d->map.ram_size))
        return fail(d, QD_DEC_E_STORE, tok, "store to 0x%08x: outside RAM or misaligned", ea);
    uint32_t off = (ea - d->map.ram_base) & ~3u;
    uint8_t *p = d->mem.ram + off;
    if (d->want_bank >= 0 && bank != d->want_bank)
        return fail(d, QD_DEC_E_STORE, tok, "bank %d for address 0x%08x (want %d)", bank, ea,
                    d->want_bank);
    if (merged && d->strict) {
        uint32_t sh = 8 * (ea & 3), mask = sz == 1 ? 0xFFu : 0xFFFFu;
        uint32_t want = (qd_ld32(p) & ~(mask << sh)) | (d->x[e->rs2] & mask) << sh;
        if (word != want)
            return fail(d, QD_DEC_E_STORE, tok, "merged word 0x%08x at 0x%08x (want 0x%08x)", word,
                        ea, want);
    }
    qd_st32(p, word);
    d->last_store = d->map.ram_base + off;
    d->stores++;
    d->n++;
    return commit(d, QD_DEC_COMMIT);
}

static int feed_start(qd_dec *d, int32_t tok, int fam, uint32_t p) {
    char eb[120];
    if (d->plan == P_BADPC)
        return fail(d, QD_DEC_E_PC, tok, "the PC (word %u) is outside the ROM", d->anchor + d->n);
    if (d->plan == P_BADN) return fail(d, QD_DEC_E_REFRESH, tok, "n = %u > %d", d->n, QD_N_MAX);
    if (!(d->allow >> fam & 1)) {
        int code = d->plan == P_R ? QD_DEC_E_REFRESH
                   : (fam == QD_FAM_FLT || d->allow == FAM(FLT)) ? QD_DEC_E_FAULT
                   : (d->shape >> fam & 1) ? d->sem_err : QD_DEC_E_SHAPE;
        fams_text(d->allow, eb, sizeof eb);
        return fail(d, code, tok, "expected %s", eb);
    }
    d->last_store = UINT32_MAX;
    switch (fam) {
    case QD_FAM_R:
        if (p != d->anchor + QD_N_MAX)
            return fail(d, QD_DEC_E_REFRESH, tok, "expected R(%u)", d->anchor + QD_N_MAX);
        d->anchor = p;
        d->n = 0;
        return commit(d, QD_DEC_COMMIT);
    case QD_FAM_F0:
        d->nv = 0;
        return commit_value(d, tok, p | (p >> 14 ? 0xFFFF8000u : 0));
    case QD_FAM_V:
        d->vc[0] = p;
        d->nv = 1;
        d->ph = d->allow & FAM(F0) ? PH_VAL : PH_SBH;
        return QD_DEC_MORE;
    case QD_FAM_QS:
        if (p > 8) return fail(d, QD_DEC_E_DIV, tok, "k0 = %u > 8", p);
        if (d->want_k0 >= 0 && (int32_t)p != d->want_k0)
            return fail(d, QD_DEC_E_DIV, tok, "k0 = %u, want %d", p, d->want_k0);
        d->k = (int)p;
        d->nv = 0;
        d->ph = p == 8 ? PH_VAL : PH_QD;
        return QD_DEC_MORE;
    case QD_FAM_NT:
        d->n++;
        return commit(d, QD_DEC_COMMIT);
    case QD_FAM_J: {
        const qd_inst *e = d->e;
        if (d->want_j >= 0 && (int32_t)p != d->want_j)
            return fail(d, QD_DEC_E_JUMP, tok, "J(%u), want J(%d)", p, d->want_j);
        if ((e->op == OP_JAL || e->op == OP_JALR) && e->rd) d->x[e->rd] = d->pc + 4;
        d->anchor = p;
        d->n = 0;
        return commit(d, QD_DEC_COMMIT);
    }
    case QD_FAM_ST0: case QD_FAM_ST1:
        return apply_store(d, tok, fam == QD_FAM_ST1, d->x[d->e->rs2], 0);
    case QD_FAM_BYTE:
        if (d->want_byte >= 0 && (int32_t)p != d->want_byte)
            return fail(d, QD_DEC_E_ECALL, tok, "BYTE(%u), want BYTE(%d) (a0 & 511)", p,
                        d->want_byte);
        d->last_byte = p;
        d->n++;
        return commit(d, p == QD_YIELD_BYTE ? QD_DEC_YIELD : QD_DEC_BYTE);
    case QD_FAM_HALT: /* ecall exit */
        d->exit_code = d->x[10];
        d->halted = QD_DEC_HALT;
        d->n++;
        return commit(d, QD_DEC_HALT);
    case QD_FAM_FLT:
        if (!(d->fault_mask >> p & 1))
            return fail(d, QD_DEC_E_FAULT, tok, "fault cause %u (%s) cannot happen here", p,
                        p < QD_N_FAULTS ? qd_fault_names[p] : "?");
        d->fcause = (int)p;
        d->hex = 0;
        d->ph = PH_HEX;
        return QD_DEC_MORE;
    }
    return fail(d, QD_DEC_E_SHAPE, tok, "unexpected"); /* (allow never holds other families) */
}

int qd_dec_feed(qd_dec *d, int32_t tok) {
    uint32_t p;
    if (d->err) return d->err;
    int fam = qd_tok_family(tok, &p);
    if (fam < 0) return fail(d, QD_DEC_E_TOKEN, tok, "not a chip token");
    if (fam == QD_FAM_GO && d->fresh) { /* the start of a run (after a snapshot) */
        d->fresh = 0;
        d->tokens++;
        return QD_DEC_MORE;
    }
    d->fresh = 0;
    int r;
    switch (d->ph) {
    case PH_START:
        r = feed_start(d, tok, fam, p);
        break;
    case PH_VAL: /* V.. then F0 | F1 | F2 by the chunks so far */
        if (fam == QD_FAM_V && d->nv < 2) {
            d->vc[d->nv++] = p;
            r = QD_DEC_MORE;
        } else if (fam == QD_FAM_F0 && d->nv == 0) {
            r = commit_value(d, tok, p | (p >> 14 ? 0xFFFF8000u : 0));
        } else if (fam == QD_FAM_F1 && d->nv == 1) {
            r = commit_value(d, tok, d->vc[0] | p << 15 | (p >> 14 ? 0xC0000000u : 0));
        } else if (fam == QD_FAM_F2 && d->nv == 2) {
            r = commit_value(d, tok, d->vc[0] | d->vc[1] << 15 | p << 30);
        } else {
            static const char *const want[3] = {"V/F0", "V/F1", "F2"};
            r = fail(d, fam == QD_FAM_V || fam == QD_FAM_F0 || fam == QD_FAM_F1 || fam == QD_FAM_F2
                     ? QD_DEC_E_VALUE : QD_DEC_E_SHAPE, tok, "expected %s", want[d->nv]);
        }
        break;
    case PH_SBH: /* V V V(0..3), then SS0 / SS1 */
        if (fam == QD_FAM_V && d->nv < 3) {
            if (d->nv == 2 && p > 3) {
                r = fail(d, QD_DEC_E_VALUE, tok, "the merged word's top chunk is %u > 3", p);
                break;
            }
            d->vc[d->nv++] = p;
            r = QD_DEC_MORE;
        } else if ((fam == QD_FAM_SS0 || fam == QD_FAM_SS1) && d->nv == 3) {
            r = apply_store(d, tok, fam == QD_FAM_SS1, d->vc[0] | d->vc[1] << 15 | d->vc[2] << 30,
                            1);
        } else {
            r = fail(d, QD_DEC_E_SHAPE, tok, "expected %s", d->nv < 3 ? "V" : "SS0/SS1");
        }
        break;
    case PH_QD:
        if (fam == QD_FAM_QD && (int)(p >> 4) == d->k) {
            if (++d->k == 8) d->ph = PH_VAL;
            r = QD_DEC_MORE;
        } else {
            r = fail(d, fam == QD_FAM_QD ? QD_DEC_E_DIV : QD_DEC_E_SHAPE, tok,
                     "expected QD with k = %d", d->k);
        }
        break;
    case PH_HEX: {
        uint32_t want = d->pc >> 4 * (7 - d->hex) & 15;
        if (fam != QD_FAM_HEX) {
            r = fail(d, QD_DEC_E_SHAPE, tok, "expected HEX");
        } else if (p != want) {
            r = fail(d, QD_DEC_E_FAULT, tok, "fault PC digit %d is %u, want %u", d->hex, p, want);
        } else {
            if (++d->hex == 8) d->ph = PH_FHALT;
            r = QD_DEC_MORE;
        }
        break;
    }
    case PH_FHALT:
        if (fam != QD_FAM_HALT) {
            r = fail(d, QD_DEC_E_SHAPE, tok, "expected HALT");
        } else {
            d->fault_cause = d->fcause;
            d->fault_pc = d->pc;
            d->halted = QD_DEC_FAULT;
            r = commit(d, QD_DEC_FAULT);
        }
        break;
    default:
        r = fail(d, QD_DEC_E_HALTED, tok, "the stream has ended");
    }
    if (r >= 0) d->tokens++;
    return r;
}

int qd_dec_feed_many(qd_dec *d, const int32_t *toks, long n, int stop_mask, long *consumed) {
    int r = d->err ? d->err : QD_DEC_MORE;
    long i = 0;
    while (i < n && r >= 0) {
        r = qd_dec_feed(d, toks[i]);
        if (r < 0) break;
        i++;
        if (r == QD_DEC_HALT || r == QD_DEC_FAULT || (stop_mask >> r & 1)) break;
        if (r != QD_DEC_MORE && (stop_mask & QD_DEC_STOP_INPUT) && qd_dec_wants_input(d)) break;
    }
    if (consumed) *consumed = i;
    return r;
}
