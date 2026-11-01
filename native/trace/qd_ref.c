/* qd_ref: the reference traced machine. record() emits the next record's tokens and commits it,
   token for token as Qwenputer's chipformer/rv/ref.py Machine.record() (MIT (c) 2026 Divyansh
   Agrawal), with qwendoom's address map (a runtime qd_map), illegal digit-file ops, and the input
   port: `csrr rd, time` (rd != x0) reads the next word of the input queue and its record is that
   word's value record (host-owned: in the model run the host injects it). */
#include <stdlib.h>
#include <string.h>

#include "qd_isa.h"

struct qd_ref {
    qd_mem mem;
    qd_map map;
    qd_icache ic;
    uint32_t x[32];
    uint32_t anchor, n;
    int status;
    uint32_t exit_code, fault_pc;
    int fault_cause;
    uint32_t last_store;
    uint64_t steps, records, tokens;
    uint32_t *inq; /* the input queue: a ring of inq_cap words */
    size_t inq_cap, inq_head, inq_len;
};

void qd_map_qwendoom(qd_map *m) {
    *m = (qd_map){QD_RAM_BASE, QD_RAM_SIZE, QD_DROM_BASE, QD_DROM_SIZE, QD_ROM_BASE, QD_ROM_SIZE,
                  QD_CSR_INPUT};
}

void qd_map_upstream(qd_map *m) {
    *m = (qd_map){QD_UP_RAM_BASE, QD_UP_RAM_SIZE, QD_UP_DROM_BASE, QD_UP_DROM_SIZE,
                  QD_UP_ROM_BASE, QD_UP_ROM_SIZE, -1};
}

int qd_icache_init(qd_icache *c, const uint8_t *rom, const qd_map *map) {
    c->rom = rom;
    c->base = map->rom_base;
    c->words = map->rom_size / 4;
    c->input_csr = map->input_csr;
    c->e = calloc(c->words ? c->words : 1, sizeof *c->e); /* raw 0 decodes to ILLEGAL (op 0) */
    return c->e != NULL;
}

void qd_icache_free(qd_icache *c) {
    free(c->e);
    c->e = NULL;
}

int qd_value_record(uint32_t v, int32_t out[3]) { return qd_emit_value(out, v); }

int qd_div_record(int op, uint32_t a, uint32_t b, int32_t out[QD_MAX_RECORD]) {
    static const int ops[4] = {OP_DIV, OP_DIVU, OP_REM, OP_REMU};
    uint32_t res;
    return qd_emit_div(out, ops[op & 3], a, b, &res);
}

qd_ref *qd_ref_new(const qd_mem *mem, const qd_map *map) {
    if (map->rom_size / 4 > 65536) return NULL;
    qd_ref *r = calloc(1, sizeof *r);
    if (!r) return NULL;
    r->mem = *mem;
    r->map = *map;
    if (!qd_icache_init(&r->ic, mem->rom, map)) {
        free(r);
        return NULL;
    }
    qd_ref_set_context(r, 0, 0, NULL);
    return r;
}

void qd_ref_free(qd_ref *r) {
    if (!r) return;
    qd_icache_free(&r->ic);
    free(r->inq);
    free(r);
}

void qd_ref_set_context(qd_ref *r, uint32_t anchor, uint32_t n, const uint32_t *regs) {
    r->anchor = anchor;
    r->n = n;
    if (regs) memcpy(r->x, regs, sizeof r->x);
    else memset(r->x, 0, sizeof r->x);
    r->x[0] = 0;
    r->status = n > QD_N_MAX ? QD_REF_ERROR : QD_REF_RUNNING;
    r->exit_code = r->fault_pc = 0;
    r->fault_cause = -1;
    r->last_store = UINT32_MAX;
    r->steps = r->records = r->tokens = 0;
    r->inq_head = r->inq_len = 0;
}

void qd_ref_get_context(const qd_ref *r, uint32_t *anchor, uint32_t *n, uint32_t *regs) {
    if (anchor) *anchor = r->anchor;
    if (n) *n = r->n;
    if (regs) memcpy(regs, r->x, sizeof r->x);
}

uint32_t qd_ref_pc(const qd_ref *r) { return r->map.rom_base + 4 * (r->anchor + r->n); }

uint32_t qd_ref_inst(const qd_ref *r) {
    uint32_t wi = r->anchor + r->n;
    return r->n == QD_N_MAX || wi >= r->ic.words ? 0 : qd_ld32(r->mem.rom + 4 * (size_t)wi);
}
void qd_ref_set_reg(qd_ref *r, int i, uint32_t v) { if (i > 0 && i < 32) r->x[i] = v; }
int qd_ref_status(const qd_ref *r) { return r->status; }
uint32_t qd_ref_exit_code(const qd_ref *r) { return r->exit_code; }
uint32_t qd_ref_last_store(const qd_ref *r) { return r->last_store; }

void qd_ref_fault(const qd_ref *r, int *cause, uint32_t *pc) {
    if (cause) *cause = r->fault_cause;
    if (pc) *pc = r->fault_pc;
}

void qd_ref_counts(const qd_ref *r, uint64_t *steps, uint64_t *records, uint64_t *tokens) {
    if (steps) *steps = r->steps;
    if (records) *records = r->records;
    if (tokens) *tokens = r->tokens;
}

void qd_ref_push_input(qd_ref *r, const uint32_t *words, size_t n) {
    if (r->inq_len + n > r->inq_cap) {
        size_t cap = r->inq_cap ? r->inq_cap : 256;
        while (cap < r->inq_len + n) cap *= 2;
        uint32_t *q = malloc(cap * sizeof *q);
        if (!q) abort();
        for (size_t i = 0; i < r->inq_len; i++) q[i] = r->inq[(r->inq_head + i) % r->inq_cap];
        free(r->inq);
        r->inq = q;
        r->inq_cap = cap;
        r->inq_head = 0;
    }
    for (size_t i = 0; i < n; i++) r->inq[(r->inq_head + r->inq_len++) % r->inq_cap] = words[i];
    if (r->status == QD_REF_NEED_INPUT && n) r->status = QD_REF_RUNNING;
}

size_t qd_ref_input_pending(const qd_ref *r) { return r->inq_len; }
void qd_ref_clear_input(qd_ref *r) { r->inq_head = r->inq_len = 0; }

int qd_ref_wants_input(qd_ref *r) {
    uint32_t wi = r->anchor + r->n;
    if (r->status != QD_REF_RUNNING && r->status != QD_REF_NEED_INPUT) return 0;
    if (r->n >= QD_N_MAX || wi >= r->ic.words) return 0;
    return qd_fetch(&r->ic, wi)->op == OP_INPUT;
}

/* One record into o; returns its length, 0 if none can run (status says why). */
static inline int step(qd_ref *r, int32_t *o, int *host) {
    uint32_t *x = r->x;
    r->last_store = UINT32_MAX;
    if (r->n == QD_N_MAX) { /* the anchor refresh comes before the instruction */
        uint32_t w = r->anchor + r->n;
        if (w >= r->ic.words) {
            r->status = QD_REF_ERROR;
            return 0;
        }
        r->anchor = w;
        r->n = 0;
        o[0] = QD_TOK_R + (int32_t)w;
        return 1;
    }
    uint32_t wi = r->anchor + r->n;
    if (wi >= r->ic.words) {
        r->status = QD_REF_ERROR;
        return 0;
    }
    const qd_inst *e = qd_fetch(&r->ic, wi);
    uint32_t pc = r->map.rom_base + 4 * wi;
    uint32_t a = x[e->rs1], b = x[e->rs2], v;
    int cause;

#define COMMIT_RD(val)                                         \
    do {                                                       \
        v = (val);                                             \
        r->n++;                                                \
        if (!e->rd) { o[0] = QD_TOK_NT; return 1; }            \
        x[e->rd] = v;                                          \
        return qd_emit_value(o, v);                            \
    } while (0)
#define FAULT(c) do { cause = (c); goto fault; } while (0)

    switch (e->op) {
    case OP_LUI:
        r->steps++;
        COMMIT_RD(e->imm);
    case OP_INPUT:
        if (!r->inq_len) {
            r->status = QD_REF_NEED_INPUT;
            return 0;
        }
        r->steps++;
        *host = 1;
        v = r->inq[r->inq_head];
        r->inq_head = (r->inq_head + 1) % r->inq_cap;
        r->inq_len--;
        COMMIT_RD(v);
    case OP_ADDI: case OP_SLTI: case OP_SLTIU: case OP_XORI: case OP_ORI: case OP_ANDI:
    case OP_SLLI: case OP_SRLI: case OP_SRAI:
        r->steps++;
        COMMIT_RD(qd_alu(e->op, a, e->imm));
    case OP_ADD: case OP_SUB: case OP_SLL: case OP_SLT: case OP_SLTU: case OP_XOR: case OP_SRL:
    case OP_SRA: case OP_OR: case OP_AND: case OP_MUL: case OP_MULH: case OP_MULHSU:
    case OP_MULHU: case OP_SH1ADD: case OP_SH2ADD: case OP_SH3ADD:
        r->steps++;
        COMMIT_RD(qd_alu(e->op, a, b));
    case OP_DIV: case OP_DIVU: case OP_REM: case OP_REMU:
        r->steps++;
        r->n++;
        if (!e->rd) {
            o[0] = QD_TOK_NT;
            return 1;
        } else {
            int len = qd_emit_div(o, e->op, a, b, &v);
            x[e->rd] = v;
            return len;
        }
    case OP_LB: case OP_LH: case OP_LW: case OP_LBU: case OP_LHU: {
        r->steps++;
        uint32_t ea = a + e->imm, sz = (uint32_t)qd_access_size(e->op), word;
        if (ea & (sz - 1)) FAULT(QD_FAULT_MISALIGNED_LOAD);
        if (qd_in(ea, r->map.ram_base, r->map.ram_size))
            word = qd_ld32(r->mem.ram + ((ea - r->map.ram_base) & ~3u));
        else if (qd_in(ea, r->map.drom_base, r->map.drom_size))
            word = qd_ld32(r->mem.drom + ((ea - r->map.drom_base) & ~3u));
        else
            FAULT(QD_FAULT_LOAD_OUT_OF_RANGE);
        word >>= 8 * (ea & 3);
        switch (e->op) {
        case OP_LB: COMMIT_RD(qd_sx(word, 8));
        case OP_LH: COMMIT_RD(qd_sx(word, 16));
        case OP_LBU: COMMIT_RD(word & 0xFF);
        case OP_LHU: COMMIT_RD(word & 0xFFFF);
        default: COMMIT_RD(word);
        }
    }
    case OP_SB: case OP_SH: case OP_SW: {
        r->steps++;
        uint32_t ea = a + e->imm, sz = (uint32_t)qd_access_size(e->op);
        if (ea & (sz - 1)) FAULT(QD_FAULT_MISALIGNED_STORE);
        if (!qd_in(ea, r->map.ram_base, r->map.ram_size)) FAULT(QD_FAULT_STORE_OUT_OF_RANGE);
        uint32_t off = (ea - r->map.ram_base) & ~3u, bank = off >> 6 & 1; /* word index bit 4 */
        uint8_t *p = r->mem.ram + off;
        r->n++;
        r->last_store = r->map.ram_base + off;
        if (e->op == OP_SW) {
            qd_st32(p, b);
            o[0] = bank ? QD_TOK_ST1 : QD_TOK_ST0;
            return 1;
        }
        uint32_t sh = 8 * (ea & 3), mask = sz == 1 ? 0xFFu : 0xFFFFu;
        uint32_t m = (qd_ld32(p) & ~(mask << sh)) | (b & mask) << sh;
        qd_st32(p, m);
        o[0] = QD_TOK_V + (int32_t)(m & 0x7FFF);
        o[1] = QD_TOK_V + (int32_t)(m >> 15 & 0x7FFF);
        o[2] = QD_TOK_V + (int32_t)(m >> 30);
        o[3] = bank ? QD_TOK_SS1 : QD_TOK_SS0;
        return 4;
    }
    case OP_BEQ: case OP_BNE: case OP_BLT: case OP_BGE: case OP_BLTU: case OP_BGEU:
        r->steps++;
        if (!qd_taken(e->op, a, b)) {
            r->n++;
            o[0] = QD_TOK_NT;
            return 1;
        }
        v = e->imm;
        goto jump;
    case OP_JAL:
        r->steps++;
        v = e->imm;
        goto jump;
    case OP_JALR:
        r->steps++;
        v = (a + e->imm) & ~1u;
        goto jump;
    case OP_ECALL:
        r->steps++;
        if (x[17] == 1) { /* putchar(a0): a byte, or 256.. a special token (256 = YIELD) */
            r->n++;
            o[0] = QD_TOK_BYTE + (int32_t)(x[10] & 0x1FF);
            return 1;
        }
        if (x[17] == 93) { /* exit(a0) */
            r->n++;
            r->status = QD_REF_HALTED;
            r->exit_code = x[10];
            o[0] = QD_TOK_HALT;
            return 1;
        }
        FAULT(QD_FAULT_ECALL);
    case OP_EBREAK:
        r->steps++;
        FAULT(QD_FAULT_EBREAK);
    case OP_FENCE:
        r->steps++;
        r->n++;
        o[0] = QD_TOK_NT;
        return 1;
    default: /* OP_ILLEGAL: unknown encodings, CSR writes and unknown CSRs, the digit file */
        r->steps++;
        FAULT(QD_FAULT_ILLEGAL);
    }

jump: /* v: the target; the link (JAL / JALR) is written only when the jump is good */
    if ((v & 3) || !qd_in(v, r->map.rom_base, r->map.rom_size)) FAULT(QD_FAULT_BAD_JUMP);
    if ((e->op == OP_JAL || e->op == OP_JALR) && e->rd) x[e->rd] = pc + 4;
    r->anchor = (v - r->map.rom_base) >> 2;
    r->n = 0;
    o[0] = QD_TOK_J + (int32_t)r->anchor;
    return 1;

fault:
    r->status = QD_REF_FAULTED;
    r->fault_cause = cause;
    r->fault_pc = pc;
    return qd_emit_fault(o, cause, pc);
#undef COMMIT_RD
#undef FAULT
}

int qd_ref_record(qd_ref *r, int32_t *out, int *flags) {
    int host = 0, len;
    if (flags) *flags = 0;
    if (r->status == QD_REF_NEED_INPUT && r->inq_len) r->status = QD_REF_RUNNING;
    if (r->status != QD_REF_RUNNING) return 0;
    len = step(r, out, &host);
    if (len) {
        r->records++;
        r->tokens += (uint64_t)len;
        if (flags && host) *flags = QD_TF_HOST;
    }
    return len;
}

long qd_ref_run(qd_ref *r, int32_t *toks, uint8_t *flags, long cap, long max_records,
                int stop_mask, int *reason) {
    int32_t scratch[QD_MAX_RECORD];
    long len = 0, recs = 0, ntok = 0;
    int why = QD_RUN_LIMIT;
    if (r->status == QD_REF_NEED_INPUT && r->inq_len) r->status = QD_REF_RUNNING;
    for (;;) {
        if (max_records >= 0 && recs >= max_records) { why = QD_RUN_LIMIT; break; }
        if (r->status != QD_REF_RUNNING) {
            why = r->status == QD_REF_HALTED ? QD_RUN_HALT : r->status == QD_REF_FAULTED
                  ? QD_RUN_FAULT : r->status == QD_REF_NEED_INPUT ? QD_RUN_INPUT : QD_RUN_ERROR;
            break;
        }
        if (toks && cap - len < QD_MAX_RECORD) { why = QD_RUN_FULL; break; }
        int32_t *o = toks ? toks + len : scratch;
        int host = 0, n = step(r, o, &host);
        if (!n) continue; /* the status changed: reported above */
        if (flags && toks) {
            memset(flags + len, host ? QD_TF_HOST : 0, (size_t)n);
            flags[len] |= QD_TF_FIRST;
        }
        recs++;
        r->records++;
        r->tokens += (uint64_t)n;
        ntok += n;
        if (toks) len += n;
        if (o[0] >= QD_TOK_BYTE && o[0] < QD_TOK_BYTE + QD_TOK_BYTE_SIZE) {
            int yield = o[0] == QD_TOK_BYTE + (int32_t)QD_YIELD_BYTE;
            if (yield && (stop_mask & QD_STOP_YIELD)) { why = QD_RUN_YIELD; break; }
            if (!yield && (stop_mask & QD_STOP_BYTE)) { why = QD_RUN_BYTE; break; }
        }
    }
    if (reason) *reason = why;
    return ntok;
}
