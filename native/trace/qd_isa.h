/* RV32IM + Zba decode and semantics shared by qd_ref and qd_dec (internal).

   Derived from Qwenputer's chipformer/rv/ref.py (decode, alu, taken, div_start, div_parts,
   rom_word; MIT (c) 2026 Divyansh Agrawal). Like the circuit's pre-decoded ROM word, the decode
   folds static facts in: AUIPC and the CSR stubs become LUI with the final value, branch and JAL
   targets are absolute. qwendoom's changes: custom-0 / custom-1 (the digit file) decode as
   ILLEGAL, and a pure read of the map's input CSR with rd != x0 is OP_INPUT. */
#ifndef QD_ISA_H
#define QD_ISA_H

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "qd_trace.h"

enum {
    OP_ILLEGAL, OP_LUI /* rd = imm: LUI, AUIPC (pc folded in), the CSR stubs */, OP_INPUT,
    OP_ADDI, OP_SLTI, OP_SLTIU, OP_XORI, OP_ORI, OP_ANDI, OP_SLLI, OP_SRLI, OP_SRAI,
    OP_ADD, OP_SUB, OP_SLL, OP_SLT, OP_SLTU, OP_XOR, OP_SRL, OP_SRA, OP_OR, OP_AND,
    OP_MUL, OP_MULH, OP_MULHSU, OP_MULHU, OP_SH1ADD, OP_SH2ADD, OP_SH3ADD,
    OP_DIV, OP_DIVU, OP_REM, OP_REMU,
    OP_LB, OP_LH, OP_LW, OP_LBU, OP_LHU, OP_SB, OP_SH, OP_SW,
    OP_BEQ, OP_BNE, OP_BLT, OP_BGE, OP_BLTU, OP_BGEU, OP_JAL, OP_JALR,
    OP_ECALL, OP_EBREAK, OP_FENCE,
    OP_COUNT
};

static const char *const qd_op_names[OP_COUNT] = {
    "illegal", "lui", "input", "addi", "slti", "sltiu", "xori", "ori", "andi", "slli", "srli",
    "srai", "add", "sub", "sll", "slt", "sltu", "xor", "srl", "sra", "or", "and", "mul", "mulh",
    "mulhsu", "mulhu", "sh1add", "sh2add", "sh3add", "div", "divu", "rem", "remu", "lb", "lh",
    "lw", "lbu", "lhu", "sb", "sh", "sw", "beq", "bne", "blt", "bge", "bltu", "bgeu", "jal",
    "jalr", "ecall", "ebreak", "fence"};

typedef struct {
    uint32_t raw;          /* the instruction word this entry was decoded from */
    uint8_t op, rd, rs1, rs2;
    uint32_t imm;          /* sign-extended immediate; LUI: the value; branch / JAL: the target */
} qd_inst;

static inline uint32_t qd_sx(uint32_t v, int bits) {
    return (uint32_t)((int32_t)(v << (32 - bits)) >> (32 - bits));
}

static inline uint32_t qd_ld32(const uint8_t *p) { uint32_t v; memcpy(&v, p, 4); return v; }
static inline void qd_st32(uint8_t *p, uint32_t v) { memcpy(p, &v, 4); }

/* ref.decode + ref.rom_word's folding, for the word w at pc */
static inline void qd_decode(uint32_t w, uint32_t pc, int32_t input_csr, qd_inst *o) {
    uint32_t opc = w & 0x7F, rd = w >> 7 & 31, f3 = w >> 12 & 7, rs1 = w >> 15 & 31,
             rs2 = w >> 20 & 31, f7 = w >> 25;
    uint32_t i_imm = qd_sx(w >> 20, 12);
    uint32_t s_imm = qd_sx((w >> 25) << 5 | (w >> 7 & 31), 12);
    uint32_t b_imm = qd_sx((w >> 31 & 1) << 12 | (w >> 7 & 1) << 11 | (w >> 25 & 63) << 5
                           | (w >> 8 & 15) << 1, 13);
    uint32_t j_imm = qd_sx((w >> 31 & 1) << 20 | (w >> 12 & 0xFF) << 12 | (w >> 20 & 1) << 11
                           | (w >> 21 & 0x3FF) << 1, 21);
    memset(o, 0, sizeof *o);
    o->raw = w;
    o->op = OP_ILLEGAL;
#define SET(OP, RD, RS1, RS2, IMM) \
    do { o->op = (OP); o->rd = (uint8_t)(RD); o->rs1 = (uint8_t)(RS1); o->rs2 = (uint8_t)(RS2); \
         o->imm = (IMM); return; } while (0)
    switch (opc) {
    case 0x37: SET(OP_LUI, rd, 0, 0, w & 0xFFFFF000u);
    case 0x17: SET(OP_LUI, rd, 0, 0, pc + (w & 0xFFFFF000u));
    case 0x6F: SET(OP_JAL, rd, 0, 0, pc + j_imm);
    case 0x67:
        if (f3 == 0) SET(OP_JALR, rd, rs1, 0, i_imm);
        return;
    case 0x63: {
        static const uint8_t br[8] = {OP_BEQ, OP_BNE, 0, 0, OP_BLT, OP_BGE, OP_BLTU, OP_BGEU};
        if (f3 != 2 && f3 != 3) SET(br[f3], 0, rs1, rs2, pc + b_imm);
        return;
    }
    case 0x03: {
        static const uint8_t ld[8] = {OP_LB, OP_LH, OP_LW, 0, OP_LBU, OP_LHU, 0, 0};
        if (ld[f3]) SET(ld[f3], rd, rs1, 0, i_imm);
        return;
    }
    case 0x23: {
        static const uint8_t st[3] = {OP_SB, OP_SH, OP_SW};
        if (f3 < 3) SET(st[f3], 0, rs1, rs2, s_imm);
        return;
    }
    case 0x13:
        if (f3 == 1) {
            if (f7 == 0) SET(OP_SLLI, rd, rs1, 0, rs2);
            return;
        }
        if (f3 == 5) {
            if (f7 == 0) SET(OP_SRLI, rd, rs1, 0, rs2);
            if (f7 == 0x20) SET(OP_SRAI, rd, rs1, 0, rs2);
            return;
        }
        {
            static const uint8_t im[8] = {OP_ADDI, 0, OP_SLTI, OP_SLTIU, OP_XORI, 0, OP_ORI,
                                          OP_ANDI};
            SET(im[f3], rd, rs1, 0, i_imm);
        }
    case 0x33:
        if (f7 == 0) {
            static const uint8_t r[8] = {OP_ADD, OP_SLL, OP_SLT, OP_SLTU, OP_XOR, OP_SRL, OP_OR,
                                         OP_AND};
            SET(r[f3], rd, rs1, rs2, 0);
        }
        if (f7 == 0x20 && f3 == 0) SET(OP_SUB, rd, rs1, rs2, 0);
        if (f7 == 0x20 && f3 == 5) SET(OP_SRA, rd, rs1, rs2, 0);
        if (f7 == 1) {
            static const uint8_t m[8] = {OP_MUL, OP_MULH, OP_MULHSU, OP_MULHU, OP_DIV, OP_DIVU,
                                         OP_REM, OP_REMU};
            SET(m[f3], rd, rs1, rs2, 0);
        }
        if (f7 == 0x10 && f3 == 2) SET(OP_SH1ADD, rd, rs1, rs2, 0);
        if (f7 == 0x10 && f3 == 4) SET(OP_SH2ADD, rd, rs1, rs2, 0);
        if (f7 == 0x10 && f3 == 6) SET(OP_SH3ADD, rd, rs1, rs2, 0);
        return;
    case 0x0F: SET(OP_FENCE, 0, 0, 0, 0);
    case 0x73:
        if (w == 0x00000073u) SET(OP_ECALL, 0, 0, 0, 0);
        if (w == 0x00100073u) SET(OP_EBREAK, 0, 0, 0, 0);
        if (f3 == 2 || f3 == 3 || f3 == 6 || f3 == 7) { /* csrrs / csrrc / csrrsi / csrrci */
            uint32_t csr = w >> 20;
            if (rs1 != 0) return; /* a CSR write: illegal */
            if (input_csr >= 0 && csr == (uint32_t)input_csr) {
                if (rd) SET(OP_INPUT, rd, 0, 0, 0);
                SET(OP_LUI, 0, 0, 0, 0); /* csrr x0, time: NT, reads nothing */
            }
            switch (csr) { /* ref.CSR_STUBS */
            case 0xC00: case 0xC01: case 0xC02: case 0xC80: case 0xC81: case 0xC82: case 0xF14:
                SET(OP_LUI, rd, 0, 0, 0);
            case 0x301: SET(OP_LUI, rd, 0, 0, QD_MISA);
            }
        }
        return; /* csrrw / csrrwi, other CSRs, mret, wfi...: illegal */
    default:
        return; /* custom-0 (0x0B), custom-1 (0x2B) and the rest: illegal */
    }
#undef SET
}

/* register-register / register-immediate results (ref.alu) for the non-DIV ops */
static inline uint32_t qd_alu(int op, uint32_t a, uint32_t b) {
    switch (op) {
    case OP_ADD: case OP_ADDI: return a + b;
    case OP_SUB: return a - b;
    case OP_SLL: case OP_SLLI: return a << (b & 31);
    case OP_SRL: case OP_SRLI: return a >> (b & 31);
    case OP_SRA: case OP_SRAI: return (uint32_t)((int32_t)a >> (b & 31));
    case OP_SLT: case OP_SLTI: return (int32_t)a < (int32_t)b;
    case OP_SLTU: case OP_SLTIU: return a < b;
    case OP_XOR: case OP_XORI: return a ^ b;
    case OP_OR: case OP_ORI: return a | b;
    case OP_AND: case OP_ANDI: return a & b;
    case OP_MUL: return a * b;
    case OP_MULH: return (uint32_t)((uint64_t)((int64_t)(int32_t)a * (int64_t)(int32_t)b) >> 32);
    case OP_MULHSU: return (uint32_t)((uint64_t)((int64_t)(int32_t)a * (int64_t)(uint64_t)b) >> 32);
    case OP_MULHU: return (uint32_t)(((uint64_t)a * b) >> 32);
    case OP_SH1ADD: return (a << 1) + b;
    case OP_SH2ADD: return (a << 2) + b;
    case OP_SH3ADD: return (a << 3) + b;
    }
    return 0;
}

static inline int qd_taken(int op, uint32_t a, uint32_t b) {
    switch (op) {
    case OP_BEQ: return a == b;
    case OP_BNE: return a != b;
    case OP_BLT: return (int32_t)a < (int32_t)b;
    case OP_BGE: return (int32_t)a >= (int32_t)b;
    case OP_BLTU: return a < b;
    case OP_BGEU: return a >= b;
    }
    return 0;
}

/* how many of the 8 nibble prefixes v >> 4i are nonzero (ref.nibbles_used) */
static inline int qd_nibbles(uint32_t v) { return v ? (32 - __builtin_clz(v) + 3) >> 2 : 0; }

/* k0, the first quotient digit a DIV-class record computes (ref.div_start) */
static inline int qd_div_start(int op, uint32_t a, uint32_t b) {
    int sb = (op == OP_DIV || op == OP_REM) && (b >> 31);
    int k = 7 - qd_nibbles(a) + qd_nibbles(sb ? ~b : b);
    if (b == 0) return 0;
    return k < 0 ? 0 : k > 8 ? 8 : k;
}

/* the quotient |A| / |B| (all ones for B = 0) and the result (ref.div_parts) */
static inline uint32_t qd_div_parts(int op, uint32_t a, uint32_t b, uint32_t *quot) {
    int sg = op == OP_DIV || op == OP_REM;
    int sa = sg && (a >> 31), sb = sg && (b >> 31);
    uint32_t ua = sa ? 0u - a : a, ub = sb ? 0u - b : b;
    uint32_t q = ub ? ua / ub : 0xFFFFFFFFu;
    uint32_t qs = (sa != sb && b) ? 0u - q : q;
    *quot = q;
    return (op == OP_DIV || op == OP_DIVU) ? qs : a - qs * b;
}

/* spec.value_record */
static inline int qd_emit_value(int32_t *o, uint32_t v) {
    int32_t s = (int32_t)v;
    if (s >= -(1 << 14) && s < (1 << 14)) {
        o[0] = QD_TOK_F0 + (int32_t)(v & 0x7FFF);
        return 1;
    }
    o[0] = QD_TOK_V + (int32_t)(v & 0x7FFF);
    if (s >= -(1 << 29) && s < (1 << 29)) {
        o[1] = QD_TOK_F1 + (int32_t)(v >> 15 & 0x7FFF);
        return 2;
    }
    o[1] = QD_TOK_V + (int32_t)(v >> 15 & 0x7FFF);
    o[2] = QD_TOK_F2 + (int32_t)(v >> 30);
    return 3;
}

/* the value record's size (1..3 tokens) */
static inline int qd_value_size(uint32_t v) {
    int32_t s = (int32_t)v;
    return (s >= -(1 << 14) && s < (1 << 14)) ? 1 : (s >= -(1 << 29) && s < (1 << 29)) ? 2 : 3;
}

static inline int qd_emit_div(int32_t *o, int op, uint32_t a, uint32_t b, uint32_t *res) {
    uint32_t q;
    int k0 = qd_div_start(op, a, b), n = 0;
    *res = qd_div_parts(op, a, b, &q);
    o[n++] = QD_TOK_QS + k0;
    for (int k = k0; k < 8; k++) o[n++] = QD_TOK_QD + (k << 4 | (int)(q >> 4 * (7 - k) & 15));
    return n + qd_emit_value(o + n, *res);
}

static inline int qd_emit_fault(int32_t *o, int cause, uint32_t pc) {
    o[0] = QD_TOK_FLT + cause;
    for (int i = 0; i < 8; i++) o[1 + i] = QD_TOK_HEX + (int32_t)(pc >> 4 * (7 - i) & 15);
    o[9] = QD_TOK_HALT;
    return 10;
}

static inline int qd_is_load(int op) { return op >= OP_LB && op <= OP_LHU; }
static inline int qd_access_size(int op) {
    switch (op) {
    case OP_LB: case OP_LBU: case OP_SB: return 1;
    case OP_LH: case OP_LHU: case OP_SH: return 2;
    }
    return 4;
}

/* A pre-decode cache over the ROM: one entry per word, re-decoded when the raw word changes. */
typedef struct {
    const uint8_t *rom;
    uint32_t base, words;
    int32_t input_csr;
    qd_inst *e;
} qd_icache;

int qd_icache_init(qd_icache *c, const uint8_t *rom, const qd_map *map);
void qd_icache_free(qd_icache *c);

static inline const qd_inst *qd_fetch(qd_icache *c, uint32_t wi) {
    qd_inst *e = &c->e[wi];
    uint32_t raw = qd_ld32(c->rom + 4 * (size_t)wi);
    if (__builtin_expect(e->raw != raw, 0)) qd_decode(raw, c->base + 4 * wi, c->input_csr, e);
    return e;
}

static inline int qd_in(uint32_t a, uint32_t base, uint32_t size) { return a - base < size; }

/* "F1(123)", "NT", "<bad 300000>" */
static inline const char *qd_tok_name(int32_t t, char buf[32]) {
    uint32_t p;
    int f = qd_tok_family(t, &p);
    if (f < 0) snprintf(buf, 32, "<bad %d>", (int)t);
    else if (f >= QD_FAM_NT) snprintf(buf, 32, "%s", qd_fam_names[f]);
    else snprintf(buf, 32, "%s(%u)", qd_fam_names[f], p);
    return buf;
}

#endif
