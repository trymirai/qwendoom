/* qd_trace: the C reference traced machine (qd_ref), the trace decoder or mirror (qd_dec) and
   the lockstep checker (qd_chk) of qwendoom. Built as build/libqdtrace.dylib; docs/trace.md has
   the API and the exact record rules.

   qd_ref is a token-exact port of the record rules of Qwenputer's chipformer/rv/ref.py
   (Machine.record, MIT (c) 2026 Divyansh Agrawal) with qwendoom's changes: a runtime address
   map, the digit-file ops (custom-0 / custom-1) are illegal instructions, and `csrr rd, time`
   reads the next word of an input queue (that record is host-owned). */
#ifndef QD_TRACE_H
#define QD_TRACE_H

#include <stddef.h>
#include <stdint.h>

#include "qd_mem.h"
#include "qd_tokens.h"

#define QD_MAX_RECORD 16 /* tokens in one record, at most (DIV: QS + 8 QD + V V F2 = 12) */

/* --- the address map ------------------------------------------------------------------------ */

typedef struct {
    uint32_t ram_base, ram_size;   /* loads and stores (words) */
    uint32_t drom_base, drom_size; /* loads only */
    uint32_t rom_base, rom_size;   /* fetch only; rom_size / 4 <= 65536 (J / R payloads) */
    int32_t input_csr;             /* the input port (QD_CSR_INPUT), or -1: none (reads 0) */
} qd_map;

void qd_map_qwendoom(qd_map *m); /* native/qd_mem.h's map, input port `time` */
void qd_map_upstream(qd_map *m); /* chipformer.rv's map (RAM 32 KB, DROM 128 KB at 0x20000,
                                    ROM 256 KB at 0x40000), no input port */

/* --- tokens --------------------------------------------------------------------------------- */

/* The value record of a 32-bit word (F0 | V F1 | V V F2 by its sign-extended size); returns the
   number of tokens (1..3) written to out. */
int qd_value_record(uint32_t v, int32_t out[3]);

/* The DIV-class record (QS, QD.., value record) of op (0 DIV, 1 DIVU, 2 REM, 3 REMU) on rs1 = a,
   rs2 = b with rd != x0; returns the number of tokens written to out. */
int qd_div_record(int op, uint32_t a, uint32_t b, int32_t out[QD_MAX_RECORD]);

/* --- the reference traced machine ----------------------------------------------------------- */

typedef struct qd_ref qd_ref;

enum { /* qd_ref_status */
    QD_REF_RUNNING = 0,
    QD_REF_HALTED = 1,     /* ecall exit (HALT); qd_ref_exit_code */
    QD_REF_FAULTED = 2,    /* a fault record ended the run; qd_ref_fault */
    QD_REF_NEED_INPUT = 3, /* the next record reads the input port and the queue is empty */
    QD_REF_ERROR = 4,      /* the PC left the ROM (no record can express it) */
};

enum { /* why qd_ref_run stopped */
    QD_RUN_LIMIT = 0, /* max_records reached */
    QD_RUN_FULL = 1,  /* the token buffer cannot hold another record */
    QD_RUN_YIELD = 2, /* BYTE(256) (with QD_STOP_YIELD) */
    QD_RUN_BYTE = 3,  /* any other BYTE (with QD_STOP_BYTE) */
    QD_RUN_HALT = 4,
    QD_RUN_FAULT = 5,
    QD_RUN_INPUT = 6, /* needs input: push words and run again */
    QD_RUN_ERROR = 7,
};
#define QD_STOP_YIELD 1
#define QD_STOP_BYTE 2

/* per-token flags (qd_ref_run's flags array) */
#define QD_TF_FIRST 1 /* the first token of a record */
#define QD_TF_HOST 2  /* host-owned: an input word's value record (not compared by checkers) */

/* A machine over mem (RAM is written in place; ROM is read through a pre-decode cache keyed by
   the raw word, so ROM changes need no invalidation). The context starts at pc = rom_base,
   registers 0. */
qd_ref *qd_ref_new(const qd_mem *mem, const qd_map *map);
void qd_ref_free(qd_ref *r);

/* The context as a snapshot holds it: anchor (PC word index of the latest J / R), n (commits
   since), x0..x31 (x0 is forced to 0; regs may be NULL: all 0). Clears the status (running),
   the counters and the input queue. */
void qd_ref_set_context(qd_ref *r, uint32_t anchor, uint32_t n, const uint32_t *regs);
void qd_ref_get_context(const qd_ref *r, uint32_t *anchor, uint32_t *n, uint32_t *regs);
uint32_t qd_ref_pc(const qd_ref *r);
uint32_t qd_ref_inst(const qd_ref *r); /* the raw word at the PC (0 before an R refresh) */
void qd_ref_set_reg(qd_ref *r, int i, uint32_t v);

int qd_ref_status(const qd_ref *r);
uint32_t qd_ref_exit_code(const qd_ref *r);
void qd_ref_fault(const qd_ref *r, int *cause, uint32_t *pc);
/* counters since the last set_context: instructions (records but R), records, tokens */
void qd_ref_counts(const qd_ref *r, uint64_t *steps, uint64_t *records, uint64_t *tokens);
/* the RAM address of the latest store record (UINT32_MAX if the latest record stored nothing) */
uint32_t qd_ref_last_store(const qd_ref *r);

void qd_ref_push_input(qd_ref *r, const uint32_t *words, size_t n);
size_t qd_ref_input_pending(const qd_ref *r);
void qd_ref_clear_input(qd_ref *r);
/* 1 if the next record reads the input port (`csrr rd, time` with rd != x0, not an R refresh) */
int qd_ref_wants_input(qd_ref *r);

/* Emit and commit the next record into out; returns its length (1..QD_MAX_RECORD), or 0 if no
   record can run now (see qd_ref_status). *flags (if not NULL) gets QD_TF_HOST for an input
   record. */
int qd_ref_record(qd_ref *r, int32_t *out, int *flags);

/* Run records into toks[0..cap) (toks NULL: count only, cap ignored) until max_records (< 0: no
   limit), a full buffer, HALT, a fault, a needed input word, or a stop_mask event. flags (if not
   NULL) gets QD_TF_* per token. Returns the number of tokens written; *reason the QD_RUN_*. */
long qd_ref_run(qd_ref *r, int32_t *toks, uint8_t *flags, long cap, long max_records,
                int stop_mask, int *reason);

/* --- the trace decoder (the mirror) --------------------------------------------------------- */

typedef struct qd_dec qd_dec;

enum { /* qd_dec_feed results: >= 0 the token was accepted */
    QD_DEC_MORE = 0,   /* the record continues */
    QD_DEC_COMMIT = 1, /* a record committed (an instruction or an R refresh) */
    QD_DEC_BYTE = 2,   /* a BYTE record committed (putchar): qd_dec_last_byte */
    QD_DEC_YIELD = 3,  /* BYTE(256) committed: the tick is done */
    QD_DEC_HALT = 4,   /* HALT (ecall exit) committed; the stream is over */
    QD_DEC_FAULT = 5,  /* a fault record committed; the stream is over */
};
/* errors (sticky until the next set_context) */
#define QD_DEC_E_TOKEN -1   /* not a chip token */
#define QD_DEC_E_SHAPE -2   /* the record grammar for the instruction at pc forbids this family */
#define QD_DEC_E_REFRESH -3 /* n == 63 needs R(anchor + 63) */
#define QD_DEC_E_VALUE -4   /* a malformed or non-canonical value record */
#define QD_DEC_E_DIV -5     /* QS / QD indices (strict: k0 != div_start) */
#define QD_DEC_E_JUMP -6    /* J target != the static one; (strict) branch direction, JALR target */
#define QD_DEC_E_STORE -7   /* a store outside RAM / misaligned; (strict) bank bit, merged word */
#define QD_DEC_E_FAULT -8   /* a fault this instruction cannot raise (strict: does not raise now),
                               a missing fault (strict), or HEX digits != pc */
#define QD_DEC_E_ECALL -9   /* (strict) BYTE / HALT disagree with a7 / a0 */
#define QD_DEC_E_PC -10     /* the PC left the ROM */
#define QD_DEC_E_HALTED -11 /* a token after the stream ended */

/* feed_many stop mask: 1 << QD_DEC_* events, plus */
#define QD_DEC_STOP_INPUT (1 << 8) /* stop when the next record reads the input port */

/* A decoder over mem: stores are applied to mem's RAM. It always checks the record grammar for
   the instruction at the PC (families, value sizes, R payloads, QD indices, static J targets,
   fault PCs). strict: also every token it can derive from its own registers and memory without
   computing an rd value: branch directions, JALR targets, whether a load / store faults, store
   banks and SB / SH merged words, DIV k0, ecall kinds and BYTE payloads. */
qd_dec *qd_dec_new(const qd_mem *mem, const qd_map *map, int strict);
void qd_dec_free(qd_dec *d);
/* Start from a snapshot context (no earlier tokens needed); clears the error and counters. The
   first token after it may be GO (the run's start; accepted as QD_DEC_MORE, nowhere else). Call
   it again after the ROM changes (a resync). */
void qd_dec_set_context(qd_dec *d, uint32_t anchor, uint32_t n, const uint32_t *regs);
void qd_dec_get_context(const qd_dec *d, uint32_t *anchor, uint32_t *n, uint32_t *regs);
uint32_t qd_dec_pc(const qd_dec *d);

int qd_dec_feed(qd_dec *d, int32_t tok);
/* Feed toks[0..n) until one of them returns an error or an event in stop_mask (or, with
   QD_DEC_STOP_INPUT, a commit after which the next record is an input record); *consumed gets
   the tokens accepted. Returns the last result. */
int qd_dec_feed_many(qd_dec *d, const int32_t *toks, long n, int stop_mask, long *consumed);

/* 1 at a record boundary when the next record reads the input port: the host injects the input
   word's value record (qd_value_record) instead of the model's prediction. */
int qd_dec_wants_input(const qd_dec *d);
int qd_dec_at_boundary(const qd_dec *d); /* no record is in progress */

int qd_dec_error(const qd_dec *d);                 /* 0 or the sticky QD_DEC_E_* */
const char *qd_dec_error_text(const qd_dec *d);    /* "" or a message with the context */
int qd_dec_halted(const qd_dec *d);                /* 0, QD_DEC_HALT or QD_DEC_FAULT */
uint32_t qd_dec_exit_code(const qd_dec *d);        /* a0 at HALT */
void qd_dec_fault(const qd_dec *d, int *cause, uint32_t *pc);
uint32_t qd_dec_last_byte(const qd_dec *d);
uint32_t qd_dec_last_store(const qd_dec *d);       /* as qd_ref_last_store */
/* counters: tokens accepted, records committed, instructions (records but R), stores applied */
void qd_dec_counts(const qd_dec *d, uint64_t *tokens, uint64_t *records, uint64_t *instructions,
                   uint64_t *stores);

/* --- the lockstep checker -------------------------------------------------------------------- */

typedef struct qd_chk qd_chk;

enum { QD_CHK_OK = 0, QD_CHK_END = 1, QD_CHK_DIVERGED = -1, QD_CHK_ERROR = -2 };

typedef struct {
    int64_t index;        /* stream position of the first divergent token */
    int64_t record;       /* its record (0-based) */
    int64_t record_start; /* stream position of that record's first token */
    int32_t got, want;    /* want = -1: the reference ended (HALT / fault) before this token */
    uint32_t pc, anchor, n, inst; /* the record's PC, context and raw instruction word */
    int32_t ref_record[QD_MAX_RECORD]; /* the reference's tokens for the record */
    int32_t ref_len;
} qd_divergence;

/* A checker that runs ref (borrowed, from its current context) against a token stream. Host-owned
   positions (input records) are not compared: the stream's value record there is parsed and, if
   ref's input queue is empty, its word is pushed as the input. A GO as the stream's first token
   is skipped. ref writes its own RAM: give it a RAM buffer of its own (DROM / ROM may be shared)
   unless the stream's stores land in the same memory with the same values anyway. */
qd_chk *qd_chk_new(qd_ref *ref);
void qd_chk_free(qd_chk *c);
void qd_chk_reset(qd_chk *c); /* restart the stream count (ref keeps its state) */
/* One stream token: QD_CHK_OK, QD_CHK_END (the reference's HALT / fault record just matched),
   QD_CHK_DIVERGED or QD_CHK_ERROR (the reference cannot run: qd_ref_status); sticky. */
int qd_chk_feed(qd_chk *c, int32_t tok);
/* Feed toks[0..n); returns the number of tokens consumed before a non-OK result (n if none);
   *state gets the last result. */
long qd_chk_stream(qd_chk *c, const int32_t *toks, long n, int *state);
int qd_chk_state(const qd_chk *c);
void qd_chk_divergence(const qd_chk *c, qd_divergence *out);
void qd_chk_counts(const qd_chk *c, int64_t *tokens, int64_t *records, int64_t *host_tokens);

/* --- test support ---------------------------------------------------------------------------- */

/* Decoder round trip: run ref record by record, feed each record to dec, and after every record
   compare the decoder's context (anchor, n, x0..x31) with ref's, and after every store the word at
   both machines' store addresses (ref and dec have separate memories). Up to max_records; at the
   end memcmp the RAM. Returns the records checked (>= 0) or -1 with msg filled. */
long qd_roundtrip(qd_ref *ref, const qd_mem *ref_mem, qd_dec *dec, const qd_mem *dec_mem,
                  const qd_map *map, long max_records, char *msg, int msg_len);

#endif
