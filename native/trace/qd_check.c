/* qd_chk: the lockstep checker. It runs the reference machine record by record against a token
   stream (the model's) and reports the first token that differs, with its record's context.
   Input records are host-owned: their tokens are the host's injection, not a prediction, so they
   are parsed as a value record but not compared (and feed the reference's input when its queue is
   empty, so a checker needs nothing but the stream). Also the decoder round trip (test support). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "qd_isa.h"

struct qd_chk {
    qd_ref *ref;
    int32_t rec[QD_MAX_RECORD];
    int len, pos;
    int in_host, hn; /* in a host-owned record: the stream's value chunks so far */
    uint32_t hv[3];
    int64_t index, records, host_tokens, rec_start;
    uint32_t rec_pc, rec_anchor, rec_n, rec_inst;
    int state;
    qd_divergence div;
};

qd_chk *qd_chk_new(qd_ref *ref) {
    qd_chk *c = calloc(1, sizeof *c);
    if (!c) return NULL;
    c->ref = ref;
    qd_chk_reset(c);
    return c;
}

void qd_chk_free(qd_chk *c) { free(c); }

void qd_chk_reset(qd_chk *c) {
    qd_ref *ref = c->ref;
    memset(c, 0, sizeof *c);
    c->ref = ref;
    c->state = QD_CHK_OK;
}

int qd_chk_state(const qd_chk *c) { return c->state; }
void qd_chk_divergence(const qd_chk *c, qd_divergence *out) { *out = c->div; }

void qd_chk_counts(const qd_chk *c, int64_t *tokens, int64_t *records, int64_t *host_tokens) {
    if (tokens) *tokens = c->index;
    if (records) *records = c->records;
    if (host_tokens) *host_tokens = c->host_tokens;
}

static int diverge(qd_chk *c, int32_t got, int32_t want) {
    qd_divergence *d = &c->div;
    d->index = c->index;
    d->record = c->records;
    d->record_start = c->rec_start;
    d->got = got;
    d->want = want;
    d->pc = c->rec_pc;
    d->anchor = c->rec_anchor;
    d->n = c->rec_n;
    d->inst = c->rec_inst;
    d->ref_len = c->in_host ? 0 : c->len;
    memcpy(d->ref_record, c->rec, sizeof c->rec);
    return c->state = QD_CHK_DIVERGED;
}

int qd_chk_feed(qd_chk *c, int32_t tok) {
    qd_ref *r = c->ref;
    if (c->state == QD_CHK_END) return diverge(c, tok, -1); /* a token after the end */
    if (c->state != QD_CHK_OK) return c->state;
    if (tok == QD_TOK_GO && c->index == 0) { /* the start of a run (after a snapshot) */
        c->index++;
        c->rec_start = 1;
        return QD_CHK_OK;
    }
    if (c->pos == c->len && !c->in_host) { /* this token starts a record */
        c->rec_start = c->index;
        qd_ref_get_context(r, &c->rec_anchor, &c->rec_n, NULL);
        c->rec_pc = qd_ref_pc(r);
        c->rec_inst = qd_ref_inst(r);
        int st = qd_ref_status(r);
        if (st == QD_REF_HALTED || st == QD_REF_FAULTED) {
            c->len = c->pos = 0;
            return diverge(c, tok, -1);
        }
        if (qd_ref_wants_input(r)) {
            c->in_host = 1;
            c->hn = 0;
            c->len = c->pos = 0;
        } else {
            c->len = qd_ref_record(r, c->rec, NULL);
            c->pos = 0;
            if (!c->len) return c->state = QD_CHK_ERROR;
        }
    }
    if (c->in_host) { /* the host's value record: parse, feed the reference, do not compare */
        uint32_t p;
        int f = qd_tok_family(tok, &p), fin = 0;
        uint32_t w = 0;
        if (f == QD_FAM_V && c->hn < 2) c->hv[c->hn++] = p;
        else if (f == QD_FAM_F0 && c->hn == 0) fin = 1, w = p | (p >> 14 ? 0xFFFF8000u : 0);
        else if (f == QD_FAM_F1 && c->hn == 1)
            fin = 1, w = c->hv[0] | p << 15 | (p >> 14 ? 0xC0000000u : 0);
        else if (f == QD_FAM_F2 && c->hn == 2) fin = 1, w = c->hv[0] | c->hv[1] << 15 | p << 30;
        else return diverge(c, tok, -2); /* -2: a value-record token was due */
        c->index++;
        c->host_tokens++;
        if (fin) {
            if (!qd_ref_input_pending(r)) qd_ref_push_input(r, &w, 1);
            c->len = qd_ref_record(r, c->rec, NULL);
            if (!c->len) return c->state = QD_CHK_ERROR;
            c->pos = c->len;
            c->in_host = 0;
            c->records++;
        }
        return QD_CHK_OK;
    }
    if (tok != c->rec[c->pos]) return diverge(c, tok, c->rec[c->pos]);
    c->pos++;
    c->index++;
    if (c->pos == c->len) {
        c->records++;
        int st = qd_ref_status(r);
        if (st == QD_REF_HALTED || st == QD_REF_FAULTED) return c->state = QD_CHK_END;
    }
    return QD_CHK_OK;
}

long qd_chk_stream(qd_chk *c, const int32_t *toks, long n, int *state) {
    long i = 0;
    int s = c->state;
    for (; i < n; i++) {
        s = qd_chk_feed(c, toks[i]);
        if (s != QD_CHK_OK) {
            if (s == QD_CHK_END) i++;
            break;
        }
    }
    if (state) *state = s;
    return i;
}

/* --- the decoder round trip ------------------------------------------------------------- */

long qd_roundtrip(qd_ref *ref, const qd_mem *ref_mem, qd_dec *dec, const qd_mem *dec_mem,
                  const qd_map *map, long max_records, char *msg, int msg_len) {
    int32_t rec[QD_MAX_RECORD];
    uint32_t ra, rn, rx[32], da, dn, dx[32];
    char tb[32];
    long i;
    for (i = 0; max_records < 0 || i < max_records; i++) {
        int len = qd_ref_record(ref, rec, NULL);
        if (!len) break;
        for (int j = 0; j < len; j++) {
            int res = qd_dec_feed(dec, rec[j]);
            if (res < 0) {
                snprintf(msg, (size_t)msg_len, "record %ld token %d %s: decoder error %d: %s", i,
                         j, qd_tok_name(rec[j], tb), res, qd_dec_error_text(dec));
                return -1;
            }
            if ((res == QD_DEC_MORE) != (j < len - 1)) {
                snprintf(msg, (size_t)msg_len, "record %ld token %d %s: decoder result %d", i, j,
                         qd_tok_name(rec[j], tb), res);
                return -1;
            }
        }
        qd_ref_get_context(ref, &ra, &rn, rx);
        qd_dec_get_context(dec, &da, &dn, dx);
        if (ra != da || rn != dn) {
            snprintf(msg, (size_t)msg_len, "record %ld: anchor/n %u/%u, decoder %u/%u", i, ra, rn,
                     da, dn);
            return -1;
        }
        for (int k = 0; k < 32; k++)
            if (rx[k] != dx[k]) {
                snprintf(msg, (size_t)msg_len, "record %ld: x%d = 0x%08x, decoder 0x%08x", i, k,
                         rx[k], dx[k]);
                return -1;
            }
        uint32_t rs = qd_ref_last_store(ref), ds = qd_dec_last_store(dec);
        if (rs != ds) {
            snprintf(msg, (size_t)msg_len, "record %ld: store at 0x%08x, decoder 0x%08x", i, rs,
                     ds);
            return -1;
        }
        if (rs != UINT32_MAX) {
            uint32_t off = rs - map->ram_base;
            uint32_t rw = qd_ld32(ref_mem->ram + off), dw = qd_ld32(dec_mem->ram + off);
            if (rw != dw) {
                snprintf(msg, (size_t)msg_len, "record %ld: RAM[0x%08x] = 0x%08x, decoder 0x%08x",
                         i, rs, rw, dw);
                return -1;
            }
        }
    }
    int st = qd_ref_status(ref), h = qd_dec_halted(dec);
    if ((st == QD_REF_HALTED) != (h == QD_DEC_HALT)
        || (st == QD_REF_FAULTED) != (h == QD_DEC_FAULT)) {
        snprintf(msg, (size_t)msg_len, "end: ref status %d, decoder halted %d", st, h);
        return -1;
    }
    if (st == QD_REF_HALTED && qd_ref_exit_code(ref) != qd_dec_exit_code(dec)) {
        snprintf(msg, (size_t)msg_len, "end: exit code %u, decoder %u", qd_ref_exit_code(ref),
                 qd_dec_exit_code(dec));
        return -1;
    }
    if (st == QD_REF_FAULTED) {
        int rc, dc;
        uint32_t rp, dp;
        qd_ref_fault(ref, &rc, &rp);
        qd_dec_fault(dec, &dc, &dp);
        if (rc != dc || rp != dp) {
            snprintf(msg, (size_t)msg_len, "end: fault %d at 0x%08x, decoder %d at 0x%08x", rc, rp,
                     dc, dp);
            return -1;
        }
    }
    if (memcmp(ref_mem->ram, dec_mem->ram, map->ram_size)) {
        for (uint32_t off = 0; off < map->ram_size; off += 4)
            if (qd_ld32(ref_mem->ram + off) != qd_ld32(dec_mem->ram + off)) {
                snprintf(msg, (size_t)msg_len, "end: RAM[0x%08x] = 0x%08x, decoder 0x%08x",
                         map->ram_base + off, qd_ld32(ref_mem->ram + off),
                         qd_ld32(dec_mem->ram + off));
                return -1;
            }
    }
    if (msg_len) msg[0] = 0;
    return i;
}
