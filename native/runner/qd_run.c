/* qd_run: one sim tick of the model / check / ref modes in C. See qd_run.h and docs/run.md. */
#include "qd_run.h"

#include <pthread.h>
#include <pthread/qos.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

struct qd_run {
    qd_engine *e;   /* model / check mode */
    qd_ref *ref;    /* ref mode: the token source */
    qd_dec *dec;    /* the mirror */
    qd_chk *chk;    /* check mode */
    int32_t pred;   /* the model's pending prediction, -1 if none */
    int forced;
    uint64_t max_tokens;
    _Atomic uint64_t live; /* tokens fed so far (written by the tick's thread only) */
    _Atomic int cancel, paused;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    int32_t *log;   /* qd_run_set_log: every token fed, and its host flag */
    uint8_t *log_host;
    uint64_t log_cap, log_len;
    qd_tick_stats last;
    char err[2048];
};

static uint64_t now_ns(void) { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }

static qd_run *alloc_run(qd_dec *dec) {
    qd_run *r = calloc(1, sizeof *r);
    if (!r) return NULL;
    r->dec = dec;
    r->pred = -1;
    r->forced = 1;
    r->max_tokens = 1ull << 26;
    pthread_mutex_init(&r->mu, NULL);
    pthread_cond_init(&r->cv, NULL);
    return r;
}

qd_run *qd_run_new(qd_engine *e, qd_dec *dec, qd_chk *chk) {
    if (!e || !dec) return NULL;
    qd_run *r = alloc_run(dec);
    if (r) r->e = e, r->chk = chk;
    return r;
}

qd_run *qd_run_new_ref(qd_ref *ref, qd_dec *dec) {
    if (!ref || !dec) return NULL;
    qd_run *r = alloc_run(dec);
    if (r) r->ref = ref;
    return r;
}

void qd_run_free(qd_run *r) {
    if (!r) return;
    pthread_cond_destroy(&r->cv);
    pthread_mutex_destroy(&r->mu);
    free(r);
}

void qd_run_set_pred(qd_run *r, int32_t pred) { r->pred = pred; }

int32_t qd_run_pred(const qd_run *r) { return r->pred; }
uint64_t qd_run_live_tokens(const qd_run *r) {
    return atomic_load_explicit(&r->live, memory_order_relaxed);
}
void qd_run_set_forced(qd_run *r, int on) { r->forced = !!on; }
void qd_run_set_max_tokens(qd_run *r, uint64_t n) { r->max_tokens = n; }
const char *qd_run_error(const qd_run *r) { return r->err; }
void qd_run_last(const qd_run *r, qd_tick_stats *st) { *st = r->last; }

void qd_run_set_log(qd_run *r, int32_t *toks, uint8_t *host, uint64_t cap) {
    r->log = toks;
    r->log_host = host;
    r->log_cap = toks ? cap : 0;
    r->log_len = 0;
}

uint64_t qd_run_log_len(const qd_run *r) { return r->log_len; }

void qd_run_cancel(qd_run *r) {
    pthread_mutex_lock(&r->mu);
    atomic_store(&r->cancel, 1);
    pthread_cond_broadcast(&r->cv);
    pthread_mutex_unlock(&r->mu);
}

void qd_run_pause(qd_run *r, int on) {
    pthread_mutex_lock(&r->mu);
    atomic_store(&r->paused, !!on);
    pthread_cond_broadcast(&r->cv);
    pthread_mutex_unlock(&r->mu);
}

int qd_run_set_qos(void) { return pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0); }

const char *qd_run_token_name(int32_t t, char *buf) {
    uint32_t p = 0;
    const int f = qd_tok_family(t, &p);
    if (t == QD_TOK_PAD) snprintf(buf, 24, "PAD");
    else if (f < 0) snprintf(buf, 24, "<bad %d>", t);
    else if (f >= QD_FAM_NT) snprintf(buf, 24, "%s", qd_fam_names[f]);
    else snprintf(buf, 24, "%s(%u)", qd_fam_names[f], p);
    return buf;
}

/* --- the tick ---------------------------------------------------------------------------------- */

/* Hold while paused (or until a cancel); returns the seconds held. */
static double hold(qd_run *r) {
    const uint64_t t0 = now_ns();
    pthread_mutex_lock(&r->mu);
    while (atomic_load(&r->paused) && !atomic_load(&r->cancel)) pthread_cond_wait(&r->cv, &r->mu);
    pthread_mutex_unlock(&r->mu);
    return (double)(now_ns() - t0) * 1e-9;
}

static void divergence_text(qd_run *r, uint64_t tick_index) {
    qd_divergence d;
    qd_chk_divergence(r->chk, &d);
    char got[24], want[40], tb[24];
    qd_run_token_name(d.got, got);
    if (d.want == -1) snprintf(want, sizeof want, "the end (HALT / fault)");
    else if (d.want == -2) snprintf(want, sizeof want, "a value record (host-owned)");
    else qd_run_token_name(d.want, want);
    int o = snprintf(r->err, sizeof r->err,
                     "check: tick token %llu, stream token %lld (record %lld, +%lld) at pc 0x%08x "
                     "(word 0x%08x, anchor %u n %u): got %s, want %s; reference record:",
                     (unsigned long long)tick_index, (long long)d.index, (long long)d.record,
                     (long long)(d.index - d.record_start), d.pc, d.inst, d.anchor, d.n, got, want);
    for (int i = 0; i < d.ref_len && o > 0 && o < (int)sizeof r->err; i++)
        o += snprintf(r->err + o, sizeof r->err - o, " %s", qd_run_token_name(d.ref_record[i], tb));
}

/* A token was fed: the live counter and the log. */
static inline void fed(qd_run *r, uint64_t *live, int32_t t, int host) {
    atomic_store_explicit(&r->live, ++*live, memory_order_relaxed);
    if (r->log_len < r->log_cap) {
        r->log[r->log_len] = t;
        if (r->log_host) r->log_host[r->log_len] = (uint8_t)host;
        r->log_len++;
    }
}

/* One token to the decoder, then the checker; *ev gets the decoder's event. 0 or an error. */
static int put(qd_run *r, int32_t t, uint64_t tick_index, int *ev) {
    const int e = qd_dec_feed(r->dec, t);
    if (e < 0) {
        snprintf(r->err, sizeof r->err, "decode: tick token %llu: %s",
                 (unsigned long long)tick_index, qd_dec_error_text(r->dec));
        return QD_TICK_E_DECODE;
    }
    if (r->chk) {
        const int s = qd_chk_feed(r->chk, t);
        if (s == QD_CHK_DIVERGED) {
            divergence_text(r, tick_index);
            return QD_TICK_E_DIVERGED;
        }
        if (s == QD_CHK_ERROR) {
            snprintf(r->err, sizeof r->err, "check: tick token %llu: the reference cannot run",
                     (unsigned long long)tick_index);
            return QD_TICK_E_REF;
        }
    }
    *ev = e;
    return 0;
}

static int model_tick(qd_run *r, const uint32_t *words, int n, qd_tick_stats *st) {
    uint64_t live = atomic_load_explicit(&r->live, memory_order_relaxed);
    int wi = 0, ev = QD_DEC_MORE;
    if (r->pred < 0) {
        snprintf(r->err, sizeof r->err, "no prediction pending (resync first)");
        return QD_TICK_E_STATE;
    }
    for (;;) {
        if (atomic_load_explicit(&r->cancel, memory_order_relaxed)) return QD_TICK_CANCELLED;
        if (atomic_load_explicit(&r->paused, memory_order_relaxed)) {
            st->paused += hold(r);
            continue;
        }
        if (st->tokens >= r->max_tokens) {
            snprintf(r->err, sizeof r->err, "no YIELD / HALT within %llu tokens (pc 0x%08x)",
                     (unsigned long long)r->max_tokens, qd_dec_pc(r->dec));
            return QD_TICK_E_LIMIT;
        }
        if (qd_dec_wants_input(r->dec)) {
            /* the input port: the next word's value record replaces the model's predictions */
            if (wi >= n) {
                snprintf(r->err, sizeof r->err,
                         "the sim reads input at pc 0x%08x with no word left (%d read)",
                         qd_dec_pc(r->dec), n);
                return QD_TICK_E_INPUT;
            }
            int32_t rec[3];
            const int k = qd_value_record(words[wi++], rec);
            for (int j = 0; j < k; j++) {
                const int err = put(r, rec[j], st->tokens, &ev);
                if (err) return err;
                st->tokens++, st->injected++;
                fed(r, &live, rec[j], 1);
                if (j + 1 < k && r->forced) {
                    qd_engine_step_forced(r->e, rec[j]); /* its prediction is not used */
                    st->forced++;
                } else {
                    r->pred = qd_engine_step(r->e, rec[j]);
                }
            }
            continue;
        }
        const int32_t t = r->pred;
        const int err = put(r, t, st->tokens, &ev);
        if (err) return err;
        st->tokens++;
        fed(r, &live, t, 0);
        if (ev == QD_DEC_HALT || ev == QD_DEC_FAULT) {
            r->pred = -1;
            break;
        }
        r->pred = qd_engine_step(r->e, t);
        if (ev == QD_DEC_YIELD) break;
    }
    st->event = ev;
    st->words = (uint64_t)wi;
    st->unread = (uint64_t)(n - wi);
    return ev == QD_DEC_YIELD ? QD_TICK_YIELD : ev == QD_DEC_HALT ? QD_TICK_HALT : QD_TICK_FAULT;
}

static int ref_tick(qd_run *r, const uint32_t *words, int n, qd_tick_stats *st) {
    uint64_t live = atomic_load_explicit(&r->live, memory_order_relaxed);
    int ev = QD_DEC_MORE;
    qd_ref_clear_input(r->ref);
    qd_ref_push_input(r->ref, words, (size_t)n);
    for (;;) {
        if (atomic_load_explicit(&r->cancel, memory_order_relaxed)) return QD_TICK_CANCELLED;
        if (atomic_load_explicit(&r->paused, memory_order_relaxed)) {
            st->paused += hold(r);
            continue;
        }
        if (st->tokens >= r->max_tokens) {
            snprintf(r->err, sizeof r->err, "no YIELD / HALT within %llu tokens (pc 0x%08x)",
                     (unsigned long long)r->max_tokens, qd_dec_pc(r->dec));
            return QD_TICK_E_LIMIT;
        }
        const int wants = qd_dec_wants_input(r->dec);
        int32_t rec[QD_MAX_RECORD];
        int flags = 0;
        const int k = qd_ref_record(r->ref, rec, &flags);
        if (k == 0) {
            const int s = qd_ref_status(r->ref);
            snprintf(r->err, sizeof r->err, "the reference stopped at pc 0x%08x: %s",
                     qd_ref_pc(r->ref),
                     s == QD_REF_NEED_INPUT ? "it reads input with no word left" : "error");
            return s == QD_REF_NEED_INPUT ? QD_TICK_E_INPUT : QD_TICK_E_REF;
        }
        const int host = (flags & QD_TF_HOST) != 0;
        if (host != wants) {
            snprintf(r->err, sizeof r->err,
                     "decode: tick token %llu at pc 0x%08x: the reference %s an input record, "
                     "the decoder %s one", (unsigned long long)st->tokens, qd_dec_pc(r->dec),
                     host ? "emits" : "does not emit", wants ? "wants" : "does not want");
            return QD_TICK_E_DECODE;
        }
        for (int j = 0; j < k; j++) {
            const int err = put(r, rec[j], st->tokens, &ev);
            if (err) return err;
            st->tokens++;
            st->injected += (uint64_t)host;
            fed(r, &live, rec[j], host);
        }
        if (ev == QD_DEC_YIELD || ev == QD_DEC_HALT || ev == QD_DEC_FAULT) break;
    }
    st->event = ev;
    st->unread = qd_ref_input_pending(r->ref);
    st->words = (uint64_t)n - st->unread;
    qd_ref_clear_input(r->ref);
    return ev == QD_DEC_YIELD ? QD_TICK_YIELD : ev == QD_DEC_HALT ? QD_TICK_HALT : QD_TICK_FAULT;
}

int qd_run_tick(qd_run *r, const uint32_t *words, int n, qd_tick_stats *out) {
    qd_tick_stats st;
    memset(&st, 0, sizeof st);
    st.event = -1;
    r->err[0] = 0;
    uint64_t rec0, ins0, rec1, ins1;
    qd_dec_counts(r->dec, NULL, &rec0, &ins0, NULL);
    const uint64_t t0 = now_ns();
    const int res = r->ref ? ref_tick(r, words, n, &st) : model_tick(r, words, n, &st);
    st.seconds = (double)(now_ns() - t0) * 1e-9 - st.paused;
    qd_dec_counts(r->dec, NULL, &rec1, &ins1, NULL);
    st.records = rec1 - rec0;
    st.instructions = ins1 - ins0;
    st.result = res;
    r->last = st;
    if (out) *out = st;
    return res;
}
