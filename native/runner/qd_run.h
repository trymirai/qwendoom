/* qd_run: one sim tick of qwendoom's model, check and ref modes, entirely in C (the hot loop of
   qwendoom/runner.py). Built as build/libqdrun.dylib, linked against build/libqdengine.dylib and
   build/libqdtrace.dylib (their public headers; @rpath = @loader_path). docs/run.md has the
   runner around it.

   A tick has trace.drive_tick's semantics: at every record boundary where the decoder wants
   input, the next input word's value record is injected token by token (each injected token is
   fed to the engine; the prediction after the record's last token is the next token, the
   earlier ones are fed as forced steps, which leave the same state); everywhere else the
   model's prediction is the token. Every token goes to the decoder (stores land in its memory:
   the host's RAM), to the checker in check mode, then to the engine. The tick ends after YIELD
   (stepped: its prediction is the next tick's first token), HALT or a fault record. In ref mode
   the reference machine produces the tokens instead of the engine (input words pushed to it);
   they go to the decoder the same way.

   Threads: qd_run_tick runs on the caller's thread (the game thread; qd_run_set_qos makes it
   user-interactive). qd_run_live_tokens, qd_run_cancel and qd_run_pause may be called from any
   other thread while a tick runs. */
#ifndef QD_RUN_H
#define QD_RUN_H

#include <stdint.h>

#include "qd_engine.h"
#include "qd_trace.h"

#ifdef __cplusplus
extern "C" {
#endif

#define QD_RUN_ABI 1

typedef struct qd_run qd_run;

/* qd_run_tick results: >= 0 the tick ended; < 0 an error (qd_run_error has the text) */
enum {
    QD_TICK_YIELD = 0,      /* BYTE(256): the tick is done, the next prediction is pending */
    QD_TICK_HALT = 1,       /* the sim exited (I_Error): no prediction pending */
    QD_TICK_FAULT = 2,      /* a fault record: no prediction pending */
    QD_TICK_CANCELLED = 3,  /* qd_run_cancel: stopped between two tokens */
};
#define QD_TICK_E_DECODE -1   /* the decoder refused a token (qd_dec_error / its text) */
#define QD_TICK_E_INPUT -2    /* the sim reads the input port with no word left */
#define QD_TICK_E_DIVERGED -3 /* check mode: the checker's divergence (qd_chk_divergence) */
#define QD_TICK_E_REF -4      /* the reference machine cannot run (its status) */
#define QD_TICK_E_LIMIT -5    /* no YIELD / HALT / fault within the token limit */
#define QD_TICK_E_STATE -6    /* no prediction pending (before qd_run_set_pred, after an end) */

typedef struct {
    uint64_t tokens;       /* tokens fed this tick (the injected ones included) */
    uint64_t injected;     /* host-injected tokens (input records) */
    uint64_t forced;       /* injected tokens fed as forced steps (no head) */
    uint64_t instructions; /* sim instructions (the decoder's records but R refreshes) */
    uint64_t records;      /* records (R refreshes included) */
    uint64_t words;        /* input words read */
    uint64_t unread;       /* input words left unread at the end */
    double seconds;        /* wall time of the tick, pauses excluded */
    double paused;         /* time spent paused (qd_run_pause) */
    int32_t event;         /* the decoder event that ended it (QD_DEC_YIELD / HALT / FAULT), or -1 */
    int32_t result;        /* what qd_run_tick returned */
} qd_tick_stats;

/* A runner over an engine (model mode), a decoder (the mirror; its memory is the host's) and a
   checker (check mode) or NULL. All borrowed. */
qd_run *qd_run_new(qd_engine *e, qd_dec *dec, qd_chk *chk);
/* A runner whose tokens come from the reference machine ref (ref mode): each tick's words are
   pushed to ref, its records go to dec (which must have its own memory or the same stores). */
qd_run *qd_run_new_ref(qd_ref *ref, qd_dec *dec);
void qd_run_free(qd_run *r);

/* After a resync: the prediction pending (what qd_engine_step(GO) returned; ignored in ref
   mode). The caller feeds GO to the decoder and the checker first (qd_dec_feed / qd_chk_feed),
   so the tick's first token is stream token 1. */
void qd_run_set_pred(qd_run *r, int32_t pred);
int32_t qd_run_pred(const qd_run *r); /* -1 if none */

/* One sim tick on the input words[0..n) (the host's pokes: (addr, value) pairs, then 0). */
int qd_run_tick(qd_run *r, const uint32_t *words, int n, qd_tick_stats *st);

/* Tokens fed since the runner was created (every mode, all ticks, the running one included).
   Any thread, any time. */
uint64_t qd_run_live_tokens(const qd_run *r);
/* Any thread: the running (or next) tick stops between two tokens with QD_TICK_CANCELLED, and
   so does every later one (the sim's state is then mid-tick: the game is over). */
void qd_run_cancel(qd_run *r);
/* Any thread: on = 1 holds the hot loop between two tokens until on = 0 (or a cancel). */
void qd_run_pause(qd_run *r, int on);

/* Options: forced = 1 (default) feeds the injected tokens before a record's last one as forced
   steps; max_tokens (default 1 << 26) bounds one tick. */
void qd_run_set_forced(qd_run *r, int on);
void qd_run_set_max_tokens(qd_run *r, uint64_t n);

/* Log every token fed from now on (and 1 / 0 for host-injected or not into host, if not NULL)
   into toks[0..cap); tokens past cap are not logged. NULL: stop logging. */
void qd_run_set_log(qd_run *r, int32_t *toks, uint8_t *host, uint64_t cap);
uint64_t qd_run_log_len(const qd_run *r); /* tokens logged */

/* The last error ("" if none): the decoder's or checker's text with the tick's token index. */
const char *qd_run_error(const qd_run *r);
/* The last tick's stats (also returned by qd_run_tick). */
void qd_run_last(const qd_run *r, qd_tick_stats *st);

/* Make the calling thread user-interactive (QOS_CLASS_USER_INTERACTIVE: macOS keeps it on the
   performance cores). Returns 0 on success. */
int qd_run_set_qos(void);

/* A token's name ("F1(5)", "NT", "PAD", "<bad 123>") into buf (>= 24 bytes); returns buf. */
const char *qd_run_token_name(int32_t t, char *buf);

#ifdef __cplusplus
}
#endif
#endif
