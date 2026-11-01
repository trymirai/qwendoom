/* The qwendoom inference engine: Qwen3.5's forward pass over a compiled checkpoint, one token at a
   time, exact (the same tokens and residuals as stock Qwen3_5ForCausalLM in bf16), sparse, for
   Apple silicon. See docs/engine.md and ARCHITECTURE.md ("The engine").

   A run: qd_engine_reset, qd_engine_step(PAD), load the snapshot rows (set_row / set_rows),
   then step the tokens. An engine is single-threaded from the caller's point of view (it may
   run worker threads inside a step); use one engine per caller thread. */
#ifndef QD_ENGINE_H
#define QD_ENGINE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define QD_ENGINE_ABI 1

typedef struct qd_engine qd_engine;

/* --- the ARCHITECTURE.md API ------------------------------------------------------------- */

/* Open weights_dir (config.json + model.safetensors, bf16). n_threads <= 0 picks the default
   (QD_THREADS or 1); with more, n_threads - 1 worker threads (user-interactive QoS) take shares
   of the big sublayers and spin between steps (sleeping after 2 ms idle). The compact layouts
   are cached under <dir of this dylib>/engine-cache (QD_ENGINE_CACHE overrides; "off"
   disables), keyed by the checkpoint's sha256. NULL on error (qd_engine_last_error() says
   why). */
qd_engine *qd_engine_open(const char *weights_dir, int n_threads);
void qd_engine_close(qd_engine *e);
void qd_engine_reset(qd_engine *e); /* zero every recurrent state and conv window */
void qd_engine_set_row(qd_engine *e, int layer, int value_head, int row, const float *vec);
int qd_engine_step(qd_engine *e, int token); /* feed one token, return the greedy next one */

/* --- extensions ---------------------------------------------------------------------------- */

#define QD_OPEN_NO_CACHE 1 /* neither read nor write the layout cache */
#define QD_OPEN_VERBOSE 2  /* print what open does to stderr */
#define QD_OPEN_GENERIC 4  /* no specialized paths (one-entry unit records, cached prefixes):
                              the same results, for tests */
#define QD_OPEN_REFRESH 8  /* drop every cached prefix at every 7th step (its base is recomputed,
                              deferred projections computed first): the same results, for tests */

/* cache_dir NULL: the default above. */
qd_engine *qd_engine_open_ex(const char *weights_dir, int n_threads, const char *cache_dir,
                             int flags);
const char *qd_engine_last_error(void);

/* n rows of one value head: rows[i] gets vecs[i * dv .. +dv]. */
void qd_engine_set_rows(qd_engine *e, int layer, int value_head, int n, const int *rows,
                        const float *vecs);
void qd_engine_get_row(qd_engine *e, int layer, int value_head, int row, float *out);

/* Feed a token whose successor the caller already knows (a host-injected record): no head, and
   the trailing layers that only feed the head are skipped. */
void qd_engine_step_forced(qd_engine *e, int token);
/* Feed n tokens; the prediction after the last one (forced steps before it). */
int qd_engine_feed(qd_engine *e, const int *tokens, int n);
/* Free-running greedy decoding: feed token, then each prediction, n times or until a
   prediction equals stop (stop < 0: never). out[i] = the i-th prediction. Returns the count. */
int qd_engine_generate(qd_engine *e, int token, int n, int stop, int *out);
/* The same with a set of stops: feed token, then each prediction, n times or until a prediction
   is one of stops[0 .. n_stops) (the stop is out's last entry and is not fed: pass it as the
   next call's token to go on). out[i] = the i-th prediction. Returns the count. */
int qd_engine_generate_until(qd_engine *e, int token, int n, const int *stops, int n_stops,
                             int *out);
/* Teacher-forced check: feed toks[0..n-2]; returns the first i with prediction != toks[i + 1]
   (skipping i where skip[i + 1] != 0 when skip is not NULL), or -1 if all agree. */
int qd_engine_check(qd_engine *e, const int *toks, int n, const uint8_t *skip);

/* 1: always compute the full tied head (vocab x code dims) and its argmax; 0 (default): read
   the sign code and fall back to the full head only when that is not provably the argmax. */
void qd_engine_set_exact_head(qd_engine *e, int on);
/* buf (n_layers x hidden floats) or NULL: when set, every step writes the residual after each
   decoder layer there (debugging, bit-identity checks). */
void qd_engine_capture(qd_engine *e, float *buf);
/* 1: time every stage (adds a few % overhead); see qd_engine_stats. */
void qd_engine_set_profile(qd_engine *e, int on);

typedef struct {
    int hidden, n_layers, vocab, key_heads, value_heads, key_dim, value_dim, conv_kernel;
    int n_threads, n_code, sign_head, last_needed_layer;
    int64_t nnz_compact; /* nonzero weights kept in the compact layouts */
    double open_seconds;
    int from_cache;
    char sha256[65];
    int prefix_dims; /* residual dims [0, prefix_dims) are constant (the cached prefix) */
} qd_engine_info;
void qd_engine_get_info(qd_engine *e, qd_engine_info *info);

/* Stages of a step (profile and work counters). */
enum {
    QD_ST_EMBED, QD_ST_NORM, QD_ST_QK, QD_ST_CONV, QD_ST_DELTA, QD_ST_VLAZY, QD_ST_OUT,
    QD_ST_GATE, QD_ST_UP, QD_ST_DOWN, QD_ST_RESID, QD_ST_HEAD, QD_ST_SYNC, QD_N_STAGES
};
typedef struct {
    uint64_t steps, forced;
    uint64_t head_sign, head_exact;     /* how the head was decided */
    uint64_t slow_transcendental;      /* silu / sigmoid / exp outside their exact ranges */
    uint64_t mac[QD_N_STAGES];          /* multiply-adds done per stage */
    double ns[QD_N_STAGES];             /* time per stage (profile on) */
    uint64_t key_writes, query_reads, active_units;
    /* sublayers where a bf16 x bf16 product might leave fp32's normal range (then a fused
       multiply-add might round differently from the reference's multiply, add): must stay 0 */
    uint64_t range_risk;
    uint64_t prefix_misses; /* cached projection prefixes recomputed (prefix_t) */
    uint64_t bound_skips;   /* eager-channel regions proved dead by their bound (qkbound_t) */
} qd_engine_stats;
void qd_engine_get_stats(qd_engine *e, qd_engine_stats *s, int reset);
const char *qd_engine_stage_name(int stage);
/* ns per round trip of the worker pool (an empty job on every thread), over n trips */
double qd_engine_bench_sync(qd_engine *e, int n);

#ifdef __cplusplus
}
#endif
#endif
