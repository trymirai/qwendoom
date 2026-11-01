/* qwendoom's hooks into DOOM: what the doom/patches call, and what qd_sim.c (the sim side,
   placed in ROM by doom/split.py) and qd_platform.c (the host side) export to each other. */
#ifndef QD_DOOM_H
#define QD_DOOM_H

#include <stddef.h>
#include <stdint.h>
#include "qd_mem.h"

/* --- the sim (qd_sim.c) --- */

void sim_main(void) __attribute__((noreturn));

/* P_KillMobj's AM_Stop: the sim cannot touch the automap (host state), so it raises this flag
   and the host closes the automap after the tick (qd_after_sim_tick) */
void qd_sim_am_stop(void);
extern int qd_am_stop_request;

/* sound events the sim (and the host) start or stop, for a later sound feature: a ring in sim RAM.
   sfx >= 0: S_StartSound(origin, sfx); sfx = -1: S_StopSound(origin). */
#define QD_SOUND_RING 32
typedef struct {
    void *origin;
    int sfx;
} qd_sound_event_t;
extern qd_sound_event_t qd_sound_ring[QD_SOUND_RING];
extern unsigned qd_sound_head;  /* events written so far; the latest is [(head - 1) % RING] */

/* the last I_Error's format and arguments, kept in RAM so every sim mode can report it */
extern const char *qd_error_fmt;
extern void *qd_error_args;

/* --- the host (qd_platform.c) --- */

void qd_sim_tick(void);         /* G_Ticker: SIM_TICK, then wait for the sim's P_Ticker */
void qd_after_sim_tick(void);   /* the host's reaction to the sim's flags */
void qd_level_begin(void);      /* P_SetupLevel starts (DROM writes are expected) */
void qd_level_loaded(void);     /* P_SetupLevel is done: the sim needs a resync */

/* read-only level data (vertexes, segs, subsectors, nodes, blockmap, reject, sector line lists)
   lives in a bump region of DROM after the sim's .rodata; it is reset at every level load */
void *qd_drom_alloc(size_t n);
/* -qdmeasure (doom/measure.py --maps): level data goes to the host heap and a large host-side
   level zone instead of DROM / sim RAM, so that any level loads and its needs can be measured
   (qd_level_ram_need / qd_drom_level_need); the sim cannot run then */
extern int qd_measure_levels;
extern size_t qd_drom_level_need;
void qd_drom_mark(void);        /* everything allocated so far is permanent (startup tables) */
void qd_drom_reset(void);       /* free back to the mark */

#define QD_IN_SIM_RAM(p) ((uintptr_t)(p) < QD_RAM_BASE + QD_RAM_SIZE)

#endif
