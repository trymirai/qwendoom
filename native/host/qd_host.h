/* The qwendoom host computer: a fast RV32IM+Zba interpreter that runs DOOM's host side
   (build/doom.elf) over the four regions of qd_mem.h, its ecall services (files, frames, keys,
   sim ticks), dirty tracking of the RAM the host writes, and the native sim mode (a second CPU
   context that runs sim_main with region checks). Built as build/libqdhost.dylib; wrapped by
   qwendoom/host.py; documented in docs/doom.md. */
#ifndef QD_HOST_H
#define QD_HOST_H

#include <stdint.h>
#include "qd_mem.h"
#include "qd_host_abi.h"

typedef struct qd_host qd_host;

/* one native sim tick (qd_host_sim_tick_native) */
typedef struct {
    uint64_t instructions;  /* sim instructions executed, the input csrr reads included */
    uint64_t tokens;        /* trace tokens the circuit emits for them (chipformer records, R
                               refreshes and the host-injected input records included) */
    uint32_t input_words;   /* csrr time reads (2 per poke + the terminating 0) */
    uint32_t loads, stores, drom_loads;
    uint32_t min_sp;        /* the lowest sp of the tick */
    uint32_t ops[64];       /* executed instructions per uop (qd_host_uop_name) */
} qd_sim_stats;

/* Create a host: load the ELF into fresh memory, set up the host CPU at _start with argv =
   {"doom", "-iwad", <wad basename>, argv...}. File reads are limited to the WAD's directory;
   writes are refused unless qd_host_set_config_dir names a directory for them. NULL on failure
   (the reason goes to stderr). */
qd_host *qd_host_new(const char *elf_path, const char *wad_path, int argc, const char *const *argv);
void qd_host_free(qd_host *h);
void qd_host_set_config_dir(qd_host *h, const char *dir);
void qd_host_set_echo(qd_host *h, int echo);  /* copy the guest's stdout / stderr to stderr */

/* Run the host CPU for at most max_instr instructions: QD_EV_FRAME, QD_EV_SIM_TICK, QD_EV_EXIT,
   QD_EV_ERROR or QD_EV_BUDGET. After QD_EV_SIM_TICK the host waits (run returns it again) until
   qd_host_complete_sim_tick. */
int qd_host_run(qd_host *h, uint64_t max_instr);

qd_mem *qd_host_mem(qd_host *h);
const char *qd_host_error(qd_host *h);  /* the last error, "" if none */
int qd_host_exit_code(qd_host *h);

/* the latest frame: 64000 palette indices (320x200) and the 768-byte RGB palette; 0, or -1 if no
   frame has been drawn yet */
int qd_host_frame(qd_host *h, uint8_t *idx, uint8_t *pal);

/* input: a DOOM key code (doomkeys.h) going down / up; queued until the guest polls */
void qd_host_key(qd_host *h, int doom_key, int down);
/* the DOOM key of a UI action (ARCHITECTURE.md: left right forward back strafeleft straferight
   fire use run menu enter weapon1 weapon2 weapon3), or -1 */
int qd_host_action_key(const char *action);

/* Host -> sim state: the RAM words the host changed since the sim last saw RAM, as sorted
   (addr, value) pairs ending with a single 0: n = 2 * pokes + 1 words. Clears the dirty set and
   queues the words as the native sim's input. Returns n, or -n (nothing taken) if max < n. */
int qd_host_take_pokes(qd_host *h, uint32_t *out, int max);
int qd_host_pending_pokes(qd_host *h);  /* pokes take_pokes would return now */

/* A level load (or a DROM write) means the model must be reloaded from a snapshot: RAM, DROM,
   ROM and the sim's entry context (pc = sim_main, sp = 0x3000, gp, other registers 0). */
int qd_host_need_resync(qd_host *h);
void qd_host_sim_entry(qd_host *h, uint32_t *pc, uint32_t *sp, uint32_t *gp);
void qd_host_resync(qd_host *h);  /* the snapshot was taken: clear the flag and the dirty set,
                                     reset the native sim context to the entry */

/* Native sim mode: run sim_main in the second context until YIELD, popping the queued pokes on
   csrr time (then 0). Any fetch outside ROM, load outside RAM / DROM, store outside RAM (or into
   the null guard below 0x100), misaligned access, other ecall or illegal instruction is an error
   (qd_host_error names the PC's symbol). Returns the instruction count, or -1. The host stays
   waiting until qd_host_complete_sim_tick. */
int64_t qd_host_sim_tick_native(qd_host *h, qd_sim_stats *stats);
/* the tick is done (in any mode; RAM now holds the sim's stores): resume the host */
int qd_host_complete_sim_tick(qd_host *h);

/* counters and lookups */
uint64_t qd_host_instructions(qd_host *h);  /* host instructions so far */
uint64_t qd_host_sim_instructions(qd_host *h);
int qd_host_counters(qd_host *h, uint64_t *out, int n);  /* see QD_CTR_* */
uint32_t qd_host_symbol(qd_host *h, const char *name);   /* address, 0 if absent */
int qd_host_symbolize(qd_host *h, uint32_t addr, char *buf, int n);
const char *qd_host_uop_name(int uop);
int qd_host_output(qd_host *h, char *buf, int n);  /* drain the guest's stdout / stderr */

enum {
    QD_CTR_FRAMES, QD_CTR_SIM_TICKS, QD_CTR_RESYNCS, QD_CTR_LEVELS, QD_CTR_DROM_WRITES,
    QD_CTR_DROM_WRITES_MIDLEVEL, QD_CTR_RAM_WRITES, QD_CTR_POKES, QD_CTR_HOST_NULL_ACCESSES,
    QD_CTR_SIM_TOKENS, QD_CTR_N
};

#endif
