/* The sim side of qwendoom's DOOM: sim_main (the entry the model's CPU runs, see ARCHITECTURE.md)
   and the stubs the playsim reaches that must not touch host state. doom/split.py places these in
   ROM (code) and RAM (data) because they are reachable from sim_main; nothing here may call host
   code or use an ecall other than YIELD (a7 = 1, a0 = 256) and exit (a7 = 93). */
#include <stdarg.h>
#include <stdint.h>

#include "doomtype.h"
#include "i_system.h"
#include "p_local.h"
#include "p_tick.h"
#include "s_sound.h"
#include "qd_doom.h"

int qd_am_stop_request;
qd_sound_event_t qd_sound_ring[QD_SOUND_RING];
unsigned qd_sound_head;
const char *qd_error_fmt;
void *qd_error_args;

/* the input port: csrr rd, time (the host injects the next input word) */
static inline uint32_t qd_in(void)
{
    uint32_t v;
    __asm__ volatile("csrr %0, time" : "=r"(v));
    return v;
}

static inline void qd_ecall2(long n, long a0, long a1, long a2)
{
    register long r0 __asm__("a0") = a0;
    register long r1 __asm__("a1") = a1;
    register long r2 __asm__("a2") = a2;
    register long r7 __asm__("a7") = n;
    __asm__ volatile("ecall" : "+r"(r0) : "r"(r1), "r"(r2), "r"(r7) : "memory");
}

void sim_main(void)
{
    for (;;) {
        for (;;) {  /* host pokes: (addr, word) pairs, addr 0 ends the list */
            uint32_t a = qd_in();
            if (!a)
                break;
            *(volatile uint32_t *)a = qd_in();
        }
        P_Ticker();
        qd_ecall2(1, QD_YIELD_BYTE, 0, 0);  /* YIELD: BYTE(256), the tick is done */
    }
}

void qd_sim_am_stop(void)
{
    qd_am_stop_request = 1;
}

static void sound_event(void *origin, int sfx)
{
    qd_sound_event_t *e = &qd_sound_ring[qd_sound_head % QD_SOUND_RING];
    e->origin = origin;
    e->sfx = sfx;
    qd_sound_head++;
}

void S_StartSound(void *origin, int sfx_id)
{
    sound_event(origin, sfx_id);
}

void S_StopSound(mobj_t *origin)
{
    sound_event(origin, -1);
}

void I_Tactile(int on, int off, int total)
{
    (void)on, (void)off, (void)total;
}

/* I_Error from anywhere (host or sim): exit with the format and its arguments in a1 / a2; the
   host interpreter formats the message from guest memory. The copies in RAM let a model-mode
   runner (which sees only the HALT token and the mirrored stores) report it too. */
void I_Error(char *error, ...)
{
    va_list ap;
    va_start(ap, error);
    qd_error_fmt = error;
    qd_error_args = (void *)ap;
    for (;;)
        qd_ecall2(93, -1, (long)error, (long)ap);
}
