/* The ABI between DOOM (the guest, built by doom/) and the host interpreter (native/host): ecall
   numbers and the events qd_host_run returns. Included by both sides, so it stays plain C with no
   dependencies. See docs/doom.md. */
#ifndef QD_HOST_ABI_H
#define QD_HOST_ABI_H

/* ecall a7 values. The sim may use only QD_EC_PUTCHAR (a0 = QD_YIELD_BYTE: YIELD) and QD_EC_EXIT,
   the two the model's circuit implements; everything else is a host service. Paths are relative
   to the config directory (reads fall back to the WAD's directory); see docs/doom.md. */
#define QD_EC_PUTCHAR  1      /* a0: a byte to the guest's stdout; in the sim a0 = 256 is YIELD */
#define QD_EC_WRITE    64     /* a0 fd (1 stdout, 2 stderr), a1 buf, a2 len -> a0 = len */
#define QD_EC_EXIT     93     /* a0 exit code; a1 != 0: I_Error, a1 = fmt, a2 = its va_list */
#define QD_EC_OPEN     0x400  /* a0 path, a1 QD_O_* -> a0 fd (>= 3) or -1 */
#define QD_EC_CLOSE    0x401  /* a0 fd -> 0 / -1 */
#define QD_EC_READ     0x402  /* a0 fd, a1 buf, a2 len -> bytes read or -1 */
#define QD_EC_FWRITE   0x403  /* a0 fd, a1 buf, a2 len -> bytes written or -1 */
#define QD_EC_SEEK     0x404  /* a0 fd, a1 offset, a2 whence (0 set, 1 cur, 2 end) -> pos or -1 */
#define QD_EC_REMOVE   0x405  /* a0 path -> 0 / -1 */
#define QD_EC_RENAME   0x406  /* a0 old path, a1 new path -> 0 / -1 */
#define QD_EC_MKDIR    0x407  /* a0 path -> 0 (also if it exists) / -1 */
#define QD_EC_FRAME    0x410  /* a0 the 320x200 screen (palette indices), a1 the palette (768 RGB) */
#define QD_EC_SIM_TICK 0x411  /* run one P_Ticker in the sim; the host waits until it is done */
#define QD_EC_GETKEY   0x412  /* -> a0 = pressed << 8 | doom key, or -1 when the queue is empty */
#define QD_EC_LEVEL    0x413  /* P_SetupLevel finished: the sim needs a resync */
#define QD_EC_TITLE    0x414  /* a0 the window title */

#define QD_O_READ  0
#define QD_O_WRITE 1  /* create / truncate */

/* qd_host_run results */
#define QD_EV_BUDGET   0   /* max_instr instructions ran without an event */
#define QD_EV_FRAME    1   /* a frame is ready (qd_host_frame) */
#define QD_EV_SIM_TICK 2   /* the host wants one sim tick (then qd_host_complete_sim_tick) */
#define QD_EV_EXIT     3   /* the guest exited (qd_host_exit_code) */
#define QD_EV_ERROR    (-1) /* a fault or an I_Error (qd_host_error) */

#endif
