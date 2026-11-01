/* qdhost: run DOOM on the host interpreter from the command line (timing and smoke runs).

     qdhost ELF WAD [-frames N] [-native] [-echo] [-ppm DIR] [-- doom args...]

   Without -native the sim ticks are run anyway (natively) so the game advances; -native only adds
   the per-tick statistics. Frames can be dumped as PPM files. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "qd_host.h"

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static void write_ppm(const char *dir, int n, const uint8_t *idx, const uint8_t *pal)
{
    char path[1024];
    snprintf(path, sizeof path, "%s/frame%05d.ppm", dir, n);
    FILE *f = fopen(path, "wb");
    if (!f)
        return;
    fprintf(f, "P6 320 200 255\n");
    for (int i = 0; i < 64000; i++)
        fwrite(pal + 3 * idx[i], 1, 3, f);
    fclose(f);
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: qdhost ELF WAD [-frames N] [-native] [-echo] [-ppm DIR] [-- args]\n");
        return 2;
    }
    int frames = 350, echo = 0, verbose = 0, dargc = 0;
    const char *ppm = NULL, *dargv[64];
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "-frames") && i + 1 < argc) frames = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-echo")) echo = 1;
        else if (!strcmp(argv[i], "-native")) verbose = 1;
        else if (!strcmp(argv[i], "-ppm") && i + 1 < argc) ppm = argv[++i];
        else if (!strcmp(argv[i], "--")) {
            for (i++; i < argc && dargc < 64; i++) dargv[dargc++] = argv[i];
        }
    }
    qd_host *h = qd_host_new(argv[1], argv[2], dargc, dargv);
    if (!h)
        return 1;
    qd_host_set_echo(h, echo);
    static uint8_t idx[64000], pal[768];
    int nframes = 0;
    uint64_t sim_instr = 0, sim_tokens = 0, ticks = 0;
    double t0 = now(), tsim = 0;
    for (;;) {
        int ev = qd_host_run(h, 1ull << 40);
        if (ev == QD_EV_FRAME) {
            if (ppm && qd_host_frame(h, idx, pal) == 0)
                write_ppm(ppm, nframes, idx, pal);
            if (++nframes >= frames)
                break;
        } else if (ev == QD_EV_SIM_TICK) {
            if (qd_host_need_resync(h))
                qd_host_resync(h);
            qd_host_take_pokes(h, NULL, 0);
            qd_sim_stats st;
            double ts = now();
            int64_t n = qd_host_sim_tick_native(h, &st);
            tsim += now() - ts;
            if (n < 0) {
                fprintf(stderr, "sim error: %s\n", qd_host_error(h));
                return 1;
            }
            sim_instr += n;
            sim_tokens += st.tokens;
            ticks++;
            if (verbose)
                printf("tick %llu: %lld instr %llu tokens\n", (unsigned long long)ticks, (long long)n,
                       (unsigned long long)st.tokens);
            qd_host_complete_sim_tick(h);
        } else if (ev == QD_EV_EXIT) {
            printf("exit %d\n", qd_host_exit_code(h));
            break;
        } else if (ev == QD_EV_ERROR) {
            fprintf(stderr, "error: %s\n", qd_host_error(h));
            return 1;
        }
    }
    double dt = now() - t0;
    uint64_t hi = qd_host_instructions(h);
    printf("%d frames, %llu ticks in %.3f s: host %llu instr (%.1f MIPS, %.0f per frame), "
           "sim %llu instr (%.1f MIPS), %llu tokens\n",
           nframes, (unsigned long long)ticks, dt, (unsigned long long)hi, hi / (dt - tsim) / 1e6,
           (double)hi / (nframes ? nframes : 1), (unsigned long long)sim_instr,
           tsim > 0 ? sim_instr / tsim / 1e6 : 0.0, (unsigned long long)sim_tokens);
    qd_host_free(h);
    return 0;
}
