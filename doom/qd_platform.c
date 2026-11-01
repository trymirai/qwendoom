/* The host side of qwendoom's DOOM platform layer: main, the doomgeneric DG_* hooks, i_system /
   i_video / s_sound replacements, and the services the patches call (sim ticks, the DROM level
   allocator). Everything here runs on the host interpreter only (doom/split.py places it in HOST);
   it talks to the interpreter through ecalls (native/host/qd_host_abi.h).

   Timing is a fake clock: one game tic per frame (singletics) and 1000/35 ms of clock per frame,
   so a run depends only on its inputs, never on wall time. Video: DOOM's own 8-bit 320x200 screen
   and the current (gamma-corrected) palette go to the host as they are. No sound or music yet. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

#include "doomtype.h"
#include "doomkeys.h"
#include "doomgeneric.h"
#include "d_event.h"
#include "d_main.h"
#include "i_system.h"
#include "i_video.h"
#include "i_sound.h"
#include "m_argv.h"
#include "s_sound.h"
#include "v_video.h"
#include "z_zone.h"
#include "am_map.h"
#include "doomstat.h"
#include "qd_doom.h"
#include "qd_sys.h"

extern boolean singletics;
extern boolean automapactive;
void M_FindResponseFile(void);
void D_DoomMain(void);

/* --- main and the fake clock ------------------------------------------------------------- */

static uint32_t frames;    /* frames drawn */
static uint32_t slept_ms;  /* DG_SleepMs total */

int main(int argc, char **argv)
{
    myargc = argc;
    myargv = argv;
    singletics = true;  /* exactly one tic per TryRunTics, whatever the clock says */
    M_FindResponseFile();
    qd_measure_levels = M_CheckParm("-qdmeasure") > 0;
    DG_Init();
    D_DoomMain();
    for (;;)
        doomgeneric_Tick();
}

void DG_Init(void) {}
void DG_DrawFrame(void) {}
void DG_SleepMs(uint32_t ms) { slept_ms += ms; }

/* ceil(frames * 1000 / 35) so that I_GetTime() (ms * 35 / 1000) is exactly the frame count */
uint32_t DG_GetTicksMs(void)
{
    return 1000 + (frames * 1000 + 34) / 35 + slept_ms;
}

int DG_GetKey(int *pressed, unsigned char *key)
{
    long k = qd_ecall(QD_EC_GETKEY, 0, 0, 0);
    if (k < 0)
        return 0;
    *pressed = (k >> 8) & 1;
    *key = k & 0xFF;
    return 1;
}

void DG_SetWindowTitle(const char *title)
{
    qd_ecall(QD_EC_TITLE, (long)title, 0, 0);
}

/* --- the sim's tick ------------------------------------------------------------------------ */

void qd_sim_tick(void)
{
    qd_ecall(QD_EC_SIM_TICK, 0, 0, 0);
}

void qd_after_sim_tick(void)
{
    if (qd_am_stop_request) {
        qd_am_stop_request = 0;
        if (automapactive)
            AM_Stop();
    }
}

void qd_level_begin(void)
{
    qd_ecall(QD_EC_LEVEL, 1, 0, 0);
}

void qd_level_loaded(void)
{
    qd_ecall(QD_EC_LEVEL, 0, 0, 0);
}

/* --- DROM: the sim's read-only level data -------------------------------------------------- */

extern char __drom_end[];  /* the end of the sim's .rodata (linker script) */
static uintptr_t drom_top, drom_mark, drom_peak;
int qd_measure_levels;
size_t qd_drom_level_need;  /* -qdmeasure: the level data allocated since the last reset */

void *qd_drom_alloc(size_t n)
{
    if (qd_measure_levels) {
        qd_drom_level_need += (n + 3) & ~(size_t)3;
        return malloc(n);  /* (never freed: a measurement run loads one level) */
    }
    if (!drom_top)
        drom_top = drom_mark = ((uintptr_t)__drom_end + 3) & ~(uintptr_t)3;
    uintptr_t p = drom_top;
    if (n > QD_DROM_BASE + QD_DROM_SIZE - p)
        I_Error("qd_drom_alloc: DROM full (%u bytes requested, %u free of %u)", (unsigned)n,
                (unsigned)(QD_DROM_BASE + QD_DROM_SIZE - p), (unsigned)QD_DROM_SIZE);
    drom_top = (p + n + 3) & ~(uintptr_t)3;
    if (drom_top > drom_peak)
        drom_peak = drom_top;
    return (void *)p;
}

void qd_drom_mark(void)
{
    qd_drom_alloc(0);
    drom_mark = drom_top;
}

void qd_drom_reset(void)
{
    qd_drom_level_need = 0;
    qd_drom_alloc(0);
    drom_top = drom_mark;
}

/* --- i_system ------------------------------------------------------------------------------- */

typedef struct atexit_entry {
    atexit_func_t func;
    struct atexit_entry *next;
} atexit_entry_t;

static atexit_entry_t *exit_funcs;

void I_AtExit(atexit_func_t func, boolean run_on_error)
{
    (void)run_on_error;  /* I_Error exits the interpreter at once; nothing runs on error */
    atexit_entry_t *e = malloc(sizeof(*e));
    e->func = func;
    e->next = exit_funcs;
    exit_funcs = e;
}

void I_Init(void) {}
void I_BindVariables(void) {}
boolean I_ConsoleStdout(void) { return false; }

#define ZONE_MB 16
byte *I_ZoneBase(int *size)
{
    *size = ZONE_MB * 1024 * 1024;
    byte *zone = malloc(*size);
    if (!zone)
        I_Error("I_ZoneBase: no memory for a %d MiB zone", ZONE_MB);
    return zone;
}

void I_PrintBanner(char *msg)
{
    printf("%s\n", msg);
}

void I_PrintDivider(void)
{
    printf("===========================================================================\n");
}

void I_PrintStartupBanner(char *gamedescription)
{
    I_PrintDivider();
    printf("%s\n", gamedescription);
    I_PrintDivider();
}

void I_Quit(void)
{
    for (atexit_entry_t *e = exit_funcs; e; e = e->next)
        e->func();
    exit(0);
}

/* the fixed DOS 6.22 dump chocolate-doom uses for NULL-read emulation (p_setup.c's null sector) */
boolean I_GetMemoryValue(unsigned int offset, void *value, int size)
{
    static const unsigned char dump[10] = {0x57, 0x92, 0x19, 0x00, 0xF4, 0x06, 0x70, 0x00, 0x16, 0x00};
    unsigned v = dump[offset] | dump[offset + 1] << 8 | dump[offset + 2] << 16 | dump[offset + 3] << 24;
    switch (size) {
    case 1: *(unsigned char *)value = v; return true;
    case 2: *(unsigned short *)value = v; return true;
    case 4: *(unsigned *)value = v; return true;
    }
    return false;
}

/* --- i_video: DOOM's 8-bit screen and palette, as they are ---------------------------------- */

byte *I_VideoBuffer;
boolean screenvisible = true;
boolean screensaver_mode;
int usegamma;
int usemouse;
float mouse_acceleration = 2.0f;
int mouse_threshold = 10;
static byte palette_rgb[768];

void I_InitGraphics(void)
{
    I_VideoBuffer = Z_Malloc(SCREENWIDTH * SCREENHEIGHT, PU_STATIC, NULL);
    screenvisible = true;
}

void I_ShutdownGraphics(void) {}
void I_StartFrame(void) {}
void I_UpdateNoBlit(void) {}

void I_GetEvent(void);
void I_StartTic(void)
{
    I_GetEvent();
}

void I_FinishUpdate(void)
{
    qd_ecall(QD_EC_FRAME, (long)I_VideoBuffer, (long)palette_rgb, 0);
    frames++;
}

void I_ReadScreen(byte *scr)
{
    memcpy(scr, I_VideoBuffer, SCREENWIDTH * SCREENHEIGHT);
}

void I_SetPalette(byte *palette)
{
    for (int i = 0; i < 768; i++)
        palette_rgb[i] = gammatable[usegamma][palette[i]];
}

int I_GetPaletteIndex(int r, int g, int b)
{
    int best = 0, best_diff = 1 << 30;
    for (int i = 0; i < 256; i++) {
        int dr = r - palette_rgb[3 * i], dg = g - palette_rgb[3 * i + 1], db = b - palette_rgb[3 * i + 2];
        int d = dr * dr + dg * dg + db * db;
        if (d < best_diff) {
            best = i;
            best_diff = d;
        }
    }
    return best;
}

void I_BeginRead(void) {}
void I_EndRead(void) {}
void I_SetWindowTitle(char *title) { DG_SetWindowTitle(title); }
void I_GraphicsCheckCommandLine(void) {}
void I_SetGrabMouseCallback(grabmouse_callback_t func) { (void)func; }
void I_EnableLoadingDisk(void) {}
void I_BindVideoVariables(void) {}
void I_DisplayFPSDots(boolean dots_on) { (void)dots_on; }
void I_CheckIsScreensaver(void) {}

/* --- s_sound: no sound or music yet (S_StartSound / S_StopSound feed the ring in qd_sim.c) --- */

void S_Init(int sfxVolume, int musicVolume) { (void)sfxVolume, (void)musicVolume; }
void S_Shutdown(void) {}
void S_Start(void) {}
void S_StartMusic(int music_id) { (void)music_id; }
void S_ChangeMusic(int music_id, int looping) { (void)music_id, (void)looping; }
boolean S_MusicPlaying(void) { return false; }
void S_StopMusic(void) {}
void S_PauseSound(void) {}
void S_ResumeSound(void) {}
void S_UpdateSounds(mobj_t *listener) { (void)listener; }
void S_SetMusicVolume(int volume) { (void)volume; }
void S_SetSfxVolume(int volume) { (void)volume; }

int sfxVolume = 8;
int musicVolume = 8;
int snd_channels = 8;
