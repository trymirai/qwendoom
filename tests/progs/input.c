// The sim's main loop in miniature: each tick reads host pokes from the input port (csrr time:
// address, word pairs, address 0 ends the list), applies them, updates a small world and yields
// (BYTE(256)); a poke to `quit` ends the run.
#include "env/rt.h"

volatile uint32_t quit;
static uint32_t world[64];
uint32_t cmd[4];  // the host pokes these (like players[].cmd)

static void tick(void) {
    for (int i = 0; i < 64; i++) world[i] = world[i] * 5 + cmd[i & 3] + (uint32_t)i;
    cmd[3] ^= world[cmd[0] & 63];
}

int main(void) {
    uint32_t ticks = 0;
    while (!quit) {
        for (;;) {
            uint32_t a = qd_in();
            if (!a) break;
            *(volatile uint32_t *)a = qd_in();
        }
        tick();
        ticks++;
        qd_yield();
    }
    uint32_t c = qd_crc32(world, sizeof world, ticks);
    qd_puts("input ");
    qd_puthex(c);
    qd_putchar('\n');
    return 0;
}
