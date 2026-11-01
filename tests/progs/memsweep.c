// RAM across the whole 256 KB and DROM across the whole 512 KB: words, halves and bytes at
// addresses that reach every RAM key head / value head / row bit and every DROM key head /
// value head / row pair (qwendoom.core.ram_layout, core.QdCore.byte_table).
#include "env/rt.h"

#define DROM_WORDS 129024  // big_rom (gen/bigrom.S): 504 KB of hash(i), the DROM's rest is code's
extern const uint32_t big_rom[DROM_WORDS];

static uint32_t hash(uint32_t x) {
    x *= 0x9E3779B1u;
    x ^= x >> 15;
    x *= 0x85EBCA77u;
    x ^= x >> 13;
    return x;
}

// RAM word index of sweep step k (injective on 0..1023): bits 0..4 and 12..15 from k's low 9
// bits (all 512 key head / value head pairs), the row bits 5..11 from a hash of k; with bits
// 12..15 zero, rows 2 / 3 keep the sweep off the null guard, the stack and the globals
// (0..0x100, 0x2000..0x3400)
static uint32_t ram_word(uint32_t k) {
    uint32_t hi = (k >> 5) & 15;
    uint32_t row = hi ? ((k * 37 + (k >> 9) * 19) & 127) : 2 + (k >> 9);
    return (k & 31) | row << 5 | hi << 12;
}

int main(void) {
    volatile uint32_t *ram = (volatile uint32_t *)0;
    uint32_t bad = 0, sum = 0;
    // words
    for (uint32_t k = 0; k < 1024; k++) ram[ram_word(k)] = hash(k);
    for (uint32_t k = 0; k < 1024; k++) {
        uint32_t v = ram[ram_word(k)];
        sum += v;
        if (v != hash(k)) bad++;
    }
    // halves and bytes (merged into the words just written), read back as all widths
    volatile uint16_t *h16 = (volatile uint16_t *)0;
    volatile uint8_t *b8 = (volatile uint8_t *)0;
    for (uint32_t k = 0; k < 256; k++) {
        uint32_t wi = ram_word(k * 4 + 1);
        if (k & 1) h16[2 * wi + (k >> 1 & 1)] = (uint16_t)(k * 0x0101 + 7);
        else b8[4 * wi + (k >> 1 & 3)] = (uint8_t)(k ^ 0x5A);
    }
    for (uint32_t k = 0; k < 256; k++) {
        uint32_t wi = ram_word(k * 4 + 1);
        sum = sum * 31 + ram[wi] + (int16_t)h16[2 * wi + 1] + (int8_t)b8[4 * wi + 3];
    }
    // DROM: every key head and value head (index bits 10..16), row pairs and lanes varied
    for (uint32_t k = 0; k < 1024; k++) {
        uint32_t wi = (k & 127) << 10 | ((k * 13 + (k >> 7)) & 63) << 4 | ((k >> 7) & 15);
        if (wi >= DROM_WORDS) wi -= DROM_WORDS;
        uint32_t v = big_rom[wi];
        if (v != hash(wi)) bad++;
        const uint8_t *bp = (const uint8_t *)&big_rom[wi];
        const uint16_t *hp = (const uint16_t *)&big_rom[wi];
        sum += v + (int8_t)bp[k & 3] + bp[(k >> 2) & 3] + (int16_t)hp[k & 1] + hp[(k >> 1) & 1];
    }
    qd_puts("memsweep ");
    qd_puthex(sum);
    qd_puts(bad ? " BAD\n" : " ok\n");
    return (int)bad;
}
