// Byte and half stores / loads: unaligned-in-word lanes, sign extension, overlapping memmove,
// packed structs, a byte histogram and crc32 over the result.
#include "env/rt.h"

static uint8_t buf[300];
static uint16_t halves[97];
static int8_t sbytes[64];
struct __attribute__((packed)) rec { uint8_t tag; uint16_t len; uint8_t flags; uint32_t val; };
static struct rec recs[12];
static uint32_t hist[16];

int main(void) {
    for (int i = 0; i < 300; i++) buf[i] = (uint8_t)(i * 7 + 3);
    for (int i = 0; i < 97; i++) halves[i] = (uint16_t)(i * 0x1357 ^ 0xA5A5);
    for (int i = 0; i < 64; i++) sbytes[i] = (int8_t)(i * 9 - 100);
    memmove(buf + 5, buf + 1, 200);
    memmove(buf + 2, buf + 9, 150);
    for (int i = 0; i < 12; i++) {
        recs[i].tag = (uint8_t)i;
        recs[i].len = (uint16_t)(i * 1000 + 1);
        recs[i].flags = (uint8_t)(0x80 | i);
        recs[i].val = 0xC0DE0000u + (uint32_t)i;
    }
    int32_t s = 0;
    for (int i = 0; i < 64; i++) s += sbytes[i] * (i & 3 ? 1 : -1);
    for (int i = 0; i < 97; i++) s ^= (int16_t)halves[i];
    for (int i = 0; i < 300; i++) hist[buf[i] & 15]++;
    uint32_t c = qd_crc32(buf, sizeof buf, 0);
    c = qd_crc32(halves, sizeof halves, c);
    c = qd_crc32(recs, sizeof recs, c);
    c = qd_crc32(hist, sizeof hist, c);
    qd_puts("bytes ");
    qd_puthex(c);
    qd_putchar(' ');
    qd_putdec(s);
    qd_putchar('\n');
    return 0;
}
