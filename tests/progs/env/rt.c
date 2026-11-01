// Runtime for freestanding test programs: printing, crc32, the memory functions clang may call,
// and the 64-bit division libcalls (rv32im has no 64-bit divide; written with 32-bit operations
// and shifts only, so they never call themselves).
#include "rt.h"

void qd_puts(const char *s) {
    while (*s) qd_putchar(*s++);
}

void qd_puthex(uint32_t v) {
    for (int i = 7; i >= 0; i--) qd_putchar("0123456789abcdef"[(v >> (4 * i)) & 15]);
}

void qd_putudec64(uint64_t v) {
    char buf[24];
    int n = 0;
    do {
        buf[n++] = '0' + (char)(v % 10);  // __umoddi3
        v /= 10;                          // __udivdi3
    } while (v);
    while (n) qd_putchar(buf[--n]);
}

void qd_putdec(int32_t v) {
    if (v < 0) {
        qd_putchar('-');
        qd_putudec64((uint64_t)(-(int64_t)v));
    } else {
        qd_putudec64((uint64_t)v);
    }
}

uint32_t qd_crc32(const void *p, size_t n, uint32_t crc) {
    const uint8_t *b = p;
    crc = ~crc;
    while (n--) {
        crc ^= *b++;
        for (int k = 0; k < 8; k++) crc = (crc >> 1) ^ (0xEDB88320u & -(crc & 1));
    }
    return ~crc;
}

void *memcpy(void *d, const void *s, size_t n) {
    uint8_t *dd = d;
    const uint8_t *ss = s;
    if ((((uintptr_t)dd | (uintptr_t)ss | n) & 3) == 0) {
        for (; n; n -= 4, dd += 4, ss += 4) *(uint32_t *)dd = *(const uint32_t *)ss;
        return d;
    }
    while (n--) *dd++ = *ss++;
    return d;
}

void *memmove(void *d, const void *s, size_t n) {
    uint8_t *dd = d;
    const uint8_t *ss = s;
    if (dd < ss) {
        while (n--) *dd++ = *ss++;
    } else {
        while (n--) dd[n] = ss[n];
    }
    return d;
}

void *memset(void *d, int c, size_t n) {
    uint8_t *dd = d;
    while (n--) *dd++ = (uint8_t)c;
    return d;
}

int memcmp(const void *a, const void *b, size_t n) {
    const uint8_t *x = a, *y = b;
    for (; n; n--, x++, y++)
        if (*x != *y) return *x - *y;
    return 0;
}

// unsigned 64-bit long division, one quotient bit per step (restoring), from n's top bit
static uint64_t udivmod64(uint64_t n, uint64_t d, uint64_t *rem) {
    if (d == 0) {  // RISC-V's DIV by zero: all ones (and the dividend as the remainder)
        if (rem) *rem = n;
        return ~(uint64_t)0;
    }
    uint64_t q = 0, r = 0;
    int top = 63;  // from n's top set bit (the quotient bits above it are 0)
    while (top > 0 && !(n >> top)) top--;
    for (int i = top; i >= 0; i--) {
        r = (r << 1) | ((n >> i) & 1);
        if (r >= d) {
            r -= d;
            q |= (uint64_t)1 << i;
        }
    }
    if (rem) *rem = r;
    return q;
}

uint64_t __udivdi3(uint64_t n, uint64_t d) { return udivmod64(n, d, 0); }

uint64_t __umoddi3(uint64_t n, uint64_t d) {
    uint64_t r;
    udivmod64(n, d, &r);
    return r;
}

int64_t __divdi3(int64_t n, int64_t d) {
    int neg = (n < 0) != (d < 0);
    uint64_t q = udivmod64(n < 0 ? -(uint64_t)n : (uint64_t)n,
                           d < 0 ? -(uint64_t)d : (uint64_t)d, 0);
    return neg ? -(int64_t)q : (int64_t)q;
}

int64_t __moddi3(int64_t n, int64_t d) {
    uint64_t r;
    udivmod64(n < 0 ? -(uint64_t)n : (uint64_t)n, d < 0 ? -(uint64_t)d : (uint64_t)d, &r);
    return n < 0 ? -(int64_t)r : (int64_t)r;
}
