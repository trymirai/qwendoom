// The qwendoom machine's system interface for freestanding test programs (no libc): putchar
// and YIELD are ecall a7 = 1, exit is a7 = 93, the input port is csrr rd, time.
#pragma once
#include <stddef.h>
#include <stdint.h>

static inline void qd_putchar(int c) {
    register long a0 __asm__("a0") = (unsigned char)c;
    register long a7 __asm__("a7") = 1;
    __asm__ volatile("ecall" : "+r"(a0) : "r"(a7) : "memory");
}

static inline void qd_yield(void) {  // BYTE(256): the tick is done
    register long a0 __asm__("a0") = 256;
    register long a7 __asm__("a7") = 1;
    __asm__ volatile("ecall" : "+r"(a0) : "r"(a7) : "memory");
}

static inline __attribute__((noreturn)) void qd_exit(int code) {
    register long a0 __asm__("a0") = code;
    register long a7 __asm__("a7") = 93;
    __asm__ volatile("ecall" : : "r"(a0), "r"(a7) : "memory");
    __builtin_unreachable();
}

static inline uint32_t qd_in(void) {  // the next input word (host-injected)
    uint32_t v;
    __asm__ volatile("csrr %0, time" : "=r"(v) : : "memory");
    return v;
}

void *memcpy(void *d, const void *s, size_t n);
void *memmove(void *d, const void *s, size_t n);
void *memset(void *d, int c, size_t n);
int memcmp(const void *a, const void *b, size_t n);
void qd_puts(const char *s);
void qd_puthex(uint32_t v);
void qd_putdec(int32_t v);
void qd_putudec64(uint64_t v);
uint32_t qd_crc32(const void *p, size_t n, uint32_t crc);
