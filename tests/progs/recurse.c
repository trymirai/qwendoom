// Deep recursion on a stack at the RAM's top (sp above 16 KB: 2-token value records), mutual
// recursion and Ackermann.
#include "env/rt.h"

__asm__(".globl __stack_top\n.set __stack_top, 0x40000");

static int depth_max;

__attribute__((noinline)) static uint32_t deep(uint32_t n, uint32_t acc) {
    volatile uint32_t frame[4] = {n, acc, n ^ acc, n * 3};  // a bigger frame, stored and reloaded
    if (n == 0) return acc;
    if ((int)n > depth_max) depth_max = (int)n;
    uint32_t r = deep(n - 1, acc * 33 + frame[2]);
    return r + frame[0] - frame[3] / 3;
}

__attribute__((noinline)) static int is_odd(uint32_t n);
__attribute__((noinline)) static int is_even(uint32_t n) { return n == 0 ? 1 : is_odd(n - 1); }
__attribute__((noinline)) static int is_odd(uint32_t n) { return n == 0 ? 0 : is_even(n - 1); }

__attribute__((noinline)) static uint32_t ack(uint32_t m, uint32_t n) {
    if (m == 0) return n + 1;
    if (n == 0) return ack(m - 1, 1);
    return ack(m - 1, ack(m, n - 1));
}

int main(void) {
    uint32_t r = deep(1200, 1);
    qd_puts("deep ");
    qd_puthex(r);
    qd_puts(" even(301) ");
    qd_putdec(is_even(301));
    qd_puts(" ack(2,3) ");
    qd_putdec((int32_t)ack(2, 3));
    qd_putchar('\n');
    return depth_max == 1200 ? 0 : 1;
}
