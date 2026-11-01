// 64-bit multiply, divide and remainder (the libcalls in env/rt.c), signed and unsigned.
#include "env/rt.h"

static uint64_t ops[] = {  // in RAM, read through volatile: nothing folds at compile time
    0, 3, 10, 0xFFFFFFFFull, 0x123456789ABCDEFull, 0x8000000000000000ull,
    0xFFFFFFFFFFFFFFFFull, 0xDEADBEEFCAFEBABEull,
};
#define N (sizeof ops / sizeof ops[0])

int main(void) {
    uint64_t acc = 0;
    for (unsigned i = 0; i < N; i++) {
        for (unsigned j = 0; j < N; j++) {
            uint64_t a = ((volatile uint64_t *)ops)[i], b = ((volatile uint64_t *)ops)[j];
            acc = acc * 0x100000001B3ull + a * b;  // 64-bit multiply (mul / mulhu inline)
            if (b) {
                acc ^= a / b;
                acc += a % b;
                int64_t sa = (int64_t)a, sb = (int64_t)b;
                if (!(sa == (int64_t)0x8000000000000000ull && sb == -1)) {
                    acc ^= (uint64_t)(sa / sb);
                    acc -= (uint64_t)(sa % sb);
                }
            }
        }
    }
    qd_puts("acc ");
    qd_putudec64(acc);
    qd_puts(" q ");
    qd_putudec64(((volatile uint64_t *)ops)[6] / 1000000007ull);
    qd_puts(" r ");
    qd_putdec((int32_t)((int64_t)-123456789012345ll % 1000003));
    qd_putchar('\n');
    return 0;
}
