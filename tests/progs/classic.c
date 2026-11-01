// The small classics: crc32 of a string, a sieve, a shell sort, string reversal and atoi.
#include "env/rt.h"

static uint8_t sieve[2000];
static int32_t arr[200];

int main(void) {
    const char *msg = "The quick brown fox jumps over the lazy dog";
    uint32_t n = 0;
    while (msg[n]) n++;
    qd_puts("crc32=");
    qd_puthex(qd_crc32(msg, n, 0));
    int primes = 0;
    for (int i = 2; i < 2000; i++) {
        if (sieve[i]) continue;
        primes++;
        for (int j = i * i; j < 2000; j += i) sieve[j] = 1;
    }
    qd_puts(" primes=");
    qd_putdec(primes);
    uint32_t x = 12345;
    for (int i = 0; i < 200; i++) {
        x = x * 1103515245u + 12345u;
        arr[i] = (int32_t)(x >> 8) % 10000 - 5000;
    }
    for (int gap = 100; gap > 0; gap /= 2)
        for (int i = gap; i < 200; i++)
            for (int j = i; j >= gap && arr[j - gap] > arr[j]; j -= gap) {
                int32_t t = arr[j];
                arr[j] = arr[j - gap];
                arr[j - gap] = t;
            }
    int sorted = 1;
    for (int i = 1; i < 200; i++) sorted &= arr[i - 1] <= arr[i];
    qd_puts(" sorted=");
    qd_putdec(sorted);
    qd_puts(" min=");
    qd_putdec(arr[0]);
    char rev[64];
    for (uint32_t i = 0; i < n && i < 63; i++) rev[i] = msg[n - 1 - i];
    rev[n < 63 ? n : 63] = 0;
    qd_puts(" rev=");
    qd_puts(rev + 30);
    int32_t v = 0;
    for (const char *p = "-98765"; *p; p++) v = *p == '-' ? v : v * 10 + (*p - '0');
    qd_puts(" atoi=");
    qd_putdec(-v);
    qd_putchar('\n');
    return 0;
}
