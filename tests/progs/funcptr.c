// Function pointers: tables in .data (RAM) and .rodata (DROM), callbacks, a comparator-driven
// insertion sort, and returning functions.
#include "env/rt.h"

typedef int32_t (*binop)(int32_t, int32_t);
static int32_t add(int32_t a, int32_t b) { return a + b; }
static int32_t sub(int32_t a, int32_t b) { return a - b; }
static int32_t mul(int32_t a, int32_t b) { return a * b; }
static int32_t dvd(int32_t a, int32_t b) { return b ? a / b : 0; }
static int32_t mod(int32_t a, int32_t b) { return b ? a % b : 0; }
static const binop rom_ops[] = {add, sub, mul, dvd, mod};
static binop ram_ops[] = {mod, dvd, mul, sub, add};

static int cmp_up(const void *a, const void *b) { return *(const int32_t *)a - *(const int32_t *)b; }
static int cmp_down(const void *a, const void *b) { return cmp_up(b, a); }

static void isort(int32_t *v, int n, int (*cmp)(const void *, const void *)) {
    for (int i = 1; i < n; i++) {
        int32_t x = v[i];
        int j = i - 1;
        while (j >= 0 && cmp(&v[j], &x) > 0) {
            v[j + 1] = v[j];
            j--;
        }
        v[j + 1] = x;
    }
}

static binop pick(int k) { return (k & 1) ? rom_ops[k % 5] : ram_ops[k % 5]; }

int main(void) {
    int32_t acc = 17, v[24];
    for (int k = 0; k < 40; k++) acc = pick(k)(acc, k * 3 - 11) + 1;
    for (int i = 0; i < 24; i++) v[i] = (i * 7919) % 101 - 50;
    isort(v, 24, (acc & 1) ? cmp_up : cmp_down);
    uint32_t c = qd_crc32(v, sizeof v, 0);
    ram_ops[0] = rom_ops[2];
    acc += ram_ops[0](6, 7);
    qd_puts("funcptr ");
    qd_putdec(acc);
    qd_putchar(' ');
    qd_puthex(c);
    qd_putchar('\n');
    return 0;
}
