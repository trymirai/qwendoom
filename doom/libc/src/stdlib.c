/* qwendoom's freestanding libc: the host heap (a K&R first-fit free list over the HOST region
   between _end and __heap_end), number conversions, exit. Host-only code. */
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <stdint.h>
#include "qd_sys.h"

int errno;

extern char _end[], __heap_end[];
static char *brk_ptr;

typedef union header {
    struct {
        union header *next;  /* the next free block (free list, address order) */
        size_t units;        /* the block size in header units, header included */
    } s;
    uint64_t align;
} header;

static header base, *freep;

static header *more_core(size_t units)
{
    if (!brk_ptr)
        brk_ptr = (char *)(((uintptr_t)_end + 15) & ~(uintptr_t)15);
    if (units < 4096)
        units = 4096;
    size_t bytes = units * sizeof(header);
    if ((size_t)(__heap_end - brk_ptr) < bytes)
        return NULL;
    header *h = (header *)brk_ptr;
    brk_ptr += bytes;
    h->s.units = units;
    free(h + 1);
    return freep;
}

void *malloc(size_t n)
{
    size_t units = (n + sizeof(header) - 1) / sizeof(header) + 1;
    header *prev = freep, *p;
    if (!prev) {
        base.s.next = freep = prev = &base;
        base.s.units = 0;
    }
    for (p = prev->s.next;; prev = p, p = p->s.next) {
        if (p->s.units >= units) {
            if (p->s.units == units) {
                prev->s.next = p->s.next;
            } else {
                p->s.units -= units;
                p += p->s.units;
                p->s.units = units;
            }
            freep = prev;
            return p + 1;
        }
        if (p == freep && !(p = more_core(units))) {
            errno = ENOMEM;
            return NULL;
        }
    }
}

void free(void *ptr)
{
    if (!ptr)
        return;
    header *b = (header *)ptr - 1, *p;
    for (p = freep; !(b > p && b < p->s.next); p = p->s.next)
        if (p >= p->s.next && (b > p || b < p->s.next))
            break;
    if (b + b->s.units == p->s.next) {
        b->s.units += p->s.next->s.units;
        b->s.next = p->s.next->s.next;
    } else {
        b->s.next = p->s.next;
    }
    if (p + p->s.units == b) {
        p->s.units += b->s.units;
        p->s.next = b->s.next;
    } else {
        p->s.next = b;
    }
    freep = p;
}

void *calloc(size_t n, size_t size)
{
    size_t t = n * size;
    void *p = malloc(t);
    return p ? memset(p, 0, t) : NULL;
}

void *realloc(void *ptr, size_t n)
{
    if (!ptr)
        return malloc(n);
    size_t have = (((header *)ptr - 1)->s.units - 1) * sizeof(header);
    if (n <= have)
        return ptr;
    void *q = malloc(n);
    if (q) {
        memcpy(q, ptr, have);
        free(ptr);
    }
    return q;
}

unsigned long strtoul(const char *s, char **end, int base)
{
    const char *p = s;
    unsigned long v = 0;
    int neg = 0, any = 0;
    while (isspace((unsigned char)*p))
        p++;
    if (*p == '+' || *p == '-')
        neg = *p++ == '-';
    if ((base == 0 || base == 16) && p[0] == '0' && (p[1] | 32) == 'x' && isxdigit((unsigned char)p[2])) {
        p += 2;
        base = 16;
    } else if (base == 0) {
        base = *p == '0' ? 8 : 10;
    }
    for (;; p++, any = 1) {
        int c = (unsigned char)*p, d;
        if (isdigit(c))
            d = c - '0';
        else if (isalpha(c))
            d = (c | 32) - 'a' + 10;
        else
            break;
        if (d >= base)
            break;
        v = v * base + d;
    }
    if (end)
        *end = (char *)(any ? p : s);
    return neg ? -v : v;
}

long strtol(const char *s, char **end, int base)
{
    return (long)strtoul(s, end, base);
}

int atoi(const char *s) { return (int)strtol(s, NULL, 10); }
long atol(const char *s) { return strtol(s, NULL, 10); }

int (abs)(int x) { return x < 0 ? -x : x; }
long labs(long x) { return x < 0 ? -x : x; }

char *getenv(const char *name)
{
    (void)name;
    return NULL;
}

int system(const char *cmd)
{
    (void)cmd;
    return -1;
}

int atexit(void (*f)(void))
{
    (void)f;
    return 0;
}

void exit(int code)
{
    for (;;)
        qd_ecall(QD_EC_EXIT, code, 0, 0);
}

void abort(void)
{
    exit(134);
}

void __qd_assert_fail(const char *expr, const char *file, int line)
{
    extern int printf(const char *, ...);
    printf("assertion failed: %s (%s:%d)\n", expr, file, line);
    exit(134);
}

static void swap_bytes(char *a, char *b, size_t n)
{
    while (n--) {
        char t = *a;
        *a++ = *b;
        *b++ = t;
    }
}

void qsort(void *base, size_t n, size_t size, int (*cmp)(const void *, const void *))
{
    /* insertion sort: DOOM sorts only short lists */
    char *b = base;
    for (size_t i = 1; i < n; i++)
        for (size_t j = i; j > 0 && cmp(b + (j - 1) * size, b + j * size) > 0; j--)
            swap_bytes(b + (j - 1) * size, b + j * size, size);
}

static unsigned long rand_state = 1;
int rand(void)
{
    rand_state = rand_state * 1103515245 + 12345;
    return (int)(rand_state >> 1) & RAND_MAX;
}
void srand(unsigned seed) { rand_state = seed; }
