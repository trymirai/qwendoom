/* qwendoom's freestanding libc: string functions. memcpy / memset / memmove also run in the sim
   (struct copies), where a misaligned word access is a fault: words only when both sides are
   aligned. */
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

void *memcpy(void *restrict d, const void *restrict s, size_t n)
{
    unsigned char *dp = d;
    const unsigned char *sp = s;
    if ((((uintptr_t)dp | (uintptr_t)sp) & 3) == 0) {
        uint32_t *dw = (uint32_t *)dp;
        const uint32_t *sw = (const uint32_t *)sp;
        for (; n >= 16; n -= 16, dw += 4, sw += 4) {
            uint32_t a = sw[0], b = sw[1], c = sw[2], e = sw[3];
            dw[0] = a; dw[1] = b; dw[2] = c; dw[3] = e;
        }
        for (; n >= 4; n -= 4)
            *dw++ = *sw++;
        dp = (unsigned char *)dw;
        sp = (const unsigned char *)sw;
    }
    while (n--)
        *dp++ = *sp++;
    return d;
}

void *memmove(void *d, const void *s, size_t n)
{
    unsigned char *dp = d;
    const unsigned char *sp = s;
    if (dp == sp || n == 0)
        return d;
    if (dp < sp || dp >= sp + n)
        return memcpy(d, s, n);
    dp += n;
    sp += n;
    if ((((uintptr_t)dp | (uintptr_t)sp | n) & 3) == 0) {
        for (; n; n -= 4) {
            dp -= 4; sp -= 4;
            *(uint32_t *)dp = *(const uint32_t *)sp;
        }
        return d;
    }
    while (n--)
        *--dp = *--sp;
    return d;
}

void *memset(void *d, int c, size_t n)
{
    unsigned char *dp = d;
    uint32_t w = (unsigned char)c * 0x01010101u;
    while (n && ((uintptr_t)dp & 3)) {
        *dp++ = (unsigned char)c;
        n--;
    }
    uint32_t *dw = (uint32_t *)dp;
    for (; n >= 16; n -= 16, dw += 4) {
        dw[0] = w; dw[1] = w; dw[2] = w; dw[3] = w;
    }
    for (; n >= 4; n -= 4)
        *dw++ = w;
    dp = (unsigned char *)dw;
    while (n--)
        *dp++ = (unsigned char)c;
    return d;
}

int memcmp(const void *a, const void *b, size_t n)
{
    const unsigned char *x = a, *y = b;
    for (; n; n--, x++, y++)
        if (*x != *y)
            return *x - *y;
    return 0;
}

void *memchr(const void *s, int c, size_t n)
{
    const unsigned char *p = s;
    for (; n; n--, p++)
        if (*p == (unsigned char)c)
            return (void *)p;
    return NULL;
}

size_t strlen(const char *s)
{
    const char *p = s;
    while (*p)
        p++;
    return p - s;
}

size_t strnlen(const char *s, size_t n)
{
    size_t i = 0;
    while (i < n && s[i])
        i++;
    return i;
}

char *strcpy(char *d, const char *s)
{
    char *r = d;
    while ((*d++ = *s++))
        ;
    return r;
}

char *strncpy(char *d, const char *s, size_t n)
{
    size_t i = 0;
    for (; i < n && s[i]; i++)
        d[i] = s[i];
    for (; i < n; i++)
        d[i] = 0;
    return d;
}

char *strcat(char *d, const char *s)
{
    strcpy(d + strlen(d), s);
    return d;
}

char *strncat(char *d, const char *s, size_t n)
{
    char *p = d + strlen(d);
    while (n-- && *s)
        *p++ = *s++;
    *p = 0;
    return d;
}

int strcmp(const char *a, const char *b)
{
    while (*a && *a == *b)
        a++, b++;
    return (unsigned char)*a - (unsigned char)*b;
}

int strncmp(const char *a, const char *b, size_t n)
{
    for (; n; n--, a++, b++) {
        if (*a != *b)
            return (unsigned char)*a - (unsigned char)*b;
        if (!*a)
            return 0;
    }
    return 0;
}

static int lower(int c) { return (unsigned)c - 'A' < 26 ? c + 32 : c; }

int strcasecmp(const char *a, const char *b)
{
    while (*a && lower(*a) == lower(*b))
        a++, b++;
    return lower((unsigned char)*a) - lower((unsigned char)*b);
}

int strncasecmp(const char *a, const char *b, size_t n)
{
    for (; n; n--, a++, b++) {
        int x = lower((unsigned char)*a), y = lower((unsigned char)*b);
        if (x != y)
            return x - y;
        if (!x)
            return 0;
    }
    return 0;
}

char *strchr(const char *s, int c)
{
    for (;; s++) {
        if (*s == (char)c)
            return (char *)s;
        if (!*s)
            return NULL;
    }
}

char *strrchr(const char *s, int c)
{
    const char *r = NULL;
    for (;; s++) {
        if (*s == (char)c)
            r = s;
        if (!*s)
            return (char *)r;
    }
}

char *strstr(const char *h, const char *n)
{
    size_t k = strlen(n);
    for (; *h; h++)
        if (!strncmp(h, n, k))
            return (char *)h;
    return k ? NULL : (char *)h;
}

char *strdup(const char *s)
{
    size_t n = strlen(s) + 1;
    char *d = malloc(n);
    return d ? memcpy(d, s, n) : NULL;
}

char *strndup(const char *s, size_t n)
{
    size_t k = strnlen(s, n);
    char *d = malloc(k + 1);
    if (!d)
        return NULL;
    memcpy(d, s, k);
    d[k] = 0;
    return d;
}

size_t strspn(const char *s, const char *accept)
{
    size_t i = 0;
    while (s[i] && strchr(accept, s[i]))
        i++;
    return i;
}

size_t strcspn(const char *s, const char *reject)
{
    size_t i = 0;
    while (s[i] && !strchr(reject, s[i]))
        i++;
    return i;
}

char *strpbrk(const char *s, const char *accept)
{
    s += strcspn(s, accept);
    return *s ? (char *)s : NULL;
}

char *strtok(char *s, const char *delim)
{
    static char *next;
    if (!s)
        s = next;
    if (!s)
        return NULL;
    s += strspn(s, delim);
    if (!*s) {
        next = NULL;
        return NULL;
    }
    char *e = s + strcspn(s, delim);
    if (*e)
        *e++ = 0;
    else
        e = NULL;
    next = e;
    return s;
}

char *strerror(int e)
{
    (void)e;
    return "error";
}
