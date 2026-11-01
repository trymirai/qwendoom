/* qwendoom's freestanding libc: stdio. Files are host file descriptors (QD_EC_OPEN and friends,
   which the host restricts to the WAD / config directory); stdout and stderr go to the host's
   output. A small read buffer serves fgetc / fscanf; fread of large blocks (WAD lumps) reads
   straight into the destination. Host-only code. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>
#include "qd_sys.h"

#define F_READ 1
#define F_WRITE 2
#define F_EOF 4
#define F_ERR 8
#define RBUF 512

struct _qd_file {
    int fd;
    int flags;
    int rpos, rlen;  /* the read buffer's unread bytes are rbuf[rpos..rlen) */
    unsigned char rbuf[RBUF];
};

static FILE std_files[3] = {{0, F_READ}, {1, F_WRITE}, {2, F_WRITE}};
FILE *stdin = &std_files[0], *stdout = &std_files[1], *stderr = &std_files[2];

FILE *fopen(const char *path, const char *mode)
{
    int w = strchr(mode, 'w') || strchr(mode, 'a');
    if (strchr(mode, '+'))
        return NULL;
    int fd = qd_ecall(QD_EC_OPEN, (long)path, w ? QD_O_WRITE : QD_O_READ, 0);
    if (fd < 0)
        return NULL;
    FILE *f = calloc(1, sizeof(FILE));
    if (!f) {
        qd_ecall(QD_EC_CLOSE, fd, 0, 0);
        return NULL;
    }
    f->fd = fd;
    f->flags = w ? F_WRITE : F_READ;
    return f;
}

int fclose(FILE *f)
{
    if (!f)
        return EOF;
    int r = qd_ecall(QD_EC_CLOSE, f->fd, 0, 0);
    if (f->fd > 2)
        free(f);
    return r ? EOF : 0;
}

static int fill(FILE *f)
{
    long n = qd_ecall(QD_EC_READ, f->fd, (long)f->rbuf, RBUF);
    if (n <= 0) {
        f->flags |= n < 0 ? F_ERR : F_EOF;
        return 0;
    }
    f->rpos = 0;
    f->rlen = n;
    return 1;
}

size_t fread(void *buf, size_t size, size_t n, FILE *f)
{
    size_t want = size * n, got = 0;
    unsigned char *p = buf;
    if (!want || !(f->flags & F_READ))
        return 0;
    if (f->rpos < f->rlen) {
        size_t k = f->rlen - f->rpos;
        if (k > want)
            k = want;
        memcpy(p, f->rbuf + f->rpos, k);
        f->rpos += k;
        got = k;
    }
    while (got < want) {
        long r = qd_ecall(QD_EC_READ, f->fd, (long)(p + got), want - got);
        if (r <= 0) {
            f->flags |= r < 0 ? F_ERR : F_EOF;
            break;
        }
        got += r;
    }
    return got / size;
}

static long raw_write(FILE *f, const void *buf, size_t n)
{
    if (f->fd == 1 || f->fd == 2)
        return qd_ecall(QD_EC_WRITE, f->fd, (long)buf, n);
    return qd_ecall(QD_EC_FWRITE, f->fd, (long)buf, n);
}

size_t fwrite(const void *buf, size_t size, size_t n, FILE *f)
{
    if (!size || !n || !(f->flags & F_WRITE))
        return 0;
    long r = raw_write(f, buf, size * n);
    if (r < 0) {
        f->flags |= F_ERR;
        return 0;
    }
    return r / size;
}

int fseek(FILE *f, long off, int whence)
{
    if (whence == SEEK_CUR)
        off -= f->rlen - f->rpos;
    f->rpos = f->rlen = 0;
    long r = qd_ecall(QD_EC_SEEK, f->fd, off, whence);
    if (r < 0)
        return -1;
    f->flags &= ~F_EOF;
    return 0;
}

long ftell(FILE *f)
{
    long r = qd_ecall(QD_EC_SEEK, f->fd, 0, SEEK_CUR);
    return r < 0 ? -1 : r - (f->rlen - f->rpos);
}

void rewind(FILE *f) { fseek(f, 0, SEEK_SET); f->flags &= ~F_ERR; }
int fflush(FILE *f) { (void)f; return 0; }
int feof(FILE *f) { return (f->flags & F_EOF) && f->rpos >= f->rlen; }
int ferror(FILE *f) { return (f->flags & F_ERR) != 0; }
void clearerr(FILE *f) { f->flags &= ~(F_EOF | F_ERR); }
int setvbuf(FILE *f, char *buf, int mode, size_t size) { (void)f; (void)buf; (void)mode; (void)size; return 0; }

int fgetc(FILE *f)
{
    if (!(f->flags & F_READ))
        return EOF;
    if (f->rpos >= f->rlen && !fill(f))
        return EOF;
    return f->rbuf[f->rpos++];
}

int getc(FILE *f) { return fgetc(f); }

int ungetc(int c, FILE *f)
{
    if (c == EOF || f->rpos == 0)
        return EOF;
    f->rbuf[--f->rpos] = (unsigned char)c;
    f->flags &= ~F_EOF;
    return c;
}

char *fgets(char *s, int n, FILE *f)
{
    int i = 0;
    while (i < n - 1) {
        int c = fgetc(f);
        if (c == EOF)
            break;
        s[i++] = (char)c;
        if (c == '\n')
            break;
    }
    if (!i)
        return NULL;
    s[i] = 0;
    return s;
}

int fputc(int c, FILE *f)
{
    unsigned char b = (unsigned char)c;
    return fwrite(&b, 1, 1, f) == 1 ? b : EOF;
}

int putc(int c, FILE *f) { return fputc(c, f); }

int fputs(const char *s, FILE *f)
{
    size_t n = strlen(s);
    return fwrite(s, 1, n, f) == n ? 0 : EOF;
}

int putchar(int c) { return fputc(c, stdout); }

int puts(const char *s)
{
    if (fputs(s, stdout))
        return EOF;
    return fputc('\n', stdout) == EOF ? EOF : 0;
}

int remove(const char *path) { return qd_ecall(QD_EC_REMOVE, (long)path, 0, 0) ? -1 : 0; }
int rename(const char *a, const char *b) { return qd_ecall(QD_EC_RENAME, (long)a, (long)b, 0) ? -1 : 0; }
int unlink(const char *path) { return remove(path); }
int mkdir(const char *path, unsigned mode) { (void)mode; return qd_ecall(QD_EC_MKDIR, (long)path, 0, 0) ? -1 : 0; }

int access(const char *path, int mode)
{
    (void)mode;
    FILE *f = fopen(path, "rb");
    if (!f)
        return -1;
    fclose(f);
    return 0;
}

/* --- printf ------------------------------------------------------------------------------- */

typedef struct {
    char *buf;
    size_t cap, len;  /* len counts every character, cap - 1 are stored */
} out_t;

static void out_c(out_t *o, char c)
{
    if (o->len + 1 < o->cap)
        o->buf[o->len] = c;
    o->len++;
}

static void out_pad(out_t *o, char c, int n)
{
    while (n-- > 0)
        out_c(o, c);
}

/* an unsigned number with the C flags: minus, zero, plus / space prefix, width, precision */
static void out_num(out_t *o, uint64_t v, int base, int upper, const char *prefix, int minus,
                    int zero, int width, int prec)
{
    char tmp[24];
    const char *digits = upper ? "0123456789ABCDEF" : "0123456789abcdef";
    int n = 0;
    if (v == 0 && prec != 0)
        tmp[n++] = '0';
    while (v) {
        tmp[n++] = digits[v % base];
        v /= base;
    }
    int plen = strlen(prefix), zeros = prec > n ? prec - n : 0;
    int total = plen + zeros + n;
    if (zero && !minus && prec < 0)
        zeros += width > total ? width - total : 0, total = plen + zeros + n;
    if (!minus)
        out_pad(o, ' ', width - total);
    while (*prefix)
        out_c(o, *prefix++);
    out_pad(o, '0', zeros);
    while (n)
        out_c(o, tmp[--n]);
    if (minus)
        out_pad(o, ' ', width - total);
}

/* %f without an FPU: the double's bits, integer arithmetic (enough for config values) */
static void out_double(out_t *o, uint64_t bits, int prec, int minus, int width)
{
    char tmp[48];
    int n = 0, neg = bits >> 63, e = (int)(bits >> 52 & 0x7FF) - 1075;
    uint64_t m = bits & ((1ull << 52) - 1), ip, fp = 0;
    int fbits = 0;
    if (prec < 0)
        prec = 6;
    if (prec > 9)
        prec = 9;
    if (bits << 1 >> 53 != 0)
        m |= 1ull << 52;
    else
        e++;
    if (e >= 0) {
        ip = e < 11 ? m << e : ~0ull;
    } else if (e > -64) {
        ip = m >> -e;
        fbits = -e;
        fp = m & ((1ull << fbits) - 1);
    } else {
        ip = 0;
    }
    /* fraction digits: fp / 2^fbits, scaled by 10 per digit (fbits <= 60 keeps fp * 10 in range) */
    while (fbits > 60) {
        fp >>= 1;
        fbits--;
    }
    uint64_t scale = 1;
    for (int i = 0; i < prec; i++)
        scale *= 10;
    uint64_t frac = 0;
    for (int i = 0; i < prec; i++) {
        fp *= 10;
        frac = frac * 10 + (fbits ? fp >> fbits : 0);
        fp = fbits ? fp & ((1ull << fbits) - 1) : 0;
    }
    if (fbits && fp >= 1ull << (fbits - 1) && ++frac == scale) {
        frac = 0;
        ip++;
    }
    for (int i = 0; i < prec; i++, frac /= 10)
        tmp[n++] = '0' + frac % 10;
    if (prec)
        tmp[n++] = '.';
    do {
        tmp[n++] = '0' + ip % 10;
        ip /= 10;
    } while (ip);
    if (neg)
        tmp[n++] = '-';
    if (!minus)
        out_pad(o, ' ', width - n);
    int k = n;
    while (n)
        out_c(o, tmp[--n]);
    if (minus)
        out_pad(o, ' ', width - k);
}

int vsnprintf(char *buf, size_t cap, const char *fmt, va_list ap)
{
    out_t o = {buf, cap, 0};
    for (; *fmt; fmt++) {
        if (*fmt != '%') {
            out_c(&o, *fmt);
            continue;
        }
        int minus = 0, zero = 0, plus = 0, space = 0, alt = 0, width = 0, prec = -1, lng = 0;
        for (;; fmt++) {
            char c = fmt[1];
            if (c == '-') minus = 1;
            else if (c == '0') zero = 1;
            else if (c == '+') plus = 1;
            else if (c == ' ') space = 1;
            else if (c == '#') alt = 1;
            else break;
        }
        fmt++;
        if (*fmt == '*') {
            width = va_arg(ap, int);
            if (width < 0)
                minus = 1, width = -width;
            fmt++;
        } else {
            while (isdigit((unsigned char)*fmt))
                width = width * 10 + *fmt++ - '0';
        }
        if (*fmt == '.') {
            fmt++;
            prec = 0;
            if (*fmt == '*') {
                prec = va_arg(ap, int);
                fmt++;
            } else {
                while (isdigit((unsigned char)*fmt))
                    prec = prec * 10 + *fmt++ - '0';
            }
        }
        while (*fmt == 'l' || *fmt == 'h' || *fmt == 'z' || *fmt == 'j' || *fmt == 't') {
            if (*fmt == 'l')
                lng++;
            fmt++;
        }
        char c = *fmt;
        if (!c)
            break;
        switch (c) {
        case 'd':
        case 'i': {
            int64_t v = lng >= 2 ? va_arg(ap, int64_t) : va_arg(ap, int);
            const char *pre = v < 0 ? "-" : plus ? "+" : space ? " " : "";
            out_num(&o, v < 0 ? -(uint64_t)v : (uint64_t)v, 10, 0, pre, minus, zero, width, prec);
            break;
        }
        case 'u':
        case 'x':
        case 'X':
        case 'o': {
            uint64_t v = lng >= 2 ? va_arg(ap, uint64_t) : va_arg(ap, unsigned);
            int base = c == 'u' ? 10 : c == 'o' ? 8 : 16;
            const char *pre = alt && v ? (c == 'o' ? "0" : c == 'x' ? "0x" : c == 'X' ? "0X" : "") : "";
            out_num(&o, v, base, c == 'X', pre, minus, zero, width, prec);
            break;
        }
        case 'p':
            out_num(&o, (uintptr_t)va_arg(ap, void *), 16, 0, "0x", minus, zero, width, prec);
            break;
        case 'c':
            if (!minus)
                out_pad(&o, ' ', width - 1);
            out_c(&o, (char)va_arg(ap, int));
            if (minus)
                out_pad(&o, ' ', width - 1);
            break;
        case 's': {
            const char *s = va_arg(ap, const char *);
            if (!s)
                s = "(null)";
            int n = prec >= 0 ? (int)strnlen(s, prec) : (int)strlen(s);
            if (!minus)
                out_pad(&o, ' ', width - n);
            for (int i = 0; i < n; i++)
                out_c(&o, s[i]);
            if (minus)
                out_pad(&o, ' ', width - n);
            break;
        }
        case 'f':
        case 'F':
        case 'g':
        case 'e': {
            union { double d; uint64_t u; } d = {.d = va_arg(ap, double)};
            out_double(&o, d.u, prec, minus, width);
            break;
        }
        case '%':
            out_c(&o, '%');
            break;
        default:
            out_c(&o, '%');
            out_c(&o, c);
        }
    }
    if (cap)
        buf[o.len < cap ? o.len : cap - 1] = 0;
    return (int)o.len;
}

int vsprintf(char *buf, const char *fmt, va_list ap) { return vsnprintf(buf, 1 << 30, fmt, ap); }

int snprintf(char *buf, size_t n, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    int r = vsnprintf(buf, n, fmt, ap);
    va_end(ap);
    return r;
}

int sprintf(char *buf, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    int r = vsprintf(buf, fmt, ap);
    va_end(ap);
    return r;
}

int vfprintf(FILE *f, const char *fmt, va_list ap)
{
    char small[512];
    va_list aq;
    va_copy(aq, ap);
    int n = vsnprintf(small, sizeof(small), fmt, ap);
    if (n < (int)sizeof(small)) {
        fwrite(small, 1, n, f);
    } else {
        char *big = malloc(n + 1);
        if (big) {
            vsnprintf(big, n + 1, fmt, aq);
            fwrite(big, 1, n, f);
            free(big);
        }
    }
    va_end(aq);
    return n;
}

int vprintf(const char *fmt, va_list ap) { return vfprintf(stdout, fmt, ap); }

int fprintf(FILE *f, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    int r = vfprintf(f, fmt, ap);
    va_end(ap);
    return r;
}

int printf(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    int r = vfprintf(stdout, fmt, ap);
    va_end(ap);
    return r;
}

/* --- scanf (the subset m_config / m_misc use: %d %i %u %x %o %s %c %[...] %n) ----------- */

typedef struct {
    const char *s;  /* a string source, or */
    FILE *f;        /* a file source */
    int count;      /* characters consumed */
} src_t;

static int src_get(src_t *src)
{
    int c = src->f ? fgetc(src->f) : (*src->s ? (unsigned char)*src->s++ : EOF);
    if (c != EOF)
        src->count++;
    return c;
}

static void src_unget(src_t *src, int c)
{
    if (c == EOF)
        return;
    src->count--;
    if (src->f)
        ungetc(c, src->f);
    else
        src->s--;
}

static int vscan(src_t *src, const char *fmt, va_list ap)
{
    int assigned = 0, c;
    for (; *fmt; fmt++) {
        if (isspace((unsigned char)*fmt)) {
            while (isspace(c = src_get(src)))
                ;
            src_unget(src, c);
            continue;
        }
        if (*fmt != '%' || fmt[1] == '%') {
            if (*fmt == '%')
                fmt++;
            c = src_get(src);
            if (c != (unsigned char)*fmt) {
                src_unget(src, c);
                return assigned ? assigned : (c == EOF ? EOF : 0);
            }
            continue;
        }
        fmt++;
        int skip = 0, width = 0;
        if (*fmt == '*')
            skip = 1, fmt++;
        while (isdigit((unsigned char)*fmt))
            width = width * 10 + *fmt++ - '0';
        while (*fmt == 'l' || *fmt == 'h')
            fmt++;
        char conv = *fmt;
        if (conv == 'n') {
            if (!skip)
                *va_arg(ap, int *) = src->count;
            continue;
        }
        if (conv != 'c' && conv != '[') {
            while (isspace(c = src_get(src)))
                ;
            src_unget(src, c);
        }
        if (!width)
            width = conv == 'c' ? 1 : 1 << 30;
        if (conv == 'd' || conv == 'i' || conv == 'u' || conv == 'x' || conv == 'o') {
            char num[40];
            int n = 0, base = conv == 'x' ? 16 : conv == 'o' ? 8 : conv == 'i' ? 0 : 10;
            while (n < width && n < 39) {
                c = src_get(src);
                int ok = isxdigit(c) || ((c == '-' || c == '+') && n == 0)
                         || ((c | 32) == 'x' && n >= 1 && num[n - 1] == '0' && (base == 0 || base == 16));
                if (ok && base == 10 && !isdigit(c) && c != '-' && c != '+')
                    ok = 0;
                if (ok && base == 8 && (c == '8' || c == '9' || isalpha(c)))
                    ok = 0;
                if (!ok) {
                    src_unget(src, c);
                    break;
                }
                num[n++] = (char)c;
            }
            num[n] = 0;
            char *end;
            long v = strtol(num, &end, base);
            if (end == num)
                return assigned ? assigned : (c == EOF ? EOF : 0);
            if (!skip) {
                *va_arg(ap, int *) = (int)v;
                assigned++;
            }
        } else if (conv == 's' || conv == 'c' || conv == '[') {
            char set[256] = {0};
            int invert = 0;
            if (conv == '[') {
                fmt++;
                if (*fmt == '^')
                    invert = 1, fmt++;
                if (*fmt == ']')
                    set[']'] = 1, fmt++;
                for (; *fmt && *fmt != ']'; fmt++)
                    set[(unsigned char)*fmt] = 1;
            }
            char *out = skip ? NULL : va_arg(ap, char *);
            int n = 0;
            while (n < width) {
                c = src_get(src);
                if (c == EOF)
                    break;
                int ok = conv == 'c' ? 1 : conv == 's' ? !isspace(c) : set[c] != invert;
                if (!ok) {
                    src_unget(src, c);
                    break;
                }
                if (out)
                    out[n] = (char)c;
                n++;
            }
            if (!n)
                return assigned ? assigned : (c == EOF ? EOF : 0);
            if (out) {
                if (conv != 'c')
                    out[n] = 0;
                assigned++;
            }
        } else {
            return assigned;
        }
    }
    return assigned;
}

int vsscanf(const char *s, const char *fmt, va_list ap)
{
    src_t src = {s, NULL, 0};
    return vscan(&src, fmt, ap);
}

int sscanf(const char *s, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    int r = vsscanf(s, fmt, ap);
    va_end(ap);
    return r;
}

int fscanf(FILE *f, const char *fmt, ...)
{
    src_t src = {NULL, f, 0};
    va_list ap;
    va_start(ap, fmt);
    int r = vscan(&src, fmt, ap);
    va_end(ap);
    return r;
}
