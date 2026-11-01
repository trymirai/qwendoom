/* qwendoom's freestanding libc: the compiler's libcalls for RV32IM (64-bit division; there is
   no compiler-rt for riscv32 here). FixedDiv's ((int64_t)a << 16) / b runs in the sim, so the
   common case (a 32-bit divisor) uses the hardware's 32-bit DIVU on 16-bit digits (Hacker's
   Delight divlu) instead of a bit loop. */
#include <stdint.h>

typedef uint64_t u64;
typedef int64_t s64;
typedef uint32_t u32;

static int clz32(u32 x)  /* x != 0; no Zbb, and __builtin_clz would call __clzsi2 */
{
    int n = 0;
    if (!(x >> 16)) { n += 16; x <<= 16; }
    if (!(x >> 24)) { n += 8; x <<= 8; }
    if (!(x >> 28)) { n += 4; x <<= 4; }
    if (!(x >> 30)) { n += 2; x <<= 2; }
    if (!(x >> 31)) { n += 1; }
    return n;
}

/* (u1:u0) / v for u1 < v: the quotient, and the remainder in *r */
static u32 divlu(u32 u1, u32 u0, u32 v, u32 *r)
{
    const u32 b = 65536;
    int s = clz32(v);
    v <<= s;
    u32 vn1 = v >> 16, vn0 = v & 0xFFFF;
    u32 un32 = s ? (u1 << s) | (u0 >> (32 - s)) : u1;
    u32 un10 = u0 << s;
    u32 un1 = un10 >> 16, un0 = un10 & 0xFFFF;
    u32 q1 = un32 / vn1, rhat = un32 - q1 * vn1;
    while (q1 >= b || q1 * vn0 > b * rhat + un1) {
        q1--;
        rhat += vn1;
        if (rhat >= b)
            break;
    }
    u32 un21 = un32 * b + un1 - q1 * v;
    u32 q0 = un21 / vn1;
    rhat = un21 - q0 * vn1;
    while (q0 >= b || q0 * vn0 > b * rhat + un0) {
        q0--;
        rhat += vn1;
        if (rhat >= b)
            break;
    }
    *r = (un21 * b + un0 - q0 * v) >> s;
    return q1 * b + q0;
}

static u64 udivmod(u64 n, u64 d, u64 *rem)
{
    u32 nh = n >> 32, nl = (u32)n;
    if (d == 0) {
        *rem = n;
        return ~(u64)0;
    }
    if (d >> 32 == 0) {
        u32 dv = (u32)d, r;
        if (nh < dv) {
            u32 q = divlu(nh, nl, dv, &r);
            *rem = r;
            return q;
        }
        u32 qh = nh / dv;
        u32 ql = divlu(nh - qh * dv, nl, dv, &r);
        *rem = r;
        return (u64)qh << 32 | ql;
    }
    /* a 64-bit divisor: the quotient fits 32 bits; restoring division over its bits */
    int s = clz32((u32)(d >> 32)) - clz32(nh ? nh : 1);
    if (n < d) {
        *rem = n;
        return 0;
    }
    if (s < 0)
        s = 0;
    u64 q = 0, dd = d << s;
    for (; s >= 0; s--, dd >>= 1) {
        q <<= 1;
        if (n >= dd) {
            n -= dd;
            q |= 1;
        }
    }
    *rem = n;
    return q;
}

u64 __udivdi3(u64 a, u64 b)
{
    u64 r;
    return udivmod(a, b, &r);
}

u64 __umoddi3(u64 a, u64 b)
{
    u64 r;
    udivmod(a, b, &r);
    return r;
}

s64 __divdi3(s64 a, s64 b)
{
    u64 r, ua = a < 0 ? -(u64)a : (u64)a, ub = b < 0 ? -(u64)b : (u64)b;
    u64 q = udivmod(ua, ub, &r);
    return (a < 0) != (b < 0) ? -(s64)q : (s64)q;
}

s64 __moddi3(s64 a, s64 b)
{
    u64 r, ua = a < 0 ? -(u64)a : (u64)a, ub = b < 0 ? -(u64)b : (u64)b;
    udivmod(ua, ub, &r);
    return a < 0 ? -(s64)r : (s64)r;
}

/* --- soft-float: the few libcalls the host side needs (v_video's mouse box, g_game's timedemo
   fps). Round to nearest even; subnormal results flush to zero. Never reached by the sim. --- */

typedef union { float f; u32 u; } fbits;
typedef union { double d; u64 u; } dbits;

static int top_bit(u64 m)
{
    int p = 0;
    while (m >> 1) {
        m >>= 1;
        p++;
    }
    return p;
}

/* the float nearest to (-1)^s * m * 2^e */
static float fpack(int s, int e, u64 m)
{
    fbits r;
    if (!m) {
        r.u = (u32)s << 31;
        return r.f;
    }
    int p = top_bit(m), sh = p - 23;
    if (sh > 0) {
        u64 rem = m & ((1ull << sh) - 1), half = 1ull << (sh - 1);
        m >>= sh;
        if (rem > half || (rem == half && (m & 1)))
            m++;
        if (m >> 24) {
            m >>= 1;
            sh++;
        }
    } else {
        m <<= -sh;
    }
    int be = e + sh + 23 + 127;
    if (be >= 255)
        r.u = (u32)s << 31 | 0x7F800000u;
    else if (be <= 0)
        r.u = (u32)s << 31;
    else
        r.u = (u32)s << 31 | (u32)be << 23 | ((u32)m & 0x7FFFFF);
    return r.f;
}

/* unpack a finite float: (-1)^s * m * 2^e; returns 0 for inf / nan (handled by the callers) */
static int funpack(float f, int *s, int *e, u64 *m)
{
    fbits b = {.f = f};
    int be = b.u >> 23 & 0xFF;
    *s = b.u >> 31;
    if (be == 255)
        return 0;
    *m = b.u & 0x7FFFFF;
    if (be)
        *m |= 1u << 23;
    else
        be = 1;
    *e = be - 127 - 23;
    return 1;
}

static float fnan(void)
{
    fbits r = {.u = 0x7FC00000u};
    return r.f;
}

float __addsf3(float a, float b)
{
    int sa, sb, ea, eb;
    u64 ma, mb;
    if (!funpack(a, &sa, &ea, &ma)) {  /* inf + -inf is a nan; otherwise a (inf or nan) */
        fbits x = {.f = a}, y = {.f = b};
        return funpack(b, &sb, &eb, &mb) || x.u == y.u ? a : fnan();
    }
    if (!funpack(b, &sb, &eb, &mb))
        return b;
    ma <<= 38;  /* 62-bit mantissas, then align the smaller exponent (sticky bit) */
    mb <<= 38;
    ea -= 38;
    eb -= 38;
    if (ea < eb) {
        u64 t = ma; ma = mb; mb = t;
        int ti = ea; ea = eb; eb = ti;
        ti = sa; sa = sb; sb = ti;
    }
    int d = ea - eb;
    if (d > 63)
        mb = mb != 0;
    else if (d)
        mb = (mb >> d) | ((mb & ((1ull << d) - 1)) != 0);
    if (sa == sb)
        return fpack(sa, ea, ma + mb);
    return ma >= mb ? fpack(sa, ea, ma - mb) : fpack(sb, ea, mb - ma);
}

float __subsf3(float a, float b)
{
    fbits nb = {.f = b};
    nb.u ^= 0x80000000u;
    return __addsf3(a, nb.f);
}

float __mulsf3(float a, float b)
{
    int sa, sb, ea, eb;
    u64 ma, mb;
    if (!funpack(a, &sa, &ea, &ma) || !funpack(b, &sb, &eb, &mb))
        return fnan();
    return fpack(sa ^ sb, ea + eb, ma * mb);
}

float __divsf3(float a, float b)
{
    int sa, sb, ea, eb;
    u64 ma, mb;
    if (!funpack(a, &sa, &ea, &ma) || !funpack(b, &sb, &eb, &mb) || !mb)
        return fnan();
    u64 q = (ma << 40) / mb, r = (ma << 40) % mb;
    return fpack(sa ^ sb, ea - eb - 40, q << 1 | (r != 0));
}

float __floatsisf(int x)
{
    return fpack(x < 0, 0, x < 0 ? -(u64)(s64)x : (u64)x);
}

int __fixsfsi(float f)
{
    int s, e;
    u64 m;
    if (!funpack(f, &s, &e, &m) || e > 7)
        return s ? (int)0x80000000 : 0x7FFFFFFF;
    u64 v = e >= 0 ? m << e : (e > -64 ? m >> -e : 0);
    return s ? -(int)v : (int)v;
}

double __extendsfdf2(float f)
{
    fbits b = {.f = f};
    dbits r;
    u64 s = (u64)(b.u >> 31) << 63;
    int be = b.u >> 23 & 0xFF;
    u64 m = b.u & 0x7FFFFF;
    if (be == 255) {
        r.u = s | 0x7FF0000000000000ull | m << 29;
    } else if (be == 0) {
        if (!m) {
            r.u = s;
        } else {  /* subnormal float: normal as a double */
            int p = top_bit(m);
            r.u = s | (u64)(p - 23 - 126 + 1023) << 52 | ((m << (52 - p)) & ((1ull << 52) - 1));
        }
    } else {
        r.u = s | (u64)(be - 127 + 1023) << 52 | m << 29;
    }
    return r.d;
}

static int dnan(u64 x) { return (x & 0x7FFFFFFFFFFFFFFFull) > 0x7FF0000000000000ull; }

/* -1 / 0 / 1 for a < / == / > b (1 if either is a nan); a total order on sign-magnitude bits */
static int dcmp(double a, double b)
{
    dbits x = {.d = a}, y = {.d = b};
    if (dnan(x.u) || dnan(y.u))
        return 1;
    if (((x.u | y.u) << 1) == 0)
        return 0;  /* +0 == -0 */
    s64 ix = x.u >> 63 ? -(s64)(x.u & 0x7FFFFFFFFFFFFFFFull) : (s64)x.u;
    s64 iy = y.u >> 63 ? -(s64)(y.u & 0x7FFFFFFFFFFFFFFFull) : (s64)y.u;
    return ix < iy ? -1 : ix > iy;
}

int __ltdf2(double a, double b) { return dcmp(a, b); }
int __ledf2(double a, double b) { return dcmp(a, b); }
int __gtdf2(double a, double b) { dbits x = {.d = a}, y = {.d = b}; return dnan(x.u) || dnan(y.u) ? -1 : dcmp(a, b); }
int __gedf2(double a, double b) { dbits x = {.d = a}, y = {.d = b}; return dnan(x.u) || dnan(y.u) ? -1 : dcmp(a, b); }
int __eqdf2(double a, double b) { return dcmp(a, b) != 0; }
int __nedf2(double a, double b) { return dcmp(a, b) != 0; }

double fabs(double x)
{
    dbits b = {.d = x};
    b.u &= 0x7FFFFFFFFFFFFFFFull;
    return b.d;
}

float fabsf(float x)
{
    fbits b = {.f = x};
    b.u &= 0x7FFFFFFFu;
    return b.f;
}
