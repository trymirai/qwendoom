/* Guest-side ecalls to the host interpreter (numbers in native/host/qd_host_abi.h). */
#ifndef QD_SYS_H
#define QD_SYS_H

#include "qd_host_abi.h"

static inline long qd_ecall(long n, long a0, long a1, long a2)
{
    register long r0 __asm__("a0") = a0;
    register long r1 __asm__("a1") = a1;
    register long r2 __asm__("a2") = a2;
    register long r7 __asm__("a7") = n;
    __asm__ volatile("ecall" : "+r"(r0) : "r"(r1), "r"(r2), "r"(r7) : "memory");
    return r0;
}

#endif
