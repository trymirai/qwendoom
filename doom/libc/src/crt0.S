# qwendoom's C runtime for the host side of DOOM: the host interpreter loads the ELF (zeroed
# .bss), sets sp to the top of the host stack and a0 / a1 to argc / argv; _start sets gp (shared
# by host and sim code: one __global_pointer$) and calls main, then exit.
    .section .text._start, "ax"
    .globl _start
_start:
    .option push
    .option norelax
    la gp, __global_pointer$
    .option pop
    call main
    call exit
1:  j 1b
