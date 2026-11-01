# Minimal C runtime for test programs: sp = __stack_top (the snapshot already holds .data and a
# zeroed .bss), then exit(main()).
    .section .text.init
    .globl _start
_start:
    la sp, __stack_top
    call main
    li a7, 93
    ecall
1:  j 1b
