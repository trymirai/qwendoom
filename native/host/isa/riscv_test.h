// riscv-tests environment for the qwendoom host computer (derived from Qwenputer's
// chipformer/rv/env/riscv_test.h, MIT (c) 2026 Divyansh Agrawal). Each test builds twice:
//   QD_ISA_SIM 1: the test is sim_main in ROM, run by the native sim context (region checks
//                 on); the host's _start asks for one sim tick. Pass = YIELD, fail = exit(TESTNUM).
//   QD_ISA_SIM 0: the test is the host's _start in HOST; pass = exit(0), fail = exit(TESTNUM).
// Test data goes to RAM (isa.ld). No CSRs, no traps.
#ifndef _ENV_QD_HOST_H
#define _ENV_QD_HOST_H

#define RVTEST_RV64U .macro init; .endm
#define RVTEST_RV32U .macro init; .endm
#define TESTNUM gp

#if QD_ISA_SIM
#define RVTEST_CODE_BEGIN                                                                       \
    .section .text.host, "ax"; .globl _start;                                                   \
    _start: li a7, 0x411; ecall; li a1, 0; li a0, 0; li a7, 93; ecall;                          \
    .section .text.sim, "ax"; .globl sim_main; sim_main: init;
#define RVTEST_PASS li a0, 256; li a7, 1; ecall; 1: j 1b
#else
#define RVTEST_CODE_BEGIN                                                                       \
    .section .text.sim, "ax"; .globl sim_main; sim_main: j sim_main;                            \
    .section .text.host, "ax"; .globl _start; _start: init;
#define RVTEST_PASS li a1, 0; li a0, 0; li a7, 93; ecall
#endif
#define RVTEST_CODE_END unimp
#define RVTEST_FAIL li a1, 0; mv a0, TESTNUM; li a7, 93; ecall
#define RVTEST_DATA_BEGIN .data; .align 4; .global begin_signature; begin_signature:
#define RVTEST_DATA_END .align 4; .global end_signature; end_signature:
#define EXTRA_DATA

#endif
