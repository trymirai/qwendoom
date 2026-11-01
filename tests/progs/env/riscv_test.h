// riscv-tests environment for the qwendoom RV32 machine (derived from Qwenputer's
// chipformer/rv/env/riscv_test.h, MIT, (c) 2026 Divyansh Agrawal; see LICENSE.chipformer): no
// CSRs, no traps; a test ends with the exit ecall (a7 = 93, a0 = 0 on pass, the failing TESTNUM
// otherwise), which ref.Rv32, ref.Machine and the circuit all understand.
#ifndef _ENV_QWENDOOM_RV_H
#define _ENV_QWENDOOM_RV_H

#define RVTEST_RV64U .macro init; .endm
#define RVTEST_RV32U .macro init; .endm
#define TESTNUM gp
#define RVTEST_CODE_BEGIN .text; .globl _start; _start: init;
#define RVTEST_CODE_END unimp
#define RVTEST_PASS li a0, 0; li a7, 93; ecall
#define RVTEST_FAIL mv a0, TESTNUM; li a7, 93; ecall
#define RVTEST_DATA_BEGIN .data; .align 4; .global begin_signature; begin_signature:
#define RVTEST_DATA_END .align 4; .global end_signature; end_signature:
#define EXTRA_DATA

#endif
