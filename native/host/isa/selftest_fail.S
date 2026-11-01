# A riscv-tests style test that fails on purpose (case 2: 1 + 1 is not 3), so that
# tests/test_host_isa.py can see the harness report a failure and its TESTNUM.
#include "riscv_test.h"
#include "test_macros.h"

RVTEST_RV32U
RVTEST_CODE_BEGIN

  TEST_RR_OP( 2, add, 0x00000003, 0x00000001, 0x00000001 );

  TEST_PASSFAIL

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN

  TEST_DATA

RVTEST_DATA_END
