"""Random RV32IM + Zba programs (qwendoom.core_fuzz: every ALU / MUL / DIV / Zba op, loads and
stores of every width into a buffer and anywhere in the 256 KB RAM, loads anywhere in the 512 KB
DROM, branches, calls, putchar, YIELD, LUI / AUIPC): fault-free in the reference and exact in
the circuit, teacher-forced. QD_SLOW=1 runs the longer seeds."""

import os

import pytest

from qwendoom import core, core_check, core_fuzz, ref
from qwendoom.core_asm import assemble

SLOW = not os.environ.get("QD_SLOW")


@pytest.fixture(scope="module")
def circ():
    return core.build()


def fuzz_machine(seed: int, iters: int) -> ref.Machine:
    return ref.Machine.from_program(assemble(core_fuzz.gen_program(seed, block=40, iters=iters)))


@pytest.mark.parametrize("seed", range(4))
def test_fuzz(circ, seed):
    m = fuzz_machine(1000 + seed, 3)
    res = core_check.check(circ, m, kernel=seed == 0)
    assert m.fault is None and m.exit_code == 0
    assert res.checked and not res.wrong, str(res)


@pytest.mark.skipif(SLOW, reason="long traces (QD_SLOW=1)")
@pytest.mark.parametrize("seed", range(4, 24))
def test_fuzz_long(circ, seed):
    m = fuzz_machine(1000 + seed, 120)
    res = core_check.check(circ, m)
    assert m.fault is None and m.exit_code == 0
    assert res.checked and not res.wrong, str(res)
