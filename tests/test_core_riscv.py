"""riscv-tests rv32ui (minus fence_i and ma_data), rv32um and rv32uzba, built by
tests/progs/build.py for the qwendoom map, teacher-forced through the circuit over their whole
run (with the HF-kernel read replay: every trace is under 1,500 tokens)."""

import pytest

from qwendoom import core, core_check
from qwendoom import spec as S

NAMES = sorted(p.stem for p in core_check.ensure_progs().glob("rv32u*.elf"))


@pytest.fixture(scope="module")
def circ():
    return core.build()


def test_suite_complete():
    suites = {n.split("-")[0] for n in NAMES}
    assert suites == {"rv32ui", "rv32um", "rv32uzba"} and len(NAMES) == 40 + 8 + 3


@pytest.mark.parametrize("name", NAMES)
def test_riscv_test(circ, name):
    res = core_check.check(circ, core_check.elf_machine(name), kernel=True)
    assert res.checked and not res.wrong, str(res)
    assert res.tokens[-1] == S.tok("HALT") and S.family(res.tokens[-2]) == ("F0", 93)
