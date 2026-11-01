"""The freestanding C programs (tests/progs/*.c, clang -O2 rv32im_zba, no libc), teacher-forced
through the circuit over their whole run: RAM across the whole 256 KB and DROM across the whole
512 KB (memsweep), deep recursion with a 2-token stack (recurse), 64-bit multiply / divide
libcalls (div64), byte / half stores (bytes), function pointers (funcptr), jump tables (switch),
the classics, and the input port with host pokes and YIELD (input). 25k-190k tokens each, about
1 ms per token: QD_SLOW=1 runs them all (the short ones always run)."""

import os

import pytest

from qwendoom import core, core_check
from qwendoom import spec as S

SLOW = not os.environ.get("QD_SLOW")
FAST = {"c-funcptr", "c-switch"}
NAMES = sorted(p.stem for p in core_check.ensure_progs().glob("c-*.elf"))


@pytest.fixture(scope="module")
def circ():
    return core.build()


@pytest.mark.parametrize("name", NAMES)
def test_c_program(circ, name):
    if SLOW and name not in FAST:
        pytest.skip("long trace (QD_SLOW=1)")
    m = core_check.elf_machine(name, core_check.elf_inputs(name))
    res = core_check.check(circ, m)
    assert m.fault is None and m.exit_code == 0, (m.fault, m.exit_code)
    assert res.checked and not res.wrong, str(res)
    assert res.tokens[-1] == S.tok("HALT")
    if name == "c-input":
        assert m.yields == 5 and len(res.injected) == 18
