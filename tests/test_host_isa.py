"""The host computer's RV32IM+Zba interpreter on riscv-tests (rv32ui, rv32um, rv32uzba), in both of
its contexts: as sim code in ROM run by the native sim mode (region checks on) and as host code.
Built by `make -C native/host isa-tests` with native/host/isa/riscv_test.h."""

import subprocess
from pathlib import Path

import pytest

from qwendoom import host as H

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "build" / "host-isa"
TESTS = ([f"rv32ui-{t}" for t in (
    "simple add addi and andi auipc beq bge bgeu blt bltu bne jal jalr lb lbu lh lhu lw ld_st lui "
    "or ori sb sh sw st_ld sll slli slt slti sltiu sltu sra srai srl srli sub xor xori").split()]
    + [f"rv32um-{t}" for t in "div divu mul mulh mulhsu mulhu rem remu".split()]
    + [f"rv32uzba-{t}" for t in "sh1add sh2add sh3add".split()])


@pytest.fixture(scope="module", autouse=True)
def build():
    for target in ("all", "isa-tests"):
        r = subprocess.run(["make", "-s", "-C", str(ROOT / "native" / "host"), target],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr


def run(context: str, name: str) -> tuple[bool, str]:
    """(passed, why) of one test ELF in the sim or host context."""
    with H.Host(elf=OUT / f"{context}-{name}.elf", args=()) as h:
        ev = h.run(10**7)
        if context == "host":
            return ev == H.EXIT and h.exit_code == 0, f"event {ev}, exit {h.exit_code} {h.error}"
        assert ev == H.SIM_TICK, h.error
        h.resync()
        h.take_pokes()
        try:
            st = h.sim_tick_native()
        except H.HostError as e:
            return False, str(e)
        return st.instructions > 0, "yielded"


@pytest.mark.parametrize("context", ["sim", "host"])
@pytest.mark.parametrize("name", TESTS)
def test_isa(context, name):
    ok, why = run(context, name)
    assert ok, why


@pytest.mark.parametrize("context", ["sim", "host"])
def test_isa_harness_sees_failures(context):
    """A test that fails on purpose (case 2) is reported as failing, with its number."""
    ok, why = run(context, "selftest_fail")
    assert not ok and ("exit(2)" in why or "exit 2" in why), why
