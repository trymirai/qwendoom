"""Teacher-forced checks of the circuit against the reference trace (the tests' engine).

    res = check(c, machine)        # c = core.build(); machine: a ref.Machine at the run's start
    assert not res.wrong

The machine's state goes into the circuit's heads (snapshot.snapshot), its trace runs through
the IR evaluator (qwendoom.q35.evaluate, chunk by chunk for long traces) with every exactness
check on, and the prediction at every position must equal the next token, except where the next
token is host-injected (an input record: Machine.injected). Short traces also replay every head
read the way the HF torch kernels compute it (``kernel``).
"""

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from qwendoom import core, snapshot
from qwendoom import ref as RF
from qwendoom import spec as S
from qwendoom.q35 import evaluate as fe

ROOT = Path(__file__).resolve().parents[1]
PROGS = ROOT / "build" / "core" / "progs"
KERNEL_MAX = 1500  # traces up to this many tokens also get the HF-kernel read replay


@dataclass
class Result:
    tokens: list[int]
    injected: set[int]
    checked: int = 0  # predictions compared
    wrong: list = field(default_factory=list)  # (position, want, got)

    def __str__(self) -> str:
        w = ", ".join(f"@{i}: want {S.name(a)} got {S.name(b) if b >= 0 else b}"
                      for i, a, b in self.wrong[:5])
        return (f"{len(self.tokens)} tokens, {len(self.injected)} injected, {self.checked} "
                f"checked, {len(self.wrong)} wrong{': ' + w if w else ''}")


def check(c, m: "RF.Machine", max_tokens: int = 10**7, chunk: int = 4096,
          kernel: bool | None = None) -> Result:
    """Run m to HALT (or max_tokens) in the reference, then teacher-force its trace through the
    circuit from m's starting state. m is consumed (it ends at the run's end)."""
    snapshot.snapshot(c, m)
    toks = m.run(max_tokens)
    inj = set(m.injected)
    res = Result(toks, inj)
    kernel = len(toks) <= KERNEL_MAX if kernel is None else kernel
    carry = {} if len(toks) > chunk else None
    for s in range(0, len(toks), chunk):
        part = toks[s:s + chunk]
        mat = fe.evaluate(c, part, check=True, carry=carry, kernel=kernel)
        pred = core.predictions(c, mat)
        for i, p in enumerate(pred):
            t = s + i + 1
            if t >= len(toks) or t in inj:
                continue
            res.checked += 1
            if p != toks[t]:
                res.wrong.append((t, toks[t], p))
    return res


def ensure_progs() -> Path:
    """build/core/progs (tests/progs/build.py runs when it is missing)."""
    if not (PROGS / "c-memsweep.elf").exists():
        subprocess.run([sys.executable, str(ROOT / "tests" / "progs" / "build.py"), "--out",
                        str(PROGS)], check=True)
    return PROGS


def elf_inputs(name: str) -> list[int]:
    """The input queue a test program reads (c-input: a few ticks of host pokes, then a poke
    to ``quit``; the other programs read none)."""
    if name != "c-input":
        return []
    from qwendoom import core_elf

    sym = core_elf.symbols(ensure_progs() / f"{name}.elf")
    cmd = sym["cmd"]
    return [cmd, 7, cmd + 4, 0xDEADBEEF, 0, cmd + 8, 1 << 20, 0, 0, cmd, 0xFFFFFFF0, 0,
            sym["quit"], 1, 0]


def elf_machine(name: str, inputs=()) -> "RF.Machine":
    from qwendoom import core_elf

    return RF.Machine.from_program(core_elf.program(ensure_progs() / f"{name}.elf"),
                                   inputs=inputs)
