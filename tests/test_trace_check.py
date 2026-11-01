"""The lockstep checker (native/trace/qd_check.c) and the generated token header: a matching
stream ends cleanly, every corruption is reported at its own token with its record's context,
host-owned input records are skipped (and feed the reference), and qd_tokens.h is current."""

import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from chipformer.rv import asm, fuzz
from chipformer.rv import spec as S
from qwendoom import trace as T

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "third_party" / "qwenputer" / "experimental" / "riscv" / "tests" / "data" / "rv"
UP = T.Map.upstream()
REGS = [0, 0, S.SP_INIT] + [0] * 29


def machine(prog=None, path=None, map_=UP, regs=REGS):
    mem = T.Memory(map_)
    entry = mem.load_elf(path) if path else mem.load_program(prog)
    r = T.Ref(mem)
    r.set_context(pc=entry, regs=regs)
    return r, entry


def test_header_is_current():
    """native/trace/qd_tokens.h equals what gen_tokens.py makes from chipformer.rv.spec."""
    p = subprocess.run([sys.executable, str(ROOT / "native" / "trace" / "gen_tokens.py"),
                        "--check"], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr


def test_token_ids():
    """The C helpers and the header agree with the spec on ids at family edges."""
    assert T.encode_value_record(0) == [S.tok("F0", 0)]
    assert T.encode_value_record(-1) == [S.tok("F0", 0x7FFF)]
    assert T.encode_value_record(1 << 14) == [S.tok("V", 1 << 14), S.tok("F1", 0)]
    assert T.encode_value_record(0xFFFFFFFF ^ (1 << 30)) == [S.tok("V", 0x7FFF),
                                                            S.tok("V", 0x7FFF), S.tok("F2", 2)]
    head = (ROOT / "native" / "trace" / "qd_tokens.h").read_text()
    assert f"#define QD_VOCAB {S.VOCAB}" in head
    for n, (base, size) in S.FAMILIES.items():
        assert f"#define QD_TOK_{n} {base}\n" in head and f"QD_TOK_{n}_SIZE {size}\n" in head
    for n, t in S.CONTROLS.items():
        assert f"#define QD_TOK_{n} {t}\n" in head


def test_qwendoom_spec_agrees():
    """qwendoom.spec (the circuit track's) has upstream's vocabulary, so qd_tokens.h's ids are
    its ids too, and its address map, input CSR and YIELD equal native/qd_mem.h's (Map.qwendoom)."""
    Q = pytest.importorskip("qwendoom.spec")
    assert (Q.FAMILIES, Q.CONTROLS, Q.VOCAB, Q.FAULTS, Q.N_MAX, Q.MISA) == \
        (S.FAMILIES, S.CONTROLS, S.VOCAB, S.FAULTS, S.N_MAX, S.MISA)
    m = T.Map.qwendoom()
    assert (m.ram_base, m.ram_size, m.drom_base, m.drom_size, m.rom_base, m.rom_size) == \
        (Q.RAM_BASE, Q.RAM_SIZE, Q.DROM_BASE, Q.DROM_SIZE, Q.ROM_BASE, Q.ROM_SIZE)
    assert m.input_csr == Q.CSR_INPUT == T.CSR_INPUT and Q.YIELD == T.YIELD_BYTE


@pytest.mark.parametrize("name", ["c-crc32p", "mpy-s1", "rv32um-rem", "c-qsort"])
def test_matching_stream(name):
    r, entry = machine(path=DATA / f"{name}.elf")
    toks, _, why = r.trace()
    ref, _ = machine(path=DATA / f"{name}.elf")
    chk = T.Checker(ref)
    n, st = chk.check(toks)
    assert (n, st) == (len(toks), T.CHK_END) and chk.divergence is None
    assert chk.counts == {"tokens": len(toks), "records": r.counts["records"], "host_tokens": 0}
    assert chk.feed(S.tok("NT")) == T.CHK_DIVERGED  # a token after the end
    assert chk.divergence.want == -1 and chk.divergence.index == len(toks)


def test_every_corruption_is_found():
    """Change one token anywhere (300 random places in a fuzz program's trace): the checker stops
    exactly there, and the divergence names the reference's record."""
    prog = asm.assemble(fuzz.gen_program(31, block=40, iters=10))
    r, _ = machine(prog)
    toks, flags, _ = r.trace()
    starts = np.nonzero(flags & T.TF_FIRST)[0]
    rng = random.Random(0)
    for _ in range(300):
        i = rng.randrange(len(toks))
        bad = toks.copy()
        bad[i] = (int(toks[i]) + rng.randrange(1, S.VOCAB)) % S.VOCAB
        ref, _ = machine(prog)
        chk = T.Checker(ref)
        n, st = chk.check(bad)
        d = chk.divergence
        assert (n, st) == (i, T.CHK_DIVERGED) and d.index == i
        start = int(starts[np.searchsorted(starts, i, side="right") - 1])
        end = int(starts[np.searchsorted(starts, i, side="right")]) if i < starts[-1] else len(toks)
        assert d.record_start == start and d.ref_record == toks[start:end].tolist()
        assert d.got == bad[i] and d.want == toks[i]
        assert str(d).startswith(f"token {i} ")


def test_truncated_and_feed():
    prog = asm.assemble(fuzz.gen_program(3, block=20, iters=3))
    r, _ = machine(prog)
    toks, _, _ = r.trace()
    ref, _ = machine(prog)
    chk = T.Checker(ref)
    for t in toks[:-1]:
        assert chk.feed(int(t)) == T.CHK_OK
    assert chk.state == T.CHK_OK  # (a prefix is fine: lockstep is token by token)
    assert chk.feed(int(toks[-1])) == T.CHK_END


POKES = """
_start: csrr a0, time
        beqz a0, tick
        csrr a1, time
        sw a1, 0(a0)
        j _start
tick:   lw t0, 0x300(zero)
        addi t0, t0, 1
        sw t0, 0x300(zero)
        li a0, 256
        li a7, 1
        ecall
        j _start
"""


@pytest.fixture
def qd_prog(monkeypatch):
    monkeypatch.setattr(asm, "ROM_BASE", 0x100000)
    monkeypatch.setattr(asm, "DROM_BASE", 0x80000)
    return asm.assemble(POKES)


def test_host_owned_positions(qd_prog):
    """Input records are host-owned: with an empty input queue the checker takes the words from
    the stream; with a queue the stream's (different) record there is not compared."""
    words = [0x104, 5, 0x200, 0x7FFFFFFF, 0, 0x108, -9, 0]
    r, _ = machine(qd_prog, map_=T.Map.qwendoom(), regs=[0, 0, 0x3000] + [0] * 29)
    r.push_input(words)
    toks, flags, why = r.run(stop_yield=True)
    more, mflags, _ = r.run(stop_yield=True)
    toks, flags = np.concatenate([toks, more]), np.concatenate([flags, mflags])
    host = int((flags & T.TF_HOST).astype(bool).sum())
    assert host == sum(len(S.value_record(w)) for w in words)

    ref, _ = machine(qd_prog, map_=T.Map.qwendoom(), regs=[0, 0, 0x3000] + [0] * 29)
    chk = T.Checker(ref)  # no input queued: the stream supplies it
    assert chk.check(toks) == (len(toks), T.CHK_OK)
    assert chk.counts["host_tokens"] == host and ref.mem.word(0x200) == 0x7FFFFFFF

    # the queue holds other words: the stream's input records are not compared, but the next
    # record that depends on them is
    ref, _ = machine(qd_prog, map_=T.Map.qwendoom(), regs=[0, 0, 0x3000] + [0] * 29)
    ref.push_input([0x104, 6, 0x200, 0x7FFFFFFF, 0, 0x108, -9, 0])
    chk = T.Checker(ref)
    n, st = chk.check(toks)
    assert st == T.CHK_OK  # the poked value goes straight to memory: no record depends on it
    assert ref.mem.word(0x104) == 6

    # a non-value token at a host-owned position is a divergence (want -2)
    bad = toks.copy()
    i = int(np.nonzero(flags & T.TF_HOST)[0][0])
    bad[i] = S.tok("NT")
    ref, _ = machine(qd_prog, map_=T.Map.qwendoom(), regs=[0, 0, 0x3000] + [0] * 29)
    chk = T.Checker(ref)
    assert chk.check(bad) == (i, T.CHK_DIVERGED) and chk.divergence.want == -2


def test_reference_error_state():
    """The checker reports CHK_ERROR when the reference cannot run (PC outside the ROM)."""
    mem = T.Memory(UP)
    ref = T.Ref(mem)
    ref.set_context(anchor=(1 << 16) - 1, n=1)
    chk = T.Checker(ref)
    assert chk.feed(S.tok("NT")) == T.CHK_ERROR and chk.state == T.CHK_ERROR


def test_leading_go():
    """A run's stream may start with GO (the token after a snapshot): the checker and the decoder
    skip it there, and only there."""
    prog = asm.assemble(fuzz.gen_program(4, block=20, iters=2))
    r, entry = machine(prog)
    toks, _, _ = r.trace()
    stream = np.concatenate([[S.tok("GO")], toks]).astype(np.int32)
    ref, _ = machine(prog)
    chk = T.Checker(ref)
    assert chk.check(stream) == (len(stream), T.CHK_END)
    mem = T.Memory(UP)
    mem.load_program(prog)
    d = T.Decoder(mem)
    d.set_context(pc=entry, regs=REGS)
    assert d.feed_many(stream) == (T.HALT, len(stream))
    d.set_context(pc=entry, regs=REGS)
    assert d.feed(int(toks[0])) >= 0 and d.feed(S.tok("GO")) == -2
    ref, _ = machine(prog)
    chk = T.Checker(ref)
    assert chk.check(np.concatenate([toks[:1], stream[:1]])) == (1, T.CHK_DIVERGED)
