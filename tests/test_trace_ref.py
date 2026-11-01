"""The C reference traced machine (native/trace/qd_ref.c) against upstream's Python
chipformer.rv.ref.Machine, token for token: every upstream test ELF, upstream's fuzz programs,
hand-made programs for the faults, CSRs, refreshes and snapshots, and qwendoom's map and input
port (upstream's machine with the map patched in, and qwendoom.ref.Machine when it imports)."""

import random
from collections import deque
from pathlib import Path

import numpy as np
import pytest

from chipformer.rv import asm, elf, fuzz, ref
from chipformer.rv import spec as S
from qwendoom import trace as T

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "third_party" / "qwenputer" / "experimental" / "riscv" / "tests" / "data" / "rv"
ELFS = sorted(p.stem for p in DATA.glob("*.elf") if not p.name.endswith(".qemu.elf"))
QD = dict(ROM_BASE=0x100000, ROM_WORDS=1 << 16, DROM_BASE=0x80000, DROM_WORDS=1 << 17,
          RAM_WORDS=1 << 16, SP_INIT=0x3000)  # qwendoom's map in upstream's spec names


def c_machine(prog_or_path, map_: T.Map, sp: int) -> tuple[T.Ref, T.Memory]:
    mem = T.Memory(map_)
    entry = mem.load_elf(prog_or_path) if isinstance(prog_or_path, Path) \
        else mem.load_program(prog_or_path)
    r = T.Ref(mem)
    regs = [0] * 32
    regs[2] = sp
    r.set_context(pc=entry, regs=regs)
    return r, mem


def py_trace(m, max_records: int = 10**8, stop=None) -> list[int]:
    """Records of a Python machine until it halts (or stop(m) before a record)."""
    out = []
    for _ in range(max_records):
        if m.halted or (stop and stop(m)):
            break
        out += m.record()
    return out


def assert_same(got: np.ndarray, want: list[int], what: str) -> None:
    w = np.asarray(want, np.int64)
    g = got.astype(np.int64)
    n = min(len(g), len(w))
    bad = np.nonzero(g[:n] != w[:n])[0]
    i = int(bad[0]) if len(bad) else n
    if i < n or len(g) != len(w):
        ctx = slice(max(0, i - 4), i + 4)
        raise AssertionError(f"{what}: first difference at token {i} of {len(w)} (ours "
                             f"{len(g)}): ours {[T.name(int(t)) for t in g[ctx]]}, upstream "
                             f"{[T.name(int(t)) for t in w[ctx]]}")


def ram_words(mem: T.Memory, n: int) -> list[int]:
    return np.frombuffer(mem.ram[:4 * n].tobytes(), "<u4").astype(np.int64).tolist()


def custom_op(m) -> bool:
    """The next record executes a digit-file op (custom-0 / custom-1)."""
    if m.n == S.N_MAX:
        return False
    return ref.decode(m.rom[(m.pc - S.ROM_BASE) >> 2]).op in ref.CUSTOM


# --- (a) every upstream test ELF --------------------------------------------------------------


@pytest.mark.parametrize("name", ELFS)
def test_upstream_elf(name):
    """Every token equals upstream's; so do the final registers, RAM, exit code and fault. ELFs
    that use the digit file match up to its first op, where ours raises an illegal fault."""
    path = DATA / f"{name}.elf"
    m = ref.Machine.from_program(elf.program(path))
    want = py_trace(m, stop=custom_op)
    r, mem = c_machine(path, T.Map.upstream(), S.SP_INIT)
    got, flags, why = r.trace()
    assert not (flags & T.TF_HOST).any()
    if not m.halted:  # stopped before a digit-file op
        assert why == "fault" and r.fault == (S.FAULTS.index("illegal"), m.pc)
        want += S.fault_record(S.FAULTS.index("illegal"), m.pc)
        assert_same(got, want, name)
        return
    assert_same(got, want, name)
    assert r.regs == m.x
    assert ram_words(mem, S.RAM_WORDS) == m.ram
    assert r.counts["steps"] == m.steps and r.counts["tokens"] == len(want)
    if m.fault:
        assert why == "fault" and r.fault == m.fault
    else:
        assert why == "halt" and r.exit_code == m.exit_code


def test_digit_file_elfs_are_covered():
    """calc.elf reaches a digit-file op (so the prefix rule above is exercised)."""
    m = ref.Machine.from_program(elf.program(DATA / "calc.elf"))
    py_trace(m, stop=custom_op)
    assert not m.halted and m.steps > 0


# --- (b) upstream's fuzz programs --------------------------------------------------------------


@pytest.mark.parametrize("chunk", range(4))
def test_fuzz_upstream_map(chunk):
    """chipformer.rv.fuzz programs (every ALU / MUL / DIV / Zba op, loads and stores of all
    widths, DROM loads, branches, calls, putchar, LUI / AUIPC), 50 seeds per chunk."""
    for seed in range(50 * chunk, 50 * chunk + 50):
        prog = asm.assemble(fuzz.gen_program(seed, block=40, iters=12))
        want = ref.Machine.from_program(prog).run(10**7)[1:]
        r, _ = c_machine(prog, T.Map.upstream(), S.SP_INIT)
        got, _, why = r.trace()
        assert why == "halt"
        assert_same(got, want, f"fuzz seed {seed}")


@pytest.fixture
def qd_map(monkeypatch):
    """Upstream's spec / assembler / machine with qwendoom's address map and input port: `time`
    pops from INPUTS when rd != x0 (as qwendoom.ref.Machine does)."""
    for k, v in QD.items():
        monkeypatch.setattr(S, k, v)
    monkeypatch.setattr(asm, "ROM_BASE", QD["ROM_BASE"])
    monkeypatch.setattr(asm, "DROM_BASE", QD["DROM_BASE"])
    inputs = deque()
    stub = ref.csr_read

    def csr_read(ins):
        v = stub(ins)
        if v is not None and ins.imm == 0xC01 and ins.rd:
            return inputs.popleft()
        return v

    monkeypatch.setattr(ref, "csr_read", csr_read)
    return inputs


@pytest.mark.parametrize("seed", range(40))
def test_fuzz_qwendoom_map(qd_map, seed):
    """The same fuzz programs at qwendoom's addresses (ROM 0x100000, DROM 0x80000, RAM 256 KB)."""
    prog = asm.assemble(fuzz.gen_program(1000 + seed, block=40, iters=8))
    m = ref.Machine.from_program(prog)
    want = m.run(10**7)[1:]
    r, mem = c_machine(prog, T.Map.qwendoom(), QD["SP_INIT"])
    got, _, why = r.trace()
    assert why == "halt"
    assert_same(got, want, f"fuzz seed {1000 + seed} (qwendoom map)")
    assert r.regs == m.x and ram_words(mem, 1 << 16) == m.ram


# --- hand-made programs ------------------------------------------------------------------------

FAULTS = {  # cause: a program that raises it
    "illegal": "_start: li a0, 1\n .word 0xFFFFFFFF",
    "ebreak": "_start: nop\n ebreak",
    "misaligned load": "_start: li a0, 0x102\n lw a1, 0(a0)",
    "misaligned store": "_start: li a0, 0x101\n sh a1, 0(a0)",
    "load out of range": "_start: li a0, 0x10000000\n lb a1, 0(a0)",
    "store out of range": "_start: li a0, 0x40000\n sw a1, 4(a0)",  # (ROM: not storable)
    "bad jump": "_start: li a0, 0x100\n jalr ra, 0(a0)",
    "ecall": "_start: li a7, 64\n ecall",
}
CSR_READ = 0xC0102573  # csrrs a0, time, x0 (a legal read)
MORE_WORDS = {
    "custom-1": 0x0000002B, "custom-0 f3=5": 0x0000500B, "mret": 0x30200073, "wfi": 0x10500073,
    "csrrw mscratch": 0x34001073, "csrrw time": 0xC0101073, "csrr unknown": 0x7C002573,
    "csrrs rs1!=0": CSR_READ | 11 << 15, "csrrwi": 0xC0105573, "R-type hole": 0x40001033,
    "fence variant": 0x0000700F, "jalr f3!=0": 0x000510E7, "branch f3=2": 0x00002063,
}


@pytest.mark.parametrize("cause", list(FAULTS))
def test_faults(cause):
    prog = asm.assemble(FAULTS[cause])
    m = ref.Machine.from_program(prog)
    want = m.run()[1:]
    assert m.fault[0] == S.FAULTS.index(cause)
    r, _ = c_machine(prog, T.Map.upstream(), S.SP_INIT)
    got, _, why = r.trace()
    assert why == "fault" and r.fault == m.fault
    assert_same(got, want, cause)


@pytest.mark.parametrize("what", list(MORE_WORDS))
def test_odd_words(what):
    """Words upstream decodes as ILLEGAL (or FENCE) give the same record; the digit-file ops
    upstream did implement are illegal here."""
    w = MORE_WORDS[what]
    prog = asm.assemble(f"_start: li a0, 5\n .word {w}\n li a7, 93\n ecall")
    r, _ = c_machine(prog, T.Map.upstream(), S.SP_INIT)
    got, _, why = r.trace()
    if ref.decode(w).op in ref.CUSTOM:
        assert why == "fault" and r.fault == (0, S.ROM_BASE + 4)
        return
    want = ref.Machine.from_program(prog).run()[1:]
    assert_same(got, want, what)
    assert why == ("halt" if ref.decode(w).op == "FENCE" else "fault")


def test_csr_stubs():
    src = """_start: csrr a0, cycle
        csrr a1, time
        csrr a2, instret
        csrr a3, mhartid
        csrr a4, misa
        csrr zero, misa
        .word 0xC8102673   # csrrs a2, timeh, x0
        .word 0xC0107773   # csrrci a4, time, 0
        li a7, 93
        ecall"""
    prog = asm.assemble(src)
    want = ref.Machine.from_program(prog).run()[1:]
    r, _ = c_machine(prog, T.Map.upstream(), S.SP_INIT)
    got, _, _ = r.trace()
    assert_same(got, want, "csr stubs")
    assert r.regs[14] == 0 and r.regs[12] == 0


def test_refresh_and_long_straight_code():
    """200 straight instructions: R(anchor + 63) before every 64th, then a jump back in."""
    body = "\n".join(f"addi a{i % 6}, a{(i + 1) % 6}, {i - 100}" for i in range(200))
    src = f"_start: li s0, 3\nloop: {body}\n addi s0, s0, -1\n bnez s0, loop\n li a7, 93\n ecall"
    prog = asm.assemble(src)
    want = ref.Machine.from_program(prog).run()[1:]
    assert sum(S.family(t)[0] == "R" for t in want) >= 9
    r, _ = c_machine(prog, T.Map.upstream(), S.SP_INIT)
    got, _, _ = r.trace()
    assert_same(got, want, "refreshes")


def test_div_edges():
    """DIV / DIVU / REM / REMU on edge operands, rd = x0 and rd != x0, through programs."""
    rng = random.Random(5)
    edge = [0, 1, 2, 7, 15, 16, 255, 256, 0x7FFFFFFF, 0x80000000, 0x80000001, 0xFFFFFFFF,
            0xFFFFFFFE, 10, 1000, 0x10000, 0xFFFF]
    lines = ["_start:"]
    for _ in range(400):
        a, b = rng.choice(edge + [rng.getrandbits(32)]), rng.choice(edge + [rng.getrandbits(16)])
        op = rng.choice(["div", "divu", "rem", "remu"])
        rd = rng.choice(["a2", "a2", "a2", "zero"])
        lines += [f"li a0, {a}", f"li a1, {b}", f"{op} {rd}, a0, a1"]
    prog = asm.assemble("\n".join(lines + ["li a7, 93", "ecall"]))
    want = ref.Machine.from_program(prog).run()[1:]
    r, _ = c_machine(prog, T.Map.upstream(), S.SP_INIT)
    got, _, _ = r.trace()
    assert_same(got, want, "div edges")


def test_div_and_value_records():
    """The C helpers equal spec.value_record and ref.div_record."""
    rng = random.Random(1)
    vals = [0, 1, -1, 2**14 - 1, -2**14, 2**14, -2**14 - 1, 2**29 - 1, -2**29, 2**29,
            -2**29 - 1, 2**31 - 1, -2**31] + [rng.getrandbits(32) for _ in range(3000)]
    for v in vals:
        assert T.encode_value_record(v) == S.value_record(v), v
    edge = [0, 1, 2, 7, 15, 16, 255, 256, 0x7FFFFFFF, 0x80000000, 0x80000001, 2**32 - 1,
            2**32 - 2, 10, 1000]
    ops = edge + [rng.getrandbits(rng.choice([4, 8, 16, 24, 32])) for _ in range(300)]
    for op in ("DIV", "DIVU", "REM", "REMU"):
        for _ in range(3000):
            a, b = rng.choice(ops), rng.choice(ops)
            assert T.div_record(op, a, b) == ref.div_record(op, a, b), (op, a, b)


def test_snapshot_context_midrun():
    """Start the C machine from a Python machine's state at many points (n > 0 included)."""
    prog = asm.assemble(fuzz.gen_program(77, block=40, iters=12))
    m = ref.Machine.from_program(prog)
    records = sum(S.is_final(t) for t in py_trace(m))
    assert m.halted and records > 600
    for k in (1, 17, 62, 63, 64, 500, records - 2):
        m = ref.Machine.from_program(prog)
        py_trace(m, max_records=k)
        assert not m.halted
        mem = T.Memory(T.Map.upstream())
        mem.load_program(prog)
        mem.ram[:] = np.asarray(m.ram, np.int64).astype("<u4").view(np.uint8)
        r = T.Ref(mem)
        r.set_context(anchor=m.anchor, n=m.n, regs=m.x)
        want = py_trace(m)
        got, _, _ = r.trace()
        assert_same(got, want, f"snapshot after {k} records")


def test_rom_rewrite_needs_no_invalidation():
    """The pre-decode cache is keyed by the raw word: rewriting the ROM takes effect."""
    p1 = asm.assemble("_start: li a0, 7\n li a7, 93\n ecall")
    p2 = asm.assemble("_start: li a0, 9\n li a7, 93\n ecall")
    r, mem = c_machine(p1, T.Map.upstream(), S.SP_INIT)
    r.trace()
    assert r.exit_code == 7
    mem.load_program(p2)
    r.set_context(pc=S.ROM_BASE)
    r.trace()
    assert r.exit_code == 9


def test_pc_outside_rom_is_an_error():
    """Running off the ROM's end, or an R refresh past word 65535, has no record: ERROR."""
    r, mem = c_machine(asm.assemble("_start: nop"), T.Map.upstream(), 0)
    nop = (0x13).to_bytes(4, "little")
    mem.write(S.ROM_BASE + 4 * 65534, nop + nop)
    r.set_context(anchor=65534)
    toks, _, why = r.run()
    assert why == "error" and r.status == T.ERROR and toks.tolist() == [S.tok("NT")] * 2
    r.set_context(anchor=65500, n=63)
    toks, _, why = r.run()
    assert why == "error" and len(toks) == 0


# --- qwendoom's map and input port -------------------------------------------------------------

POKES = """
_start: csrr a0, time       # the host's pokes: (addr, word) pairs until addr 0
        beqz a0, tick
        csrr a1, time
        sw a1, 0(a0)
        j _start
tick:   csrr zero, time     # (rd = x0: reads nothing, NT)
        lw t0, 0x300(zero)
        addi t0, t0, 1
        sw t0, 0x300(zero)
        li a0, 256
        li a7, 1
        ecall               # YIELD
        j _start
"""


def reads_input(m) -> bool:
    """The next record of a Python machine is `csrr rd, time` with rd != x0."""
    if m.n == S.N_MAX:
        return False
    ins = ref.decode(m.rom[(m.pc - S.ROM_BASE) >> 2])
    pure = ins.op in ("CSRRS", "CSRRC", "CSRRSI", "CSRRCI") and ins.rs1 == 0
    return pure and ins.imm == 0xC01 and ins.rd != 0


def test_input_port(qd_map):
    """csrr rd, time pops the input queue (host-owned value records); rd = x0 pops nothing; an
    empty queue stops the machine (NEED_INPUT) until words are pushed."""
    prog = asm.assemble(POKES)
    ticks = [[(0x104, 5), (0x200, 0xDEADBEEF), (0x108, -3)], [], [(0x3FFFC, 1 << 20)]]
    words = [w for t in ticks for p in t + [(0, None)] for w in p if w is not None]
    qd_map.extend(words)
    m = ref.Machine.from_program(prog)
    want = py_trace(m, max_records=10**5, stop=lambda m: not qd_map and reads_input(m))
    r, mem = c_machine(prog, T.Map.qwendoom(), QD["SP_INIT"])
    got, flags, yields = [], [], 0
    for t in ticks:
        r.push_input([w for p in t + [(0, None)] for w in p if w is not None])
        toks, fl, why = r.run(stop_yield=True)
        assert why == "yield"
        got += toks.tolist()
        flags += fl.tolist()
        yields += 1
    toks, fl, why = r.run()
    assert why == "input" and r.status == T.NEED_INPUT and r.wants_input
    got += toks.tolist()
    flags += fl.tolist()
    assert_same(np.asarray(got), want, "pokes")
    assert ram_words(mem, 1 << 16) == m.ram and r.regs == m.x
    assert mem.word(0x104) == 5 and mem.word(0x200) == 0xDEADBEEF and mem.word(0x300) == 3
    # host-owned tokens: exactly the input records' (value records of the pushed words)
    host = [t for t, f in zip(got, flags) if f & T.TF_HOST]
    assert host == [t for w in words for t in S.value_record(w)]
    r.push_input([0])
    toks, _, why = r.run(stop_yield=True)
    assert why == "yield" and mem.word(0x300) == 4


def test_qwendoom_ref_machine():
    """The C machine is a token-exact port of qwendoom.ref.Machine (another track's module; skipped
    while it does not import)."""
    qref = pytest.importorskip("qwendoom.ref")
    qasm = pytest.importorskip("qwendoom.core_asm")
    try:
        prog = qasm.assemble(POKES)
        m = qref.Machine.from_program(prog, inputs=[0x104, 5, 0x200, -7, 0, 0])
        want = []
        while not m.halted and m.yields < 2:
            want += m.record()
    except Exception as e:  # noqa: BLE001  (an API in flux on the other track)
        pytest.skip(f"qwendoom.ref is not usable yet: {e!r}")
    r, mem = c_machine(prog, T.Map.qwendoom(), 0x3000)
    r.push_input([0x104, 5, 0x200, -7, 0, 0])
    got, _, _ = r.run(stop_yield=True)
    more, _, _ = r.run(stop_yield=True)
    assert_same(np.concatenate([got, more]), want, "qwendoom.ref pokes")
    for seed in range(10):
        prog = qasm.assemble(fuzz.gen_program(seed, block=40, iters=5))
        m = qref.Machine.from_program(prog)
        want = m.run(10**7)[1:]
        r, mem = c_machine(prog, T.Map.qwendoom(), 0x3000)
        got, _, _ = r.trace()
        assert_same(got, want, f"qwendoom.ref fuzz {seed}")


OPCODES = [0x37, 0x17, 0x6F, 0x67, 0x63, 0x03, 0x23, 0x13, 0x33, 0x0F, 0x73, 0x0B, 0x2B]


def random_word(rng) -> int:
    """An instruction word biased to the decoder's paths: a real opcode (or none), random fields
    with the funct7 / CSR fields often set to the values that decode."""
    w = rng.getrandbits(32)
    if rng.random() < 0.9:
        w = w & ~0x7F | rng.choice(OPCODES)
    if rng.random() < 0.5:
        w = w & 0x01FFFFFF | rng.choice([0, 0x20, 1, 0x10]) << 25
    if w & 0x7F == 0x73 and rng.random() < 0.6:
        csr = rng.choice([0xC00, 0xC01, 0xC02, 0xC80, 0xC81, 0xC82, 0xF14, 0x301, 0x340])
        w = w & 0x000FFFFF | csr << 20
        if rng.random() < 0.6:
            w &= ~(31 << 15)  # rs1 = 0: a pure read
    if w & 0x7F in (0x03, 0x23) and rng.random() < 0.7:
        w &= ~(0x7F << 25 | 0x1F << 7 if w & 0x7F == 0x23 else 0xFFF << 20)  # offset 0
        w = w & ~(7 << 12) | rng.choice([0, 1, 2] if w & 0x7F == 0x23 else [0, 1, 2, 4, 5]) << 12
    if rng.random() < 0.08:
        w = rng.choice([0x73, 0x73, 0x00100073, 0x30200073])
    return w


def random_regs(rng) -> list[int]:
    pool = [0, 1, 2, 3, 4, 93, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 0x100, 0x7FFC, 0x7FFE, 0x8000,
            0x20000, 0x3FFFF, 0x40000, 0x40004, 0x40100, 0x7FFFF, 256, 0x1FFFF, 0x40, 0x1C4,
            0x2000, 0x2042, 0x20040]
    regs = [rng.choice(pool) if rng.random() < 0.7 else rng.getrandbits(32) for _ in range(32)]
    regs[0] = 0
    regs[17] = rng.choice([1, 93, 64, rng.getrandbits(32)])  # a7: the ecall
    return regs


def random_words(seed: int, count: int) -> dict:
    """Run count random words (seed) through upstream's machine, ours and the decoder; returns
    how many of the words' records began with each family."""
    rng = random.Random(seed)
    exit_seq = [0x05D00893, 0x00000073]  # li a7, 93; ecall
    mem = T.Memory(T.Map.upstream())
    dmem = T.Memory(T.Map.upstream())
    r, d = T.Ref(mem), T.Decoder(dmem)
    m = ref.Machine([0] * S.ROM_WORDS, [0] * S.RAM_WORDS, [0] * 32, 0)
    kinds = {}
    for i in range(count):
        w, regs, n = random_word(rng), random_regs(rng), rng.choice([0, 0, 5, 62, 63])
        words = [0] * 68
        words[n:n + 3] = [w] + exit_seq
        if ref.decode(w).op in ref.CUSTOM:
            continue
        rom = np.asarray(words, np.int64).astype("<u4").view(np.uint8)
        mem.rom[:len(rom)], dmem.rom[:len(rom)] = rom, rom
        m.rom[:len(words)] = words
        m.x, m.anchor, m.n, m.halted, m.fault, m.exit_code = list(regs), 0, n, False, None, None
        want = py_trace(m, max_records=6)
        r.set_context(anchor=0, n=n, regs=regs)
        got, flags, why = r.run(max_records=6)
        assert_same(got, want, f"word {w:#010x} (n {n})")
        assert r.regs == m.x, hex(w)
        st = r.last_store
        if st is not None:
            assert mem.word(st) == m.ram[st >> 2]
        d.set_context(anchor=0, n=n, regs=regs)
        dmem.ram[:] = mem.ram if st is None else dmem.ram
        res, used = d.feed_many(got)
        assert res >= 0 and used == len(got), (hex(w), d.error_text)
        assert d.regs == r.regs
        dmem.ram[:] = mem.ram
        first = np.nonzero(flags & T.TF_FIRST)[0][1 if n == 63 else 0]  # the word's record
        k = S.family(int(got[first]))[0]
        kinds[k] = kinds.get(k, 0) + 1
    return kinds


def test_random_words():
    """20,000 random instruction words (every decode path, faults of every kind, CSRs, the digit
    file's opcodes, random n before an R) in random register files: upstream's machine and ours
    give the same records, registers and stored words; the decoder (strict) follows."""
    kinds = random_words(2026, 20000)
    # every kind of record came up
    assert all(kinds.get(k, 0) > 20 for k in ("FLT", "J", "NT", "F0", "V", "ST0", "ST1", "QS",
                                               "BYTE", "HALT")), str(kinds)
