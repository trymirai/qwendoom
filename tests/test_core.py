"""The circuit (qwendoom.core) against the reference trace, teacher-forced in the IR with every
exactness check on (and the HF-kernel read replay on short traces): small programs, faults at the
map's edges, the R refresh, snapshots mid-run, CSR stubs, putchar / YIELD, MUL / DIV / ALU fuzz,
Zba and DROM loads, the input port with host-injected records; plus the circuit's shape, the fast
evaluator against chipformer's, and the snapshot rows."""

import random
import time

import numpy as np
import pytest

from chipformer.q35 import ir
from qwendoom import core, core_check, ref, snapshot
from qwendoom import spec as S
from qwendoom.core_asm import assemble
from qwendoom.q35 import evaluate as fe

EXIT = "\nli a7, 93\necall"


@pytest.fixture(scope="module")
def circ():
    return core.build()


def run_check(c, prog, inputs=(), **kw) -> core_check.Result:
    res = core_check.check(c, ref.Machine.from_program(prog, inputs=inputs), **kw)
    assert res.checked and not res.wrong, str(res)
    return res


def test_shape(circ):
    """16 layers; every sublayer within Qwen3.5's shapes (16 key heads, 4 value heads each,
    the MLP width); 128 RAM heads (256 KB), 32 DROM heads (512 KB), 16 ROM heads (256 KB)."""
    st = circ.stats()
    assert st["layers"] == 16 and st["max_heads"] <= circ.target.key_heads == 16, st
    assert st["max_units"] <= circ.target.intermediate
    assert all(len(n.vheads()) <= core.VHEADS for n in circ.nodes if isinstance(n, ir.Head))
    names = [n.name for n in circ.nodes if isinstance(n, ir.Head)]
    assert sum(n.startswith("ram") for n in names) == 128
    assert sum(n.startswith("drom") for n in names) == 32
    assert sum(n.startswith("rom") for n in names) == 16
    # every head read is one-hot, or 2-hot for the byte tables (a key is one-hot by ir's check)
    for n in circ.nodes:
        if isinstance(n, ir.Head) and n.levels == "byte":
            assert len(n.q) == 128


SIGNEXT = """_start: la t0, vals
        lb a0, 0(t0)
        lbu a1, 0(t0)
        lh a2, 2(t0)
        lhu a3, 2(t0)
        lb a4, 5(t0)
        lh a5, 6(t0)
        li t1, -2
        sh t1, 8(t0)
        sb t1, 11(t0)
        lw a6, 8(t0)
        lui a7, 0xABCDE
        srai s2, a7, 7
        srli s3, a7, 7
        slli s4, a7, 3
        slt s5, a7, zero
        sltu s6, a7, zero
        sltiu s7, a0, -1
        xori s8, a1, 0x7FF
        ori s9, a0, 0x123
        andi s10, a2, -256
        sub s11, a2, a7""" + EXIT + """
.data
vals:   .word 0x8F7F80FF, 0x7F7F0080, 0x11223344"""

CALL = """_start: li a0, 7
        call fact
        mv s1, a0
        li a0, 12
        call sum
        li a7, 93
        ecall
fact:   addi sp, sp, -16
        sw ra, 12(sp)
        sw s0, 8(sp)
        mv s0, a0
        li t0, 2
        blt a0, t0, fact1
        addi a0, a0, -1
        call fact
        mul a0, a0, s0
        j fact2
fact1:  li a0, 1
fact2:  lw ra, 12(sp)
        lw s0, 8(sp)
        addi sp, sp, 16
        ret
sum:    addi sp, sp, -16
        sw ra, 12(sp)
        sw a0, 8(sp)
        beqz a0, sum0
        addi a0, a0, -1
        call sum
        lw t0, 8(sp)
        add a0, a0, t0
        j sum1
sum0:   li a0, 0
sum1:   lw ra, 12(sp)
        addi sp, sp, 16
        ret"""

# the map's corners: the RAM's first and last words (every RAM key head bit pattern is in
# memsweep.c), the DROM's first and last words, words / halves / bytes, stores at 0x3FFFC
MEMORY = """_start: li t0, 0x3FFFC
        li a0, 0x89ABCDEF
        sw a0, 0(t0)
        lw a1, 0(t0)
        sb a0, 1(t0)
        sh a0, 2(t0)
        lw a2, 0(t0)
        lh a3, 2(t0)
        lbu a4, 1(t0)
        li t1, 0x100
        sw a0, 0(t1)
        sw a1, 4(t1)
        sw a2, 64(t1)
        lw s2, 0(t1)
        lw s3, 64(t1)
        li t2, 0x80000
        lw s4, 0(t2)
        lb s5, 1(t2)
        li t3, 0xFFFFC
        lw s6, 0(t3)
        lhu s7, 2(t3)
        lh s8, 2(t3)
        li t4, 0x2A000
        sw a0, 0(t4)
        sw a1, 4(t4)
        sw a2, 8(t4)
        sw a3, 12(t4)
        lw s9, 4(t4)
        lw s10, 12(t4)
        lw s11, 8(t4)""" + EXIT


def memory_prog():
    p = assemble(MEMORY)
    rod = bytearray(S.DROM_SIZE)
    rod[0:4] = (0x80FF7F01).to_bytes(4, "little")
    rod[-4:] = (0xC3B2A190).to_bytes(4, "little")
    p.rodata = bytes(rod)
    return p


@pytest.mark.parametrize("src", ["signext", "call", "memory"])
def test_programs(circ, src):
    prog = {"signext": lambda: assemble(SIGNEXT), "call": lambda: assemble(CALL),
            "memory": memory_prog}[src]()
    run_check(circ, prog)


def test_memory_values():
    m = ref.Machine.from_program(memory_prog())
    m.run()
    assert m.x[11] == 0x89ABCDEF and m.x[12] == 0xCDEFEFEF and m.x[22] == 0xC3B2A190
    assert m.x[20] == 0x80FF7F01 and m.x[21] == 0x7F and m.x[23] == 0xC3B2


def test_fast_evaluate_equals_ir(circ):
    """qwendoom.q35.evaluate == chipformer.q35.ir.evaluate, every dim at every position (both
    with the kernel replay and every check)."""
    prog = assemble(CALL)
    snapshot.snapshot(circ, ref.Machine.from_program(prog))
    toks = ref.Machine.from_program(prog).run()[:200]
    a = fe.evaluate(circ, toks, check=True, kernel=True)
    b = ir.evaluate(circ, toks, check=True, kernel=True)
    assert np.array_equal(np.asarray(a), np.asarray(b))
    # chunked (with carry) == one piece
    carry = {}
    parts = [fe.evaluate(circ, toks[s:s + 64], check=True, carry=carry) for s in (0, 64, 128, 192)]
    assert np.array_equal(np.concatenate([np.asarray(p) for p in parts]), np.asarray(a))


def test_refresh_after_63(circ):
    src = "_start: " + "\n".join(f"addi a0, a0, {i % 7}" for i in range(140)) + EXIT
    res = run_check(circ, assemble(src))
    assert sum(S.family(t)[0] == "R" for t in res.tokens) == 2


@pytest.mark.parametrize("skip", [5, 40, 133])
def test_snapshot_mid_run(circ, skip):
    """A snapshot mid-run (n > 0, live registers and RAM) continues exactly."""
    m = ref.Machine.from_program(assemble(CALL))
    for _ in range(skip):
        m.record()
    assert 0 < m.n < S.N_MAX
    res = core_check.check(circ, m)
    assert res.checked and not res.wrong, str(res)


FAULTS = {
    "illegal": "_start: li a0, 7\n.word 0x0000000b",  # custom-0 (the removed digit file)
    "illegal (custom-1)": "_start: li a0, 7\n.word 0x0000002b",
    "illegal (csr write)": "_start: li a0, 1\ncsrw mscratch, a0",
    "ebreak": "_start: nop\nebreak",
    "misaligned load": "_start: li a0, 0x102\nlw a1, 0(a0)",
    "misaligned store": "_start: li a0, 0x101\nsh a0, 0(a0)",
    "load out of range": "_start: li a0, 0x40000\nlw a1, 0(a0)",
    "load out of range (rom)": "_start: li a0, 0x100000\nlw a1, 0(a0)",
    "load out of range (below drom)": "_start: li a0, 0x7FFFC\nlw a1, 0(a0)",
    "load out of range (host)": "_start: li a0, 0x1000000\nlbu a1, 0(a0)",
    "store out of range": "_start: li a0, -4\nsb a0, 0(a0)",
    "store out of range (drom)": "_start: li a0, 0x80000\nsw a0, 0(a0)",
    "store out of range (ram end)": "_start: li a0, 0x40000\nsh a0, 0(a0)",
    "bad jump": "_start: li a0, 0x40000\njr a0",
    "bad jump (rom end)": "_start: li a0, 0x140000\njalr a0",
    "bad jump (misaligned)": "_start: la a0, t\naddi a0, a0, 2\njr a0\nt: nop",
    "bad jump (jal)": "_start: nop\nj 0x140004",
    "bad jump (branch)": "_start: beq zero, zero, 0xFFFF0",
    "ecall": "_start: li a7, 5\necall",
}


@pytest.mark.parametrize("name", list(FAULTS))
def test_faults(circ, name):
    res = run_check(circ, assemble(FAULTS[name]))
    cause = name.split(" (")[0]
    fams = [S.family(t) for t in res.tokens]
    assert fams[-10] == ("FLT", S.FAULTS.index(cause)) and fams[-1][0] == "HALT"


def test_csr_stubs_putchar_yield(circ):
    src = """_start: csrr a0, mhartid
        csrr a1, misa
        csrr a2, cycle
        csrr a3, instret
        csrr a4, timeh
        li a7, 1
        li a0, 72
        ecall
        li a0, 256
        ecall
        li a0, 105
        ecall
        li a0, 0x1FF
        ecall""" + EXIT
    res = run_check(circ, assemble(src))
    assert [S.family(t) for t in res.tokens if S.family(t)[0] == "BYTE"] == [
        ("BYTE", 72), ("BYTE", S.YIELD), ("BYTE", 105), ("BYTE", 511)]


# ---------------------------------------------------------------- the input port

INPUT = """_start: csrr a0, time
        mv s0, a0
        csrr a1, time
        csrr a2, time
        csrr s3, time
        mv s1, a1
        mv s2, a2
        add s4, a0, a1
        csrr zero, time
        csrr t0, time
        sw s3, 0(t0)
        lw s5, 0(t0)
        csrr t1, time
        lw s6, 0(t1)
        csrr t2, time
        sb a2, 0(t2)
        lw s7, 0(t2)
        csrr ra, time
        csrr sp, time
        csrr t6, time
        csrr t6, time
        xor s8, ra, sp
        sub s9, t6, s8
        csrr a0, time
        li a7, 1
        ecall
        li a0, 256
        ecall
loop:   csrr a3, time
        beqz a3, done
        csrr a4, time
        sw a4, 0(a3)
        j loop
done:   lw s10, 0x104(zero)
        li t3, 0x3FFF8
        lw s11, 0(t3)""" + EXIT

INPUT_WORDS = [5, -1, 0x12345, 0xDEADBEEF,    # 1-, 1-, 2- and 3-token records
               0x3F000, 0x80004, 0x2000,     # addresses: RAM (2 tokens), DROM, RAM
               0x7FFFFFFF, -0x4000, 0x4000, 0x20000000,  # ra, sp, t6, t6 back to back
               0x41,                         # putchar's byte
               0x104, 77, 0x3FFF8, 0x80000000, 0]  # pokes, then the end of the list


def test_input_injection(circ):
    """The host's input records (1-, 2- and 3-token values, rd = several registers, back-to-back
    reads, loads / stores through the injected value right after) are committed exactly: every
    non-injected token is predicted exactly, including the value records of instructions that
    copy and combine the injected registers; the circuit itself predicts F0(0) (the folded CSR
    stub) where each input record starts."""
    prog = assemble(INPUT)
    prog.rodata = bytes(4) + (0xCAFEF00D).to_bytes(4, "little")
    m = ref.Machine.from_program(prog, inputs=INPUT_WORDS)
    snapshot.snapshot(circ, ref.Machine.from_program(prog, inputs=INPUT_WORDS))
    res = run_check(circ, prog, inputs=INPUT_WORDS)
    toks, inj = res.tokens, sorted(res.injected)
    m.run()
    want = [t for w in INPUT_WORDS for t in S.value_record(w)]
    assert [toks[i] for i in inj] == want
    lens = sorted({len(S.value_record(w)) for w in INPUT_WORDS})
    assert lens == [1, 2, 3]
    assert (m.x[8], m.x[9], m.x[18], m.x[19]) == (5, 0xFFFFFFFF, 0x12345, 0xDEADBEEF)
    assert m.x[21] == 0xDEADBEEF and m.x[22] == 0xCAFEF00D and m.x[23] == 0x45
    assert m.x[26] == 77 and m.x[27] == 0x80000000 and m.out == b"A" and m.yields == 1
    # the circuit's own prediction at each input record's first token: F0(0)
    mat = fe.evaluate(circ, toks, check=True)
    pred = core.predictions(circ, mat)
    starts, ends, pos = [], [], 1  # each input record's first and last positions
    m2 = ref.Machine.from_program(prog, inputs=INPUT_WORDS)
    while not m2.halted:
        rec = m2.record()
        if m2.last_input:
            starts.append(pos)
            ends.append(pos + len(rec) - 1)
        pos += len(rec)
    assert len(starts) == len(INPUT_WORDS) and all(pred[i - 1] == S.tok("F0", 0) for i in starts)
    # the register file holds exactly the injected value right after the record: the read
    # ports at its last token (where the circuit decodes the next instruction)
    A = core.probe_value(circ, mat, circ.probe["A"])
    B = core.probe_value(circ, mat, circ.probe["B"])
    for k, port, w in ((0, A, 5), (3, A, -1), (4, A, 0x3F000), (5, A, 0x80004), (6, A, 0x2000),
                       (10, A, 0x7FFFFFFF), (10, B, -0x4000)):
        assert port[ends[k]] == w & 0xFFFFFFFF, (k, hex(port[ends[k]]), hex(w))


def test_input_poke_loop(circ):
    """input.c-style poke loops with 3-token addresses / words through the loop above."""
    words = [5, 6, 7, 8, 0x3F000, 0x80000, 0x2000, 1, 2, 3, 4, 0x41]
    words += [a for k in range(12) for a in (0x200 + 4 * k, (k * 0x9E3779B9) & 0xFFFFFFFF)] + [0]
    prog = assemble(INPUT)
    prog.rodata = bytes(8)
    run_check(circ, prog, inputs=words)


# ---------------------------------------------------------------- fuzz


def random_alu_program(rng: random.Random, n: int) -> str:
    regs = ["a0", "a1", "a2", "a3", "a4", "a5", "t0", "t1", "t2", "s1"]
    lines = [f"li {r}, {rng.randint(-2**31, 2**31 - 1)}" for r in regs]
    ops3 = ["add", "sub", "and", "or", "xor", "sll", "srl", "sra", "slt", "sltu", "sh1add",
            "sh2add", "sh3add"]
    opsi = ["addi", "andi", "ori", "xori", "slti", "sltiu"]
    opss = ["slli", "srli", "srai"]
    for _ in range(n):
        k = rng.random()
        d, a, b = rng.choice(regs), rng.choice(regs), rng.choice(regs)
        if k < 0.5:
            lines.append(f"{rng.choice(ops3)} {d}, {a}, {b}")
        elif k < 0.8:
            lines.append(f"{rng.choice(opsi)} {d}, {a}, {rng.randint(-2048, 2047)}")
        elif k < 0.9:
            lines.append(f"{rng.choice(opss)} {d}, {a}, {rng.randint(0, 31)}")
        else:
            lines.append(f"{rng.choice(['lui', 'auipc'])} {d}, {rng.randint(0, 2**20 - 1)}")
    return "_start: " + "\n".join(lines) + EXIT


@pytest.mark.parametrize("seed", range(4))
def test_alu_fuzz(circ, seed):
    run_check(circ, assemble(random_alu_program(random.Random(seed), 150)))


def test_branch_fuzz(circ):
    rng = random.Random(7)
    lines = ["_start: li s0, 40"]
    for i in range(16):
        a, b = rng.choice([rng.randint(-5, 5), rng.randint(-2**31, 2**31 - 1)]), rng.randint(-5, 5)
        br = rng.choice(["beq", "bne", "blt", "bge", "bltu", "bgeu"])
        lines += [f"li a0, {a}", f"li a1, {b}", f"{br} a0, a1, skip{i}", "addi s1, s1, 1",
                  f"skip{i}: addi s0, s0, -1"]
    lines += ["loop: addi s0, s0, -1", "bgtz s0, loop"]
    run_check(circ, assemble("\n".join(lines) + EXIT))


@pytest.mark.parametrize("seed", range(3))
def test_mul_fuzz(circ, seed):
    rng = random.Random(100 + seed)
    regs = ["a0", "a1", "a2", "a3", "t0", "t1"]
    lines = [f"li {r}, {rng.choice([rng.randint(-2**31, 2**31 - 1), rng.randint(-9, 9)])}"
             for r in regs]
    for _ in range(80):
        op = rng.choice(["mul", "mulh", "mulhsu", "mulhu", "add", "xor"])
        lines.append(f"{op} {rng.choice(regs)}, {rng.choice(regs)}, {rng.choice(regs)}")
    run_check(circ, assemble("_start: " + "\n".join(lines) + EXIT))


DIV_EDGE = [0, 1, -1, 2, -2, 7, -7, 15, 16, 255, 256, 1000, 2**31 - 1, -2**31, -2**31 + 1, 0xFFFF,
            -0x10000, 10, 3]


@pytest.mark.parametrize("seed", range(4))
def test_div_fuzz(circ, seed):
    """DIV / DIVU / REM / REMU on edge values and random operands of every size (digits skipped
    or not, signs, division by 0, overflow), rd = x0 and rd == rs1 included."""
    rng = random.Random(200 + seed)
    regs = ["a0", "a1", "a2", "a3", "t0", "t1", "s1"]

    def val():
        return rng.choice([rng.choice(DIV_EDGE), rng.randint(-2**31, 2**31 - 1),
                           rng.randint(-2**(rng.randint(1, 31)), 2**rng.randint(1, 31))])

    lines = [f"li {r}, {val()}" for r in regs]
    for i in range(40):
        if i % 6 == 5:
            lines += [f"li {r}, {val()}" for r in rng.sample(regs, 3)]
        op = rng.choice(["div", "divu", "rem", "remu"])
        d = "zero" if rng.random() < 0.1 else rng.choice(regs)
        lines.append(f"{op} {d}, {rng.choice(regs)}, {rng.choice(regs)}")
    run_check(circ, assemble("_start: " + "\n".join(lines) + EXIT))


def test_zba_and_rodata(circ):
    src = """_start: li a0, 0x1234567
        li a1, -77
        sh1add a2, a0, a1
        sh2add a3, a1, a0
        sh3add a4, a0, a0
        la s0, tab
        lw t0, 0(s0)
        lw t1, 4(s0)
        lb t2, 5(s0)
        lbu t3, 7(s0)
        lh t4, 6(s0)
        lhu t5, 2(s0)
        sh2add t6, a0, s0""" + EXIT + """
        .rodata
        tab: .word 0x89abcdef, 0x80ff7f01"""
    prog = assemble(src)
    run_check(circ, prog)
    m = ref.Machine.from_program(prog)
    m.run()
    assert m.x[12] == (0x1234567 * 2 - 77) & 0xFFFFFFFF
    assert (m.x[5], m.x[7], m.x[28], m.x[29], m.x[30]) == (0x89abcdef, 0x7f, 0x80, 0xFFFF80FF,
                                                           0x89ab)


# ---------------------------------------------------------------- snapshot rows


def full_images(seed=0):
    rng = np.random.default_rng(seed)
    ram = rng.integers(0, 256, S.RAM_SIZE, dtype=np.uint8).tobytes()
    drom = rng.integers(0, 256, S.DROM_SIZE, dtype=np.uint8).tobytes()
    rom = rng.integers(0, 2**32, S.ROM_WORDS, dtype=np.uint64).astype("<u4").tobytes()
    regs = [int(x) for x in rng.integers(0, 2**32, 32)]
    return ram, drom, rom, regs


def test_state_rows_speed_and_layout(circ):
    """state_rows of full 256 KB / 512 KB / 256 KB images: < 1 s, every row inside its value
    head's state, and the same rows as the IR snapshot of the same machine."""
    ram, drom, rom, regs = full_images()
    snapshot.state_rows(ram, drom, rom, regs, S.ROM_BASE)  # (warm-up: the head map)
    t = time.perf_counter()
    rows = snapshot.state_rows(ram, drom, rom, regs, S.ROM_BASE + 4 * 1234)
    dt = time.perf_counter() - t
    assert dt < 1.0, dt
    assert rows.first == S.tok("GO") and rows.vec.dtype == np.float32
    assert rows.vec.shape == (len(rows), 128) and rows.row.max() < 128
    assert rows.vhead.max() < circ.target.value_heads and rows.layer.max() < circ.n_layers()
    # == the IR snapshot (core's heads' init) mapped through head_slots
    m = ref.Machine.from_images(ram, drom, rom, regs, S.ROM_BASE + 4 * 1234)
    snapshot.snapshot(circ, m)
    slots = snapshot.head_slots(circ)
    r = circ.target.value_heads // circ.target.key_heads
    want = {}
    for name, node in circ.tables.items():
        L, j, _ = slots[name]
        for x, init in enumerate(node.init):
            for row, vec in init.items():
                want[L, j * r + x, row] = vec
    got = {(int(a), int(b), int(c)): v for a, b, c, v in zip(rows.layer, rows.vhead, rows.row,
                                                             rows.vec)}
    assert got.keys() == want.keys()
    assert all(np.array_equal(got[k], want[k].astype(np.float32)) for k in want)


def test_state_rows_run(circ):
    """A run from state_rows: the rows, written into the IR's heads by (layer, value head, row),
    predict the trace exactly (the snapshot path other tracks use)."""
    prog = assemble(CALL)
    m = ref.Machine.from_program(prog)
    ram = b"".join(w.to_bytes(4, "little") for w in m.ram)
    rom = b"".join(w.to_bytes(4, "little") for w in m.rom)
    rows = snapshot.state_rows(ram, m.drom, rom, m.x, m.pc)
    by_slot = {(L, vh): {} for L, vh in zip(rows.layer.tolist(), rows.vhead.tolist())}
    for L, vh, r, v in zip(rows.layer.tolist(), rows.vhead.tolist(), rows.row.tolist(),
                           rows.vec):
        by_slot[L, vh][r] = v.astype(np.float64)
    rv = circ.target.value_heads // circ.target.key_heads
    for name, (L, j, nvh) in snapshot.head_slots(circ).items():
        circ.tables[name].init = [by_slot.get((L, j * rv + x), {}) for x in range(nvh)]
    toks = ref.Machine.from_program(prog).run()
    assert toks[0] == rows.first
    pred = core.predictions(circ, fe.evaluate(circ, toks, check=True))
    assert pred[:-1] == toks[1:]
