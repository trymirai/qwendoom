"""The reference machine (qwendoom.ref): the token vocabulary, the ROM word, the ISA emulator
against the self-checking riscv-tests and C programs with outputs computed here, the traced
machine against the ISA emulator, DIV records, faults and the input port."""

import random
from pathlib import Path

import numpy as np
import pytest

from qwendoom import core_check, core_elf, ref
from qwendoom import spec as S
from qwendoom.core_asm import assemble

PROGS = core_check.ensure_progs()
ELFS = sorted(p.stem for p in PROGS.glob("*.elf"))
EXIT = "\nli a7, 93\necall"


def run_rv32(name: str):
    cpu = ref.Rv32()
    core_elf.load_into(cpu, PROGS / f"{name}.elf")
    cpu.x[2] = S.SP_INIT
    cpu.inputs.extend(core_check.elf_inputs(name))
    while not cpu.halted and cpu.steps < 10**7:
        cpu.step()
    return cpu


def test_vocabulary_is_upstreams():
    """Token ids are stable across every tool: the families, sizes, order and controls of
    upstream chipformer.rv.spec, PAD = VOCAB."""
    from chipformer.rv import spec as U

    assert S.FAMILY_SIZES == U.FAMILY_SIZES
    assert S.FAMILIES == U.FAMILIES and S.CONTROLS == U.CONTROLS
    assert S.VOCAB == U.VOCAB == S.PAD == 230324
    for t in [0, 32767, 32768, 98304, 98308, 229380, 229380 + 256, 229892, 230059, S.VOCAB - 1]:
        assert S.family(t) == U.family(t) and S.name(t) == U.name(t)


def test_address_map_matches_qd_mem_h():
    h = (Path(__file__).parents[1] / "native" / "qd_mem.h").read_text()
    want = {"QD_RAM_BASE": S.RAM_BASE, "QD_RAM_SIZE": S.RAM_SIZE, "QD_DROM_BASE": S.DROM_BASE,
            "QD_DROM_SIZE": S.DROM_SIZE, "QD_ROM_BASE": S.ROM_BASE, "QD_ROM_SIZE": S.ROM_SIZE,
            "QD_HOST_BASE": S.HOST_BASE, "QD_HOST_SIZE": S.HOST_SIZE,
            "QD_SIM_STACK_TOP": S.SP_INIT, "QD_YIELD_BYTE": S.YIELD, "QD_CSR_INPUT": S.CSR_INPUT}
    for name, v in want.items():
        line = next(ln for ln in h.splitlines() if ln.startswith(f"#define {name} "))
        assert int(line.split()[2].rstrip("u"), 0) == v, name


def test_value_records():
    rng = random.Random(1)
    for _ in range(20000):
        v = rng.choice([rng.getrandbits(32), rng.randint(-2**15, 2**15),
                        rng.randint(-2**30, 2**30)]) & 0xFFFFFFFF
        rec = S.value_record(v)
        assert S.decode_value(rec) == v
        s = S.sign32(v)
        assert len(rec) == (1 if -2**14 <= s < 2**14 else 2 if -2**29 <= s < 2**29 else 3)
        assert [S.is_final(t) for t in rec] == [False] * (len(rec) - 1) + [True]


@pytest.mark.parametrize("name", ELFS)
def test_rom_words_vectorized(name):
    """rom_words (a whole image at once, the snapshot's path) == rom_word per instruction."""
    prog = core_elf.program(PROGS / f"{name}.elf")
    words = list(prog.text)
    pcs = [S.ROM_BASE + 4 * i for i in range(len(words))]
    assert ref.rom_words(words).tolist() == [ref.rom_word(w, pc) for w, pc in zip(words, pcs)]


def test_rom_words_random():
    rng = random.Random(5)
    words = [rng.getrandbits(32) for _ in range(30000)]
    for i in range(0, len(words), 3):  # often a real opcode
        words[i] = words[i] & ~0x7F | rng.choice([0x37, 0x17, 0x6F, 0x67, 0x63, 0x03, 0x23,
                                                  0x13, 0x33, 0x0F, 0x73, 0x0B, 0x2B])
    got = ref.rom_words(words).tolist()
    assert got == [ref.rom_word(w, S.ROM_BASE + 4 * i) for i, w in enumerate(words)]


@pytest.mark.parametrize("name", [n for n in ELFS if n.startswith("rv32u")])
def test_riscv_tests_pass(name):
    """The ISA emulator passes the self-checking compliance tests (a0 = 0 at exit)."""
    cpu = run_rv32(name)
    assert cpu.halted and cpu.exit_code == 0, f"failed test {cpu.exit_code}"


C_OUT = {  # computed independently (Python), not by either emulator
    "c-classic": None,  # crc32 / primes checked below
    "c-div64": "acc 11571176827909942706 q 18446743944 r -643091\n",
    "c-switch": "10 9 8 7 6 5 4 3 2 1 109 | switch 109 13 9 4 6 3 1 2 1\n",
    "c-memsweep": None,
}


def c_div64_expected() -> str:
    m = (1 << 64) - 1
    ops = [0, 3, 10, 0xFFFFFFFF, 0x123456789ABCDEF, 1 << 63, m, 0xDEADBEEFCAFEBABE]

    def s64(x):
        return x - (1 << 64) if x >> 63 else x

    def tdiv(a, b):
        q = abs(a) // abs(b)
        return q if (a < 0) == (b < 0) else -q

    acc = 0
    for a in ops:
        for b in ops:
            acc = (acc * 0x100000001B3 + a * b) & m
            if b:
                acc ^= a // b
                acc = (acc + a % b) & m
                sa, sb = s64(a), s64(b)
                if not (sa == -(1 << 63) and sb == -1):
                    q = tdiv(sa, sb)
                    acc ^= q & m
                    acc = (acc - (sa - sb * q)) & m
    r = -(123456789012345 % 1000003)
    return f"acc {acc} q {m // 1000000007} r {r}\n"


def test_c_outputs():
    """C programs print what plain Python computes (the ISA emulator's results are right)."""
    import zlib

    assert run_rv32("c-div64").out.decode() == c_div64_expected() == C_OUT["c-div64"]
    out = run_rv32("c-classic").out.decode()
    fox = b"The quick brown fox jumps over the lazy dog"
    primes = sum(all(i % d for d in range(2, int(i**0.5) + 1)) for i in range(2, 2000))
    assert out.startswith(f"crc32={zlib.crc32(fox):08x} primes={primes} sorted=1")
    assert out.endswith(f"rev={fox[::-1][30:].decode()} atoi=-98765\n")
    assert run_rv32("c-memsweep").out.decode().endswith(" ok\n")
    assert run_rv32("c-recurse").exit_code == 0
    assert run_rv32("c-switch").out.decode() == C_OUT["c-switch"]


@pytest.mark.parametrize("name", ELFS)
def test_machine_equals_isa(name):
    """The traced machine (records, implied PC, pre-decoded ROM, the region map) ends in the ISA
    emulator's state: registers, RAM, output, exit code, instruction count."""
    m = core_check.elf_machine(name, core_check.elf_inputs(name))
    toks = m.run(10**7)
    cpu = run_rv32(name)
    assert m.halted and m.fault is None and cpu.halted
    assert m.x == cpu.x and m.exit_code == cpu.exit_code and m.out == cpu.out
    assert ref.trace_stats(toks)["instructions"] == cpu.steps == m.steps
    ram = np.array([cpu.mem.get(i, 0) for i in range(S.RAM_SIZE)], np.uint8)
    assert ram.view("<u4").tolist() == m.ram


def test_div_records_random():
    """The DIV-class decomposition (k0, digits, result) against the ISA on random and edge
    operands (div_parts asserts quotient digits above k0 are 0 and the result equals alu)."""
    rng = random.Random(3)
    edge = [0, 1, 2, 3, 7, 15, 16, 255, 256, 0x7FFFFFFF, 0x80000000, 0x80000001, 0xFFFFFFFF,
            0xFFFFFFFE, 0xFFFF, 0x10000, 0xFFFFFFF0]
    for _ in range(40000):
        a, b = (rng.choice([rng.choice(edge), rng.getrandbits(32),
                            rng.getrandbits(rng.randint(1, 32))]) for _ in range(2))
        op = rng.choice(sorted(ref.DIV_OPS))
        rec = ref.div_record(op, a, b)
        assert S.decode_value(rec[-3:] if len(rec) >= 3 else rec) == ref.alu(op, a, b)


FAULTS = {  # the qwendoom map's edges
    "illegal": "_start: li a0, 7\n.word 0x0000000b",  # custom-0: the removed digit file
    "illegal (custom-1)": "_start: li a0, 7\n.word 0x0000002b",
    "illegal (csr write)": "_start: li a0, 1\ncsrw mscratch, a0",
    "illegal (time write)": "_start: li a0, 1\ncsrw time, a0",
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
    "ecall": "_start: li a7, 5\necall",
}


@pytest.mark.parametrize("name", list(FAULTS))
def test_faults(name):
    m = ref.Machine.from_program(assemble(FAULTS[name]))
    toks = m.run()
    cause = name.split(" (")[0]
    assert m.fault[0] == S.FAULTS.index(cause)
    assert toks[-10:] == S.fault_record(S.FAULTS.index(cause), m.fault[1])


def test_region_edges_ok():
    """The last word of the RAM and of the DROM load; the RAM's last word stores."""
    src = """_start: li t0, 0x3FFFC
        li a0, -123456
        sw a0, 0(t0)
        lw a1, 0(t0)
        lbu a2, 3(t0)
        li t1, 0xFFFFC
        lw a3, 0(t1)
        lb a4, 3(t1)""" + EXIT
    prog = assemble(src)
    prog.rodata = bytes(S.DROM_SIZE - 4) + bytes([1, 2, 3, 0xF4])
    m = ref.Machine.from_program(prog)
    m.run()
    assert m.fault is None and m.x[11] == (-123456) & 0xFFFFFFFF and m.x[12] == 0xFF
    assert m.x[13] == 0xF4030201 and m.x[14] == 0xFFFFFFF4


def test_input_port():
    """csrr rd, time pops the next input word; its record is the word's value record (marked
    host-injected); csrr x0, time consumes nothing; other counters read 0; an empty queue stops
    before the instruction without committing it."""
    src = """_start: csrr a0, time
        csrr zero, time
        csrr a1, time
        csrr s2, time
        csrr a2, cycle
        csrr a3, instret
        csrr a4, timeh
        mv a5, s2""" + EXIT
    words = [5, 0x12345, 0xDEADBEEF]
    m = ref.Machine.from_program(assemble(src), inputs=words)
    toks = m.run()
    assert (m.x[10], m.x[11], m.x[18], m.x[15]) == (5, 0x12345, 0xDEADBEEF, 0xDEADBEEF)
    assert m.x[12] == m.x[13] == m.x[14] == 0
    assert [toks[i] for i in m.injected] == [t for w in words for t in S.value_record(w)]
    assert len(m.injected) == 1 + 2 + 3
    # an empty queue: InputEmpty, nothing committed; pushing a word resumes
    m = ref.Machine.from_program(assemble(src), inputs=[1])
    m.record()
    m.record()
    pc, n = m.pc, m.n
    with pytest.raises(ref.InputEmpty):
        m.record()
    assert (m.pc, m.n, m.x[11]) == (pc, n, 0)
    m.inputs.append(-2)
    assert m.record() == S.value_record(-2) and m.last_input and m.x[11] == 0xFFFFFFFE


def test_yield_and_putchar():
    src = """_start: li a7, 1
        li a0, 72
        ecall
        li a0, 256
        ecall
        li a0, 0x141
        ecall""" + EXIT
    m = ref.Machine.from_program(assemble(src))
    toks = m.run()
    assert [S.family(t) for t in toks if S.family(t)[0] == "BYTE"] == [
        ("BYTE", 72), ("BYTE", S.YIELD), ("BYTE", 0x141)]
    assert m.out == b"H" and m.yields == 1


def test_from_images_matches_from_program():
    prog = core_elf.program(PROGS / "c-funcptr.elf")
    a = ref.Machine.from_program(prog)
    ram = bytes(prog.data_base - S.RAM_BASE) + prog.data
    rom = b"".join(w.to_bytes(4, "little") for w in prog.text)
    b = ref.Machine.from_images(ram.ljust(S.RAM_SIZE, b"\0"), prog.rodata, rom, a.x, a.pc)
    assert a.run() == b.run() and a.out == b.out
