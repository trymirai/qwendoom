"""The trace decoder (the mirror, native/trace/qd_decoder.c): reference tokens in, machine state
out. Round trips against the C reference (every record) and against upstream's Python machine
(independent of the C one), snapshot starts, input injection, and its token checks: every error
code on a corrupted stream, strict against lenient."""

import random
import subprocess
from pathlib import Path

import numpy as np
import pytest

from chipformer.rv import asm, elf, fuzz, ref
from chipformer.rv import spec as S
from qwendoom import trace as T

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "third_party" / "qwenputer" / "experimental" / "riscv" / "tests" / "data" / "rv"
ELFS = sorted(p.stem for p in DATA.glob("*.elf") if not p.name.endswith(".qemu.elf"))
UP = T.Map.upstream()
CLANG = Path("/opt/homebrew/opt/llvm/bin/clang")
LLD = Path("/opt/homebrew/bin/ld.lld")


def pair(load, map_: T.Map, sp: int, strict: bool = True):
    """A reference machine and a decoder on separate memories with the same image."""
    rmem = T.Memory(map_)
    entry = load(rmem)
    dmem = rmem.copy()
    regs = [0] * 32
    regs[2] = sp
    r, d = T.Ref(rmem), T.Decoder(dmem, strict=strict)
    r.set_context(pc=entry, regs=regs)
    d.set_context(pc=entry, regs=regs)
    return r, d


def elf_pair(name, strict=True):
    return pair(lambda m: m.load_elf(DATA / f"{name}.elf"), UP, S.SP_INIT, strict)


def prog_pair(prog, map_=UP, sp=S.SP_INIT, strict=True):
    return pair(lambda m: m.load_program(prog), map_, sp, strict)


# --- (c) round trips -----------------------------------------------------------------------------


@pytest.mark.parametrize("strict", [True, False])
@pytest.mark.parametrize("name", ELFS)
def test_roundtrip_elf(name, strict):
    """Every upstream ELF: after every record the decoder's anchor, n and registers equal the
    reference's, every store lands on the same word with the same value, and RAM is equal at
    the end (qd_roundtrip)."""
    r, d = elf_pair(name, strict)
    n = T.roundtrip(r, d)
    assert n == r.counts["records"] and d.counts["tokens"] == r.counts["tokens"]
    assert d.halted in (T.HALT, T.FAULT)


@pytest.mark.parametrize("seed", range(60))
def test_roundtrip_fuzz(seed):
    r, d = prog_pair(asm.assemble(fuzz.gen_program(seed, block=40, iters=10)))
    T.roundtrip(r, d)
    assert d.halted == T.HALT


@pytest.mark.parametrize("name", ["c-crc32p", "rv32ui-sb", "rv32ui-sh", "rv32um-div", "c-sort"])
def test_roundtrip_against_python(name):
    """Upstream's Python machine's tokens through the decoder, record by record: the registers
    equal the Python machine's after every record, the stored word after every store, and all of
    RAM at the end (no C reference involved)."""
    m = ref.Machine.from_program(elf.program(DATA / f"{name}.elf"))
    mem = T.Memory(UP)
    entry = mem.load_elf(DATA / f"{name}.elf")
    d = T.Decoder(mem)
    d.set_context(pc=entry, regs=m.x)
    while not m.halted:
        rec = m.record()
        for i, t in enumerate(rec):
            res = d.feed(t)
            assert res >= 0, d.error_text
            assert (res == T.MORE) == (i < len(rec) - 1)
        a, n, x = d.context
        assert (a, n, x) == (m.anchor, m.n, m.x)
        fam = S.family(rec[-1])[0]
        if fam in ("ST0", "ST1", "SS0", "SS1"):
            st = d.last_store
            assert st is not None and mem.word(st) == m.ram[st >> 2]
        else:
            assert d.last_store is None
    ram = np.frombuffer(mem.ram.tobytes(), "<u4").astype(np.int64).tolist()
    assert ram == m.ram
    assert d.halted == T.HALT and d.exit_code == m.exit_code


def test_snapshot_start():
    """A decoder started mid-run from a context (anchor, n > 0, registers) and the RAM at that
    point decodes the rest without the earlier tokens."""
    prog = asm.assemble(fuzz.gen_program(9, block=40, iters=10))
    r, _ = prog_pair(prog)
    full, _, _ = r.trace()
    for k in (3, 63, 64, 100, 700):
        r, _ = prog_pair(prog)
        r.run(max_records=k)
        a, n, x = r.context
        dmem = r.mem.copy()
        d = T.Decoder(dmem)
        d.set_context(anchor=a, n=n, regs=x)
        rest, _, _ = r.trace()
        assert d.feed_checked(rest) == T.HALT
        assert d.context == r.context and np.array_equal(dmem.ram, r.mem.ram)
        assert d.counts["tokens"] == len(rest) < len(full)


# --- (d) input injection ---------------------------------------------------------------------

POKES = """
_start: csrr a0, time       # the host's pokes: (addr, word) pairs until addr 0
        beqz a0, tick
        csrr a1, time
        sw a1, 0(a0)
        j _start
tick:   lw t0, 0x300(zero)  # the tick: count, and store a byte
        addi t0, t0, 1
        sw t0, 0x300(zero)
        sb t0, 0x305(zero)
        li a0, 256
        li a7, 1
        ecall               # YIELD
        j _start
"""


@pytest.fixture
def qd_asm(monkeypatch):
    """Upstream's assembler at qwendoom's addresses."""
    monkeypatch.setattr(asm, "ROM_BASE", 0x100000)
    monkeypatch.setattr(asm, "DROM_BASE", 0x80000)


def model_run(r: T.Ref, d: T.Decoder, ticks):
    """Drive a decoder as the runner drives the model: at every record boundary where the
    decoder wants input, inject the next host word's value record (in place of the model's
    prediction); otherwise take the model's tokens (here the reference's, which has the same
    words queued). Returns the decoder's events per tick."""
    events = []
    for pokes in ticks:
        words = [w for a, v in pokes for w in (a, v)] + [0]
        r.push_input(words)
        host = list(words)
        evs = []
        while True:
            want_in = d.wants_input
            assert want_in == r.wants_input
            rec, owned = r.record()
            assert owned == want_in and rec
            if want_in:
                rec = T.encode_value_record(host.pop(0))
            for t in rec:
                ev = d.feed(t)
                assert ev >= 0, d.error_text
            evs.append(ev)
            if ev == T.YIELD:
                break
        assert not host and r.input_pending == 0
        events.append(evs)
    return events


def test_input_injection(qd_asm):
    prog = asm.assemble(POKES)
    r, d = prog_pair(prog, T.Map.qwendoom(), 0x3000)
    ticks = [[(0x104, 5), (0x200, 0xDEADBEEF), (0x108, -3)], [], [(0x3FFFC, 1 << 20)] * 2]
    events = model_run(r, d, ticks)
    mem = d.mem
    assert mem.word(0x104) == 5 and mem.word(0x200) == 0xDEADBEEF and mem.word(0x108) == 2**32 - 3
    assert mem.word(0x3FFFC) == 1 << 20 and mem.word(0x300) == 3 and mem.read(0x305, 1) == b"\x03"
    assert np.array_equal(d.mem.ram, r.mem.ram) and d.context == r.context
    assert [e.count(T.YIELD) for e in events] == [1, 1, 1]
    # after the jump back the sim reads input again: the decoder says so, the reference waits
    rec, _ = r.record()
    assert not d.wants_input and d.feed_many(rec) == (T.COMMIT, 1)
    assert d.wants_input and r.wants_input
    assert r.run()[2] == "input"


def test_input_value_sizes(qd_asm):
    """Injected words of every value-record size (F0, V F1, V V F2) reach the mirror."""
    prog = asm.assemble(POKES)
    r, d = prog_pair(prog, T.Map.qwendoom(), 0x3000)
    vals = [0, 1, -1, 2**14 - 1, -2**14, 2**14, 2**29 - 1, -2**29, 2**29, 2**31 - 1, -2**31]
    pokes = [(0x1000 + 4 * i, v) for i, v in enumerate(vals)]
    model_run(r, d, [pokes])
    assert [d.mem.word(a) for a, _ in pokes] == [v & 0xFFFFFFFF for v in vals]
    assert sorted({len(T.encode_value_record(v)) for v in vals}) == [1, 2, 3]


def test_input_from_snapshot_and_feed_many(qd_asm):
    """feed_many with DEC_STOP_INPUT stops right before each input record."""
    prog = asm.assemble(POKES)
    r, d = prog_pair(prog, T.Map.qwendoom(), 0x3000)
    r.push_input([0x104, 77, 0])
    toks, flags, why = r.run(stop_yield=True)
    assert why == "yield"
    i, stops = 0, int(d.wants_input)  # (the first record reads input)
    while i < len(toks):
        res, used = d.feed_many(toks[i:], stop=T.DEC_STOP_INPUT)
        assert res >= 0, d.error_text
        i += used
        if d.wants_input:
            stops += 1
            assert flags[i] & T.TF_HOST and flags[i] & T.TF_FIRST
    assert stops == 3 and res == T.YIELD and d.mem.word(0x104) == 77


def model_setup(prog, pc: int | None = None):
    """A RefModel (its own memory), a decoder (the mirror) and a checker whose reference takes
    its input from the stream, all at pc (default: the program's entry) on qwendoom's map."""
    regs = [0, 0, 0x3000] + [0] * 29
    mems = [T.Memory(T.Map.qwendoom()) for _ in range(3)]
    entry = [m.load_program(prog) for m in mems][0]
    model_ref, dec, cref = T.Ref(mems[0]), T.Decoder(mems[1]), T.Ref(mems[2])
    for x in (model_ref, dec, cref):
        x.set_context(pc=entry if pc is None else pc, regs=regs)
    return model_ref, dec, T.Checker(cref), mems


def test_drive_tick(qd_asm):
    """drive_tick with a perfect model: the input records are injected (the model's F0(0) stubs
    replaced), the mirror and the checker's reference end every tick with the model's RAM."""
    prog = asm.assemble(POKES)
    model_ref, dec, chk, mems = model_setup(prog)
    model = T.RefModel(model_ref)
    pred = T.start_stream(model.step, dec, chk)  # GO to the decoder, the checker and the model
    assert pred == S.tok("F0", 0) and dec.wants_input
    ticks = [[0x104, 5, 0x200, 0xDEADBEEF, 0], [0], [0x3FFFC, 1 << 20, 0x108, -2**31, 0]]
    for words in ticks:
        model_ref.push_input(words)
        tick = T.drive_tick(model.step, pred, dec, words, chk)
        pred = tick.pred
        assert tick.event == T.YIELD and not tick.unread
        injected = [t for w in words for t in T.encode_value_record(w)]
        assert tick.tokens[tick.host].tolist() == injected
        assert np.array_equal(mems[1].ram, mems[0].ram) and np.array_equal(mems[2].ram, mems[0].ram)
        assert dec.context == chk.ref.context  # (the model's reference runs a record ahead)
    assert mems[1].word(0x200) == 0xDEADBEEF and mems[1].word(0x108) == 2**31
    assert mems[1].word(0x300) == 3
    assert chk.counts["host_tokens"] == sum(len(T.encode_value_record(w)) for t in ticks
                                            for w in t) == 17

    # too few words: the sim reads input with none left
    model_ref.push_input([0x104])
    with pytest.raises(T.DecodeError, match="no word left"):
        T.drive_tick(model.step, pred, dec, [0x104], chk)


def test_drive_tick_catches_a_wrong_prediction(qd_asm):
    """A wrong value the decoder cannot see (an rd value) is the checker's: CheckError at its
    exact stream index (GO is token 0); a grammar violation is the decoder's (DecodeError)."""
    prog = asm.assemble(POKES)
    words = [0x104, 77, 0]
    model_ref, dec, chk, _ = model_setup(prog)
    model_ref.push_input(words)
    model = T.RefModel(model_ref)
    good = T.drive_tick(model.step, T.start_stream(model.step, dec), dec, words)
    # the load of 0x300 (a model-owned F0 after the input records)
    k = next(i for i, (t, h) in enumerate(zip(good.tokens, good.host))
             if not h and S.family(int(t))[0] == "F0")
    model_ref, dec, chk, _ = model_setup(prog)
    model_ref.push_input(words)
    model = T.RefModel(model_ref, corrupt={k: int(good.tokens[k]) ^ 1})
    with pytest.raises(T.CheckError) as e:
        T.drive_tick(model.step, T.start_stream(model.step, dec, chk), dec, words, chk)
    assert e.value.divergence.index == k + 1 and e.value.divergence.want == good.tokens[k]
    model_ref, dec, chk, _ = model_setup(prog)
    model_ref.push_input(words)
    model = T.RefModel(model_ref, corrupt={k: S.tok("NT")})  # NT where a value record is due
    with pytest.raises(T.DecodeError, match="shape"):
        T.drive_tick(model.step, T.start_stream(model.step, dec), dec, words)


def test_drive_tick_go_after_go(qd_asm):
    """start_stream feeds GO to the decoder and the checker as well as to the model, so the
    model's first prediction is stream token 1, where a GO is an error. A model that predicts GO
    after GO fails the tick; with GO in the model only (a fresh decoder and checker take a GO at
    token 0 as the start of a run) it passed. A normal tick's tokens are the same either way."""
    prog = asm.assemble(POKES)
    at = prog.labels["tick"]  # (a model-owned first record: at _start the host injects input)

    def start(go_to_all: bool, pc=at, words=(), corrupt=None):
        model_ref, dec, chk, _ = model_setup(prog, pc)
        model_ref.push_input(words)
        model = T.RefModel(model_ref, corrupt)
        pred = T.start_stream(model.step, dec, chk) if go_to_all else model.step(T.GO)
        return model, pred, dec, chk

    model, pred, dec, chk = start(True, corrupt={0: T.GO})  # the model predicts GO after GO
    assert pred == T.GO
    with pytest.raises(T.DecodeError, match="shape"):
        T.drive_tick(model.step, pred, dec, [], chk)
    model, _, dec, chk = start(True)
    with pytest.raises(T.DecodeError, match="shape"):
        T.drive_tick(model.step, T.GO, dec, [], chk)  # GO forced as the first prediction
    _, _, _, chk = start(True)
    assert chk.feed(T.GO) == T.CHK_DIVERGED and chk.divergence.index == 1  # the checker as well
    model, _, dec, chk = start(False)  # the old start: the same forced GO passed
    tick = T.drive_tick(model.step, T.GO, dec, [], chk)
    assert tick.event == T.YIELD and tick.tokens[0] == T.GO and chk.state == T.CHK_OK

    # a normal tick (input injected): the same tokens and host mask either way
    words = [0x104, 77, 0]
    ticks = []
    for go_to_all in (True, False):
        model, pred, dec, chk = start(go_to_all, None, words)
        ticks.append(T.drive_tick(model.step, pred, dec, words, chk))
        assert ticks[-1].event == T.YIELD and chk.state == T.CHK_OK
    assert np.array_equal(ticks[0].tokens, ticks[1].tokens) and len(ticks[0].tokens) > 10
    assert np.array_equal(ticks[0].host, ticks[1].host)
    assert ticks[0].host.sum() == sum(len(T.encode_value_record(w)) for w in words)


SIM_C = r"""
static inline unsigned qd_in(void) {
    unsigned v;
    __asm__ volatile("csrr %0, time" : "=r"(v));
    return v;
}
static inline void qd_yield(void) {
    register long a0 __asm__("a0") = 256, a7 __asm__("a7") = 1;
    __asm__ volatile("ecall" : : "r"(a0), "r"(a7) : "memory");
}
static const unsigned short mix[8] = {3, 1, 4, 1, 5, 9, 2, 6};
unsigned counter, table[32];
void sim_main(void) {
    for (;;) {
        for (;;) {
            unsigned a = qd_in();
            if (!a) break;
            *(volatile unsigned *)a = qd_in();
        }
        counter++;
        for (int i = 0; i < 32; i++)
            table[i] = table[i] * mix[i & 7] + counter / (i + 1) + (unsigned char)i;
        qd_yield();
    }
}
"""
LINK = """ENTRY(_start)
MEMORY {
  RAM (rw) : ORIGIN = 0x3000, LENGTH = 0x3D000
  DROM (r) : ORIGIN = 0x80000, LENGTH = 512K
  ROM (rx) : ORIGIN = 0x100000, LENGTH = 256K
}
SECTIONS {
  .text : { *(.text.start) *(.text .text.*) } > ROM
  .rodata : { *(.rodata .rodata.* .srodata .srodata.*) } > DROM
  .data : { *(.data .data.* .sdata .sdata.*) } > RAM
  .bss : { *(.bss .bss.* .sbss .sbss.*) *(COMMON) } > RAM
}
"""
START = """.section .text.start
.globl _start
_start: li sp, 0x3000
        call sim_main
"""


@pytest.mark.skipif(not (CLANG.exists() and LLD.exists()), reason="no RISC-V clang / ld.lld")
def test_clang_sim_main(tmp_path):
    """A clang-built sim_main poke loop (as ARCHITECTURE.md's) on qwendoom's map: model-style
    injection through the decoder; the mirror equals the reference's RAM after every tick."""
    (tmp_path / "sim.c").write_text(SIM_C)
    (tmp_path / "start.S").write_text(START)
    (tmp_path / "link.ld").write_text(LINK)
    cc = [str(CLANG), "--target=riscv32-unknown-elf", "-march=rv32im_zba", "-mabi=ilp32", "-O2",
          "-ffreestanding", "-nostdlib", f"-fuse-ld={LLD}", "-T", str(tmp_path / "link.ld"),
          "-o", str(tmp_path / "sim.elf"), str(tmp_path / "start.S"), str(tmp_path / "sim.c")]
    subprocess.run(cc, check=True, capture_output=True)
    path = tmp_path / "sim.elf"
    r, d = pair(lambda m: m.load_elf(path), T.Map.qwendoom(), 0)
    syms = subprocess.run([str(CLANG.parent / "llvm-nm"), str(path)], capture_output=True,
                          text=True).stdout if (CLANG.parent / "llvm-nm").exists() else ""
    counter = next((int(s.split()[0], 16) for s in syms.splitlines() if s.endswith(" counter")),
                   None)
    rng = random.Random(3)
    ticks = [[(4 * a, rng.getrandbits(32))
              for a in rng.sample(range(0x400, 0x800), rng.randrange(5))]
             for _ in range(20)]
    for t in ticks:
        model_run(r, d, [t])
        assert np.array_equal(d.mem.ram, r.mem.ram) and d.context == r.context
        assert all(d.mem.word(a) == v for a, v in t)
    assert d.counts["records"] > 20 * 200
    if counter is not None:
        assert d.mem.word(counter) == len(ticks)


# --- token checks: every error code ------------------------------------------------------------


def stream(src: str, map_=UP):
    """(decoder over a fresh memory, the reference's tokens with per-token flags) of a program."""
    prog = asm.assemble(src)
    r, d = prog_pair(prog, map_)
    toks, flags, _ = r.trace()
    return d, toks.tolist(), flags


def feed_until(d: T.Decoder, toks, upto: int):
    for t in toks[:upto]:
        assert d.feed(t) >= 0, d.error_text


PROG = """_start: li a0, 5
    li a1, 7
    add a2, a0, a1
    beq a0, a1, out
    jal ra, f
    sw a2, 0x140(zero)
    sh a1, 0x102(zero)
    divu a3, a2, a0
    li a7, 1
    ecall
    li a7, 93
    ecall
f:  ret
out: ebreak
"""


def idx(toks, name_, k=0):
    """The position of the k-th token of family name_."""
    return [i for i, t in enumerate(toks) if S.family(t)[0] == name_][k]


def corrupt(toks, at, new):
    return toks[:at] + list(new) + toks[at + 1:]


V, F0, F1, NT, J = (lambda p: S.tok("V", p)), (lambda p: S.tok("F0", p)), \
    (lambda p: S.tok("F1", p)), S.tok("NT"), (lambda p: S.tok("J", p))
CASES = {  # name: (position, replacement tokens, error, raised at +k, also when lenient)
    "pad": (lambda t: 0, [S.VOCAB], -1, 0, True),
    "nt for a value": (lambda t: 2, [NT], -2, 0, True),
    "f1 alone": (lambda t: 0, [F1(5)], -2, 0, True),
    "non-canonical": (lambda t: 0, [V(5), F1(0)], -4, 1, True),  # 5 is a 1-token value
    "v v f0": (lambda t: 0, [V(5), V(0), F0(1)], -4, 2, True),
    "branch taken": (lambda t: idx(t, "NT"), [J(13)], -6, 0, False),  # beq 5, 7 is not taken
    "jal target": (lambda t: idx(t, "J"), [J(3)], -6, 0, True),  # (static: always checked)
    "store bank": (lambda t: idx(t, "ST1"), [S.tok("ST0")], -7, 0, False),
    "merged word": (lambda t: idx(t, "SS0") - 3, [V(1)], -7, 3, False),
    "merged top chunk": (lambda t: idx(t, "SS0") - 1, [V(4)], -4, 0, True),
    "div k0": (lambda t: idx(t, "QS"), [S.tok("QS", 0)], -5, 0, False),
    "qs > 8": (lambda t: idx(t, "QS"), [S.tok("QS", 9)], -5, 0, True),
    "qd index": (lambda t: idx(t, "QD"), [S.tok("QD", 0x35)], -5, 0, True),
    "byte payload": (lambda t: idx(t, "BYTE"), [S.tok("BYTE", 6)], -9, 0, False),
    "halt for byte": (lambda t: idx(t, "BYTE"), [S.tok("HALT")], -9, 0, False),
    "fault at add": (lambda t: 2, [S.tok("FLT", 0)], -8, 0, True),
    "after halt": (lambda t: len(t), [NT], -11, 0, True),
}


@pytest.mark.parametrize("strict", [True, False])
@pytest.mark.parametrize("case", list(CASES))
def test_error_codes(case, strict):
    """A corrupted stream fails at the right token with the right code (and stays failed); in
    lenient mode the register-dependent checks let the token through."""
    pos, repl, code, k, lenient_too = CASES[case]
    d, toks, _ = stream(PROG)
    if not strict:
        d = T.Decoder(d.mem, strict=False)
        d.set_context(pc=S.ROM_BASE, regs=[0, 0, S.SP_INIT] + [0] * 29)
    at = pos(toks)
    bad = corrupt(toks, at, repl) if at < len(toks) else toks + repl
    if strict or lenient_too:
        res, used = d.feed_many(bad)
        if res == T.HALT:  # (feed_many stops at the end; the next token is refused)
            res = d.feed(bad[used])
        assert (res, used) == (code, at + k), (res, used, at, d.error_text)
        assert d.error == code and d.error_text
        assert d.feed(toks[0]) == code  # sticky
    else:
        res, used = d.feed_many(bad[:at + k + 1])
        assert res >= 0 and used == at + k + 1, d.error_text


def test_refresh_errors():
    body = "\n".join(f"addi a0, a0, {i}" for i in range(70))
    d, toks, _ = stream(f"_start: {body}\n li a7, 93\n ecall")
    r_at = idx(toks, "R")
    for bad in (S.tok("NT"), S.tok("R", 5), S.tok("F0", 1)):
        d2 = T.Decoder(d.mem)
        d2.set_context(pc=S.ROM_BASE, regs=[0, 0, S.SP_INIT] + [0] * 29)
        res, used = d2.feed_many(corrupt(toks, r_at, [bad]))
        assert res == -3 and used == r_at, d2.error_text
    d.feed_checked(toks)


def test_fault_record_checks():
    d, toks, _ = stream("_start: li a0, 0x102\n lw a1, 0(a0)")
    assert S.family(toks[-10])[0] == "FLT"
    res, used = d.feed_many(corrupt(toks, len(toks) - 3, [S.tok("HEX", 15)]))  # a wrong PC digit
    assert res == -8
    d, toks, _ = stream("_start: li a0, 0x102\n lw a1, 0(a0)")
    res, used = d.feed_many(corrupt(toks, len(toks) - 10, [S.tok("FLT", 4)]))  # (strict) cause
    assert res == -8
    d, toks, _ = stream("_start: li a0, 0x102\n lw a1, 0(a0)")
    assert d.feed_checked(toks) == T.FAULT and d.fault == (2, S.ROM_BASE + 4)


def test_store_outside_ram_lenient():
    """A lenient decoder still refuses to apply a store it cannot place."""
    d, toks, _ = stream("_start: li a0, 0x40000\n sw a1, 4(a0)")  # faults: store out of range
    d = T.Decoder(d.mem, strict=False)
    d.set_context(pc=S.ROM_BASE, regs=[0, 0, S.SP_INIT] + [0] * 29)
    at = idx(toks, "FLT")
    assert d.feed_many(toks[:at] + [S.tok("ST0")]) == (-7, at)


def test_pc_outside_rom():
    mem = T.Memory(UP)
    d = T.Decoder(mem)
    d.set_context(anchor=1 << 16)
    assert d.feed(S.tok("NT")) == -10
    d.set_context(anchor=0, n=64)
    assert d.feed(S.tok("NT")) == -3


def test_lenient_follows_the_model():
    """Lenient mode takes a branch the reference did not (the model's word); strict refuses."""
    d, toks, _ = stream(PROG)
    at = idx(toks, "NT")  # beq a0, a1 (not taken)
    bad = corrupt(toks, at, [S.tok("J", 13)])[:at + 1]
    d.set_context(pc=S.ROM_BASE, regs=[0, 0, S.SP_INIT] + [0] * 29)
    assert d.feed_many(bad)[0] == -6
    lz = T.Decoder(d.mem, strict=False)
    lz.set_context(pc=S.ROM_BASE, regs=[0, 0, S.SP_INIT] + [0] * 29)
    assert lz.feed_many(bad) == (T.COMMIT, len(bad)) and lz.pc == S.ROM_BASE + 4 * 13
