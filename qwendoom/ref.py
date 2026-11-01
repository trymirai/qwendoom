"""Reference RV32IM + Zba: an ISA emulator, the pre-decoded ROM, and the trace (records) the
circuit must produce.

Derived from Qwenputer's chipformer/rv/ref.py (MIT, (c) 2026 Divyansh Agrawal; see
LICENSE.chipformer), without the digit-file coprocessor (custom-0 / custom-1 are illegal
instructions here) and the P3 text buffer, with qwendoom's address map and its input port.

``Rv32``       plain RV32IM + Zba semantics with a flat byte memory (the cross-check).
``rom_word``   the circuit's view of an instruction: one pre-decoded 32-bit ROM word (see below);
               ``rom_words`` is the same for a whole image, vectorized.
``Machine``    the traced machine: ``record()`` returns the next record's tokens and commits it.

The input port: ``csrr rd, time`` (CSR 0xC01, rd != x0) reads the next word of ``inputs``, and its
record is that word's value record. The circuit folds the CSR to ``rd = 0`` like every CSR stub
(it predicts F0(0)); the host injects the real record, so those tokens are host-owned
(``Machine.last_input`` after ``record()``, ``Machine.injected`` after ``run()``). A read into x0
consumes nothing (its record is NT, like any rd = x0 write).

The ROM word is four bytes, each a field the circuit decodes in one MLP stage:

  b0 = uop | S0 << 7    b1 = RD << 3 | S1..S3    b2 = RS1 << 3 | S4..S6    b3 = RS2 << 3 | S7..S9

RD / RS1 / RS2 are 5-bit register slots (the regfile's three read ports query them) and S0..S9
are spare bits. Static facts are folded in, so every format fits:

  R-type (ALU, MUL, DIV, Zba)   RD = rd, RS1 = rs1, RS2 = rs2
  I-type (ALU imm, loads, JALR)  RD = rd, RS1 = rs1, imm12 = RS2 (bits 0..4) + S0..S6 (bits 5..11)
  stores                         RD = the data register, RS1 = base, imm12 as I-type
  branches                       RS1, RS2; the target as a signed word offset: RD (bits 0..4) +
                                 S0..S5 (bits 5..10); S6 = the target is bad (misaligned / not ROM)
  JAL                            RD = rd; the target's ROM word index: RS1 (0..4) + RS2 (5..9) +
                                 S0..S5 (10..15); S6 = bad
  LUI / AUIPC                    RD = rd; the value's upper 20 bits: RS1 (0..4) + RS2 (5..9) +
                                 S0..S9 (10..19); AUIPC's are (pc + imm) >> 12 (its low 12 bits
                                 are the PC's, in-circuit)
  CSR reads (stubs)              LUI rd, 0 (time, the input port, included), or MISA rd;
                                 anything else ILLEGAL
  ECALL                          RS1 = 17 (a7), RS2 = 10 (a0)
"""

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from qwendoom import spec as S

UOPS = ["ILLEGAL", "LUI", "AUIPC", "MISA",
        "ADDI", "SLTI", "SLTIU", "XORI", "ORI", "ANDI", "SLLI", "SRLI", "SRAI",
        "ADD", "SUB", "SLL", "SLT", "SLTU", "XOR", "SRL", "SRA", "OR", "AND",
        "LB", "LH", "LW", "LBU", "LHU", "SB", "SH", "SW",
        "BEQ", "BNE", "BLT", "BGE", "BLTU", "BGEU", "JAL", "JALR",
        "ECALL", "EBREAK", "FENCE",
        "MUL", "MULH", "MULHSU", "MULHU", "DIV", "DIVU", "REM", "REMU",
        "SH1ADD", "SH2ADD", "SH3ADD"]  # upstream's uop numbers (64.. were the digit-file ops)
CSR_STUBS = {0xC00: 0, 0xC01: 0, 0xC02: 0, 0xC80: 0, 0xC81: 0, 0xC82: 0, 0xF14: 0,
             0x301: S.MISA}  # cycle, time (the input port), instret (+h), mhartid, misa
U = {n: i for i, n in enumerate(UOPS)}
WRITES_RD = {"LUI", "AUIPC", "MISA", "ADDI", "SLTI", "SLTIU", "XORI", "ORI", "ANDI", "SLLI",
             "SRLI", "SRAI", "ADD", "SUB", "SLL", "SLT", "SLTU", "XOR", "SRL", "SRA", "OR", "AND",
             "LB", "LH", "LW", "LBU", "LHU", "JAL", "JALR",
             "MUL", "MULH", "MULHSU", "MULHU", "DIV", "DIVU", "REM", "REMU",
             "SH1ADD", "SH2ADD", "SH3ADD"}
MUL_OPS = {"MUL", "MULH", "MULHSU", "MULHU"}
DIV_OPS = {"DIV", "DIVU", "REM", "REMU"}
ZBA = {"SH1ADD": 1, "SH2ADD": 2, "SH3ADD": 3}
IMM_OPS = {"ADDI", "SLTI", "SLTIU", "XORI", "ORI", "ANDI", "SLLI", "SRLI", "SRAI"}
BRANCH_OPS = {"BEQ", "BNE", "BLT", "BGE", "BLTU", "BGEU"}
LOAD_OPS = {"LB", "LH", "LW", "LBU", "LHU"}
STORE_OPS = {"SB", "SH", "SW"}
M32 = 0xFFFFFFFF


def sx(v: int, bits: int) -> int:
    v &= (1 << bits) - 1
    return v - (1 << bits) if v >> (bits - 1) else v


@dataclass
class Inst:
    op: str
    rd: int = 0
    rs1: int = 0
    rs2: int = 0
    imm: int = 0  # sign-extended immediate (branch/jump: byte offset; CSR: the CSR number)


def decode(w: int) -> Inst:
    opc, rd, f3 = w & 0x7F, w >> 7 & 31, w >> 12 & 7
    rs1, rs2, f7 = w >> 15 & 31, w >> 20 & 31, w >> 25
    i_imm = sx(w >> 20, 12)
    s_imm = sx((w >> 25) << 5 | (w >> 7 & 31), 12)
    b_imm = sx((w >> 31 & 1) << 12 | (w >> 7 & 1) << 11 | (w >> 25 & 63) << 5
               | (w >> 8 & 15) << 1, 13)
    j_imm = sx((w >> 31 & 1) << 20 | (w >> 12 & 0xFF) << 12 | (w >> 20 & 1) << 11
               | (w >> 21 & 0x3FF) << 1, 21)
    if opc == 0x37:
        return Inst("LUI", rd, imm=sx(w & 0xFFFFF000, 32))
    if opc == 0x17:
        return Inst("AUIPC", rd, imm=sx(w & 0xFFFFF000, 32))
    if opc == 0x6F:
        return Inst("JAL", rd, imm=j_imm)
    if opc == 0x67 and f3 == 0:
        return Inst("JALR", rd, rs1, imm=i_imm)
    if opc == 0x63 and f3 not in (2, 3):
        return Inst(["BEQ", "BNE", "", "", "BLT", "BGE", "BLTU", "BGEU"][f3], 0, rs1, rs2, b_imm)
    if opc == 0x03 and f3 in (0, 1, 2, 4, 5):
        return Inst(["LB", "LH", "LW", "", "LBU", "LHU"][f3], rd, rs1, imm=i_imm)
    if opc == 0x23 and f3 in (0, 1, 2):
        return Inst(["SB", "SH", "SW"][f3], 0, rs1, rs2, s_imm)
    if opc == 0x13:
        if f3 == 1 and f7 == 0:
            return Inst("SLLI", rd, rs1, imm=rs2)
        if f3 == 5 and f7 in (0, 0x20):
            return Inst("SRAI" if f7 else "SRLI", rd, rs1, imm=rs2)
        if f3 not in (1, 5):
            return Inst(["ADDI", "", "SLTI", "SLTIU", "XORI", "", "ORI", "ANDI"][f3], rd, rs1,
                        imm=i_imm)
    if opc == 0x33:
        if f7 == 0:
            return Inst(["ADD", "SLL", "SLT", "SLTU", "XOR", "SRL", "OR", "AND"][f3], rd, rs1, rs2)
        if f7 == 0x20 and f3 in (0, 5):
            return Inst("SUB" if f3 == 0 else "SRA", rd, rs1, rs2)
        if f7 == 1:
            return Inst(["MUL", "MULH", "MULHSU", "MULHU", "DIV", "DIVU", "REM", "REMU"][f3], rd,
                        rs1, rs2)
        if f7 == 0x10 and f3 in (2, 4, 6):
            return Inst({2: "SH1ADD", 4: "SH2ADD", 6: "SH3ADD"}[f3], rd, rs1, rs2)
    if w == 0x00000073:
        return Inst("ECALL")
    if opc == 0x73 and f3 in (1, 2, 3, 5, 6, 7):
        return Inst(["", "CSRRW", "CSRRS", "CSRRC", "", "CSRRWI", "CSRRSI", "CSRRCI"][f3], rd, rs1,
                    imm=w >> 20)
    if w == 0x00100073:
        return Inst("EBREAK")
    if opc == 0x0F:
        return Inst("FENCE")
    return Inst("ILLEGAL")


def alu(op: str, a: int, b: int) -> int:
    """Register-register / register-immediate results (a, b as unsigned 32-bit)."""
    sa, sb = sx(a, 32), sx(b, 32)
    if op in ("ADD", "ADDI"):
        r = a + b
    elif op == "SUB":
        r = a - b
    elif op in ("SLL", "SLLI"):
        r = a << (b & 31)
    elif op in ("SRL", "SRLI"):
        r = a >> (b & 31)
    elif op in ("SRA", "SRAI"):
        r = sa >> (b & 31)
    elif op in ("SLT", "SLTI"):
        r = int(sa < sb)
    elif op in ("SLTU", "SLTIU"):
        r = int(a < b)
    elif op in ("XOR", "XORI"):
        r = a ^ b
    elif op in ("OR", "ORI"):
        r = a | b
    elif op in ("AND", "ANDI"):
        r = a & b
    elif op == "MUL":
        r = a * b
    elif op == "MULH":
        r = (sa * sb) >> 32
    elif op == "MULHSU":
        r = (sa * b) >> 32
    elif op == "MULHU":
        r = (a * b) >> 32
    elif op == "DIV":
        r = -1 if sb == 0 else (sa if (sa, sb) == (-1 << 31, -1) else int(sa / sb))
    elif op == "DIVU":
        r = M32 if b == 0 else a // b
    elif op == "REM":
        r = sa if sb == 0 else (0 if (sa, sb) == (-1 << 31, -1) else sa - sb * int(sa / sb))
    elif op == "REMU":
        r = a if b == 0 else a % b
    elif op in ZBA:
        r = (a << ZBA[op]) + b
    else:
        raise ValueError(op)
    return r & M32


def taken(op: str, a: int, b: int) -> bool:
    sa, sb = sx(a, 32), sx(b, 32)
    return {"BEQ": a == b, "BNE": a != b, "BLT": sa < sb, "BGE": sa >= sb,
            "BLTU": a < b, "BGEU": a >= b}[op]


def nibbles_used(v: int) -> int:
    """How many of the 8 nibble prefixes v >> 4i (i = 0..7) are nonzero."""
    return sum(1 for i in range(8) if v >> 4 * i)


def div_start(op: str, a: int, b: int) -> int:
    """k0, the first quotient digit a DIV-class record computes (digits 0..k0-1 are 0).

    |A| < 16^dA and |B| >= 16^(dB-1), so |A| / |B| has at most dA - dB + 1 hex digits. dA counts
    the nonzero prefixes of a's raw bits (8 when a negative signed operand sets bit 31: an
    overestimate, which is safe) and dB those of |B| - 1 for a negative signed b, else of b (an
    underestimate when b < 0, also safe). b = 0 computes all 8 digits (the quotient is all ones)."""
    sb = op in ("DIV", "REM") and b >> 31
    da = nibbles_used(a)
    db = nibbles_used((~b & M32) if sb else b)
    return 0 if b == 0 else min(max(7 - da + db, 0), 8)


def div_parts(op: str, a: int, b: int) -> tuple[int, list[int], int]:
    """(k0, the quotient digits k0..7 of |A| / |B|, the result) of a DIV-class instruction."""
    signed = op in ("DIV", "REM")
    sa, sb = signed and a >> 31 == 1, signed and b >> 31 == 1
    ua, ub = (-a) & M32 if sa else a, (-b) & M32 if sb else b
    q = ua // ub if ub else M32
    k0 = div_start(op, a, b)
    assert q >> 4 * (8 - k0) == 0 if k0 < 8 else q == 0, (op, a, b, k0)
    digits = [q >> 4 * (7 - k) & 15 for k in range(k0, 8)]
    qs = (-q) & M32 if sa != sb and b else q  # the signed quotient
    res = qs if op in ("DIV", "DIVU") else (a - qs * b) & M32
    assert res == alu(op, a, b), (op, a, b)
    return k0, digits, res


def div_record(op: str, a: int, b: int) -> list[int]:
    k0, digits, res = div_parts(op, a, b)
    return ([S.tok("QS", k0)] + [S.tok("QD", (k0 + i) << 4 | q) for i, q in enumerate(digits)]
            + S.value_record(res))


def csr_read(ins: Inst) -> int | None:
    """The value of a supported pure CSR read (csrr), else None (an illegal-instruction fault).
    The time CSR (the input port) reads 0 here: the ROM's view; Machine injects the input."""
    pure = ins.op in ("CSRRS", "CSRRC", "CSRRSI", "CSRRCI") and ins.rs1 == 0
    return CSR_STUBS.get(ins.imm) if pure else None


def is_input(ins: Inst) -> bool:
    """csrr rd, time with rd != x0: a read of the input port."""
    return csr_read(ins) is not None and ins.imm == S.CSR_INPUT and ins.rd != 0


class Fault(Exception):
    pass


class InputEmpty(Exception):
    """csrr rd, time with an empty input queue; nothing was committed (push inputs, retry)."""


@dataclass
class Rv32:
    """Plain RV32IM + Zba with a flat byte memory (dict) and ecall exit / putchar. CSR reads
    follow the same stubs; time reads the next word of ``inputs``."""

    mem: dict[int, int] = field(default_factory=dict)
    x: list[int] = field(default_factory=lambda: [0] * 32)
    pc: int = 0
    out: bytearray = field(default_factory=bytearray)
    halted: bool = False
    exit_code: int | None = None
    steps: int = 0
    inputs: deque = field(default_factory=deque)

    def load(self, addr: int, n: int) -> int:
        return int.from_bytes(bytes(self.mem.get(addr + i, 0) for i in range(n)), "little")

    def store(self, addr: int, v: int, n: int) -> None:
        for i in range(n):
            self.mem[addr + i] = v >> 8 * i & 0xFF

    def step(self) -> Inst:
        w = self.load(self.pc, 4)
        ins = decode(w)
        op, x = ins.op, self.x
        nxt = self.pc + 4
        a, b = x[ins.rs1], x[ins.rs2]
        res = None
        if op == "LUI":
            res = ins.imm & M32
        elif op == "AUIPC":
            res = (self.pc + ins.imm) & M32
        elif op in ("JAL", "JALR"):
            res = nxt
            nxt = (self.pc + ins.imm) & M32 if op == "JAL" else (a + ins.imm) & M32 & ~1
        elif op in BRANCH_OPS:
            if taken(op, a, b):
                nxt = (self.pc + ins.imm) & M32
        elif op == "EBREAK":
            self.halted = True
        elif op in LOAD_OPS:
            ea = (a + ins.imm) & M32
            n = {"LB": 1, "LBU": 1, "LH": 2, "LHU": 2, "LW": 4}[op]
            v = self.load(ea, n)
            res = (sx(v, 8 * n) if op in ("LB", "LH") else v) & M32
        elif op in STORE_OPS:
            ea = (a + ins.imm) & M32
            self.store(ea, b, {"SB": 1, "SH": 2, "SW": 4}[op])
        elif op == "ECALL":
            if x[17] == 93:
                self.halted, self.exit_code = True, x[10]
            elif x[17] == 1:  # putchar(a0); 256.. (YIELD) prints nothing
                if x[10] & 0x1FF < 256:
                    self.out.append(x[10] & 0xFF)
            else:
                self.halted = True
        elif op == "FENCE":
            pass
        elif op.startswith("CSR"):
            v = csr_read(ins)
            if v is None:
                raise Fault(f"unsupported CSR access {w:#010x} at {self.pc:#x}")
            if is_input(ins):
                if not self.inputs:
                    raise InputEmpty
                v = self.inputs.popleft() & M32
            res = v
        elif op == "ILLEGAL":
            raise Fault(f"unsupported instruction {w:#010x} at {self.pc:#x}")
        elif op in IMM_OPS:
            res = alu(op, a, ins.imm & M32)
        else:
            res = alu(op, a, b)
        if res is not None and ins.rd:
            x[ins.rd] = res
        self.pc = nxt & M32
        self.steps += 1
        return ins


# --- the circuit's ROM ------------------------------------------------------------------------


def in_rom(t: int) -> bool:
    return t % 4 == 0 and S.ROM_BASE <= t < S.ROM_BASE + S.ROM_SIZE


def pack(uop: str, rd: int = 0, rs1: int = 0, rs2: int = 0, spare: int = 0) -> int:
    """The ROM word from its uop, register slots and spare bits S0..S9."""
    b0 = U[uop] | (spare & 1) << 7
    b1 = rd << 3 | (spare >> 1 & 7)
    b2 = rs1 << 3 | (spare >> 4 & 7)
    b3 = rs2 << 3 | (spare >> 7 & 7)
    return b0 | b1 << 8 | b2 << 16 | b3 << 24


def rom_word(w: int, pc: int) -> int:
    """The pre-decoded ROM word of the instruction word w at pc (format in the module doc)."""
    ins = decode(w)
    op, imm = ins.op, ins.imm
    if op.startswith("CSR"):
        v = csr_read(ins)
        if v is None:
            return pack("ILLEGAL")
        return pack("MISA", ins.rd) if v else pack("LUI", ins.rd)
    if op in ("LUI", "AUIPC"):
        up = (imm if op == "LUI" else pc + imm) >> 12 & 0xFFFFF
        return pack(op, ins.rd, up & 31, up >> 5 & 31, up >> 10)
    if op in IMM_OPS or op in LOAD_OPS or op == "JALR":
        i = imm & 0xFFF
        return pack(op, ins.rd, ins.rs1, i & 31, i >> 5)
    if op in STORE_OPS:
        i = imm & 0xFFF
        return pack(op, ins.rs2, ins.rs1, i & 31, i >> 5)
    if op in BRANCH_OPS:
        t = (pc + imm) & M32
        off = (imm >> 2) & 0x7FF  # signed word offset, 11 bits
        bad = int(not in_rom(t))
        return pack(op, off & 31, ins.rs1, ins.rs2, off >> 5 | bad << 6)
    if op == "JAL":
        t = (pc + imm) & M32
        wi = (t - S.ROM_BASE) >> 2 & 0xFFFF
        bad = int(not in_rom(t))
        return pack(op, ins.rd, wi & 31, wi >> 5 & 31, wi >> 10 | bad << 6)
    if op == "ECALL":
        return pack(op, 0, 17, 10)
    if op in ("EBREAK", "FENCE", "ILLEGAL"):
        return pack(op)
    return pack(op, ins.rd, ins.rs1, ins.rs2)  # R-type


def rom_words(words, base: int = S.ROM_BASE) -> np.ndarray:
    """``rom_word`` of a whole image (words at base, base + 4, ..), vectorized: uint32 array."""
    w = np.asarray(words, dtype=np.int64) & M32
    pc = base + 4 * np.arange(len(w), dtype=np.int64)
    opc, rd, f3 = w & 0x7F, w >> 7 & 31, w >> 12 & 7
    rs1, rs2, f7 = w >> 15 & 31, w >> 20 & 31, w >> 25

    def sxv(v, bits):
        v = v & ((1 << bits) - 1)
        return np.where(v >> (bits - 1) & 1, v - (1 << bits), v)

    i_imm = sxv(w >> 20, 12)
    s_imm = sxv((w >> 25) << 5 | (w >> 7 & 31), 12)
    b_imm = sxv((w >> 31 & 1) << 12 | (w >> 7 & 1) << 11 | (w >> 25 & 63) << 5
                | (w >> 8 & 15) << 1, 13)
    j_imm = sxv((w >> 31 & 1) << 20 | (w >> 12 & 0xFF) << 12 | (w >> 20 & 1) << 11
                | (w >> 21 & 0x3FF) << 1, 21)
    u_imm = sxv(w & 0xFFFFF000, 32)

    uop = np.zeros(len(w), dtype=np.int64)  # ILLEGAL
    kind = np.zeros(len(w), dtype=np.int64)  # 0 bare, 1 R, 2 I, 3 S, 4 B, 5 J, 6 U, 7 ecall, 8 csr

    def put(mask, name, k):
        uop[mask] = U[name]
        kind[mask] = k

    put(opc == 0x37, "LUI", 6)
    put(opc == 0x17, "AUIPC", 6)
    put(opc == 0x6F, "JAL", 5)
    put((opc == 0x67) & (f3 == 0), "JALR", 2)
    for i, nm in enumerate(["BEQ", "BNE", "", "", "BLT", "BGE", "BLTU", "BGEU"]):
        if nm:
            put((opc == 0x63) & (f3 == i), nm, 4)
    for i, nm in enumerate(["LB", "LH", "LW", "", "LBU", "LHU"]):
        if nm:
            put((opc == 0x03) & (f3 == i), nm, 2)
    for i, nm in enumerate(["SB", "SH", "SW"]):
        put((opc == 0x23) & (f3 == i), nm, 3)
    op_i = opc == 0x13
    put(op_i & (f3 == 1) & (f7 == 0), "SLLI", 2)
    put(op_i & (f3 == 5) & (f7 == 0), "SRLI", 2)
    put(op_i & (f3 == 5) & (f7 == 0x20), "SRAI", 2)
    for i, nm in enumerate(["ADDI", "", "SLTI", "SLTIU", "XORI", "", "ORI", "ANDI"]):
        if nm:
            put(op_i & (f3 == i), nm, 2)
    op_r = opc == 0x33
    for i, nm in enumerate(["ADD", "SLL", "SLT", "SLTU", "XOR", "SRL", "OR", "AND"]):
        put(op_r & (f7 == 0) & (f3 == i), nm, 1)
    put(op_r & (f7 == 0x20) & (f3 == 0), "SUB", 1)
    put(op_r & (f7 == 0x20) & (f3 == 5), "SRA", 1)
    for i, nm in enumerate(["MUL", "MULH", "MULHSU", "MULHU", "DIV", "DIVU", "REM", "REMU"]):
        put(op_r & (f7 == 1) & (f3 == i), nm, 1)
    for i, nm in ((2, "SH1ADD"), (4, "SH2ADD"), (6, "SH3ADD")):
        put(op_r & (f7 == 0x10) & (f3 == i), nm, 1)
    put(w == 0x00000073, "ECALL", 7)
    csr = (opc == 0x73) & np.isin(f3, (1, 2, 3, 5, 6, 7))
    kind[csr] = 8
    put(w == 0x00100073, "EBREAK", 0)
    put((opc == 0x0F), "FENCE", 0)

    def in_rom_v(t):
        return (t % 4 == 0) & (t >= S.ROM_BASE) & (t < S.ROM_BASE + S.ROM_SIZE)

    out_rd, out_rs1, out_rs2, spare = rd.copy(), rs1.copy(), rs2.copy(), np.zeros_like(w)
    # I-type and stores: imm12 in RS2 + S0..S6
    m = kind == 2
    i12 = np.where(m, i_imm, s_imm) & 0xFFF
    out_rs2 = np.where(m, i12 & 31, out_rs2)
    spare = np.where(m, i12 >> 5, spare)
    shift = np.isin(uop, [U["SLLI"], U["SRLI"], U["SRAI"]])  # shamt: imm = rs2
    out_rs2 = np.where(m & shift, rs2, out_rs2)
    spare = np.where(m & shift, 0, spare)
    m = kind == 3
    out_rd, out_rs2 = np.where(m, rs2, out_rd), np.where(m, i12 & 31, out_rs2)
    spare = np.where(m, i12 >> 5, spare)
    m = kind == 4  # branches
    t = (pc + b_imm) & M32
    off = (b_imm >> 2) & 0x7FF
    bad = (~in_rom_v(t)).astype(np.int64)
    out_rd = np.where(m, off & 31, out_rd)
    spare = np.where(m, off >> 5 | bad << 6, spare)
    m = kind == 5  # JAL
    t = (pc + j_imm) & M32
    wi = (t - S.ROM_BASE) >> 2 & 0xFFFF
    bad = (~in_rom_v(t)).astype(np.int64)
    out_rs1, out_rs2 = np.where(m, wi & 31, out_rs1), np.where(m, wi >> 5 & 31, out_rs2)
    spare = np.where(m, wi >> 10 | bad << 6, spare)
    m = kind == 6  # LUI / AUIPC
    up = np.where(uop == U["LUI"], u_imm, pc + u_imm) >> 12 & 0xFFFFF
    out_rs1, out_rs2 = np.where(m, up & 31, out_rs1), np.where(m, up >> 5 & 31, out_rs2)
    spare = np.where(m, up >> 10, spare)
    m = kind == 7  # ECALL
    out_rd, out_rs1, out_rs2 = np.where(m, 0, out_rd), np.where(m, 17, out_rs1), \
        np.where(m, 10, out_rs2)
    m = kind == 8  # CSR stubs: pure reads only
    pure = np.isin(f3, (2, 3, 6, 7)) & (rs1 == 0)
    num = w >> 20
    known = np.isin(num, list(CSR_STUBS))
    misa = num == 0x301
    uop = np.where(m, np.where(pure & known, np.where(misa, U["MISA"], U["LUI"]), U["ILLEGAL"]),
                   uop)
    ok = m & pure & known
    out_rd = np.where(m, np.where(ok, rd, 0), out_rd)
    out_rs1, out_rs2 = np.where(m, 0, out_rs1), np.where(m, 0, out_rs2)
    spare = np.where(m, 0, spare)
    bare = (kind == 0) | (uop == U["ILLEGAL"])  # EBREAK, FENCE, ILLEGAL: no fields
    out_rd, out_rs1, out_rs2 = (np.where(bare, 0, v) for v in (out_rd, out_rs1, out_rs2))
    spare = np.where(bare, 0, spare)
    b0 = uop | (spare & 1) << 7
    b1 = out_rd << 3 | (spare >> 1 & 7)
    b2 = out_rs1 << 3 | (spare >> 4 & 7)
    b3 = out_rs2 << 3 | (spare >> 7 & 7)
    return (b0 | b1 << 8 | b2 << 16 | b3 << 24).astype(np.uint32)


# --- the traced machine -----------------------------------------------------------------------


def words_of(b: bytes, n: int | None = None) -> list[int]:
    """Little-endian words of a byte string (zero-padded; n words if given)."""
    b = bytes(b) + bytes(-len(b) % 4)
    if n is not None:
        assert len(b) <= 4 * n, (len(b), n)
        b += bytes(4 * n - len(b))
    return np.frombuffer(b, dtype="<u4").astype(np.int64).tolist()


@dataclass
class Machine:
    """State the trace commits: registers, RAM words, anchor, n. ROM and DROM are read-only."""

    rom: list[int]  # instruction words from ROM_BASE (ROM_WORDS)
    ram: list[int]  # words from RAM_BASE (RAM_WORDS)
    x: list[int]
    anchor: int  # PC word index of the latest J / R
    n: int = 0
    halted: bool = False
    steps: int = 0  # instructions executed
    drom: bytes = b""  # read-only data from DROM_BASE
    inputs: deque = field(default_factory=deque)  # the input port's queue (csrr rd, time)
    out: bytearray = field(default_factory=bytearray)  # putchar output (bytes < 256)
    fault: tuple[int, int] | None = None  # (cause, pc) of the fault that halted the run
    exit_code: int | None = None
    yields: int = 0  # BYTE(256) records
    last_input: bool = False  # the latest record was an input record (host-injected tokens)
    injected: list[int] = field(default_factory=list)  # run(): positions of injected tokens

    @classmethod
    def from_program(cls, prog, entry: int | None = None, inputs=()) -> "Machine":
        """A machine at a program's entry (an asm / ELF Program: text words at ROM_BASE, data
        bytes at data_base, rodata at DROM_BASE), sp = SP_INIT."""
        ram = bytes(prog.data_base - S.RAM_BASE) + bytes(prog.data)
        assert len(ram) <= S.RAM_SIZE, "data does not fit the RAM"
        rodata = bytes(getattr(prog, "rodata", b""))
        assert len(rodata) <= S.DROM_SIZE, "rodata does not fit the DROM"
        assert len(prog.text) <= S.ROM_WORDS, "code does not fit the ROM"
        x = [0] * 32
        x[2] = S.SP_INIT
        e = prog.entry if entry is None else entry
        return cls(list(prog.text) + [0] * (S.ROM_WORDS - len(prog.text)),
                   words_of(ram, S.RAM_WORDS), x, (e - S.ROM_BASE) >> 2, drom=rodata,
                   inputs=deque(inputs))

    @classmethod
    def from_images(cls, ram: bytes, drom: bytes, rom: bytes, regs, pc: int,
                    inputs=()) -> "Machine":
        """A machine at pc (n = 0) from raw region images (as qwendoom.snapshot takes them)."""
        assert in_rom(pc), hex(pc)
        x = [int(v) & M32 for v in regs]
        x[0] = 0
        return cls(words_of(rom, S.ROM_WORDS), words_of(ram, S.RAM_WORDS), x,
                   (pc - S.ROM_BASE) >> 2, drom=bytes(drom), inputs=deque(inputs))

    @property
    def pc(self) -> int:
        return S.ROM_BASE + 4 * (self.anchor + self.n)

    def ram_index(self, ea: int) -> int | None:
        return (ea - S.RAM_BASE) >> 2 if 0 <= ea - S.RAM_BASE < S.RAM_SIZE else None

    def drom_word(self, ea: int) -> int | None:
        """The DROM word holding byte ea, or None outside the DROM."""
        o = ea - S.DROM_BASE
        if not 0 <= o < S.DROM_SIZE:
            return None
        o &= ~3
        return int.from_bytes(self.drom[o:o + 4].ljust(4, b"\0"), "little")

    def record(self) -> list[int]:
        """The next record's tokens; commits it (state as after its last token). Raises
        InputEmpty (committing nothing) at an input read with no input queued."""
        self.last_input = False
        if self.n == S.N_MAX:
            w = self.anchor + self.n
            self.anchor, self.n = w, 0
            return [S.tok("R", w)]
        pc = self.pc
        word = self.rom[(pc - S.ROM_BASE) >> 2]
        ins = decode(word)
        op, x = ins.op, self.x
        a, b = x[ins.rs1], x[ins.rs2]
        if op.startswith("CSR") and is_input(ins) and not self.inputs:
            raise InputEmpty(f"csrr x{ins.rd}, time at {pc:#x}")
        self.steps += 1

        def fault(cause):
            self.halted, self.fault = True, (S.FAULTS.index(cause), pc)
            return S.fault_record(S.FAULTS.index(cause), pc)

        def commit_rd(v):
            self.n += 1
            if ins.rd:
                x[ins.rd] = v & M32
                return S.value_record(v)
            return [S.tok("NT")]

        def jump(target, link):
            if not in_rom(target):
                return fault("bad jump")
            if link and ins.rd:
                x[ins.rd] = (pc + 4) & M32
            self.anchor, self.n = (target - S.ROM_BASE) >> 2, 0
            return [S.tok("J", self.anchor)]

        if op in ("LUI", "AUIPC") or op.startswith("CSR"):
            v = ins.imm if op == "LUI" else pc + ins.imm if op == "AUIPC" else csr_read(ins)
            if v is None:
                return fault("illegal")
            if op.startswith("CSR") and is_input(ins):
                v = self.inputs.popleft() & M32
                self.last_input = True  # the host injects this value record
            return commit_rd(v)
        if op in ("JAL", "JALR"):
            target = (pc + ins.imm) & M32 if op == "JAL" else (a + ins.imm) & M32 & ~1
            return jump(target, True)
        if op in BRANCH_OPS:
            if taken(op, a, b):
                return jump((pc + ins.imm) & M32, False)
            self.n += 1
            return [S.tok("NT")]
        if op in LOAD_OPS:
            ea = (a + ins.imm) & M32
            n = {"LB": 1, "LBU": 1, "LH": 2, "LHU": 2, "LW": 4}[op]
            if ea % n:
                return fault("misaligned load")
            i = self.ram_index(ea)
            word = self.ram[i] if i is not None else self.drom_word(ea)
            if word is None:
                return fault("load out of range")
            wv = word >> 8 * (ea & 3) & ((1 << 8 * n) - 1)
            return commit_rd(sx(wv, 8 * n) if op in ("LB", "LH") else wv)
        if op in STORE_OPS:
            ea = (a + ins.imm) & M32
            n = {"SB": 1, "SH": 2, "SW": 4}[op]
            if ea % n:
                return fault("misaligned store")
            i = self.ram_index(ea)
            if i is None:
                return fault("store out of range")
            w4 = ea >> 6 & 1  # ST0 / ST1, SS0 / SS1: address bit 6
            self.n += 1
            if op == "SW":
                self.ram[i] = b
                return [S.tok(f"ST{w4}")]
            sh, mask = 8 * (ea & 3), (1 << 8 * n) - 1
            m = self.ram[i] & ~(mask << sh) & M32 | (b & mask) << sh
            self.ram[i] = m
            return [S.tok("V", m & 0x7FFF), S.tok("V", m >> 15 & 0x7FFF), S.tok("V", m >> 30),
                    S.tok(f"SS{w4}")]
        if op == "ECALL":
            a, b = x[17], x[10]  # the call (a7) and its argument (a0)
            if a == 1:  # putchar(a0): a byte, or (256..) a special token: 256 is YIELD
                if b & 0x1FF < 256:
                    self.out.append(b & 0xFF)
                elif b & 0x1FF == S.YIELD:
                    self.yields += 1
                self.n += 1
                return [S.tok("BYTE", b & 0x1FF)]
            if a == 93:  # exit(a0)
                self.halted, self.exit_code = True, b
                self.n += 1
                return [S.tok("HALT")]
            return fault("ecall")
        if op == "EBREAK":
            return fault("ebreak")
        if op == "FENCE":
            self.n += 1
            return [S.tok("NT")]
        if op == "ILLEGAL":
            return fault("illegal")
        if op in DIV_OPS:
            self.n += 1
            if not ins.rd:
                return [S.tok("NT")]
            x[ins.rd] = div_parts(op, a, b)[2]
            return div_record(op, a, b)
        if op in IMM_OPS:
            return commit_rd(alu(op, a, ins.imm & M32))
        return commit_rd(alu(op, a, b))

    def run(self, max_tokens: int = 10**6) -> list[int]:
        """The whole trace from GO to HALT (or max_tokens); ``injected`` then lists the positions
        of the host-injected tokens (input records) in it."""
        toks = [S.tok("GO")]
        self.injected = []
        while not self.halted and len(toks) < max_tokens:
            rec = self.record()
            if self.last_input:
                self.injected.extend(range(len(toks), len(toks) + len(rec)))
            toks += rec
        return toks


def trace_stats(toks: list[int]) -> dict:
    fams = [S.family(t)[0] for t in toks]
    # one final token per instruction, except R refreshes; a fault's FLT .. HALT is one
    instr = sum(1 for f in fams if f in S.FINAL and f != "R")
    return {"tokens": len(toks), "instructions": instr,
            "tokens_per_instruction": len(toks) / max(instr, 1),
            "refreshes": fams.count("R")}
