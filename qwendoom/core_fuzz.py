"""Random RV32IM + Zba programs for the circuit: fault-free in the reference, every token of their
traces predicted exactly in the IR (teacher-forced; tests/test_core_fuzz.py).

Derived from Qwenputer's chipformer/rv/fuzz.py (MIT, (c) 2026 Divyansh Agrawal; see
LICENSE.chipformer). A program is a loop over a random block: every ALU, shift, compare, MUL, DIV
and Zba op on random registers; loads and stores (all widths) through masked addresses into a
4 KB buffer and anywhere in the 256 KB RAM; loads from a 4 KB DROM table and anywhere in the
512 KB DROM; forward branches; calls to a leaf function; putchar of a register's low byte; YIELD;
LUI / AUIPC.
"""

import random

from qwendoom import spec as S

REGS = ["a0", "a1", "a2", "a3", "a4", "a5", "t0", "t1", "t2", "t3", "t4", "s1", "s2", "s3", "s4"]
OPS3 = ["add", "sub", "and", "or", "xor", "sll", "srl", "sra", "slt", "sltu", "mul", "mulh",
        "mulhsu", "mulhu", "div", "divu", "rem", "remu", "sh1add", "sh2add", "sh3add"]
OPSI = ["addi", "andi", "ori", "xori", "slti", "sltiu"]
OPSS = ["slli", "srli", "srai"]
LOADS = [("lw", 0xFFC), ("lh", 0xFFE), ("lhu", 0xFFE), ("lb", 0xFFF), ("lbu", 0xFFF)]
STORES = [("sw", 0xFFC), ("sh", 0xFFE), ("sb", 0xFFF)]
# the full-region masks live in registers: s5 / s6 / s7 = RAM word / half / byte, s8 = DROM word
WIDE = {4: "s5", 2: "s6", 1: "s7"}


def gen_program(seed: int, block: int = 40, iters: int = 30) -> str:
    """A random fault-free RV32IM + Zba program (s0: loop counter, s11: the RAM buffer, s10: the
    DROM table, s9: DROM_BASE, t5: addresses)."""
    rng = random.Random(seed)

    def val():
        return rng.choice([rng.randint(-2**31, 2**31 - 1), rng.randint(-40, 40),
                           rng.randint(0, 0x3FFFF)])

    width = {"lw": 4, "sw": 4, "lh": 2, "lhu": 2, "sh": 2, "lb": 1, "lbu": 1, "sb": 1}
    lines = ["_start: la s11, buf", "la s10, rtab", f"li s9, {S.DROM_BASE}",
             f"li s5, {S.RAM_SIZE - 4}", f"li s6, {S.RAM_SIZE - 2}", f"li s7, {S.RAM_SIZE - 1}",
             f"li s8, {S.DROM_SIZE - 4}", f"li s0, {iters}"]
    lines += [f"li {r}, {val()}" for r in REGS]
    lines.append("loop:")
    skip = 0
    for _ in range(block):
        k = rng.random()
        d, a, b = rng.choice(REGS), rng.choice(REGS), rng.choice(REGS)
        if k < 0.33:
            lines.append(f"{rng.choice(OPS3)} {d}, {a}, {b}")
        elif k < 0.46:
            lines.append(f"{rng.choice(OPSI)} {d}, {a}, {rng.randint(-2048, 2047)}")
        elif k < 0.52:
            lines.append(f"{rng.choice(OPSS)} {d}, {a}, {rng.randint(0, 31)}")
        elif k < 0.58:
            op, mask = rng.choice(LOADS)
            lines += [f"andi t5, {a}, {mask & 0x7FF}", "add t5, t5, s11", f"{op} {d}, 0(t5)"]
        elif k < 0.62:
            op, _ = rng.choice(LOADS)
            lines += [f"and t5, {a}, {WIDE[width[op]]}", f"{op} {d}, 0(t5)"]
        elif k < 0.66:
            op, mask = rng.choice(LOADS)
            lines += [f"andi t5, {a}, {mask & 0x7FF}", "add t5, t5, s10", f"{op} {d}, 0(t5)"]
        elif k < 0.69:
            op, _ = rng.choice(LOADS)
            off = rng.choice([0, 2, 3]) if width[op] == 1 else 0
            lines += [f"and t5, {a}, s8", "add t5, t5, s9", f"{op} {d}, {off}(t5)"]
        elif k < 0.77:
            op, mask = rng.choice(STORES)
            lines += [f"andi t5, {a}, {mask & 0x7FF}", "add t5, t5, s11", f"{op} {b}, 0(t5)"]
        elif k < 0.81:
            op, _ = rng.choice(STORES)
            lines += [f"and t5, {a}, {WIDE[width[op]]}", f"{op} {b}, 0(t5)"]
        elif k < 0.89:
            br = rng.choice(["beq", "bne", "blt", "bge", "bltu", "bgeu"])
            lines += [f"{br} {a}, {b}, skip{skip}", f"{rng.choice(OPS3)} {d}, {a}, {b}",
                      f"skip{skip}:"]
            skip += 1
        elif k < 0.93:
            lines += [f"mv a6, {a}", "call leaf", f"mv {d}, a6"]
        elif k < 0.96:
            lines += [f"andi a0, {a}, 127", "li a7, 1", "ecall"]
        elif k < 0.97:
            lines += [f"li a0, {S.YIELD}", "li a7, 1", "ecall"]
        else:
            lines.append(f"lui {d}, {rng.randint(0, 2**20 - 1)}" if rng.random() < 0.5
                         else f"auipc {d}, {rng.randint(0, 2**20 - 1)}")
    lines += ["addi s0, s0, -1", "bnez s0, loop", "li a0, 0", "li a7, 93", "ecall",
              "leaf: slli a6, a6, 3", "xori a6, a6, 0x5a5", "add a6, a6, s0", "ret",
              ".data", "buf: .space 4096",
              ".rodata", "rtab: .word " + ", ".join(str(rng.getrandbits(32)) for _ in range(1024))]
    return "\n".join(lines)
