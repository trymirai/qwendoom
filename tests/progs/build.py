"""Build the core's test programs for the qwendoom address map with clang + ld.lld:

    .venv/bin/python tests/progs/build.py [--out build/core/progs]

* riscv-tests rv32ui (minus fence_i: self-modifying code, and ma_data: misaligned access, as
  upstream), rv32um and rv32uzba from third_party/riscv-tests/isa, with env/riscv_test.h (a test
  ends with the exit ecall: a0 = 0 on pass, the failing test number otherwise);
* the freestanding C programs here (no libc; env/rt.c has the runtime and the 64-bit divide
  libcalls), -O2, rv32im_zba, crt0 + env/qd.ld (ROM 0x100000, DROM 0x80000, RAM 0x3000..).

Outputs OUT/<suite>-<name>.elf and OUT/c-<name>.elf. memsweep's 504 KB DROM table is generated
(OUT/gen/bigrom.bin, included by OUT/gen/bigrom.S). The approach follows Qwenputer's
chipformer/rv/env/build_tests.sh (MIT, (c) 2026 Divyansh Agrawal; see LICENSE.chipformer).
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ISA = ROOT / "third_party" / "riscv-tests" / "isa"
CLANG = "/opt/homebrew/opt/llvm/bin/clang"
LLD = "/opt/homebrew/bin/ld.lld"
ARCH = ["--target=riscv32-unknown-elf", "-march=rv32im_zba", "-mabi=ilp32", "-mno-relax"]
CFLAGS = ["-O2", "-ffreestanding", "-fno-delete-null-pointer-checks", "-fno-pic", "-Wall",
          "-Werror", "-g0"]
SKIP = {"fence_i", "ma_data"}
SUITES = ["rv32ui", "rv32um", "rv32uzba"]
BIGROM_WORDS = 129024  # memsweep.c's big_rom: 504 KB, leaving 8 KB of the DROM for other rodata


def run(cmd: list) -> None:
    p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if p.returncode:
        sys.exit(f"failed: {' '.join(map(str, cmd))}\n{p.stdout}{p.stderr}")


def hash32(x: np.ndarray) -> np.ndarray:
    """memsweep.c's hash, vectorized (uint32 wraparound)."""
    x = (x.astype(np.uint64) * 0x9E3779B1) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 0x85EBCA77) & 0xFFFFFFFF
    x ^= x >> 13
    return x.astype(np.uint32)


def gen_bigrom(gen: Path) -> Path:
    gen.mkdir(parents=True, exist_ok=True)
    hash32(np.arange(BIGROM_WORDS)).astype("<u4").tofile(gen / "bigrom.bin")
    s = gen / "bigrom.S"
    s.write_text('    .section .rodata.bigrom, "a"\n    .globl big_rom\n    .p2align 2\n'
                 f'big_rom:\n    .incbin "{gen / "bigrom.bin"}"\n')
    return s


def link(out: Path, objs: list) -> None:
    run([LLD, "-T", HERE / "env" / "qd.ld", "--no-relax", "-static", "-o", out, *objs])


def build(out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    obj = out / "obj"
    obj.mkdir(exist_ok=True)
    built = []
    for suite in SUITES:
        for src in sorted((ISA / suite).glob("*.S")):
            name = f"{suite}-{src.stem}"
            if src.stem in SKIP:
                continue
            o = obj / f"{name}.o"
            run([CLANG, *ARCH, "-c", "-I", HERE / "env", "-I", ISA / "macros" / "scalar", "-o", o,
                 src])
            link(out / f"{name}.elf", [o])
            built.append(out / f"{name}.elf")
    common = []
    for src in (HERE / "env" / "crt0.S", HERE / "env" / "rt.c"):
        o = obj / f"env-{src.stem}.o"
        run([CLANG, *ARCH, *CFLAGS, "-c", "-o", o, src])
        common.append(o)
    bigrom = obj / "bigrom.o"
    run([CLANG, *ARCH, "-c", "-o", bigrom, gen_bigrom(out / "gen")])
    for src in sorted(HERE.glob("*.c")):
        o = obj / f"c-{src.stem}.o"
        run([CLANG, *ARCH, *CFLAGS, "-I", HERE, "-c", "-o", o, src])
        extra = [bigrom] if src.stem == "memsweep" else []
        link(out / f"c-{src.stem}.elf", [*common, o, *extra])
        built.append(out / f"c-{src.stem}.elf")
    return built


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(ROOT / "build" / "core" / "progs"))
    args = ap.parse_args()
    built = build(Path(args.out))
    print(f"{len(built)} programs in {args.out}")


if __name__ == "__main__":
    main()
