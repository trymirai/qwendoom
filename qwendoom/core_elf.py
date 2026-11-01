"""Load a static RV32 ELF (clang + ld.lld, linked for the qwendoom map) into a Program (for
ref.Machine and the circuit's snapshot) or into a ref.Rv32 (flat memory).

Derived from Qwenputer's chipformer/rv/elf.py (MIT, (c) 2026 Divyansh Agrawal; see
LICENSE.chipformer).
"""

import struct
from pathlib import Path

from qwendoom import spec as S
from qwendoom.core_asm import Program


def segments(path) -> tuple[int, list[tuple[int, bytes]]]:
    """(entry, [(vaddr, bytes incl. zero-filled bss)]) of the PT_LOAD segments."""
    b = Path(path).read_bytes()
    assert b[:4] == b"\x7fELF" and b[4] == 1, "ELF32 only"
    entry, phoff = struct.unpack_from("<II", b, 0x18)
    phentsize, phnum = struct.unpack_from("<HH", b, 0x2A)
    segs = []
    for i in range(phnum):
        p_type, off, vaddr, _, filesz, memsz, _, _ = struct.unpack_from("<8I", b,
                                                                    phoff + i * phentsize)
        if p_type == 1 and memsz:
            segs.append((vaddr, b[off:off + filesz] + bytes(memsz - filesz)))
    return entry, segs


def symbols(path) -> dict[str, int]:
    """name -> value of the ELF's symbol table (for tests that poke or inspect variables)."""
    b = Path(path).read_bytes()
    shoff, = struct.unpack_from("<I", b, 0x20)
    shentsize, shnum = struct.unpack_from("<HH", b, 0x2E)
    secs = [struct.unpack_from("<10I", b, shoff + i * shentsize) for i in range(shnum)]
    out = {}
    for sh in secs:
        if sh[1] != 2:  # SHT_SYMTAB
            continue
        strtab = secs[sh[6]]
        for k in range(sh[5] // 16):
            name, value, _, _, _, _ = struct.unpack_from("<IIIBBH", b, sh[4] + 16 * k)
            s = b[strtab[4] + name:b.index(b"\0", strtab[4] + name)].decode()
            if s:
                out[s] = value
    return out


def program(path) -> Program:
    """A qwendoom-map ELF as a Program: code words from ROM_BASE, read-only data from
    DROM_BASE, data bytes from RAM_BASE."""
    entry, segs = segments(path)
    regions = {"rom": (S.ROM_BASE, S.ROM_SIZE), "drom": (S.DROM_BASE, S.DROM_SIZE),
               "ram": (S.RAM_BASE, S.RAM_SIZE)}
    bufs = {k: bytearray() for k in regions}
    for vaddr, blob in segs:
        for k, (base, size) in regions.items():
            if base <= vaddr and vaddr + len(blob) <= base + size:
                b = bufs[k]
                end = vaddr - base + len(blob)
                b += bytes(max(0, end - len(b)))
                b[vaddr - base:end] = blob
                break
        else:
            raise AssertionError(f"segment at {vaddr:#x} (+{len(blob)}) outside ROM/DROM/RAM")
    text = bufs["rom"] + bytes(-len(bufs["rom"]) % 4)
    words = [int.from_bytes(text[i:i + 4], "little") for i in range(0, len(text), 4)]
    return Program(words, bytes(bufs["ram"]), {"_start": entry}, data_base=S.RAM_BASE,
                   rodata=bytes(bufs["drom"]))


def load_into(cpu, path) -> None:
    """Load any static ELF into a ref.Rv32 (flat memory) and set its PC."""
    entry, segs = segments(path)
    for vaddr, blob in segs:
        for i, x in enumerate(blob):
            cpu.mem[vaddr + i] = x
    cpu.pc = entry
