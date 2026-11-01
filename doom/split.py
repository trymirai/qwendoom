"""The sim/host split of DOOM: which code and data the sim (sim_main -> P_Ticker) can reach, and
the linker script that places it (ARCHITECTURE.md's address map).

    python split.py analyze --out-ld doom.ld --out-json doom-split.json OBJS...
    python split.py check build/doom.elf build/doom-split.json
    python split.py why build/doom-split.json SYMBOL      # the reference chain that makes it sim

``analyze`` reads the relocatable objects (built with -ffunction-sections -fdata-sections, so
every function and variable is its own section) and walks the section graph their relocations
form, from these roots:

* ``sim_main`` (qd_sim.c), the sim's entry;
* every function of a playsim object (p_*.o) whose address is taken anywhere: thinkers
  (T_MoveFloor, T_LightFlash, ...) are stored into thinker_t by host code (P_SpawnSpecials at
  level load, savegames) and called by the sim through P_RunThinkers; the A_* actions come in
  through ``states[]`` and the PIT_/PTR_ iterators through sim code anyway.

Everything reached is sim: code goes to ROM, read-only data to DROM, mutable data to RAM, except
the tables in READONLY (mutable in C, never written after startup), which go to DROM. All other
sections fall into HOST. ``check`` verifies the linked ELF against the analysis and reports the
region usage; the native sim mode of the host interpreter then checks every fetch, load and
store at run time.
"""

import argparse
import json
import struct
import sys
from collections import deque
from pathlib import Path

RAM_ORIGIN, RAM_END = 0x3000, 0x40000      # sim data from 0x3000; the level zone follows to 256 KB
DROM_ORIGIN, DROM_SIZE = 0x80000, 0x80000
ROM_ORIGIN, ROM_SIZE = 0x100000, 0x40000
HOST_ORIGIN, HOST_SIZE = 0x1000000, 0x4000000
HEAP_END = 0x4F00000                       # the host heap ends 1 MB below the top (host stack)
GP_WINDOW = 0x800                          # gp-relative reach: [gp - 2 KB, gp + 2 KB)

# Mutable in C but written only at startup (or never): read-only to the sim, so DROM. The host
# interpreter flags any DROM write after startup (it forces a resync); a sim store faults.
READONLY = {
    "info.o": ["states", "mobjinfo", "sprnames"],
    "d_items.o": ["weaponinfo"],
    "p_enemy.o": ["xspeed", "yspeed", "opposite", "diags"],
    "p_inter.o": ["maxammo", "clipammo"],
}

# Playsim objects: their address-taken functions are roots (see the module doc).
PLAYSIM_EXCLUDE = {"p_setup.o", "p_saveg.o"}

# Objects whose functions in ROM would mean host code leaked into the sim (reported).
HOST_ONLY_PREFIXES = ("r_", "st_", "hu_", "am_", "m_menu", "w_", "v_", "i_", "d_", "g_", "f_",
                      "wi_", "s_", "stdio", "stdlib")
HOST_OK = {"R_PointInSubsector", "R_PointToAngle", "R_PointToAngle2", "G_ExitLevel",
           "G_SecretExitLevel", "I_Error", "I_Tactile"}

SHF_WRITE, SHF_ALLOC, SHF_EXECINSTR = 1, 2, 4
SHT_SYMTAB, SHT_RELA, SHT_NOBITS = 2, 4, 8
SHN_UNDEF, SHN_LORESERVE, SHN_ABS = 0, 0xFF00, 0xFFF1
R_NONE, R_BRANCH, R_JAL, R_CALL, R_CALL_PLT, R_ALIGN, R_RELAX = 0, 16, 17, 18, 19, 43, 51
CALL_RELOCS = {R_BRANCH, R_JAL, R_CALL, R_CALL_PLT}
SKIP_RELOCS = {R_NONE, R_ALIGN, R_RELAX}
STT_FUNC, STT_SECTION = 2, 3


class Obj:
    """The sections, symbols and relocations of one ELF32 relocatable object."""

    def __init__(self, path: str):
        self.path = path
        self.name = Path(path).name
        b = Path(path).read_bytes()
        if b[:4] != b"\x7fELF" or b[4] != 1:
            raise ValueError(f"{path}: not an ELF32 object")
        shoff, = struct.unpack_from("<I", b, 0x20)
        shentsize, shnum, shstrndx = struct.unpack_from("<HHH", b, 0x2E)
        hdrs = [struct.unpack_from("<10I", b, shoff + i * shentsize) for i in range(shnum)]
        strtab = hdrs[shstrndx]
        names = b[strtab[4]:strtab[4] + strtab[5]]

        def cstr(tab: bytes, off: int) -> str:
            return tab[off:tab.index(b"\0", off)].decode()

        # (name, type, flags, size, link, info)
        self.sections = [(cstr(names, h[0]), h[1], h[2], h[5], h[6], h[7]) for h in hdrs]
        self.symbols = []  # (name, value, size, type, bind, shndx)
        self.relocs = {}   # target section index -> [(type, symbol index)]
        for i, h in enumerate(hdrs):
            if h[1] == SHT_SYMTAB:
                st = hdrs[h[6]]
                stab = b[st[4]:st[4] + st[5]]
                for k in range(h[5] // 16):
                    nm, val, size, info, _, shndx = struct.unpack_from("<IIIBBH", b, h[4] + 16 * k)
                    self.symbols.append((cstr(stab, nm), val, size, info & 15, info >> 4, shndx))
        for i, h in enumerate(hdrs):
            if h[1] == SHT_RELA:
                rel = []
                for k in range(h[5] // 12):
                    _, rinfo, _ = struct.unpack_from("<IIi", b, h[4] + 12 * k)
                    rel.append((rinfo & 0xFF, rinfo >> 8))
                self.relocs.setdefault(h[7], []).extend(rel)

    def kind(self, i: int) -> str | None:
        """text / rodata / data / bss for an allocated section, else None."""
        name, typ, flags, *_ = self.sections[i]
        if not flags & SHF_ALLOC:
            return None
        if flags & SHF_EXECINSTR:
            return "text"
        if not flags & SHF_WRITE:
            return "rodata"
        return "bss" if typ == SHT_NOBITS else "data"


def analyze(paths: list[str]) -> dict:
    objs = [Obj(p) for p in paths]
    # global definitions: name -> (object, section)
    globs = {}
    for oi, o in enumerate(objs):
        for name, _, _, typ, bind, shndx in o.symbols:
            if bind in (1, 2) and shndx != SHN_UNDEF and shndx < SHN_LORESERVE:
                if name in globs and bind == 2:
                    continue
                globs[name] = (oi, shndx)

    def section_label(node):
        oi, si = node
        return f"{objs[oi].name}({objs[oi].sections[si][0]})"

    # the section graph, and which function sections have their address taken
    edges, addr_taken, undefined = {}, set(), set()
    for oi, o in enumerate(objs):
        for si, rel in o.relocs.items():
            if o.kind(si) is None:
                continue
            out = edges.setdefault((oi, si), set())
            for rtype, symi in rel:
                if rtype in SKIP_RELOCS:
                    continue
                name, _, _, styp, bind, shndx = o.symbols[symi]
                if shndx == SHN_UNDEF:
                    if name not in globs:
                        undefined.add(name)
                        continue
                    tgt = globs[name]
                elif shndx >= SHN_LORESERVE:
                    continue
                else:
                    tgt = (oi, shndx)
                out.add(tgt)
                # a function's address: a non-call relocation against its symbol. Jump tables
                # (.rodata.F -> .text.F) use local .L labels instead, which are not.
                if (rtype not in CALL_RELOCS and styp in (STT_FUNC, STT_SECTION)
                        and objs[tgt[0]].kind(tgt[1]) == "text"):
                    addr_taken.add(tgt)

    roots = [globs["sim_main"]]
    for node in sorted(addr_taken):
        oi, _ = node
        n = objs[oi].name
        if n.startswith("p_") and n not in PLAYSIM_EXCLUDE:
            roots.append(node)

    # breadth-first from the roots, remembering who reached each section (for `why`)
    parent = {r: None for r in roots}
    queue = deque(roots)
    while queue:
        node = queue.popleft()
        for tgt in sorted(edges.get(node, ())):
            if tgt not in parent and objs[tgt[0]].kind(tgt[1]) is not None:
                parent[tgt] = node
                queue.append(tgt)

    readonly = {(o, s) for o, syms in READONLY.items() for s in syms}
    placement = {"ROM": [], "DROM": [], "RAM_SMALL": [], "RAM": []}
    sim_funcs, sizes = [], {"ROM": 0, "DROM": 0, "RAM": 0}
    for node in sorted(parent, key=lambda n: (objs[n[0]].name, objs[n[0]].sections[n[1]][0])):
        oi, si = node
        o = objs[oi]
        name, _, _, size, *_ = o.sections[si]
        k = o.kind(si)
        var = name.split(".", 2)[-1] if name.count(".") >= 2 else name
        entry = {"obj": o.name, "path": o.path, "section": name, "size": size}
        if k == "text":
            region = "ROM"
            for sname, _, ssize, styp, _, shndx in o.symbols:
                if shndx == si and styp == STT_FUNC:
                    sim_funcs.append({"name": sname, "obj": o.name, "size": ssize})
        elif k == "rodata" or (o.name, var) in readonly:
            region = "DROM"
        else:
            region = "RAM_SMALL" if name.startswith((".sdata", ".sbss")) else "RAM"
        placement[region].append(entry)
        sizes["ROM" if region == "ROM" else "DROM" if region == "DROM" else "RAM"] += size

    suspicious = []
    for f in sim_funcs:
        if f["obj"].startswith(HOST_ONLY_PREFIXES) and f["name"] not in HOST_OK:
            suspicious.append(f)
    whys = {}
    for node, par in parent.items():
        whys[section_label(node)] = section_label(par) if par else None
    return {
        "placement": placement,
        "sim_functions": sim_funcs,
        "sizes": sizes,
        "roots": [section_label(r) for r in roots],
        "suspicious": suspicious,
        "undefined": sorted(undefined),
        "why": whys,
    }


def linker_script(a: dict) -> str:
    def lines(entries):
        # lld matches the file pattern against the path as given on the command line
        return "\n".join(f"    *{e['path'][e['path'].rindex('/', 0, e['path'].rindex('/')):]}"
                         f"({e['section']})" for e in entries)

    p = a["placement"]
    return f"""/* Generated by doom/split.py from the objects' relocations: do not edit.
   ROM / DROM / RAM: everything sim_main (P_Ticker) can reach; HOST: the rest. */
ENTRY(_start)
MEMORY
{{
  RAM  (rw)  : ORIGIN = {RAM_ORIGIN:#x}, LENGTH = {RAM_END - RAM_ORIGIN:#x}
  DROM (r)   : ORIGIN = {DROM_ORIGIN:#x}, LENGTH = {DROM_SIZE:#x}
  ROM  (rx)  : ORIGIN = {ROM_ORIGIN:#x}, LENGTH = {ROM_SIZE:#x}
  HOST (rwx) : ORIGIN = {HOST_ORIGIN:#x}, LENGTH = {HEAP_END - HOST_ORIGIN:#x}
}}
SECTIONS
{{
  .rom : {{
    __rom_start = .;
{lines(p["ROM"])}
    __rom_end = .;
  }} > ROM
  .drom : {{
    __drom_start = .;
{lines(p["DROM"])}
    . = ALIGN(4);
    __drom_end = .;
  }} > DROM
  .sim_sdata : {{
    __sim_ram_start = .;
{lines(p["RAM_SMALL"])}
  }} > RAM
  __global_pointer$ = __sim_ram_start + {GP_WINDOW:#x};
  .sim_data : {{
{lines(p["RAM"])}
    . = ALIGN(16);
    __sim_ram_end = .;
  }} > RAM

  .text : {{ *(.text._start) *(.text .text.*) }} > HOST
  .rodata : {{ *(.rodata .rodata.* .srodata .srodata.*) }} > HOST
  .data : {{ *(.data .data.* .sdata .sdata.*) }} > HOST
  .bss : {{ *(.bss .bss.* .sbss .sbss.*) . = ALIGN(16); }} > HOST
  _end = .;
  __heap_end = {HEAP_END:#x};
}}
"""


# --- the linked ELF ----------------------------------------------------------------------------

def elf_symbols(path: str) -> list[tuple[str, int, int, int]]:
    """(name, value, size, type) of the ELF's symbol table."""
    o = Obj.__new__(Obj)
    Obj.__init__(o, path)
    return [(n, v, s, t) for n, v, s, t, _, sh in o.symbols if sh != SHN_UNDEF]


def rom_words(path: str) -> dict[int, int]:
    """address -> instruction word of every allocated byte the ELF puts in ROM."""
    b = Path(path).read_bytes()
    shoff, = struct.unpack_from("<I", b, 0x20)
    shentsize, shnum = struct.unpack_from("<HH", b, 0x2E)
    words = {}
    for i in range(shnum):
        _, typ, flags, addr, off, size, *_ = struct.unpack_from("<10I", b, shoff + i * shentsize)
        if flags & SHF_ALLOC and typ != SHT_NOBITS and ROM_ORIGIN <= addr < ROM_ORIGIN + ROM_SIZE:
            for k in range(0, size - 3, 4):
                words[addr + k], = struct.unpack_from("<I", b, off + k)
    return words


def sim_isa_error(w: int, pc: int) -> str | None:
    """Why the instruction word w at pc is outside what the circuit runs (RV32IM + Zba, csrr time,
    ecall, FENCE; static branch / JAL targets inside ROM), or None."""
    opc, f3, f7, rs1 = w & 0x7F, w >> 12 & 7, w >> 25, w >> 15 & 31
    if opc == 0x6F or opc == 0x63:
        if opc == 0x6F:
            imm = ((w >> 31) << 20 | (w >> 12 & 0xFF) << 12 | (w >> 20 & 1) << 11
                   | (w >> 21 & 0x3FF) << 1)
            imm -= (w >> 31) << 21
        else:
            if f3 in (2, 3):
                return "bad branch"
            imm = (w >> 31) << 12 | (w >> 7 & 1) << 11 | (w >> 25 & 63) << 5 | (w >> 8 & 15) << 1
            imm -= (w >> 31) << 13
        t = pc + imm
        return None if ROM_ORIGIN <= t < ROM_ORIGIN + ROM_SIZE and not t & 3 else f"jump to {t:#x}"
    ok = {
        0x37: True, 0x17: True,
        0x67: f3 == 0,
        0x03: f3 in (0, 1, 2, 4, 5),
        0x23: f3 in (0, 1, 2),
        0x13: f3 not in (1, 5) or f7 == 0 or (f3 == 5 and f7 == 0x20),
        0x33: f7 in (0, 1) or (f7 == 0x20 and f3 in (0, 5)) or (f7 == 0x10 and f3 in (2, 4, 6)),
        0x0F: f3 == 0,  # FENCE; FENCE.I (f3 = 1) is not allowed
        0x73: w == 0x73 or (f3 in (2, 3, 6, 7) and rs1 == 0 and w >> 20 == 0xC01),
    }.get(opc, False)
    return None if ok else "not RV32IM+Zba / csrr time / ecall"


def check(elf: str, js: str) -> int:
    a = json.loads(Path(js).read_text())
    syms = elf_symbols(elf)
    by = {n: v for n, v, _, _ in syms}
    # every instruction of every sim function must be one the circuit runs
    words = rom_words(elf)
    isa_bad, rom_instrs = [], 0
    for n, v, size, t in syms:
        if t == STT_FUNC and ROM_ORIGIN <= v < ROM_ORIGIN + ROM_SIZE:
            for pc in range(v, v + size, 4):
                rom_instrs += 1
                err = sim_isa_error(words.get(pc, 0), pc)
                if err:
                    isa_bad.append(f"{n}+{pc - v:#x} ({pc:#x}): {words.get(pc, 0):#010x} {err}")
    sim_names = {f["name"] for f in a["sim_functions"]}
    bad = []
    rom_funcs = 0
    for n, v, size, t in syms:
        if t != STT_FUNC:
            continue
        in_rom = ROM_ORIGIN <= v < ROM_ORIGIN + ROM_SIZE
        rom_funcs += in_rom
        if in_rom != (n in sim_names) and not (n in sim_names and not in_rom and size == 0):
            bad.append(f"{n} at {v:#x} ({'ROM' if in_rom else 'not ROM'})")
    gc_dropped = sorted(n for n in sim_names if n not in by)
    rom = by["__rom_end"] - by["__rom_start"]
    drom = by["__drom_end"] - by["__drom_start"]
    ram = by["__sim_ram_end"] - RAM_ORIGIN
    small = sum(e["size"] for e in a["placement"]["RAM_SMALL"])
    zone = RAM_END - by["__sim_ram_end"]
    sim_sizes = sorted(((f["size"], f["name"]) for f in a["sim_functions"] if f["name"] in by),
                       reverse=True)
    report = {
        "rom_bytes": rom, "rom_functions": rom_funcs, "drom_rodata_bytes": drom,
        "ram_sim_data_bytes": ram, "ram_small_data_bytes": small,
        "gp": by["__global_pointer$"], "level_zone_bytes": zone,
        "level_zone_start": by["__sim_ram_end"], "drom_level_start": by["__drom_end"],
        "drom_level_bytes": DROM_ORIGIN + DROM_SIZE - by["__drom_end"],
        "rom_instructions": rom_instrs, "isa_errors": len(isa_bad),
        "sim_function_bytes": {"total": sum(s for s, _ in sim_sizes),
                               "median": sim_sizes[len(sim_sizes) // 2][0] if sim_sizes else 0},
        "largest_sim_functions": [[n, s] for s, n in sim_sizes[:10]],
        "dropped_by_gc": len(gc_dropped),
    }
    a["linked"] = report
    Path(js).write_text(json.dumps(a, indent=1))
    print(f"split: ROM {rom} B ({rom_funcs} functions), DROM .rodata {drom} B, sim RAM data "
          f"{ram} B ({small} B small, gp {by['__global_pointer$']:#x}), level zone {zone} B")
    if a["suspicious"]:
        print("split: host-side functions reachable from the sim:", file=sys.stderr)
        for f in a["suspicious"]:
            print(f"  {f['obj']}: {f['name']}", file=sys.stderr)
    if bad:
        print("split: ROM placement does not match the analysis:", *bad[:20], sep="\n  ",
              file=sys.stderr)
    if isa_bad:
        print("split: sim instructions the circuit cannot run:", *isa_bad[:20], sep="\n  ",
              file=sys.stderr)
    return 1 if bad or isa_bad else 0


def why(js: str, name: str) -> None:
    a = json.loads(Path(js).read_text())
    w = a["why"]
    keys = [k for k in w if k.endswith(f".{name})") or k.endswith(f"({name})")]
    for k in keys:
        chain = [k]
        while w.get(chain[-1]):
            chain.append(w[chain[-1]])
        print(" <- ".join(chain))
    if not keys:
        print(f"{name}: not sim")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    an = sub.add_parser("analyze")
    an.add_argument("--out-ld", required=True)
    an.add_argument("--out-json", required=True)
    an.add_argument("objs", nargs="+")
    ck = sub.add_parser("check")
    ck.add_argument("elf")
    ck.add_argument("json")
    wy = sub.add_parser("why")
    wy.add_argument("json")
    wy.add_argument("symbol")
    args = ap.parse_args()
    if args.cmd == "analyze":
        a = analyze(args.objs)
        Path(args.out_ld).write_text(linker_script(a))
        Path(args.out_json).write_text(json.dumps(a, indent=1))
        s = a["sizes"]
        print(f"split: {len(a['sim_functions'])} sim functions; ROM {s['ROM']} B, "
              f"DROM {s['DROM']} B,"
              f" RAM {s['RAM']} B (before linking); {len(a['roots'])} roots")
        return 0
    if args.cmd == "check":
        return check(args.elf, args.json)
    why(args.json, args.symbol)
    return 0


if __name__ == "__main__":
    sys.exit(main())
