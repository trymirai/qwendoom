"""ctypes bindings for build/libqdtrace.dylib (native/trace): the C reference traced machine
(``Ref``), the trace decoder or mirror (``Decoder``) and the lockstep checker (``Checker``).

    mem = Memory(Map.qwendoom()); mem.load_elf("build/doom.elf", skip_outside=True)
    ref = Ref(mem); ref.set_context(pc=sim_main, regs=regs); ref.push_input(pokes)
    toks, flags, why = ref.run(stop_yield=True)    # one sim tick: why == "yield"

    mirror = Memory.from_arrays(host.ram, host.drom, host.rom)   # the host's memory, shared
    dec = Decoder(mirror); dec.set_context(pc=sim_main, regs=regs)
    for t in model_tokens: ev = dec.feed(t)        # stores land in the host's RAM; ev < 0: error

    pred = start_stream(engine.step, dec, checker)              # after a resync: GO, to all three
    tick = drive_tick(engine.step, pred, dec, pokes, checker)   # one model-mode tick, injected

The record rules and the API are in docs/trace.md. The library is (re)built on first use when a
source under native/trace is newer than it (QD_TRACE_LIB=path loads another build instead).
"""

import ctypes as C
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from chipformer.rv import spec as S

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "native" / "trace"
LIB_PATH = ROOT / "build" / "libqdtrace.dylib"

MAX_RECORD = 16
CSR_INPUT = 0xC01
YIELD_BYTE = 256
GO = S.tok("GO")  # a run's first token (after a state snapshot): start_stream feeds it

# qd_ref status / qd_ref_run stop reasons / per-token flags
RUNNING, HALTED, FAULTED, NEED_INPUT, ERROR = range(5)
RUN_REASONS = ["limit", "full", "yield", "byte", "halt", "fault", "input", "error"]
STOP_YIELD, STOP_BYTE = 1, 2
TF_FIRST, TF_HOST = 1, 2
# qd_dec_feed results (>= 0) and errors (< 0)
MORE, COMMIT, BYTE, YIELD, HALT, FAULT = range(6)
DEC_STOP_INPUT = 1 << 8
DEC_ERRORS = {-1: "token", -2: "shape", -3: "refresh", -4: "value", -5: "div", -6: "jump",
              -7: "store", -8: "fault", -9: "ecall", -10: "pc", -11: "halted"}
# qd_chk states
CHK_OK, CHK_END, CHK_DIVERGED, CHK_ERROR = 0, 1, -1, -2


def name(t: int) -> str:
    """A token's name (``F1(5)``, ``NT``; ``PAD``; ``<bad t>``)."""
    if t == S.VOCAB:
        return "PAD"
    try:
        return S.name(t)
    except ValueError:
        return f"<bad {t}>"


class Map(C.Structure):
    """qd_map: the address map (regions and the input port CSR, -1 for none)."""

    _fields_ = [("ram_base", C.c_uint32), ("ram_size", C.c_uint32),
                ("drom_base", C.c_uint32), ("drom_size", C.c_uint32),
                ("rom_base", C.c_uint32), ("rom_size", C.c_uint32), ("input_csr", C.c_int32)]

    @classmethod
    def qwendoom(cls) -> "Map":
        m = cls()
        lib().qd_map_qwendoom(C.byref(m))
        return m

    @classmethod
    def upstream(cls) -> "Map":
        m = cls()
        lib().qd_map_upstream(C.byref(m))
        return m

    def regions(self) -> dict:
        return {"ram": (self.ram_base, self.ram_size), "drom": (self.drom_base, self.drom_size),
                "rom": (self.rom_base, self.rom_size)}

    def __repr__(self):
        return ("Map(" + ", ".join(f"{k}={b:#x}+{s:#x}" for k, (b, s) in self.regions().items())
                + f", input_csr={self.input_csr:#x})")


class _QdMem(C.Structure):
    _fields_ = [("ram", C.c_void_p), ("drom", C.c_void_p), ("rom", C.c_void_p),
                ("host", C.c_void_p)]


class Memory:
    """The RAM, DROM and ROM of a map as numpy byte arrays, passed to C as a qd_mem (no HOST: the
    trace components never touch it). ``Memory.wrap`` views a qd_mem owned by another library."""

    def __init__(self, map: Map | None = None):
        self.map = map or Map.qwendoom()
        self.ram = np.zeros(self.map.ram_size, np.uint8)
        self.drom = np.zeros(self.map.drom_size, np.uint8)
        self.rom = np.zeros(self.map.rom_size, np.uint8)
        self._sync()

    def _sync(self):
        self.c = _QdMem(self.ram.ctypes.data, self.drom.ctypes.data, self.rom.ctypes.data, None)

    @classmethod
    def wrap(cls, qd_mem_address: int, map: Map | None = None, own_ram: bool = False) -> "Memory":
        """Views (no copy) of an existing ``qd_mem`` at that address (e.g. the host's). own_ram:
        RAM is a private copy instead (DROM and ROM stay shared), as a checker's reference
        needs; ``sync_ram()`` copies the source's RAM into it again (at a resync)."""
        map = map or Map.qwendoom()
        src = _QdMem.from_address(qd_mem_address)

        def view(ptr, size):
            return np.ctypeslib.as_array(C.cast(ptr, C.POINTER(C.c_uint8)), (size,))

        return cls.from_arrays(view(src.ram, map.ram_size), view(src.drom, map.drom_size),
                               view(src.rom, map.rom_size), map, own_ram)

    @classmethod
    def from_arrays(cls, ram: np.ndarray, drom: np.ndarray, rom: np.ndarray,
                    map: Map | None = None, own_ram: bool = False) -> "Memory":
        """A Memory over existing contiguous uint8 arrays (e.g. qwendoom.host.Host's ``ram``,
        ``drom``, ``rom`` views); the caller keeps them alive. own_ram as in ``wrap``."""
        self = cls.__new__(cls)
        self.map = map or Map.qwendoom()
        for a, size in ((ram, self.map.ram_size), (drom, self.map.drom_size),
                        (rom, self.map.rom_size)):
            assert a.dtype == np.uint8 and a.flags.c_contiguous and a.size >= size, (a.dtype,
                                                                                     a.size)
        self._src_ram = ram
        self.ram = ram.copy() if own_ram else ram
        self.drom, self.rom = drom, rom
        self._sync()
        return self

    def sync_ram(self) -> None:
        """A wrapped Memory with its own RAM: copy the source's RAM into it (in place)."""
        src = getattr(self, "_src_ram", None)
        if src is not None and src is not self.ram:
            self.ram[:] = src

    def copy(self) -> "Memory":
        m = Memory(self.map)
        m.ram[:], m.drom[:], m.rom[:] = self.ram, self.drom, self.rom
        return m

    def region(self, addr: int, size: int = 1) -> tuple[np.ndarray, int]:
        """(the region's array, offset) holding [addr, addr + size)."""
        for k, (base, rsize) in self.map.regions().items():
            if base <= addr and addr + size <= base + rsize:
                return getattr(self, k), addr - base
        raise ValueError(f"{addr:#x} (+{size}) is outside RAM / DROM / ROM")

    def write(self, addr: int, data: bytes) -> None:
        arr, off = self.region(addr, len(data))
        arr[off:off + len(data)] = np.frombuffer(bytes(data), np.uint8)

    def read(self, addr: int, size: int) -> bytes:
        arr, off = self.region(addr, size)
        return arr[off:off + size].tobytes()

    def word(self, addr: int) -> int:
        return int.from_bytes(self.read(addr, 4), "little")

    def load_segments(self, segments, skip_outside: bool = False) -> None:
        """[(vaddr, bytes)] (chipformer.rv.elf.segments' PT_LOAD list) into their regions.
        skip_outside: skip segments in none of RAM / DROM / ROM (doom.elf's HOST ones) instead
        of raising."""
        for vaddr, blob in segments:
            if skip_outside and not any(b <= vaddr < b + s for b, s in self.map.regions().values()):
                continue
            self.write(vaddr, blob)

    def load_program(self, prog) -> int:
        """An assembler Program (text words from rom_base, data bytes at data_base, rodata from
        drom_base); returns its entry point."""
        text = np.asarray(prog.text, np.int64).astype("<u4").tobytes()
        self.write(self.map.rom_base, text)
        if len(prog.data):
            self.write(prog.data_base, bytes(prog.data))
        if len(getattr(prog, "rodata", b"")):
            self.write(self.map.drom_base, bytes(prog.rodata))
        return prog.entry

    def load_elf(self, path, skip_outside: bool = False) -> int:
        """Load a static ELF's PT_LOAD segments; returns the entry point (skip_outside as in
        load_segments)."""
        from chipformer.rv import elf

        entry, segs = elf.segments(path)
        self.load_segments(segs, skip_outside)
        return entry


def _context_args(m: Map, pc, regs, anchor, n):
    if anchor is None:
        anchor = (pc - m.rom_base) >> 2 if pc is not None else 0
    regs = regs if regs is not None else [0] * 32
    arr = (C.c_uint32 * 32)(*[int(v) & 0xFFFFFFFF for v in regs])
    return anchor, n, arr


class Ref:
    """The reference traced machine over a Memory (RAM written in place)."""

    def __init__(self, mem: Memory, map: Map | None = None):
        self.mem, self.map = mem, map or mem.map
        self.p = lib().qd_ref_new(C.byref(mem.c), C.byref(self.map))
        if not self.p:
            raise MemoryError("qd_ref_new failed")

    def __del__(self):
        if getattr(self, "p", None):
            lib().qd_ref_free(self.p)
            self.p = None

    def set_context(self, pc: int | None = None, regs=None, anchor: int | None = None,
                    n: int = 0) -> None:
        """Start at pc (or anchor + n) with x0..x31 = regs; clears status, counters, input."""
        a, n, arr = _context_args(self.map, pc, regs, anchor, n)
        lib().qd_ref_set_context(self.p, a, n, arr)

    @property
    def context(self) -> tuple[int, int, list[int]]:
        a, n, x = C.c_uint32(), C.c_uint32(), (C.c_uint32 * 32)()
        lib().qd_ref_get_context(self.p, C.byref(a), C.byref(n), x)
        return a.value, n.value, list(x)

    @property
    def regs(self) -> list[int]:
        return self.context[2]

    @property
    def pc(self) -> int:
        return lib().qd_ref_pc(self.p)

    def set_reg(self, i: int, v: int) -> None:
        lib().qd_ref_set_reg(self.p, i, v & 0xFFFFFFFF)

    @property
    def status(self) -> int:
        return lib().qd_ref_status(self.p)

    @property
    def exit_code(self) -> int:
        return lib().qd_ref_exit_code(self.p)

    @property
    def fault(self) -> tuple[int, int] | None:
        c, pc = C.c_int(), C.c_uint32()
        lib().qd_ref_fault(self.p, C.byref(c), C.byref(pc))
        return (c.value, pc.value) if c.value >= 0 else None

    @property
    def counts(self) -> dict:
        s, r, t = C.c_uint64(), C.c_uint64(), C.c_uint64()
        lib().qd_ref_counts(self.p, C.byref(s), C.byref(r), C.byref(t))
        return {"steps": s.value, "records": r.value, "tokens": t.value}

    @property
    def last_store(self) -> int | None:
        a = lib().qd_ref_last_store(self.p)
        return None if a == 0xFFFFFFFF else a

    def push_input(self, words) -> None:
        arr = np.ascontiguousarray(np.asarray(words, dtype=np.int64) & 0xFFFFFFFF, np.uint32)
        lib().qd_ref_push_input(self.p, arr.ctypes.data_as(C.POINTER(C.c_uint32)), len(arr))

    @property
    def input_pending(self) -> int:
        return lib().qd_ref_input_pending(self.p)

    def clear_input(self) -> None:
        lib().qd_ref_clear_input(self.p)

    @property
    def wants_input(self) -> bool:
        return bool(lib().qd_ref_wants_input(self.p))

    def record(self) -> tuple[list[int], bool]:
        """(the next record's tokens, host-owned); ([], False) if none can run (see status)."""
        buf, fl = (C.c_int32 * MAX_RECORD)(), C.c_int()
        n = lib().qd_ref_record(self.p, buf, C.byref(fl))
        return list(buf[:n]), bool(fl.value & TF_HOST)

    def run(self, max_records: int = -1, cap: int = 1 << 22, stop_yield: bool = False,
            stop_byte: bool = False) -> tuple[np.ndarray, np.ndarray, str]:
        """Records until a stop (see qd_ref_run): (tokens int32, per-token TF_* flags, reason)."""
        toks = np.empty(cap, np.int32)
        flags = np.empty(cap, np.uint8)
        why = C.c_int()
        n = lib().qd_ref_run(self.p, toks.ctypes.data_as(C.POINTER(C.c_int32)),
                             flags.ctypes.data_as(C.POINTER(C.c_uint8)), cap, max_records,
                             (STOP_YIELD if stop_yield else 0) | (STOP_BYTE if stop_byte else 0),
                             C.byref(why))
        return toks[:n], flags[:n], RUN_REASONS[why.value]

    def run_count(self, max_records: int = -1, stop_yield: bool = False) -> tuple[int, str]:
        """Run without keeping tokens: (tokens produced, reason)."""
        why = C.c_int()
        n = lib().qd_ref_run(self.p, None, None, 0, max_records,
                             STOP_YIELD if stop_yield else 0, C.byref(why))
        return n, RUN_REASONS[why.value]

    def trace(self, max_tokens: int = 1 << 26) -> tuple[np.ndarray, np.ndarray, str]:
        """Run to a stop other than a full buffer, in chunks: (tokens, flags, reason)."""
        parts, fparts, total = [], [], 0
        while True:
            t, f, why = self.run(cap=min(1 << 22, max_tokens - total + MAX_RECORD))
            parts.append(t)
            fparts.append(f)
            total += len(t)
            if why != "full" or total >= max_tokens:
                return np.concatenate(parts), np.concatenate(fparts), why


class DecodeError(Exception):
    pass


class Decoder:
    """The trace decoder over a Memory: tracks the context from the tokens, applies stores."""

    def __init__(self, mem: Memory, map: Map | None = None, strict: bool = True):
        self.mem, self.map = mem, map or mem.map
        self.p = lib().qd_dec_new(C.byref(mem.c), C.byref(self.map), int(strict))
        if not self.p:
            raise MemoryError("qd_dec_new failed")

    def __del__(self):
        if getattr(self, "p", None):
            lib().qd_dec_free(self.p)
            self.p = None

    def set_context(self, pc: int | None = None, regs=None, anchor: int | None = None,
                    n: int = 0) -> None:
        a, n, arr = _context_args(self.map, pc, regs, anchor, n)
        lib().qd_dec_set_context(self.p, a, n, arr)

    def feed(self, tok: int) -> int:
        """One token: MORE / COMMIT / BYTE / YIELD / HALT / FAULT, or a negative error."""
        return lib().qd_dec_feed(self.p, int(tok))

    def feed_many(self, toks, stop: int = 0) -> tuple[int, int]:
        """(last result, tokens consumed); stops at errors, HALT / FAULT, events in ``stop``
        (1 << result, or DEC_STOP_INPUT)."""
        arr = np.ascontiguousarray(toks, np.int32)
        used = C.c_long()
        r = lib().qd_dec_feed_many(self.p, arr.ctypes.data_as(C.POINTER(C.c_int32)), len(arr),
                                   stop, C.byref(used))
        return r, used.value

    def feed_checked(self, toks) -> int:
        """Feed all of toks; raise DecodeError on an error. Returns the last result."""
        r, used = self.feed_many(toks)
        if r < 0:
            raise DecodeError(f"{DEC_ERRORS.get(r, r)}: {self.error_text}")
        if used != len(toks):
            raise DecodeError(f"stream ended after {used} of {len(toks)} tokens")
        return r

    @property
    def context(self) -> tuple[int, int, list[int]]:
        a, n, x = C.c_uint32(), C.c_uint32(), (C.c_uint32 * 32)()
        lib().qd_dec_get_context(self.p, C.byref(a), C.byref(n), x)
        return a.value, n.value, list(x)

    @property
    def regs(self) -> list[int]:
        return self.context[2]

    @property
    def pc(self) -> int:
        return lib().qd_dec_pc(self.p)

    @property
    def wants_input(self) -> bool:
        return bool(lib().qd_dec_wants_input(self.p))

    @property
    def at_boundary(self) -> bool:
        return bool(lib().qd_dec_at_boundary(self.p))

    @property
    def error(self) -> int:
        return lib().qd_dec_error(self.p)

    @property
    def error_text(self) -> str:
        return lib().qd_dec_error_text(self.p).decode()

    @property
    def halted(self) -> int:
        return lib().qd_dec_halted(self.p)

    @property
    def exit_code(self) -> int:
        return lib().qd_dec_exit_code(self.p)

    @property
    def fault(self) -> tuple[int, int] | None:
        c, pc = C.c_int(), C.c_uint32()
        lib().qd_dec_fault(self.p, C.byref(c), C.byref(pc))
        return (c.value, pc.value) if c.value >= 0 else None

    @property
    def last_byte(self) -> int:
        return lib().qd_dec_last_byte(self.p)

    @property
    def last_store(self) -> int | None:
        a = lib().qd_dec_last_store(self.p)
        return None if a == 0xFFFFFFFF else a

    @property
    def counts(self) -> dict:
        v = [C.c_uint64() for _ in range(4)]
        lib().qd_dec_counts(self.p, *map(C.byref, v))
        return dict(zip(("tokens", "records", "instructions", "stores"), (x.value for x in v)))


@dataclass
class Divergence:
    index: int  # stream position of the first divergent token
    record: int
    record_start: int
    got: int
    want: int  # -1: the reference had ended; -2: a host-owned value record was due
    pc: int
    anchor: int
    n: int
    inst: int
    ref_record: list[int]

    def __str__(self):
        want = {-1: "the end (HALT / fault)", -2: "a value record (host-owned)"}.get(
            self.want, name(self.want))
        return (f"token {self.index} (record {self.record}, +{self.index - self.record_start}) "
                f"at pc {self.pc:#010x} (word {self.inst:#010x}, anchor {self.anchor} n {self.n}): "
                f"got {name(self.got)}, want {want}; reference record: "
                + " ".join(name(t) for t in self.ref_record))


class _QdDivergence(C.Structure):
    _fields_ = [("index", C.c_int64), ("record", C.c_int64), ("record_start", C.c_int64),
                ("got", C.c_int32), ("want", C.c_int32), ("pc", C.c_uint32),
                ("anchor", C.c_uint32), ("n", C.c_uint32), ("inst", C.c_uint32),
                ("ref_record", C.c_int32 * MAX_RECORD), ("ref_len", C.c_int32)]


class Checker:
    """Lockstep: the reference (from its current context) against a token stream."""

    def __init__(self, ref: Ref):
        self.ref = ref
        self.p = lib().qd_chk_new(ref.p)

    def __del__(self):
        if getattr(self, "p", None):
            lib().qd_chk_free(self.p)
            self.p = None

    def reset(self) -> None:
        lib().qd_chk_reset(self.p)

    def feed(self, tok: int) -> int:
        return lib().qd_chk_feed(self.p, int(tok))

    def check(self, toks) -> tuple[int, int]:
        """(tokens consumed, state): CHK_OK if all matched, CHK_END at the reference's end."""
        arr = np.ascontiguousarray(toks, np.int32)
        st = C.c_int()
        n = lib().qd_chk_stream(self.p, arr.ctypes.data_as(C.POINTER(C.c_int32)), len(arr),
                                C.byref(st))
        return n, st.value

    @property
    def state(self) -> int:
        return lib().qd_chk_state(self.p)

    @property
    def divergence(self) -> Divergence | None:
        if self.state != CHK_DIVERGED:
            return None
        d = _QdDivergence()
        lib().qd_chk_divergence(self.p, C.byref(d))
        return Divergence(d.index, d.record, d.record_start, d.got, d.want, d.pc, d.anchor, d.n,
                          d.inst, list(d.ref_record[:d.ref_len]))

    @property
    def counts(self) -> dict:
        v = [C.c_int64() for _ in range(3)]
        lib().qd_chk_counts(self.p, *map(C.byref, v))
        return dict(zip(("tokens", "records", "host_tokens"), (x.value for x in v)))


def encode_value_record(v: int) -> list[int]:
    """The tokens of a 32-bit word's value record (what the host injects for an input word)."""
    buf = (C.c_int32 * 3)()
    n = lib().qd_value_record(int(v) & 0xFFFFFFFF, buf)
    return list(buf[:n])


def div_record(op: str, a: int, b: int) -> list[int]:
    """The DIV-class record (op DIV / DIVU / REM / REMU, rd != x0)."""
    buf = (C.c_int32 * MAX_RECORD)()
    n = lib().qd_div_record(["DIV", "DIVU", "REM", "REMU"].index(op), a & 0xFFFFFFFF,
                            b & 0xFFFFFFFF, buf)
    return list(buf[:n])


def roundtrip(ref: Ref, dec: Decoder, max_records: int = -1) -> int:
    """Run ref, feed its records to dec, compare contexts at every record and the stored words
    at every store (ref and dec on separate Memories), RAM at the end. Returns the records
    checked; raises AssertionError at the first mismatch."""
    msg = C.create_string_buffer(512)
    n = lib().qd_roundtrip(ref.p, C.byref(ref.mem.c), dec.p, C.byref(dec.mem.c),
                           C.byref(ref.map), max_records, msg, len(msg))
    if n < 0:
        raise AssertionError(msg.value.decode())
    return n


# --- the model-mode loop ----------------------------------------------------------------------


class CheckError(Exception):
    """The lockstep checker refused a model token (``divergence``; None: the reference could not
    run, see its status)."""

    def __init__(self, divergence: Divergence | None, msg: str = ""):
        super().__init__(msg or str(divergence))
        self.divergence = divergence


@dataclass
class Tick:
    """One sim tick driven through the model (``drive_tick``)."""

    tokens: np.ndarray  # int32: every token fed to the model, in order
    host: np.ndarray  # bool per token: host-injected (an input record), not the model's
    event: int  # the decoder event that ended it: YIELD, HALT or FAULT
    pred: int  # the model's prediction after the last token (None after HALT / FAULT)
    unread: list[int]  # input words the sim did not read


def start_stream(step, dec: Decoder, checker: Checker | None = None, go: int = GO) -> int:
    """After a resync (dec.set_context; the checker's reference's set_context, then its reset;
    the snapshot loaded into the engine): the run's first token, GO, to the decoder, to
    ``checker`` if given, then to the model. Returns step(GO), the first tick's pred. The tick's
    first token is then stream token 1, where both reject a GO (fresh, either takes one at token
    0 as the start of a run, so a model predicting GO after GO would pass). GO is no tick's token
    (Tick.tokens / host are the tick's own). Raises DecodeError or CheckError."""
    if dec.feed(go) < 0:
        raise DecodeError(f"GO: {dec.error_text}")
    if checker is not None and checker.feed(go) != CHK_OK:
        raise CheckError(checker.divergence, "the checker refused GO")
    return step(go)


def drive_tick(step, pred: int, dec: Decoder, words, checker: Checker | None = None,
               max_tokens: int = 1 << 24) -> Tick:
    """One sim tick in the model / check modes. step(token) feeds one token to the model and
    returns its greedy prediction; pred is the prediction pending now (after a resync, the one
    start_stream returned). At every record boundary where the decoder wants input, the next word of
    ``words`` is injected (its value record replaces the model's predictions, token by token);
    everywhere else the model's prediction is the token. Each token goes to the decoder (stores
    land in its Memory), to ``checker`` if given, then to the model. Ends after YIELD (stepped:
    its prediction starts the next tick), HALT or a fault record. Raises DecodeError (an illegal
    token, input wanted with no word left, or no end within max_tokens) or CheckError."""
    words = [int(w) & 0xFFFFFFFF for w in words][::-1]
    toks, host = [], []
    ev = MORE
    while ev not in (YIELD, HALT, FAULT):
        if len(toks) >= max_tokens:
            raise DecodeError(f"no YIELD / HALT within {max_tokens} tokens (pc {dec.pc:#x})")
        inject = dec.wants_input
        if inject and not words:
            raise DecodeError(f"the sim reads input at pc {dec.pc:#x} with no word left")
        for t in encode_value_record(words.pop()) if inject else [pred]:
            ev = dec.feed(t)
            if ev < 0:
                raise DecodeError(f"{DEC_ERRORS.get(ev, ev)}: {dec.error_text}")
            if checker is not None:
                st = checker.feed(t)
                if st == CHK_DIVERGED:
                    raise CheckError(checker.divergence)
                if st == CHK_ERROR:
                    raise CheckError(None, "the reference cannot run (status "
                                           f"{checker.ref.status})")
            toks.append(t)
            host.append(inject)
            pred = step(t) if ev not in (HALT, FAULT) else None
    return Tick(np.asarray(toks, np.int32), np.asarray(host, bool), ev, pred, words[::-1])


class RefModel:
    """A stand-in for the engine (test support for a model-mode runner): a perfect model that
    predicts the reference's tokens, and at an input record the circuit's view of the csrr stub
    (F0(0), then the reference's tokens of the real word; the runner overrides them all). Queue
    each tick's input words on ``ref`` before driving it. step(GO) before the first token is the
    run's start (start_stream; no reference token): it returns predict(). corrupt: {index:
    token} makes those predictions wrong (for testing checkers; index: the tokens stepped
    before it, GO not counted, so a tick token's index in the first tick)."""

    def __init__(self, ref: Ref, corrupt: dict | None = None):
        self.ref, self.queue, self.fed, self.corrupt = ref, [], 0, dict(corrupt or {})

    def _fill(self) -> bool:
        if not self.queue:
            if self.ref.wants_input and not self.ref.input_pending:
                return False  # the next tick's words are not queued yet
            rec, owned = self.ref.record()
            if not rec:
                raise RuntimeError(f"the reference cannot run (status {self.ref.status})")
            self.queue = [(t, owned and i == 0) for i, t in enumerate(rec)]
        return True

    def predict(self) -> int:
        """The prediction for the next position (what step(GO) returns after a resync)."""
        if self.fed in self.corrupt:
            return self.corrupt[self.fed]
        if not self._fill():
            return S.tok("F0", 0)
        t, stub = self.queue[0]
        return S.tok("F0", 0) if stub else t

    def step(self, tok: int) -> int:
        if tok == GO and not self.fed:  # the run's start
            return self.predict()
        self._fill()
        want, _ = self.queue.pop(0)
        if tok != want:  # (a perfect model cannot follow a stream that left the reference)
            raise RuntimeError(f"RefModel: token {self.fed} is {name(tok)}, the reference's "
                               f"{name(want)}")
        self.fed += 1
        return self.predict()


# --- the library ------------------------------------------------------------------------------

_LIB = None


def build(force: bool = False) -> Path:
    """make -C native/trace when the dylib is missing or older than a source."""
    srcs = [*SRC.glob("*.c"), *SRC.glob("*.h"), SRC / "Makefile", ROOT / "native" / "qd_mem.h"]
    if force or not LIB_PATH.exists() or LIB_PATH.stat().st_mtime < max(
            p.stat().st_mtime for p in srcs):
        subprocess.run(["make", "-s", "-C", str(SRC)], check=True)
    return LIB_PATH


def lib() -> C.CDLL:
    global _LIB
    if _LIB is None:
        # QD_TRACE_LIB: another build of the library (e.g. with sanitizers)
        L = C.CDLL(os.environ.get("QD_TRACE_LIB") or str(build()))
        u32, i32, vp, sz = C.c_uint32, C.c_int32, C.c_void_p, C.c_size_t
        pu32, pi32 = C.POINTER(C.c_uint32), C.POINTER(C.c_int32)
        sig = {
            "qd_map_qwendoom": (None, [vp]), "qd_map_upstream": (None, [vp]),
            "qd_value_record": (C.c_int, [u32, pi32]),
            "qd_div_record": (C.c_int, [C.c_int, u32, u32, pi32]),
            "qd_ref_new": (vp, [vp, vp]), "qd_ref_free": (None, [vp]),
            "qd_ref_set_context": (None, [vp, u32, u32, pu32]),
            "qd_ref_get_context": (None, [vp, pu32, pu32, pu32]),
            "qd_ref_pc": (u32, [vp]), "qd_ref_inst": (u32, [vp]),
            "qd_ref_set_reg": (None, [vp, C.c_int, u32]),
            "qd_ref_status": (C.c_int, [vp]), "qd_ref_exit_code": (u32, [vp]),
            "qd_ref_fault": (None, [vp, C.POINTER(C.c_int), pu32]),
            "qd_ref_counts": (None, [vp] + [C.POINTER(C.c_uint64)] * 3),
            "qd_ref_last_store": (u32, [vp]),
            "qd_ref_push_input": (None, [vp, pu32, sz]),
            "qd_ref_input_pending": (sz, [vp]), "qd_ref_clear_input": (None, [vp]),
            "qd_ref_wants_input": (C.c_int, [vp]),
            "qd_ref_record": (C.c_int, [vp, pi32, C.POINTER(C.c_int)]),
            "qd_ref_run": (C.c_long, [vp, pi32, C.POINTER(C.c_uint8), C.c_long, C.c_long, C.c_int,
                                      C.POINTER(C.c_int)]),
            "qd_dec_new": (vp, [vp, vp, C.c_int]), "qd_dec_free": (None, [vp]),
            "qd_dec_set_context": (None, [vp, u32, u32, pu32]),
            "qd_dec_get_context": (None, [vp, pu32, pu32, pu32]),
            "qd_dec_pc": (u32, [vp]), "qd_dec_feed": (C.c_int, [vp, i32]),
            "qd_dec_feed_many": (C.c_int, [vp, pi32, C.c_long, C.c_int, C.POINTER(C.c_long)]),
            "qd_dec_wants_input": (C.c_int, [vp]), "qd_dec_at_boundary": (C.c_int, [vp]),
            "qd_dec_error": (C.c_int, [vp]), "qd_dec_error_text": (C.c_char_p, [vp]),
            "qd_dec_halted": (C.c_int, [vp]), "qd_dec_exit_code": (u32, [vp]),
            "qd_dec_fault": (None, [vp, C.POINTER(C.c_int), pu32]),
            "qd_dec_last_byte": (u32, [vp]), "qd_dec_last_store": (u32, [vp]),
            "qd_dec_counts": (None, [vp] + [C.POINTER(C.c_uint64)] * 4),
            "qd_chk_new": (vp, [vp]), "qd_chk_free": (None, [vp]), "qd_chk_reset": (None, [vp]),
            "qd_chk_feed": (C.c_int, [vp, i32]),
            "qd_chk_stream": (C.c_long, [vp, pi32, C.c_long, C.POINTER(C.c_int)]),
            "qd_chk_state": (C.c_int, [vp]), "qd_chk_divergence": (None, [vp, vp]),
            "qd_chk_counts": (None, [vp] + [C.POINTER(C.c_int64)] * 3),
            "qd_roundtrip": (C.c_long, [vp, vp, vp, vp, vp, C.c_long, C.c_char_p, C.c_int]),
        }
        for fn, (res, args) in sig.items():
            f = getattr(L, fn)
            f.restype, f.argtypes = res, args
        _LIB = L
    return _LIB
