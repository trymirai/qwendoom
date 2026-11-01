"""ctypes bindings for the host computer (native/host -> build/libqdhost.dylib).

The host interpreter runs DOOM's host side (build/doom.elf: main loop, renderer, menus, level
loading) and stops at events: a frame is ready, the host wants one sim tick (P_Ticker), exit, error.
A runner answers SIM_TICK in one of the sim modes of ARCHITECTURE.md; ``Host.sim_tick_native``
is the native one (the sim context of the same interpreter, with region checks):

    h = Host()                       # Freedoom E1M1, skill 3
    while True:
        ev = h.run()
        if ev == FRAME: idx, pal = h.frame()
        elif ev == SIM_TICK:
            if h.need_resync():                     # a level was loaded
                # (model modes: snapshot h.ram, h.drom, h.rom and h.sim_entry() here)
                h.resync()
            pokes = h.take_pokes()                  # the tick's input words (csrr time)
            stats = h.sim_tick_native()             # (model modes: run the model to YIELD,
            h.complete_sim_tick()                   #  its stores mirrored into h.ram)

See docs/doom.md for the C API this wraps.
"""

import ctypes
from ctypes import POINTER, c_char_p, c_int, c_int64, c_uint8, c_uint32, c_uint64, c_void_p
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LIB = ROOT / "build" / "libqdhost.dylib"
DEFAULT_ELF = ROOT / "build" / "doom.elf"
DEFAULT_WAD = ROOT / "third_party" / "freedoom" / "freedoom1.wad"
DEFAULT_ARGS = ("-warp", "1", "1", "-skill", "3")  # straight into E1M1, Hurt Me Plenty

# qd_host_run events (native/host/qd_host_abi.h)
BUDGET, FRAME, SIM_TICK, EXIT, ERROR = 0, 1, 2, 3, -1

# the address map (native/qd_mem.h)
RAM_BASE, RAM_SIZE = 0x0, 0x40000
DROM_BASE, DROM_SIZE = 0x80000, 0x80000
ROM_BASE, ROM_SIZE = 0x100000, 0x40000
HOST_BASE, HOST_SIZE = 0x1000000, 0x4000000

# DOOM key codes (doomgeneric's doomkeys.h) and the UI actions of ARCHITECTURE.md
KEY_RIGHTARROW, KEY_LEFTARROW, KEY_UPARROW, KEY_DOWNARROW = 0xAE, 0xAC, 0xAD, 0xAF
KEY_STRAFE_L, KEY_STRAFE_R, KEY_USE, KEY_FIRE = 0xA0, 0xA1, 0xA2, 0xA3
KEY_ESCAPE, KEY_ENTER, KEY_RSHIFT = 27, 13, 0x80 + 0x36
ACTION_KEYS = {
    "left": KEY_LEFTARROW, "right": KEY_RIGHTARROW, "forward": KEY_UPARROW,
    "back": KEY_DOWNARROW, "strafeleft": KEY_STRAFE_L, "straferight": KEY_STRAFE_R,
    "fire": KEY_FIRE, "use": KEY_USE, "run": KEY_RSHIFT, "menu": KEY_ESCAPE, "enter": KEY_ENTER,
    "weapon1": ord("1"), "weapon2": ord("2"), "weapon3": ord("3"),
}

COUNTERS = ["frames", "sim_ticks", "resyncs", "levels", "drom_writes", "drom_writes_midlevel",
            "ram_writes", "pokes", "host_null_accesses", "sim_tokens"]


class SimStats(ctypes.Structure):
    """One native sim tick (qd_sim_stats)."""
    _fields_ = [("instructions", c_uint64), ("tokens", c_uint64), ("input_words", c_uint32),
                ("loads", c_uint32), ("stores", c_uint32), ("drom_loads", c_uint32),
                ("min_sp", c_uint32), ("ops", c_uint32 * 64)]

    def op_counts(self, lib) -> dict:
        return {lib.qd_host_uop_name(i).decode(): self.ops[i] for i in range(64) if self.ops[i]}


class QdMem(ctypes.Structure):
    _fields_ = [("ram", c_void_p), ("drom", c_void_p), ("rom", c_void_p), ("host", c_void_p)]


class HostError(RuntimeError):
    pass


_lib = None


def lib():
    """The loaded libqdhost (built by `make -C native/host`)."""
    global _lib
    if _lib is None:
        L = ctypes.CDLL(str(LIB))
        sig = {
            "qd_host_new": (c_void_p, [c_char_p, c_char_p, c_int, POINTER(c_char_p)]),
            "qd_host_free": (None, [c_void_p]),
            "qd_host_set_config_dir": (None, [c_void_p, c_char_p]),
            "qd_host_set_echo": (None, [c_void_p, c_int]),
            "qd_host_run": (c_int, [c_void_p, c_uint64]),
            "qd_host_mem": (POINTER(QdMem), [c_void_p]),
            "qd_host_error": (c_char_p, [c_void_p]),
            "qd_host_exit_code": (c_int, [c_void_p]),
            "qd_host_frame": (c_int, [c_void_p, POINTER(c_uint8), POINTER(c_uint8)]),
            "qd_host_key": (None, [c_void_p, c_int, c_int]),
            "qd_host_action_key": (c_int, [c_char_p]),
            "qd_host_take_pokes": (c_int, [c_void_p, POINTER(c_uint32), c_int]),
            "qd_host_pending_pokes": (c_int, [c_void_p]),
            "qd_host_need_resync": (c_int, [c_void_p]),
            "qd_host_sim_entry": (None, [c_void_p, POINTER(c_uint32), POINTER(c_uint32),
                                         POINTER(c_uint32)]),
            "qd_host_resync": (None, [c_void_p]),
            "qd_host_sim_tick_native": (c_int64, [c_void_p, POINTER(SimStats)]),
            "qd_host_complete_sim_tick": (c_int, [c_void_p]),
            "qd_host_instructions": (c_uint64, [c_void_p]),
            "qd_host_sim_instructions": (c_uint64, [c_void_p]),
            "qd_host_counters": (c_int, [c_void_p, POINTER(c_uint64), c_int]),
            "qd_host_symbol": (c_uint32, [c_void_p, c_char_p]),
            "qd_host_symbolize": (c_int, [c_void_p, c_uint32, c_char_p, c_int]),
            "qd_host_uop_name": (c_char_p, [c_int]),
            "qd_host_output": (c_int, [c_void_p, c_char_p, c_int]),
        }
        for name, (res, args) in sig.items():
            f = getattr(L, name)
            f.restype, f.argtypes = res, args
        _lib = L
    return _lib


class Host:
    """One host computer running DOOM (see the module doc)."""

    def __init__(self, elf=DEFAULT_ELF, wad=DEFAULT_WAD, args=DEFAULT_ARGS, config_dir=None,
                 echo=False):
        L = lib()
        argv = (c_char_p * max(len(args), 1))(*[str(a).encode() for a in args])
        self._h = L.qd_host_new(str(elf).encode(), str(wad).encode(), len(args), argv)
        if not self._h:
            raise HostError(f"cannot start the host on {elf} / {wad}")
        if config_dir is not None:
            L.qd_host_set_config_dir(self._h, str(config_dir).encode())
        L.qd_host_set_echo(self._h, int(echo))
        m = L.qd_host_mem(self._h).contents
        # numpy views of the four regions (live: they see every write)
        self.ram = np.ctypeslib.as_array(ctypes.cast(m.ram, POINTER(c_uint8)), (RAM_SIZE,))
        self.drom = np.ctypeslib.as_array(ctypes.cast(m.drom, POINTER(c_uint8)), (DROM_SIZE,))
        self.rom = np.ctypeslib.as_array(ctypes.cast(m.rom, POINTER(c_uint8)), (ROM_SIZE,))
        self.host = np.ctypeslib.as_array(ctypes.cast(m.host, POINTER(c_uint8)), (HOST_SIZE,))
        self._idx = (c_uint8 * 64000)()
        self._pal = (c_uint8 * 768)()
        self._pokes = (c_uint32 * (2 * RAM_SIZE // 4 + 1))()

    @property
    def mem_address(self) -> int:
        """The address of the host's ``qd_mem`` (for ``qwendoom.trace.Memory.wrap``)."""
        return ctypes.addressof(lib().qd_host_mem(self._h).contents)

    # --- lifecycle and the host CPU ---

    def close(self):
        if self._h:
            lib().qd_host_free(self._h)
            self._h = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def run(self, max_instr: int = 1 << 40) -> int:
        """Run the host CPU until an event (FRAME, SIM_TICK, EXIT, ERROR) or the budget."""
        return lib().qd_host_run(self._h, max_instr)

    @property
    def error(self) -> str:
        return lib().qd_host_error(self._h).decode(errors="replace")

    @property
    def exit_code(self) -> int:
        return lib().qd_host_exit_code(self._h)

    def instructions(self) -> int:
        return lib().qd_host_instructions(self._h)

    def sim_instructions(self) -> int:
        return lib().qd_host_sim_instructions(self._h)

    def counters(self) -> dict:
        out = (c_uint64 * len(COUNTERS))()
        lib().qd_host_counters(self._h, out, len(COUNTERS))
        return dict(zip(COUNTERS, out))

    def output(self) -> str:
        """Drain what the guest printed (stdout and stderr)."""
        buf = ctypes.create_string_buffer(1 << 16)
        n = lib().qd_host_output(self._h, buf, len(buf))
        return buf.raw[:n].decode(errors="replace")

    # --- I/O ---

    def frame(self) -> tuple[np.ndarray, np.ndarray]:
        """(320x200 palette indices, 256x3 RGB palette) of the latest frame."""
        if lib().qd_host_frame(self._h, self._idx, self._pal) < 0:
            raise HostError("no frame yet")
        idx = np.frombuffer(self._idx, np.uint8).reshape(200, 320).copy()
        pal = np.frombuffer(self._pal, np.uint8).reshape(256, 3).copy()
        return idx, pal

    def frame_rgb(self) -> np.ndarray:
        idx, pal = self.frame()
        return pal[idx]

    def key(self, doom_key: int, down: bool):
        lib().qd_host_key(self._h, doom_key, int(down))

    def action(self, name: str, down: bool):
        """A UI action of ARCHITECTURE.md (forward, fire, use, menu, ...)."""
        self.key(ACTION_KEYS[name], down)

    # --- the sim ---

    def need_resync(self) -> bool:
        return bool(lib().qd_host_need_resync(self._h))

    def sim_entry(self) -> tuple[int, int, int]:
        """(pc, sp, gp) the sim starts from after a resync."""
        pc, sp, gp = c_uint32(), c_uint32(), c_uint32()
        lib().qd_host_sim_entry(self._h, pc, sp, gp)
        return pc.value, sp.value, gp.value

    def resync(self):
        lib().qd_host_resync(self._h)

    def pending_pokes(self) -> int:
        return lib().qd_host_pending_pokes(self._h)

    def take_pokes(self) -> np.ndarray:
        """The tick's input words: sorted (addr, value) pairs, then 0."""
        n = lib().qd_host_take_pokes(self._h, self._pokes, len(self._pokes))
        if n < 0:
            raise HostError("poke buffer too small")
        return np.frombuffer(self._pokes, np.uint32, n).copy()

    def sim_tick_native(self) -> SimStats:
        st = SimStats()
        if lib().qd_host_sim_tick_native(self._h, st) < 0:
            raise HostError(self.error)
        return st

    def complete_sim_tick(self):
        if lib().qd_host_complete_sim_tick(self._h) < 0:
            raise HostError("no sim tick in progress")

    def native_tick(self) -> tuple[SimStats, np.ndarray]:
        """Answer a SIM_TICK in native mode: resync if needed, pokes, the tick, resume."""
        if self.need_resync():
            self.resync()
        pokes = self.take_pokes()
        st = self.sim_tick_native()
        self.complete_sim_tick()
        return st, pokes

    def next_frame(self, on_tick=None) -> int:
        """Run (native sim) until the next frame; returns the event that stopped it (FRAME or
        EXIT). on_tick(stats, pokes) sees every sim tick."""
        while True:
            ev = self.run()
            if ev == SIM_TICK:
                st, pokes = self.native_tick()
                if on_tick:
                    on_tick(st, pokes)
            elif ev in (FRAME, EXIT):
                return ev
            elif ev == ERROR:
                raise HostError(self.error)

    # --- symbols ---

    def symbol(self, name: str) -> int:
        return lib().qd_host_symbol(self._h, name.encode())

    def symbolize(self, addr: int) -> str:
        buf = ctypes.create_string_buffer(256)
        lib().qd_host_symbolize(self._h, addr, buf, len(buf))
        return buf.value.decode()

    def word(self, addr: int) -> int:
        """A 32-bit word of guest memory (any region)."""
        for base, size, arr in ((RAM_BASE, RAM_SIZE, self.ram), (DROM_BASE, DROM_SIZE, self.drom),
                                (ROM_BASE, ROM_SIZE, self.rom), (HOST_BASE, HOST_SIZE, self.host)):
            if base <= addr < base + size:
                return int(arr[addr - base:addr - base + 4].view(np.uint32)[0])
        raise ValueError(f"{addr:#x} is outside the address map")

    def uop_name(self, i: int) -> str:
        return lib().qd_host_uop_name(i).decode()
