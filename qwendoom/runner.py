"""The runner: DOOM on the host computer, its sim ticks answered in one of the sim modes.

    r = Runner("model")                  # or "check", "native", "ref"
    while True:
        idx, pal = r.step_frame()        # runs the host to its next frame (and any sim tick)
        r.action("forward", True)        # keys go to the host between frames
    print(r.tokens, r.ticks, r.frames, r.tps)

Modes (ARCHITECTURE.md, "Sim modes"):

  model   the transformer (native/engine) runs sim_main: its predictions are the tokens, the host
          injects the input words' value records, the trace decoder mirrors its stores into the
          host's RAM
  check   model, with the C reference machine in lockstep: every model token must equal the
          reference's (a CheckError at the first divergence, with its context)
  ref     the C reference machine produces the tokens; the decoder mirrors them the same way
  native  the host interpreter's sim context runs sim_main (fast, for development)

A model / check / ref tick runs entirely in C (native/runner, build/libqdrun.dylib; the GIL is
released while it runs): another thread can read ``live_tokens`` and call ``cancel`` / ``pause``
meanwhile. Every mode consumes the same input words, so token and instruction counts agree
across modes. A resync (level start, death and reborn, loading a save) loads the host's RAM,
DROM, ROM and the sim's entry context into the model (``qwendoom.snapshot.state_rows``).
docs/run.md has the details and the measurements.
"""

from __future__ import annotations

import ctypes as C
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from . import trace as T
from .host import (DEFAULT_ARGS, DEFAULT_ELF, DEFAULT_WAD, ERROR, EXIT, FRAME, SIM_TICK, Host,
                   HostError, ACTION_KEYS)

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "native" / "runner"
LIB_PATH = ROOT / "build" / "libqdrun.dylib"
DEFAULT_WEIGHTS = ROOT / "weights"

MODES = ("model", "check", "native", "ref")
# The engine's threads: 1 runs real E1M1 ticks fastest on the M5 Max (31.9k tok/s, against 30.8k
# with 2 and 30.7k with 4; docs/run.md). QD_THREADS overrides.
DEFAULT_THREADS = int(os.environ.get("QD_THREADS") or 1)
MODEL_MODES = ("model", "check")

# qd_run_tick results
TICK_YIELD, TICK_HALT, TICK_FAULT, TICK_CANCELLED = 0, 1, 2, 3
TICK_ERRORS = {-1: "decode", -2: "input", -3: "diverged", -4: "ref", -5: "limit", -6: "state"}

# A button script for Freedoom E1M1 from the start (after the level start's ~38-frame wipe): along
# the hall, round the corner, a right turn to the door at (960, -152), open it, walk into the room
# beyond; shots on the way. 400 tics = 362 sim ticks.
E1M1_WALK = ("forward@40-128+141-225+254-400,right@80-92+226-253,left@128-140,use@296-298,"
             "fire@200-203+215-218+330-333+345-348+380-383")

# button scripts: the UI actions (host.ACTION_KEYS) and a few aliases
ALIASES = {"turnleft": "left", "turnright": "right", "backward": "back", "backwards": "back",
           "open": "use", "shoot": "fire", "esc": "menu", "escape": "menu", "strafe_left":
           "strafeleft", "strafe_right": "straferight"}


class RunError(RuntimeError):
    """The sim cannot go on: it exited (``SimHalt``), faulted, or a tick was cancelled."""


class SimHalt(RunError):
    """The sim called I_Error (ecall exit): the message is formatted from sim RAM."""


class SimFault(RunError):
    """The sim's CPU faulted (a FLT record)."""


class Cancelled(RunError):
    """``Runner.cancel()`` stopped a tick midway: the runner cannot go on."""


class TickStatsC(C.Structure):
    """qd_tick_stats."""
    _fields_ = [("tokens", C.c_uint64), ("injected", C.c_uint64), ("forced", C.c_uint64),
                ("instructions", C.c_uint64), ("records", C.c_uint64), ("words", C.c_uint64),
                ("unread", C.c_uint64), ("seconds", C.c_double), ("paused", C.c_double),
                ("event", C.c_int32), ("result", C.c_int32)]


@dataclass
class Tick:
    """One sim tick, in any mode."""

    index: int  # sim ticks before this one (0-based)
    frame: int  # frames shown before it
    tokens: int  # trace tokens (the injected ones included)
    injected: int  # host-injected tokens (input records)
    instructions: int
    words: int  # input words (the pokes: 2 per poke + the closing 0)
    seconds: float  # wall time of the tick (pauses excluded)
    resync: bool  # a resync came right before it
    forced: int = 0  # injected tokens fed to the engine as forced steps (model modes)

    @property
    def tps(self) -> float:
        return self.tokens / self.seconds if self.seconds > 0 else 0.0


# --- the library ------------------------------------------------------------------------------

_LIB = None


def build(force: bool = False) -> Path:
    """make -C native/runner, every time (~10 ms when the dylib is current): make decides from
    the Makefile's prerequisites, which include the engine, trace and qd_mem headers compiled
    into it (their struct layouts must match the libraries it calls), not just native/runner's
    files. force: remake the dylib even when make finds it current."""
    subprocess.run(["make", "-s", "-C", str(SRC), *(["-W", "qd_run.c"] if force else [])],
                   check=True)
    return LIB_PATH


def lib() -> C.CDLL:
    global _LIB
    if _LIB is None:
        from . import engine as E
        # load the engine and the trace library first: dyld then binds libqdrun's
        # @rpath/libqdengine.dylib and @rpath/libqdtrace.dylib to those images (by install name;
        # a QD_ENGINE_LIB / QD_TRACE_LIB build included), so the runner calls the code that made
        # the Python objects it is given. Check that nothing loaded a second copy.
        T.lib()
        E.lib()
        L = C.CDLL(str(build()))
        libs = _loaded_images()
        for name in ("libqdengine", "libqdtrace"):
            copies = [p for p in libs if Path(p).name.startswith(name)]
            if len(copies) != 1:
                raise RuntimeError(f"the runner needs one {name}.dylib, loaded: {copies}")
        vp, u64 = C.c_void_p, C.c_uint64
        sig = {
            "qd_run_new": (vp, [vp, vp, vp]), "qd_run_new_ref": (vp, [vp, vp]),
            "qd_run_free": (None, [vp]), "qd_run_set_pred": (None, [vp, C.c_int32]),
            "qd_run_pred": (C.c_int32, [vp]),
            "qd_run_tick": (C.c_int, [vp, C.POINTER(C.c_uint32), C.c_int,
                                      C.POINTER(TickStatsC)]),
            "qd_run_live_tokens": (u64, [vp]), "qd_run_cancel": (None, [vp]),
            "qd_run_pause": (None, [vp, C.c_int]), "qd_run_set_forced": (None, [vp, C.c_int]),
            "qd_run_set_max_tokens": (None, [vp, u64]), "qd_run_error": (C.c_char_p, [vp]),
            "qd_run_last": (None, [vp, C.POINTER(TickStatsC)]), "qd_run_set_qos": (C.c_int, []),
            "qd_run_set_log": (None, [vp, C.POINTER(C.c_int32), C.POINTER(C.c_uint8), u64]),
            "qd_run_log_len": (u64, [vp]),
        }
        for fn, (res, args) in sig.items():
            f = getattr(L, fn)
            f.restype, f.argtypes = res, args
        _LIB = L
    return _LIB


def _loaded_images() -> list[str]:
    dyld = C.CDLL(None)
    dyld._dyld_image_count.restype = C.c_uint32
    dyld._dyld_get_image_name.restype = C.c_char_p
    dyld._dyld_get_image_name.argtypes = [C.c_uint32]
    return [dyld._dyld_get_image_name(i).decode() for i in range(dyld._dyld_image_count())]


def set_qos() -> bool:
    """Make the calling thread user-interactive (macOS keeps it on the performance cores)."""
    return lib().qd_run_set_qos() == 0


# --- guest strings (I_Error's message) --------------------------------------------------------


def guest_string(h: Host, addr: int, limit: int = 512) -> str:
    out = bytearray()
    while addr and len(out) < limit:
        try:
            b = h.word(addr & ~3) >> 8 * (addr & 3) & 0xFF
        except ValueError:
            break
        if not b:
            break
        out.append(b)
        addr += 1
    return out.decode("latin-1")


def guest_format(h: Host, fmt: int, ap: int) -> str:
    """printf of a guest format string and va_list (word-sized arguments on the sim's stack), as
    the host interpreter formats I_Error's message (%d %i %u %x %X %o %c %s %%)."""
    import re

    def arg() -> int:
        nonlocal ap
        try:
            v = h.word(ap)
        except ValueError:
            v = 0
        ap += 4
        return v

    def conv(m: re.Match) -> str:
        flags, c = m.group(1), m.group(2)
        flags = re.sub("[lhz]", "", flags)
        if c == "%":
            return "%"
        if c == "s":
            return ("%" + flags + "s") % guest_string(h, arg())
        v = arg()
        if c in "di":
            v = v - (1 << 32) if v >> 31 else v
            c = "d"
        elif c == "u":
            c = "d"
        elif c == "c":
            v = chr(v & 0xFF)
        return ("%" + flags + c) % v

    return re.sub(r"%([-+ #0-9.lhz]*)([diuxXocs%])", conv, guest_string(h, fmt))


# --- button scripts -----------------------------------------------------------------------------


def parse_buttons(spec: str) -> Callable[[int], frozenset]:
    """'forward@10-120,fire@60-64+90-94,turnleft@130-150' -> tic -> the actions held. Entries
    are actions joined by '+', '@', then tic ranges (a-b inclusive, or a) joined by '+'. Actions
    are host.ACTION_KEYS' (left right forward back strafeleft straferight fire use run menu enter
    weapon1 weapon2 weapon3) or an alias (turnleft, turnright, backward, open, shoot, esc)."""
    ranges = []
    for part in filter(None, (s.strip() for s in spec.split(","))):
        names, sep, tics = part.partition("@")
        if not sep or not names or not tics:
            raise ValueError(f"bad button entry {part!r} (want action[+action]@a-b[+c-d])")
        acts = []
        for nm in names.split("+"):
            a = ALIASES.get(nm.strip().lower(), nm.strip().lower())
            if a not in ACTION_KEYS:
                raise ValueError(f"unknown action {nm!r} (known: {', '.join(ACTION_KEYS)})")
            acts.append(a)
        for tr in tics.split("+"):
            a, dash, b = tr.partition("-")
            try:
                lo = int(a)
                hi = int(b) if dash else lo  # (an open range, 'a-', is an error)
            except ValueError:
                raise ValueError(f"bad tic range {tr!r} (want a-b or a)") from None
            if hi < lo:
                raise ValueError(f"empty tic range {tr!r}")
            ranges.append((lo, hi, tuple(acts)))

    def held(tic: int) -> frozenset:
        return frozenset(x for lo, hi, acts in ranges if lo <= tic <= hi for x in acts)

    return held


# --- the runner ---------------------------------------------------------------------------------


class Runner:
    """A host computer running DOOM and the machinery for its sim mode (see the module doc).

    threads: the engine's thread count (None: DEFAULT_THREADS). engine: an open
    ``qwendoom.engine.Engine`` to use instead of opening one (it is not closed by ``close``).
    log: keep every model-mode token (``token_log()``; for checks and the bench), up to that
    many. forced: feed the injected tokens before a record's last one as forced steps."""

    def __init__(self, mode: str = "model", weights=DEFAULT_WEIGHTS, threads: int | None = None,
                 elf=DEFAULT_ELF, wad=DEFAULT_WAD, args=DEFAULT_ARGS, config_dir=None,
                 engine=None, log: int = 0, forced: bool = True, echo: bool = False):
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r} (known: {', '.join(MODES)})")
        self.mode = mode
        self.weights = Path(weights)
        self.engine = self.dec = self.chk = self.ref = None
        self._run = None
        self._log = None
        self._log_len = 0  # tokens logged, kept by close
        self._own_engine = False
        self.heads = None
        if mode in MODEL_MODES:
            meta = self.weights / "qwendoom.json"
            if not (self.weights / "model.safetensors").exists() or not meta.exists():
                raise FileNotFoundError(f"no checkpoint in {self.weights}: run "
                                        "`python -m qwendoom build --weights`")
            self.heads = json.loads(meta.read_text())["heads"]
            from . import snapshot  # noqa: F401  (imports torch: ~1 s, paid here, not at a resync)
        self.host = Host(elf, wad, args, config_dir=config_dir, echo=echo)
        h = self.host
        if mode != "native":
            L = lib()
            self.mirror = T.Memory.from_arrays(h.ram, h.drom, h.rom)  # the host's own memory
            self.dec = T.Decoder(self.mirror)
            if mode in MODEL_MODES:
                from . import engine as E
                if engine is None:
                    engine = E.Engine(self.weights, n_threads=threads or DEFAULT_THREADS)
                    self._own_engine = True
                self.engine = engine
                if mode == "check":
                    self.chk = T.Checker(T.Ref(T.Memory.from_arrays(h.ram, h.drom, h.rom,
                                                                    own_ram=True)))
                self._run = L.qd_run_new(engine.h, self.dec.p, self.chk.p if self.chk else None)
            else:
                self.ref = T.Ref(T.Memory.from_arrays(h.ram, h.drom, h.rom, own_ram=True))
                self._run = L.qd_run_new_ref(self.ref.p, self.dec.p)
            if not self._run:
                raise MemoryError("qd_run_new failed")
            L.qd_run_set_forced(self._run, int(forced))
            if log:
                self._log = (np.zeros(log, np.int32), np.zeros(log, np.uint8))
                L.qd_run_set_log(self._run, self._log[0].ctypes.data_as(C.POINTER(C.c_int32)),
                                 self._log[1].ctypes.data_as(C.POINTER(C.c_uint8)), log)
        self._st = TickStatsC()
        self._keep_rows = bool(log)
        self._tokens = 0  # at the end of the latest tick (live_tokens is the running count)
        self.ticks = self.frames = self.resyncs = 0  # ticks: clean ones (not a failed tick)
        self.injected = self.instructions = self.forced = 0  # (a failed tick's included)
        self.model_seconds = 0.0  # time inside sim ticks (pauses excluded)
        self.resync_seconds = 0.0
        self.last_tick: Tick | None = None
        self.ticked = False  # the latest frame came after a sim tick
        self.exited = False
        self.on_tick: Callable[[Tick], None] | None = None
        self.snapshot_rows = None  # the latest resync's state rows (model modes, with log)
        self._held: set[str] = set()

    # --- lifecycle ---

    def close(self) -> None:
        if self._run:
            self._tokens = lib().qd_run_live_tokens(self._run)
            self._log_len = lib().qd_run_log_len(self._run)
            lib().qd_run_free(self._run)
            self._run = None
        if self._own_engine and self.engine is not None:
            self.engine.close()
        self.engine = None
        if getattr(self, "host", None) is not None:
            self.host.close()
            self.host = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # --- counters (tokens and live_tokens may be read from any thread) ---

    @property
    def live_tokens(self) -> int:
        """Tokens so far, the running tick's included."""
        if self._run:
            return lib().qd_run_live_tokens(self._run)
        return self._tokens

    @property
    def tokens(self) -> int:
        return self.live_tokens

    @property
    def tps(self) -> float:
        """Tokens per second of sim-tick time (resyncs and the host's work excluded)."""
        return self.tokens / self.model_seconds if self.model_seconds > 0 else 0.0

    def counters(self) -> dict:
        out = {"mode": self.mode, "tokens": self.tokens, "injected": self.injected,
               "instructions": self.instructions, "ticks": self.ticks, "frames": self.frames,
               "resyncs": self.resyncs, "tick_seconds": round(self.model_seconds, 3),
               "resync_seconds": round(self.resync_seconds, 3), "tps": round(self.tps, 1)}
        if self.chk is not None:
            c = self.chk.counts
            out["compared"] = self.tokens - self.injected
            out["check_records"] = c["records"]
        return out

    def token_log(self) -> tuple[np.ndarray, np.ndarray]:
        """(tokens, host-injected flags) logged so far (model / check / ref modes with log)."""
        if self._log is None:
            raise RuntimeError("the runner was made without log=")
        n = lib().qd_run_log_len(self._run) if self._run else self._log_len
        return self._log[0][:n], self._log[1][:n].astype(bool)

    # --- control (any thread) ---

    def cancel(self) -> None:
        """Stop the running tick between two tokens (it raises Cancelled; the runner is done)."""
        if self._run:
            lib().qd_run_cancel(self._run)

    def pause(self, on: bool) -> None:
        """Hold the running tick between two tokens (on) or let it go on (off)."""
        if self._run:
            lib().qd_run_pause(self._run, int(on))

    # --- input (the game thread) ---

    def action(self, name: str, down: bool) -> None:
        """A UI action (ARCHITECTURE.md: forward, fire, use, menu, ...) going down / up."""
        self.host.action(ALIASES.get(name, name), down)

    def key(self, doom_key: int, down: bool) -> None:
        self.host.key(doom_key, down)

    def char(self, ch: str, down: bool) -> None:
        self.host.key(ord(ch.lower()), down)

    def hold(self, actions) -> None:
        """Hold exactly these actions (key edges for the difference with the last call)."""
        actions = set(actions)
        for a in sorted(self._held - actions):
            self.action(a, False)
        for a in sorted(actions - self._held):
            self.action(a, True)
        self._held = actions

    # --- running ---

    def step(self) -> str:
        """Run the host to its next event and answer it: "tick" (a sim tick ran: last_tick),
        "frame" (a frame is ready: frame()) or "exit"."""
        h = self.host
        while True:
            ev = h.run()
            if ev == SIM_TICK:
                self._sim_tick()
                return "tick"
            if ev == FRAME:
                self.frames += 1
                return "frame"
            if ev == EXIT:
                self.exited = True
                return "exit"
            if ev == ERROR:
                raise HostError(h.error)

    def step_frame(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Run the host to its next frame, doing any sim tick on the way: (320x200 palette
        indices, 256x3 RGB palette), or None when DOOM exited. ``ticked`` says whether a sim
        tick ran since the previous frame."""
        self.ticked = False
        while True:
            ev = self.step()
            if ev == "tick":
                self.ticked = True
            elif ev == "frame":
                return self.host.frame()
            else:
                return None

    def frame(self) -> tuple[np.ndarray, np.ndarray]:
        return self.host.frame()

    def resync(self) -> float:
        """Load the host's state into the sim (done by the next tick when the host asks)."""
        h = self.host
        t0 = time.perf_counter()
        pc, sp, gp = h.sim_entry()
        regs = [0] * 32
        regs[2], regs[3] = sp, gp
        if self.mode in MODEL_MODES:
            from . import engine as E
            from . import snapshot, spec
            rows = snapshot.state_rows(bytes(h.ram), bytes(h.drom), bytes(h.rom), regs, pc,
                                       heads=self.heads)
            E.start(self.engine, rows, spec.PAD)
            self.dec.set_context(pc=pc, regs=regs)
            if self.chk is not None:
                self.chk.ref.mem.sync_ram()
                self.chk.ref.set_context(pc=pc, regs=regs)
                self.chk.reset()
            self._start_stream(rows.first)
            lib().qd_run_set_pred(self._run, self.engine.step(rows.first))
            if self._keep_rows:
                self.snapshot_rows = rows
        elif self.mode == "ref":
            from . import spec
            self.ref.mem.sync_ram()
            self.ref.set_context(pc=pc, regs=regs)
            self.dec.set_context(pc=pc, regs=regs)
            self._start_stream(spec.tok("GO"))
            lib().qd_run_set_pred(self._run, 0)
        h.resync()
        dt = time.perf_counter() - t0
        self.resyncs += 1
        self.resync_seconds += dt
        return dt

    def _start_stream(self, go: int) -> None:
        """The run's first token (GO, which the engine steps) to the decoder and the checker as
        well: the first token of the tick is then stream token 1, where both reject a GO (a
        fresh decoder or checker takes one at token 0 as the start of a run). Not logged, not
        counted (the counters and the token log are the ticks' tokens)."""
        if self.dec.feed(go) < 0:
            raise T.DecodeError(f"resync: GO: {self.dec.error_text}")
        if self.chk is not None and self.chk.feed(go) != T.CHK_OK:
            raise T.CheckError(self.chk.divergence, "resync: the checker refused GO")

    def _sim_tick(self) -> Tick:
        h = self.host
        resync = h.need_resync()
        if resync:
            self.resync()
        words = h.take_pokes()
        if self.mode == "native":
            t0 = time.perf_counter()
            st = h.sim_tick_native()  # HostError on the sim's I_Error / a region check
            dt = time.perf_counter() - t0
            v = words.astype(np.int64)
            v = np.where(v >> 31, v - (1 << 32), v)
            injected = int(np.sum(np.where((v >= -(1 << 14)) & (v < 1 << 14), 1,
                                           np.where((v >= -(1 << 29)) & (v < 1 << 29), 2, 3))))
            tick = Tick(self.ticks, self.frames, st.tokens, injected, st.instructions,
                        len(words), dt, resync)
        else:
            res = lib().qd_run_tick(self._run, words.ctypes.data_as(C.POINTER(C.c_uint32)),
                                    len(words), C.byref(self._st))
            st = self._st
            tick = Tick(self.ticks, self.frames, st.tokens, st.injected, st.instructions,
                        st.words, st.seconds, resync, st.forced)
            if res != TICK_YIELD or st.unread:
                self._account(tick, done=False)
                if res != TICK_YIELD:
                    self._raise(res)
                raise T.DecodeError(f"tick {tick.index}: the sim left {st.unread} of "
                                    f"{len(words)} input words unread")
        h.complete_sim_tick()
        self._account(tick)
        if self.on_tick:
            self.on_tick(tick)
        return tick

    def _account(self, tick: Tick, done: bool = True) -> None:
        """Add a tick to the counters. done = False: a failed tick (its tokens, injected tokens,
        instructions and time count; ``ticks`` stays the number of clean ticks)."""
        self.ticks += done
        self._tokens += tick.tokens
        self.forced += tick.forced
        self.injected += tick.injected
        self.instructions += tick.instructions
        self.model_seconds += tick.seconds
        self.last_tick = tick

    def _raise(self, res: int) -> None:
        msg = lib().qd_run_error(self._run).decode(errors="replace")
        where = f"tick {self.ticks}"  # the failing tick (ticks: the clean ones before it)
        if res == TICK_CANCELLED:
            raise Cancelled(f"{where}: cancelled")
        if res == TICK_HALT:
            raise SimHalt(f"{where}: the sim exited: {self.error_message()}")
        if res == TICK_FAULT:
            cause, pc = self.dec.fault or (-1, 0)
            from . import spec
            name = spec.FAULTS[cause] if 0 <= cause < len(spec.FAULTS) else str(cause)
            raise SimFault(f"{where}: the sim faulted ({name}) at pc {pc:#x} "
                           f"({self.host.symbolize(pc)})")
        if res == -3:  # diverged
            raise T.CheckError(self.chk.divergence, f"{where}: {msg}")
        if res == -4:
            raise T.CheckError(None, f"{where}: {msg}")
        raise T.DecodeError(f"{where}: {TICK_ERRORS.get(res, res)}: {msg}")

    def error_message(self) -> str:
        """I_Error's message from sim RAM (qd_error_fmt / qd_error_args), or the exit code."""
        h = self.host
        fa, aa = h.symbol("qd_error_fmt"), h.symbol("qd_error_args")
        fmt = h.word(fa) if fa else 0
        if not fmt:
            return f"exit({self.dec.exit_code if self.dec else '?'})"
        return guest_format(h, fmt, h.word(aa) if aa else 0)

    # --- convenience ---

    def run_frames(self, n: int, buttons: Callable[[int], frozenset] | None = None,
                   on_frame: Callable[[int, np.ndarray, np.ndarray], None] | None = None) -> int:
        """Run n frames (tics of DOOM's clock: one per frame), holding buttons(tic) before each;
        on_frame(tic, idx, pal) sees every frame. Returns the frames run (fewer at an exit)."""
        for tic in range(n):
            if buttons is not None:
                self.hold(buttons(tic))
            fr = self.step_frame()
            if fr is None:
                return tic
            if on_frame:
                on_frame(tic, *fr)
        return n
