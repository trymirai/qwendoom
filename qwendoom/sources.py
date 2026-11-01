"""Game sources: what the web server streams to the browser.

A source runs the game (in its own thread, usually) and publishes what the page shows: frames
(a 768-byte RGB palette + 320x200 palette indices, DOOM's native screen format) and stats (the
CPU's token counters and the game state). The server only calls the control methods (`start`,
`stop`, `pause`, `resume`, `key`, `close`) and listens for what the source publishes.

The real source, `ModelSource` (`--source model`, the default: DOOM's playsim on the CPU in the
weights), runs `qwendoom.runner.Runner` in a background thread and calls `publish_frame` /
`publish_stats` from there; `MockSource` does the same with an animated test pattern, so the page
and the protocol can be exercised without the model, and `HostSource` (`--source native`) runs
real DOOM with its sim on the host interpreter.
"""

from __future__ import annotations

import colorsys
import importlib
import math
import random
import threading
import time
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

WIDTH, HEIGHT = 320, 200
PALETTE_BYTES = 256 * 3
PIXEL_BYTES = WIDTH * HEIGHT

# Game states, as reported in stats["state"] (ARCHITECTURE.md, "The browser UI and server").
IDLE, STARTING, RUNNING, PAUSED, ENDED, ERROR = "idle", "starting", "running", "paused", "ended", "error"
STATES = (IDLE, STARTING, RUNNING, PAUSED, ENDED, ERROR)

# Key actions the page sends ({"type": "key", "action": A, "down": bool}).
ACTIONS = (
    "left", "right", "forward", "back", "strafeleft", "straferight",
    "fire", "use", "run", "menu", "enter", "weapon1", "weapon2", "weapon3",
)


@dataclass(frozen=True)
class Frame:
    """One screen: `palette` is 256 RGB triples, `pixels` 320x200 indices, row-major."""

    palette: bytes
    pixels: bytes
    seq: int

    def message(self) -> bytes:
        """The WebSocket binary frame: 0x01 + palette + pixels."""
        return b"\x01" + self.palette + self.pixels


@dataclass(frozen=True)
class Stats:
    """Counters for the page's LCD. None means unknown (the page shows a dash)."""

    tokens: int | None = 0
    tps: float | None = None
    frames: int | None = 0
    state: str = IDLE

    def message(self) -> dict:
        return {"type": "stats", "tokens": self.tokens, "tps": self.tps,
                "frames": self.frames, "state": self.state}


Listener = Callable[[str, object], None]  # (kind, payload): ("frame", Frame) | ("stats", Stats) | ("status", dict)

_KEEP = object()


def take_edges(queue: list[tuple[str | None, bool, str | None]]) -> list:
    """Pop the key edges to apply before the next frame: all of them, up to the release of a key
    that was pressed in the same batch. That release and everything after it wait for the next
    frame, so a tap shorter than a frame still holds its key for one tic (DOOM reads the keys once
    a tic, after all events, and a tic of the model can take a second or more)."""
    pressed = set()
    n = len(queue)
    for i, (action, down, char) in enumerate(queue):
        key = action if action is not None else ("char", char)
        if down:
            pressed.add(key)
        elif key in pressed:
            n = i
            break
    taken = queue[:n]
    del queue[:n]
    return taken


class GameSource:
    """Base class for game sources.

    Control methods are called on the server's event loop thread and must return quickly (hand
    work to the game thread). The publish_* methods may be called from any thread; listeners run
    on the publishing thread and must be cheap and thread-safe (the server's listener only hops
    onto its event loop with `call_soon_threadsafe`). Publish stats a few dozen times a second at
    most; the server sends them on at 10 Hz anyway (state changes at once).

    A game thread must not publish after a `stop` or `pause` it has not seen yet (a stale frame
    or a `running` stats would undo it on the page): check the state under the lock that the
    control methods take, and publish while holding it, as `MockSource` does.
    """

    name = "source"

    def __init__(self) -> None:
        self._pub_lock = threading.Lock()
        self._listeners: list[Listener] = []
        self._frame: Frame | None = None
        self._stats = Stats()
        self._status: dict = {}
        self._seq = 0

    # -- control (event loop thread) ----------------------------------------------------------
    def start(self) -> None:
        """Start (or restart after `ended`/`error`) the game. Idempotent while running."""
        raise NotImplementedError

    def stop(self) -> None:
        """End the game and go back to idle (a dark screen). A later `start` begins anew."""
        raise NotImplementedError

    def pause(self) -> None:
        raise NotImplementedError

    def resume(self) -> None:
        raise NotImplementedError

    def key(self, action: str | None, down: bool, char: str | None = None) -> None:
        """A key edge: `action` is one of ACTIONS, or `char` a single lowercase character."""
        raise NotImplementedError

    def close(self) -> None:
        """Stop the game thread and release resources. Called once, at server shutdown."""

    # -- output (any thread) ------------------------------------------------------------------
    def publish_frame(self, palette, pixels) -> Frame:
        """Publish a screen. Accepts bytes-like objects or numpy arrays (copied)."""
        palette, pixels = bytes(palette), bytes(pixels)
        if len(palette) != PALETTE_BYTES or len(pixels) != PIXEL_BYTES:
            raise ValueError(f"frame must be {PALETTE_BYTES} + {PIXEL_BYTES} bytes, "
                             f"got {len(palette)} + {len(pixels)}")
        with self._pub_lock:
            self._seq += 1
            frame = self._frame = Frame(palette, pixels, self._seq)
            listeners = list(self._listeners)
        for fn in listeners:
            fn("frame", frame)
        return frame

    def publish_stats(self, *, tokens=_KEEP, tps=_KEEP, frames=_KEEP, state=_KEEP) -> Stats:
        """Update the counters; fields not given keep their previous values."""
        if state is not _KEEP and state not in STATES:
            raise ValueError(f"unknown state {state!r}")
        changes = {k: v for k, v in dict(tokens=tokens, tps=tps, frames=frames, state=state).items()
                   if v is not _KEEP}
        with self._pub_lock:
            stats = self._stats = replace(self._stats, **changes)
            listeners = list(self._listeners)
        for fn in listeners:
            fn("stats", stats)
        return stats

    def publish_status(self, message: str | None = None, **fields) -> dict:
        """Publish free-form status (e.g. "loading E1M1", an error) shown by the page."""
        status = {"message": message, **fields}
        with self._pub_lock:
            self._status = status
            listeners = list(self._listeners)
        for fn in listeners:
            fn("status", status)
        return status

    # -- reading ------------------------------------------------------------------------------
    @property
    def frame(self) -> Frame | None:
        return self._frame

    @property
    def stats(self) -> Stats:
        return self._stats

    @property
    def status(self) -> dict:
        return dict(self._status)

    @property
    def state(self) -> str:
        return self._stats.state

    def add_listener(self, fn: Listener) -> None:
        with self._pub_lock:
            self._listeners.append(fn)

    def remove_listener(self, fn: Listener) -> None:
        with self._pub_lock:
            if fn in self._listeners:
                self._listeners.remove(fn)


# ------------------------------------------------------------------------------------------------
# The mock: an animated test pattern driven by the same keys as DOOM.

def _mock_palette() -> np.ndarray:
    """16 ramps of 16 shades (bright to dark), loosely like DOOM's PLAYPAL layout."""
    ramps = [  # (hue degrees, saturation)
        (0, 0.0), (30, 0.55), (110, 0.6), (0, 0.75), (215, 0.6), (52, 0.8), (28, 0.85), (170, 0.5),
        (280, 0.5), (38, 0.35), (75, 0.5), (330, 0.45), (200, 0.45), (140, 0.45), (210, 0.15), (20, 0.4),
    ]
    pal = np.zeros((256, 3), np.uint8)
    for r, (h, s) in enumerate(ramps):
        for shade in range(16):
            v = 1.0 - shade * 0.058
            pal[r * 16 + shade] = [round(c * 255) for c in colorsys.hsv_to_rgb(h / 360, s, v)]
    return pal


GRAY, BROWN, GREEN, RED, BLUE, YELLOW, ORANGE, TEAL, PURPLE, TAN, OLIVE, PINK, SKY, DKGREEN, STEEL, FLESH = (
    16 * i for i in range(16))

# A 3x5 pixel font for the overlay text (rows top to bottom, 3 bits each).
_FONT = {
    "0": "111101101101111", "1": "010110010010111", "2": "111001111100111", "3": "111001111001111",
    "4": "101101111001001", "5": "111100111001111", "6": "111100111101111", "7": "111001001010010",
    "8": "111101111101111", "9": "111101111001111", "A": "010101111101101", "B": "110101110101110",
    "C": "011100100100011", "D": "110101101101110", "E": "111100110100111", "F": "111100110100100",
    "G": "011100101101011", "H": "101101111101101", "I": "111010010010111", "J": "001001001101010",
    "K": "101101110101101", "L": "100100100100111", "M": "101111111101101", "N": "110101101101101",
    "O": "010101101101010", "P": "110101110100100", "Q": "010101101110011", "R": "110101110101101",
    "S": "011100010001110", "T": "111010010010010", "U": "101101101101111", "V": "101101101101010",
    "W": "101101111111101", "X": "101101010101101", "Y": "101101010010010", "Z": "111001010100111",
    " ": "000000000000000", ":": "000010000010000", "/": "001001010100100", "-": "000000111000000",
    ".": "000000000000010",
}
_GLYPHS = {c: np.array([int(b) for b in bits], bool).reshape(5, 3) for c, bits in _FONT.items()}

# Status-bar key indicators: (action, label) in the order of the page's key tiles.
_HUD_KEYS = (("left", "TL"), ("forward", "FW"), ("right", "TR"), ("strafeleft", "SL"),
             ("back", "BK"), ("straferight", "SR"), ("fire", "FI"), ("use", "US"), ("run", "RN"),
             ("menu", "ES"), ("enter", "EN"), ("weapon1", "1"), ("weapon2", "2"), ("weapon3", "3"))


def _text(img: np.ndarray, x: int, y: int, s: str, color: int, scale: int = 1) -> None:
    for i, ch in enumerate(s.upper()):
        g = _GLYPHS.get(ch)
        if g is None:
            continue
        if scale > 1:
            g = g.repeat(scale, 0).repeat(scale, 1)
        x0 = x + i * 4 * scale
        h, w = g.shape
        if x0 < 0 or y < 0 or x0 + w > WIDTH or y + h > HEIGHT:
            continue
        img[y:y + h, x0:x0 + w][g] = color


class MockSource(GameSource):
    """An animated test pattern with fake token counters, driven by the page's keys.

    It walks a camera over a checkered floor (turn, move, strafe, run), flashes the palette on fire
    and use, toggles a menu box on Esc (Enter closes it), switches the "weapon" on 1-3, and lights
    a status-bar indicator for every held action, so every control on the page is visible on
    screen. Frames run at `fps`; the token counter advances at about `tps` tokens per second.
    """

    name = "mock"
    VIEW_H = 168  # like DOOM: the bottom 32 rows are the status bar
    HORIZON = 84
    FOCAL = 160.0

    def __init__(self, fps: float = 35.0, tps: float = 24000.0, startup: float = 0.6, seed: int = 0):
        super().__init__()
        self.fps, self.tps, self.startup = float(fps), float(tps), float(startup)
        self._rng = random.Random(seed)
        self._lock = threading.Lock()
        self._held: set[str] = set()
        self._events: list[str] = []
        self._chars: list[str] = []
        self._state = IDLE
        self._ready_at = 0.0
        self._stop = threading.Event()
        self._base_pal = _mock_palette()
        self._reset_world()
        self._precompute()
        self._thread = threading.Thread(target=self._run, name="mock-source", daemon=True)
        self._thread.start()

    # -- control ------------------------------------------------------------------------------
    def start(self) -> None:
        with self._lock:
            if self._state in (STARTING, RUNNING, PAUSED):
                return
            self._reset_world()
            self._state = STARTING
            self._ready_at = time.monotonic() + self.startup
        self.publish_stats(tokens=0, tps=None, frames=0, state=STARTING)
        self.publish_status("loading the test pattern")

    def stop(self) -> None:
        with self._lock:
            if self._state not in (STARTING, RUNNING, PAUSED):
                return
            self._state = IDLE
            self._held.clear()
            self._events.clear()
            self._chars.clear()
        with self._pub_lock:
            self._frame = None  # late joiners see a dark screen, not the last picture
        self.publish_status(None)
        self.publish_stats(tokens=0, tps=None, frames=0, state=IDLE)

    def pause(self) -> None:
        with self._lock:
            if self._state != RUNNING:
                return
            self._state = PAUSED
        self.publish_stats(tps=0.0, state=PAUSED)

    def resume(self) -> None:
        with self._lock:
            if self._state != PAUSED:
                return
            self._state = RUNNING
        self.publish_stats(state=RUNNING)

    def key(self, action: str | None, down: bool, char: str | None = None) -> None:
        with self._lock:
            if action is not None:
                if down:
                    if action not in self._held:
                        self._events.append(action)
                    self._held.add(action)
                else:
                    self._held.discard(action)
            elif char is not None and down:
                self._chars.append(char)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    # -- the game thread ------------------------------------------------------------------------
    def _reset_world(self) -> None:
        self._x, self._y, self._angle = 0.0, 0.0, math.pi / 2
        self._tic = 0
        self._tokens = 0.0
        self._frames = 0
        self._weapon = 1
        self._menu = False
        self._flash = 0
        self._use = 0
        self._last_char = ""
        self._tps_now = 0.0

    def _precompute(self) -> None:
        cols = np.arange(WIDTH) - WIDTH / 2 + 0.5
        self._col_off = np.arctan(cols / self.FOCAL)  # ray angle offset per column
        self._col_cos = np.cos(self._col_off)
        rows = np.arange(self.HORIZON, self.VIEW_H) - self.HORIZON + 0.5
        self._row_dist = 41.0 * self.FOCAL / rows  # camera height 41, like DOOM's view height
        self._shade = np.clip(self._row_dist[:, None] / 70.0, 0, 15).astype(np.int16)
        sky = np.arange(self.HORIZON)
        self._sky = (SKY + np.clip(15 - sky * 15 // self.HORIZON, 0, 15)).astype(np.uint8)
        self._palstrip = (np.arange(WIDTH) * 256 // WIDTH).astype(np.uint8)

    def _run(self) -> None:
        period = 1.0 / self.fps
        next_tick = time.monotonic()
        last_stats = 0.0
        last_t = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            dt, last_t = now - last_t, now
            with self._lock:
                ready = self._state == STARTING and now >= self._ready_at
                if ready:
                    self._state = RUNNING
                state = self._state
                held = frozenset(self._held)
                events, self._events = self._events, []
                chars, self._chars = self._chars, []
            if ready:
                self.publish_status(None)
            if state == RUNNING:
                self._tick(held, events, chars)
                pal, pix = self._render(held)
                tps = self.tps * (1 + 0.04 * math.sin(self._tic / 17) + self._rng.uniform(-0.02, 0.02))
                with self._lock:  # a Stop or Pause that came in meanwhile wins: nothing stale goes out
                    if self._state == RUNNING:
                        self._frames += 1
                        self._tokens += tps * dt
                        self._tps_now = tps
                        self.publish_frame(pal, pix)
                        if now - last_stats >= 0.1:
                            last_stats = now
                            self.publish_stats(tokens=int(self._tokens), tps=round(tps, 1),
                                               frames=self._frames, state=RUNNING)
            elif now - last_stats >= 0.1:
                last_stats = now
                if state != self.stats.state:
                    self.publish_stats(state=state)
            next_tick += period
            delay = next_tick - time.monotonic()
            if delay < -0.25:  # fell behind (e.g. machine asleep): don't try to catch up
                next_tick = time.monotonic()
                delay = 0
            self._stop.wait(max(0.0, delay))

    def _tick(self, held: frozenset, events: list, chars: list) -> None:
        self._tic += 1
        for ev in events:
            if ev == "menu":
                self._menu = not self._menu
            elif ev == "enter" and self._menu:
                self._menu = False
            elif ev.startswith("weapon"):
                self._weapon = int(ev[-1])
            elif ev == "use":
                self._use = 8
            elif ev == "fire":
                self._flash = 6
        if chars:
            self._last_char = chars[-1]
        if self._menu:
            return
        if "fire" in held and self._flash == 0 and self._tic % 8 == 0:
            self._flash = 6  # auto-fire while held, like the pistol
        run = 2.0 if "run" in held else 1.0
        turn = 0.045 * run
        if "left" in held:
            self._angle += turn
        if "right" in held:
            self._angle -= turn
        fwd = (("forward" in held) - ("back" in held)) * 8.0 * run
        side = (("straferight" in held) - ("strafeleft" in held)) * 6.0 * run
        self._x += fwd * math.cos(self._angle) + side * math.cos(self._angle - math.pi / 2)
        self._y += fwd * math.sin(self._angle) + side * math.sin(self._angle - math.pi / 2)
        self._flash = max(0, self._flash - 1)
        self._use = max(0, self._use - 1)

    def _render(self, held: frozenset) -> tuple[np.ndarray, np.ndarray]:
        img = np.empty((HEIGHT, WIDTH), np.uint8)
        a = self._angle - self._col_off  # screen right = clockwise
        # sky with a horizon silhouette that scrolls as the camera turns
        img[:self.HORIZON] = self._sky[:, None]
        ridge = (self.HORIZON - 12 - 7 * np.sin(3 * a) - 4 * np.sin(7 * a + 1)).astype(int)
        rows = np.arange(self.HORIZON)[:, None]
        img[:self.HORIZON][rows >= ridge[None, :]] = DKGREEN + 11
        img[0:6] = self._palstrip[None, :]  # every palette entry, left to right
        # checkered floor (mode-7 style)
        d = self._row_dist[:, None] / self._col_cos[None, :]
        wx = self._x + d * np.cos(a)[None, :]
        wy = self._y + d * np.sin(a)[None, :]
        checker = ((np.floor(wx / 64).astype(np.int64) + np.floor(wy / 64).astype(np.int64)) & 1)
        base = np.where(checker == 1, TAN, BROWN).astype(np.int16)
        img[self.HORIZON:self.VIEW_H] = (base + np.minimum(self._shade + 2, 15)).astype(np.uint8)
        # weapon and crosshair
        wcol = (STEEL + 4, ORANGE + 3, TEAL + 3)[self._weapon - 1]
        for i, y in enumerate(range(self.VIEW_H - 30, self.VIEW_H)):
            half = 10 + i // 3
            img[y, 160 - half:160 + half] = wcol + (2 if i % 6 == 0 else 0)
        if self._flash:
            img[self.VIEW_H - 46:self.VIEW_H - 30, 150:170] = YELLOW + 1
        img[self.HORIZON, 156:165] = GRAY
        img[self.HORIZON - 4:self.HORIZON + 5, 160] = GRAY
        _text(img, 4, 10, "QWENDOOM MOCK SOURCE", GRAY, 1)
        _text(img, 4, 18, f"TIC {self._tic}", GRAY + 2, 1)
        if self._last_char:
            _text(img, 4, 26, f"CHAR {self._last_char}", GRAY + 2, 1)
        if self._menu:
            img[60:110, 90:230] = GRAY + 14
            img[60, 90:230] = img[109, 90:230] = img[60:110, 90] = img[60:110, 229] = RED + 2
            _text(img, 136, 70, "MENU", RED + 1, 3)
            _text(img, 104, 96, "ENTER OR ESC TO CLOSE", GRAY + 3, 1)
        # status bar: one indicator per action, lit while held
        img[self.VIEW_H:] = STEEL + 9
        img[self.VIEW_H] = STEEL + 4
        for i, (action, label) in enumerate(_HUD_KEYS):
            x0, y0 = 6 + i * 22, self.VIEW_H + 8
            lit = action in held
            img[y0:y0 + 16, x0:x0 + 19] = (GREEN + 3) if lit else (STEEL + 12)
            _text(img, x0 + 2 + (2 if len(label) == 1 else 0), y0 + 5, label,
                  GRAY + (15 if lit else 4), 1)
        pal = self._base_pal
        if self._flash or self._use:  # palette effects, like DOOM's pickup/damage tints
            tint = np.array([40, 30, 0] if self._flash else [0, 36, 8], np.int16)
            pal = np.clip(pal.astype(np.int16) + tint, 0, 255).astype(np.uint8)
        return pal, img


# ------------------------------------------------------------------------------------------------
# DOOM on the host computer, with the sim in `native` mode (no model): real frames for the page.

class HostSource(GameSource):
    """DOOM on `qwendoom.host.Host`, its sim ticks answered by the host interpreter (`native`).

    A development source: the playsim runs on the fast interpreter, not on the model, so the page
    can be tried against real DOOM before the engine is wired in. The token counter is the native
    sim's count of the tokens the model would emit (the same count in every sim mode); tokens per
    second is left unknown, since nothing here runs at the model's speed. The game's clock is one
    tic per frame (doom/qd_platform.c), so frames are paced to `fps` (35, DOOM's tic rate).

    Each Start runs a fresh Host in its own thread (a "generation"); Stop or close only flags it,
    and a thread whose generation is over publishes nothing more.
    """

    name = "native"

    def __init__(self, fps: float = 35.0, **host_options):
        super().__init__()
        self.fps = float(fps)
        self._host_options = host_options  # passed to Host(): elf, wad, args, ...
        self._lock = threading.Lock()
        self._edges: list[tuple[str | None, bool, str | None]] = []
        self._state = IDLE
        self._gen = 0
        self._done = threading.Event()  # the current generation's stop flag
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        with self._lock:
            if self._state in (STARTING, RUNNING, PAUSED):
                return
            self._state = STARTING
            self._edges.clear()
            self._gen += 1
            self._done = threading.Event()
            self._thread = threading.Thread(target=self._run, args=(self._gen, self._done),
                                            name="doom-host", daemon=True)
        self.publish_stats(tokens=0, tps=None, frames=0, state=STARTING)
        self.publish_status("starting DOOM (native sim)")
        self._thread.start()

    def stop(self) -> None:
        with self._lock:
            if self._state not in (STARTING, RUNNING, PAUSED):
                return
            self._state = IDLE
            self._gen += 1
            self._done.set()
        with self._pub_lock:
            self._frame = None
        self.publish_status(None)
        self.publish_stats(tokens=0, tps=None, frames=0, state=IDLE)

    def pause(self) -> None:
        self._switch(RUNNING, PAUSED)

    def resume(self) -> None:
        self._switch(PAUSED, RUNNING)

    def _switch(self, old: str, new: str) -> None:
        with self._lock:
            if self._state != old:
                return
            self._state = new
        self.publish_stats(state=new)

    def key(self, action: str | None, down: bool, char: str | None = None) -> None:
        with self._lock:  # applied by the game thread: Host is not thread-safe
            if self._state in (STARTING, RUNNING, PAUSED):
                self._edges.append((action, down, char))

    def close(self) -> None:
        with self._lock:
            self._gen += 1
            self._done.set()
            thread = self._thread
        if thread is not None:
            thread.join(timeout=5)

    def _run(self, gen: int, done: threading.Event) -> None:
        from .host import EXIT, Host  # needs build/libqdhost.dylib; only this source does

        try:
            host = Host(**self._host_options)
        except Exception as e:
            self._end(gen, ERROR, f"cannot start DOOM: {e}")
            return
        tokens = frames = 0

        def on_tick(st, pokes) -> None:
            nonlocal tokens
            tokens += st.tokens

        try:
            with self._lock:
                if self._gen == gen and self._state == STARTING:
                    self._state = RUNNING
                    self.publish_status(None)
            period, next_t, last_stats = 1.0 / self.fps, time.monotonic(), 0.0
            while not done.is_set():
                with self._lock:
                    state, edges = self._state, []
                    if state != PAUSED:
                        edges = take_edges(self._edges)
                if state == PAUSED:
                    done.wait(0.05)
                    next_t = time.monotonic()
                    continue
                for action, down, char in edges:
                    if action is not None:
                        host.action(action, down)
                    else:
                        host.key(ord(char), down)
                if host.next_frame(on_tick) == EXIT:
                    self._end(gen, ENDED, f"DOOM exited ({host.exit_code})")
                    return
                idx, pal = host.frame()
                frames += 1
                now = time.monotonic()
                with self._lock:  # a Stop or Pause that came in meanwhile wins: nothing stale goes out
                    if self._gen != gen or self._state != RUNNING:
                        continue
                    self.publish_frame(pal, idx)
                    if now - last_stats >= 0.1:
                        last_stats = now
                        self.publish_stats(tokens=tokens, tps=None, frames=frames, state=RUNNING)
                next_t += period
                delay = next_t - time.monotonic()
                if delay < -0.25:
                    next_t, delay = time.monotonic(), 0.0
                done.wait(max(0.0, delay))
        except Exception as e:
            self._end(gen, ERROR, f"{type(e).__name__}: {e}")
        finally:
            host.close()

    def _end(self, gen: int, state: str, message: str) -> None:
        with self._lock:
            if self._gen != gen:
                return  # stopped meanwhile: the page already went back to idle
            self._state = state
            self.publish_stats(state=state)
            self.publish_status(message)


# ------------------------------------------------------------------------------------------------
# DOOM with its playsim on the CPU in the weights: the real thing.

class ModelSource(GameSource):
    """DOOM on `qwendoom.runner.Runner`: the host computer draws, the model runs `P_Ticker`.

    `mode` is the runner's: "model" (the transformer, default), "check" (the model with the C
    reference in lockstep; a divergence ends the game with an error), "ref" or "native".
    `threads` is the engine's thread count (None: runner.DEFAULT_THREADS, 1). Other keyword
    arguments go to the Runner (weights, elf, wad, args, config_dir).

    Frames are not paced by a clock: a tic takes the model as long as it takes. Frames without a
    sim tick (menus, the screen wipe at a level start) come at host speed and are paced to `fps`
    (DOOM's 35), as is everything in the fast modes. Stats go out at ~10 Hz from their own thread,
    also while a tick runs (the runner's token counter is live): tokens = the model's tokens so
    far, tps = tokens per second over the last ~2 s (None when no token came in that window:
    menus, paused), frames = frames shown. The state stays `starting` until the first frame
    (the resync and the first tick: the counters move meanwhile). Pause holds the model between
    two tokens; Stop cancels the running tick.

    Each Start runs a fresh Runner in its own thread (a "generation"), as `HostSource` does.
    """

    name = "model"

    def __init__(self, mode: str = "model", threads: int | None = None, fps: float = 35.0,
                 **runner_options):
        super().__init__()
        self.mode, self.threads, self.fps = mode, threads, float(fps)
        self._runner_options = runner_options
        self._lock = threading.Lock()
        self._edges: list[tuple[str | None, bool, str | None]] = []
        self._state = IDLE
        self._gen = 0
        self._done = threading.Event()
        self._thread: threading.Thread | None = None
        self._runner = None  # the current generation's Runner (cancel / pause from here)
        self._frames = 0
        # the snapshot code imports torch (~1 s): do it now, not on the first Start
        threading.Thread(target=importlib.import_module, args=("qwendoom.snapshot",),
                         name="model-preload", daemon=True).start()

    def start(self) -> None:
        with self._lock:
            if self._state in (STARTING, RUNNING, PAUSED):
                return
            self._state = STARTING
            self._edges.clear()
            self._gen += 1
            self._done = threading.Event()
            self._runner = None
            self._frames = 0
            self._thread = threading.Thread(target=self._run, args=(self._gen, self._done),
                                            name="doom-model", daemon=True)
        self.publish_stats(tokens=0, tps=None, frames=0, state=STARTING)
        self.publish_status(f"starting DOOM ({self.mode} mode)")
        self._thread.start()

    def stop(self) -> None:
        with self._lock:
            if self._state not in (STARTING, RUNNING, PAUSED):
                return
            self._state = IDLE
            self._end_generation()
        with self._pub_lock:
            self._frame = None
        self.publish_status(None)
        self.publish_stats(tokens=0, tps=None, frames=0, state=IDLE)

    def _end_generation(self) -> None:
        """(holding the lock) The current game thread must finish: flag it, cancel its tick."""
        self._gen += 1
        self._done.set()
        if self._runner is not None:
            self._runner.cancel()

    def pause(self) -> None:
        with self._lock:
            if self._state != RUNNING:
                return
            self._state = PAUSED
            if self._runner is not None:
                self._runner.pause(True)  # holds the model between two tokens
        self.publish_stats(tps=None, state=PAUSED)

    def resume(self) -> None:
        with self._lock:
            if self._state != PAUSED:
                return
            self._state = RUNNING
            if self._runner is not None:
                self._runner.pause(False)
        self.publish_stats(state=RUNNING)

    def key(self, action: str | None, down: bool, char: str | None = None) -> None:
        with self._lock:  # applied by the game thread between frames: Host is not thread-safe
            if self._state in (STARTING, RUNNING, PAUSED):
                self._edges.append((action, down, char))

    def close(self) -> None:
        with self._lock:
            self._end_generation()
            thread = self._thread
        if thread is not None:
            thread.join(timeout=10)

    # -- the game thread ------------------------------------------------------------------------
    def _run(self, gen: int, done: threading.Event) -> None:
        from .runner import Cancelled, Runner, set_qos  # needs the native libraries

        set_qos()  # the game thread runs the model's hot loop: keep it on a performance core
        try:
            runner = Runner(self.mode, threads=self.threads, **self._runner_options)
        except Exception as e:
            self._end(gen, ERROR, f"cannot start DOOM: {type(e).__name__}: {e}")
            return
        with self._lock:
            live = self._gen == gen
            if live:
                self._runner = runner
                self.publish_status("loading the level into the model, then its first tic")
        if not live:
            runner.close()
            return
        fin = threading.Event()  # the stats thread's end (it reads the runner: stop it first)
        stats = threading.Thread(target=self._stats_loop, args=(gen, fin, runner),
                                 name="doom-model-stats", daemon=True)
        stats.start()
        try:
            period, last_t = 1.0 / self.fps, 0.0
            while not done.is_set():
                with self._lock:
                    state, edges = self._state, []
                    if state != PAUSED:
                        edges = take_edges(self._edges)
                if state == PAUSED:
                    done.wait(0.05)
                    continue
                for action, down, char in edges:
                    if action is not None:
                        runner.action(action, down)
                    else:
                        runner.char(char, down)
                fr = runner.step_frame()
                if fr is None:
                    self._end(gen, ENDED, f"DOOM exited ({runner.host.exit_code})")
                    return
                idx, pal = fr
                delay = last_t + period - time.monotonic()  # at most fps (menus, the wipe)
                if delay > 0:
                    done.wait(delay)
                while not done.is_set():  # a frame finished while paused waits for Resume
                    with self._lock:
                        if self._gen != gen:
                            return
                        if self._state == STARTING:
                            self._state = RUNNING
                            self.publish_stats(state=RUNNING)
                            self.publish_status(None)
                        if self._state == RUNNING:
                            self._frames += 1
                            self.publish_frame(pal, idx)
                            break
                    done.wait(0.05)
                last_t = time.monotonic()
        except Cancelled:
            pass  # stopped
        except Exception as e:
            self._end(gen, ERROR, f"{type(e).__name__}: {e}")
        finally:
            fin.set()
            stats.join()
            with self._lock:
                if self._runner is runner:
                    self._runner = None
            runner.close()

    def _stats_loop(self, gen: int, fin: threading.Event, runner) -> None:
        samples: list[tuple[float, int]] = []  # (time, tokens) over the last ~2 s
        while not fin.wait(0.1):
            now, tokens = time.monotonic(), runner.live_tokens
            with self._lock:  # a Stop or Pause that came in meanwhile wins: nothing stale goes out
                if self._gen != gen:
                    return
                if self._state not in (STARTING, RUNNING):
                    samples.clear()
                    continue
                samples.append((now, tokens))
                while len(samples) > 2 and now - samples[1][0] >= 2.0:
                    samples.pop(0)
                t0, k0 = samples[0]
                tps = (tokens - k0) / (now - t0) if tokens > k0 and now > t0 else None
                self.publish_stats(tokens=tokens, tps=None if tps is None else round(tps, 1),
                                   frames=self._frames, state=self._state)

    def _end(self, gen: int, state: str, message: str) -> None:
        with self._lock:
            if self._gen != gen:
                return  # stopped meanwhile: the page already went back to idle
            self._state = state
            self.publish_stats(tps=None, state=state)
            self.publish_status(message)


_REGISTRY: dict[str, Callable[..., GameSource]] = {"mock": MockSource, "native": HostSource,
                                                   "model": ModelSource}


def register_source(name: str, factory: Callable[..., GameSource]) -> None:
    """Make `--source NAME` available (the real DOOM source registers itself here)."""
    _REGISTRY[name] = factory


def make_source(spec: str, **options) -> GameSource:
    """Build a source from a registered name or a `package.module:Factory` path."""
    if spec in _REGISTRY:
        return _REGISTRY[spec](**options)
    if ":" in spec:
        module, _, attr = spec.partition(":")
        return getattr(importlib.import_module(module), attr)(**options)
    raise ValueError(f"unknown source {spec!r} (known: {', '.join(sorted(_REGISTRY))}, "
                     f"or package.module:Factory)")
