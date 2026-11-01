"""Measure DOOM on the qwendoom host computer in native sim mode: the numbers the project is sized
by. Sim instructions, trace tokens and pokes per tic (overall and while monsters are awake), host
instructions per frame, interpreter speed, RAM / DROM / ROM use, sim functions.

The player is a small deterministic bot (``Bot``): it follows waypoints, opens doors, turns to
and shoots awake monsters, and respawns when killed. It reads only sim RAM (the player and the
thinker list), which every sim mode keeps identical, so a run is a pure function of the build.

    python doom/measure.py [--tics 2100] [--frames 1,350,...] [--json build/doom-measure.json]
    python doom/measure.py --ref                     # also check every tick against the trace
                                                     # track's reference machine
    python doom/measure.py --maps all [--tics 1050]   # every map of the IWAD: does it fit?
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qwendoom import host as H  # noqa: E402

FRAMES_DIR = ROOT / "build" / "doom-frames"

# struct offsets on RV32 (ILP32), from d_think.h, p_mobj.h, r_defs.h, d_player.h, z_zone.c
TH_NEXT, TH_FUNC = 4, 8
MO_X, MO_Y, MO_Z, MO_ANGLE = 12, 16, 20, 32
MO_FLAGS, MO_HEALTH, MO_TARGET = 104, 108, 120
LN_V1, LN_V2, LN_FRONT, LN_BACK, LINE_SIZE = 0, 4, 48, 52, 64
SEC_FLOOR, SEC_CEIL, SECTOR_SIZE = 0, 4, 88
PL_MO, PL_STATE, PL_HEALTH, PL_KILLS = 0, 4, 40, 212
PST_DEAD = 1
MF_SHOOTABLE, MF_COUNTKILL = 0x4, 0x400000
ZB_SIZE, ZB_TAG, ZB_NEXT = 0, 8, 16  # memblock_t
PU_FREE = 4
TAG_NAMES = {1: "static", 5: "level", 6: "levspec", 7: "purgelevel", 8: "cache"}

# Freedoom E1M1 from the start (-416, 256) facing east: through the hall south of the pit and
# its lift, down the corridor to the door at (960, -152) (opened with "use"), round the big room
# beyond (zombieman, shotgun guys), and back.
E1M1_ROUTE = [(-150, 256), (200, 60), (960, 60), (960, -100, "use"), (960, -250), (900, -450),
              (1050, -650), (900, -450), (960, -250), (960, 60), (200, 60)]


class SimView:
    """DOOM's sim state, read from RAM (the player, the mobjs, the level zone)."""

    def __init__(self, h: H.Host):
        self.h = h
        self.u32 = h.ram.view(np.uint32)
        self.s32 = h.ram.view(np.int32)
        self.players = h.symbol("players")
        self.thinkercap = h.symbol("thinkercap")
        self.mobj_thinker = h.symbol("P_MobjThinker")
        self.zone = h.word(h.symbol("levelzone"))  # set by Z_Init, before any level

    def u(self, a: int) -> int:
        return int(self.u32[a >> 2])

    def s(self, a: int) -> int:
        return int(self.s32[a >> 2])

    def player(self) -> dict:
        p = self.players
        mo = self.u(p + PL_MO)
        d = {"state": self.s(p + PL_STATE), "health": self.s(p + PL_HEALTH),
             "kills": self.s(p + PL_KILLS), "mo": mo}
        if mo:
            d.update(x=self.s(mo + MO_X) / 65536, y=self.s(mo + MO_Y) / 65536, z=self.s(mo + MO_Z),
                     angle=self.u(mo + MO_ANGLE) * 360 / 2**32)
        return d

    def thinkers(self):
        """(address, function) of every thinker, in list order."""
        t, n = self.u(self.thinkercap + TH_NEXT), 0
        while t != self.thinkercap and n < 100000:
            yield t, self.u(t + TH_FUNC)
            t, n = self.u(t + TH_NEXT), n + 1

    def monsters(self) -> list[dict]:
        out = []
        for t, f in self.thinkers():
            if f != self.mobj_thinker:
                continue
            flags = self.s(t + MO_FLAGS)
            if flags & MF_COUNTKILL and flags & MF_SHOOTABLE and self.s(t + MO_HEALTH) > 0:
                out.append({"addr": t, "x": self.s(t + MO_X) / 65536,
                            "y": self.s(t + MO_Y) / 65536, "awake": self.u(t + MO_TARGET) != 0})
        return out

    def walls(self):
        """The level's lines as arrays: endpoints (map units) and front / back sector indices
        (-1: none). Vertexes are DROM, lines and sectors RAM."""
        h = self.h
        lines, n = h.word(h.symbol("lines")), h.word(h.symbol("numlines"))
        sectors = self.u(h.symbol("sectors"))
        L = self.s32[lines >> 2:(lines + n * LINE_SIZE) >> 2].reshape(n, LINE_SIZE // 4)
        v1, v2 = L[:, LN_V1 // 4].astype(np.int64), L[:, LN_V2 // 4].astype(np.int64)
        dv = h.drom.view(np.int32)
        def xy(v):
            return dv[(v - H.DROM_BASE) >> 2] / 65536, dv[(v + 4 - H.DROM_BASE) >> 2] / 65536
        (x1, y1), (x2, y2) = xy(v1), xy(v2)
        sec = lambda p: np.where(p != 0, (p.astype(np.int64) - sectors) // SECTOR_SIZE, -1)
        return x1, y1, x2, y2, sec(L[:, LN_FRONT // 4]), sec(L[:, LN_BACK // 4]), sectors

    def sight(self, walls, ax, ay, bx, by, z) -> bool:
        """A rough line of sight from a to b at eye level (floor z): no one-sided line, closed
        two-sided line (a shut door) or two-sided line whose floor rises above the eye or whose
        ceiling comes down below it crosses it (good enough to aim a bot)."""
        x1, y1, x2, y2, front, back, sectors = walls
        ns = int(max(front.max(), back.max())) + 1
        S = self.s32[sectors >> 2:(sectors + ns * SECTOR_SIZE) >> 2].reshape(ns, SECTOR_SIZE // 4)
        floor, ceil = S[:, SEC_FLOOR // 4], S[:, SEC_CEIL // 4]
        f, b = np.maximum(front, 0), np.maximum(back, 0)
        top, bottom = np.minimum(ceil[f], ceil[b]), np.maximum(floor[f], floor[b])
        eye = z + 41 * 65536
        closed = (back < 0) | (top <= bottom) | (bottom > eye) | (top < eye)
        def side(px, py, qx, qy, rx, ry):
            return (qx - px) * (ry - py) - (qy - py) * (rx - px)
        d1, d2 = side(ax, ay, bx, by, x1, y1), side(ax, ay, bx, by, x2, y2)
        d3, d4 = side(x1, y1, x2, y2, ax, ay), side(x1, y1, x2, y2, bx, by)
        cross = (d1 * d2 < 0) & (d3 * d4 < 0)
        return not np.any(cross & closed)

    def census(self) -> dict:
        """Thinker / mobj counts."""
        th = mo = 0
        for _, f in self.thinkers():
            th += 1
            mo += f == self.mobj_thinker
        return {"thinkers": th, "mobjs": mo}

    def zone_usage(self) -> dict:
        """The level zone (sim RAM): bytes in use (headers included) by tag, the highest end of a
        used block, the largest free block."""
        if not self.zone:
            self.zone = self.h.word(self.h.symbol("levelzone"))
        u = self.u if self.zone < H.RAM_SIZE else self.h.word  # (-qdmeasure: a host-side zone)
        s = lambda a: u(a) - (u(a) >> 31 << 32)
        head = self.zone + 4  # &levelzone->blocklist
        used, by_tag, top, free_max = 0, {}, 0, 0
        b, n = u(head + ZB_NEXT), 0
        while b != head and n < 1 << 20:
            size, tag = s(b + ZB_SIZE), s(b + ZB_TAG)
            if tag == PU_FREE:
                free_max = max(free_max, size)
            else:
                used += size
                name = TAG_NAMES.get(tag, str(tag))
                by_tag[name] = by_tag.get(name, 0) + size
                top = max(top, b + size)
            b, n = u(b + ZB_NEXT), n + 1
        return {"used": used, "by_tag": by_tag, "top": top, "free_max": free_max,
                "size": s(self.zone)}


class Bot:
    """Deterministic keyboard player: waypoints, doors, aim and fire at awake monsters."""

    TURN_TOL = 4.0  # degrees
    FIRE_TOL = 3.0
    ENGAGE = 1100.0  # map units
    GIVE_UP = 140  # tics on one foe

    def __init__(self, h: H.Host, route=E1M1_ROUTE):
        self.h, self.view, self.route = h, SimView(h), list(route)
        self.wp, self.down, self.tic = 0, set(), 0
        self.last_pos, self.stuck_tics, self.strafe_left = None, 0, 0
        self.walls, self.lines = None, None
        self.foe, self.foe_tics, self.ignore = None, 0, {}  # give up on a foe after a while
        self.leveltime, self.last_leveltime = h.symbol("leveltime"), 0

    def _set(self, want: set):
        for k in sorted(self.down - want):
            self.h.key(k, False)
        for k in sorted(want - self.down):
            self.h.key(k, True)
        self.down = want

    @staticmethod
    def _diff(a: float, b: float) -> float:
        return (a - b + 180) % 360 - 180

    def step(self):
        """Choose the keys held for the next tic."""
        self.tic += 1
        p = self.view.player()
        want = set()
        mons = self.view.monsters()
        if p["state"] == PST_DEAD or not p["mo"]:
            if self.tic % 20 < 10:  # tap use to respawn
                want.add(H.KEY_USE)
            self._set(want)
            return
        x, y, ang = p["x"], p["y"], p["angle"]
        leveltime = self.view.u(self.leveltime)
        if leveltime < self.last_leveltime:  # the level restarted (died): start the route over
            self.wp, self.ignore, self.stuck_tics = 0, {}, 0
        self.last_leveltime = leveltime
        lines = self.h.word(self.h.symbol("lines"))
        if lines != self.lines:  # a new level
            self.walls, self.lines = self.view.walls(), lines
        # the nearest awake monster in range and in sight: turn to it, fire when lined up
        foe = min(((math.hypot(m["x"] - x, m["y"] - y), m) for m in mons if m["awake"]
                   and self.ignore.get(m["addr"], 0) < self.tic
                   and math.hypot(m["x"] - x, m["y"] - y) < self.ENGAGE
                   and self.view.sight(self.walls, x, y, m["x"], m["y"], p["z"])),
                  default=(None, None), key=lambda t: t[0])
        if foe[0] is not None:
            m = foe[1]
            self.foe_tics = self.foe_tics + 1 if m["addr"] == self.foe else 0
            self.foe = m["addr"]
            if self.foe_tics > self.GIVE_UP:  # out of reach: leave it alone for a while
                self.ignore[m["addr"]] = self.tic + 10 * self.GIVE_UP
            d = self._diff(math.degrees(math.atan2(m["y"] - y, m["x"] - x)), ang)
            if d > self.TURN_TOL:
                want.add(H.KEY_LEFTARROW)
            elif d < -self.TURN_TOL:
                want.add(H.KEY_RIGHTARROW)
            if abs(d) < self.FIRE_TOL:
                want.add(H.KEY_FIRE)
            self._set(want)
            return
        if self.tic < 40:  # a few shots at the start wake up whoever can hear them
            if self.tic % 8 < 4:
                want.add(H.KEY_FIRE)
            self._set(want)
            return
        if not self.route:  # no route (other maps): hold the ground, fight what comes
            self._set(want)
            return
        wp = self.route[self.wp % len(self.route)]
        dist = math.hypot(wp[0] - x, wp[1] - y)
        if dist < 48:
            self.wp += 1
            wp = self.route[self.wp % len(self.route)]
            dist = math.hypot(wp[0] - x, wp[1] - y)
        d = self._diff(math.degrees(math.atan2(wp[1] - y, wp[0] - x)), ang)
        if d > self.TURN_TOL:
            want.add(H.KEY_LEFTARROW)
        elif d < -self.TURN_TOL:
            want.add(H.KEY_RIGHTARROW)
        if abs(d) < 30:
            want.add(H.KEY_UPARROW)
            if dist > 256:
                want.add(H.KEY_RSHIFT)
        if len(wp) > 2 and wp[2] == "use" and dist < 96 and self.tic % 10 < 5:
            want.add(H.KEY_USE)
        # walking into something: it may be a door (use), else strafe for a moment
        moved = self.last_pos and math.dist((x, y), self.last_pos) >= 0.5
        self.stuck_tics = self.stuck_tics + 1 if H.KEY_UPARROW in want and not moved else 0
        if 8 <= self.stuck_tics < 14:
            want.add(H.KEY_USE)
        if self.stuck_tics > 60:
            self.strafe_left, self.stuck_tics = 16, 0
        if self.strafe_left:
            self.strafe_left -= 1
            want.add(H.KEY_STRAFE_R if (self.wp // 2) % 2 else H.KEY_STRAFE_L)
        self.last_pos = (x, y)
        self._set(want)


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else 0.0


def summary(a) -> dict:
    a = np.asarray(a, np.float64)
    if not len(a):
        return {}
    return {"min": int(a.min()), "median": int(np.median(a)), "p95": int(pct(a, 95)),
            "max": int(a.max()), "mean": round(float(a.mean()), 1), "n": int(len(a))}


def save_png(path: Path, rgb: np.ndarray):
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).resize((640, 400), Image.NEAREST).save(path)


def play(tics: int, args=H.DEFAULT_ARGS, route=E1M1_ROUTE, frames=(), frames_dir=FRAMES_DIR,
         prefix="e1m1", bot=True, log=None, ref=False) -> dict:
    """Run `tics` sim tics in native mode with the bot; return the measurements. With ref, the
    trace track's reference machine runs every tick too, fed only the snapshot and the pokes
    (as the model would be), and must match the native tick's RAM, instruction and token
    counts (``ref_mismatches``)."""
    h = H.Host(args=args)
    checker = RefCheck(h) if ref else None
    b = Bot(h, route) if bot else None
    view = SimView(h)
    rec = {k: [] for k in ("sim_instr", "sim_tokens", "pokes", "input_words", "awake", "min_sp",
                           "loads", "stores")}
    host_per_frame, zone_used, zone_top = [], [], []
    t_host = t_sim = 0.0
    nframes = ntics = 0
    resyncs_at, trail = [], []
    last_hi = h.instructions()
    error = None
    census = {}
    try:
        pending = sorted(frames)  # save the first frame drawn after each of these tics
        while ntics < tics or pending and pending[0] <= tics:
            t0 = time.perf_counter()
            ev = h.run()
            t_host += time.perf_counter() - t0
            if ev == H.SIM_TICK:
                if h.need_resync():
                    resyncs_at.append(ntics)
                    h.resync()
                    if checker:
                        checker.resync()
                pokes = h.take_pokes()
                if checker:
                    checker.run(pokes)
                t0 = time.perf_counter()
                st = h.sim_tick_native()
                t_sim += time.perf_counter() - t0
                if checker:
                    checker.compare(st, ntics)
                h.complete_sim_tick()
                ntics += 1
                rec["sim_instr"].append(st.instructions)
                rec["sim_tokens"].append(st.tokens)
                rec["pokes"].append((len(pokes) - 1) // 2)
                rec["input_words"].append(st.input_words)
                rec["min_sp"].append(st.min_sp)
                rec["loads"].append(st.loads)
                rec["stores"].append(st.stores)
                z = view.zone_usage()
                zone_used.append(z["used"])
                zone_top.append(z["top"])
                rec["awake"].append(sum(m["awake"] for m in view.monsters()))
                if ntics % 5 == 0:
                    pl = view.player()
                    trail.append((ntics, round(pl.get("x", 0)), round(pl.get("y", 0))))
                if ntics in frames or ntics == tics:
                    census[ntics] = view.census()
            elif ev == H.FRAME:
                hi = h.instructions()
                host_per_frame.append(hi - last_hi)
                last_hi = hi
                nframes += 1
                while pending and pending[0] <= ntics:
                    if frames_dir:
                        save_png(Path(frames_dir) / f"{prefix}-tic{ntics:05d}.png", h.frame_rgb())
                    pending.pop(0)
                if b and ntics:
                    b.step()
            elif ev == H.EXIT:
                error = f"exit {h.exit_code}"
                break
            elif ev == H.ERROR:
                error = h.error
                break
            if log and ntics and ntics % 700 == 0 and ev == H.SIM_TICK:
                log(f"  tic {ntics}: {view.player()}")
    except H.HostError as e:
        error = str(e)
    c = h.counters()
    si = np.asarray(rec["sim_instr"], np.int64)
    awake = np.asarray(rec["awake"])
    res = {
        "args": list(args), "tics": ntics, "frames": nframes, "error": error,
        "sim_instructions": summary(si),
        "sim_instructions_awake": summary(si[awake > 0]) if len(si) else {},
        "sim_instructions_idle": summary(si[awake == 0]) if len(si) else {},
        "sim_tokens": summary(rec["sim_tokens"]),
        "tokens_per_instruction": round(float(np.sum(rec["sim_tokens"]) / max(1, si.sum())), 3),
        "pokes": summary(rec["pokes"]),
        "pokes_after_first": summary(rec["pokes"][1:]),
        "input_words": summary(rec["input_words"]),
        "loads": summary(rec["loads"]), "stores": summary(rec["stores"]),
        "awake_monsters": summary(rec["awake"]),
        "tics_with_awake_monsters": int((awake > 0).sum()),
        "host_instructions_per_frame": summary(host_per_frame[1:]),
        "host_instructions_total": h.instructions(),
        "sim_instructions_total": int(si.sum()),
        "host_mips": round(h.instructions() / t_host / 1e6, 1) if t_host else 0,
        "sim_mips": round(si.sum() / t_sim / 1e6, 1) if t_sim else 0,
        "wall_s": {"host": round(t_host, 3), "sim": round(t_sim, 3)},
        "stack_peak_bytes": 0x3000 - min(rec["min_sp"]) if rec["min_sp"] else 0,
        "level_zone": {"size": view.zone_usage()["size"], "peak_used": max(zone_used, default=0),
                       "peak_top": max(zone_top, default=0),
                       "final": view.zone_usage()},
        "drom": drom_usage(h),
        "census": census,
        "resyncs_at_tic": resyncs_at,
        "counters": c,
        "ref_ticks_checked": checker.ticks if checker else 0,
        "ref_mismatches": checker.mismatches if checker else None,
        "player": view.player(),
        "trail": trail,
    }
    if b:
        res["bot_waypoint"] = b.wp
    h.close()
    return res


class RefCheck:
    """The trace track's reference machine (qwendoom.trace.Ref) as a model stand-in: its own
    RAM, loaded at each resync and then fed only the pokes; it must stay equal to the host's."""

    def __init__(self, h: H.Host):
        from qwendoom import trace as T
        self.T, self.h, self.ref = T, h, None
        self.mem = T.Memory(T.Map.qwendoom())
        self.ticks, self.mismatches = 0, []

    def resync(self):
        h, m = self.h, self.mem
        m.ram[:], m.drom[:], m.rom[:] = h.ram, h.drom, h.rom
        pc, sp, gp = h.sim_entry()
        self.ref = self.T.Ref(m)
        self.ref.set_context(pc=pc, regs=[0, 0, sp, gp] + [0] * 28)

    def run(self, pokes):
        self.ref.push_input(pokes)
        s0 = self.ref.counts["steps"]
        self.tokens, self.why = self.ref.run_count(stop_yield=True)
        self.steps = self.ref.counts["steps"] - s0

    def compare(self, st, tic: int):
        self.ticks += 1
        diff = np.nonzero(self.mem.ram != self.h.ram)[0]
        if self.why != "yield" or (self.steps, self.tokens) != (st.instructions, st.tokens) \
                or len(diff):
            self.mismatches.append({"tic": tic, "ref": [self.why, self.steps, self.tokens],
                                    "native": [st.instructions, st.tokens],
                                    "ram_diffs": [int(a) for a in diff[:8]]})


def drom_usage(h: H.Host) -> dict:
    """DROM: the sim's .rodata and the level-data bump region (qd_platform.c's allocator)."""
    start, end = h.symbol("__drom_start"), h.symbol("__drom_end")
    top, mark, peak = (h.word(h.symbol(n)) for n in ("drom_top", "drom_mark", "drom_peak"))
    return {"rodata": end - start, "permanent": (mark - end) if mark else 0,
            "level": (top - mark) if top else 0, "peak_end": peak,
            "peak_used": (peak - H.DROM_BASE) if peak else end - H.DROM_BASE,
            "size": H.DROM_SIZE}


def level_needs(episode: int, map_: int) -> dict:
    """What a level needs at load time (before its first tic): level-zone bytes (sim RAM) by tag
    and level-data bytes (DROM), from a -qdmeasure run (level data in host memory)."""
    h = H.Host(args=("-warp", str(episode), str(map_), "-skill", "3", "-qdmeasure"))
    while True:
        ev = h.run()
        if ev == H.SIM_TICK:
            break
        if ev in (H.ERROR, H.EXIT):
            raise H.HostError(h.error)
    z = SimView(h).zone_usage()
    drom_rodata = h.symbol("__drom_end") - H.DROM_BASE
    need = {"zone_used": z["used"], "zone_by_tag": z["by_tag"],
            "drom_level": h.word(h.symbol("qd_drom_level_need")),
            "ram_available": H.RAM_SIZE - h.symbol("__sim_ram_end"),
            "drom_available": H.DROM_SIZE - drom_rodata - drom_permanent(h)}
    h.close()
    return need


def drom_permanent(h: H.Host) -> int:
    """Startup data in the DROM bump region (textureheight), measured on a normal build."""
    return 4 * h.word(h.symbol("numtextures"))


def map_names(wad=H.DEFAULT_WAD) -> list[tuple[int, int]]:
    """(episode, map) of every ExMy map in the WAD."""
    import re
    import struct
    b = Path(wad).read_bytes()
    n, off = struct.unpack_from("<II", b, 4)
    out = []
    for i in range(n):
        nm = b[off + 16 * i + 8:off + 16 * i + 16].rstrip(b"\0").decode(errors="replace")
        m = re.fullmatch(r"E(\d)M(\d)", nm)
        if m:
            out.append((int(m[1]), int(m[2])))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tics", type=int, default=35 * 60)
    ap.add_argument("--frames", default="1,140,420,700,1050,1400,1750,2100")
    ap.add_argument("--json", default=str(ROOT / "build" / "doom-measure.json"))
    ap.add_argument("--maps", help="'all' or a list like E1M1,E2M3: one run per map")
    ap.add_argument("--ref", action="store_true",
                    help="check every tick against the trace reference machine (model stand-in)")
    args = ap.parse_args()
    frames = {int(f) for f in args.frames.split(",") if f}
    if args.maps:
        maps = map_names() if args.maps == "all" else [
            (int(m[1]), int(m[3])) for m in args.maps.upper().split(",")]
        out = {}
        for e, m in maps:
            name = f"E{e}M{m}"
            r = play(args.tics, args=("-warp", str(e), str(m), "-skill", "3"),
                     route=E1M1_ROUTE if (e, m) == (1, 1) else [],
                     frames={args.tics}, prefix=name.lower())
            r["needs_at_load"] = need = level_needs(e, m)
            out[name] = r
            z, d = r["level_zone"], r["drom"]
            print(f"{name}: {'OK  ' if not r['error'] else 'FAIL'} at load: zone "
                  f"{need['zone_used']:7} / {need['ram_available']} B, DROM level data "
                  f"{need['drom_level']:7} / {need['drom_available']} B | {r['tics']:4} tics: "
                  f"zone peak {z['peak_used']:6} B, stack {r['stack_peak_bytes']:4} B, "
                  f"sim instr/tic median {r['sim_instructions'].get('median')} "
                  f"max {r['sim_instructions'].get('max')}"
                  + (f" | {r['error']}" if r["error"] else ""), flush=True)
        Path(args.json).write_text(json.dumps(out, indent=1))
        return 0
    r = play(args.tics, frames=frames, log=print, ref=args.ref)
    Path(args.json).write_text(json.dumps(r, indent=1))
    print(json.dumps({k: v for k, v in r.items() if k not in ("counters",)}, indent=1))
    return 0 if not r["error"] else 1


if __name__ == "__main__":
    sys.exit(main())
