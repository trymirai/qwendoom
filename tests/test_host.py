"""The host computer (native/host) running DOOM (build/doom.elf): boot into E1M1, the native sim
mode and its region checks, pokes (every host write reaches the sim), determinism, menus, the
title screen without demos, and the sim against the trace track's reference machine.

The builds are brought up to date first (make -C doom, make -C native/host)."""

import importlib.util
import subprocess
import struct
from pathlib import Path

import numpy as np
import pytest

from qwendoom import host as H

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module", autouse=True)
def build():
    for d in ("doom", "native/host"):
        r = subprocess.run(["make", "-s", "-C", str(ROOT / d)], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr


def measure():
    spec = importlib.util.spec_from_file_location("measure", ROOT / "doom" / "measure.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# a fixed key script: tic -> [(action, down)], applied at the frame after that tic
SCRIPT = {
    3: [("forward", True)], 40: [("left", True)], 52: [("left", False), ("fire", True)],
    70: [("fire", False), ("strafeleft", True)], 90: [("strafeleft", False), ("run", True)],
    120: [("right", True)], 150: [("right", False), ("use", True)], 155: [("use", False)],
    200: [("forward", False), ("back", True)], 230: [("back", False), ("fire", True)],
    260: [("fire", False), ("run", False)],
}


def play(h: H.Host, tics: int, script=SCRIPT, on_tick=None) -> list[int]:
    """Run `tics` native sim tics, pressing the script's keys; returns instructions per tic."""
    counts, n = [], 0
    while n < tics:
        ev = h.run()
        if ev == H.SIM_TICK:
            st, pokes = h.native_tick()
            counts.append(st.instructions)
            n += 1
            if on_tick:
                on_tick(st, pokes)
        elif ev == H.FRAME:
            for action, down in script.get(n, ()):
                h.action(action, down)
        else:
            raise H.HostError(f"event {ev}: {h.error}")
    return counts


def playpal(wad=H.DEFAULT_WAD) -> bytes:
    b = Path(wad).read_bytes()
    n, off = struct.unpack_from("<II", b, 4)
    for i in range(n):
        pos, size, name = struct.unpack_from("<II8s", b, off + 16 * i)
        if name.rstrip(b"\0") == b"PLAYPAL":
            return b[pos:pos + size]
    raise KeyError("PLAYPAL")


def test_boot_e1m1():
    """Straight into E1M1 at skill 3 (sk_medium = 2), one sim tick per frame, the right palette."""
    with H.Host() as h:
        ev = h.run()
        while ev != H.SIM_TICK:  # the level loads before the first tick
            assert ev in (H.FRAME,), h.error
            ev = h.run()
        assert h.need_resync()
        pc, sp, gp = h.sim_entry()
        assert pc == h.symbol("sim_main") and sp == 0x3000 and gp == h.symbol("__global_pointer$")
        assert H.ROM_BASE <= pc < H.ROM_BASE + H.ROM_SIZE and 0x3000 <= gp < 0x4000
        h.native_tick()
        assert (h.word(h.symbol("gameepisode")), h.word(h.symbol("gamemap")),
                h.word(h.symbol("gameskill"))) == (1, 1, 2)
        for _ in range(60):  # past the wipe: then exactly one tic per frame
            h.next_frame()
        ticks = []
        for _ in range(35):
            t0 = h.counters()["sim_ticks"]
            h.next_frame()
            ticks.append(h.counters()["sim_ticks"] - t0)
        assert ticks == [1] * 35
        idx, pal = h.frame()
        assert idx.shape == (200, 320) and len(np.unique(idx)) > 50
        gt = h.symbol("gammatable") - H.HOST_BASE  # DOOM's gamma 0 (i -> i + 1, mostly)
        expect = h.host[gt:gt + 256][np.frombuffer(playpal()[:768], np.uint8)]
        assert np.array_equal(pal.reshape(-1), expect)  # palette 0: no damage / pickup tint
        c = h.counters()
        assert c["levels"] == 1 and c["resyncs"] == 1 and c["drom_writes_midlevel"] == 0
        assert c["host_null_accesses"] == 0


def test_short_run_and_pokes():
    """A scripted run: no region violation; pokes are sorted aligned RAM words ending with 0."""
    seen = {"pokes": 0, "min_sp": 0x3000}

    def check(st, pokes):
        assert pokes[-1] == 0 and len(pokes) % 2 == 1
        addrs = pokes[:-1:2]
        assert np.all(np.diff(addrs.astype(np.int64)) > 0)
        assert np.all(addrs % 4 == 0) and np.all(addrs >= 0x100) and np.all(addrs < H.RAM_SIZE)
        assert st.input_words == len(pokes)
        seen["pokes"] += len(addrs)
        seen["min_sp"] = min(seen["min_sp"], st.min_sp)

    with H.Host() as h:
        counts = play(h, 280, on_tick=check)
        assert len(counts) == 280 and min(counts) > 1000
        assert 0x100 < seen["min_sp"] < 0x3000 and seen["pokes"] > 280
        assert h.counters()["drom_writes_midlevel"] == 0


def test_determinism():
    """Same inputs, same machine: RAM, DROM, frames and instruction counts after N tics; a
    different input diverges (so the comparison can tell)."""
    runs = []
    for script in (SCRIPT, SCRIPT, {**SCRIPT, 100: [("fire", True)]}):
        with H.Host() as h:
            counts = play(h, 300, script)
            h.next_frame()
            runs.append((h.ram.copy(), h.drom.copy(), h.frame()[0], counts,
                         h.instructions(), h.counters()))
    a, b, c = runs
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert np.array_equal(a[2], b[2]) and a[3] == b[3] and a[4] == b[4] and a[5] == b[5]
    assert not np.array_equal(a[0], c[0])


def test_pokes_reach_the_sim():
    """Model-mode data flow: a mirror that sees only the snapshot, the pokes and the sim's own
    stores holds exactly the host's RAM before every tick (the dirty tracking misses nothing)."""
    with H.Host() as h:
        mirror, n = None, 0
        while n < 300:
            ev = h.run()
            if ev == H.SIM_TICK:
                if h.need_resync():
                    h.resync()
                    mirror = h.ram.copy()
                pokes = h.take_pokes()
                for a, v in zip(pokes[:-1:2], pokes[1:-1:2]):
                    mirror[a:a + 4] = np.frombuffer(np.uint32(v).tobytes(), np.uint8)
                diff = np.nonzero(mirror != h.ram)[0]
                missed = [h.symbolize(int(a)) for a in diff[:8]]
                assert not len(diff), f"tick {n}: host writes not poked at {missed}"
                h.sim_tick_native()
                h.complete_sim_tick()
                mirror = h.ram.copy()  # (in model mode the trace decoder applies the stores)
                n += 1
            elif ev == H.FRAME:
                for action, down in SCRIPT.get(n, ()):
                    h.action(action, down)
            else:
                raise H.HostError(h.error)


def test_against_reference_machine():
    """The native sim against the trace track's C reference machine fed only the snapshot and
    the pokes: the same RAM after every tick, the same instruction and token counts."""
    T = pytest.importorskip("qwendoom.trace")
    try:
        T.lib()
    except Exception as e:  # the other track's library is not built
        pytest.skip(f"libqdtrace unavailable: {e}")
    with H.Host() as h:
        mem, ref, n = T.Memory(T.Map.qwendoom()), None, 0
        while n < 200:
            ev = h.run()
            if ev == H.SIM_TICK:
                if h.need_resync():
                    h.resync()
                    mem.ram[:], mem.drom[:], mem.rom[:] = h.ram, h.drom, h.rom
                    pc, sp, gp = h.sim_entry()
                    ref = T.Ref(mem)
                    ref.set_context(pc=pc, regs=[0, 0, sp, gp] + [0] * 28)
                pokes = h.take_pokes()
                ref.push_input(pokes)
                steps0 = ref.counts["steps"]
                ntok, why = ref.run_count(stop_yield=True)
                st = h.sim_tick_native()
                h.complete_sim_tick()
                assert why == "yield"
                assert (ref.counts["steps"] - steps0, ntok) == (st.instructions, st.tokens)
                assert np.array_equal(mem.ram, h.ram), f"tick {n}"
                n += 1
            elif ev == H.FRAME:
                for action, down in SCRIPT.get(n, ()):
                    h.action(action, down)
            else:
                raise H.HostError(h.error)


def tick_with(corrupt) -> str:
    """Boot, run 5 tics, corrupt sim RAM with corrupt(h), run a tick: the error text."""
    with H.Host() as h:
        play(h, 5, {})
        while h.run() != H.SIM_TICK:
            pass
        corrupt(h)
        h.take_pokes()
        with pytest.raises(H.HostError) as e:
            h.sim_tick_native()
        assert h.run() == H.ERROR
        return str(e.value)


def set_word(h: H.Host, addr: int, value: int):
    h.ram[addr:addr + 4] = np.frombuffer(np.uint32(value).tobytes(), np.uint8)


def player_mo(h):
    return h.symbol("players")  # players[0].mo is the first word


def test_region_check_load():
    msg = tick_with(lambda h: set_word(h, player_mo(h), h.symbol("gamestate")))  # a HOST variable
    assert "sim load" in msg and "HOST" in msg and "P_PlayerThink" in msg, msg


def test_region_check_store():
    # a mobj in DROM: its loads pass, its first store does not
    msg = tick_with(lambda h: set_word(h, player_mo(h), H.DROM_BASE + 0x1000))
    assert "sim store" in msg and "0x81" in msg and "DROM" in msg and "P_" in msg, msg


def test_region_check_null_guard():
    msg = tick_with(lambda h: set_word(h, player_mo(h), 0))
    assert "sim load" in msg and "from 0x68" in msg, msg  # mo->flags of a NULL mo


def test_region_check_fetch():
    def corrupt(h):
        first = h.word(h.symbol("thinkercap") + 4)  # thinkercap.next
        set_word(h, first + 8, h.symbol("D_DoomMain"))  # its function: host code
    msg = tick_with(corrupt)
    assert "sim jump to" in msg and "(D_DoomMain+0, HOST)" in msg and "P_Ticker" in msg, msg


def test_menu_pauses_the_sim():
    """Esc opens the menu on the host: no sim ticks while it is up; Esc again resumes."""
    with H.Host() as h:
        play(h, 70, {})
        menuactive = h.symbol("menuactive")
        h.action("menu", True)
        h.next_frame()
        h.action("menu", False)
        h.next_frame()
        assert h.word(menuactive) == 1
        t0 = h.counters()["sim_ticks"]
        for _ in range(35):
            assert h.next_frame() == H.FRAME
        assert h.counters()["sim_ticks"] == t0
        h.action("menu", True)
        h.next_frame()
        h.action("menu", False)
        for _ in range(5):
            h.next_frame()
        assert h.word(menuactive) == 0 and h.counters()["sim_ticks"] > t0


def test_title_without_demos_then_new_game():
    """Without -warp: the title pages and no attract demos (no sim tick); the menu starts a game."""
    with H.Host(args=()) as h:
        for _ in range(800):
            h.next_frame()
        assert h.counters()["sim_ticks"] == 0 and h.counters()["levels"] == 0
        for action in ("menu", "enter", "enter", "enter"):  # new game, episode 1, default skill
            h.action(action, True)
            h.next_frame()
            h.action(action, False)
            for _ in range(8):
                h.next_frame()
        for _ in range(40):
            h.next_frame()
        assert h.counters()["levels"] == 1 and h.counters()["sim_ticks"] > 0
        assert h.word(h.symbol("gameskill")) == 2


def test_action_keys():
    actions = ["left", "right", "forward", "back", "strafeleft", "straferight", "fire", "use",
               "run", "menu", "enter", "weapon1", "weapon2", "weapon3"]
    for a in actions:
        assert H.lib().qd_host_action_key(a.encode()) == H.ACTION_KEYS[a]
    assert H.lib().qd_host_action_key(b"jump") == -1


def test_split_is_consistent():
    """The linked ELF: sim functions in ROM and nothing else there, every ROM instruction one the
    circuit runs, sim data within the regions, and E1M1 measured to fit (doom/measure.py)."""
    import json
    a = json.loads((ROOT / "build" / "doom-split.json").read_text())
    L = a["linked"]
    assert L["isa_errors"] == 0 and L["dropped_by_gc"] == 0 and not a["suspicious"]
    assert L["rom_bytes"] <= H.ROM_SIZE and L["rom_instructions"] * 4 == L["rom_bytes"]
    with H.Host() as h:
        for name in ("sim_main", "P_Ticker", "P_MobjThinker", "T_MoveFloor", "A_Chase", "FixedDiv"):
            assert H.ROM_BASE <= h.symbol(name) < H.ROM_BASE + H.ROM_SIZE, name
        for name in ("D_DoomMain", "R_RenderPlayerView", "G_Ticker", "P_SetupLevel", "M_Drawer"):
            assert h.symbol(name) >= H.HOST_BASE, name
        for name in ("players", "thinkercap", "leveltime", "levelzone"):
            assert 0x3000 <= h.symbol(name) < H.RAM_SIZE, name
        for name in ("states", "mobjinfo", "finesine"):
            assert H.DROM_BASE <= h.symbol(name) < H.DROM_BASE + H.DROM_SIZE, name
    m = measure()
    need = m.level_needs(1, 1)
    assert need["zone_used"] < need["ram_available"] and need["drom_level"] < need["drom_available"]


def test_save_and_load(tmp_path):
    """Savegames through the menus (F2 / F3) in the config directory; loading one reloads the
    level (a resync) and the pokes stay complete after it."""
    with H.Host(config_dir=tmp_path) as h:
        def press(key, frames=6):
            h.key(key, True)
            h.next_frame()
            h.key(key, False)
            for _ in range(frames):
                h.next_frame()

        play(h, 60, {3: [("forward", True)], 40: [("forward", False)]})
        saved = h.word(h.symbol("leveltime"))
        press(0x80 + 0x3c)  # F2: save; slot 0, name "a", enter
        press(H.KEY_ENTER)
        press(ord("a"))
        press(H.KEY_ENTER, 10)
        assert (tmp_path / ".savegame" / "doomsav0.dsg").stat().st_size > 1000
        play(h, 100, {})
        assert h.word(h.symbol("leveltime")) > saved + 100
        levels = h.counters()["levels"]
        press(0x80 + 0x3d, 3)  # F3: load slot 0 (enter, below)
        h.key(H.KEY_ENTER, True)
        mirror, n, frames = None, 0, 0
        while n < 30:  # the first tick after the load resyncs; the mirror then holds
            ev = h.run()
            if ev == H.FRAME:
                frames += 1
                if frames == 1:
                    h.key(H.KEY_ENTER, False)
            if ev == H.SIM_TICK:
                if h.need_resync():
                    h.resync()
                    mirror = h.ram.copy()
                    assert abs(h.word(h.symbol("leveltime")) - saved) < 20
                pokes = h.take_pokes()
                for a, v in zip(pokes[:-1:2], pokes[1:-1:2]):
                    mirror[a:a + 4] = np.frombuffer(np.uint32(v).tobytes(), np.uint8)
                assert mirror is not None and np.array_equal(mirror, h.ram)
                h.sim_tick_native()
                h.complete_sim_tick()
                mirror = h.ram.copy()
                n += 1
            elif ev != H.FRAME:
                raise H.HostError(h.error)
        assert h.counters()["levels"] == levels + 1


def test_palettes_follow_playpal():
    """Every frame's palette is one of PLAYPAL's 14 through the gamma table: the bot (doom/
    measure.py) takes damage (red, 1-8) and picks things up (gold, 9-12) within 1,200 frames."""
    with H.Host() as h:
        bot = measure().Bot(h)
        gt = h.symbol("gammatable") - H.HOST_BASE
        pp = np.frombuffer(playpal(), np.uint8)[:14 * 768].reshape(14, 768)
        expect = [h.host[gt:gt + 256][pp[k]].tobytes() for k in range(14)]
        seen = set()
        for _ in range(1200):
            h.next_frame()
            bot.step()
            seen.add(expect.index(h.frame()[1].tobytes()))  # ValueError: not a PLAYPAL palette
        assert 0 in seen and seen & set(range(1, 9)) and seen & set(range(9, 13)), seen
