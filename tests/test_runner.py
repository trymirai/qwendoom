"""The runner (qwendoom/runner.py + native/runner): DOOM with its sim ticks in the model, check,
ref and native modes, and the model game source.

Fast by default: button scripts, the dylib's make, I_Error's message, ref mode against native
mode (the C hot loop with the reference machine as the token source), a failed tick's counts,
the token log after close, a few model-mode ticks (check mode, and a GO after the resync failing
it; forced steps against full steps, state rows included; pause / cancel / the live counter from
another thread) and the model source over the WebSocket. QD_SLOW=1 adds the long runs of
docs/run.md: check mode over E1M1_WALK (362 ticks), model mode against native mode (RAM after
every tick, every frame) and a save / load resync in model mode."""

import asyncio
import os
import subprocess
import threading
import time

import numpy as np
import pytest

from qwendoom import host as H
from qwendoom import runner as R
from qwendoom import spec
from qwendoom import trace as T

SLOW = not os.environ.get("QD_SLOW")
pytestmark = pytest.mark.skipif(
    not (H.LIB.exists() and H.DEFAULT_ELF.exists() and H.DEFAULT_WAD.exists()),
    reason="needs build/libqdhost.dylib, build/doom.elf and freedoom1.wad")
needs_weights = pytest.mark.skipif(
    not (R.DEFAULT_WEIGHTS / "model.safetensors").exists(), reason="needs weights/")

KEY_F2, KEY_F3 = 0x80 + 0x3C, 0x80 + 0x3D
SHORT_WALK = "forward@40-100,right@60-70,fire@50-53+80-83,use@90-92"


def lockstep(a: R.Runner, b: R.Runner, tics: int, buttons=None, keys=None,
             on_tick=None) -> dict:
    """Run two runners on the same input, event by event: after every sim tick their RAM (all
    256 KB) and the tick's token and instruction counts must be equal, after every frame the
    frames. keys: {tic: [(doom key, down)]} pressed before that tic's frame."""
    out = {"ticks": 0, "frames": 0, "tokens": 0}
    for tic in range(tics):
        for r in (a, b):
            if buttons is not None:
                r.hold(buttons(tic))
            for k, down in (keys or {}).get(tic, []):
                r.key(k, down)
        while True:
            ea, eb = a.step(), b.step()
            assert ea == eb, (tic, ea, eb)
            if ea == "tick":
                ta, tb = a.last_tick, b.last_tick
                assert (ta.tokens, ta.instructions, ta.injected, ta.words) == \
                    (tb.tokens, tb.instructions, tb.injected, tb.words), (tic, ta, tb)
                assert np.array_equal(a.host.ram, b.host.ram), f"tic {tic}: RAM differs"
                out["ticks"] += 1
                out["tokens"] += ta.tokens
                if on_tick:
                    on_tick(tic, ta)
            elif ea == "frame":
                (ia, pa), (ib, pb) = a.frame(), b.frame()
                assert np.array_equal(ia, ib) and np.array_equal(pa, pb), f"tic {tic}: frame"
                out["frames"] += 1
                break
            else:
                return out
    return out


# --- fast ---------------------------------------------------------------------------------------


def test_parse_buttons():
    b = R.parse_buttons("forward@10-120,fire@60-64+90-94,turnleft@130-150,run+right@5")
    assert b(0) == frozenset() and b(5) == {"run", "right"}
    assert b(10) == {"forward"} and b(62) == {"forward", "fire"} and b(95) == {"forward"}
    assert b(130) == {"left"} and b(151) == frozenset()
    assert R.parse_buttons(R.E1M1_WALK)(297) == {"forward", "use"}
    for bad in ("jump@1-2", "forward", "forward@", "forward@5-2", "forward@x", "forward@40-",
                "forward@-40", "forward@10-20+40-", "forward@1-2-3"):
        with pytest.raises(ValueError):
            R.parse_buttons(bad)


def test_build_asks_make(monkeypatch):
    """build() runs make every time and lets it decide: the Makefile's prerequisites include the
    engine / trace / qd_mem headers compiled into libqdrun (a struct changed there and a stale
    dylib would smash the stack), not just native/runner's files."""
    R.build()
    for dep in ("qd_run.c", "../engine/qd_engine.h", "../trace/qd_trace.h",
                "../trace/qd_tokens.h", "../qd_mem.h"):  # make -q: is the dylib current?
        q = subprocess.run(["make", "-q", "-C", str(R.SRC), "-W", dep], capture_output=True)
        assert q.returncode == 1, f"make does not remake libqdrun when {dep} changes"
    assert subprocess.run(["make", "-q", "-C", str(R.SRC)]).returncode == 0
    calls = []
    monkeypatch.setattr(R.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    assert R.build() == R.LIB_PATH and calls == [["make", "-s", "-C", str(R.SRC)]]


def test_guest_format():
    """I_Error's message as the host interpreter formats it, from strings and a va_list in sim
    RAM (a model-mode HALT has only the mirrored RAM to go on)."""
    with R.Runner("native") as r:
        h = r.host
        fmt, s, ap = 0x3E000, 0x3E100, 0x3E200
        h.ram[fmt:fmt + 64] = 0
        h.ram[s:s + 16] = 0
        text = b"R_TextureNumForName: %s not found (%d, %u, 0x%04x, %c%%)\0"
        h.ram[fmt:fmt + len(text)] = np.frombuffer(text, np.uint8)
        h.ram[s:s + 6] = np.frombuffer(b"BIGDR\0", np.uint8)
        args = np.array([s, -7 & 0xFFFFFFFF, 3000000000, 0xBEEF, ord("x")], "<u4")
        h.ram[ap:ap + 20] = args.view(np.uint8)
        assert R.guest_format(h, fmt, ap) == \
            "R_TextureNumForName: BIGDR not found (-7, 3000000000, 0xbeef, x%)"
        h.ram[h.symbol("qd_error_fmt"):][:4] = np.array([fmt], "<u4").view(np.uint8)
        h.ram[h.symbol("qd_error_args"):][:4] = np.array([ap], "<u4").view(np.uint8)
        assert r.error_message().startswith("R_TextureNumForName: BIGDR not found")


def test_ref_mode_equals_native():
    """Ref mode (the C reference machine's tokens through the C loop and the decoder into the
    host's RAM) against native mode: the same RAM after every tick, the same token and
    instruction counts, the same frames; the live counter and the counters add up."""
    with R.Runner("ref", log=1 << 22) as ref, R.Runner("native") as nat:
        res = lockstep(ref, nat, 110, R.parse_buttons(SHORT_WALK))
        assert res["ticks"] >= 70 and res["frames"] == 110
        assert ref.tokens == nat.tokens == res["tokens"] == ref.live_tokens
        assert ref.instructions == nat.instructions and ref.injected == nat.injected
        toks, host = ref.token_log()
        assert len(toks) == ref.tokens and host.sum() == ref.injected
        assert ref.counters()["resyncs"] == 1


def halt_at_start(r: R.Runner, text: bytes) -> None:
    """Make a ref-mode runner's resync start the sim in I_Error(text, 42) instead of sim_main
    (the reference and the decoder alike): its first tick ends in a real HALT, a SimHalt."""
    resync = r.resync

    def to_i_error():
        dt = resync()
        h = r.host
        pc, sp, gp = h.sim_entry()
        fmt = sp - 0x800  # in the stack, unused at the entry
        h.ram[fmt:fmt + len(text) + 1] = np.frombuffer(text + b"\0", np.uint8)
        regs = [0] * 32
        regs[2], regs[3], regs[10], regs[11] = sp, gp, fmt, 42
        r.ref.mem.sync_ram()
        for m in (r.ref, r.dec):
            m.set_context(pc=h.symbol("I_Error"), regs=regs)
        return dt

    r.resync = to_i_error


def test_failed_tick_counts():
    """A tick that fails is not counted in ``ticks`` (the clean ones), its error names it, and
    its tokens, injected tokens and instructions are counted; alike for every way a tick fails:
    an error from the C loop (the token limit), input words left unread, a HALT."""
    walk = R.parse_buttons(SHORT_WALK)
    with R.Runner("ref") as r:  # the token limit, after a few clean ticks
        r.run_frames(45, walk)
        n, injected, instructions = r.ticks, r.injected, r.instructions
        assert n >= 3 and r.last_tick.index == n - 1
        R.lib().qd_run_set_max_tokens(r._run, 1000)
        with pytest.raises(T.DecodeError, match=f"^tick {n}: limit: no YIELD"):
            r.run_frames(10, walk)
        t = r.last_tick
        assert r.ticks == n and t.index == n and t.tokens >= 1000 and t.injected > 0
        assert r.injected == injected + t.injected
        assert r.instructions == instructions + t.instructions
    with R.Runner("ref") as r:  # a word the sim does not read, in tick 0
        take = r.host.take_pokes
        r.host.take_pokes = lambda: np.append(take(), np.uint32(7))
        with pytest.raises(T.DecodeError, match="^tick 0: the sim left 1 of"):
            r.run_frames(45)
        t = r.last_tick
        assert r.ticks == 0 and t.index == 0 and t.injected > 0 and r.injected == t.injected
        assert r.tokens == t.tokens and r.instructions == t.instructions > 0
    with R.Runner("ref") as r:  # I_Error in tick 0
        halt_at_start(r, b"test halt %d")
        with pytest.raises(R.SimHalt, match="^tick 0: the sim exited: test halt 42$"):
            r.run_frames(45)
        assert r.ticks == 0 and r.last_tick.index == 0 and r.tokens == r.last_tick.tokens > 0


def test_token_log_after_close():
    """The token log stays readable after close (the C runner is freed by then)."""
    r = R.Runner("ref", log=1 << 20)
    r.run_frames(42)
    toks, host = r.token_log()
    assert len(toks) == r.tokens > 0 and host.sum() == r.injected
    r.close()
    toks2, host2 = r.token_log()
    assert np.array_equal(toks, toks2) and np.array_equal(host, host2) and r.tokens == len(toks)


@needs_weights
def test_model_ticks_check_mode_and_forced_steps():
    """The first ticks of E1M1 in check mode (every model token equal to the reference's), and
    again in model mode with every injected token fully stepped (forced = False): the same tokens,
    and the same recurrent state in every value head afterwards. So feeding the injected tokens
    before a record's last one as forced steps (no head, trailing layers skipped) is exact."""
    walk = R.parse_buttons("forward@40-60,fire@41-43")
    with R.Runner("check", log=1 << 20) as chk:
        chk.run_frames(43, walk)  # 5 ticks; the ticcmd's pokes: hundreds of injected tokens
        assert chk.ticks == 5 and chk.injected > 400
        toks, host = chk.token_log()
        c = chk.counters()
        assert c["compared"] == chk.tokens - chk.injected > 50000
        with R.Runner("model", engine=chk.engine, log=1 << 20, forced=False) as full:
            full.run_frames(43, walk)
            toks2, host2 = full.token_log()
            assert full.forced == 0
            assert np.array_equal(toks, toks2) and np.array_equal(host, host2)
            assert np.array_equal(full.host.ram, chk.host.ram)
            full_state = _state(full.engine)
    # forced (the default) from the same snapshot, by the engine's own calls: the hot loop's
    # sequence of steps, then the state
    with R.Runner("model", log=1 << 20) as m:
        m.run_frames(43, walk)
        toks3, _ = m.token_log()
        assert np.array_equal(toks, toks3)
        assert m.forced > 100 and m.engine.stats()["forced"] == m.forced + 1  # + PAD's
        forced_state = _state(m.engine)
    for k in full_state:
        assert np.array_equal(full_state[k], forced_state[k]), k


@needs_weights
def test_check_mode_go_after_resync():
    """After a resync the engine steps GO, and so do the decoder and the checker: the model's
    first prediction is stream token 1, where a GO is an error (a fresh decoder or checker takes
    a GO at token 0 as the start of a run). A model that predicts GO after GO fails the check."""
    with R.Runner("check") as r:
        resync = r.resync

        def resync_go():
            dt = resync()
            R.lib().qd_run_set_pred(r._run, spec.tok("GO"))
            return dt

        r.resync = resync_go
        with pytest.raises((T.DecodeError, T.CheckError), match="^tick 0: "):
            r.run_frames(45)
        assert r.resyncs == 1 and r.ticks == 0 and r.tokens == 0


def _state(engine) -> dict:
    """Every nonzero recurrent state row of every value head (rows 0..127) of the Gated DeltaNet
    layers."""
    out = {}
    for L in range(engine.info["last_needed_layer"] + 1):
        for vh in range(engine.info["value_heads"]):
            rows = np.stack([engine.get_row(L, vh, r) for r in range(engine.info["key_dim"])])
            if rows.any():
                out[(L, vh)] = rows
    return out


@needs_weights
def test_live_counter_pause_cancel():
    """A model tick runs with the GIL released: another thread sees the token counter move, a
    pause holds it between two tokens, resume lets it go on, cancel ends the tick."""
    r = R.Runner("model")
    done, err = threading.Event(), []

    def game():
        try:
            while True:
                r.step_frame()
        except R.Cancelled as e:
            err.append(e)
        finally:
            done.set()

    t = threading.Thread(target=game)
    t.start()
    try:
        deadline = time.monotonic() + 30
        while r.live_tokens < 2000 and time.monotonic() < deadline:
            time.sleep(0.01)
        a = r.live_tokens
        time.sleep(0.2)
        assert r.live_tokens > a, "the counter does not move during a tick"
        r.pause(True)
        time.sleep(0.05)
        p = r.live_tokens
        time.sleep(0.4)
        assert r.live_tokens == p, "pause does not hold the hot loop"
        r.pause(False)
        time.sleep(0.2)
        assert r.live_tokens > p
        r.cancel()
        assert done.wait(5) and err and "cancelled" in str(err[0])
    finally:
        r.cancel()
        t.join(10)
        r.close()


def _ws_session(source, check_paused_tokens: bool) -> dict:
    """Start / frames / stats / keys / pause / resume / stop over the WebSocket."""
    from aiohttp import WSMsgType
    from aiohttp.test_utils import TestClient, TestServer

    from qwendoom.server import make_app
    from qwendoom.sources import PALETTE_BYTES, PIXEL_BYTES

    import json

    async def recv(ws, want, timeout=30.0):
        deadline = time.monotonic() + timeout
        while True:
            msg = await ws.receive(timeout=max(0.01, deadline - time.monotonic()))
            assert msg.type in (WSMsgType.TEXT, WSMsgType.BINARY), msg
            m = json.loads(msg.data) if msg.type == WSMsgType.TEXT else msg.data
            if want(m):
                return m

    def stats(**kv):
        return lambda m: isinstance(m, dict) and m.get("type") == "stats" and all(
            m.get(k) == v for k, v in kv.items())

    def frame(m):
        return isinstance(m, bytes)

    out = {}

    async def main():
        client = TestClient(TestServer(make_app(source)))
        await client.start_server()
        try:
            ws = await client.ws_connect("/ws")
            await ws.send_json({"type": "start"})
            f = await recv(ws, frame)
            assert len(f) == 1 + PALETTE_BYTES + PIXEL_BYTES and f[0] == 1
            s1 = await recv(ws, lambda m: stats(state="running")(m) and m["tokens"] > 0)
            await ws.send_json({"type": "key", "action": "forward", "down": True})
            s2 = await recv(ws, lambda m: stats(state="running")(m)
                            and m["tokens"] > s1["tokens"] and m["frames"] > s1["frames"])
            assert s2["tps"] is None or s2["tps"] > 0
            out["tps"] = s2["tps"]
            await ws.send_json({"type": "key", "action": "forward", "down": False})
            await ws.send_json({"type": "pause"})
            p = await recv(ws, stats(state="paused"))
            assert p["tps"] is None
            if check_paused_tokens:  # the model holds between two tokens
                held = source._runner.live_tokens
                await asyncio.sleep(0.5)
                assert source._runner.live_tokens == held
            await ws.send_json({"type": "resume"})
            await recv(ws, stats(state="running"))
            s3 = await recv(ws, lambda m: stats(state="running")(m) and m["tokens"] > p["tokens"])
            out["tokens"] = s3["tokens"]
            await ws.send_json({"type": "stop"})
            await recv(ws, stats(state="idle"))
            out["pix"] = np.frombuffer(f, np.uint8, PIXEL_BYTES, 1 + PALETTE_BYTES)
        finally:
            await client.close()

    asyncio.run(asyncio.wait_for(main(), 120))
    return out


def test_model_source_protocol_ref_mode():
    """The model source (with the reference machine as the token source: fast) over the
    WebSocket: frames, stats with the token counter moving, keys, pause, resume, stop."""
    from qwendoom.sources import ModelSource, make_source

    assert isinstance(make_source("model", mode="ref"), ModelSource)
    source = ModelSource(mode="ref")
    out = _ws_session(source, check_paused_tokens=False)
    assert out["tokens"] > 10000
    assert source._thread is not None and not source._thread.is_alive()


@needs_weights
def test_model_source_protocol_model_mode():
    """The same with the transformer: the first frame comes after the resync and the first tick,
    the counters move meanwhile, and Pause holds the model mid-tick."""
    from qwendoom.sources import ModelSource

    source = ModelSource()
    out = _ws_session(source, check_paused_tokens=True)
    assert out["tokens"] > 10000 and len(np.unique(out["pix"])) > 1
    assert source._thread is not None and not source._thread.is_alive()


# --- slow (QD_SLOW=1) ------------------------------------------------------------------------------


@needs_weights
@pytest.mark.skipif(SLOW, reason="~15 min (QD_SLOW=1)")
def test_check_mode_e1m1_walk():
    """Check mode over E1M1_WALK (walk, turn, open the door, fire): 362 ticks, every model token
    equal to the reference machine's."""
    with R.Runner("check") as r:
        r.run_frames(400, R.parse_buttons(R.E1M1_WALK))
        c = r.counters()
        print(f"\ncheck mode, E1M1_WALK: {c}")
        assert c["ticks"] >= 300 and c["compared"] > 5_000_000


@needs_weights
@pytest.mark.skipif(SLOW, reason="~15 min (QD_SLOW=1)")
def test_model_equals_native():
    """Model mode against native mode on E1M1_WALK: the same RAM after every one of 300+ ticks
    (the decoder mirrors the model's stores into the host's RAM), the same frames."""
    with R.Runner("model") as m, R.Runner("native") as n:
        res = lockstep(m, n, 340, R.parse_buttons(R.E1M1_WALK))
        print(f"\nmodel vs native: {res}, {m.counters()}")
        assert res["ticks"] >= 300


@needs_weights
@pytest.mark.skipif(SLOW, reason="~6 min (QD_SLOW=1)")
def test_model_resync_after_load(tmp_path):
    """A resync mid-game: save (F2) after a walk, walk on, load (F3): the load reloads the level,
    so the model gets a new snapshot; model mode stays equal to native mode (RAM after every
    tick, frames) through it and after."""
    keys: dict = {}

    def press(tic, key):
        keys.setdefault(tic, []).append((key, True))
        keys.setdefault(tic + 1, []).append((key, False))

    press(110, KEY_F2)  # the save menu, slot 0, name "a", save
    press(118, H.KEY_ENTER)
    press(126, ord("a"))
    press(134, H.KEY_ENTER)
    press(190, KEY_F3)  # the load menu, slot 0
    press(198, H.KEY_ENTER)
    walk = R.parse_buttons("forward@40-100+150-185+245-290,right@60-70+160-170+250-256,"
                           "fire@170-173+260-263")
    (tmp_path / "m").mkdir()
    (tmp_path / "n").mkdir()
    with R.Runner("model", config_dir=tmp_path / "m") as m, \
            R.Runner("native", config_dir=tmp_path / "n") as n:
        resync_ticks = []
        res = lockstep(m, n, 300, walk, keys,
                       on_tick=lambda tic, t: t.resync and resync_ticks.append((tic, t.index)))
        print(f"\nsave / load in model mode: {res}, resyncs at (tic, tick) {resync_ticks}, "
              f"{m.counters()}")
        assert (tmp_path / "m" / ".savegame" / "doomsav0.dsg").exists()
        assert m.resyncs == 2 and len(resync_ticks) == 2
        assert res["ticks"] - resync_ticks[1][1] >= 50  # the model goes on after the load
        # (the load shows a screen wipe first: ~38 frames without a tick)
