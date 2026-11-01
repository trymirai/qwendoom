"""The web server and the game-source interface: static files, the WebSocket protocol
(ARCHITECTURE.md, "The browser UI and server") with the mock source, control between tabs,
input validation, the Origin and Host checks, the page's key handling (under node), and frame
delivery from a source thread."""

import asyncio
import json
import shutil
import subprocess
import threading
import time

import numpy as np
import pytest
from aiohttp import WSMsgType, WSServerHandshakeError
from aiohttp.test_utils import TestClient, TestServer

from qwendoom.server import WEB_DIR, _host_ok, _parse_opt, make_app
from qwendoom.sources import (ACTIONS, IDLE, PALETTE_BYTES, PIXEL_BYTES, RUNNING, PAUSED, GameSource,
                              HostSource, MockSource, make_source, take_edges, _mock_palette)

FRAME_BYTES = 1 + PALETTE_BYTES + PIXEL_BYTES
WEB_JS = WEB_DIR / "app.js"


class FakeSource(GameSource):
    """Records every control call; state changes are published like a real source would."""

    name = "fake"

    def __init__(self):
        super().__init__()
        self.calls = []
        self.closed = False

    def start(self):
        self.calls.append(("start",))
        self.publish_stats(state=RUNNING)

    def stop(self):
        self.calls.append(("stop",))
        self.publish_stats(state=IDLE)

    def pause(self):
        self.calls.append(("pause",))
        self.publish_stats(state=PAUSED)

    def resume(self):
        self.calls.append(("resume",))
        self.publish_stats(state=RUNNING)

    def key(self, action, down, char=None):
        self.calls.append(("key", action, down, char))

    def close(self):
        self.closed = True

    def keys(self):
        return [c[1:] for c in self.calls if c[0] == "key"]


def run(coro):
    return asyncio.run(asyncio.wait_for(coro, 30))


async def started(source):
    client = TestClient(TestServer(make_app(source)))
    await client.start_server()
    return client


async def recv_json(ws, pred=lambda m: True, timeout=3.0):
    """The next JSON message matching `pred` (binary frames and other messages are skipped)."""
    deadline = time.monotonic() + timeout
    while True:
        msg = await ws.receive(timeout=max(0.01, deadline - time.monotonic()))
        assert msg.type in (WSMsgType.TEXT, WSMsgType.BINARY), msg
        if msg.type == WSMsgType.TEXT:
            m = json.loads(msg.data)
            if pred(m):
                return m


async def recv_frame(ws, timeout=3.0):
    deadline = time.monotonic() + timeout
    while True:
        msg = await ws.receive(timeout=max(0.01, deadline - time.monotonic()))
        assert msg.type in (WSMsgType.TEXT, WSMsgType.BINARY), msg
        if msg.type == WSMsgType.BINARY:
            return msg.data


async def settle(ws, quiet=0.15):
    """Drain everything the server sends until it has been quiet for `quiet` seconds."""
    out = []
    while True:
        try:
            msg = await ws.receive(timeout=quiet)
        except asyncio.TimeoutError:
            return out
        out.append(json.loads(msg.data) if msg.type == WSMsgType.TEXT else msg.data)


def is_status(**want):
    return lambda m: m.get("type") == "status" and all(m.get(k) == v for k, v in want.items())


def is_stats(**want):
    return lambda m: m.get("type") == "stats" and all(m.get(k) == v for k, v in want.items())


# -- static files ----------------------------------------------------------------------------------

def test_serves_the_page():
    async def main():
        client = await started(FakeSource())
        try:
            r = await client.get("/")
            assert r.status == 200 and "text/html" in r.headers["Content-Type"]
            html = await r.text()
            assert "qwen<span>doom</span>" in html and 'src="app.js"' in html
            for path, kind in (("/app.js", "javascript"), ("/style.css", "text/css"),
                               ("/favicon.svg", "image/svg+xml")):
                r = await client.get(path)
                assert r.status == 200 and kind in r.headers["Content-Type"], path
                assert r.headers.get("Cache-Control") == "no-cache"
            assert (await client.get("/missing.js")).status == 404
            assert (await client.get("/../pyproject.toml")).status == 404
        finally:
            await client.close()
    run(main())


# -- the protocol with the mock --------------------------------------------------------------------

def test_mock_round_trip():
    source = MockSource(fps=60, tps=1000.0, startup=0.05)

    async def main():
        client = await started(source)
        try:
            ws = await client.ws_connect("/ws")
            first = await recv_json(ws, is_status())
            assert first["source"] == "mock" and first["state"] == "idle"
            assert first["controller"] is False and first["clients"] == 1
            assert (await recv_json(ws, is_stats())) == {
                "type": "stats", "tokens": 0, "tps": None, "frames": 0, "state": "idle"}

            await ws.send_json({"type": "start"})
            await recv_json(ws, is_status(controller=True))
            await recv_json(ws, is_stats(state="starting"))
            await recv_json(ws, is_stats(state="running"))

            data = await recv_frame(ws)
            assert len(data) == FRAME_BYTES and data[0] == 1
            pal = np.frombuffer(data, np.uint8, PALETTE_BYTES, 1).reshape(256, 3)
            pix = np.frombuffer(data, np.uint8, PIXEL_BYTES, 1 + PALETTE_BYTES).reshape(200, 320)
            assert np.array_equal(pal, _mock_palette())
            assert np.array_equal(pix[0], np.arange(320) * 256 // 320)  # the palette strip

            # counters move while running
            s1 = await recv_json(ws, is_stats(state="running"))
            s2 = await recv_json(ws, lambda m: m.get("type") == "stats" and m["frames"] > s1["frames"])
            assert s2["tokens"] > s1["tokens"] >= 0 and s2["tps"] > 0

            # a held key reaches the game thread and lights its status-bar indicator
            await ws.send_json({"type": "key", "action": "fire", "down": True})
            for _ in range(20):
                f = np.frombuffer(await recv_frame(ws), np.uint8, PIXEL_BYTES, 1 + PALETTE_BYTES)
                if f.reshape(200, 320)[176 + 1, 6 + 6 * 22 + 1] != f.reshape(200, 320)[176 + 1, 6 + 1]:
                    break
            else:
                pytest.fail("the fire indicator never lit")
            assert "fire" in source._held
            await ws.send_json({"type": "key", "action": "fire", "down": False})
            await ws.send_json({"type": "key", "char": "Y", "down": True})
            await ws.send_json({"type": "key", "char": "Y", "down": False})

            await ws.send_json({"type": "pause"})
            paused = await recv_json(ws, is_stats(state="paused"))
            assert paused["tps"] == 0.0
            await settle(ws)
            assert not [m for m in await settle(ws, 0.3) if isinstance(m, bytes)], "frames while paused"
            assert "fire" not in source._held

            await ws.send_json({"type": "resume"})
            await recv_json(ws, is_stats(state="running"))
            assert len(await recv_frame(ws)) == FRAME_BYTES
            await ws.close()
        finally:
            await client.close()
    run(main())
    assert not source._thread.is_alive(), "the app's cleanup closes the source"


class SlowMock(MockSource):
    """A game thread that spends most of each frame computing, so a control message usually
    lands between its reading the state and publishing (the window for stale output)."""

    def _render(self, held):
        time.sleep(0.004)
        return super()._render(held)


def test_mock_stop_goes_dark():
    source = SlowMock(fps=120, tps=1000.0, startup=0.0)

    async def main():
        client = await started(source)
        try:
            ws = await client.ws_connect("/ws")
            await ws.send_json({"type": "start"})
            await recv_json(ws, is_stats(state="running"))
            await recv_frame(ws)
            await ws.send_json({"type": "key", "action": "forward", "down": True})
            await asyncio.sleep(0.006)  # into the next frame's computation
            await ws.send_json({"type": "stop"})
            # nothing stale after the stop: no frame, no "running" (the game thread races it)
            while True:
                m = await ws.receive(timeout=3)
                if m.type == WSMsgType.TEXT and json.loads(m.data).get("state") == "idle":
                    break
            after = await settle(ws, 0.4)
            assert not [m for m in after if isinstance(m, bytes)], "frames after stop"
            assert all(m.get("state") == "idle" for m in after if "state" in m), after
            st = source.stats
            assert (st.state, st.tokens, st.frames, st.tps) == (IDLE, 0, 0, None)
            assert source.frame is None and not source._held

            late = await client.ws_connect("/ws")  # a tab opened now sees a dark screen
            first = await settle(late)
            assert not [m for m in first if isinstance(m, bytes)]
            assert any(isinstance(m, dict) and m.get("state") == "idle" for m in first)

            await ws.send_json({"type": "start"})  # a new game, counters from zero
            s = await recv_json(ws, is_stats(state="running"))
            assert s["frames"] <= 10
            assert len(await recv_frame(late)) == FRAME_BYTES
        finally:
            await client.close()
    run(main())


def test_pause_is_not_undone_by_the_game_thread():
    """Pausing many times while frames fly: the paused state always sticks."""
    source = SlowMock(fps=120, tps=1000.0, startup=0.0)

    async def main():
        client = await started(source)
        try:
            ws = await client.ws_connect("/ws")
            await ws.send_json({"type": "start"})
            await recv_json(ws, is_stats(state="running"))
            for i in range(15):
                await recv_frame(ws)
                await asyncio.sleep(i % 5 * 0.0015)  # pause at different points of the frame
                await ws.send_json({"type": "pause"})
                await recv_json(ws, is_stats(state="paused"))
                got = await settle(ws, 0.12)
                assert not [m for m in got if isinstance(m, bytes)], "a frame after pause"
                assert all(m.get("state") == "paused" for m in got if "state" in m), got
                await ws.send_json({"type": "resume"})
                await recv_json(ws, is_stats(state="running"))
        finally:
            await client.close()
    run(main())


# -- control between tabs --------------------------------------------------------------------------

def test_one_controller_and_takeover():
    source = FakeSource()

    async def main():
        client = await started(source)
        try:
            a = await client.ws_connect("/ws")
            await recv_json(a, is_status(clients=1))
            b = await client.ws_connect("/ws")
            await recv_json(b, is_status(clients=2))
            await recv_json(a, is_status(clients=2))

            await a.send_json({"type": "start"})
            await recv_json(a, is_status(controller=True))
            await recv_json(b, is_status(controller=False, has_controller=True))
            await a.send_json({"type": "key", "action": "forward", "down": True})
            await a.send_json({"type": "key", "action": "forward", "down": True})  # auto-repeat
            await b.send_json({"type": "key", "action": "fire", "down": True})  # a viewer
            await b.send_json({"type": "pause"})
            await b.send_json({"type": "stop"})
            await settle(a), await settle(b)
            assert source.calls == [("start",), ("key", "forward", True, None)]

            # Start from a viewer takes control of the running game (no restart) and lifts the
            # keys the old controller held
            await b.send_json({"type": "start"})
            await recv_json(b, is_status(controller=True))
            await recv_json(a, is_status(controller=False))
            assert source.calls[1:] == [("key", "forward", True, None), ("key", "forward", False, None)]
            await a.send_json({"type": "key", "action": "forward", "down": False})  # ignored now
            await b.send_json({"type": "key", "action": "use", "down": True})
            await b.send_json({"type": "pause"})
            await recv_json(b, is_stats(state="paused"))
            await recv_json(a, is_stats(state="paused"))

            # the controller leaves holding a key: it is released and control is free again
            await b.close()
            await recv_json(a, is_status(clients=1, has_controller=False))
            assert source.keys()[-2:] == [("use", True, None), ("use", False, None)]
            await a.send_json({"type": "resume"})  # the first control message takes control
            await recv_json(a, is_status(controller=True))
            await recv_json(a, is_stats(state="running"))
            assert source.calls[-1] == ("resume",)
            assert sum(c == ("start",) for c in source.calls) == 1
        finally:
            await client.close()
    run(main())
    assert source.closed


def test_rejects_bad_messages():
    source = FakeSource()

    async def main():
        client = await started(source)
        try:
            ws = await client.ws_connect("/ws")
            await settle(ws)
            bad = [
                "not json",
                json.dumps(["start"]),
                json.dumps({"type": "launch"}),
                json.dumps({"type": "key", "action": "jump", "down": True}),
                json.dumps({"type": "key", "action": "fire", "down": 1}),
                json.dumps({"type": "key", "action": "fire", "char": "x", "down": True}),
                json.dumps({"type": "key", "char": "xy", "down": True}),
                json.dumps({"type": "key", "char": "\n", "down": True}),
                json.dumps({"type": "key", "down": True}),
                # only printable ASCII: DOOM keeps one byte, 'τ' (U+03C4) would be F10 (0xc4);
                # 'İ' lowers to two code points and must not reach the game thread's ord()
                json.dumps({"type": "key", "char": "τ", "down": True}),
                json.dumps({"type": "key", "char": "İ", "down": True}),
                json.dumps({"type": "key", "char": " ", "down": True}),
                json.dumps({"type": "key", "char": "\x7f", "down": True}),
                json.dumps({"type": "key", "char": " ", "down": True}),
            ]
            for text in bad:
                await ws.send_str(text)
                err = await recv_json(ws, lambda m: "error" in m)
                assert err["type"] == "status" and err["error"], text
            assert source.keys() == []
            for action in ACTIONS:  # every documented action is accepted
                await ws.send_json({"type": "key", "action": action, "down": True})
                await ws.send_json({"type": "key", "action": action, "down": False})
            await settle(ws)
            assert source.keys() == [(a, d, None) for a in ACTIONS for d in (True, False)]
            del source.calls[:]
            for char in ("Y", "!", "~", "0"):  # the ends of the range; letters lowercased
                await ws.send_json({"type": "key", "char": char, "down": True})
                await ws.send_json({"type": "key", "char": char, "down": False})
            await settle(ws)
            assert source.keys() == [(None, d, c) for c in "y!~0" for d in (True, False)]
        finally:
            await client.close()
    run(main())


def test_websocket_checks_origin():
    """Only the page itself (or a client that sends no Origin) gets the socket: another site open
    in the same browser must not be able to start, stop or steer the game."""
    source = FakeSource()

    async def main():
        client = await started(source)
        here = f"{client.host}:{client.port}"
        try:
            for origin in ("http://evil.example", f"http://{client.host}:{client.port + 1}",
                           "null", "http://[::1"):
                with pytest.raises(WSServerHandshakeError) as e:
                    await client.ws_connect("/ws", headers={"Origin": origin})
                assert e.value.status == 403, origin
            same = await client.ws_connect("/ws", headers={"Origin": f"http://{here}"})
            await recv_json(same, is_status(clients=1))
            # --host 0.0.0.0, the page opened by LAN IP; host names compare case-insensitively
            lan = await client.ws_connect("/ws", headers={"Host": f"192.168.1.5:{client.port}",
                                                          "Origin": f"http://192.168.1.5:{client.port}"})
            await recv_json(lan, is_status(clients=2))
            mixed = await client.ws_connect("/ws", headers={"Host": f"LocalHost:{client.port}",
                                                            "Origin": f"http://localhost:{client.port}"})
            await recv_json(mixed, is_status(clients=3))
            bare = await client.ws_connect("/ws")  # no Origin: CLI tools, these tests
            await recv_json(bare, is_status(clients=4))
            await bare.send_json({"type": "start"})
            await recv_json(bare, is_status(controller=True))
            assert source.calls == [("start",)]
        finally:
            await client.close()
    run(main())


def test_host_names():
    ok = [("localhost", None), ("LocalHost:8080", None), ("127.0.0.1:8080", None), ("[::1]:8080", None),
          ("[::1]", None), ("x.local", None), ("Mac-Mini.local:8080", None), ("game.localhost:80", None),
          ("192.168.1.5:8080", None), ("Doom-Box.lan:8080", "doom-box.LAN"), ("10.0.0.7", "0.0.0.0")]
    bad = [(None, None), ("", None), ("evil.example:8080", None), ("localhost.evil.example", None),
           ("127.0.0.1.nip.io", None), ("xlocal", None), ("doom-box.lan", None),
           ("doom-box.lan.evil.example", "doom-box.lan"), ("evil.example", ""),
           ("::1", None),  # IPv6 needs brackets
           ("[1.2.3.4]", None), ("[evil]", None), ("evil@127.0.0.1", None), ("localhost/x", None),
           ("localhost#x", None), ("localhost:abc", None), ("localhost:99999", None)]
    for host, bind in ok:
        assert _host_ok(host, bind), (host, bind)
    for host, bind in bad:
        assert not _host_ok(host, bind), (host, bind)


def test_host_check_stops_dns_rebinding():
    """A page on evil.example whose name now resolves to 127.0.0.1 is same-origin with itself
    (matching Origin and Host), so only the Host check keeps it from the page and the socket. LAN
    names and the --host the server was started with still get in."""
    source = FakeSource()

    async def main():
        client = TestClient(TestServer(make_app(source, host="Doom-Box.lan")))
        await client.start_server()
        port = client.port
        try:
            for name in ("evil.example", "127.0.0.1.nip.io", "doom-box.lan.evil.example"):
                host = f"{name}:{port}"
                for path in ("/", "/app.js"):
                    r = await client.get(path, headers={"Host": host})
                    assert r.status == 403, (path, host)
                    assert "evil" not in await r.text()  # the Host is not echoed back
                with pytest.raises(WSServerHandshakeError) as e:
                    await client.ws_connect("/ws", headers={"Host": host, "Origin": f"http://{host}"})
                assert e.value.status == 403, host
            reader, writer = await asyncio.open_connection(client.host, port)
            writer.write(b"GET / HTTP/1.0\r\n\r\n")  # no Host at all (browsers always send one)
            assert (await reader.readline()).split()[1] == b"403"
            writer.close()
            n = 0
            for name in ("localhost", "127.0.0.1", "[::1]", "x.local", "game.localhost",
                         "192.168.1.5", "doom-box.LAN"):
                host = f"{name}:{port}"
                r = await client.get("/", headers={"Host": host})
                assert r.status == 200 and "qwen<span>doom</span>" in await r.text(), host
                ws = await client.ws_connect("/ws", headers={"Host": host, "Origin": f"http://{host}"})
                n += 1
                await recv_json(ws, is_status(clients=n))
            assert source.calls == []
        finally:
            await client.close()
    run(main())


# -- the page's keys (app.js under node with a minimal DOM) -----------------------------------------

PAGE_HARNESS = r"""
const vm = require('vm'), fs = require('fs');
const [path, events] = JSON.parse(fs.readFileSync(0, 'utf8'));
const els = {}, on = {}, sent = [];
const el = () => ({
  dataset: {}, style: { setProperty() {} }, textContent: '',
  classList: { add() {}, remove() {}, toggle() {} },
  addEventListener() {}, focus() {}, replaceChildren() {},
  getContext: () => ({ createImageData: (w, h) => ({ data: new Uint8ClampedArray(w * h * 4) }),
                       fillRect() {}, putImageData() {} }),
});
class WebSocket { constructor() { WebSocket.last = this; this.readyState = 1; }
                  send(s) { sent.push(JSON.parse(s)); } }
WebSocket.OPEN = 1;
const document = { getElementById: (id) => els[id] ??= el(), querySelectorAll: () => [],
                   addEventListener() {}, createElement: el, body: {}, documentElement: {} };
const window = { addEventListener: (type, fn) => { on[type] = fn; } };
vm.runInNewContext(fs.readFileSync(path, 'utf8'), {
  document, window, WebSocket, location: { protocol: 'http:', host: 'localhost' }, console,
  requestAnimationFrame() {}, setInterval() {}, setTimeout() {}, performance: { now: () => 0 },
});
WebSocket.last.onopen();
document.activeElement = els.screen;  // the game screen has the keyboard
for (const [type, key, code] of events) {
  on[type]({ type, key, code, repeat: false, metaKey: false, altKey: false, ctrlKey: false,
             preventDefault() {} });
}
process.stdout.write(JSON.stringify(sent));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_page_keys():
    """What the page sends for a key: printable ASCII chars, the physical key for a letter or
    digit typed on another layout, nothing for the rest, and F as a second fire key."""
    presses = [  # (key, code) -> what goes out, on down and on up
        ("Y", "KeyY", {"char": "y"}),
        ("~", "Backquote", {"char": "~"}),
        ("τ", "KeyT", {"char": "t"}),  # Greek
        ("н", "KeyY", {"char": "y"}),  # Russian: y/n prompts still answer
        ("İ", "KeyI", {"char": "i"}),  # Turkish
        ("é", "Digit0", {"char": "0"}),  # Czech
        ("f", "KeyF", {"action": "fire"}),  # never also sent as the char 'f'
        ("а", "KeyF", {"action": "fire"}),  # Russian, the same physical key
        ("ü", "BracketLeft", None),  # no ASCII and no letter key: nothing
        ("Dead", "Quote", None),
        ("F10", "F10", None),
    ]
    events = [[t, key, code] for key, code, _ in presses for t in ("keydown", "keyup")]
    r = subprocess.run(["node", "-e", PAGE_HARNESS], input=json.dumps([str(WEB_JS), events]),
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    want = [{"type": "key", **what, "down": d} for _, _, what in presses if what for d in (True, False)]
    assert json.loads(r.stdout) == want


# -- frames from a source thread -------------------------------------------------------------------

def test_frames_from_a_thread_newest_wins():
    source = FakeSource()
    pal = bytes(range(256)) * 3

    async def main():
        client = await started(source)
        try:
            ws = await client.ws_connect("/ws")
            await settle(ws)

            def produce():  # a game thread publishing as fast as it can
                for i in range(200):
                    source.publish_frame(pal, bytes([i]) * PIXEL_BYTES)
                    source.publish_stats(tokens=i * 1000, tps=12.5, frames=i + 1)

            t = threading.Thread(target=produce)
            t.start()
            await asyncio.get_running_loop().run_in_executor(None, t.join)
            got = [m for m in await settle(ws, 0.4)]
            frames = [m for m in got if isinstance(m, bytes)]
            stats = [m for m in got if isinstance(m, dict) and m["type"] == "stats"]
            assert frames and all(len(f) == FRAME_BYTES and f[0] == 1 for f in frames)
            assert frames[-1][1 + PALETTE_BYTES] == 199, "the newest frame always arrives"
            assert [f[1 + PALETTE_BYTES] for f in frames] == sorted(f[1 + PALETTE_BYTES] for f in frames)
            assert stats[-1] == {"type": "stats", "tokens": 199000, "tps": 12.5, "frames": 200,
                                 "state": "idle"}
            assert len(stats) < 50, "stats are coalesced"

            # a tab that connects later gets the current picture and counters at once
            late = await client.ws_connect("/ws")
            first = await settle(late)
            assert any(isinstance(m, bytes) and m[1 + PALETTE_BYTES] == 199 for m in first)
            assert any(isinstance(m, dict) and m.get("tokens") == 199000 for m in first)

            source.publish_status("loading E1M1", level="E1M1")
            st = await recv_json(ws, is_status(message="loading E1M1"))
            assert st["level"] == "E1M1" and st["source"] == "fake"
        finally:
            await client.close()
    run(main())


# -- the source interface --------------------------------------------------------------------------

def test_source_interface():
    s = FakeSource()
    with pytest.raises(ValueError):
        s.publish_frame(b"\0" * 767, b"\0" * PIXEL_BYTES)
    with pytest.raises(ValueError):
        s.publish_frame(b"\0" * PALETTE_BYTES, b"\0" * 63999)
    with pytest.raises(ValueError):
        s.publish_stats(state="exploded")
    f = s.publish_frame(np.zeros((256, 3), np.uint8), np.ones((200, 320), np.uint8))
    assert f.message()[:1] == b"\x01" and len(f.message()) == FRAME_BYTES and f.seq == 1
    st = s.publish_stats(tokens=5)
    assert (st.tokens, st.tps, st.frames, st.state) == (5, None, 0, "idle")
    assert s.publish_stats(tps=1.5).tokens == 5  # unspecified fields are kept

    m = make_source("mock", fps=5)
    try:
        assert isinstance(m, MockSource) and m.fps == 5.0
    finally:
        m.close()
    t = make_source("qwendoom.sources:MockSource", fps=7)  # a package.module:Factory path
    try:
        assert isinstance(t, MockSource) and t.fps == 7.0
    finally:
        t.close()
    with pytest.raises(ValueError):
        make_source("nope")
    assert _parse_opt("fps=20") == ("fps", 20) and _parse_opt("mode=model") == ("mode", "model")


# -- real DOOM: the host interpreter with the native sim (needs the other tracks' builds) -----------

def _host_ready():
    try:
        from qwendoom import host
    except Exception:
        return False
    return all(p.exists() for p in (host.LIB, host.DEFAULT_ELF, host.DEFAULT_WAD))


def test_short_taps_last_a_frame():
    q = [("forward", True, None), ("fire", True, None), ("forward", False, None),
         ("fire", False, None), ("right", True, None)]
    assert take_edges(q) == [("forward", True, None), ("fire", True, None)]
    assert take_edges(q) == [("forward", False, None), ("fire", False, None), ("right", True, None)]
    assert take_edges(q) == [] and q == []
    q = [("left", False, None), (None, True, "y"), (None, False, "y"), (None, True, "n")]
    assert take_edges(q) == [("left", False, None), (None, True, "y")]  # an earlier press's release
    assert take_edges(q) == [(None, False, "y"), (None, True, "n")]


@pytest.mark.skipif(not _host_ready(), reason="needs libqdhost.dylib, doom.elf and Freedoom")
def test_native_source_plays_doom():
    source = HostSource()

    async def main():
        client = await started(source)
        try:
            ws = await client.ws_connect("/ws")
            await ws.send_json({"type": "start"})
            await recv_json(ws, is_stats(state="running"), timeout=10)
            frames = [await recv_frame(ws, timeout=10) for _ in range(40)]
            pix = np.frombuffer(frames[-1], np.uint8, PIXEL_BYTES, 1 + PALETTE_BYTES)
            assert len(np.unique(pix)) > 20, "a DOOM screen, not a blank one"
            s = await recv_json(ws, lambda m: m.get("type") == "stats" and m["frames"] >= 40)
            assert s["tokens"] > 0 and s["tps"] is None and s["state"] == "running"

            # walking forward changes the view
            await ws.send_json({"type": "key", "action": "forward", "down": True})
            for _ in range(20):
                moved = await recv_frame(ws)
            await ws.send_json({"type": "key", "action": "forward", "down": False})
            a = np.frombuffer(moved, np.uint8, PIXEL_BYTES, 1 + PALETTE_BYTES).reshape(200, 320)
            b = pix.reshape(200, 320)
            assert (a[:168] != b[:168]).mean() > 0.05, "the 3D view did not change"

            await ws.send_json({"type": "pause"})
            await recv_json(ws, is_stats(state="paused"))
            await ws.send_json({"type": "stop"})
            await recv_json(ws, is_stats(state="idle"))
            assert not [m for m in await settle(ws, 0.3) if isinstance(m, bytes)]
            await ws.send_json({"type": "start"})  # a fresh game
            await recv_json(ws, is_stats(state="running"), timeout=10)
            await recv_frame(ws, timeout=10)
        finally:
            await client.close()
    run(main())
    assert source._thread is not None and not source._thread.is_alive()
