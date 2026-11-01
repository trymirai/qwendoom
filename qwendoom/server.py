"""The web server: serves web/ and streams a game source to browsers over a WebSocket.

    python -m qwendoom.server --port 8080 [--source model|native|mock|pkg.module:Factory]
                              [--threads N] [--opt fps=35] [--open]
    python -m qwendoom serve ...            (the same options)

Protocol (ARCHITECTURE.md, "The browser UI and server"): the server sends binary frames
(0x01 + 768-byte palette + 64,000 palette indices) and JSON `stats` / `status` messages; the page
sends `start`, `stop`, `pause`, `resume` and `key` messages.

Any number of tabs can watch. One of them is the controller: the first to send a control message,
or whoever presses Start; input from the others is ignored. Each tab gets its own sender that only
ever holds the newest frame, so a slow tab drops frames instead of queueing them.
"""

from __future__ import annotations

import argparse
import asyncio
import ipaddress
import json
import logging
from pathlib import Path
from urllib.parse import urlsplit

from aiohttp import WSMsgType, web

from .sources import ACTIONS, ENDED, ERROR, IDLE, Frame, GameSource, Stats, make_source

log = logging.getLogger("qwendoom.server")

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
STATS_INTERVAL = 0.1  # seconds between stats messages (state changes go out at once)


class Client:
    """One connected tab: a coalescing outbox drained by its own sender task."""

    def __init__(self, ws: web.WebSocketResponse, cid: int):
        self.ws, self.id = ws, cid
        self.held: set[tuple[str | None, str | None]] = set()  # keys down, as (action, char)
        self._json: list[dict] = []  # status messages, in order
        self._stats: dict | None = None  # newest only
        self._frame: bytes | None = None  # newest only
        self._wake = asyncio.Event()

    def send_json(self, msg: dict) -> None:
        self._json.append(msg)
        self._wake.set()

    def send_stats(self, msg: dict) -> None:
        self._stats = msg
        self._wake.set()

    def send_frame(self, data: bytes) -> None:
        self._frame = data
        self._wake.set()

    def drop_frame(self) -> None:
        self._frame = None

    async def run(self) -> None:
        try:
            while not self.ws.closed:
                await self._wake.wait()
                self._wake.clear()
                while self._json:
                    await self.ws.send_str(json.dumps(self._json.pop(0)))
                if self._stats is not None:
                    msg, self._stats = self._stats, None
                    await self.ws.send_str(json.dumps(msg))
                if self._frame is not None:
                    data, self._frame = self._frame, None
                    await self.ws.send_bytes(data)
        except (ConnectionError, RuntimeError):
            pass  # the tab went away mid-send; the receive loop cleans up


class Hub:
    """Fans the source out to clients and routes control messages back to it."""

    def __init__(self, source: GameSource):
        self.source = source
        self.clients: dict[int, Client] = {}
        self.controller: Client | None = None
        self._next_id = 1
        self._loop: asyncio.AbstractEventLoop | None = None
        self._frame_msg: bytes | None = None
        self._stats_dirty = False
        self._last_state = source.state
        self._pump: asyncio.Task | None = None

    # -- lifecycle ----------------------------------------------------------------------------
    async def startup(self, app: web.Application | None = None) -> None:
        self._loop = asyncio.get_running_loop()
        if self.source.frame is not None:
            self._frame_msg = self.source.frame.message()
        self.source.add_listener(self._listener)
        self._pump = asyncio.create_task(self._stats_pump())

    async def shutdown(self, app: web.Application | None = None) -> None:
        self.source.remove_listener(self._listener)
        if self._pump:
            self._pump.cancel()
        for c in list(self.clients.values()):
            await c.ws.close(code=1001, message=b"server shutting down")

    # -- source events (any thread -> event loop) ---------------------------------------------
    def _listener(self, kind: str, payload) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(self._on_event, kind, payload)
        except RuntimeError:  # loop closed between the check and the call (shutdown)
            pass

    def _on_event(self, kind: str, payload) -> None:
        if kind == "frame":
            assert isinstance(payload, Frame)
            self._frame_msg = payload.message()
            for c in self.clients.values():
                c.send_frame(self._frame_msg)
        elif kind == "stats":
            assert isinstance(payload, Stats)
            if payload.state != self._last_state:  # buttons should react now, not in 100 ms
                self._last_state = payload.state
                if payload.state == IDLE:  # stopped: a dark screen, for open tabs and late joiners
                    self._frame_msg = None
                    for c in self.clients.values():
                        c.drop_frame()
                self._broadcast_stats()
                self.broadcast_status()
            else:
                self._stats_dirty = True
        elif kind == "status":
            self.broadcast_status()

    async def _stats_pump(self) -> None:
        while True:
            await asyncio.sleep(STATS_INTERVAL)
            if self._stats_dirty:
                self._broadcast_stats()

    def _broadcast_stats(self) -> None:
        self._stats_dirty = False
        msg = self.source.stats.message()
        for c in self.clients.values():
            c.send_stats(msg)

    def status_for(self, c: Client) -> dict:
        extra = self.source.status
        return {
            **{k: v for k, v in extra.items() if k not in ("type",)},
            "type": "status",
            "state": self.source.state,
            "source": self.source.name,
            "controller": self.controller is c,
            "has_controller": self.controller is not None,
            "clients": len(self.clients),
            "message": extra.get("message"),
        }

    def broadcast_status(self) -> None:
        for c in self.clients.values():
            c.send_json(self.status_for(c))

    # -- clients ------------------------------------------------------------------------------
    def connect(self, ws: web.WebSocketResponse) -> Client:
        c = Client(ws, self._next_id)
        self._next_id += 1
        self.clients[c.id] = c
        self.broadcast_status()  # everyone's client count changed; c gets its first status
        c.send_stats(self.source.stats.message())
        if self._frame_msg is not None:
            c.send_frame(self._frame_msg)
        log.info("client %d connected (%d open)", c.id, len(self.clients))
        return c

    def disconnect(self, c: Client) -> None:
        if self.clients.pop(c.id, None) is None:
            return
        if self.controller is c:
            self._release(c)
            self.controller = None
        self.broadcast_status()
        log.info("client %d disconnected (%d open)", c.id, len(self.clients))

    def _release(self, c: Client) -> None:
        """Lift every key `c` holds, so a lost tab never leaves the player running."""
        for action, char in sorted(c.held, key=str):
            self._call(self.source.key, action, False, char)
        c.held.clear()

    def _take_control(self, c: Client) -> None:
        if self.controller is c:
            return
        if self.controller is not None:
            self._release(self.controller)
        self.controller = c
        self.broadcast_status()

    def _call(self, fn, *args) -> bool:
        try:
            fn(*args)
            return True
        except Exception as e:  # a broken source must not take the server down
            log.exception("source.%s failed", getattr(fn, "__name__", fn))
            self.source.publish_status(f"{type(e).__name__}: {e}")
            return False

    def handle(self, c: Client, msg) -> None:
        """Apply one JSON message from client `c`."""
        kind = msg.get("type") if isinstance(msg, dict) else None
        if kind == "start":
            self._take_control(c)
            if self.source.state in (IDLE, ENDED, ERROR):
                self._call(self.source.start)
            return
        if kind not in ("stop", "pause", "resume", "key"):
            c.send_json({"type": "status", "error": f"unknown message type {kind!r}"})
            return
        if self.controller is None:
            self._take_control(c)
        if self.controller is not c:
            return  # a viewer: its input is ignored (its status says controller: false)
        if kind == "stop":
            self._call(self.source.stop)
        elif kind == "pause":
            self._call(self.source.pause)
        elif kind == "resume":
            self._call(self.source.resume)
        else:
            self._key(c, msg)

    def _key(self, c: Client, msg: dict) -> None:
        action, char, down = msg.get("action"), msg.get("char"), msg.get("down")
        if not isinstance(down, bool):
            c.send_json({"type": "status", "error": "key: 'down' must be a boolean"})
            return
        # a char is one printable ASCII character, checked after lowercasing ('İ' lowers to two
        # code points): DOOM takes a key as one byte, so 'τ' would become an unrelated key (F10...)
        ch = char.lower() if isinstance(char, str) else None
        if action is not None and char is None and action in ACTIONS:
            k = (action, None)
        elif action is None and ch is not None and len(ch) == 1 and "!" <= ch <= "~":
            k = (None, ch)
        else:
            c.send_json({"type": "status", "error": f"key: bad action/char in {msg!r}"})
            return
        if down == (k in c.held):
            return  # auto-repeat or a stray release: the source only sees edges
        (c.held.add if down else c.held.discard)(k)
        self._call(self.source.key, k[0], down, k[1])


HUB = web.AppKey("hub", Hub)


def _host_ok(host: str | None, bind: str | None = None) -> bool:
    """True when a Host header names this machine in a way no other site can: localhost, a name
    under .localhost or .local (mDNS), an IP literal (IPv6 bracketed), or the --host the server
    was started with (`bind`). A DNS-rebinding page (evil.example re-pointed at 127.0.0.1) is
    same-origin with itself, so it passes the Origin check, but its Host is evil.example."""
    if not host or "@" in host:
        return False
    try:
        u = urlsplit(f"//{host}")
        name, _ = u.hostname, u.port  # .port raises on a malformed port
    except ValueError:  # e.g. "[evil]" or "::1" (an IPv6 literal without brackets)
        return False
    if u.netloc != host or not name:  # "a/b", or characters urlsplit drops
        return False
    try:
        return ipaddress.ip_address(name).version == (6 if host.startswith("[") else 4)
    except ValueError:
        pass
    return (name == "localhost" or name.endswith((".localhost", ".local"))
            or name == (bind or "").lower())


def _same_origin(request: web.Request) -> bool:
    """False when a browser opened the socket from another site's page. Browsers send Origin on
    every WebSocket handshake; clients without one (CLI tools, tests) are let in."""
    origin = request.headers.get("Origin")
    if origin is None:
        return True
    try:
        return urlsplit(origin).netloc.lower() == request.host.lower()
    except ValueError:  # e.g. "http://[::1" (a malformed IPv6 host)
        return False


async def _ws_handler(request: web.Request) -> web.WebSocketResponse:
    hub = request.app[HUB]
    if not _same_origin(request):  # any page in the user's browser could drive the game
        log.warning("refused a WebSocket from origin %s", request.headers.get("Origin"))
        raise web.HTTPForbidden(text="cross-origin WebSocket refused")
    ws = web.WebSocketResponse(heartbeat=20, max_msg_size=1 << 16)
    await ws.prepare(request)
    c = hub.connect(ws)
    sender = asyncio.create_task(c.run())
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                except ValueError:
                    c.send_json({"type": "status", "error": "message is not JSON"})
                    continue
                hub.handle(c, data)
            elif msg.type == WSMsgType.ERROR:
                break
    finally:
        hub.disconnect(c)
        sender.cancel()
    return ws


async def _no_cache(request: web.Request, response: web.StreamResponse) -> None:
    # the page is tiny and changes during development; never serve a stale one
    response.headers.setdefault("Cache-Control", "no-cache")


def make_app(source: GameSource, web_dir: Path = WEB_DIR, close_source: bool = True,
             host: str | None = None) -> web.Application:
    """The aiohttp application: `/` and the files of `web_dir`, plus the WebSocket `/ws`. `host`
    is the --host the server binds; a request whose Host is neither that nor a local name or IP
    literal (`_host_ok`) gets a 403, page and socket alike."""
    web_dir = Path(web_dir).resolve()
    hub = Hub(source)

    @web.middleware
    async def check_host(request: web.Request, handler):
        if not _host_ok(request.headers.get("Host"), host):  # DNS rebinding
            log.warning("refused a request for host %s", request.headers.get("Host"))
            raise web.HTTPForbidden(text="Host refused: open the page by localhost, an IP address "
                                         "or a .local name, or start the server with --host NAME")
        return await handler(request)

    app = web.Application(middlewares=[check_host])
    app[HUB] = hub

    async def index(request: web.Request) -> web.FileResponse:
        return web.FileResponse(web_dir / "index.html")

    app.router.add_get("/ws", _ws_handler)
    app.router.add_get("/", index)
    app.router.add_static("/", web_dir, follow_symlinks=False)
    app.on_startup.append(hub.startup)
    app.on_shutdown.append(hub.shutdown)
    app.on_response_prepare.append(_no_cache)
    if close_source:
        async def _close(app: web.Application) -> None:
            await asyncio.get_running_loop().run_in_executor(None, source.close)
        app.on_cleanup.append(_close)
    return app


def _parse_opt(text: str) -> tuple[str, object]:
    key, sep, value = text.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {text!r}")
    try:
        return key, json.loads(value)  # numbers, booleans, null, quoted strings
    except ValueError:
        return key, value


def add_arguments(p: argparse.ArgumentParser) -> None:
    """The server's options (shared with the `qwendoom serve` CLI)."""
    p.add_argument("--host", default="127.0.0.1", help="interface to bind (default 127.0.0.1)")
    p.add_argument("--port", type=int, default=8080, help="port (default 8080)")
    p.add_argument("--source", default="model",
                   help="game source: model (DOOM, sim on the model: the default), native (DOOM, "
                        "sim on the host interpreter), mock (a test pattern), another registered "
                        "name, or package.module:Factory")
    p.add_argument("--threads", type=int, default=None, metavar="N",
                   help="the model's engine threads (model source; default 1, or QD_THREADS)")
    p.add_argument("--mode", default=None, choices=["model", "check", "ref", "native"],
                   help="the model source's sim mode (default model)")
    p.add_argument("--opt", action="append", type=_parse_opt, default=[], metavar="KEY=VALUE",
                   help="keyword argument for the source (repeatable), e.g. --opt fps=20")
    p.add_argument("--web", type=Path, default=WEB_DIR, help="directory with the page")
    p.add_argument("--open", action="store_true", help="open the page in the default browser")


def serve(args: argparse.Namespace) -> None:
    options = dict(args.opt)
    for key in ("threads", "mode"):  # the model source's own options
        if getattr(args, key, None) is not None:
            if args.source != "model":
                raise SystemExit(f"--{key} is an option of the model source")
            options.setdefault(key, getattr(args, key))
    source = make_source(args.source, **options)
    app = make_app(source, args.web, host=args.host)
    url = f"http://{'localhost' if args.host in ('127.0.0.1', '::1') else args.host}:{args.port}/"
    if getattr(args, "open", False):
        async def _open(app: web.Application) -> None:
            import webbrowser
            asyncio.get_running_loop().call_later(0.3, webbrowser.open, url)
        app.on_startup.append(_open)
    print(f"qwendoom: serving {url} (source: {source.name})", flush=True)
    web.run_app(app, host=args.host, port=args.port, print=None, access_log=None)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m qwendoom.server", description=__doc__.split("\n")[0])
    add_arguments(p)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    serve(args)


if __name__ == "__main__":
    main()
