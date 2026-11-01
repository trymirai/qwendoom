"""qwendoom: DOOM's game logic on a RISC-V CPU compiled into the weights of a stock Qwen3.5 model.

    python -m qwendoom serve [--port 8080] [--source model|native|mock] [--threads N] [--open]
    python -m qwendoom run [--tics N] [--mode model|check|native|ref] [--buttons SCRIPT]
                           [--gif PATH] [--png DIR] [--every K] [--scale S] [--threads N]
    python -m qwendoom check [--tics N] [--buttons SCRIPT]   the model against the reference
    python -m qwendoom bench [--tics N] [--threads N]        model-mode tokens/s on real tics
    python -m qwendoom build [--weights] [--native] [--doom] (all three without a flag)
    python -m qwendoom info [WEIGHTS]                        describe the checkpoint

The model runs DOOM's playsim (P_Ticker) token by token; the host computer (native/host) runs
the rest of DOOM and draws the frames. Tics count frames of DOOM's clock (one tic per frame); a
level start shows ~38 frames of screen wipe without a sim tick. Button scripts: actions joined
by '+', '@', tic ranges joined by '+', entries by ',': forward@10-120,fire@60-64+90-94,
turnleft@130-150 (actions: left right forward back strafeleft straferight fire use run menu enter
weapon1 weapon2 weapon3; aliases turnleft turnright backward open). docs/run.md has more.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from .runner import E1M1_WALK

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = ROOT / "weights"


def _runner(a, mode: str, log: int = 0):
    from .host import DEFAULT_WAD, HostError
    from .runner import Runner, set_qos
    wad = getattr(a, "wad", None)
    if not Path(wad or DEFAULT_WAD).is_file():
        sys.exit(f"--wad: {wad}: no such file" if wad else
                 f"no IWAD at {DEFAULT_WAD} (`make third_party` fetches Freedoom; or --wad)")
    set_qos()
    try:
        return Runner(mode, weights=a.weights, threads=a.threads, log=log,
                      **({"wad": wad} if wad else {}))
    except (FileNotFoundError, HostError) as e:
        sys.exit(str(e))


def _buttons(a):
    from .runner import parse_buttons
    try:
        return parse_buttons(a.buttons) if a.buttons else None
    except ValueError as e:
        sys.exit(f"--buttons: {e}")


def _error(e: BaseException) -> str:
    """'Type: message' on one line (an I_Error's message may end in a newline)."""
    return f"{type(e).__name__}: {' '.join(str(e).split())}"


def _tick_line(t, tic: int) -> str:
    return (f"tic {tic:4d}  tick {t.index:4d}: {t.tokens:7,} tokens ({t.injected:3} injected), "
            f"{t.instructions:7,} instructions, {t.seconds:6.3f} s, {t.tps:8,.0f} tok/s"
            + ("  [resync]" if t.resync else ""))


def _summary(r, wall: float) -> str:
    c = r.counters()
    per = c["tokens"] / c["ticks"] if c["ticks"] else 0
    return (f"{c['ticks']} sim ticks in {c['frames']} frames ({r.mode} mode): {c['tokens']:,} "
            f"tokens ({per:,.0f} per tick, {c['injected']:,} injected), {c['instructions']:,} "
            f"instructions; {c['tps']:,.0f} tok/s in ticks ({c['tick_seconds']:.1f} s), "
            f"{c['resyncs']} resyncs ({c['resync_seconds']:.2f} s), {wall:.1f} s wall")


def save_frames(frames, gif: str | None, png_dir: str | None, every: int, scale: int) -> None:
    """frames: [(idx 200x320, pal 256x3)] (already every K-th) as a GIF and / or PNGs."""
    from PIL import Image

    ims = []
    for idx, pal in frames:
        im = Image.fromarray(idx, "P")
        im.putpalette(pal.reshape(-1).tobytes())
        if scale != 1:
            im = im.resize((320 * scale, 200 * scale), Image.NEAREST)
        ims.append(im)
    if png_dir:
        Path(png_dir).mkdir(parents=True, exist_ok=True)
        for i, im in enumerate(ims):
            im.convert("RGB").save(Path(png_dir) / f"frame{i * every:05d}.png")
    if gif and ims:
        ims[0].save(gif, save_all=True, append_images=ims[1:], loop=0,
                    duration=max(20, round(1000 * every / 35)), optimize=False)


def cmd_run(a) -> None:
    from .host import HostError
    from .runner import RunError
    from .trace import CheckError, DecodeError
    buttons = _buttons(a)
    r = _runner(a, a.mode)
    frames, tic = [], [0]
    r.on_tick = lambda t: print(_tick_line(t, tic[0]), flush=True) if not a.quiet else None
    t0 = time.time()
    err = None
    try:
        for tic[0] in range(a.tics):
            if buttons is not None:
                r.hold(buttons(tic[0]))
            fr = r.step_frame()
            if fr is None:
                print(f"DOOM exited ({r.host.exit_code})")
                break
            if tic[0] % a.every == 0:
                frames.append(fr)
    # (HostError: an I_Error in the host's DOOM, e.g. a bad WAD, or in native mode's sim)
    except (RunError, CheckError, DecodeError, HostError) as e:
        err = e
    wall = time.time() - t0
    print(_summary(r, wall))
    out = a.gif or (None if a.png else "qwendoom.gif")
    if frames and (out or a.png):
        save_frames(frames, out, a.png, a.every, a.scale)
        print(f"saved {len(frames)} frames" + (f" to {out}" if out else "")
              + (f" and {a.png}/" if a.png else ""))
    r.close()
    if err is not None:
        sys.exit(_error(err))


def cmd_check(a) -> None:
    from .host import HostError
    from .runner import RunError
    from .trace import CheckError, DecodeError
    buttons = _buttons(a)
    r = _runner(a, "check")
    tic = [0]
    r.on_tick = lambda t: print(_tick_line(t, tic[0]), flush=True) if a.verbose else None
    t0 = time.time()
    err = None
    try:
        for tic[0] in range(a.tics):
            if buttons is not None:
                r.hold(buttons(tic[0]))
            if r.step_frame() is None:
                break
            if not a.verbose and r.ticks and r.ticked and r.ticks % 25 == 0:
                print(f"  {r.ticks} ticks, {r.tokens - r.injected:,} tokens compared", flush=True)
    except (RunError, CheckError, DecodeError, HostError) as e:
        err = e
    print(_summary(r, time.time() - t0))
    c, compared = r.counters(), r.tokens - r.injected  # (ticks: the clean ones)
    r.close()
    if isinstance(err, (CheckError, DecodeError)):  # a model token the reference does not make
        sys.exit(f"MISMATCH after {r.ticks} clean ticks ({compared:,} tokens compared): "
                 + _error(err))
    if isinstance(err, RunError):  # SimHalt / SimFault (the reference's sim exits too), Cancelled
        sys.exit(f"{_error(err)}; not a model mismatch: all {compared:,} model tokens equal to "
                 f"the reference ({r.ticks} clean ticks before it)")
    if err is not None:  # the host's DOOM (e.g. a bad WAD), not the sim
        sys.exit(_error(err))
    print(f"OK: {c['compared']:,} model tokens compared with the reference machine, all equal "
          f"({c['injected']:,} host-injected input tokens not compared), {c['ticks']} ticks, "
          f"{c['resyncs']} resyncs")


def cmd_bench(a) -> None:
    import numpy as np

    from . import engine as E
    from . import spec
    from .host import HostError
    from .runner import RunError
    from .trace import CheckError, DecodeError
    buttons = _buttons(a)
    r = _runner(a, "model", log=1 << 26)
    t0 = time.time()
    try:
        for tic in range(a.tics):
            if buttons is not None:
                r.hold(buttons(tic))
            if r.step_frame() is None:
                break
    except (RunError, CheckError, DecodeError, HostError) as e:
        print(_summary(r, time.time() - t0))
        r.close()
        sys.exit(_error(e))
    print(_summary(r, time.time() - t0))
    c = r.counters()
    print(f"hot loop (model mode, {r.engine.info['n_threads']} engine threads): {c['tokens']:,} "
          f"tokens in {c['tick_seconds']:.2f} s of ticks = {c['tps']:,.0f} tok/s")
    if a.no_engine or r.resyncs != 1:
        r.close()
        return
    # the engine alone on the same tokens from the same snapshot (teacher-forced in C)
    toks, host = r.token_log()
    rows, eng = r.snapshot_rows, r.engine
    seq = np.concatenate([[rows.first], toks]).astype(np.int32)
    skip = np.concatenate([[0], host]).astype(np.uint8)
    E.start(eng, rows, spec.PAD)
    t1 = time.perf_counter()
    bad = eng.check(seq, skip)
    dt = time.perf_counter() - t1
    print(f"engine alone (qd_engine_check, same tokens): {len(toks):,} tokens in {dt:.2f} s = "
          f"{len(toks) / dt:,.0f} tok/s" + ("" if bad < 0 else f"; MISMATCH at {bad}"))
    print(f"hot loop / engine alone: {c['tps'] / (len(toks) / dt):.3f}")
    r.close()


def cmd_build(a) -> None:
    every = not (a.weights or a.native or a.doom)

    def make(*args):
        print("$ make " + " ".join(args), flush=True)
        subprocess.run(["make", *args], cwd=ROOT, check=True)

    t0 = time.time()
    if a.doom or every:
        make("-C", "doom")
    if a.native or every:
        for d in ("native/host", "native/trace", "native/engine", "native/runner"):
            make("-C", d)
    if a.weights or every:
        from . import weights
        print(f"$ python -m qwendoom.weights --out {a.out}", flush=True)
        res = weights.write_checkpoint(Path(a.out))
        print(json.dumps(res, indent=1))
    print(f"built in {time.time() - t0:.0f} s")


def cmd_info(a) -> None:
    w = Path(a.weights)
    try:
        meta = json.loads((w / "qwendoom.json").read_text())
        cfg = json.loads((w / "config.json").read_text())
    except FileNotFoundError as e:
        sys.exit(f"{e.filename}: not found (python -m qwendoom build --weights)")
    tc = cfg.get("text_config", cfg)
    if a.json:
        print(json.dumps({"config": cfg, "qwendoom": meta}, indent=1))
        return
    st = w / "model.safetensors"
    size = f"{st.stat().st_size / 1e9:.2f} GB bf16" if st.exists() else "model.safetensors missing"
    print(f"{w}: {cfg.get('architectures', ['?'])[0]} ({tc.get('model_type')}), {size}")
    print(f"  {meta['machine']}")
    types = tc.get("layer_types", [])
    print(f"  hidden {tc['hidden_size']}, {tc['num_hidden_layers']} layers "
          f"({types.count('linear_attention')} Gated DeltaNet + {types.count('full_attention')} "
          f"full attention), {tc['linear_num_key_heads']} key / {tc['linear_num_value_heads']} "
          f"value heads of {tc.get('linear_key_head_dim', 128)}, intermediate "
          f"{tc['intermediate_size']}, vocabulary {tc['vocab_size']:,} (chip tokens "
          f"{meta['vocab']:,}, PAD {meta['pad']}, GO {meta['controls']['GO']})")
    nz = meta["nonzeros"]
    print(f"  nonzero weights: {nz['chip_layers']:,} in the circuit layers + {nz['embedding']:,} "
          f"embedding = {nz['total']:,}")
    c = meta["circuit"]
    print(f"  circuit: {c['layers']} layers, {c['dims']:,} IR dims, {c['heads']} heads, "
          f"{c['ffn_units']:,} MLP units (at most {c['max_units']:,} per layer)")
    per_layer: dict = {}
    for name, (L, j, nvh) in meta["heads"].items():
        per_layer.setdefault(L, []).append(name)
    print("  layers (0-based: tables / nonzero weights):")
    for L in range(tc["num_hidden_layers"]):
        names = per_layer.get(L, [])
        groups: dict = {}
        for n in names:
            base = n.rstrip("0123456789") if not n.startswith("regs") else n
            groups[base] = groups.get(base, 0) + 1
        desc = ", ".join(f"{k} x{v}" if v > 1 else k for k, v in groups.items()) or (
            "(all zero: full attention)" if types and types[L] == "full_attention" else "-")
        print(f"    L{L:<2} {desc:<52} {nz['per_layer'].get(str(L), 0):>9,}")
    am = meta["address_map"]
    print("  address map: " + ", ".join(f"{k.upper()} {v[0]:#x}+{v[1] // 1024} KB"
                                         for k, v in am.items() if isinstance(v, list))
          + f"; sim sp {am.get('sp_init', 0):#x}")
    print(f"  input: {meta['io']['input']}")
    print(f"  yield: {meta['io']['yield']}; exit: {meta['io']['exit']}")


def cmd_serve(a) -> None:
    import logging

    from . import server
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    server.serve(a)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="qwendoom", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="serve the browser UI (the model plays DOOM)")
    from . import server
    server.add_arguments(s)
    s.set_defaults(fn=cmd_serve)

    def game(p, tics: int, buttons: str = ""):
        p.add_argument("--tics", type=int, default=tics,
                       help=f"frames of DOOM's clock to run (default {tics})")
        p.add_argument("--buttons", default=buttons, metavar="SCRIPT",
                       help="input script, e.g. 'forward@10-120,fire@60-64+90-94,turnleft@130-150'"
                            + (f" (default: {buttons})" if buttons else ""))
        p.add_argument("--threads", type=int, default=None,
                       help="engine threads (default 1, or QD_THREADS)")
        p.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
        p.add_argument("--wad", default=None, help="IWAD (default Freedoom's freedoom1.wad)")

    p = sub.add_parser("run", help="play headless and save a GIF")
    game(p, 105)
    p.add_argument("--mode", choices=["model", "check", "native", "ref"], default="model")
    p.add_argument("--gif", default=None, help="animated GIF (default qwendoom.gif)")
    p.add_argument("--png", default=None, help="save the kept frames as PNGs in this folder "
                                                 "(alone: no GIF)")
    p.add_argument("--every", type=int, default=1, help="keep every K-th frame")
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--quiet", action="store_true", help="no per-tick lines")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("check", help="the model against the reference machine, token by token")
    game(p, 105)
    p.add_argument("--verbose", action="store_true", help="a line per tick")
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("bench", help="model-mode tokens per second on real tics")
    game(p, 105, E1M1_WALK)
    p.add_argument("--no-engine", action="store_true",
                   help="skip the engine-alone replay of the same tokens")
    p.set_defaults(fn=cmd_bench)

    p = sub.add_parser("build", help="build doom.elf, the native libraries, the checkpoint")
    p.add_argument("--weights", action="store_true", help="emit the checkpoint (~12 s)")
    p.add_argument("--native", action="store_true", help="make the host, trace, engine and "
                                                         "runner libraries")
    p.add_argument("--doom", action="store_true", help="make build/doom.elf")
    p.add_argument("--out", default=str(DEFAULT_WEIGHTS), help="checkpoint directory")
    p.set_defaults(fn=cmd_build)

    p = sub.add_parser("info", help="describe the checkpoint")
    p.add_argument("weights", nargs="?", default=str(DEFAULT_WEIGHTS))
    p.add_argument("--json", action="store_true", help="the raw config and qwendoom.json")
    p.set_defaults(fn=cmd_info)

    a = ap.parse_args(argv)
    if getattr(a, "every", 1) < 1 or getattr(a, "scale", 1) < 1:
        ap.error("--every and --scale must be >= 1")
    a.fn(a)


if __name__ == "__main__":
    main()
