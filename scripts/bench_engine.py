"""Benchmark the native engine (native/engine, build/libqdengine.dylib):

    .venv/bin/python scripts/bench_engine.py [--tiny] [--weights] [--doom] [--ref]
                                             [--tokens 100000] [--threads 1,2,3,4,6]
                                             [--repeat 3] [--json build/engine-bench/x.json]
    .venv/bin/python scripts/bench_engine.py --doom-capture        # (once) the DOOM ticks
    .venv/bin/python scripts/bench_engine.py --ab A.dylib,B.dylib  # interleaved A/B, DOOM ticks

--tiny (the default when no set is named): rv-tiny (build/rv-tiny, upstream chipformer.rv),
free-running greedy decoding from GO over the first --tokens tokens of an upstream ELF's trace
(c-qsort: 240k tokens), checked against the reference trace. --weights: the qwendoom checkpoint
(weights/) on a C test program, teacher-forced (every prediction must equal the trace). --doom:
weights/ on real DOOM ticks (below), teacher-forced, every prediction checked except the
host-injected input records. Each set runs for every thread count, then a per-stage profile of
the single-thread and the best configuration. --ref: the numpy RefEngine's speed on rv-tiny.

The DOOM ticks (--doom-capture, build/engine-bench/doom/): Freedoom E1M1 played by doom/measure.py's
bot in native sim mode, with the C reference machine (qwendoom.trace.Ref, its own RAM, fed only
the snapshot and each tick's pokes, as the model is) running every tick alongside; it must match
the native tick's instruction and token counts and RAM. For each window of ticks (--windows
name:first_tick:ticks) it saves the snapshot at the window's start (the reference's RAM, DROM,
ROM, registers and position, and their state rows) and the reference's tokens of the window's
ticks: toks = [GO] + the ticks' tokens, host = the injected input records' positions.

The calling thread asks for the user-interactive QoS class (macOS then prefers P-cores), as the
engine's worker threads do. Timings on a loaded machine vary; the configurations take turns,
--repeat rounds, and each reports its best and median.
"""

import argparse
import ctypes
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qwendoom import engine as E  # noqa: E402

RV_TINY = ROOT / "build" / "rv-tiny"
RV_DATA = ROOT / "third_party" / "qwenputer" / "experimental" / "riscv" / "tests" / "data" / "rv"
WEIGHTS = ROOT / "weights"
DOOM_DIR = ROOT / "build" / "engine-bench" / "doom"
# the level start (nothing awake: ~15k tokens a tick) and a fight (18 monsters awake: ~37k)
DOOM_WINDOWS = "start:0:16,fight:1000:10"


def interactive_qos():
    try:
        ctypes.CDLL("/usr/lib/libSystem.B.dylib").pthread_set_qos_class_self_np(0x21, 0)
    except OSError:
        pass


def rv_setup(prog_name, n):
    """(start(engine), the trace's first n + 1 tokens) for an upstream ELF on rv-tiny."""
    from chipformer.rv import core, elf, ref
    from chipformer.rv import spec as S

    prog = elf.program(RV_DATA / f"{prog_name}.elf")
    want = ref.Machine.from_program(prog).run(n + 1)
    c = core.build()
    rows = E.rv_rows(c, ref.Machine.from_program(prog))
    return (lambda e: E.start(e, rows, S.VOCAB)), want


def qd_setup(prog_name, n):
    from qwendoom import core_check, snapshot
    from qwendoom import spec as S

    m = core_check.elf_machine(prog_name, core_check.elf_inputs(prog_name))
    want = m.run(n + 1)
    skip = np.zeros(len(want), np.uint8)
    skip[m.injected] = 1
    rows = snapshot.machine_rows(core_check.elf_machine(prog_name,
                                                        core_check.elf_inputs(prog_name)))
    return (lambda e: E.start(e, rows, S.PAD)), want, skip


# --- the DOOM ticks ---------------------------------------------------------------------------


def doom_capture(windows: str, out_dir: Path) -> None:
    """Play E1M1 with the bot and save the windows' snapshots and reference streams."""
    sys.path.insert(0, str(ROOT / "doom"))
    import measure as M

    from qwendoom import host as H
    from qwendoom import snapshot
    from qwendoom import trace as T

    wins = [(n, int(a), int(k)) for n, a, k in (w.split(":") for w in windows.split(","))]
    heads = json.loads((WEIGHTS / "qwendoom.json").read_text())["heads"]
    out_dir.mkdir(parents=True, exist_ok=True)
    h = H.Host()
    bot = M.Bot(h)
    view = M.SimView(h)
    mem, ref = T.Memory(T.Map.qwendoom()), None
    end = max(a + k for _, a, k in wins)
    tick, cur = 0, None
    while tick < end:
        ev = h.run()
        if ev == H.FRAME:
            if tick:
                bot.step()
            continue
        if ev != H.SIM_TICK:
            raise SystemExit(f"host event {ev}: {h.error}")
        if h.need_resync():
            mem.ram[:], mem.drom[:], mem.rom[:] = h.ram, h.drom, h.rom
            pc, sp, gp = h.sim_entry()
            ref = T.Ref(mem)
            ref.set_context(pc=pc, regs=[0, 0, sp, gp] + [0] * 28)
            h.resync()
        start = next((w for w in wins if w[1] == tick), None)
        if start:
            anchor, n, regs = ref.context
            pc = ref.pc
            rows = snapshot.state_rows(bytes(mem.ram), bytes(mem.drom), bytes(mem.rom), regs, pc,
                                       heads=heads, n=n)
            cur = {"name": start[0], "first_tick": tick, "ticks": start[2], "rows": rows,
                   "state": dict(ram=mem.ram.copy(), drom=mem.drom.copy(), rom=mem.rom.copy(),
                                 regs=np.asarray(regs, np.uint32), pc=pc, n=n),
                   "toks": [np.asarray([rows.first], np.int32)], "host": [np.zeros(1, np.uint8)],
                   "tick_tokens": [], "tick_instructions": [], "awake": []}
        pokes = h.take_pokes()
        ref.push_input(pokes)
        s0 = ref.counts["steps"]
        if cur:
            toks, flags, why = ref.run(stop_yield=True)
            ntok = len(toks)
        else:
            ntok, why = ref.run_count(stop_yield=True)
        st = h.sim_tick_native()
        if why != "yield" or (ref.counts["steps"] - s0, ntok) != (st.instructions, st.tokens):
            raise SystemExit(f"tick {tick}: the reference ({why}, {ref.counts['steps'] - s0} "
                             f"instructions, {ntok} tokens) differs from the native sim "
                             f"({st.instructions}, {st.tokens})")
        h.complete_sim_tick()
        if not np.array_equal(mem.ram, h.ram):
            raise SystemExit(f"tick {tick}: the reference's RAM differs from the host's")
        if cur:
            cur["toks"].append(toks)
            cur["host"].append((flags & T.TF_HOST != 0).astype(np.uint8))
            cur["tick_tokens"].append(ntok)
            cur["tick_instructions"].append(st.instructions)
            cur["awake"].append(sum(m["awake"] for m in view.monsters()))
        tick += 1
        if cur and tick == cur["first_tick"] + cur["ticks"]:
            save_window(out_dir, cur)
            cur = None


def save_window(out_dir: Path, w: dict) -> None:
    toks, host, rows = np.concatenate(w["toks"]), np.concatenate(w["host"]), w["rows"]
    np.savez_compressed(out_dir / f"{w['name']}.npz", toks=toks, host=host,
                        tick_tokens=np.asarray(w["tick_tokens"], np.int64),
                        layer=rows.layer, vhead=rows.vhead, row=rows.row, vec=rows.vec,
                        first=rows.first, **w["state"])
    info = {k: w[k] for k in ("name", "first_tick", "ticks", "tick_tokens", "tick_instructions",
                              "awake")}
    info.update(tokens=len(toks) - 1, injected=int(host.sum()), rows=len(rows),
                pc=int(w["state"]["pc"]), n=int(w["state"]["n"]))
    (out_dir / f"{w['name']}.json").write_text(json.dumps(info, indent=1))
    print(f"{w['name']}: ticks {w['first_tick']}..{w['first_tick'] + w['ticks'] - 1}, "
          f"{len(toks) - 1} tokens ({int(host.sum())} injected), {len(rows)} state rows",
          flush=True)


def doom_windows(names=None, max_tokens=0, directory=DOOM_DIR):
    """[(name, start(engine), toks, skip)] of the captured windows (the first max_tokens + 1
    tokens of each when max_tokens > 0)."""
    from qwendoom import snapshot
    from qwendoom import spec as S

    out = []
    paths = sorted(Path(directory).glob("*.npz"))
    if not paths:
        raise SystemExit(f"no DOOM ticks in {directory}: run with --doom-capture first")
    for p in paths:
        if names and p.stem not in names:
            continue
        z = np.load(p)
        rows = snapshot.Rows(z["layer"], z["vhead"], z["row"], z["vec"], int(z["first"]))
        n = max_tokens + 1 if max_tokens > 0 else len(z["toks"])
        out.append((p.stem, (lambda e, rows=rows: E.start(e, rows, S.PAD)), z["toks"][:n],
                    z["host"][:n].astype(np.uint8)))
    return out


# --- runs -------------------------------------------------------------------------------------


def run_once(e, begin, want, skip=None):
    """Seconds for one run of len(want) - 1 predictions (free-running when skip is None)."""
    begin(e)
    if skip is None:
        t = time.perf_counter()
        got = e.generate(want[0], len(want) - 1)
        dt = time.perf_counter() - t
        if got != list(want[1:]):
            bad = next(i for i, (g, w) in enumerate(zip(got, want[1:])) if g != w)
            raise SystemExit(f"MISMATCH at position {bad}")
    else:
        toks = np.asarray(want, np.int32)
        t = time.perf_counter()
        r = e.check(toks, skip)
        dt = time.perf_counter() - t
        if r != -1:
            raise SystemExit(f"MISMATCH at position {r}")
    return dt


def run_set(e, cases):
    """Seconds for one pass over cases [(name, begin, want, skip)]."""
    return sum(run_once(e, begin, want, skip) for _, begin, want, skip in cases)


def ntokens(cases):
    return sum(len(want) - 1 for _, _, want, _ in cases)


def profile(e, cases):
    e.profile(True)
    e.stats(reset=True)
    dt = run_set(e, cases)
    st = e.stats(reset=True)
    e.profile(False)
    n = st["steps"] - len(cases)  # (the PAD steps)
    tot = sum(st["ns"].values())
    stages = {k: {"ns_per_token": st["ns"][k] / n, "share": st["ns"][k] / tot,
                  "macs_per_token": st["mac"][k] / n} for k in E.STAGES}
    counters = {k: st[k] / n for k in ("key_writes", "query_reads", "active_units", "bound_skips")}
    counters.update({k: st[k] for k in ("head_sign", "head_exact", "slow_transcendental",
                                        "range_risk", "prefix_misses", "forced")})
    return {"tokens_per_s": ntokens(cases) / dt, "stages": stages, "counters": counters}


def print_profile(title, p):
    print(f"\n{title}: {p['tokens_per_s']:.0f} tok/s with profiling on")
    print(f"  {'stage':10s} {'ns/token':>9s} {'share':>6s} {'MACs/token':>11s}")
    for k, v in p["stages"].items():
        if v["ns_per_token"] or v["macs_per_token"]:
            print(f"  {k:10s} {v['ns_per_token']:9.0f} {100 * v['share']:5.1f}% "
                  f"{v['macs_per_token']:11.0f}")
    c = p["counters"]
    print(f"  per token: {c['key_writes']:.2f} key writes, {c['query_reads']:.2f} query reads, "
          f"{c['active_units']:.0f} active MLP units, {c['bound_skips']:.1f} channel groups "
          f"skipped by their bound")
    print(f"  head: {c['head_sign']} sign code / {c['head_exact']} full argmax; "
          f"{c['forced']} forced steps; slow transcendentals {c['slow_transcendental']}, "
          f"range risks {c['range_risk']}, prefix recomputes {c['prefix_misses']}")


def bench(name, model_dir, cases, threads, repeat):
    n = ntokens(cases)
    print(f"\n== {name}: {n} tokens ==")
    out = {"tokens": n, "cases": {c[0]: len(c[2]) - 1 for c in cases}, "configs": {}}
    engines = {nth: E.Engine(model_dir, n_threads=nth) for nth in threads}
    e = engines[threads[0]]
    out["open_seconds"] = e.info["open_seconds"]
    out["from_cache"] = bool(e.info["from_cache"])
    out["nnz_compact"] = e.info["nnz_compact"]
    # the repeats go round the configurations, so a change in the machine's load hits them alike
    times = {nth: [] for nth in threads}
    for _ in range(repeat):
        for nth in threads:
            times[nth].append(run_set(engines[nth], cases))
    for nth in threads:
        e = engines[nth]
        tps = [n / t for t in times[nth]]
        sync = e.bench_sync(20000) if nth > 1 else 0.0
        out["configs"][nth] = {"best_tok_s": max(tps), "median_tok_s": statistics.median(tps),
                               "us_per_token": 1e6 / max(tps), "pool_round_trip_ns": sync}
        print(f"  {nth} thread{'s' if nth > 1 else ' '}: best {max(tps):8.0f} tok/s "
              f"({1e6 / max(tps):6.1f} us/token), median {statistics.median(tps):8.0f}"
              + (f"; pool round trip {sync:.0f} ns" if nth > 1 else ""), flush=True)
    best = max(out["configs"], key=lambda k: out["configs"][k]["best_tok_s"])
    out["best_threads"] = best
    out["profile_1"] = profile(engines[threads[0]], cases)
    print_profile(f"profile, {threads[0]} thread", out["profile_1"])
    if best != threads[0]:
        out["profile_best"] = profile(engines[best], cases)
        print_profile(f"profile, {best} threads", out["profile_best"])
    for e in engines.values():
        e.close()
    return out


def ab(libs, cases, nth, repeat):
    """Interleaved A/B runs of engine builds (ABAB...) on the same cases: per build the best and
    median tok/s and the per-round ratios to the first build."""
    engines = []
    for path in libs:
        E.lib(path)
        engines.append(E.Engine(WEIGHTS, n_threads=nth))
    n = ntokens(cases)
    times = [[] for _ in libs]
    for r in range(repeat):
        order = range(len(libs)) if r % 2 == 0 else reversed(range(len(libs)))
        for i in order:
            times[i].append(run_set(engines[i], cases))
        print(f"  round {r + 1}: " + "  ".join(f"{n / t[-1]:8.0f}" for t in times), flush=True)
    res = {}
    for i, path in enumerate(libs):
        tps = [n / t for t in times[i]]
        ratio = [times[0][k] / times[i][k] for k in range(repeat)]
        res[path] = {"best_tok_s": max(tps), "median_tok_s": statistics.median(tps),
                     "median_speedup": statistics.median(ratio)}
        print(f"{path}: best {max(tps):8.0f} tok/s, median {statistics.median(tps):8.0f}, "
              f"speedup over the first {statistics.median(ratio):.3f} (median of rounds; "
              f"min {min(ratio):.3f}, max {max(ratio):.3f})")
    for e in engines:
        e.close()
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tiny", action="store_true", help="rv-tiny (the default set)")
    ap.add_argument("--weights", action="store_true", help="the qwendoom checkpoint, a C program")
    ap.add_argument("--doom", action="store_true", help="the qwendoom checkpoint, DOOM ticks")
    ap.add_argument("--doom-capture", action="store_true", help="capture the DOOM ticks")
    ap.add_argument("--windows", default=DOOM_WINDOWS, help="for --doom-capture")
    ap.add_argument("--only", default="", help="--doom / --ab: these windows only (a,b)")
    ap.add_argument("--ab", default="", help="interleaved A/B of engine builds on the DOOM ticks")
    ap.add_argument("--tokens", type=int, default=100_000,
                    help="rv-tiny / --weights: trace length; --ab: per DOOM window (0: all)")
    ap.add_argument("--threads", default="1,2,3,4,6")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--prog", default="c-qsort", help="upstream ELF for rv-tiny")
    ap.add_argument("--qd-prog", default="c-memsweep", help="C test program for --weights")
    ap.add_argument("--ref", action="store_true", help="also RefEngine's speed (short run)")
    ap.add_argument("--json", default=str(ROOT / "build" / "engine-bench" / "bench.json"))
    args = ap.parse_args()
    interactive_qos()
    if args.doom_capture:
        doom_capture(args.windows, DOOM_DIR)
        return
    threads = [int(t) for t in args.threads.split(",")]
    only = set(args.only.split(",")) - {""}
    if args.ab:
        ab(args.ab.split(","), doom_windows(only, args.tokens), threads[0], args.repeat)
        return
    res = {"machine": machine(), "lib": str(E.LIB)}
    if args.tiny or args.ref or not (args.weights or args.doom):
        begin, want = rv_setup(args.prog, args.tokens)
        res["rv_tiny"] = {"prog": args.prog,
                          **bench(f"rv-tiny, {args.prog}", RV_TINY,
                                  [(args.prog, begin, want, None)], threads, args.repeat)}
    if args.ref:
        r = E.RefEngine(RV_TINY)
        n = 300
        t = time.perf_counter()
        begin(r)
        tok = want[0]
        for i in range(n):
            tok = r.step(tok)
            assert tok == want[i + 1]
        res["rv_tiny"]["ref_engine_tok_s"] = n / (time.perf_counter() - t)
        print(f"\nRefEngine (numpy): {res['rv_tiny']['ref_engine_tok_s']:.0f} tok/s")
    if args.weights:
        begin, want, skip = qd_setup(args.qd_prog, args.tokens)
        res["weights"] = {"prog": args.qd_prog,
                          **bench(f"weights/, {args.qd_prog}", WEIGHTS,
                                  [(args.qd_prog, begin, want, skip)], threads, args.repeat)}
    if args.doom:
        cases = doom_windows(only)
        res["doom"] = {"windows": [json.loads((DOOM_DIR / f"{c[0]}.json").read_text())
                                   for c in cases],
                       **bench("weights/, DOOM ticks (" + ", ".join(c[0] for c in cases) + ")",
                               WEIGHTS, cases, threads, args.repeat)}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(res, indent=1))
    print(f"\nwrote {args.json}")


def machine():
    import platform
    import subprocess

    def sysctl(k):
        try:
            return subprocess.run(["sysctl", "-n", k], capture_output=True, text=True).stdout.strip()
        except OSError:
            return ""

    return {"cpu": sysctl("machdep.cpu.brand_string"), "p_cores": sysctl("hw.perflevel0.physicalcpu"),
            "e_cores": sysctl("hw.perflevel1.physicalcpu"), "os": platform.platform(),
            "load": sysctl("vm.loadavg")}


if __name__ == "__main__":
    main()
