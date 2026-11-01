"""The real sim: build/doom.elf's sim_main (P_Ticker) on the C reference machine, tick by tick, in
lockstep with the host's native sim mode (native/host, another track's interpreter) on Freedoom
E1M1 with scripted keys: the same pokes in, the same instruction and trace-token counts and the
same RAM out. The decoder (the mirror) rebuilds that RAM and the register file from the
reference's tokens alone, and the lockstep checker (taking the input words from the stream)
accepts them. Skipped when the host library, doom.elf or the WAD is missing."""

import random
import time

import numpy as np
import pytest

from qwendoom import trace as T

H = pytest.importorskip("qwendoom.host")
pytestmark = pytest.mark.skipif(
    not (H.LIB.exists() and H.DEFAULT_ELF.exists() and H.DEFAULT_WAD.exists()),
    reason="needs build/libqdhost.dylib, build/doom.elf and freedoom1.wad")

TICKS = 1050  # 30 s of game time
ACTIONS = ["forward", "back", "left", "right", "strafeleft", "straferight", "fire", "use", "run"]


def key_script(seed: int = 7) -> dict:
    """tick -> [(action, down)]: random key changes every few ticks (walk, turn, fire, open)."""
    rng, held, script = random.Random(seed), set(), {}
    for t in range(35, TICKS, 9):
        ev = []
        for a in ACTIONS:
            want = rng.random() < (0.45 if a in ("forward", "fire") else 0.15)
            if want != (a in held):
                ev.append((a, want))
                (held.add if want else held.discard)(a)
        script[t] = ev
    return script


def test_doom_ticks_against_native_sim():
    try:
        h = H.Host()
    except Exception as e:  # noqa: BLE001  (the host track's library in flux)
        pytest.skip(f"the host does not start: {e!r}")
    mems = [T.Memory.from_arrays(h.ram, h.drom, h.rom, own_ram=True) for _ in range(3)]
    ref, dec, cref = T.Ref(mems[0]), T.Decoder(mems[1]), T.Ref(mems[2])
    chk = T.Checker(cref)
    script = key_script()
    players, leveltime = h.symbol("players"), h.symbol("leveltime")
    places = set()  # the player's (x, y) after each tick: the keys reach the playsim
    tick = steps0 = 0
    totals = {"tokens": 0, "host_tokens": 0, "records": 0, "instructions": 0, "pokes": 0,
              "resyncs": 0, "ref_s": 0.0, "dec_s": 0.0, "chk_s": 0.0}
    while tick < TICKS:
        ev = h.run()
        if ev == H.FRAME:
            continue
        assert ev == H.SIM_TICK, (ev, h.error)
        for action, down in script.get(tick, []):
            h.action(action, down)
        if h.need_resync():  # a level load: the snapshot is RAM, DROM, ROM and the entry context
            h.resync()
            pc, sp, gp = h.sim_entry()
            regs = [0] * 32
            regs[2], regs[3] = sp, gp
            for m in mems:
                m.sync_ram()
            for x in (ref, dec, cref):
                x.set_context(pc=pc, regs=regs)
            chk.reset()
            steps0 = 0
            totals["resyncs"] += 1
        pokes = h.take_pokes()
        assert len(pokes) % 2 == 1 and pokes[-1] == 0

        ref.push_input(pokes)
        t0 = time.perf_counter()
        toks, flags, why = ref.run(stop_yield=True)
        totals["ref_s"] += time.perf_counter() - t0
        assert why == "yield", (tick, why, ref.status, hex(ref.pc))
        steps = ref.counts["steps"]
        host_toks = int(np.count_nonzero(flags & T.TF_HOST))
        assert host_toks == sum(len(T.encode_value_record(int(w))) for w in pokes)

        st = h.sim_tick_native()
        assert st.input_words == len(pokes)
        assert (st.instructions, st.tokens) == (steps - steps0, len(toks)), tick
        assert np.array_equal(mems[0].ram, h.ram), f"tick {tick}: RAM differs from the native sim"

        t0 = time.perf_counter()
        res, used = dec.feed_many(toks, stop=1 << T.YIELD)
        totals["dec_s"] += time.perf_counter() - t0
        assert (res, used) == (T.YIELD, len(toks)), (tick, res, used, dec.error_text)
        assert dec.context == ref.context and np.array_equal(mems[1].ram, h.ram), tick

        t0 = time.perf_counter()
        n, state = chk.check(toks)
        totals["chk_s"] += time.perf_counter() - t0
        assert (n, state) == (len(toks), T.CHK_OK), (tick, chk.divergence)
        assert np.array_equal(mems[2].ram, h.ram) and cref.context == ref.context

        h.complete_sim_tick()
        if players and h.word(players):
            mo = h.word(players)  # players[0].mo; mobj_t: thinker (3 words), x, y
            places.add((h.word(mo + 12), h.word(mo + 16)))
        totals["tokens"] += len(toks)
        totals["host_tokens"] += host_toks
        totals["records"] += int(np.count_nonzero(flags & T.TF_FIRST))
        totals["instructions"] += steps - steps0
        totals["pokes"] += (len(pokes) - 1) // 2
        steps0 = steps
        tick += 1
    c = h.counters()
    assert c["sim_ticks"] == TICKS and c["sim_tokens"] == totals["tokens"]
    if players and leveltime:
        assert len(places) > 100 and h.word(leveltime) > TICKS // 2, (len(places), leveltime)
    print(f"\n{TICKS} DOOM ticks (E1M1, {totals['resyncs']} resyncs): "
          f"{totals['instructions']:,} instructions, {totals['records']:,} records, "
          f"{totals['tokens']:,} tokens ({totals['tokens'] / TICKS:,.0f} per tick, "
          f"{totals['host_tokens']:,} host-owned), {totals['pokes']:,} pokes, the player at "
          f"{len(places)} places, leveltime {h.word(leveltime) if leveltime else '?'}")
    print(f"  ref {totals['records'] / totals['ref_s'] / 1e6:.1f} M records/s, decoder "
          f"{totals['tokens'] / totals['dec_s'] / 1e6:.1f} M tokens/s, checker "
          f"{totals['tokens'] / totals['chk_s'] / 1e6:.1f} M tokens/s")


def test_doom_model_mode_with_a_perfect_model():
    """Model mode end to end with RefModel standing in for the engine: two hosts take the same
    keys; one runs its ticks natively, the other through drive_tick (input words injected, the
    decoder's stores landing in that host's own RAM, the checker in lockstep). After every tick
    the two hosts' RAM is equal, and so is the frame they render at the end."""
    try:
        native, model_host = H.Host(), H.Host()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"the host does not start: {e!r}")
    h = model_host
    mirror = T.Memory.from_arrays(h.ram, h.drom, h.rom)  # the host's RAM itself
    own = [T.Memory.from_arrays(h.ram, h.drom, h.rom, own_ram=True) for _ in range(2)]
    dec, model_ref, cref = T.Decoder(mirror), T.Ref(own[0]), T.Ref(own[1])
    chk = T.Checker(cref)
    script = {5: [("forward", True), ("fire", True)], 30: [("right", True)],
              45: [("right", False), ("fire", False)], 60: [("strafeleft", True)]}
    model = pred = None
    tokens = injected = 0
    for tick in range(70):
        for host in (native, model_host):
            while (ev := host.run()) == H.FRAME:
                pass
            assert ev == H.SIM_TICK, (ev, host.error)
            for action, down in script.get(tick, []):
                host.action(action, down)
        native.native_tick()
        if h.need_resync():
            h.resync()
            pc, sp, gp = h.sim_entry()
            regs = [0] * 32
            regs[2], regs[3] = sp, gp
            for m in own:
                m.sync_ram()
            for x in (dec, model_ref, cref):
                x.set_context(pc=pc, regs=regs)
            chk.reset()
            model = T.RefModel(model_ref)
            pred = T.start_stream(model.step, dec, chk)  # GO to the decoder, checker and model
        pokes = h.take_pokes()
        model_ref.push_input(pokes)
        res = T.drive_tick(model.step, pred, dec, pokes, chk)
        pred = res.pred
        assert res.event == T.YIELD and not res.unread
        h.complete_sim_tick()
        assert np.array_equal(native.ram, h.ram), f"tick {tick}: the mirrored RAM differs"
        tokens += len(res.tokens)
        injected += int(res.host.sum())
    assert native.next_frame() == H.FRAME
    while (ev := h.run()) != H.FRAME:
        assert ev == H.SIM_TICK, (ev, h.error)
        pokes = h.take_pokes()
        model_ref.push_input(pokes)
        pred = T.drive_tick(model.step, pred, dec, pokes, chk).pred
        h.complete_sim_tick()
    f1, f2 = native.frame(), h.frame()
    assert np.array_equal(f1[0], f2[0]) and np.array_equal(f1[1], f2[1])
    assert len(np.unique(f1[0])) > 20  # (a real 3D view, not a blank screen)
    print(f"\nmodel mode, 70 ticks: {tokens:,} tokens ({injected:,} injected), "
          f"checker {chk.counts}")
