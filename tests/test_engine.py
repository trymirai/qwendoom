"""The engine's gates (docs/engine.md), on the development checkpoint build/rv-tiny (upstream
chipformer.rv, built by `python -m chipformer.rv.tiny --out build/rv-tiny`) and, when it is
there, on the qwendoom checkpoint weights/:

(a) free-running greedy decoding with the native engine equals the reference machine's trace
    (upstream ELFs on rv-tiny: >= 200k tokens; the C test programs on weights/), and on real
    DOOM ticks (Freedoom E1M1, captured in the test) every prediction equals the reference's
    token except at the host-injected input records;
(b) RefEngine (numpy) and the native engine agree token for token and residual for residual
    (every layer's output, every position) over the first few thousand positions;
plus the engine's options and API: the exact head, the generic paths, forced base changes
(refresh), threads, forced steps, generate_until, the layout cache, state rows, and a synthetic
checkpoint with a residual wider than 32,768 dims. (c), stock transformers, is in
test_engine_hf.py.

Residuals are compared as values (np.array_equal): +0 and -0 are equal (the sign of a zero
residual never reaches a token: see docs/engine.md).
"""

import json
import os
import struct
from pathlib import Path

import numpy as np
import pytest

from qwendoom import engine as E

ROOT = Path(__file__).resolve().parents[1]
RV_TINY = ROOT / "build" / "rv-tiny"
WEIGHTS = ROOT / "weights"
RV_DATA = ROOT / "third_party" / "qwenputer" / "experimental" / "riscv" / "tests" / "data" / "rv"
LONG = os.environ.get("QD_LONG") == "1"  # longer runs (more programs, more positions)

need_lib = pytest.mark.skipif(not E.LIB.exists(), reason="make -C native/engine")
need_tiny = pytest.mark.skipif(not (RV_TINY / "model.safetensors").exists() or not RV_DATA.exists(),
                               reason="build/rv-tiny or the upstream test ELFs are missing")
need_weights = pytest.mark.skipif(not (WEIGHTS / "model.safetensors").exists(),
                                  reason="weights/ (python -m qwendoom build) is missing")


# --- rv-tiny ------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def circuit():
    from chipformer.rv import core

    return core.build()


def rv_program(name):
    from chipformer.rv import elf

    return elf.program(RV_DATA / f"{name}.elf")


def rv_trace(name, n=10**7):
    from chipformer.rv import ref

    return ref.Machine.from_program(rv_program(name)).run(n)


def rv_begin(engine, circuit, name):
    from chipformer.rv import ref
    from chipformer.rv import spec as S

    E.rv_start(engine, circuit, ref.Machine.from_program(rv_program(name)), S.VOCAB)


@pytest.fixture(scope="module")
def tiny():
    e = E.Engine(RV_TINY)
    yield e
    e.close()


# c-qsort alone is 240k tokens; the set is 377k
RV_PROGS = ["c-crc32", "c-qsort", "c-sieve", "c-gmpmul", "c-sort", "rv32um-div", "mpy-s1"]
if LONG:
    RV_PROGS += ["c-sieve4", "mpy-s5", "c-crc32p", "c-strops", "rv32um-mulh", "rv32uzba-sh1add"]


@need_lib
@need_tiny
@pytest.mark.parametrize("name", RV_PROGS)
def test_rv_free_running(tiny, circuit, name):
    """(a): greedy decoding from GO reproduces the reference trace to HALT."""
    from chipformer.rv import spec as S

    want = rv_trace(name)
    rv_begin(tiny, circuit, name)
    tiny.stats(reset=True)
    got = tiny.generate(want[0], len(want) + 8, stop=S.tok("HALT"))
    bad = next((i for i, (g, w) in enumerate(zip(got, want[1:])) if g != w), None)
    assert got == want[1:], f"{name}: first difference at {bad} of {len(want)}"
    st = tiny.stats()
    assert st["range_risk"] == 0 and st["slow_transcendental"] == 0, st


def residual_compare(ref_engine, engine, toks, n):
    """Teacher-force toks[:n] through both engines (already started); the first position whose
    prediction or any layer's residual differs, or None."""
    cap = engine.capture(True)
    ref_engine.capture = []
    try:
        for i in range(n):
            a = ref_engine.step(toks[i])
            b = engine.step(toks[i])
            if a != b or not np.array_equal(ref_engine.capture[-1], cap):
                diff = np.nonzero((ref_engine.capture[-1] != cap).any(1))[0]
                return i, a, b, diff[:5].tolist()
            ref_engine.capture.clear()
    finally:
        engine.capture(False)
        ref_engine.capture = None
    return None


@pytest.fixture(scope="module")
def tiny_ref():
    return E.RefEngine(RV_TINY)


@need_lib
@need_tiny
@pytest.mark.parametrize("name,n", [("mpy-s1", 6000 if LONG else 2500), ("c-qsort", 1500)])
def test_rv_ref_equals_native(tiny, tiny_ref, circuit, name, n):
    """(b): RefEngine == the native engine, every layer's residual at every position."""
    toks = rv_trace(name, n + 1)
    rv_begin(tiny_ref, circuit, name)
    rv_begin(tiny, circuit, name)
    assert residual_compare(tiny_ref, tiny, toks, min(n, len(toks) - 1)) is None


@need_lib
@need_tiny
def test_rv_ref_equals_trace(tiny_ref, circuit):
    """RefEngine alone (free-running) reproduces a short trace."""
    want = rv_trace("c-crc32", 600)
    rv_begin(tiny_ref, circuit, "c-crc32")
    tok, got = want[0], []
    for _ in range(len(want) - 1):
        tok = tiny_ref.step(tok)
        got.append(tok)
    assert got == want[1:]


@need_lib
@need_tiny
def test_rv_exact_head(tiny, circuit):
    """The sign-code head and the full vocabulary argmax pick the same tokens."""
    want = rv_trace("mpy-s1", 1500)
    rv_begin(tiny, circuit, "mpy-s1")
    tiny.exact_head = True
    try:
        tiny.stats(reset=True)
        assert tiny.check(want) == -1
        st = tiny.stats()
        assert st["head_exact"] == len(want) - 1 and st["head_sign"] == 0
    finally:
        tiny.exact_head = False


@need_lib
@need_tiny
def test_rv_generic_and_threads(tiny, circuit):
    """The generic paths (no unit records, no cached prefixes), a base change every 7th step
    (refresh) and 1..6 threads give the same residuals as the default engine at every
    position."""
    toks = rv_trace("mpy-s1", 3000)
    others = [E.Engine(RV_TINY, generic=True), E.Engine(RV_TINY, refresh=True)] + [
        E.Engine(RV_TINY, n_threads=t) for t in (2, 3, 6)]
    engines = [tiny] + others
    try:
        caps = [e.capture(True) for e in engines]
        for e in engines:
            rv_begin(e, circuit, "mpy-s1")
        for i in range(len(toks) - 1):
            preds = [e.step(toks[i]) for e in engines]
            assert preds == [toks[i + 1]] * len(engines), (i, preds)
            for k in range(1, len(engines)):
                assert np.array_equal(caps[0], caps[k]), (i, k)
    finally:
        for e in engines:
            e.capture(False)
        for e in others:
            e.close()


@need_lib
@need_tiny
def test_rv_forced_and_check(tiny, circuit):
    """Forced steps (no head, trailing layers skipped) leave the same state as full steps; check
    and feed agree with step."""
    toks = rv_trace("c-sieve", 3000)
    rv_begin(tiny, circuit, "c-sieve")
    assert tiny.check(toks) == -1
    # every third position forced: the predictions at the others still match
    rv_begin(tiny, circuit, "c-sieve")
    skip = np.zeros(len(toks), np.uint8)
    skip[1::3] = 1
    assert tiny.check(toks, skip) == -1
    rv_begin(tiny, circuit, "c-sieve")
    assert tiny.feed(toks[:1000]) == toks[1000]
    # a wrong token is reported where it is predicted
    bad = list(toks)
    bad[500] = (bad[500] + 1) % 1000
    rv_begin(tiny, circuit, "c-sieve")
    assert tiny.check(bad) == 499


@need_lib
@need_tiny
def test_rv_rows_roundtrip(tiny):
    tiny.reset()
    v = np.arange(128, dtype=np.float32) - 64
    tiny.set_row(3, 5, 77, v)
    assert np.array_equal(tiny.get_row(3, 5, 77), v)
    tiny.set_rows(3, 5, [1, 2], np.stack([v, -v]))
    assert np.array_equal(tiny.get_row(3, 5, 2), -v)
    tiny.reset()
    assert not tiny.get_row(3, 5, 77).any()


@need_lib
@need_tiny
def test_rv_layout_cache(tmp_path, circuit):
    """A second open reads the layouts from the cache; both engines run identically."""
    a = E.Engine(RV_TINY, cache_dir=tmp_path)
    b = E.Engine(RV_TINY, cache_dir=tmp_path)
    try:
        assert not a.info["from_cache"] and b.info["from_cache"]
        assert a.info["sha256"] == b.info["sha256"]
        assert len(list(tmp_path.glob("*.qdc"))) == 1
        want = rv_trace("c-crc32")
        for e in (a, b):
            rv_begin(e, circuit, "c-crc32")
            assert e.generate(want[0], len(want) - 1) == want[1:]
    finally:
        a.close()
        b.close()


def test_sum16_order():
    """sum16 adds lane j's values in order (np.add.reduce over axis 0 is sequential)."""
    rng = np.random.default_rng(0)
    v = (rng.standard_normal(4096) * 10.0 ** rng.integers(-8, 8, 4096)).astype(np.float32)
    lanes = np.zeros(16, np.float32)
    for i, x in enumerate(v):
        lanes[i % 16] = np.float32(lanes[i % 16] + x)
    c = (lanes[0:4] + lanes[8:12]) + (lanes[4:8] + lanes[12:16])
    assert E.sum16(v) == np.float32(np.float32(c[0] + c[1]) + np.float32(c[2] + c[3]))


def test_bf16r():
    x = np.array([1.0, 1.00390625, 1.005859375, 1.0078125, -3.0e38, 0.0, -0.0], np.float32)
    got = E.bf16r(x)
    # 1 + 2^-8 is a tie: to even (1.0); 1 + 1.5 * 2^-8 rounds up
    assert got[0] == 1.0 and got[1] == 1.0 and got[2] == 1.0078125 and got[3] == 1.0078125
    assert np.array_equal(E.bf16r(got), got)


# --- a synthetic checkpoint wider than 32,768 dims ----------------------------------------------


def write_checkpoint(d, cfg, tensors):
    """config.json and a bf16 model.safetensors (the arrays must be exact in bf16) in d."""
    (d / "config.json").write_text(json.dumps(cfg))
    header, blobs, off = {}, [], 0
    for name, a in tensors.items():
        a = np.ascontiguousarray(a, np.float32)
        assert np.array_equal(E.bf16r(a), a), name
        b = (a.view(np.uint32) >> 16).astype("<u2").tobytes()
        header[name] = {"dtype": "BF16", "shape": list(a.shape),
                        "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    h = json.dumps(header).encode()
    h += b" " * (-len(h) % 8)
    (d / "model.safetensors").write_bytes(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def wide_checkpoint(d, H=33024, hi=33000, lo=32800):
    """One linear-attention layer (a key and a value head, dk 64, dv 16, no MLP) over H dims.
    Tokens 1-3 hold a ballast 1 at dim 0 and 1 / 2 at hi (tokens 1, 2) or 2 at lo (token 3);
    dims below lo are the eager channels' constant prefix. The 16 single-tap channels (q and k
    dims 0-7) read every dim (+1; dim 0: -2, hi: +3), so the group bound tables have a row per
    dim, rows past 32,767 for hi and lo, and the single-tap group is live only through hi's term
    (else bounded dead). The 4 ring keys (k dims 8-11, conv over all 4 taps) read hi (-1) and lo
    (+4): a token 1 or 2 proves the ring group dead (its projection deferred), a later token 3
    in the window needs it. Value dims 0 and 1 read hi and lo, both z gates hi; out_proj writes
    them to dims 5 and 6. Every pre-activation is 0, <= -89 or >= 17 (no slow silu)."""
    V, dk, dv, KS, nu = 4, 64, 16, 4, 16
    C = 2 * dk + dv
    emb = np.zeros((V, H), np.float32)
    emb[1:, 0] = 1
    emb[1, hi], emb[2, hi], emb[3, lo] = 1, 2, 2
    qkv = np.zeros((C, H), np.float32)
    conv = np.zeros((C, 1, KS), np.float32)
    for c in list(range(8)) + list(range(dk, dk + 8)):
        qkv[c] = 1
        qkv[c, 0], qkv[c, hi] = -2, 3
        conv[c, 0, KS - 1] = 1
    for c in range(dk + 8, dk + 12):
        qkv[c, hi], qkv[c, lo] = -1, 4
        conv[c, 0] = 1
    qkv[2 * dk, hi] = qkv[2 * dk + 1, lo] = 1
    conv[2 * dk:2 * dk + 2, 0, KS - 1] = 1
    z = np.zeros((dv, H), np.float32)
    z[0, hi] = z[1, hi] = 1
    out = np.zeros((H, dv), np.float32)
    out[5, 0] = out[6, 1] = 1
    p = "model.layers.0."
    zero = np.zeros(H, np.float32)
    write_checkpoint(d, {
        "hidden_size": H, "num_hidden_layers": 1, "vocab_size": V, "linear_num_key_heads": 1,
        "linear_num_value_heads": 1, "linear_key_head_dim": dk, "linear_value_head_dim": dv,
        "linear_conv_kernel_dim": KS, "rms_norm_eps": 1e-6, "tie_word_embeddings": True,
        "layer_types": ["linear_attention"], "intermediate_size": nu,
    }, {
        "model.embed_tokens.weight": emb, "model.norm.weight": zero,
        p + "input_layernorm.weight": zero, p + "post_attention_layernorm.weight": zero,
        p + "linear_attn.in_proj_qkv.weight": qkv, p + "linear_attn.conv1d.weight": conv,
        p + "linear_attn.in_proj_z.weight": z,
        p + "linear_attn.in_proj_a.weight": np.zeros((1, H)),
        p + "linear_attn.in_proj_b.weight": np.zeros((1, H)),  # beta 0.5
        p + "linear_attn.out_proj.weight": out,
        p + "linear_attn.dt_bias": np.full(1, -200.0),  # with A_log 0: decay exactly 1
        p + "linear_attn.A_log": np.zeros(1), p + "linear_attn.norm.weight": np.ones(dv),
        p + "mlp.gate_proj.weight": np.zeros((nu, H)), p + "mlp.up_proj.weight": np.zeros((nu, H)),
        p + "mlp.down_proj.weight": np.zeros((H, nu)),
    })


@need_lib
def test_wide_hidden(tmp_path):
    """A residual wider than 32,768 dims (open takes up to 65,536): the group bounds, the live
    groups' scatter and the deferred ring projections read the inputs past dim 32,767 (an int16
    table row once dropped them, x[5] after token 1 was 0, and wrote out of bounds at open).
    The default engine, refresh and the generic paths equal RefEngine, every residual and
    prediction at every position."""
    wide_checkpoint(tmp_path)
    toks = [int(c) for c in "010300002311032000001233010030"]
    ref_engine = E.RefEngine(tmp_path)
    for kw in ({}, {"refresh": True}, {"generic": True}):
        e = E.Engine(tmp_path, no_cache=True, **kw)
        try:
            ref_engine.reset()
            assert residual_compare(ref_engine, e, toks, len(toks)) is None, kw
            st = e.stats()
            assert st["range_risk"] == 0 and st["slow_transcendental"] == 0, (kw, st)
            assert (st["bound_skips"] > 0) == (not kw.get("generic")), (kw, st)
            e.reset()
            cap = e.capture(True)
            e.step(0)
            e.step(1)
            assert cap[0, 5] != 0, kw  # (token 1's query, key, value and gate all read hi)
        finally:
            e.capture(False)
            e.close()


@need_lib
def test_open_refuses_16bit_overflow(tmp_path):
    """Shapes whose eager channels' 16-row blocks or key heads overflow their 16-bit fields
    (key heads x key dim > 524,288) are refused at open, before the weights are read."""
    cfg = {"hidden_size": 64, "num_hidden_layers": 1, "vocab_size": 4, "linear_num_key_heads": 8193,
           "linear_num_value_heads": 8193, "linear_key_head_dim": 64, "linear_value_head_dim": 16,
           "layer_types": ["linear_attention"]}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(RuntimeError, match="unsupported shapes"):
        E.Engine(tmp_path, no_cache=True)


# --- broken checkpoints and layout caches: an error, never a crash -------------------------------


def small_checkpoint(d):
    """wide_checkpoint over 128 dims (opens in milliseconds); its model.safetensors."""
    wide_checkpoint(d, H=128, hi=100, lo=80)
    return d / "model.safetensors"


def header_end(st):
    return 8 + struct.unpack("<Q", st.read_bytes()[:8])[0]


def edit_header(st, edit):
    """Apply edit to the parsed safetensors header and write it back (the data unchanged)."""
    b, n = st.read_bytes(), header_end(st)
    h = json.loads(b[8:n])
    edit(h)
    h = json.dumps(h).encode()
    h += b" " * (-len(h) % 8)
    st.write_bytes(struct.pack("<Q", len(h)) + h + b[n:])


def set_entry(name, **kv):
    return lambda h: h[name].update(kv)


NORM = "model.norm.weight"
BROKEN_WEIGHTS = {
    "missing": (lambda st: st.unlink(), "cannot open .*model.safetensors: No such file"),
    "directory": (lambda st: (st.unlink(), st.mkdir()), "model.safetensors: not a regular file"),
    "empty": (lambda st: os.truncate(st, 0), r"model.safetensors: truncated \(0 bytes\)"),
    "header cut": (lambda st: os.truncate(st, 40), "truncated or not safetensors"),
    "data cut": (lambda st: os.truncate(st, st.stat().st_size - 100), "past the end of the file"),
    "no data": (lambda st: os.truncate(st, header_end(st)), "past the end of the file"),
    "garbage": (lambda st: st.write_bytes(np.random.default_rng(0).bytes(4096)),
                "truncated or not safetensors"),
    "garbage header": (
        lambda st: st.write_bytes(struct.pack("<Q", 16) + b"\x00{]garbage\xff\"[ 1e"),
        "bad safetensors header"),
    "deep header": (lambda st: st.write_bytes(struct.pack("<Q", 200_000) + b"[" * 200_000),
                    "bad safetensors header"),
    "header not an object": (lambda st: st.write_bytes(struct.pack("<Q", 8) + b"[1, 2]  "),
                             "bad safetensors header"),
    "dtype": (lambda st: edit_header(st, set_entry(NORM, dtype=16)), "norm.weight is not BF16"),
    "no shape": (lambda st: edit_header(st, lambda h: h[NORM].pop("shape")), "bad shape"),
    "shape": (lambda st: edit_header(st, set_entry(NORM, shape=[64, 2])),
              r"norm.weight has shape \[64, 2, 1\], expected \[128, 1, 1\]"),
    "offsets": (lambda st: edit_header(st, set_entry(NORM, data_offsets=[0, 2])),
                "do not hold its shape"),
    "offsets past the end": (
        lambda st: edit_header(st, set_entry(NORM, data_offsets=[10**9, 10**9 + 256])),
        "past the end of the file"),
}


@need_lib
@pytest.mark.parametrize("case", BROKEN_WEIGHTS)
def test_open_broken_weights(tmp_path, case):
    """A missing, unreadable, truncated or malformed model.safetensors fails open with an error
    naming the file (a missing one was hashed through an mmap of fd -1: a segfault)."""
    mutate, match = BROKEN_WEIGHTS[case]
    st = small_checkpoint(tmp_path)
    mutate(st)
    for kw in ({"no_cache": True}, {"cache_dir": tmp_path / "cache"}):
        with pytest.raises(RuntimeError, match=match):
            E.Engine(tmp_path, **kw)


@need_lib
@pytest.mark.skipif(os.geteuid() == 0, reason="root reads any file")
def test_open_unreadable_weights(tmp_path):
    st = small_checkpoint(tmp_path)
    E.Engine(tmp_path, cache_dir=tmp_path / "cache").close()  # (the sha256 memoized)
    os.chmod(st, 0)
    try:
        with pytest.raises(RuntimeError,
                           match="cannot open .*model.safetensors: Permission denied"):
            E.Engine(tmp_path, cache_dir=tmp_path / "cache")
    finally:
        os.chmod(st, 0o644)


def set_config(**kv):
    def edit(d):
        cfg = json.loads((d / "config.json").read_text())
        cfg.update(kv)
        (d / "config.json").write_text(json.dumps({k: v for k, v in cfg.items() if v is not None}))
    return edit


BROKEN_CONFIGS = {
    "missing": (lambda d: (d / "config.json").unlink(), "cannot open .*config.json"),
    "not json": (lambda d: (d / "config.json").write_text('{"hidden_size": 12'),
                 "bad .*config.json"),
    "deep json": (lambda d: (d / "config.json").write_text("[" * 200_000), "bad .*config.json"),
    "no hidden_size": (set_config(hidden_size=None), "config.json: hidden_size is missing"),
    "hidden_size": (set_config(hidden_size=-128), "hidden_size is not a positive integer"),
    "key dim": (set_config(linear_key_head_dim=0), "linear_key_head_dim is not a positive integer"),
    "conv": (set_config(linear_conv_kernel_dim=2.5), "linear_conv_kernel_dim is not a positive"),
    "layer_types object": (set_config(layer_types={"linear_attention": 1}),
                           "layer_types is not a list"),
    "layer_types numbers": (set_config(layer_types=[1]), "layer_types is not a list"),
    "layer_types length": (set_config(layer_types=[]), "layer_types is not a list"),
    "layer type": (set_config(layer_types=["mamba"]), "layer type mamba"),
}


@need_lib
@pytest.mark.parametrize("case", BROKEN_CONFIGS)
def test_open_broken_config(tmp_path, case):
    """A missing or unreadable config.json, or one whose shapes or layer_types are missing or
    not what they must be, fails open with an error naming the key."""
    mutate, match = BROKEN_CONFIGS[case]
    small_checkpoint(tmp_path)
    mutate(tmp_path)
    with pytest.raises(RuntimeError, match=match):
        E.Engine(tmp_path, no_cache=True)


@need_lib
def test_open_damaged_cache(tmp_path):
    """A layout cache that is truncated, of another version or shape, garbage, unreadable or not
    a file is stale: open rebuilds the layouts from the weights (the same predictions) and
    replaces it."""
    small_checkpoint(tmp_path)
    cache = tmp_path / "cache"
    toks = [int(c) for c in "0103000023110320000012"]
    with_cache = E.Engine(tmp_path, cache_dir=cache)
    want = [with_cache.step(t) for t in toks]
    with_cache.close()
    (qdc,) = cache.glob("*.qdc")
    good = qdc.read_bytes()
    damage = {f"cut {n}": good[:n] for n in (0, 7, 8, 30, 52, 53, len(good) // 2, len(good) - 1)}
    damage["version"] = good[:8] + struct.pack("<i", 4) + good[12:]
    damage["shape"] = good[:12] + struct.pack("<i", 64) + good[16:]
    damage["garbage"] = np.random.default_rng(0).bytes(len(good))
    if os.geteuid() != 0:
        damage["unreadable"] = good

    def reopen():
        e = E.Engine(tmp_path, cache_dir=cache)
        try:
            assert [e.step(t) for t in toks] == want
            return e.info["from_cache"]
        finally:
            e.close()

    for name, data in damage.items():
        qdc.write_bytes(data)
        if name == "unreadable":
            os.chmod(qdc, 0)
        assert not reopen(), name  # rebuilt
        assert qdc.read_bytes() == good and reopen(), name  # replaced
    qdc.unlink()
    qdc.mkdir()
    assert not reopen() and qdc.is_dir() and not list(cache.glob("*.tmp"))


# --- the qwendoom checkpoint ----------------------------------------------------------------------


def qd_machine(name):
    from qwendoom import core_check

    return core_check.elf_machine(name, core_check.elf_inputs(name))


def qd_begin(engine, name):
    from qwendoom import snapshot
    from qwendoom import spec as S

    rows = snapshot.machine_rows(qd_machine(name))
    E.start(engine, rows, S.PAD)
    return rows.first


@pytest.fixture(scope="module")
def qd():
    e = E.Engine(WEIGHTS)
    yield e
    e.close()


QD_PROGS = ["c-classic", "c-input", "c-recurse", "rv32um-div", "rv32ui-sb"]
if LONG:
    QD_PROGS += ["c-memsweep", "c-bytes", "c-div64", "c-funcptr", "c-switch"]


@need_lib
@need_weights
@pytest.mark.parametrize("name", QD_PROGS)
def test_qd_free_running(qd, name):
    """(a) on weights/: the model's own predictions equal the trace at every position the host
    does not own (input records are injected: fed, not compared); with every prediction equal,
    teacher forcing is free-running decoding."""
    m = qd_machine(name)
    want = m.run(10**7)
    skip = np.zeros(len(want), np.uint8)
    skip[m.injected] = 1
    first = qd_begin(qd, name)
    assert first == want[0]
    qd.stats(reset=True)
    assert qd.check(want, skip) == -1
    st = qd.stats()
    assert st["range_risk"] == 0 and st["slow_transcendental"] == 0, st


@need_lib
@need_weights
def test_qd_ref_equals_native(qd):
    """(b) on weights/."""
    ref_engine = E.RefEngine(WEIGHTS)
    m = qd_machine("c-classic")
    toks = m.run(3001 if LONG else 1201)
    qd_begin(ref_engine, "c-classic")
    qd_begin(qd, "c-classic")
    assert residual_compare(ref_engine, qd, toks, len(toks) - 1) is None


@need_lib
@need_weights
def test_qd_generate_until(qd):
    """generate_until stops right after the first stop token (not fed), resumes from it, and
    chained calls reproduce the free-running trace (c-classic: 75 putchar BYTEs, then HALT);
    without a stop it runs n tokens."""
    from qwendoom import spec as S

    want = qd_machine("c-classic").run(10**7)
    b0, nb = S.FAMILIES["BYTE"]
    f0, nf = S.FAMILIES["FLT"]
    stops = list(range(b0, b0 + nb)) + list(range(f0, f0 + nf)) + [S.tok("HALT")]
    assert qd_begin(qd, "c-classic") == want[0]
    got, tok, calls = [], want[0], 0
    while not got or got[-1] != S.tok("HALT"):
        out = qd.generate_until(tok, len(want), stops)
        assert out and out[-1] in stops and not any(t in stops for t in out[:-1])
        got += out
        tok, calls = out[-1], calls + 1
    assert got == want[1:]
    assert calls == sum(b0 <= t < b0 + nb for t in want) + 1
    qd_begin(qd, "c-classic")
    assert qd.generate_until(want[0], 500, []) == want[1:501]
    assert qd.generate_until(want[500], 0, stops) == []
    assert qd.generate_until(want[500], 1, stops) == want[501:502]


# --- real DOOM ticks (needs the DOOM build, the host and Freedoom; skipped otherwise) ----------


def doom_ready():
    from qwendoom import host as H

    return all(p.exists() for p in (H.DEFAULT_ELF, H.LIB, H.DEFAULT_WAD))


need_doom = pytest.mark.skipif(not (WEIGHTS / "model.safetensors").exists() or not E.LIB.exists()
                               or not doom_ready(), reason="weights/ or the DOOM build is missing")


@pytest.fixture(scope="module")
def doom_ticks(tmp_path_factory):
    """Freedoom E1M1 windows of ticks captured as scripts/bench_engine.py --doom-capture does:
    the level start (ticks 0-2, 39,462 tokens) and a fight (ticks 700-701, 10 monsters awake,
    55,301 tokens); with QD_LONG the benchmark's windows."""
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import bench_engine as B

    d = tmp_path_factory.mktemp("doom")
    B.doom_capture(B.DOOM_WINDOWS if LONG else "start:0:3,fight:700:2", d)
    return B.doom_windows(directory=d)


@need_lib
@need_doom
def test_doom_ticks(qd, doom_ticks):
    """(a) on real DOOM ticks: every prediction equals the reference machine's token except at
    the host-injected input records (fed, not compared)."""
    for name, begin, toks, skip in doom_ticks:
        begin(qd)
        qd.stats(reset=True)
        assert qd.check(toks, skip) == -1, name
        st = qd.stats()
        assert st["range_risk"] == 0 and st["slow_transcendental"] == 0, st
        assert st["bound_skips"] > 0


@need_lib
@need_doom
def test_doom_paths_agree(qd, doom_ticks):
    """The group bounds, deferred ring projections and their flush at a base change (refresh: a
    forced base change every 7th step), the generic paths and 3 threads: the same residual
    after every layer and the same prediction at every position of real DOOM ticks."""
    others = [E.Engine(WEIGHTS, refresh=True), E.Engine(WEIGHTS, generic=True),
              E.Engine(WEIGHTS, n_threads=3)]
    engines = [qd] + others
    try:
        caps = [e.capture(True) for e in engines]
        for name, begin, toks, skip in doom_ticks:
            for e in engines:
                begin(e)
            for i in range(min(len(toks) - 1, 8000 if LONG else 2500)):
                if skip[i + 1]:
                    for e in engines:
                        e.step_forced(int(toks[i]))
                else:
                    preds = [e.step(int(toks[i])) for e in engines]
                    assert preds == [toks[i + 1]] * len(engines), (name, i, preds)
                for k in range(1, len(engines)):
                    assert np.array_equal(caps[0], caps[k]), (name, i, k)
        assert others[0].stats()["prefix_misses"] > 1000
    finally:
        for e in engines:
            e.capture(False)
        for e in others:
            e.close()
