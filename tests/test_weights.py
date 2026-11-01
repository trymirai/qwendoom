"""The checkpoint emitter (qwendoom.weights): its layers equal chipformer's emitter's, every weight
is bf16, the embedding equals the IR's token features, the files in weights/ are consistent, and
(QD_SLOW=1) stock transformers runs the checkpoint exactly (qwendoom.hfcheck)."""

import hashlib
import json
import os
import struct
from pathlib import Path

import numpy as np
import pytest

from qwendoom import core, snapshot, weights
from qwendoom import spec as S

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights"
SLOW = not os.environ.get("QD_SLOW")


@pytest.fixture(scope="module")
def circ():
    return core.build()


@pytest.fixture(scope="module")
def lay_tc(circ):
    lay = weights.pick_layout(len(circ.dims))
    return lay, weights.text_config(circ, lay)


def test_layout(circ, lay_tc):
    lay, tc = lay_tc
    assert (lay.hidden, lay.m) == (2304, 9) and lay.m + len(circ.dims) <= lay.hidden
    assert lay.scale == 2.0**-36  # m B^2 / hidden = 9 * 2^80 / 2304 = 4^36
    assert tc["linear_num_key_heads"] == 16 and tc["linear_num_value_heads"] == 64
    assert tc["num_hidden_layers"] == circ.n_layers() + 1 == 17
    assert tc["layer_types"][-1] == "full_attention"
    assert tc["vocab_size"] % 64 == 0 and tc["vocab_size"] > S.PAD


def test_layers_equal_chipformer_emitter(circ, lay_tc):
    """Every circuit layer equals chipformer.q35.emit.emit_layers' (torch) bit for bit."""
    torch = pytest.importorskip("torch")
    from chipformer.q35.emit import Layout, emit_layers

    lay, tc = lay_tc
    want = emit_layers(circ, Layout(hidden=lay.hidden, m=lay.m), tc, 0, "model.", None,
                       set(circ.keep))
    for L in range(circ.n_layers()):
        got = weights.chip_layer(circ, lay, tc, L)
        for name, t in got.items():
            w = want.pop(name)
            assert t.dtype == np.float32 and torch.equal(torch.from_numpy(t), w), name
    assert not want


def test_embedding_equals_features(circ, lay_tc):
    lay, _ = lay_tc
    e = weights.embedding_matrix(circ, lay)
    feats = sorted(circ.token_fns)
    rng = np.random.default_rng(1)
    toks = rng.integers(0, S.VOCAB, 3000).tolist() + list(S.CONTROLS.values())
    toks += [b + s - 1 for b, s in S.FAMILIES.values()] + [b for b, _ in S.FAMILIES.values()]
    for t in toks:
        assert (e[t, :lay.m] == lay.ballast).all()
        assert [float(e[t, lay.dim(d)]) for d in feats] == [float(circ.feature_value(d, t))
                                                             for d in feats], t
    assert not e[S.PAD:].any()  # PAD and the padding rows are zero
    used = set(range(lay.m)) | {lay.dim(d) for d in feats}
    assert not e[:, [i for i in range(lay.hidden) if i not in used]].any()


def test_code_rows_decode(circ, lay_tc):
    """Each token's code (its embedding on the code dims) is distinct and decodes back."""
    lay, _ = lay_tc
    toks = list(range(0, S.VOCAB, 97)) + list(S.CONTROLS.values())
    for t in toks:
        bits = core.token_code(t)
        fid = sum(b << i for i, b in enumerate(bits[:core.NFAM]))
        p = sum(b << i for i, b in enumerate(bits[core.NFAM:]))
        assert core.decode_code(fid, p) == t


needs_weights = pytest.mark.skipif(not (WEIGHTS / "model.safetensors").exists(),
                                   reason="weights/ not emitted (python -m qwendoom.weights)")


@needs_weights
def test_checkpoint_files(circ, lay_tc):
    lay, tc = lay_tc
    cfg = json.loads((WEIGHTS / "config.json").read_text())
    assert cfg["architectures"] == ["Qwen3_5ForCausalLM"] and cfg["model_type"] == "qwen3_5_text"
    assert {k: cfg[k] for k in tc} == json.loads(json.dumps(tc))
    meta = json.loads((WEIGHTS / "qwendoom.json").read_text())
    assert meta["heads"] == {k: list(v) for k, v in snapshot.head_slots(circ).items()}
    assert meta["code"]["dims"] == [lay.dim(d) for d in circ.keep]
    assert meta["families"] == {k: list(v) for k, v in S.FAMILIES.items()}
    assert meta["controls"] == S.CONTROLS and meta["pad"] == S.PAD
    # the safetensors header: every tensor in bf16 with the expected shape, contiguous
    with open(WEIGHTS / "model.safetensors", "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    header.pop("__metadata__")
    shapes = weights.tensor_shapes(tc)
    assert [k for k, _ in shapes] == sorted(header, key=lambda k: header[k]["data_offsets"][0])
    off = 0
    for name, shape in shapes:
        h = header[name]
        assert h["dtype"] == "BF16" and tuple(h["shape"]) == shape
        assert h["data_offsets"][0] == off
        off = h["data_offsets"][1]
    assert (WEIGHTS / "model.safetensors").stat().st_size == 8 + n + off
    # SHA256SUMS
    for line in (WEIGHTS / "SHA256SUMS").read_text().split("\n"):
        if line and not line.endswith("model.safetensors"):
            digest, name = line.split()
            assert hashlib.sha256((WEIGHTS / name).read_bytes()).hexdigest() == digest, name


@needs_weights
def test_checkpoint_tensors(circ, lay_tc):
    """The saved bf16 tensors equal the emitter's (a few layers, the norm, the embedding)."""
    lay, tc = lay_tc
    path = WEIGHTS / "model.safetensors"
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    raw = np.memmap(path, np.uint16, "r", offset=8 + n)

    def get(name):
        a, b = header[name]["data_offsets"]
        t = (raw[a // 2:b // 2].astype(np.uint32) << 16).view(np.float32)
        return t.reshape(header[name]["shape"])

    for L in (0, 2, 7, circ.n_layers() - 1):
        for name, t in weights.chip_layer(circ, lay, tc, L).items():
            assert np.array_equal(get(name), t), name
    assert np.array_equal(get("model.norm.weight"), weights.final_norm(circ, lay))
    e = get("model.embed_tokens.weight")
    assert np.array_equal(e[::997], weights.embedding_matrix(circ, lay)[::997])


@pytest.mark.skipif(SLOW, reason="loads the checkpoint in transformers (QD_SLOW=1)")
@needs_weights
@pytest.mark.parametrize("prog", ["call", "input"])
def test_transformers_exact(circ, prog):
    """Stock transformers, bf16, CPU: the residual before the exit equals the IR's on every
    circuit dim, and teacher-forced (prefill and decode) and free-running predictions equal the
    reference trace."""
    from qwendoom import hfcheck

    model, meta = hfcheck.load_model(str(WEIGHTS))
    res = hfcheck.residual(model, meta, circ, hfcheck.program(prog))
    assert res["inexact_dims"] == 0 and res["dims"] > 2000, res
    for mode in ("prefill", "decode"):
        res = hfcheck.teacher_forced(model, meta, hfcheck.program(prog), mode)
        assert res["n_wrong"] == 0 and res["checked"] >= 10, res
    res = hfcheck.generate(model, meta, hfcheck.program(prog))
    assert res["equal"], res
