"""The checkpoint's forward pass, one token at a time: a numpy reference and the native engine.

RefEngine computes exactly what stock ``Qwen3_5ForCausalLM`` computes in bf16 on the CPU: RMSNorm
with a 1 + w gain, the qkv / z / a / b projections, the causal conv + silu, l2-normalized q and k,
the gated delta rule in fp32, the gated RMSNorm, out_proj, the SwiGLU MLP, the final norm and the
tied output head with a greedy argmax, rounding to bf16 where the bf16 model rounds. It only skips
work that is exactly zero: all-zero weight rows and columns, value heads that cannot reach the
output, and state rows a key or query does not touch. It is generic in the hidden size, the number
of key / value heads (value head h reads key head h // (nv / nk), as ``repeat_interleave`` does),
the head dims and the conv width.

Engine wraps the native engine (native/engine, build/libqdengine.dylib), which computes the same
thing fast (docs/engine.md); tests/test_engine*.py hold the two (and stock transformers) to each
other: the same tokens and the same residual after every layer at every position.

Derived from nesformer/engine.py (MIT, (c) 2026 Drew Breunig; see LICENSE.nesformer).

A run (ARCHITECTURE.md): reset, step(PAD), load the snapshot rows, then step the tokens;
start() does the first three for either snapshot format.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import struct
from pathlib import Path

import numpy as np
import scipy.sparse as sp

F32 = np.float32
ROOT = Path(__file__).resolve().parents[1]
# (QD_ENGINE_LIB: another build of the engine, e.g. one side of an A/B comparison)
LIB = Path(os.environ.get("QD_ENGINE_LIB") or ROOT / "build" / "libqdengine.dylib")


def bf16r(x) -> np.ndarray:
    """Round float32 to the nearest bf16 (ties to even), as float32."""
    x = np.ascontiguousarray(x, dtype=F32)
    u = x.view(np.uint32)
    u = (u + (0x7FFF + ((u >> 16) & 1))) & 0xFFFF0000
    return u.astype(np.uint32).view(F32)


def silu(x):
    x = np.asarray(x, F32)
    with np.errstate(over="ignore"):
        return (x / (F32(1) + np.exp(-x))).astype(F32)


def rsqrt(v):
    """torch.rsqrt on the CPU: 1 / sqrt(v), each correctly rounded in fp32."""
    return (F32(1) / np.sqrt(np.asarray(v, F32))).astype(F32)


def sum16(v) -> np.float32:
    """The engine's fp32 reduction order (norms): 16 lanes, lane j summing v[j], v[j + 16], ...
    in order, then (l[j] + l[j + 8]) + (l[j + 4] + l[j + 12]) for j < 4, then (c0 + c1) +
    (c2 + c3). (np.add.reduce along axis 0 adds the rows in order; tests check it.) The circuit's
    sums are exact, so every order gives the same value; fixing one keeps the two engines
    bit-identical even where they would not be."""
    v = np.asarray(v, F32).reshape(-1)
    pad = -len(v) % 16
    if pad:
        v = np.concatenate([v, np.zeros(pad, F32)])
    acc = np.add.reduce(v.reshape(-1, 16), axis=0, dtype=F32)
    c = (acc[0:4] + acc[8:12]) + (acc[4:8] + acc[12:16])
    return F32(F32(c[0] + c[1]) + F32(c[2] + c[3]))


def rmsnorm(x, gain, eps):
    """Qwen3_5RMSNorm: (x * rsqrt(mean(x^2) + eps)) * (1 + w), then bf16 (mean = sum / n, as
    torch computes it on the CPU)."""
    var = F32(sum16(x * x) / F32(x.shape[0]))
    return bf16r(x * rsqrt(var + F32(eps)) * gain)


class Weights:
    """Lazy bf16 safetensors reader (memory-mapped)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            self.header = json.loads(fh.read(n))
        self.header.pop("__metadata__", None)
        self.base = 8 + n
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")

    def raw(self, name) -> np.ndarray:
        info = self.header[name]
        a, b = info["data_offsets"]
        assert info["dtype"] == "BF16", (name, info["dtype"])
        return self.mm[self.base + a:self.base + b].view("<u2").reshape(info["shape"])

    def get(self, name) -> np.ndarray:
        return (self.raw(name).astype(np.uint32) << 16).view(F32)


def csr(w: np.ndarray) -> sp.csr_matrix:
    """fp32 CSR with sorted columns: scipy's matvec then sums each row in increasing column order
    in fp32, as a row-major dot product would."""
    m = sp.csr_matrix(w.astype(F32))
    m.sort_indices()
    return m


class RefLayer:
    """One linear-attention decoder layer (token mixer + MLP)."""

    def __init__(self, W: Weights, L: int, cfg: dict):
        p = f"model.layers.{L}."
        self.nk, self.nv = cfg["linear_num_key_heads"], cfg["linear_num_value_heads"]
        self.dk, self.dv = cfg["linear_key_head_dim"], cfg["linear_value_head_dim"]
        self.eps = F32(cfg["rms_norm_eps"])
        assert self.nv % self.nk == 0
        self.r = self.nv // self.nk
        nk, nv, dk, dv = self.nk, self.nv, self.dk, self.dv
        self.g_in = F32(1) + W.get(p + "input_layernorm.weight")
        self.g_post = F32(1) + W.get(p + "post_attention_layernorm.weight")
        # projections: only rows that are not all zero
        qkv = W.get(p + "linear_attn.in_proj_qkv.weight")
        conv = W.get(p + "linear_attn.conv1d.weight")[:, 0, :]  # [channels, K]
        self.ksize = conv.shape[1]
        self.ch = np.nonzero((qkv != 0).any(1) & (conv != 0).any(1))[0]
        self.Wqkv = csr(qkv[self.ch])
        self.conv = conv[self.ch]
        self.hist = np.zeros((self.ksize, len(self.ch)), F32)
        kd = nk * dk
        self.kind = np.where(self.ch < kd, 0, np.where(self.ch < 2 * kd, 1, 2))
        rel = np.where(self.kind == 0, self.ch, np.where(self.kind == 1, self.ch - kd,
                                                          self.ch - 2 * kd))
        self.head = np.where(self.kind == 2, rel // dv, rel // dk)
        self.dim = np.where(self.kind == 2, rel % dv, rel % dk)
        zw = W.get(p + "linear_attn.in_proj_z.weight")
        self.zch = np.nonzero((zw != 0).any(1))[0]
        self.Wz = csr(zw[self.zch])
        ow = W.get(p + "linear_attn.out_proj.weight")
        self.ocols = np.nonzero((ow != 0).any(0))[0]
        self.Wo = csr(ow[:, self.ocols])
        # value heads that can reach the output: out_proj columns and a z gate (silu(0) = 0)
        self.vh = np.intersect1d(np.unique(self.ocols // dv), np.unique(self.zch // dv))
        self.Wa = csr(W.get(p + "linear_attn.in_proj_a.weight")[self.vh])
        self.Wb = csr(W.get(p + "linear_attn.in_proj_b.weight")[self.vh])
        self.dt_bias = W.get(p + "linear_attn.dt_bias")[self.vh]
        self.A = np.exp(W.get(p + "linear_attn.A_log")[self.vh]).astype(F32)
        self.norm_w = W.get(p + "linear_attn.norm.weight")
        self.S = np.zeros((nv, dk, dv), F32)
        g = W.get(p + "mlp.gate_proj.weight")
        u = W.get(p + "mlp.up_proj.weight")
        d = W.get(p + "mlp.down_proj.weight")
        # a unit with a zero gate row (silu(0) = 0), up row or down column adds exactly 0
        self.units = np.nonzero((g != 0).any(1) & (u != 0).any(1) & (d != 0).any(0))[0]
        self.Wg, self.Wu = csr(g[self.units]), csr(u[self.units])
        self.Wd = csr(d[:, self.units])
        self.has_attn = len(self.vh) > 0 and len(self.ch) > 0
        self.qscale = F32(dk ** 0.5)

    def reset(self):
        self.hist[:] = 0
        self.S[:] = 0

    def attn(self, x):
        nk, nv, dk, dv = self.nk, self.nv, self.dk, self.dv
        xn = rmsnorm(x, self.g_in, self.eps)
        # projections and the causal conv + silu (taps summed oldest first)
        qkv = bf16r(self.Wqkv @ xn)
        self.hist[:-1] = self.hist[1:]
        self.hist[-1] = qkv
        acc = np.zeros(len(self.ch), F32)
        for t in range(self.ksize):
            acc = (acc + self.conv[:, t] * self.hist[t]).astype(F32)
        cv = bf16r(silu(bf16r(acc)))
        q = np.zeros((nk, dk), F32)
        k = np.zeros((nk, dk), F32)
        v = np.zeros((nv, dv), F32)
        for kind, arr in enumerate((q, k, v)):
            m = self.kind == kind
            arr[self.head[m], self.dim[m]] = cv[m]
        z = np.zeros(nv * dv, F32)
        z[self.zch] = bf16r(self.Wz @ xn)
        z = z.reshape(nv, dv)
        b = bf16r(self.Wb @ xn)
        a = bf16r(self.Wa @ xn)
        with np.errstate(over="ignore"):
            beta = bf16r(F32(1) / (F32(1) + np.exp(-b, dtype=F32)))
        with np.errstate(over="ignore", under="ignore"):
            xa = (a + self.dt_bias).astype(F32)
            spl = np.where(xa > 20, xa, np.log1p(np.exp(xa, dtype=F32))).astype(F32)
            decay = np.exp((-self.A * spl).astype(F32), dtype=F32)
        full = np.zeros((nv, dv), F32)
        for i, h in enumerate(self.vh):
            kh = h // self.r
            S = self.S[h]
            Q, K = q[kh], k[kh]
            Q = (Q * rsqrt(sum16(Q * Q) + F32(1e-6))).astype(F32)
            K = (K * rsqrt(sum16(K * K) + F32(1e-6))).astype(F32)
            Q = (Q / self.qscale).astype(F32)
            if decay[i] != 1:
                S *= decay[i]
            kr = np.nonzero(K)[0]
            if len(kr):
                kv = np.zeros(dv, F32)
                for r_ in kr:
                    kv = (kv + S[r_] * K[r_]).astype(F32)
                delta = ((v[h] - kv) * beta[i]).astype(F32)
                for r_ in kr:
                    S[r_] = (S[r_] + K[r_] * delta).astype(F32)
            o = np.zeros(dv, F32)
            for r_ in np.nonzero(Q)[0]:
                o = (o + S[r_] * Q[r_]).astype(F32)
            o = bf16r(o)
            # the gated RMSNorm of this value head
            var = F32(sum16(o * o) / F32(dv))
            y = bf16r(o * rsqrt(var + self.eps))
            y = bf16r(self.norm_w * y)
            full[h] = bf16r(y * silu(z[h]))
        att = bf16r(self.Wo @ full.reshape(-1)[self.ocols])
        return bf16r(x + att)

    def mlp(self, x):
        xn = rmsnorm(x, self.g_post, self.eps)
        act = bf16r(bf16r(silu(bf16r(self.Wg @ xn))) * bf16r(self.Wu @ xn))
        return bf16r(x + bf16r(self.Wd @ act))

    def __call__(self, x):
        if self.has_attn:  # (otherwise no head reaches the output: the mixer adds exactly 0)
            x = self.attn(x)
        return self.mlp(x) if len(self.units) else x


def assert_zero_layer(W: Weights, L: int) -> None:
    """A full-attention layer must be all zero (norms aside): then it adds exactly 0."""
    for name in W.header:
        if name.startswith(f"model.layers.{L}.") and "norm" not in name:
            assert not W.raw(name).any(), f"{name} is not zero: unsupported"


class RefEngine:
    """step(token) -> the model's greedy next token (numpy, slow, the reference)."""

    def __init__(self, model_dir, exact_head: bool = False):
        d = Path(model_dir)
        self.cfg = json.loads((d / "config.json").read_text())
        cfg = self.cfg
        W = Weights(d / "model.safetensors")
        self.layers = {}
        for L, t in enumerate(cfg["layer_types"]):
            if t == "linear_attention":
                self.layers[L] = RefLayer(W, L, cfg)
            else:
                assert_zero_layer(W, L)
        self.E = W.raw("model.embed_tokens.weight")  # bf16 bits, memory-mapped
        assert cfg.get("tie_word_embeddings", False), "the head must be tied to the embedding"
        self.gain = F32(1) + W.get("model.norm.weight")
        self.eps = F32(cfg["rms_norm_eps"])
        self.code = np.nonzero(self.gain)[0]  # the final norm zeroes every other dim
        self.Ecode = (self.E[:, self.code].astype(np.uint32) << 16).view(F32)
        self.head = SignHead(self.Ecode)
        self.exact_head = exact_head
        self.capture = None  # a list: residuals after every layer are appended per step
        self.steps = 0

    def reset(self):
        for layer in self.layers.values():
            layer.reset()

    def set_row(self, layer: int, value_head: int, row: int, vec) -> None:
        self.layers[layer].S[value_head, row] = np.asarray(vec, F32)

    def embed(self, t: int) -> np.ndarray:
        return (self.E[t].astype(np.uint32) << 16).view(F32).copy()

    def forward(self, t: int) -> np.ndarray:
        x = self.embed(t)
        res = [] if self.capture is not None else None
        for L in range(len(self.cfg["layer_types"])):
            if L in self.layers:
                x = self.layers[L](x)
            if res is not None:
                res.append(x.copy())
        if res is not None:
            self.capture.append(np.stack(res))
        self.steps += 1
        return x

    def step(self, t: int) -> int:
        x = self.forward(t)
        y = rmsnorm(x, self.gain, self.eps)[self.code]
        if not self.exact_head:
            tok = self.head.fast(y)
            if tok >= 0:
                return tok
        return int(np.argmax(bf16r(self.Ecode @ y)))

    def step_forced(self, t: int) -> None:
        self.forward(t)


class SignHead:
    """The tied head's argmax without the vocabulary matmul, when the embedding allows it.

    If every token row is, on the code dims (those the final norm keeps), either all zero or all
    +-A for one A, and y (the normed residual there) has equal nonzero magnitudes m, then a row
    scores A m (n - 2 d) with d its sign disagreements with y: exact in fp32 and (n <= 256) in
    bf16, so the row with d = 0 is the unique argmax. Otherwise (or if no row has y's signs)
    the caller computes the exact head.
    """

    def __init__(self, Ecode: np.ndarray):
        self.n = Ecode.shape[1]
        nz = Ecode != 0
        rows = nz.any(1)
        amp = np.unique(np.abs(Ecode[rows]))
        self.ok = (0 < self.n <= 24 and len(amp) == 1 and amp[0] > 0
                   and bool((nz[rows].all(1)).all()))
        self.table = None
        if self.ok:
            ids = np.nonzero(rows)[0]
            bits = ((Ecode[ids] > 0).astype(np.int64) << np.arange(self.n)).sum(1)
            # (duplicate codes: the first token wins, as argmax's first maximum)
            codes, first = np.unique(bits, return_index=True)
            self.table = np.full(1 << self.n, -1, np.int32)
            self.table[codes] = ids[first]

    def fast(self, y) -> int:
        if not self.ok:
            return -1
        a = np.abs(y)
        if a[0] == 0 or (a != a[0]).any():
            return -1
        return int(self.table[int(((y > 0).astype(np.int64) << np.arange(self.n)).sum())])


# --- the native engine ---------------------------------------------------------------------------


def checkpoint_hash(path) -> str:
    """sha256 of the safetensors file (the engine cache key; the native engine computes the same)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


class _Info(ctypes.Structure):
    _fields_ = [(n, ctypes.c_int) for n in (
        "hidden", "n_layers", "vocab", "key_heads", "value_heads", "key_dim", "value_dim",
        "conv_kernel", "n_threads", "n_code", "sign_head", "last_needed_layer")] + [
        ("nnz_compact", ctypes.c_int64), ("open_seconds", ctypes.c_double),
        ("from_cache", ctypes.c_int), ("sha256", ctypes.c_char * 65), ("prefix_dims", ctypes.c_int)]


STAGES = ["embed", "norm", "qk_proj", "conv", "delta", "v_lazy", "out_proj", "gate", "up", "down",
          "residual", "head", "sync"]


class _Stats(ctypes.Structure):
    _fields_ = [("steps", ctypes.c_uint64), ("forced", ctypes.c_uint64),
                ("head_sign", ctypes.c_uint64), ("head_exact", ctypes.c_uint64),
                ("slow_transcendental", ctypes.c_uint64),
                ("mac", ctypes.c_uint64 * len(STAGES)), ("ns", ctypes.c_double * len(STAGES)),
                ("key_writes", ctypes.c_uint64), ("query_reads", ctypes.c_uint64),
                ("active_units", ctypes.c_uint64), ("range_risk", ctypes.c_uint64),
                ("prefix_misses", ctypes.c_uint64), ("bound_skips", ctypes.c_uint64)]


_lib = None


def lib(path=None) -> ctypes.CDLL:
    """The native engine (build it with `make -C native/engine`)."""
    global _lib
    if _lib is None or path:
        L = ctypes.CDLL(str(path or LIB))
        P, I, F = ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float)
        IP = ctypes.POINTER(ctypes.c_int)
        sig = {
            "qd_engine_open_ex": (P, [ctypes.c_char_p, I, ctypes.c_char_p, I]),
            "qd_engine_last_error": (ctypes.c_char_p, []),
            "qd_engine_close": (None, [P]),
            "qd_engine_reset": (None, [P]),
            "qd_engine_set_row": (None, [P, I, I, I, F]),
            "qd_engine_set_rows": (None, [P, I, I, I, IP, F]),
            "qd_engine_get_row": (None, [P, I, I, I, F]),
            "qd_engine_step": (I, [P, I]),
            "qd_engine_step_forced": (None, [P, I]),
            "qd_engine_feed": (I, [P, IP, I]),
            "qd_engine_generate": (I, [P, I, I, I, IP]),
            "qd_engine_generate_until": (I, [P, I, I, IP, I, IP]),
            "qd_engine_check": (I, [P, IP, I, ctypes.POINTER(ctypes.c_uint8)]),
            "qd_engine_set_exact_head": (None, [P, I]),
            "qd_engine_capture": (None, [P, F]),
            "qd_engine_set_profile": (None, [P, I]),
            "qd_engine_get_info": (None, [P, ctypes.POINTER(_Info)]),
            "qd_engine_get_stats": (None, [P, ctypes.POINTER(_Stats), I]),
            "qd_engine_bench_sync": (ctypes.c_double, [P, I]),
        }
        for name, (res, args) in sig.items():
            fn = getattr(L, name)
            fn.restype, fn.argtypes = res, args
        _lib = L
    return _lib


def _ip(a: np.ndarray):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_int))


def _fp(a: np.ndarray):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


class Engine:
    """The native engine with RefEngine's interface (plus bulk calls that stay in C).

    n_threads <= 0: QD_THREADS or 1. cache_dir: the layout cache (default build/engine-cache).
    generic: no specialized paths (unit records, cached prefixes, channel bounds), for tests.
    refresh: every cached prefix is dropped at every 7th step (base recomputed), for tests."""

    def __init__(self, model_dir, n_threads: int = 0, cache_dir=None, no_cache=False,
                 verbose=False, exact_head=False, generic=False, refresh=False):
        self.L = lib()
        flags = ((1 if no_cache else 0) | (2 if verbose else 0) | (4 if generic else 0)
                 | (8 if refresh else 0))
        self.h = self.L.qd_engine_open_ex(str(model_dir).encode(), n_threads,
                                          str(cache_dir).encode() if cache_dir else None, flags)
        if not self.h:
            raise RuntimeError(f"qd_engine_open: {self.L.qd_engine_last_error().decode()}")
        info = _Info()
        self.L.qd_engine_get_info(self.h, ctypes.byref(info))
        self.info = {k: (getattr(info, k).decode() if k == "sha256" else getattr(info, k))
                     for k, _ in _Info._fields_}
        self.hidden, self.n_layers = info.hidden, info.n_layers
        self.dv = info.value_dim
        self._cap = None
        self.exact_head = exact_head

    def close(self):
        if self.h:
            self.L.qd_engine_close(self.h)
            self.h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @property
    def exact_head(self) -> bool:
        return self._exact

    @exact_head.setter
    def exact_head(self, on: bool):
        self._exact = bool(on)
        self.L.qd_engine_set_exact_head(self.h, int(on))

    def reset(self):
        self.L.qd_engine_reset(self.h)

    def set_row(self, layer: int, value_head: int, row: int, vec) -> None:
        v = np.ascontiguousarray(vec, F32)
        assert v.shape == (self.dv,)
        self.L.qd_engine_set_row(self.h, layer, value_head, row, _fp(v))

    def set_rows(self, layer: int, value_head: int, rows, vecs) -> None:
        r = np.ascontiguousarray(rows, np.int32)
        v = np.ascontiguousarray(vecs, F32).reshape(len(r), self.dv)
        self.L.qd_engine_set_rows(self.h, layer, value_head, len(r), _ip(r), _fp(v))

    def get_row(self, layer: int, value_head: int, row: int) -> np.ndarray:
        out = np.zeros(self.dv, F32)
        self.L.qd_engine_get_row(self.h, layer, value_head, row, _fp(out))
        return out

    def step(self, t: int) -> int:
        return self.L.qd_engine_step(self.h, int(t))

    def step_forced(self, t: int) -> None:
        self.L.qd_engine_step_forced(self.h, int(t))

    def feed(self, toks) -> int:
        a = np.ascontiguousarray(toks, np.int32)
        return self.L.qd_engine_feed(self.h, _ip(a), len(a))

    def generate(self, t: int, n: int, stop: int = -1) -> list[int]:
        out = np.zeros(n, np.int32)
        k = self.L.qd_engine_generate(self.h, int(t), n, stop, _ip(out))
        return out[:k].tolist()

    def generate_until(self, t: int, n: int, stops=()) -> list[int]:
        """Free-running greedy decoding from t until a prediction is one of ``stops`` (it is
        the last element and is not fed: pass it as the next call's t), at most n tokens."""
        out = np.zeros(n, np.int32)
        st = np.ascontiguousarray(stops, np.int32)
        k = self.L.qd_engine_generate_until(self.h, int(t), n, _ip(st), len(st), _ip(out))
        return out[:k].tolist()

    def check(self, toks, skip=None) -> int:
        """Teacher-forced: the first i whose prediction != toks[i + 1], or -1."""
        a = np.ascontiguousarray(toks, np.int32)
        sk = None
        if skip is not None:
            sk = np.ascontiguousarray(skip, np.uint8)
            sk = sk.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        return self.L.qd_engine_check(self.h, _ip(a), len(a), sk)

    def capture(self, on: bool = True) -> np.ndarray | None:
        """With on, every step writes the residual after each layer into the returned
        [n_layers, hidden] array (overwritten per step)."""
        if on:
            self._cap = np.zeros((self.n_layers, self.hidden), F32)
            self.L.qd_engine_capture(self.h, _fp(self._cap))
        else:
            self._cap = None
            self.L.qd_engine_capture(self.h, None)
        return self._cap

    def bench_sync(self, n: int = 10000) -> float:
        """ns per round trip of the worker pool (0 workers: the call overhead)."""
        return self.L.qd_engine_bench_sync(self.h, n)

    def profile(self, on: bool = True) -> None:
        self.L.qd_engine_set_profile(self.h, int(on))

    def stats(self, reset: bool = False) -> dict:
        s = _Stats()
        self.L.qd_engine_get_stats(self.h, ctypes.byref(s), int(reset))
        return {"steps": s.steps, "forced": s.forced, "head_sign": s.head_sign,
                "head_exact": s.head_exact, "slow_transcendental": s.slow_transcendental,
                "key_writes": s.key_writes, "query_reads": s.query_reads,
                "active_units": s.active_units, "range_risk": s.range_risk,
                "prefix_misses": s.prefix_misses, "bound_skips": s.bound_skips,
                "mac": dict(zip(STAGES, list(s.mac))), "ns": dict(zip(STAGES, list(s.ns)))}


# --- the development checkpoint (rv-tiny, upstream chipformer.rv) ---------------------------------


def rv_rows(c, machine) -> list[tuple[int, int, dict]]:
    """A machine's snapshot as (layer, value head, {row: vec}) for set_row: core.snapshot fills
    the circuit's head states, written as chipformer.rv.hf.inject writes them into HF's cache
    (value head j * r + x of key head j)."""
    from chipformer.rv import core, hf

    core.snapshot(c, machine)
    r = c.target.value_heads // c.target.key_heads
    out = []
    for ni, (L, j) in hf.head_slots(c).items():
        init = c.nodes[ni].init or []
        for x in range(len(c.nodes[ni].vheads())):
            out.append((L, j * r + x, init[x] if x < len(init) else {}))
    return out


def load_rows(engine, rows) -> None:
    """Write state rows into an engine (RefEngine, Engine) after reset and step(PAD): rv_rows'
    list, or a qwendoom.snapshot.Rows (arrays layer, vhead, row, vec)."""
    if not isinstance(rows, list):
        order = np.lexsort((rows.row, rows.vhead, rows.layer))
        L, vh, rr, vec = rows.layer[order], rows.vhead[order], rows.row[order], rows.vec[order]
        cut = np.flatnonzero((np.diff(L) != 0) | (np.diff(vh) != 0)) + 1
        rows = [(int(L[a]), int(vh[a]), dict(zip(rr[a:b].tolist(), vec[a:b])))
                for a, b in zip(np.r_[0, cut], np.r_[cut, len(L)]) if b > a]
    for L, vh, rr in rows:
        if not rr:
            continue
        if hasattr(engine, "set_rows"):
            idx = sorted(rr)
            engine.set_rows(L, vh, idx, np.stack([np.asarray(rr[i], F32) for i in idx]))
        else:
            for row, vec in rr.items():
                engine.set_row(L, vh, row, vec)


def start(engine, rows, pad: int) -> None:
    """A run's start (ARCHITECTURE.md): reset, step(PAD), the snapshot rows. The engine is then
    ready to step the snapshot's first token (GO). (PAD's prediction is not used: a forced step
    skips the head, which for PAD's all-zero code would take the full vocabulary argmax.)"""
    engine.reset()
    engine.step_forced(pad)
    load_rows(engine, rows)


def rv_start(engine, c, machine, pad: int) -> None:
    """start() with an upstream chipformer.rv machine's snapshot (rv-tiny)."""
    start(engine, rv_rows(c, machine), pad)
