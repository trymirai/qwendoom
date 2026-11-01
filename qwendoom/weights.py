"""Compile the circuit into a stock Qwen3_5ForCausalLM checkpoint:

    .venv/bin/python -m qwendoom.weights --out weights

writes config.json, generation_config.json, model.safetensors (bf16), qwendoom.json (the
vocabulary, the output code, where every state table lives, the address map, the run procedure)
and SHA256SUMS.

Derived from Qwenputer's chipformer/q35/emit.py and chipformer/rv/tiny.py (MIT, (c) 2026
Divyansh Agrawal; see LICENSE.chipformer) and nesformer/weights.py
(MIT, (c) 2026 Drew Breunig; see LICENSE.nesformer): the same weights as
chipformer's ``emit_layers`` (tests/test_weights.py compares them), computed in numpy one layer
at a time and streamed into the safetensors file, so the 4.5 GB checkpoint is never held in
memory whole.

Residual layout of a chip token:

    [ballast x m] [unused] [IR dims 1..n-1] [unused]      (IR dim d at residual index m + d)

Chip tokens carry B = 2^40 in each ballast dim, so every RMSNorm divides by a known power of 2
and a residual value x reaches the projections as exactly x * c (``Layout.scale``); the weights
carry 1 / c, and the IR's constant ONE rides on the ballast. The circuit's layers are linear-
attention (Gated DeltaNet) layers with Qwen3.5's head shapes (16 key heads of 128 dims, 4 value
heads per key head: value head 4 j + x reads key head j, as the model's repeat_interleave does);
one all-zero full-attention layer comes last (HF's cache counts positions there). The exit (the
last MLP) clears every dim but the output code, ballast included, and writes the predicted code
as KAPPA * (2 bit - 1); the final norm passes only the code dims (gain 1 + w = 2^20 + 1, 0
elsewhere) and the output head is tied to the embedding, whose rows hold CODE_AMP * (2 bit - 1)
of their own token's code: the argmax is the token whose code bits all agree.
"""

import argparse
import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from chipformer.dsl.graph import ONE, Expr
from chipformer.q35.ir import A_BIT, A_REF, FFN, HD, K, REF_COLS, Head
from qwendoom import core, snapshot
from qwendoom import spec as S

BASE = {  # Qwen3.5 text config fields we do not size ourselves (as chipformer.q35.tiny.BASE)
    "hidden_act": "silu", "linear_conv_kernel_dim": 4, "linear_key_head_dim": HD,
    "linear_value_head_dim": HD, "num_attention_heads": 8, "num_key_value_heads": 2,
    "head_dim": 256, "rms_norm_eps": 1e-6, "attn_output_gate": True, "attention_bias": False,
    "attention_dropout": 0.0, "initializer_range": 0.02, "use_cache": True,
    "rope_parameters": {"rope_type": "default", "rope_theta": 10000000,
                        "partial_rotary_factor": 0.25, "mrope_section": [11, 11, 10],
                        "mrope_interleaved": True},
    "max_position_embeddings": 1 << 30, "tie_word_embeddings": True,
}
CODE_GAIN = 2.0**20  # final-norm weight on the output code (chipformer.q35.tiny.CODE_GAIN)
N_VOCAB = -(-(S.VOCAB + 1) // 64) * 64  # chip tokens, PAD, zero rows up to a multiple of 64


@dataclass
class Layout:
    """chipformer.q35.emit.Layout without slot sharing (off = 0)."""

    hidden: int
    m: int  # ballast dims
    log2_ballast: int = 40

    @property
    def ballast(self) -> float:
        return 2.0**self.log2_ballast

    @property
    def scale(self) -> float:
        """c: the normed value of a residual value 1 on a chip token."""
        log4 = math.log(self.m * self.ballast**2 / self.hidden, 4)
        assert abs(log4 - round(log4)) < 1e-12, "m * B^2 / hidden must be a power of 4"
        return 2.0 ** -round(log4)

    def dim(self, d: int) -> int:
        """Residual index of IR dim d >= 1."""
        return self.m + d


def pick_layout(n_dims: int) -> Layout:
    """chipformer.q35.tiny.pick_layout: the smallest hidden = m * 4^k (a multiple of 64) that
    fits m >= 3 ballast dims and the IR's dims."""
    best = None
    for m in range(3, 16):
        k = 0
        while m * 4**k < m + n_dims or (m * 4**k) % 64:
            k += 1
        if best is None or m * 4**k < best.hidden:
            best = Layout(hidden=m * 4**k, m=m)
    return best


def bf16_round(x: float) -> float:
    """x rounded to bf16 as torch does (double -> float -> bf16, ties to even)."""
    u = np.array([x], np.float32).view(np.uint32)
    u = (u + (0x7FFF + ((u >> 16) & 1))) & 0xFFFF0000
    return float(u.astype(np.uint32).view(np.float32)[0])


def bf16_parts(x: float, n: int) -> list[float]:
    """x as at most n bf16 values that sum to it exactly (a constant over the ballast dims)."""
    parts = []
    for _ in range(n):
        if x == 0:
            break
        p = bf16_round(x)
        parts.append(p)
        x -= p
    if x != 0:
        raise ValueError(f"constant needs more than {n} bf16 parts")
    return parts


def row(e: Expr, lay: Layout, out: np.ndarray, w: float = 1.0) -> None:
    """Add w * e as a projection row (input side: the normed residual)."""
    c = lay.scale
    for d, coef in e.terms.items():
        if d == ONE:
            for i, p in enumerate(bf16_parts(w * float(coef) / (lay.ballast * c), lay.m)):
                out[i] += p
        else:
            out[lay.dim(d)] += w * float(coef) / c


def check_bf16(name: str, t: np.ndarray) -> None:
    """Every weight is a finite bf16 value (chipformer.q35.emit.check_bf16)."""
    t = np.ascontiguousarray(t, np.float32)
    bad = np.argwhere(((t.view(np.uint32) & 0xFFFF) != 0) | ~np.isfinite(t))
    if len(bad):
        i = tuple(bad[0])
        raise ValueError(f"{name}: weight {t[i]!r} at {i} is not a bf16 value")


# --- the configuration ----------------------------------------------------------------------


def text_config(c, lay: Layout) -> dict:
    n_chip = c.n_layers()
    return dict(BASE, hidden_size=lay.hidden, vocab_size=N_VOCAB, num_hidden_layers=n_chip + 1,
                layer_types=["linear_attention"] * n_chip + ["full_attention"],
                linear_num_key_heads=c.target.key_heads,
                linear_num_value_heads=c.target.value_heads,
                intermediate_size=c.target.intermediate, eos_token_id=S.tok("HALT"),
                pad_token_id=S.PAD, bos_token_id=S.PAD)


def tensor_shapes(tc: dict) -> list[tuple[str, tuple]]:
    """Every tensor of the checkpoint, in file order (layer by layer, embedding, final norm)."""
    h, nk, nv, hd = (tc["hidden_size"], tc["linear_num_key_heads"], tc["linear_num_value_heads"],
                     HD)
    inter, kw = tc["intermediate_size"], tc["linear_conv_kernel_dim"]
    qkv = 2 * nk * hd + nv * hd
    out = []
    for L, kind in enumerate(tc["layer_types"]):
        p = f"model.layers.{L}."
        out += [(p + "input_layernorm.weight", (h,)), (p + "post_attention_layernorm.weight", (h,))]
        if kind == "linear_attention":
            out += [(p + "linear_attn.in_proj_qkv.weight", (qkv, h)),
                    (p + "linear_attn.in_proj_z.weight", (nv * hd, h)),
                    (p + "linear_attn.in_proj_a.weight", (nv, h)),
                    (p + "linear_attn.in_proj_b.weight", (nv, h)),
                    (p + "linear_attn.conv1d.weight", (qkv, 1, kw)),
                    (p + "linear_attn.dt_bias", (nv,)), (p + "linear_attn.A_log", (nv,)),
                    (p + "linear_attn.norm.weight", (hd,)),
                    (p + "linear_attn.out_proj.weight", (h, nv * hd))]
        else:
            nh, nkv, ahd = tc["num_attention_heads"], tc["num_key_value_heads"], tc["head_dim"]
            out += [(p + "self_attn.q_proj.weight", (nh * ahd * 2, h)),
                    (p + "self_attn.k_proj.weight", (nkv * ahd, h)),
                    (p + "self_attn.v_proj.weight", (nkv * ahd, h)),
                    (p + "self_attn.o_proj.weight", (h, nh * ahd)),
                    (p + "self_attn.q_norm.weight", (ahd,)),
                    (p + "self_attn.k_norm.weight", (ahd,))]
        out += [(p + "mlp.gate_proj.weight", (inter, h)), (p + "mlp.up_proj.weight", (inter, h)),
                (p + "mlp.down_proj.weight", (h, inter))]
    out += [("model.embed_tokens.weight", (tc["vocab_size"], h)), ("model.norm.weight", (h,))]
    return out


# --- the weights ----------------------------------------------------------------------------


def chip_layer(c, lay: Layout, tc: dict, L: int) -> dict:
    """The tensors of circuit layer L (float32 arrays of bf16 values), as emit_layers makes them
    with ``keep`` = the output code."""
    h, nk, nv = tc["hidden_size"], tc["linear_num_key_heads"], tc["linear_num_value_heads"]
    inter = tc["intermediate_size"]
    r = nv // nk
    assert r * nk == nv
    f32 = np.float32
    keep = set(c.keep)
    exit_level = max(n.level for n in c.nodes if n.name.startswith("exit."))
    assert all(n.level < exit_level for n in c.nodes if not n.name.startswith("exit."))
    one = Expr({ONE: 1})
    qkv = np.zeros((2 * nk * HD + nv * HD, h), f32)
    convw = np.zeros((2 * nk * HD + nv * HD, 1, tc["linear_conv_kernel_dim"]), f32)
    zw = np.zeros((nv * HD, h), f32)
    bw = np.zeros((nv, h), f32)
    aw = np.zeros((nv, h), f32)
    ow = np.zeros((h, nv * HD), f32)
    heads = [n for n in c.nodes if isinstance(n, Head) and n.level == 2 * L + 1]
    assert len(heads) <= nk, (L, len(heads))
    for j, n in enumerate(heads):
        vhs = n.vheads()
        assert len(vhs) <= r, (n.name, len(vhs), r)
        for base, chans in ((j * HD, n.q), (nk * HD + j * HD, n.k)):
            for i, ch in chans.items():
                row(ch.expr, lay, qkv[base + i])
                for lag, w in enumerate(ch.taps):
                    convw[base + i, 0, -1 - lag] = w
        for x, vhd in enumerate(vhs):
            vh = j * r + x  # value head x of key head j
            base = 2 * nk * HD + vh * HD
            for i, ch in vhd.v.items():
                row(ch.expr, lay, qkv[base + i])
                for lag, w in enumerate(ch.taps):
                    convw[base + i, 0, -1 - lag] = w
            # beta = sigmoid(128) = 1, or a write gate sigmoid(256 g - 128): 1 or exactly 0
            beta = Expr({ONE: 128}) if vhd.beta is None else 256 * vhd.beta - 128 * one
            row(beta, lay, bw[vh])
            row(vhd.a, lay, aw[vh])
            for col, e in vhd.z.items():
                row(e, lay, zw[vh * HD + col])
            for col, d, w in vhd.out:
                ow[lay.dim(d), vh * HD + col] += w
    units = []  # (coef, a, b, output dim)
    for n in c.nodes:
        if isinstance(n, FFN) and n.level == 2 * L + 2:
            units += [(coef, a, b, n.out) for coef, a, b in n.terms]
    exiting = exit_level == 2 * L + 2
    if exiting:  # the exit clears every chip dim but the output code
        units += [(-1, Expr({o: 1}), one, o) for o in range(1, len(c.dims)) if o not in keep]
    assert len(units) + (lay.m if exiting else 0) <= inter, (L, len(units), inter)
    gate = np.zeros((inter, h), f32)
    up = np.zeros((inter, h), f32)
    down = np.zeros((h, inter), f32)
    for u, (coef, a, b, d) in enumerate(units):
        row(b, lay, gate[u], K)
        row(a, lay, up[u], 1 / K)
        down[lay.dim(d), u] = float(coef)
    if exiting:  # ... and the ballast: -B per ballast dim
        for i in range(lay.m):
            u = len(units) + i
            row(Expr({ONE: K}), lay, gate[u])
            up[u, i] = (lay.ballast / K) / (lay.ballast * lay.scale)
            down[i, u] = -1.0
    p = f"model.layers.{L}."
    return {
        p + "input_layernorm.weight": np.zeros(h, f32),
        p + "post_attention_layernorm.weight": np.zeros(h, f32),
        p + "linear_attn.in_proj_qkv.weight": qkv,
        p + "linear_attn.in_proj_z.weight": zw,
        p + "linear_attn.in_proj_a.weight": aw,
        p + "linear_attn.in_proj_b.weight": bw,
        p + "linear_attn.conv1d.weight": convw,
        p + "linear_attn.dt_bias": np.full(nv, -110.0, f32),
        p + "linear_attn.A_log": np.zeros(nv, f32),
        p + "linear_attn.norm.weight": np.ones(HD, f32),
        p + "linear_attn.out_proj.weight": ow,
        p + "mlp.gate_proj.weight": gate,
        p + "mlp.up_proj.weight": up,
        p + "mlp.down_proj.weight": down,
    }


def embedding_matrix(c, lay: Layout) -> np.ndarray:
    """(N_VOCAB, hidden): every chip token's features and the ballast; PAD and the padding rows
    are 0. Within a large family the features are affine in the payload bits (each is a family
    flag, a payload bit, a code bit CODE_AMP * (2 bit - 1) or 0), so a family's rows are its
    first row plus the bits times per-bit deltas; small families are evaluated token by token.
    Spot checks against the IR's own feature functions."""
    feats = sorted(c.token_fns)
    cols = np.array([lay.dim(d) for d in feats])

    def vec(t):
        return np.array([float(c.feature_value(d, t)) for d in feats])

    e = np.zeros((N_VOCAB, lay.hidden), np.float32)
    rng = np.random.default_rng(0)
    for name, (base, size) in S.FAMILIES.items():
        if size <= 4096:
            rows = np.stack([vec(base + i) for i in range(size)])
        else:
            nb = (size - 1).bit_length()
            v0 = vec(base)
            deltas = np.stack([vec(base + (1 << i)) - v0 for i in range(nb)])
            bits = (np.arange(size)[:, None] >> np.arange(nb)) & 1
            rows = v0 + bits @ deltas
            for t in [size - 1, size // 3] + rng.integers(0, size, 64).tolist():
                assert (vec(base + t) == rows[t]).all(), (name, t)
        e[base:base + size, cols] = rows
        e[base:base + size, :lay.m] = lay.ballast
    for t in S.CONTROLS.values():
        e[t, cols] = vec(t)
        e[t, :lay.m] = lay.ballast
    check_bf16("embedding", e)
    return e


def final_norm(c, lay: Layout) -> np.ndarray:
    norm = np.full(lay.hidden, -1.0, np.float32)  # (1 + w) = 0: the head sees only the code
    norm[[lay.dim(d) for d in c.keep]] = CODE_GAIN
    return norm


def tensors(c, lay: Layout, tc: dict):
    """(name, float32 array) of every tensor, in tensor_shapes order, one layer at a time."""
    n_chip = c.n_layers()
    for L in range(n_chip):
        yield from chip_layer(c, lay, tc, L).items()
    for name, shape in tensor_shapes(tc):
        if name.startswith(f"model.layers.{n_chip}."):  # the full-attention layer: zeros
            yield name, np.zeros(shape, np.float32)
    yield "model.embed_tokens.weight", embedding_matrix(c, lay)
    yield "model.norm.weight", final_norm(c, lay)


# --- metadata -------------------------------------------------------------------------------


def head_layout(c) -> dict:
    """Table -> [layer, key head, value heads]: where the snapshot's rows go (value head index
    within the layer = key head * value_heads_per_key_head + x)."""
    return {k: list(v) for k, v in snapshot.head_slots(c).items()}


def metadata(c, lay: Layout, tc: dict, nonzeros: dict | None = None) -> dict:
    vb, rb, hb = core.ram_layout(c.target.value_heads // c.target.key_heads)
    return {
        "machine": "qwendoom: RV32IM + Zba (DOOM's playsim) in Qwen3.5 Gated DeltaNet layers",
        "vocab": S.VOCAB, "pad": S.PAD, "n_vocab": N_VOCAB,
        "families": {k: list(v) for k, v in S.FAMILIES.items()}, "controls": S.CONTROLS,
        "final": sorted(S.FINAL), "faults": S.FAULTS,
        "code": {
            "dims": [lay.dim(d) for d in c.keep],
            "nfam": core.NFAM, "npay": core.NPAY, "out_names": core.OUT_NAMES,
            "code_amp": core.CODE_AMP, "kappa": core.KAPPA, "gain": CODE_GAIN,
            "decoding": "greedy argmax of the tied head; code bit i of a token = family id bit "
                        "i (i < nfam, the family's index in out_names), then payload bit "
                        "i - nfam; its embedding row holds code_amp * (2 bit - 1) on dims[i]"},
        "layout": {"hidden": lay.hidden, "ballast_dims": lay.m,
                   "log2_ballast": lay.log2_ballast, "ir_dims": len(c.dims),
                   "ir_dim_index": "residual index = ballast_dims + IR dim"},
        "heads": head_layout(c),
        "value_heads_per_key_head": c.target.value_heads // c.target.key_heads,
        "state_rows": {
            "a_ref": A_REF, "a_bit": A_BIT, "ref_cols": list(REF_COLS),
            "tables": {
                "pc": "row 0: ballast; anchor (PC word index) bit i at column 2 + i",
                "count/j/vcnt": "row 0: a_bit * 2^-k at column 2 (k = n for count, 0 for j "
                                "and vcnt); row 1: ballast",
                "regs.rs1/rs2/rd": "row r (1..31): ballast; bit i of x_r at column "
                                   "core.reg_columns()[i][0]",
                "quo*/lag": "row 0: ballast only",
                "rom*/drom*": "byte tables (core.QdCore.byte_table) of the word index w "
                              "((pc - rom base) >> 2 of the pre-decoded ROM words, ref.rom_words; "
                              "(addr - drom base) >> 2): table h = w >> 12, value head "
                              "(w >> 10) & 3, row (w >> 4) & 63 | (w & 8) << 3, byte b at column "
                              "2 + 4 * (w & 15) + b (values 0..255); every row carries the "
                              "ballast, zero bytes included",
                "ram*": f"word index w = (addr >> 2): value head bits {vb}, row bits {rb}, key "
                        f"head bits {hb} of w; the word's bits at columns 2..33; one row per "
                        "nonzero word"}},
        "address_map": {"ram": [S.RAM_BASE, S.RAM_SIZE], "drom": [S.DROM_BASE, S.DROM_SIZE],
                        "rom": [S.ROM_BASE, S.ROM_SIZE], "host": [S.HOST_BASE, S.HOST_SIZE],
                        "sp_init": S.SP_INIT},
        "io": {"input": "csrr rd, time (CSR 0xC01): the circuit predicts F0(0); the host "
                        "injects the next input word's value record (F0 | V F1 | V V F2)",
               "yield": f"ecall a7 = 1, a0 = {S.YIELD}: BYTE({S.YIELD})",
               "putchar": "ecall a7 = 1: BYTE(a0 & 511)", "exit": "ecall a7 = 93: HALT"},
        "run": ["reset every recurrent state and conv window to 0",
                f"step PAD (token {S.PAD}, a zero embedding row: every state stays 0)",
                "write the snapshot rows (qwendoom.snapshot.state_rows): recurrent "
                "state[layer][value head][row, :] = vec (layer 0-based, value head within the "
                "layer); every other row stays 0",
                f"step GO (token {S.tok('GO')}); its prediction is the first token of the "
                "record at pc; then feed each predicted (or host-injected) token"],
        "circuit": c.stats(), "nonzeros": nonzeros or {},
        "config": {k: tc[k] for k in ("hidden_size", "num_hidden_layers",
                                      "linear_num_key_heads", "linear_num_value_heads",
                                      "intermediate_size")},
    }


# --- safetensors (bf16), streamed -----------------------------------------------------------


def write_checkpoint(out: Path, c=None) -> dict:
    """Emit everything into out/; returns a summary (sizes, nonzeros, hashes)."""
    c = c or core.build()
    lay = pick_layout(len(c.dims))
    tc = text_config(c, lay)
    out.mkdir(parents=True, exist_ok=True)
    shapes = tensor_shapes(tc)
    header, off = {}, 0
    for name, shape in shapes:
        n = 2 * int(np.prod(shape))
        header[name] = {"dtype": "BF16", "shape": list(shape), "data_offsets": [off, off + n]}
        off += n
    header["__metadata__"] = {"format": "pt"}
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    nz, params = {}, 0
    path = out / "model.safetensors"
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(hb)))
        fh.write(hb)
        order = iter(shapes)
        for name, t in tensors(c, lay, tc):
            want, shape = next(order)
            assert name == want and t.shape == tuple(shape), (name, want, t.shape, shape)
            check_bf16(name, t)
            u = np.ascontiguousarray(t, np.float32).view(np.uint32)
            fh.write((u >> 16).astype("<u2").tobytes())
            key = name.split(".")[2] if name.startswith("model.layers.") else name
            nz[key] = nz.get(key, 0) + int(np.count_nonzero(t))
            params += t.size
    assert path.stat().st_size == 8 + len(hb) + off
    chip_nz = sum(v for k, v in nz.items() if k.isdigit())
    nonzeros = {"chip_layers": chip_nz, "per_layer": {k: v for k, v in nz.items()
                                                      if k.isdigit()},
                "embedding": nz["model.embed_tokens.weight"], "total": sum(nz.values())}
    config = {"architectures": ["Qwen3_5ForCausalLM"], "model_type": "qwen3_5_text",
              "dtype": "bfloat16", **tc}
    (out / "config.json").write_text(json.dumps(config, indent=1) + "\n")
    (out / "generation_config.json").write_text(json.dumps(
        {"eos_token_id": S.tok("HALT"), "pad_token_id": S.PAD, "bos_token_id": S.PAD,
         "do_sample": False}, indent=1) + "\n")
    (out / "qwendoom.json").write_text(json.dumps(metadata(c, lay, tc, nonzeros), indent=1)
                                       + "\n")
    sums = []
    for f in ("config.json", "generation_config.json", "model.safetensors", "qwendoom.json"):
        h = hashlib.sha256()
        with open(out / f, "rb") as fh:
            for blk in iter(lambda: fh.read(1 << 24), b""):
                h.update(blk)
        sums.append(f"{h.hexdigest()}  {f}")
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    return {"dir": str(out), "bytes": path.stat().st_size, "params": params, "hidden": lay.hidden,
            "layers": tc["num_hidden_layers"], "nonzeros": nonzeros, "circuit": c.stats()}


def main() -> None:
    ap = argparse.ArgumentParser(description="Emit the qwendoom checkpoint")
    ap.add_argument("--out", default="weights")
    args = ap.parse_args()
    import time

    t = time.time()
    res = write_checkpoint(Path(args.out))
    print(json.dumps(res, indent=1))
    print(f"{time.time() - t:.0f} s")


if __name__ == "__main__":
    main()
