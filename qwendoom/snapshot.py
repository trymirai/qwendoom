"""The machine state as the circuit's head states: what a resync loads into the model.

    rows = state_rows(ram, drom, rom, regs, pc)   # numpy, < 1 s for full images
    # rows.layer, rows.vhead, rows.row: int32 (N,); rows.vec: float32 (N, 128); rows.first: GO

A run (the engine, stock transformers, the IR alike):

  1. reset: every recurrent state and conv window 0;
  2. step PAD (token id spec.PAD, a zero embedding row: it leaves every state and window at 0);
  3. write each row: recurrent state[layer][value head][row, :] = vec (all other rows stay 0);
  4. step rows.first (GO); its prediction is the first token of the record at pc; then feed
     each predicted (or host-injected) token in turn.

``layer`` counts the checkpoint's layers from 0 and ``vhead`` is the value head within the layer
(key head j holds value heads 4 j .. 4 j + 3), as in config.json's linear attention shapes. Which
table sits where comes from the built circuit (``head_map``; also in weights/qwendoom.json).

State rows (ir.A_REF = 2^24 in the two ballast columns 0, 1 of every row that carries data;
ir.A_BIT = 128 for a set bit):

  pc         row 0: the anchor (PC word index) in bits, columns 2..17
  count      rows 0 / 1: A_BIT 2^-n in column 2 / the ballast (n: commits since the anchor)
  j, vcnt    the same with n = 0
  regs.*     rows 1..31: register x_r's bits (core.reg_columns()[i][0]), three read ports alike
  quo*, lag  row 0: the ballast only
  rom*, drom*  byte tables: every row of every value head; byte values 0..255 (core.byte_table)
  ram*       one row per nonzero RAM word: its bits in columns 2..33 (core.ram_layout)
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from chipformer.q35.ir import A_BIT, A_REF, HD, Head
from qwendoom import core
from qwendoom import ref as RF
from qwendoom import spec as S


@dataclass
class Rows:
    """State rows in the emitted checkpoint's coordinates (see the module doc)."""

    layer: np.ndarray  # int32 (N,)
    vhead: np.ndarray  # int32 (N,): value head index within the layer
    row: np.ndarray  # int32 (N,): key dim (row of the 128 x 128 state)
    vec: np.ndarray  # float32 (N, 128)
    first: int = S.tok("GO")  # the token to step after loading (its prediction starts the run)

    def __len__(self) -> int:
        return len(self.row)

    def as_dict(self) -> dict:
        """(layer, value head) -> {row: vec} (for writing into an HF cache)."""
        out: dict = {}
        for L, vh, r, v in zip(self.layer.tolist(), self.vhead.tolist(), self.row.tolist(),
                               self.vec):
            out.setdefault((L, vh), {})[r] = v
        return out


def _ballast(n: int) -> np.ndarray:
    rows = np.zeros((n, HD), np.float32)
    rows[:, 0:2] = A_REF
    return rows


def _bit_rows(words: np.ndarray, cols) -> np.ndarray:
    """Ballast rows holding the 32 bits of each word in the given columns."""
    rows = _ballast(len(words))
    bits = (words[:, None].astype(np.int64) >> np.arange(32)) & 1
    rows[:, cols] = A_BIT * bits
    return rows


def _byte_table(words: np.ndarray, prefix: str, r: int) -> dict:
    """name -> per value head (rows, vecs) of a byte table (core.QdCore.byte_table's layout)."""
    per = core.TABLE_WORDS
    n_heads = len(words) // (per * r)
    blk = words.reshape(n_heads, r, 64, 2, 8).astype(np.int64)  # head, vhead, pair, half, word
    vec = np.zeros((n_heads, r, 2, 64, HD), np.float32)  # head, vhead, half (row block), pair
    vec[..., 0:2] = A_REF
    for half in range(2):
        for b in range(4):
            cols = 2 + 32 * half + 4 * np.arange(8) + b
            vec[:, :, half, :, cols] = np.moveaxis((blk[:, :, :, half, :] >> 8 * b) & 0xFF, -1, 0)
    rows = np.arange(128)
    vec = vec.reshape(n_heads, r, 128, HD)  # row = pair + 64 * half
    return {f"{prefix}{h}": [(rows, vec[h, x]) for x in range(r)] for h in range(n_heads)}


def _counter(n: int) -> list:
    v = np.zeros((2, HD), np.float32)
    v[0, 2] = A_BIT * 2.0**-n
    v[1, 0:2] = A_REF
    return [(np.arange(2), v)]


def tables(ram, drom, rom, regs, anchor: int, n: int = 0, r: int = core.VHEADS,
           rom_decoded: bool = False) -> dict:
    """Table name -> per value head (rows int array, vecs float32 (k, 128)): the IR's view.

    ram: RAM_SIZE bytes (or RAM_WORDS words); drom: up to DROM_SIZE bytes; rom: up to ROM_SIZE
    bytes of raw instructions (or words; pre-decoded already with rom_decoded); regs: 32 ints;
    anchor: the PC word index (pc = ROM_BASE + 4 (anchor + n))."""
    def words(img, n_words, name):
        a = (np.frombuffer(bytes(img).ljust(4 * n_words, b"\0"), "<u4")
             if isinstance(img, (bytes, bytearray, memoryview)) else
             np.asarray(img, dtype=np.int64).astype(np.uint32))
        if len(a) < n_words:
            a = np.concatenate([a, np.zeros(n_words - len(a), np.uint32)])
        assert len(a) == n_words, f"{name} is larger than its region"
        return a

    ramw = words(ram, S.RAM_WORDS, "ram")
    dromw = words(drom, S.DROM_WORDS, "drom")
    romw = words(rom, S.ROM_WORDS, "rom")
    if not rom_decoded:
        romw = RF.rom_words(romw)
    out = {}
    pc = np.zeros((1, HD), np.float32)
    pc[0, 0:2] = A_REF
    pc[0, 2:18] = A_BIT * ((anchor >> np.arange(16)) & 1)
    out["pc"] = [(np.zeros(1, np.int64), pc)]
    out["count"] = _counter(n)
    out["j"] = _counter(0)
    out["vcnt"] = _counter(0)
    regw = np.array([int(v) & 0xFFFFFFFF for v in regs], dtype=np.int64)
    regw[0] = 0
    cols = [c[0] for _, c in sorted(core.reg_columns().items())]
    reg_rows = (np.arange(1, 32), _bit_rows(regw[1:], cols))
    for p in ("rs1", "rs2", "rd"):
        out[f"regs.{p}"] = [reg_rows]
    for h in range(8 // r):
        out[f"quo{h}"] = [(np.zeros(1, np.int64), _ballast(1)) for _ in range(r)]
    if r == 4:
        out["lag"] = [(np.zeros(1, np.int64), _ballast(1))]
    out.update(_byte_table(romw, "rom", r))
    out.update(_byte_table(dromw, "drom", r))
    # RAM: one row per nonzero word, at (key head, value head, row) of core.ram_layout
    vb, rb, hb = core.ram_layout(r)
    wi = np.nonzero(ramw)[0]

    def gather(bits):
        return sum(((wi >> b) & 1) << k for k, b in enumerate(bits))

    x, row, h = gather(vb), gather(rb), gather(hb)
    vecs = _bit_rows(ramw[wi], np.arange(2, 34))
    order = np.lexsort((row, x, h))
    wi_h, wi_x, wi_r, vecs = h[order], x[order], row[order], vecs[order]
    bounds = np.searchsorted(wi_h * r + wi_x, np.arange((1 << len(hb)) * r + 1))
    for hh in range(1 << len(hb)):
        out[f"ram{hh}"] = [(wi_r[bounds[hh * r + xx]:bounds[hh * r + xx + 1]],
                            vecs[bounds[hh * r + xx]:bounds[hh * r + xx + 1]])
                           for xx in range(r)]
    return out


def machine_tables(m: "RF.Machine", r: int = core.VHEADS) -> dict:
    """``tables`` of a reference machine's state (any n, as at the start of a record)."""
    return tables(m.ram, m.drom, m.rom, m.x, m.anchor, m.n, r)


def load_ir(c, tabs: dict) -> None:
    """Set the circuit's head ``init`` states (the IR evaluator's snapshot) from ``tables``."""
    for name, node in c.tables.items():
        vhs = tabs.get(name)
        assert vhs is not None, f"no snapshot rows for table {name}"
        node.init = [{int(r): v.astype(np.float64) for r, v in zip(rows, vecs)}
                     for rows, vecs in vhs]


def snapshot(c, m: "RF.Machine") -> None:
    """Load a reference machine's state into the circuit's heads (the IR's snapshot)."""
    load_ir(c, machine_tables(m, c.target.value_heads // c.target.key_heads))


def head_slots(c) -> dict:
    """Table name -> (layer, key head, value heads) in the emitted checkpoint."""
    out = {}
    for L in range(c.n_layers()):
        heads = [n for n in c.nodes if isinstance(n, Head) and n.level == 2 * L + 1]
        for j, n in enumerate(heads):
            out[n.name] = (L, j, len(n.vheads()))
    return out


@lru_cache(maxsize=1)
def head_map() -> dict:
    """head_slots of the default circuit (built once per process)."""
    return head_slots(core.build())


def emitted(tabs: dict, heads: dict | None = None, r: int = core.VHEADS) -> Rows:
    """``tables`` in checkpoint coordinates (layer, value head, row)."""
    heads = heads or head_map()
    Ls, vhs, rows, vecs = [], [], [], []
    for name, per in tabs.items():
        L, j, nvh = heads[name]
        assert len(per) <= nvh, name
        for x, (rr, vv) in enumerate(per):
            if len(rr):
                Ls.append(np.full(len(rr), L, np.int32))
                vhs.append(np.full(len(rr), j * r + x, np.int32))
                rows.append(np.asarray(rr, np.int32))
                vecs.append(vv)
    return Rows(np.concatenate(Ls), np.concatenate(vhs), np.concatenate(rows),
                np.concatenate(vecs).astype(np.float32, copy=False))


def state_rows(ram, drom, rom, regs, pc: int, heads: dict | None = None, n: int = 0) -> Rows:
    """The model's state rows for a machine at pc (n = 0: the start of a record) from raw
    images: ram (RAM_SIZE bytes), drom (<= DROM_SIZE bytes), rom (<= ROM_SIZE bytes of raw
    instructions from ROM_BASE), regs (32 ints; x0 ignored). ``heads``: head_slots of the
    checkpoint (default: of core.build(), cached; or weights/qwendoom.json's "heads")."""
    assert RF.in_rom(pc), f"pc {pc:#x} outside the ROM"
    anchor = (pc - S.ROM_BASE) >> 2
    return emitted(tables(ram, drom, rom, regs, anchor - n, n), heads)


def machine_rows(m: "RF.Machine", heads: dict | None = None) -> Rows:
    """state_rows of a reference machine (any n)."""
    return emitted(machine_tables(m), heads)
