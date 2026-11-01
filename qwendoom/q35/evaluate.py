"""A faster ``chipformer.q35.ir.evaluate``: the same values and the same exactness checks.

Derived from Qwenputer's chipformer/q35/ir.py (MIT, (c) 2026 Divyansh Agrawal; see
LICENSE.chipformer). Upstream's evaluator gathers every linear form's columns from a row-major
(tokens x dims) matrix, one form at a time; with qwendoom's 186 heads (128 RAM heads of 256 query
and key channels each) that dominates. Here the residual is kept dim-major (each dim's values
contiguous), every node's forms are one sparse matrix applied at once, and the per-term checks
run on whole blocks. The arithmetic of each value is upstream's (linear forms and sums of
integers and powers of 2 are exact in float64), and tests/test_core.py checks the result equals
``ir.evaluate``'s bit for bit.
"""

import numpy as np
import scipy.sparse as sp

from chipformer.dsl.graph import ONE
from chipformer.q35.ir import (A_BIT, A_REF, BYTE_LEVELS, DECAY, DECAY_LEVELS, FFN, K,
                               REF_COLS, RESIDUAL_MAX, ExactnessError, Graph, bf16,
                               kernel_check, require, silu_ok)


def is_bf16(x: np.ndarray) -> np.ndarray:
    """ir.is_bf16 (x is a finite bf16 value) without rounding: x is a float32 value whose low
    16 bits are 0."""
    x = np.asarray(x, dtype=np.float64)
    f = x.astype(np.float32)
    return (f == x) & ((f.view(np.uint32) & 0xFFFF) == 0) & np.isfinite(x)


def _forms(exprs, n_dims: int) -> sp.csr_matrix:
    """The linear forms as rows of a sparse (len(exprs) x n_dims) matrix."""
    rows, cols, vals = [], [], []
    for i, e in enumerate(exprs):
        for d, c in e.terms.items():
            rows.append(i)
            cols.append(d)
            vals.append(float(c))
    return sp.csr_matrix((vals, (rows, cols)), shape=(len(exprs), n_dims))


def _sig(W: sp.csr_matrix, *extra) -> bytes:
    """A content key of a block of forms (equal blocks in different heads share results)."""
    return b"|".join([W.indptr.tobytes(), W.indices.tobytes(), W.data.tobytes(),
                      *(np.asarray(e).tobytes() for e in extra)])


class _Ctx:
    """One evaluate call: per-dim write versions and the results of blocks shared by heads."""

    def __init__(self, n_dims: int, shared: set):
        self.ver = np.zeros(n_dims, np.int64)
        self.shared = shared  # signatures of blocks that occur more than once
        self.cache: dict = {}

    def key(self, sig: bytes, dims: np.ndarray, *extra):
        if sig not in self.shared:
            return None
        return (sig, self.ver[dims].tobytes(), *extra)


class _Chans:
    """A block of conv channels (Chan): their forms and taps."""

    def __init__(self, chans, n_dims: int):
        chans = list(chans)
        self.W = _forms([c.expr for c in chans], n_dims)
        self.taps = np.array([c.taps for c in chans], dtype=np.float64).reshape(len(chans), 4)
        self.lagged = self.taps[:, 1:].any(1)
        self.sig = _sig(self.W, self.taps)
        self.dims = np.unique(self.W.indices)

    def __call__(self, mt, check, carry, key, ctx: "_Ctx") -> np.ndarray:
        """silu(conv(projection)) of every channel: (channels, tokens)."""
        use_carry = carry is not None and self.lagged.any()
        prev = carry.get(key) if use_carry else None
        ck = ctx.key(self.sig, self.dims, None if prev is None else prev.tobytes())
        if ck is not None and ck in ctx.cache:
            out, carry_out = ctx.cache[ck]
            if use_carry:
                carry[key] = carry_out
            return out
        x = np.asarray(self.W @ mt) if self.W.shape[0] else np.zeros((0, mt.shape[1]))
        if check:
            require(is_bf16(x), "projection", x)
        n = x.shape[1]
        p = 0
        xx = x
        carry_out = None
        if use_carry:
            if prev is not None:  # the previous chunk's last 3 values (lagged channels)
                xx = np.concatenate([prev, x], 1)
                p = prev.shape[1]
            carry_out = xx[:, -3:].copy()
            carry[key] = carry_out
        out = np.zeros(xx.shape)
        for lag in range(4):
            w = self.taps[:, lag]
            if w.any():
                out[:, lag:] += w[:, None] * xx[:, :xx.shape[1] - lag]
        out = out[:, p:]  # (a channel without lag taps reads only its own position)
        if check:
            require(is_bf16(out) & silu_ok(out), "conv output (silu input)", out)
        assert out.shape[1] == n
        out = np.maximum(out, 0.0)
        if ck is not None:
            ctx.cache[ck] = (out, carry_out)
        return out


class _Compiled:
    """Per-node sparse forms of a graph (built once, reused by every evaluate call)."""

    def __init__(self, g: Graph):
        D = len(g.dims)
        self.nodes = []
        for node in g.nodes:
            if isinstance(node, FFN):
                self.nodes.append({
                    "B": _forms([b for _, _, b in node.terms], D),
                    "A": _forms([a for _, a, _ in node.terms], D),
                    "coef": np.array([float(c) for c, _, _ in node.terms])})
                continue
            vhs = []
            for vh in node.vheads():
                zexprs, zidx, seen = [], {}, {}
                for col, e in vh.z.items():
                    if id(e) not in seen:
                        seen[id(e)] = len(zexprs)
                        zexprs.append(e)
                    zidx[col] = seen[id(e)]
                misc = [vh.a] + ([] if vh.beta is None else [vh.beta])
                out = (np.array([c for c, _, _ in vh.out], np.int64),
                       np.array([d for _, d, _ in vh.out], np.int64),
                       np.array([w for _, _, w in vh.out], np.float64),
                       np.array([zidx.get(c, -1) for c, _, _ in vh.out], np.int64))
                zf = _forms(zexprs, D)
                vhs.append({"cols": list(vh.v), "v": _Chans(vh.v.values(), D),
                            "z": zf, "zsig": _sig(zf), "zdims": np.unique(zf.indices),
                            "zidx": zidx, "misc": _forms(misc, D), "out": out})
            self.nodes.append({"q": _Chans(node.q.values(), D), "k": _Chans(node.k.values(), D),
                               "vh": vhs})
        from collections import Counter
        sigs = Counter()
        for cn in self.nodes:
            if "vh" in cn:
                sigs.update([cn["q"].sig, cn["k"].sig])
                for cv in cn["vh"]:
                    sigs.update([cv["v"].sig, cv["zsig"]])
        self.shared = {k for k, v in sigs.items() if v > 1}


def compiled(g: Graph) -> _Compiled:
    cg = getattr(g, "_fast_eval", None)
    if cg is None or len(cg.nodes) != len(g.nodes):
        cg = _Compiled(g)
        g._fast_eval = cg
    return cg


def evaluate(g: Graph, tokens: list[int], check: bool = True,
             carry: dict | None = None, kernel: bool = False) -> np.ndarray:
    """``ir.evaluate``: the residual after every node, (n_tokens, n_dims) (a view of a dim-major
    array). ``carry`` evaluates a long stream chunk by chunk (pass the same dict, empty at the
    start); ``kernel`` also replays every head read the way the HF torch kernels compute it."""
    cg = compiled(g)
    n = len(tokens)
    mt = np.zeros((len(g.dims), n))
    toks = np.asarray(tokens, dtype=np.int64)
    chip = toks >= 0
    mt[ONE] = chip
    uniq, inv = np.unique(toks, return_inverse=True)
    for d in g.token_fns:
        col = np.array([float(g.feature_value(d, int(u))) if u >= 0 else 0.0 for u in uniq])[inv]
        if check:
            require(is_bf16(col), f"token feature {g.dims[d].name}", col)
        mt[d] = col
    ctx = _Ctx(len(g.dims), cg.shared)
    for i, (node, cn) in enumerate(zip(g.nodes, cg.nodes)):
        try:
            if isinstance(node, FFN):
                _eval_ffn(node, cn, mt, check)
                ctx.ver[node.out] += 1
            else:
                _eval_head(node, cn, mt, chip, check, carry, i, kernel, ctx)
        except ExactnessError as e:
            raise ExactnessError(f"{node.name}: {e}") from None
    if check and n:
        big = np.abs(mt[1:]).max(1)
        if len(big) and big.max() > RESIDUAL_MAX:
            d = int(big.argmax()) + 1
            raise ExactnessError(f"{g.dims[d].name}: residual value {big[d - 1]!r} would move "
                                 f"the norms (> {RESIDUAL_MAX})")
    return mt.T


def _eval_ffn(node: FFN, cn: dict, mt: np.ndarray, check: bool) -> None:
    bv = np.asarray(cn["B"] @ mt)
    av = np.asarray(cn["A"] @ mt)
    coef = cn["coef"]
    ideal = np.zeros(mt.shape[1])
    for t in range(len(coef)):  # upstream's order of accumulation
        ideal += coef[t] * av[t] * np.maximum(bv[t], 0.0)
    if check:
        gate = bf16(K * bv)
        require(silu_ok(gate), "gate outside silu's exact range", gate)
        up = bf16(av / K)
        prod = bf16(np.maximum(gate, 0.0) * up)
        hw = np.zeros(mt.shape[1])
        for t in range(len(coef)):
            hw += coef[t] * prod[t]
        hw = bf16(hw)
        require(hw == ideal, "bf16 output differs from exact", np.stack([hw, ideal], 1)[:, 0])
    if node.into:
        ideal = mt[node.out] + ideal
        if check:
            require(is_bf16(ideal), "residual add", ideal)
    mt[node.out] = ideal


def _eval_head(node, cn: dict, mt: np.ndarray, chip: np.ndarray, check: bool,
               carry: dict | None, ni: int, kernel: bool, ctx: _Ctx) -> None:
    n = mt.shape[1]
    q = cn["q"](mt, check, carry, (ni, "q"), ctx).T
    k = cn["k"](mt, check, carry, (ni, "k"), ctx).T
    qi, ki = list(node.q), list(node.k)
    if check:
        nz = (k > 0).sum(1)
        require(nz <= 1, "key must be one-hot", nz)
        require(~((nz > 0) & ~chip), "key on a text token", nz)
        for t in np.nonzero((q > 0).sum(1) > 1)[0]:
            vals = q[t][q[t] > 0]
            require(vals == vals[0], "query must be multi-hot", vals)
    has_key = k.any(1) if k.shape[1] else np.zeros(n, dtype=bool)
    karg = np.argmax(k, 1) if k.shape[1] else np.zeros(n, dtype=np.int64)
    key_row = [ki[int(karg[t])] if has_key[t] else None for t in range(n)]
    has_q = q.any(1) if q.shape[1] else np.zeros(n, dtype=bool)
    for x_, (vh, cv) in enumerate(zip(node.vheads(), cn["vh"])):
        v = np.zeros((n, node.n_cols))
        if cv["cols"]:
            v[:, cv["cols"]] = cv["v"](mt, check, carry, (ni, "v", x_), ctx).T
        misc = np.asarray(cv["misc"] @ mt)
        a = misc[0]
        zk = ctx.key(cv["zsig"], cv["zdims"], check)
        if zk is not None and zk in ctx.cache:
            zall = ctx.cache[zk]
        else:
            zall = np.asarray(cv["z"] @ mt) if cv["z"].shape[0] else np.zeros((0, n))
            if check:
                require(is_bf16(zall) & silu_ok(zall), "gate z", zall)
            if zk is not None:
                ctx.cache[zk] = zall
        beta = np.ones(n) if vh.beta is None else misc[1]
        writes = has_key & (beta > 0)
        if check:
            require(is_bf16(a) & np.isin(a, list(DECAY)), "decay selector a", a)
            require((beta == 1.0) | ((beta <= 0) & (beta == np.round(beta))), "write gate beta",
                    beta)
            written = v[writes]
            if len(written):
                data = np.delete(written, REF_COLS, 1)
                require(np.isin(data, [0.0, A_BIT]), "state row data column", data)
                ref = written[:, list(REF_COLS)]
                require((ref == 0) | (ref == A_REF), "state row ballast column", ref)
        if carry is not None and (ni, "S", x_) in carry:
            state = carry[ni, "S", x_]
        else:
            state = {r: np.asarray(row, dtype=np.float64).copy()
                     for r, row in ((node.init or [{}] * (x_ + 1))[x_] if node.init
                                    and x_ < len(node.init) else {}).items()}
            if check and state:
                rows = np.stack(list(state.values()))
                data = np.delete(rows, REF_COLS, 1)
                ok = np.isin(data, BYTE_LEVELS if node.levels == "byte" else
                             DECAY_LEVELS if node.levels == "decay" else [0.0, A_BIT])
                require(ok, f"initial state data ({node.levels})", data)
                ref = rows[:, list(REF_COLS)]
                require((ref == 0) | (ref == A_REF), "initial state ballast column", ref)
        out = np.zeros((n, node.n_cols))
        decay = np.select([a == 0.0, a == 110.0, a == 238.0], [1.0, 0.5, 0.0], np.nan)
        events = np.nonzero((decay != 1.0) | writes | has_q)[0]
        zero_row = np.zeros(node.n_cols)
        for t in events:
            if decay[t] == 0.0:
                state.clear()
            elif decay[t] == 0.5:
                state = {s: row * 0.5 for s, row in state.items()}
            if writes[t]:
                state[key_row[t]] = v[t].copy()
            if has_q[t]:
                hot = [(qi[j], q[t, j]) for j in np.nonzero(q[t] > 0)[0]]
                o = sum((state[s] for s, _ in hot if s in state), zero_row)
                ref = (o[:2] ** 2).sum()  # REF_COLS
                if ref == 0:
                    if check:
                        require(o == 0, "read without ballast", o)
                    continue
                if check:
                    data = (o[2:] ** 2).sum()
                    if not data <= ref * 2.0**-26:
                        require(np.array([False]), "data not absorbed by ballast",
                                np.array([data / ref]))
                out[t] = o / np.sqrt(ref / node.n_cols)
                if kernel and check:
                    rows = [(state[s], j) for j, (s, _) in enumerate(hot) if s in state]
                    kernel_check(rows, q[t], node.n_cols, out[t])
        if carry is not None:
            carry[ni, "S", x_] = state
        if not vh.out:
            continue
        # only positions that read a row have outputs (elsewhere every output is exactly 0)
        oc, od, ow, oz = cv["out"]
        pos = np.nonzero(out.any(1))[0]
        if not vh.into:
            mt[od] = 0.0
        if not len(pos):
            continue
        normed = out[pos][:, oc].T  # (outputs, positions)
        zz = np.where(oz[:, None] >= 0, zall[np.maximum(oz, 0)][:, pos] if len(zall) else 0.0,
                      0.0)
        contrib = normed * np.maximum(zz, 0.0) * ow[:, None]
        if check:
            small = np.abs(normed) < node.tiny
            require(is_bf16(normed) | small, "normed column", normed)
            require(is_bf16(contrib) | small, "head output column", contrib)
        if vh.into:
            # upstream adds the outputs one by one and checks every partial sum: the partial
            # sums per dim, in the outputs' order, then the totals
            order = np.argsort(od, kind="stable")
            ds, cs = od[order], contrib[order]
            first = np.r_[True, ds[1:] != ds[:-1]]
            start = np.maximum.accumulate(np.where(first, np.arange(len(ds)), 0))
            run = np.cumsum(cs, 0)
            before = np.where(start[:, None] > 0, run[np.maximum(start - 1, 0)], 0.0)
            partial = mt[ds][:, pos] + (run - before)
            if check:
                require(is_bf16(partial), "head output sum", partial)
            last = np.r_[ds[1:] != ds[:-1], True]
            mt[np.ix_(ds[last], pos)] = partial[last]
        else:
            mt[np.ix_(od, pos)] = contrib
        ctx.ver[od] += 1
