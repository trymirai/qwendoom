"""Measurements of the circuit for docs/core.md:

    .venv/bin/python -m qwendoom.core_measure [--variants] [--tokens] > build/core/measure.json

Depth, IR dims, heads and MLP units per sublayer, the emitted hidden size and nonzero weights
(qwendoom.weights.chip_layer), tokens per instruction on the C programs; ``--variants`` also
measures the same circuit with 2 value heads per key head (upstream's shape) and upstream's own
core (chipformer.rv.core: digit file, 32 KB RAM, 128 KB DROM), the "before" of the fork.

The approach follows Qwenputer's chipformer/rv/measure.py (MIT, (c) 2026 Divyansh Agrawal; see
LICENSE.chipformer).
"""

import argparse
import json
from collections import Counter

import numpy as np

from chipformer.q35.ir import FFN, Head, Target
from qwendoom import core, weights


def per_sublayer(c) -> list[dict]:
    units, heads, names = Counter(), Counter(), {}
    for n in c.nodes:
        if isinstance(n, FFN):
            units[n.level] += len(n.terms)
        else:
            heads[n.level] += 1
            key = n.name.rstrip("0123456789").rstrip(".")
            names.setdefault(n.level, Counter())[key] += 1
    return [{"layer": L + 1, "heads": heads[2 * L + 1], "units": units[2 * L + 2],
             "tables": dict(names.get(2 * L + 1, {}))} for L in range(c.n_layers())]


def nonzeros(c) -> dict:
    """Nonzero weights of the emitted circuit layers (the exit's clears included)."""
    lay = weights.pick_layout(len(c.dims))
    tc = weights.text_config(c, lay)
    per = {}
    for L in range(c.n_layers()):
        per[L] = sum(int(np.count_nonzero(t)) for t in weights.chip_layer(c, lay, tc, L).values())
    return {"hidden": lay.hidden, "ballast_dims": lay.m, "per_layer": per,
            "total": sum(per.values())}


def summary(c, with_nonzeros: bool = True) -> dict:
    out = {"stats": c.stats(), "target": vars(c.target).copy(),
           "value_heads_per_key_head": c.target.value_heads // c.target.key_heads,
           "sublayers": per_sublayer(c)}
    if with_nonzeros:
        out["nonzeros"] = nonzeros(c)
    return out


def c_program_tokens() -> dict:
    """Instructions, tokens and tokens per instruction of the C programs (reference traces)."""
    from qwendoom import core_check, ref

    progs = core_check.ensure_progs()
    out, ti, tt = {}, 0, 0
    for f in sorted(progs.glob("c-*.elf")):
        m = core_check.elf_machine(f.stem, core_check.elf_inputs(f.stem))
        st = ref.trace_stats(m.run(10**8))
        out[f.stem] = {k: st[k] for k in ("instructions", "tokens", "refreshes")}
        out[f.stem]["tokens_per_instruction"] = round(st["tokens_per_instruction"], 3)
        ti += st["instructions"]
        tt += st["tokens"]
    out["all"] = {"instructions": ti, "tokens": tt, "tokens_per_instruction": round(tt / ti, 3)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", action="store_true")
    ap.add_argument("--tokens", action="store_true")
    args = ap.parse_args()
    res = {"qwendoom": summary(core.build())}
    if args.variants:
        t = core.TARGET
        res["qwendoom_2_value_heads"] = summary(
            core.build(Target(t.key_heads, 2 * t.key_heads, t.intermediate)))
        from chipformer.rv import core as upstream

        u = upstream.build()
        res["upstream"] = summary(u)
        dig = [n for n in u.nodes if n.name.startswith("bn")]
        res["upstream"]["digit_file"] = {
            "heads": sum(isinstance(n, Head) for n in dig),
            "units": sum(len(n.terms) for n in dig if isinstance(n, FFN)),
            "nodes": len(dig)}
    if args.tokens:
        res["c_programs"] = c_program_tokens()
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
