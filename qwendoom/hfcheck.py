"""Run the emitted checkpoint in stock transformers (Qwen3_5ForCausalLM, bf16, CPU: the torch
reference kernels) and check it against the reference machine:

    .venv/bin/python -m qwendoom.hfcheck weights [--progs call,input,div] [--modes ...]

For each program, with the snapshot (qwendoom.snapshot.state_rows) written into the model's
DeltaNet cache after a PAD step:

* ``residual``: one prefill of the reference trace; the residual before the exit layer equals
  the IR evaluator's on every circuit dim at every position (bit for bit);
* ``prefill`` / ``decode``: teacher-forced, the greedy prediction at every position equals the
  next token (host-injected input records are fed, not compared); prefill runs the trace as one
  chunk (torch_chunk_gated_delta_rule), decode token by token (torch_recurrent_gated_delta_rule);
* ``generate``: free-running greedy decoding from GO, the model's own tokens fed back (input
  records injected by the host where the reference reads its input port), equals the trace.

Derived from nesformer/hfcheck.py (MIT, (c) 2026 Drew Breunig; see LICENSE.nesformer) and Qwenputer's chipformer/rv/hf.py and tinycheck.py
(MIT, (c) 2026 Divyansh Agrawal; see LICENSE.chipformer).
"""

import argparse
import json
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from qwendoom import ref, snapshot
from qwendoom import spec as S


def load_model(model_dir: str):
    import torch
    from transformers import Qwen3_5ForCausalLM

    model = Qwen3_5ForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).eval()
    meta = json.loads((Path(model_dir) / "qwendoom.json").read_text())
    return model, meta


def start_cache(model, meta: dict, rows: "snapshot.Rows"):
    """A cache after step(PAD) with the snapshot rows written into the recurrent states."""
    import torch

    with torch.no_grad():
        out = model(input_ids=torch.tensor([[S.PAD]]), use_cache=True)
    cache = out.past_key_values
    by_layer: dict = {}
    for L, vh, r, v in zip(rows.layer.tolist(), rows.vhead.tolist(), rows.row.tolist(),
                           rows.vec):
        by_layer.setdefault(L, []).append((vh, r, v))
    n_chip = meta["circuit"]["layers"]
    for L in range(n_chip):
        st = (cache.get_recurrent_state(L, state_idx=0) if hasattr(cache, "get_recurrent_state")
              else cache.layers[L].recurrent_states[0])
        assert not st.any(), "PAD left a nonzero state"
        for vh, r, v in by_layer.get(L, []):
            st[0, vh, r] = torch.tensor(v, dtype=st.dtype)
    return cache


@contextmanager
def one_thread():
    """Token-by-token steps on one CPU thread: transformers' depthwise conv fallback
    (causal_conv1d_update) spends nearly all its time in OpenMP fork / join otherwise
    (measured 0.31 s per token on one thread, minutes per token on 18)."""
    import torch

    n = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(n)


def machine_rows(m: "ref.Machine", meta: dict) -> "snapshot.Rows":
    heads = {k: tuple(v) for k, v in meta["heads"].items()}
    return snapshot.machine_rows(m, heads)


def teacher_forced(model, meta, m: "ref.Machine", mode: str, max_tokens=10**5) -> dict:
    """Feed the reference trace; compare every non-injected prediction."""
    import torch

    rows = machine_rows(m, meta)
    toks = m.run(max_tokens)
    inj = set(m.injected)
    cache = start_cache(model, meta, rows)
    preds = []
    with torch.no_grad():
        if mode == "prefill":
            o = model(input_ids=torch.tensor([toks]), past_key_values=cache, use_cache=True)
            preds = o.logits[0].float().argmax(-1).tolist()
        else:
            with one_thread():
                for t in toks:
                    o = model(input_ids=torch.tensor([[t]]), past_key_values=cache,
                              use_cache=True)
                    cache = o.past_key_values
                    preds.append(int(o.logits[0, -1].float().argmax()))
    bad = [(i + 1, S.name(toks[i + 1]), preds[i]) for i in range(len(toks) - 1)
           if i + 1 not in inj and preds[i] != toks[i + 1]]
    return {"tokens": len(toks), "checked": len(toks) - 1 - len(inj), "wrong": bad[:10],
            "n_wrong": len(bad)}


def generate(model, meta, m: "ref.Machine", max_tokens=10**5) -> dict:
    """Free-running greedy decoding from GO; the host injects input records (where the
    reference's trace has them), everything else is the model's own output."""
    import torch

    rows = machine_rows(m, meta)
    want = m.run(max_tokens)
    inj = set(m.injected)
    cache = start_cache(model, meta, rows)
    got = [want[0]]
    with torch.no_grad(), one_thread():
        while len(got) < len(want):
            o = model(input_ids=torch.tensor([[got[-1]]]), past_key_values=cache, use_cache=True)
            cache = o.past_key_values
            p = int(o.logits[0, -1].float().argmax())
            got.append(want[len(got)] if len(got) in inj else p)
            if got[-1] != want[len(got) - 1]:
                break
    return {"tokens": len(want), "generated": len(got) - 1 - len(inj), "equal": got == want,
            "first_diff": next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), None)}


def residual(model, meta, c, m: "ref.Machine", max_tokens=10**5) -> dict:
    """The residual before the exit layer == the IR's, every circuit dim, every position."""
    import torch

    from qwendoom.q35 import evaluate as fe

    rows = machine_rows(m, meta)
    snapshot.load_ir(c, snapshot.machine_tables(m))
    toks = m.run(max_tokens)
    want = np.asarray(fe.evaluate(c, toks, check=False))
    cache = start_cache(model, meta, rows)
    with torch.no_grad():
        o = model(input_ids=torch.tensor([toks]), past_key_values=cache, use_cache=True,
                  output_hidden_states=True)
    n_chip = c.n_layers()
    h = o.hidden_states[n_chip - 1][0].double().numpy()
    m_ = meta["layout"]["ballast_dims"]
    keep = set(c.keep)
    dims = [d for d in range(1, len(c.dims))
            if c.dims[d].level < 2 * n_chip - 1 and d not in keep]
    bad = [d for d in dims if (h[:, m_ + d] != want[:, d]).any()]
    out = {"positions": len(toks), "dims": len(dims), "inexact_dims": len(bad)}
    if bad:
        d = bad[0]
        t = int(np.nonzero(h[:, m_ + d] != want[:, d])[0][0])
        out["first"] = (c.dims[d].name, t, float(h[t, m_ + d]), float(want[t, d]))
    return out


# --- programs -------------------------------------------------------------------------------

CALL = """_start: li a0, 7
        call fact
        mv s1, a0
        li a0, 1000
        li a1, -7
        div a2, a0, a1
        rem a3, a0, a1
        li t0, 0x3F000
        sw a2, 0(t0)
        sb a3, 5(t0)
        lw a4, 4(t0)
        li t1, 0x80004
        lw a5, 0(t1)
        sh2add a6, a5, a4
        li a7, 1
        li a0, 72
        ecall
        li a0, 256
        ecall
        li a7, 93
        li a0, 0
        ecall
fact:   addi sp, sp, -16
        sw ra, 12(sp)
        sw s0, 8(sp)
        mv s0, a0
        li t0, 2
        blt a0, t0, fact1
        addi a0, a0, -1
        call fact
        mul a0, a0, s0
        j fact2
fact1:  li a0, 1
fact2:  lw ra, 12(sp)
        lw s0, 8(sp)
        addi sp, sp, 16
        ret"""

INPUT = """_start: csrr a0, time
        csrr a1, time
        csrr s3, time
        add a2, a0, a1
        csrr t0, time
        sw s3, 0(t0)
        lw a3, 0(t0)
        mv a4, s3
        li a7, 93
        ecall"""


def program(name: str) -> "ref.Machine":
    from qwendoom.core_asm import assemble

    if name == "call":
        p = assemble(CALL)
        p.rodata = bytes(4) + (0xCAFEF00D).to_bytes(4, "little")
        return ref.Machine.from_program(p)
    if name == "input":
        return ref.Machine.from_program(assemble(INPUT),
                                        inputs=[5, 0x12345, 0xDEADBEEF, 0x3F000])
    from qwendoom import core_check

    return core_check.elf_machine(name)  # an ELF of tests/progs (e.g. rv32um-div)


def main() -> None:
    import torch

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model", nargs="?", default="weights")
    ap.add_argument("--progs", default="call,input")
    ap.add_argument("--modes", default="residual,prefill,decode,generate")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    t = time.time()
    model, meta = load_model(args.model)
    print(f"loaded {args.model} ({time.time() - t:.0f} s)", flush=True)
    c = None
    ok = True
    for name in args.progs.split(","):
        for mode in args.modes.split(","):
            t = time.time()
            m = program(name)
            if mode == "residual":
                if c is None:
                    from qwendoom import core

                    c = core.build()
                res = residual(model, meta, c, m)
                good = res["inexact_dims"] == 0
            elif mode == "generate":
                res = generate(model, meta, m)
                good = res["equal"]
            else:
                res = teacher_forced(model, meta, m, mode)
                good = res["n_wrong"] == 0
            ok &= good
            print(f"{name:10s} {mode:8s} {'ok ' if good else 'BAD'} {res} "
                  f"({time.time() - t:.1f} s)", flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
