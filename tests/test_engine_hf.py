"""Gate (c): stock transformers (Qwen3_5ForCausalLM, bf16, CPU, the torch reference kernels)
agrees with RefEngine, teacher-forced through a reference trace with the snapshot written into
the DeltaNet cache after a PAD step: at every position the greedy prediction is the same and
every layer's output residual is the same (as values: +0 == -0).

Slow (seconds per position on the CPU): QD_HF_POSITIONS sets the positions (default 48; the
reported gate runs 300: `QD_HF_POSITIONS=300 pytest tests/test_engine_hf.py`).
"""

import os
import time
from pathlib import Path

import numpy as np
import pytest

from qwendoom import engine as E

ROOT = Path(__file__).resolve().parents[1]
RV_TINY = ROOT / "build" / "rv-tiny"
WEIGHTS = ROOT / "weights"
RV_DATA = ROOT / "third_party" / "qwenputer" / "experimental" / "riscv" / "tests" / "data" / "rv"
N = int(os.environ.get("QD_HF_POSITIONS", "48"))


def hf_model(model_dir):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.set_num_threads(int(os.environ.get("QD_HF_THREADS", "4")))
    return transformers.Qwen3_5ForCausalLM.from_pretrained(str(model_dir),
                                                           dtype=torch.bfloat16).eval()


def hf_start(model, rows, pad):
    """The HF cache after a PAD step with the state rows written into it (as
    chipformer.rv.hf.inject writes them)."""
    import torch

    with torch.no_grad():
        cache = model(input_ids=torch.tensor([[pad]]), use_cache=True).past_key_values
        for L, vh, rr in rows:
            st = cache.layers[L].recurrent_states[0]
            for row, vec in rr.items():
                st[0, vh, row] = torch.tensor(np.asarray(vec, np.float32), dtype=st.dtype)
    return cache


def rows_list(rows):
    """rv_rows' list format from either format (see engine.load_rows)."""
    if isinstance(rows, list):
        return rows
    out = {}
    for L, vh, r, v in zip(rows.layer.tolist(), rows.vhead.tolist(), rows.row.tolist(), rows.vec):
        out.setdefault((L, vh), {})[r] = v
    return [(L, vh, rr) for (L, vh), rr in sorted(out.items())]


def hf_vs_ref(model_dir, rows, pad, toks, n, log):
    import torch

    model = hf_model(model_dir)
    ref_engine = E.RefEngine(model_dir)
    rows = rows_list(rows)
    cache = hf_start(model, rows, pad)
    ref_engine.reset()
    ref_engine.step(pad)
    E.load_rows(ref_engine, rows)
    ref_engine.capture = []
    nl = len(ref_engine.cfg["layer_types"])
    t0 = time.time()
    with torch.no_grad():
        for i in range(n):
            o = model(input_ids=torch.tensor([[toks[i]]]), past_key_values=cache, use_cache=True,
                      output_hidden_states=True)
            hf_tok = int(o.logits[0, -1].float().argmax())
            ref_tok = ref_engine.step(toks[i])
            res = ref_engine.capture.pop()
            # hidden_states: the embedding, then each layer's output (the last after the final
            # norm): layers 0 .. nl - 2 are compared directly
            hs = np.stack([o.hidden_states[L + 1][0, -1].float().numpy() for L in range(nl - 1)])
            same = np.array_equal(hs, res[:nl - 1])
            if log:
                log(f"{i}: token {toks[i]} -> hf {hf_tok} ref {ref_tok} trace {toks[i + 1]} "
                    f"residuals equal {same} ({time.time() - t0:.0f} s)")
            assert hf_tok == ref_tok, (i, hf_tok, ref_tok)
            if not same:
                bad = [L for L in range(nl - 1) if not np.array_equal(hs[L], res[L])]
                raise AssertionError(f"position {i}: residuals differ in layers {bad}")
    return n


@pytest.mark.skipif(not (RV_TINY / "model.safetensors").exists() or not RV_DATA.exists(),
                    reason="build/rv-tiny or the upstream ELFs are missing")
def test_hf_rv_tiny():
    from chipformer.rv import core, elf, ref
    from chipformer.rv import spec as S

    prog = elf.program(RV_DATA / "c-crc32.elf")
    toks = ref.Machine.from_program(prog).run(N + 1)
    rows = E.rv_rows(core.build(), ref.Machine.from_program(prog))
    hf_vs_ref(RV_TINY, rows, S.VOCAB, toks, min(N, len(toks) - 1), print)


@pytest.mark.skipif(not (WEIGHTS / "model.safetensors").exists(), reason="weights/ is missing")
def test_hf_weights():
    from qwendoom import core_check, snapshot
    from qwendoom import spec as S

    toks = core_check.elf_machine("c-classic").run(N + 1)
    rows = snapshot.machine_rows(core_check.elf_machine("c-classic"))
    hf_vs_ref(WEIGHTS, rows, S.PAD, toks, min(N, len(toks) - 1), print)
