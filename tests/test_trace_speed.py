"""(e) Speed of the C trace machine, decoder and checker on upstream's longest ELFs (c-sudoku,
c-fib: 2.6M / 2.1M instructions). Prints the measured rates (pytest -s); the floors are far
below them, there to catch an accidental slow path."""

import time
from pathlib import Path

import pytest

from chipformer.rv import spec as S
from qwendoom import trace as T

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "third_party" / "qwenputer" / "experimental" / "riscv" / "tests" / "data" / "rv"
REGS = [0, 0, S.SP_INIT] + [0] * 29


def machine(name):
    mem = T.Memory(T.Map.upstream())
    entry = mem.load_elf(DATA / f"{name}.elf")
    r = T.Ref(mem)
    r.set_context(pc=entry, regs=REGS)
    return r, mem, entry


def best(fn, reps=3):
    """The fastest of reps runs of fn() (which sets itself up and returns its timed seconds)."""
    return min(fn() for _ in range(reps))


@pytest.mark.parametrize("name", ["c-sudoku", "c-fib"])
def test_speed(name):
    r, _, _ = machine(name)
    toks, _, why = r.trace()
    assert why == "halt"
    records, ntok = r.counts["records"], len(toks)

    def ref_count():
        r, _, _ = machine(name)
        t = time.perf_counter()
        r.run_count()
        return time.perf_counter() - t

    def ref_tokens():
        r, _, _ = machine(name)
        t = time.perf_counter()
        r.run(cap=ntok + 64)
        return time.perf_counter() - t

    def decoder():
        _, mem, entry = machine(name)
        d = T.Decoder(mem)
        d.set_context(pc=entry, regs=REGS)
        t = time.perf_counter()
        res, used = d.feed_many(toks)
        dt = time.perf_counter() - t
        assert (res, used) == (T.HALT, ntok)
        return dt

    def checker():
        r, _, _ = machine(name)
        c = T.Checker(r)
        t = time.perf_counter()
        assert c.check(toks) == (ntok, T.CHK_END)
        return time.perf_counter() - t

    rates = {"ref records/s (count only)": records / best(ref_count),
             "ref records/s (tokens kept)": records / best(ref_tokens),
             "decoder tokens/s": ntok / best(decoder),
             "checker tokens/s (ref + compare)": ntok / best(checker)}
    print(f"\n{name}: {records:,} records, {ntok:,} tokens")
    for k, v in rates.items():
        print(f"  {k:34s} {v / 1e6:7.1f} M")
    assert rates["ref records/s (count only)"] > 20e6
    assert rates["decoder tokens/s"] > 10e6
    assert rates["checker tokens/s (ref + compare)"] > 10e6
