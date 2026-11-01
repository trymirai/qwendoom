# The inference engine

`native/engine` builds `build/libqdengine.dylib` (`make -C native/engine`, Apple clang,
`-O3 -mcpu=native -ffp-contract=off`). It computes the forward pass of the stock
`Qwen3_5ForCausalLM` checkpoint for one position at a time and returns the greedy next token,
exactly: the same tokens, and the same residual after every layer, as stock transformers in bf16
on the CPU. `qwendoom/engine.py` loads it with ctypes (`Engine`) and holds the numpy reference
(`RefEngine`) the engine is checked against.

| File | Contents |
|---|---|
| `native/engine/qd_engine.h` | the C API (ARCHITECTURE.md's five calls, plus bulk calls, options, stats) |
| `native/engine/model.c` | opening a checkpoint: config, safetensors (mmap), the compact layouts, the layout cache |
| `native/engine/step.c` | a step: norms, projections, conv, the delta rule, MLP, head; the worker pool |
| `native/engine/internal.h` | the layouts and the run state |
| `qwendoom/engine.py` | `RefEngine` (numpy), `Engine` (ctypes), `start` / `load_rows` (a run's start) |
| `tests/test_engine.py`, `tests/test_engine_hf.py` | the gates (below) |
| `scripts/bench_engine.py` | speed and per-stage profile; captures real DOOM ticks to bench on; interleaved A/B of engine builds |

## Using it

```c
qd_engine *qd_engine_open(const char *weights_dir, int n_threads);   /* NULL: qd_engine_last_error() */
void qd_engine_close(qd_engine *);
void qd_engine_reset(qd_engine *);                       /* zero every state and conv window */
void qd_engine_set_row(qd_engine *, int layer, int value_head, int row, const float *vec128);
int  qd_engine_step(qd_engine *, int token);             /* feed one token, return the greedy next */
```

A run (ARCHITECTURE.md): reset, step PAD, write the snapshot rows, then step the snapshot's first
token (GO) and each token after it. `engine.start(engine, rows, pad)` does the first three for
either row format (`qwendoom.snapshot.Rows` or upstream rv-tiny's `engine.rv_rows`), stepping
PAD as a forced step (its prediction is never used).

Extensions (all in `qd_engine.h`, all wrapped by `Engine`):

* `qd_engine_set_rows` (a value head's rows in one call), `qd_engine_get_row`;
* `qd_engine_step_forced(token)`: feed a token whose successor the host already knows (an
  injected input record): no head, and the trailing layers that only feed the head are skipped;
* `qd_engine_feed` (forced steps, then one step), `qd_engine_generate` (free-running greedy
  decoding in C, stopping at a token), `qd_engine_check` (teacher forcing against a trace,
  skipping host-owned positions; returns the first wrong position);
* `qd_engine_generate_until(e, token, n, stops, n_stops, out)`: free-running greedy decoding that
  stops after emitting any of the stop tokens (a runner's YIELD = BYTE(256), HALT and the FLT
  family); the stop is out's last entry and is not fed (pass it as the next call's token).
  Returns the count. Python: `Engine.generate_until(t, n, stops)`;
* `qd_engine_set_exact_head`: always take the full vocabulary argmax;
* `qd_engine_capture`: write every layer's output residual after each step (bit checks);
* `qd_engine_set_profile`, `qd_engine_get_stats`: time per stage and work counters;
* `qd_engine_open_ex(dir, n_threads, cache_dir, flags)`: `QD_OPEN_NO_CACHE`, `QD_OPEN_VERBOSE`
  (prints the discovered structure per layer), `QD_OPEN_GENERIC` (turns off the specialized paths:
  unit records, cached prefixes, channel bounds; for tests), `QD_OPEN_REFRESH` (drops every
  cached prefix at every 7th step, which forces the base-change path: deferred projections
  computed, bounds refitted; for tests);
* `qd_engine_bench_sync`: the worker pool's round trip.

Another build of the library (one side of an A/B comparison, a build under test) goes
elsewhere with `make -C native/engine OUT=/abs/path/libqdengine.dylib`; `QD_ENGINE_LIB` points
`qwendoom/engine.py` at it. The Makefile writes a new file and renames it into place, so a
process that has the old library loaded keeps running it.

`n_threads <= 0` means `QD_THREADS` or 1. With more, `n_threads - 1` workers (user-interactive
QoS, so macOS prefers P-cores) take shares of the large sublayers; they spin between steps and
sleep after 2 ms idle. The calling thread is not changed; a latency-critical caller should ask
for the user-interactive QoS class itself (`bench_engine.py` does).

## What the engine computes

Per layer (16 + 1 in the qwendoom checkpoint: hidden 2,304, 16 key and 64 value heads; 11 + 1 in
rv-tiny: hidden 2,816, 16 key and 32 value heads; in general any head counts with value heads a
multiple of key heads, head dims multiples of 64 up to 256, key heads times key head dim up to
524,288, conv width up to 8, hidden up to 65,536):

1. `xn = bf16(RMSNorm(x) * (1 + w))`, the projections `bf16(W xn)` (qkv, z, a, b), the causal
   conv over the last 4 positions (taps oldest first, multiply then add) and `bf16(silu(bf16(.)))`;
2. per value head h (reading key head `h / (nv / nk)`): l2-normed q and k (`x * rsqrt(sum x^2 +
   1e-6)`, q also `/ sqrt(128)`), `beta = bf16(sigmoid(b))`, `g = -exp(A_log) softplus(a +
   dt_bias)`; the gated delta rule in fp32: `S *= exp(g)`, `kv = sum_r S[r] k[r]`, `S[r] += k[r]
   (v - kv) beta`, `o = bf16(sum_r S[r] q[r])`;
3. the gated RMSNorm `bf16(bf16(norm_w * bf16(o * rsqrt(mean o^2 + eps))) * silu(z))`, out_proj,
   `x = bf16(x + bf16(.))`;
4. the MLP `bf16(bf16(silu(bf16(gate))) * bf16(up))`, down, `x = bf16(x + bf16(.))`;

then the final norm and the tied head's argmax. The one full-attention layer must be all zero
(it then adds exactly 0; open refuses anything else).

## Design

The checkpoint is a compiled circuit: >99% of the weights are 0, and at any position most
activations are exactly 0 too. The residual has 2,304 (2,816) dims, of which 25 to 420 are
nonzero; a sublayer reads a few of those; a few hundred of its thousands of MLP units have a
nonzero silu; one or two of 2,048 query channels fire. The engine finds the nonzero structure at
open and does only the work whose result is not exactly zero (or exactly known).

**Open** (`model.c`): mmap `model.safetensors`, find per layer the value heads that can reach the
output (an out_proj column and a z row both nonzero), their key heads' query and key channels
("eager" channels, computed every step), their value channels (computed lazily, only when the
head writes), the z / a / b rows ("pool" rows, CSR), the MLP units with a nonzero gate row, up
row and down column, and the residual dims each sublayer reads and writes. Projections over the
residual become blocked CSC matrices: per input dim, the 16-row blocks of outputs it touches with
16 bf16 weights each (the circuit's columns are dense over runs of rows). The embedding keeps its
nonzero columns, the head its code dims (where the final norm's `1 + w` is nonzero: 21 of them).
That takes 1.3 s for rv-tiny (4.2 GB) and is cached in `build/engine-cache/<sha256>-v5.qdc`
(sha256 memoized by path, size, mtime, inode): a cached open takes 0.03 s. Open checks what it
reads: config.json's shapes are positive integers and layer_types names every layer, every tensor
is bf16 with the shape the code reads and its data inside the file, JSON nests at most 64 deep, and
a missing, unreadable or truncated file fails open with an error naming it. A cache file that
cannot be read whole (truncated, another version or shapes) is stale: rebuilt and replaced.

**A step** (`step.c`), per sublayer:

* *Sparse residual.* `x` keeps a bitmap of its nonzero dims. The RMSNorm's sum of squares runs
  dense over the 64-dim words that hold a nonzero dim (in the reference's 16-lane order: the
  other words would add +0), and 1 / rms(x) is kept until a residual update changes a value
  (most attention sublayers of the table layers write nothing). The normed values are computed
  only at the nonzero dims the sublayer reads; the residual update walks a bitmap of the dims
  the sublayer wrote.
* *Cached prefix* (`prefix_t`). The residual dims below a per-sublayer bound (the ballast: 9 to
  11 huge constant dims that pin the RMS) never change during a run, and they are the lowest
  input dims, so their terms come first in every output's in-order sum. Their partial sums
  ("base") are kept; a step compares the prefix's (dim, value) pairs with the cached ones (a few
  compares) and scatters only the other inputs. That halves the MLP gate's block visits. The
  accumulator holds base between steps; the touched blocks are put back after each step.
* *Blocked scatter.* `acc[block] += w[16] * xn[c]` with NEON over the nonzero inputs, marking
  the blocks touched (a column's blocks are in increasing order, so a bitmap word's bits gather in
  a register); only those, plus the blocks live at base, are examined afterwards.
* *Eager channels* (queries and keys). Single-tap channels (conv weight only on the current
  position) need no history; ring channels keep the bf16 projections of the last 4 positions.
  The nonzero queries and keys go to per-key-head lists in dim order.
* *Channel-group bounds* (`qkbound_t`). In the qwendoom checkpoint every eager channel of the 11
  table layers (ROM, 2 DROM, 8 RAM) is an address detector: weights `+-2^44` on 10 to 17 bit
  inputs, and a base (the ballast's term) of `128 - 256 popcount(pattern)`, so its projection is
  `128 - 256 * HammingDistance(pattern, address bits)`; on real DOOM ticks one or two of a
  layer's 2,048 or 4,096 channels fire. Channels are grouped by 128 (8 blocks; a key head's
  queries or keys), and each group's projections are bounded in one pass over the few nonzero
  inputs. Per input row and group the channels fall into three weight classes (`w > 0`,
  `w < 0`, `w = 0`); base is written as `r_j` plus one offset per row and class (fitted when base
  changes: per group, a class's mean base minus the group's), so per row the bound takes the
  largest (smallest) over the classes of offset + the class's extreme weights times the input.
  Any offsets give a valid bound; for the detectors the fit is exact (`r_j` is constant) and the
  bound is the exact best case `128 - 256 * mismatches` of the bits that are fixed in the group,
  so a group whose key-head bits disagree with the address is proved dead (pre-activation
  `<= -89`, silu exactly -0). The previous per-input extreme-weight bound (offsets 0) proved
  only 31-55% of the RAM layers' single-tap groups dead; this one proves all but the matching
  one (300.9 of the 308 groups per token). The two regions (all single-tap groups, all ring groups)
  are bounded first as groups of their own: when a region is dead (the 3 layer-select bits of
  a RAM layer disagree, or the key's previous token was not a store) its groups need no test.
  A dead group is neither projected nor convolved; the live ones are scattered one at a time
  (`gptr`: where each group's blocks start in each input's column).
* *Deferred ring projections.* A ring group's projection at this position is needed only by
  the convs of the next positions that read this slot and are not themselves proved dead, so a
  dead ring group's projection is deferred (`pend`); a later position whose bound leaves the
  group live computes it first, from that position's normed inputs (kept in a ring of 4 for the
  lazy value channels) and base, in the same order as the scatter. Likewise its bound range at
  that slot (`rr_ok`: while the ring region is dead only the region's range is stored). Before
  base changes (`qk_flush`) every deferred projection is computed and a missing range takes the
  region's (a looser bound on the same values).
* *Value heads.* The decay is a precomputed constant where the a row is zero; only the key heads
  with a key, a query or a decay other than 1 are visited; the delta rule and the read touch only
  the key's / query's nonzero rows; a head's value channels are computed only when it writes
  (from the normed inputs of the last 4 positions) and not at all when beta is exactly 0 and
  `v - kv` is provably finite (`write_noop`); pool rows over the constant prefix only are cached.
  The z rows (gates) of the output dims are deduplicated at open (a RAM layer's 4,096 output dims
  share 4 rows, a ROM / DROM layer's 64) and each distinct `silu(bf16(z))` is computed once per
  step; a head whose gates are all 0 (3 of the 4 value heads of a RAM read) skips its gated
  norm and out_proj.
* *MLP.* Units become records (down dim, down weight, the first up entry) when every unit has one
  down entry; a block of 16 units whose up rows each have one entry is done with NEON at once
  (`silu(g) = g` for `g >= 17`, `-0` for `g <= -89`, then up, activation), no branch per unit,
  and when its 16 units read one up input (every such block of layers 1-7) that input is loaded
  once; other blocks take the scalar path. Activations go to a per-unit array with a 16-bit mask
  of the nonzero ones per block and a per-thread list of blocks; the down sum walks them in unit
  order, a run of units writing one dim accumulating in a register.
* *Head.* If every token's row is 0 or `+-A` on the code dims and the normed code has equal
  magnitudes, the argmax is the row with the code's signs (a 2^21-entry table); otherwise (and
  with `set_exact_head`) the full argmax over the vocabulary's code dims.
* *bf16 rounding* is one BFCVT instruction (`FEAT_BF16`).

**Threads.** The two big jobs per layer (the eager channels; the MLP's gate and units) are split
by output blocks over the workers; per-thread outputs are concatenated in thread order, so every
sum keeps its order and the results are bit-identical for any thread count. A job's thread count
follows its size (small jobs run on the caller alone; with the group bounds the eager channels
almost always are). The rest (norms, value heads, down, residual) is serial.

**Not Metal.** Tokens are strictly sequential (each needs the previous argmax) and a step is
~30 us of sparse, data-dependent work. On this M5 Max one trivial Metal compute dispatch
costs 86 us (best; 112 us mean) from commit to completion, and 40 dependent dispatches in one
command buffer 140-240 us: more than a whole CPU step, before any work.

## Exactness

The claim: for every position, the engine's residual after every layer and its token equal
RefEngine's, and RefEngine's equal stock `Qwen3_5ForCausalLM` (bf16, CPU). The argument, then
the evidence.

1. **Same operations, same order.** RefEngine fixes an order where torch's is unspecified:
   projections sum in increasing input order (scipy CSR), norms in the 16-lane `sum16` order, the
   delta rule over rows in increasing order. The engine does the same additions in the same
   order. Every skipped term is exactly `+-0` (a zero weight, a zero input, a unit whose silu is
   exactly 0), and adding `+-0` to a sum leaves it unchanged except for the sign of a zero
   result. Signs of zeros never reach a token (every later use is a product, a sum, a
   comparison or silu, which map `+-0` to `+-0` or to the same value), so residuals are compared
   as values.
2. **Fused multiply-add only where it is exact.** A bf16 x bf16 product has 16 significant bits,
   so it is exact in fp32 while it stays in the normal range; then `fma(w, x, s) = s + w * x`.
   The engine uses fma only for such products and checks the range per sublayer: the nonzero
   |weight| range of each matrix (open) times the |input| range (each step) must stay within
   [2^-126, 2^127] (`stats.range_risk`, 0 on every run here). Everything else (conv, delta rule,
   norms) multiplies and adds separately; `-ffp-contract=off` keeps the compiler from fusing.
3. **bf16 rounding.** BFCVT equals round-to-nearest-even on the bits: checked on all 4.28 billion
   finite fp32 inputs (denormals included) on this M5 Max.
4. **Transcendentals only where every implementation agrees.** silu, sigmoid and the decay are
   evaluated in ranges with exact results (`silu(x) = x` for `x >= 17`, `-0` for `x <= -89`,
   `exp(0) = 1`, `exp(g) = 0` for `g <= -104`, softplus' linear branch above 20, and `exp(-log1p(1))
   = 0.5` for the circuit's halving heads, 0.03 ulp from 0.5). Anything else is computed with libm
   and counted (`stats.slow_transcendental`, 0 on every run here); numpy, torch and libm may
   differ there in the last bit.
5. **Cached prefix.** The base is the fp32 partial sum over the same leading terms in the same
   order; it is reused only after comparing those terms' inputs bit for bit.
6. **Bounds.** A group is skipped only if every channel's pre-activation is provably `<= -89`.
   With base_j = r_j + sum over the rows of `o(row, class of w_j,row)` for any offsets o (r_j is
   defined by it, in double), channel j's exact projection is r_j + sum over the rows of
   (`o(row, class)` + `w_j,row x_row`) <= max_j r_j + sum over the rows of the largest, over the
   group's classes, `o(row, class)` + (the class's extreme weights times x_row) (rows with
   x = 0 add their largest offset; the minimum likewise). That is widened by `(2 n + 8) 2^-21`
   of the magnitude (|base| + the offsets' magnitude + sum |w x|): fp32 rounding of the engine's
   n additions is at most gamma_n of it, and the float evaluation of the bound, whose terms also
   hold offsets, at most `3 (n + 2) u` of up to three times it. bf16 roundings widen by 2^-8
   relatively, the tap products and sums by gamma_KS. A region's bound is the same over all its
   channels, so a dead region's groups are dead. The normal path would have produced exactly the
   same -0's.
7. **Deferred projections** are computed later from the same normed inputs (kept bit for bit
   in the ring) and the same base (any deferred projection is computed before base changes), by
   the same fused multiply-adds in the same input order; a ring slot without its group ranges
   uses the region's, a valid (looser) bound on the same values.
8. **Skipped value-head work.** A head whose gates `silu(z)` are all `+-0` adds `y z = +-0`
   wherever y is finite, and y is finite when o is (|o / rms(o)| <= sqrt(dv)): o is computed and
   checked, then the norm and out_proj are skipped. A write with beta = 0 adds
   `k ((v - kv) 0) = +-0` to each row while `v - kv` is finite: kv is computed and checked
   (|kv| < 2^126), and |v| <= 1.1 vbound max|xn| + 0.3 < 2^101 (vbound: per value channel the
   sum of |taps| times the sum of |w| of its row; the bf16 roundings; |silu(x)| <= max(|x|,
   0.28)), so v is not computed. Equal z rows (the same CSR entries) give equal z.
9. **Reused values.** 1 / rms(x) is recomputed whenever a residual update changes any value of
   x; the dense sum of squares adds the same values per lane in the same order (the skipped
   words would add +0 to sums that are >= +0).
10. **Sign head.** When all token rows are 0 or `+-A` on the n code dims and |y| is the same m on
   all of them, a row with d sign disagreements scores `A m (n - 2d)` exactly (n <= 24), so the
   row with y's signs is the unique maximum; otherwise the full argmax runs.
11. **Threads** split outputs, never a sum.

**Evidence** (`tests/test_engine.py`, `tests/test_engine_hf.py`; results below):

* (a) free-running greedy decoding equals the reference machine's trace to HALT on upstream ELFs
  on rv-tiny: c-crc32, c-qsort, c-sieve, c-gmpmul, c-sort, rv32um-div, mpy-s1 (371,608 predicted
  tokens; with `QD_LONG=1` also c-sieve4, mpy-s5, c-crc32p, c-strops, rv32um-mulh,
  rv32uzba-sh1add: 1,166,810), and on weights/ with the C test programs: c-classic, c-input (18
  host-injected input tokens), c-recurse, rv32um-div, rv32ui-sb (162,421; long: also c-memsweep,
  c-bytes, c-div64, c-funcptr, c-switch: 614,442);
* (a') on real DOOM ticks (`test_doom_ticks`; Freedoom E1M1 captured in the test as
  `bench_engine.py --doom-capture` does, the C reference machine's tokens): every prediction
  equals the reference's token except at the host-injected input records (fed, not compared):
  ticks 0-2 and 700-701 (94,763 tokens, 547 injected); with `QD_LONG=1` the benchmark's two
  windows (640,468 tokens, 2,340 injected). The benchmark checks the same on every run;
* (b) the engine's residual after every layer equals RefEngine's at every position (rv-tiny:
  2,500 positions of mpy-s1 (long: 6,000) and 1,500 of c-qsort; weights/: 1,200 of c-classic
  (long: 3,000)); the generic paths (no bounds, no deferred projections, no unit records, no
  cached prefixes), `QD_OPEN_REFRESH` (a base change every 7th step: deferred projections
  flushed, bounds refitted) and 2, 3 and 6 threads equal the default engine at every position
  of 3,000 of mpy-s1 on rv-tiny, and (generic, refresh, 3 threads) at the first 2,500 positions
  (long: 8,000) of each DOOM window on weights/; on a synthetic one-layer checkpoint of hidden
  33,024 (the group bound tables' rows, the live groups' scatter and the deferred projections
  past dim 32,767) the default, refresh and generic paths equal RefEngine at every position;
  the exact head equals the sign head;
  `generate_until` (chained calls reproduce c-classic to HALT, stopping after each of its 75
  putchar BYTEs), forced steps, `check`, `feed`, the layout cache and the row calls are tested
  too. Every run reports 0 range risks and 0 slow transcendentals;
* (c) stock transformers (CPU, bf16, the torch reference kernels: recurrent delta rule, conv
  update) equals RefEngine, residual after every layer and prediction, at 300 positions of
  c-crc32 on rv-tiny and 300 of c-classic on weights/ (7.5 and 12 minutes).

## Performance

**The benchmark: real DOOM ticks.** `scripts/bench_engine.py --doom-capture` plays Freedoom E1M1
with `doom/measure.py`'s bot in native sim mode while the C reference machine (`trace.Ref`, its
own RAM, fed only the snapshot and each tick's pokes, as the model is) runs every tick alongside
and must match the native tick's instruction and token counts and RAM. For each window of ticks
it saves to `build/engine-bench/doom/` the snapshot at the window's start (RAM, DROM, ROM,
registers, position and their state rows) and the reference's tokens: `toks` = GO + the ticks'
tokens, `host` = the injected input records. Two windows (3 s to capture):

| Window | Ticks | Tokens (injected) | Tokens per tick | Monsters awake |
|---|---|---|---|---|
| start | 0-15 (the level start) | 236,736 (1,621) | 12,345-22,478, median 14,196 | 0 |
| fight | 1000-1009 | 403,732 (719) | 22,296-62,133, median 40,518 | 18 |

(The whole 2,100-tick run of docs/doom.md: median 29.7k tokens per tick, mean 36.1k.) `--doom`
runs them teacher-forced (`check`: every prediction must equal the next reference token, the
injected positions fed as forced steps) for each thread count; `--ab A.dylib,B.dylib` runs
engine builds in turns on them (AB, BA, AB, ...) in one process.

Measured on the M5 Max (6 P + 12 E cores) with other work running (load average 3.6-5), the
thread counts taking turns, 3 rounds, best and median (`build/engine-bench/bench.log`, `.json`):

| Checkpoint, program | 1 thread | 2 | 3 | 4 | 6 |
|---|---|---|---|---|---|
| weights/, DOOM ticks (640,468 tokens, teacher-forced) | **30,893** tok/s (32.4 us), median 30,880 | 30,490 | 30,698 | 30,519 | 29,764 |
| weights/, c-memsweep (100,002, teacher-forced) | **27,330** (36.6 us), median 27,071 | 27,050 | **28,160** | 27,941 | 27,143 |
| rv-tiny, c-qsort (100,000, free-running) | **24,230** (41.3 us), median 23,962 | 26,103 | 27,256 | **27,956** | 27,378 |

Against the engine before this round (interleaved on the same DOOM windows, 3 rounds,
`build/engine-bench/ab-final.log`): 14,110 -> 30,948 tok/s median, **2.195x** (2.186-2.195).
Against the previous round's measurements (not interleaved): c-memsweep 15,966 -> 27,330
(1.71x), rv-tiny c-qsort 21,834 -> 24,230 (1.11x: rv-tiny's eager channels are not bounded as
tightly; 43 groups per token proved dead, 41.7k qk MACs per token remain). For scale:
RefEngine (numpy) 168 tok/s; upstream nesformer's numba engine ~1,650 tok/s (one core),
Qwenputer's Triton executor ~5,600 tok/s on an NVIDIA L4. Opening a checkpoint takes 0.03-0.05 s
from the layout cache (1.3 s to build it; 1.02 M / 1.43 M nonzero weights kept).

Per stage on the DOOM ticks, one thread (ns per token, disjoint stages; MACs = multiply-adds
done, 16 per block visit; "norm" includes the prefix checks and the group bounds):

| Stage | before | | now | |
|---|---|---|---|---|
| norm (+ prefix checks, channel bounds) | 5,845 | 8.7% | 4,260 | 13.3% |
| qk_proj (eager channel scatter) | 18,371 | 27.4% (167,608 MACs) | 377 | 1.2% (1,806) |
| conv (ring, conv, silu, q / k lists) | 9,774 | 14.6% | 533 | 1.7% |
| delta (value heads: delta rule, read, gated norm, out_proj) | 5,568 | 8.3% | 3,477 | 10.8% |
| v_lazy (value channels, when a head writes) | 1,556 | 2.3% (972) | 1,004 | 3.1% (634) |
| gate (MLP scatter) | 7,790 | 11.6% (65,368) | 6,245 | 19.5% (65,368) |
| up (candidate blocks: up, activation) | 11,705 | 17.4% | 9,505 | 29.6% |
| down | 4,768 | 7.1% | 4,791 | 14.9% |
| residual, embed, head | 1,791 | 2.6% | 1,873 | 5.9% |

(Before, the bound's own time was in no stage: the stages summed to 67 of 71 us.) Per token:
7.6 key writes, 26.0 query reads, 6,057 MLP units with a nonzero activation, 300.9
of the 308 eager-channel groups proved dead (was 127.8); every head decided by the sign code; 0
range risks, 0 slow transcendentals. The step is now the MLP (gate + up + down: 64%), whose
work is mostly real: about 7.9k units per token have a live gate, most groups of 128 units hold
a live one in layers 1, 2 and 15, and 42% in layer 6.

How it got here (interleaved A/B runs, one thread, the first 80,000 tokens of each DOOM window;
median ratio of 4-8 rounds, every token checked; every kept change but the key-head visiting
(0.995-1.035 over 8 rounds) won in every round):

| Change | Speedup |
|---|---|
| decomposed group bounds (class offsets fitted to base) + deferred ring projections | 1.507 |
| 1 / rms(x) kept while x is unchanged | 1.061 |
| regions bounded first; live groups scattered one by one (gptr) | 1.141 |
| the scatter's touched bits gathered in a register | 1.023 |
| z rows deduplicated, computed once per step; heads with all gates 0 skip their norm | 1.038 |
| only the key heads with a key, a query or a decay are visited | 1.020 |
| unit records per block (layers 0, 3, 4 have some multi-entry up rows) | 1.024 |
| activations dense with block masks instead of item lists | 1.015 |
| the sum of squares dense over the words holding a nonzero dim | 1.035 |
| ring groups' ranges deferred while the ring region is dead; dead regions in bulk | 1.031 |
| writes with beta = 0 skipped (write_noop) | 1.017 |
| one up input per block loaded once | 1.016 |

Earlier (C programs): the CSC scalar-scatter engine 7.9k tok/s -> blocked NEON scatter + sparse
residual 11.0k (1.40x) -> branch-free unit items, bitmap residual (1.06x) -> cached MLP prefix
(gate MACs 110k -> 63k per token) -> BFCVT rounding (1.12x) -> 16-unit NEON MLP blocks (1.12x)
-> skipping dead blocks early (1.10x), 19.4k -> per-input extreme-weight channel bounds (1.10x
on weights/ c-classic).

Tried and dropped: interleaving the down sums of 4 run-aligned segments (0.913x; 0.920x with the
chains in registers: down is not latency-bound here), NEON over dense 64-dim words in the norm
and residual updates (0.981x at 12 and at 32 dims per word), bounds on the MLP gate's groups (on
the DOOM ticks a perfect group-of-128 bound would avoid 37% of the gate's block visits, but with
58-437 nonzero inputs per MLP sublayer the bound costs about what it saves), a per-block CSR gate
(10x the visits), whole-layer channel bounds (loose), reusing the previous step's projections
(inputs repeat only 29% of the time), Metal (above).

**Threads** no longer help on the DOOM ticks (best 30,698 at 3 threads against 30,893 on one):
the eager channels are now too small to split, and of the MLP the per-thread split balances the
matrices' blocks, not the step's live units (with 2 threads one worker's share holds 2.2x the
other's activations; the two were busy about 9 and 14 us per token where one thread takes about
16, in an earlier build), on top of a 0.2-0.3 us pool round trip per job. Batch-1 decoding is a
chain of ~25 dependent sublayers of 0.5-5 us each.

**Leads.** The last layer's MLP (layer 15, ~4 us per token) mostly clears the residual dims
outside the 21 code dims; those only reach the head through the final norm's rms, which the
sign code does not depend on when the code's magnitudes are equal, so with a proof that the rms
stays finite the clears could be skipped when the head is the sign code (not done: the engine
skips only exactly-zero work). In check mode, where the reference already supplies every token,
a pipelined teacher-forced check (layers split between two threads, token i + 1's early layers
while token i's late ones run) could approach 2x; free-running decoding cannot pipeline.

What DOOM needs: a median tick (29.7k tokens) is ~1.0 s at 31k tok/s; the level start 0.45 s,
the heaviest tick seen (193k) 6.3 s.

## Running the gates

```sh
make -C native/engine
.venv/bin/python -m pytest tests/test_engine.py            # (a), (a'), (b), options: ~1.6 min
QD_LONG=1 .venv/bin/python -m pytest tests/test_engine.py  # more programs, ticks, positions: ~4 min
QD_HF_POSITIONS=300 .venv/bin/python -m pytest tests/test_engine_hf.py -s   # (c): ~20 min
.venv/bin/python scripts/bench_engine.py --doom-capture     # once: build/engine-bench/doom/
.venv/bin/python scripts/bench_engine.py --tiny --weights --doom --ref
make -C native/engine OUT=$PWD/build/engine-ab/new/libqdengine.dylib   # a build to compare
.venv/bin/python scripts/bench_engine.py --ab build/libqdengine.dylib,build/engine-ab/new/libqdengine.dylib --tokens 80000 --repeat 6
```

rv-tiny is built by `python -m chipformer.rv.tiny --out build/rv-tiny` (35 s on the CPU; no
change to upstream needed).
