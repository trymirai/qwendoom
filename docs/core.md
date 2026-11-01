# The CPU in the weights

`qwendoom/` holds an RV32IM + Zba CPU compiled into a stock `Qwen3_5ForCausalLM` checkpoint. It
is a fork of Qwenputer's `chipformer.rv` (`experimental/riscv`, commit `bec1a0c`, MIT) and uses
its q35 compiler (`chipformer.q35.ir`, `prims`, `chipformer.dsl`) from the pinned package.

| File | What |
|---|---|
| `qwendoom/spec.py` | tokens, records, the address map (constants shared with `native/qd_mem.h`) |
| `qwendoom/ref.py` | `Rv32` (plain ISA emulator), `rom_word` / `rom_words` (the pre-decoded ROM), `Machine` (the traced reference) |
| `qwendoom/core.py` | the circuit (`build()`), its prediction decoding |
| `qwendoom/snapshot.py` | machine state -> the model's DeltaNet state rows (`state_rows`) |
| `qwendoom/weights.py` | the checkpoint emitter (`python -m qwendoom.weights`) |
| `qwendoom/hfcheck.py` | the checkpoint in stock transformers against the reference |
| `qwendoom/q35/evaluate.py` | a faster `ir.evaluate` (same values and checks) |
| `qwendoom/core_asm.py`, `core_elf.py`, `core_fuzz.py`, `core_check.py`, `core_measure.py` | assembler, ELF loader, fuzzer, teacher-forced checker, measurements |
| `tests/progs/` | the test programs (C, riscv-tests environment) and `build.py` |
| `tests/test_ref.py`, `test_core*.py`, `test_weights.py` | the tests |

## What changed from upstream

* **The address map** (ARCHITECTURE.md): RAM 256 KB at 0, DROM 512 KB at `0x80000`, ROM 256 KB
  at `0x100000`. The circuit derives every region test and table from `spec.py` (sizes are
  powers of 2 and bases multiples of their size, asserted). J / R payloads stay 16-bit ROM word
  indices (`PC = 0x100000 + 4 w`). Faults: loads outside RAM ∪ DROM, stores outside RAM, jumps
  outside ROM (and misaligned), as upstream.
* **The vocabulary is upstream's, id for id** (230,324 tokens, PAD = 230,324;
  `test_ref.test_vocabulary_is_upstreams`). BD and SC (the digit file) and GO keep their ids.
* **No digit-file coprocessor**: custom-0 / custom-1 are illegal instructions (fault 0). **No
  P3 text bridge.** DIV / REM (QS / QD records), MUL and Zba are upstream's.
* **Four value heads per key head** (Qwen3.5 head shapes: 16 key heads, 64 value heads of 128),
  so a sublayer holds 32 KB of RAM words and 256 KB of ROM / DROM bytes (below).
* **The input port**: `csrr rd, time` (CSR `0xC01`).
* Dead nodes of upstream's adders and DIV columns are not built (the top nibble's generate /
  propagate without a carry out, nibble 0's propagate, Q x |B| products no column reads) and only
  the token features the circuit reads exist: 2,231 IR dims, which fit a hidden size of 2304
  (9 ballast dims) instead of 2560.

## The input port and YIELD

`ref.Machine` (and the C trace machine) gives `csrr rd, time` (rd != x0) the next word of an
input queue (`Machine.inputs`); its record is that word's value record (`F0 | V F1 | V V F2`).
`csrr x0, time` consumes nothing (its record is NT). With an empty queue `record()` raises
`InputEmpty` and commits nothing.

The ROM folds every CSR read to a constant, so the circuit decodes `csrr rd, time` as
`lui rd, 0` and predicts `F0(0)`. The host injects the real record in its place. That works
because the register file is written from the token payloads (the V at t-1 / t-2 and the final
token's payload, `core.reg_columns`), not from the circuit's own value: whatever value record
arrives is committed. Inside an injected multi-token record the circuit's predictions (F1(0),
F2(0)) are not compared; everything after it is. The host-owned positions of a run are
`Machine.injected` (after `run()`), or `Machine.last_input` after each `record()`.

`tests/test_core.py::test_input_injection` proves it in the IR, teacher-forced: 1-, 2- and
3-token inputs (5, -1, 0x12345, 0xDEADBEEF), rd = a0, a1, a2, s3, t0, t1, t2, ra, sp, t6, back-to-
back reads (twice into the same register), `csrr x0`, a store / load through an injected RAM
address right after it, a load through an injected DROM address, a byte store through an
injected address, putchar of an injected byte, YIELD, then a poke loop (address, word pairs; 0
ends it). Every non-injected token is predicted exactly; the circuit's own prediction where each
input record starts is `F0(0)`; the register file's read ports hold exactly the injected values
at each record's last token. `test_input_poke_loop` and the C program `input.c` (sim_main's loop
in miniature: 18 injected tokens, 5 YIELDs) run the same way.

`ecall` with a7 = 1 and a0 = 256 emits `BYTE(256)` = YIELD (the payload is a0 & 511).

## The circuit

16 layers, 2,231 IR dims, 186 heads (at most 16 key heads per sublayer, 4 value heads each),
36,534 MLP units (at most 8,192 per sublayer), **1,521,437 nonzero weights** in the 16 circuit
layers (the exit's clears included). The emitted hidden size is 2304.

| Layer | Attention (key heads) | MLP units | Nonzero weights |
|---|---|---|---|
| L1 | pc, count, j, vcnt (4) | 198: n / j / v thermometers, the input code's clear | 1,987 |
| L2 | — | 1,116: PC word w, link w + 1, digits so far | 77,961 |
| L3 | ROM 16 (256 KB, pre-decoded words) | 4,090: field bits, uop and register one-hots, the link | 73,979 |
| L4 | registers x 3 ports, quotient heads x 2 (5) | 7,746: adder s1, MUL nibble products, shifter s1, logic, branch target s1, \|B\|, store data | 108,503 |
| L5 | lag (1): EA[2] at t-1 | 5,722: adder s2 (EA), shifter s2, MUL product splits, DIV m\|B\| / Q\|B\| products, \|A\| >> 4s, the signed quotient | 37,714 |
| L6 | DROM 16 | 2,936: MUL / REM column splits, Zba s1, DIV product splits, k0, region tests, misaligned faults | 92,309 |
| L7 | DROM 16 (512 KB) | 8,192: MUL adder s1, the 15 DIV compare columns, DROM bytes to bits, Zba result | 221,961 |
| L8 | RAM 16 | 4,379: DROM bits, MUL result, DIV lookahead, branch decision, range faults | 146,974 |
| L9 | RAM 16 | 290: the DIV compare signs | 105,603 |
| L10 | RAM 16 | 37: early output-select bits | 103,865 |
| L11-L14 | RAM 16 each | 0 | 103,104 each |
| L15 | RAM 16 (256 KB of words) | 1,203: loads (lanes straight from the word), SB / SH merge, rd value, its size | 112,568 |
| L16 | — | 625: the output code (+ the exit's 2,209 dim clears and 9 ballast clears) | 25,597 |

The depth is the memory: the effective address is ready after L5, then 10 table sublayers (2
DROM + 8 RAM), the value select and the exit. The MLPs of L9-L14 are almost empty; the work is
front-loaded (L4, L5, L7).

### Tables

| Table | Layout | Read | Write |
|---|---|---|---|
| ROM | 16 key heads x 4 value heads x 64 row pairs x 16 words; the pre-decoded word's 4 bytes as byte values (`core.QdCore.byte_table`) | 2-hot query {p, p + 64} (exact for bytes) | snapshot only |
| DROM | the same layout, 32 key heads (2 sublayers) | 2-hot | snapshot only |
| RAM | 128 key heads x 4 value heads x 128 rows x 1 word: value head = word index bits 4, 0 (EA[6], EA[2]), row = bits 5..11 (EA[13:7]), key head = bits 1, 2, 3, 12..15 (`core.ram_layout`) | one-hot query; z picks the value head | key: (row, head) of EA at t-1 AND a store at t-1; beta at t: ST0/ST1/SS0/SS1 (EA[6]) AND the lag head's EA[2] |
| registers | 3 heads (RS1, RS2, RD-slot ports), rows 1..31 | one-hot | key rd(t-1); beta F0/F1/F2/J |
| quotient | 2 heads x 4 value heads (8 digits) | one-hot gated by nd | at QD tokens |
| pc, count, j, vcnt | 1 head each | | J / R payload; 2^-k counters |
| lag | 1 head: EA[2] (stage-1 digit bit) of t-1 | one-hot | every token |

Every read is one-hot or 2-hot (upstream's docs/rv32.md: multi-hot reads of often-rewritten rows
left fp32 residues in fla's chunk kernel).

### 2 or 4 value heads per key head

Measured (`python -m qwendoom.core_measure --variants`), the same circuit and address map:

| Value heads per key head | Layers | Heads | RAM sublayers | DROM / ROM sublayers | Nonzero weights | Hidden |
|---|---|---|---|---|---|---|
| 2 (upstream's shape) | 27 | 363 | 16 | 4 / 2 | 2,205,431 | 2304 |
| **4 (built)** | **16** | **186** | **8** | **2 / 1** | **1,521,437** | **2304** |

Four value heads cut 11 layers and 31% of the nonzero weights (the per-token work of a sparse
engine), at the price of one small head (the lag head: a RAM write's beta has no conv tap, so
the second value-head bit EA[2] comes from t-1 through a head) and twice the value heads per
sublayer in a dense engine.

### Before / after the fork

| | Layers | IR dims | Heads | MLP units | Nonzero weights | Hidden |
|---|---|---|---|---|---|---|
| upstream `chipformer.rv.core` (digit file; RAM 32 KB, DROM 128 KB, ROM 256 KB; 2 value heads) | 11 | 2,679 | 99 | 42,014 | 1,048,631 | 2816 |
| of which the digit file (`bn*` nodes) | | 277 | 8 | 4,103 | | |
| qwendoom (no digit file, no bridge; RAM 256 KB, DROM 512 KB, ROM 256 KB; 4 value heads) | 16 | 2,231 | 186 | 36,534 | 1,521,437 | 2304 |

(Nonzeros counted the same way for both: `qwendoom.weights.chip_layer`, exit clears included.)
The extra layers and nonzeros are the 8x RAM and 4x DROM: 128 RAM heads and 32 DROM heads
against upstream's 32 + 16.

## Tokens per instruction

Reference traces of the C programs (`tests/progs`, clang -O2 rv32im_zba):

| Program | Instructions | Tokens | tok/instr |
|---|---|---|---|
| bytes (byte / half stores, memmove, packed structs) | 35,481 | 65,088 | 1.83 |
| classic (crc32, sieve, shell sort, strings) | 55,199 | 69,404 | 1.26 |
| div64 (64-bit mul / div / rem libcalls) | 166,840 | 186,731 | 1.12 |
| funcptr (tables in RAM and DROM, callbacks) | 8,129 | 12,434 | 1.53 |
| input (pokes through the input port, 5 YIELDs) | 14,248 | 25,417 | 1.78 |
| memsweep (all of RAM and DROM) | 112,436 | 181,517 | 1.61 |
| recurse (1,200 deep, stack at the RAM's top: 2-token sp) | 39,161 | 66,923 | 1.71 |
| switch (jump tables) | 5,235 | 6,256 | 1.20 |
| **all** | **436,729** | **613,770** | **1.41** |

Byte / half stores cost 4 tokens (V V V SS), values outside ±16K cost 2-3.

## The snapshot

```python
from qwendoom import snapshot
rows = snapshot.state_rows(ram, drom, rom, regs, pc)  # bytes(256K), bytes(<=512K), bytes(<=256K) raw code, 32 ints
rows.layer, rows.vhead, rows.row   # int32 (N,): checkpoint layer (0-based), value head in the layer, state row
rows.vec                           # float32 (N, 128)
rows.first                         # GO: the token to step after loading
```

Full images take 85 ms (90,221 rows, 46 MB; the ROM is pre-decoded with `ref.rom_words`,
vectorized). Zero RAM words get no row; ROM and DROM rows are always written (they carry the
ballast the 2-hot reads need). `state_rows(..., heads=meta["heads"])` uses the map from
`weights/qwendoom.json` instead of building the circuit (about 2 s, once per process).

**A run** (the engine, stock transformers and the IR alike):

1. reset: every recurrent state and conv window 0;
2. step PAD (230,324: a zero embedding row, so every state and window stays 0);
3. write each row: `recurrent_state[layer][vhead][row, :] = vec`; every other row stays 0;
4. step `rows.first` (GO); its prediction is the first token of the record at `pc`; then feed
   each predicted (or host-injected) token.

## The checkpoint

`.venv/bin/python -m qwendoom.weights --out weights` (12 s) writes:

* `config.json`: `Qwen3_5ForCausalLM`, `qwen3_5_text`, hidden 2304, 17 layers (16 linear
  attention + 1 all-zero full attention), 16 key heads, 64 value heads, head dims 128,
  intermediate 8192, vocabulary 230,336 (230,324 chip tokens, PAD, zero rows), tied embeddings;
* `generation_config.json` (eos HALT, pad PAD, greedy);
* `model.safetensors`: bf16, 2,572,356,864 parameters, **5,144,741,064 bytes**; 10,449,740
  nonzero (1,521,437 in the circuit layers, 8,925,999 in the embedding); streamed one layer at a
  time;
* `qwendoom.json`: the vocabulary, the output code (dims, family / payload bits, decoding), the
  layout, every table's (layer, key head, value heads), the state-row encoding, the address map,
  the I/O, the run procedure, the circuit's stats and nonzeros;
* `SHA256SUMS`.

The emitter is chipformer's `emit_layers` in numpy (`test_weights` checks every layer equal, bit
for bit) and checks every weight is a bf16 value.

**Stock transformers** (5.22, bf16, CPU, the torch reference kernels; `python -m qwendoom.hfcheck
weights --progs call,input,rv32um-div`): the checkpoint loads with no custom code; the snapshot
goes into the DeltaNet cache after a PAD step; then, for each of `call` (recursion with MUL,
DIV / REM, RAM and DROM loads, a byte store, Zba, putchar, YIELD; 160 tokens), `input` (1-, 2-
and 3-token injected inputs, a store / load through an injected address; 20 tokens, 8 of them
injected) and riscv-tests `rv32um-div` (142 tokens):

| Check | call | input | rv32um-div |
|---|---|---|---|
| residual before the exit == the IR's, all 2,192 circuit dims, every position | exact | exact | exact |
| teacher-forced, one prefill (chunked delta rule) | 159 / 159 | 11 / 11 | 141 / 141 |
| teacher-forced, token by token (recurrent delta rule) | 159 / 159 | 11 / 11 | 141 / 141 |
| free-running greedy decoding (host injects only the input records) | equal | equal | equal |

Token-by-token steps run on one CPU thread (0.31 s per token): transformers' depthwise
`causal_conv1d_update` fallback spends nearly all its time in OpenMP fork / join on 18 threads.

## Verification

| Suite | What | Result |
|---|---|---|
| `tests/test_ref.py` | vocabulary == upstream's; constants == `qd_mem.h`; value records; `rom_words` == `rom_word` (all programs, 30k random words); the ISA emulator passes all 51 riscv-tests; C outputs equal plain Python (crc32, primes, the div64 sums, switch); `Machine` == `Rv32` on all 59 programs (registers, RAM, output, instruction count); 40k DIV records; 18 faults; the input port; YIELD | 197 passed |
| `tests/test_core.py` | the circuit's shape; the fast evaluator == `ir.evaluate`; asm programs, the map's corners, R refresh, snapshots mid-run, 19 faults, CSR stubs, putchar / YIELD, the input port, ALU / branch / MUL / DIV fuzz, Zba + DROM; `state_rows` speed and equality with the IR snapshot, and a run from it | 47 passed |
| `tests/test_core_riscv.py` | rv32ui (40), rv32um (8), rv32uzba (3), teacher-forced with the HF-kernel read replay | 52 passed |
| `tests/test_core_fuzz.py` | random programs (every op, loads / stores over all of RAM and DROM, calls, putchar, YIELD); the 20 long ones are 186,741 instructions, 295,358 tokens | 4 + 20 (QD_SLOW): all passed |
| `tests/test_core_progs.py` | the 8 C programs, 436,729 instructions, 613,770 tokens teacher-forced | 2 + 6 (QD_SLOW): all passed (15 min) |
| `tests/test_weights.py` | layers == chipformer's emitter; embedding == the IR's features; files, header, hashes, saved tensors; transformers (QD_SLOW) | 6 + 2 (QD_SLOW): all passed |

Without QD_SLOW: 308 passed, 28 skipped in 103 s. The C trace machine (`native/trace`, another
track) produces the same tokens as `ref.Machine` on all 59 programs.

Every IR run checks exactness on every node (bf16 values, silu ranges, one-hot keys, write
gates, ballast absorption) and compares every non-injected prediction.

## Building and verifying

```sh
.venv/bin/python tests/progs/build.py                  # riscv-tests + C programs -> build/core/progs
.venv/bin/python -m pytest tests/test_ref.py tests/test_core*.py tests/test_weights.py
QD_SLOW=1 .venv/bin/python -m pytest tests/test_core_progs.py tests/test_core_fuzz.py   # ~22 min
.venv/bin/python -m qwendoom.weights --out weights     # the checkpoint
.venv/bin/python -m qwendoom.hfcheck weights            # stock transformers, bf16, CPU
QD_SLOW=1 .venv/bin/python -m pytest tests/test_weights.py
.venv/bin/python -m qwendoom.core_measure --variants --tokens   # the numbers above
```

## Limits

* The HF-kernel read replay (`ir.kernel_check`) runs on traces up to 1,500 tokens (all
  riscv-tests and the asm tests), as upstream; longer traces use the IR's direct reads.
* Stock transformers was run on the CPU only (bf16, the torch fallbacks; no fla / causal-conv1d
  kernels), on 3 short programs (322 tokens).
* The checkpoint is dense: 2.57B parameters of which 10.4M are nonzero (5.1 GB in bf16). Residual
  slot sharing (`chipformer.q35.emit.allocate`) could shrink the hidden size further; it is not
  used, because a handover is exact only when the old and new values differ by a bf16 value.
* The MLPs of L9-L14 are almost empty: the depth is set by the 10 table sublayers.

