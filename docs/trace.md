# The trace machine, the decoder and the lockstep checker

`native/trace` builds `build/libqdtrace.dylib` (`make -C native/trace`; `qwendoom/trace.py` loads
it with ctypes and rebuilds it when a source is newer). It holds three things:

| Part | C | Python | Used by the sim modes |
|---|---|---|---|
| **qd_ref**, the reference traced machine | `qd_ref_*` | `trace.Ref` | `ref` (emits the reference tokens), `check` (inside the checker) |
| **qd_dec**, the trace decoder (the mirror) | `qd_dec_*` | `trace.Decoder` | `model`, `check`: turns the model's tokens back into registers, PC and RAM |
| **qd_chk**, the lockstep checker | `qd_chk_*` | `trace.Checker` | `check`: every model token must equal the reference's |

`qd_ref` is a token-exact C port of `Machine.record()` in Qwenputer's `chipformer/rv/ref.py`
(MIT, (c) 2026 Divyansh Agrawal) and of `qwendoom.ref.Machine`, with qwendoom's changes (below).
Token ids come from `native/trace/qd_tokens.h`, generated from the installed `chipformer.rv.spec`
by `native/trace/gen_tokens.py` (`make -C native/trace tokens`; `--check` fails if it is stale,
and a test runs that check), so they cannot drift from the vocabulary.

## Memory and the address map

All three work on a `qd_mem` (`native/qd_mem.h`: RAM, DROM, ROM, HOST pointers; HOST is never
touched here). RAM is read and written in place; DROM is read; ROM holds raw RISC-V instruction
words and is read through a pre-decode cache keyed by the raw word, so rewriting the ROM needs no
invalidation call.

The map is a runtime parameter, `qd_map`:

```c
typedef struct {
    uint32_t ram_base, ram_size, drom_base, drom_size, rom_base, rom_size;
    int32_t input_csr;   /* 0xC01 (time), or -1: no input port */
} qd_map;
void qd_map_qwendoom(qd_map *m);  /* RAM 256 KB @0, DROM 512 KB @0x80000, ROM 256 KB @0x100000, time = input */
void qd_map_upstream(qd_map *m);  /* RAM 32 KB @0, DROM 128 KB @0x20000, ROM 256 KB @0x40000, no input */
```

`rom_size / 4` must be at most 65,536 (J / R payloads are 16-bit ROM word indices). Python:
`Map.qwendoom()`, `Map.upstream()`; `Memory(map)` allocates the three regions as numpy byte
arrays (`load_elf(path, skip_outside=False)`, `load_program`, `write`, `read`, `word`, `copy`;
`skip_outside=True` skips segments in none of the three regions, such as `doom.elf`'s HOST ones).
Another component's memory is used in place: `Memory.from_arrays(host.ram, host.drom, host.rom)`
(e.g. `qwendoom.host.Host`'s live views) or `Memory.wrap(qd_mem_address, map)`; with
`own_ram=True` the RAM is a private copy while DROM and ROM stay shared, which is what a
checker's reference (or a stand-in model) needs, and `sync_ram()` re-copies the source's RAM at
a resync.

## The record rules (exact)

A trace is one record per instruction. State changes only at a record's last token. The PC is
implied: `PC = rom_base + 4 * (anchor + n)`, where `anchor` is the payload of the latest J or R and
`n` counts the commits since then.

**Before every instruction:** if `n == 63`, the record is `R(anchor + 63)`: anchor += 63, n = 0,
nothing executes (not counted as an instruction). Then the instruction at the PC decodes (as
`ref.decode`, with AUIPC's `pc + imm`, branch and JAL targets folded in like the ROM word does):

| Instruction | Record | Commit |
|---|---|---|
| ALU, ALU-imm, MUL*, SH*ADD, LUI, AUIPC, CSR stub, load; rd != x0 | value record of the result | rd = value, n += 1 |
| the same with rd = x0, FENCE (any `0x0F`) | `NT` | n += 1 |
| DIV, DIVU, REM, REMU, rd != x0 | `QS(k0) QD(k<<4 \| q_k)` for k = k0..7, then the value record | rd = value, n += 1 |
| DIV-class, rd = x0 | `NT` | n += 1 |
| branch not taken | `NT` (even if its target is bad) | n += 1 |
| branch taken, JAL, JALR (target good) | `J(word)`, word = (target - rom_base) / 4 | link rd = pc + 4 (JAL / JALR, rd != x0), anchor = word, n = 0 |
| SW | `ST0` / `ST1` (bit 6 of the address's offset in RAM) | RAM word = rs2, n += 1 |
| SB, SH | `V(m & 0x7FFF) V(m >> 15 & 0x7FFF) V(m >> 30) SS0/SS1`, m = the RAM word with the byte / half merged in | RAM word = m, n += 1 |
| ECALL, a7 = 1 | `BYTE(a0 & 511)` (256 = YIELD, the tick is done) | n += 1 |
| ECALL, a7 = 93 | `HALT` | exit code a0, n += 1, halted |
| `csrr rd, time`, rd != x0 (qwendoom map) | value record of the next input word (**host-owned**) | rd = word, n += 1 |
| a fault | `FLT(cause) HEX×8 HALT` (the PC, most significant nibble first) | halted; n unchanged |

**Value record** of a 32-bit v (s = v as signed): `F0(v & 0x7FFF)` if -2^14 <= s < 2^14; else
`V(v & 0x7FFF) F1(v >> 15 & 0x7FFF)` if -2^29 <= s < 2^29; else
`V(v & 0x7FFF) V(v >> 15 & 0x7FFF) F2(v >> 30)`. Always the shortest (the decoder rejects any other
size).

**DIV-class record** (`ref.div_start`, `ref.div_parts`): with `nib(v)` the number of nonzero
nibble prefixes of v, and sb = (DIV or REM) and b < 0: `k0 = 0` if b == 0, else
`clamp(7 - nib(a) + nib(sb ? ~b : b), 0, 8)`. q = |a| / |b| (|x| only for the signed ops;
q = 0xFFFFFFFF if b == 0); the digits are `q >> 4(7-k) & 15`. The result is RISC-V's (DIV by 0 =
-1, REM by 0 = a, the overflow case INT_MIN / -1 = INT_MIN rem 0).

**Faults**, causes in `FLT` order: 0 illegal, 1 ebreak, 2 misaligned load, 3 misaligned store,
4 load out of range, 5 store out of range, 6 bad jump, 7 ecall. Checks in this order:

* loads: misaligned (ea % size) first, then ea outside RAM ∪ DROM;
* stores: misaligned first, then ea outside RAM (DROM and ROM are not storable);
* taken branch / JAL / JALR: target not 4-aligned or outside ROM (JALR clears bit 0 first); the
  link is not written;
* ECALL with a7 not 1 or 93; EBREAK; ILLEGAL: any undecodable word, the custom-0 (`0x0B`) and
  custom-1 (`0x2B`) opcodes (upstream's digit file is removed here), CSRRW / CSRRWI, any CSR op
  with rs1 != 0, any CSR not a stub, `mret`, `wfi`...

**CSR stubs** (pure reads: CSRRS / CSRRC / CSRRSI / CSRRCI with rs1 = 0): cycle, time, instret,
cycleh, timeh, instreth, mhartid read 0; misa reads 0x40001100. With the qwendoom map, `time`
(0xC01) is the **input port**: with rd != x0 it pops the next word of the input queue and its
record is that word's value record (the circuit predicts F0(0); the host injects the real record,
so checkers do not compare these tokens). `csrr x0, time` records `NT` and pops nothing (as
`qwendoom.ref.Machine`). With an empty queue the machine stops before the instruction
(`QD_REF_NEED_INPUT`, nothing committed) until words are pushed.

**Machine errors** with no record: the PC (anchor + n) outside the ROM, or an R refresh whose
payload would leave it: `QD_REF_ERROR`, nothing emitted (upstream's Python raises there).

A record is at most 12 tokens (`QD_MAX_RECORD` = 16 is the buffer size to use).

## qd_ref: the reference machine

```c
qd_ref *qd_ref_new(const qd_mem *mem, const qd_map *map);     /* RAM written in place */
void qd_ref_set_context(qd_ref *, uint32_t anchor, uint32_t n, const uint32_t regs[32]);
void qd_ref_push_input(qd_ref *, const uint32_t *words, size_t n);
int  qd_ref_record(qd_ref *, int32_t out[16], int *flags);   /* one record; 0: see status */
long qd_ref_run(qd_ref *, int32_t *toks, uint8_t *flags, long cap, long max_records,
                int stop_mask, int *reason);                  /* toks NULL: count only */
```

`qd_ref_run` runs records until `max_records`, a full buffer (fewer than 16 free slots), HALT, a
fault, a needed input word, an error, or (with `QD_STOP_YIELD` / `QD_STOP_BYTE`) a YIELD / BYTE
record, which is included. It returns the tokens produced; `flags[i]` gets `QD_TF_FIRST` (a
record's first token) and `QD_TF_HOST` (an input record's tokens). Also: `qd_ref_get_context`,
`qd_ref_pc`, `qd_ref_inst` (the raw word at the PC), `qd_ref_status` (`RUNNING HALTED FAULTED
NEED_INPUT ERROR`), `qd_ref_exit_code`, `qd_ref_fault` (cause, pc), `qd_ref_counts` (instructions,
records, tokens), `qd_ref_last_store`, `qd_ref_wants_input`, `qd_ref_input_pending`,
`qd_ref_clear_input`. `qd_value_record(v, out)` and `qd_div_record(op, a, b, out)` expose the two
encoders.

Python: `Ref(mem)`, `.set_context(pc= | anchor=, n=, regs=)`, `.push_input(words)`, `.record()` ->
(tokens, host_owned), `.run(max_records, cap, stop_yield, stop_byte)` -> (int32 tokens, flags,
reason), `.run_count()`, `.trace()` (to the end), `.context`, `.regs`, `.pc`, `.status`,
`.exit_code`, `.fault`, `.counts`, `.last_store`, `.wants_input`; `encode_value_record(v)`,
`div_record(op, a, b)`.

A sim tick in `ref` mode: `push_input(host.take_pokes())` (the (addr, word) pairs and the closing
0), `run(stop_yield=True)` -> reason `"yield"`; at a resync, `set_context(pc=entry, regs)` with
the RAM copied from the host's (`Memory.from_arrays(..., own_ram=True)`, `sync_ram()`), or the
host's RAM itself when nothing else writes it during the tick.

## qd_dec: the decoder (the mirror)

The decoder reads tokens one at a time and keeps the state they imply: anchor / n / PC, the
register file (written only from value records' payloads and J links, never computed) and RAM
(stores applied to its `qd_mem`: SW writes `regs[rs2]` at `regs[rs1] + imm`; SB / SH write the
merged word from the three V payloads). It needs no earlier tokens: `set_context(anchor, n, regs)`
starts it anywhere (a snapshot), with RAM as of that point. The first token after
`set_context` may be `GO` (the run's start; accepted there only).

```c
qd_dec *qd_dec_new(const qd_mem *mem, const qd_map *map, int strict);
int qd_dec_feed(qd_dec *, int32_t tok);       /* >= 0 an event, < 0 an error (sticky) */
int qd_dec_feed_many(qd_dec *, const int32_t *toks, long n, int stop_mask, long *consumed);
int qd_dec_wants_input(const qd_dec *);       /* the next record is an input record */
```

Events: `MORE` (record continues), `COMMIT`, `BYTE` (`qd_dec_last_byte`), `YIELD` (BYTE(256)),
`HALT` (exit code = a0), `FAULT` (`qd_dec_fault`: cause, pc). `feed_many` stops at an error, at
HALT / FAULT, at an event in `stop_mask` (`1 << event`), and with `QD_DEC_STOP_INPUT` after a
commit that leaves an input record next.

**Checks.** The decoder knows the instruction at the PC, so every token is checked at its
position. Always (the grammar): the token is a chip token; its family can start / continue this
instruction's record (e.g. an ADD with rd != x0 takes `V`/`F0` only); value records are well formed
and the shortest; `R` comes exactly at n = 63 with payload anchor + 63; QS <= 8 and QD indices run
k0..7; the SB / SH merged word's top chunk is <= 3; a static J target (branch, JAL) is the encoded
one; a fault's cause is one the instruction can raise and its HEX digits spell the PC; a store has
a valid RAM address (or it cannot be applied); nothing follows HALT. **Strict** (default) adds
everything the decoder can derive from its own registers and memory without computing an rd
value: branch directions, JALR targets, whether a load / store faults and with which cause,
ST / SS bank bits, the SB / SH merged word, DIV k0, the ecall kind and BYTE payload. Not checked
(the model's word): rd values (ALU, MUL, loads, DIV digits and results, input words).

Errors (`qd_dec_error`, `qd_dec_error_text` has the token index, the record, the PC, the
instruction and what was expected): `-1 TOKEN` not a chip token, `-2 SHAPE` family not allowed
here, `-3 REFRESH`, `-4 VALUE`, `-5 DIV`, `-6 JUMP`, `-7 STORE`, `-8 FAULT`, `-9 ECALL`, `-10 PC`
(the PC left the ROM), `-11 HALTED` (a token after the end).

**Input injection** (the model run): at a record boundary where `qd_dec_wants_input()` is true,
the host takes the next input word, feeds `qd_value_record(word)` (1 to 3 tokens) to the engine
and to the decoder instead of the model's predictions, and continues with the model's prediction
after the last injected token.

Python: `Decoder(mem, strict=True)`, `.set_context(...)`, `.feed(t)`, `.feed_many(toks, stop)` ->
(result, consumed), `.feed_checked(toks)` (raises `DecodeError`), `.wants_input`, `.at_boundary`,
`.context`, `.regs`, `.pc`, `.error`, `.error_text`, `.halted`, `.exit_code`, `.fault`,
`.last_byte`, `.last_store`, `.counts`.

## qd_chk: the lockstep checker

```c
qd_chk *qd_chk_new(qd_ref *ref);              /* ref from its current context, borrowed */
int  qd_chk_feed(qd_chk *, int32_t tok);      /* OK, END, DIVERGED (sticky), ERROR */
long qd_chk_stream(qd_chk *, const int32_t *toks, long n, int *state);
void qd_chk_divergence(const qd_chk *, qd_divergence *out);
```

At each record start the checker runs the reference's next record and compares token by token.
Input records are host-owned: the stream's value record there is parsed (a non-value token is a
divergence with `want = -2`) and not compared; if the reference's input queue is empty, the parsed
word becomes its input, so a checker needs nothing but the stream. A leading `GO` is skipped.
`END` comes when the reference's HALT / fault record has matched; a token after it diverges with
`want = -1`. The divergence holds the stream index, the record number and start, got / want, the
record's PC, anchor, n, raw instruction word and the reference's whole record. Python:
`Checker(ref)`, `.feed(t)`, `.check(toks)` -> (consumed, state), `.divergence` (printable),
`.counts`.

The reference writes its RAM: in `check` mode give it a RAM buffer of its own (its `qd_mem` may
share the DROM and ROM pointers), loaded from the same snapshot as the model.

## The model-mode loop (Python)

`start_stream(step, dec, checker=None)` begins a run after a resync (the decoder's and the
checker's reference's `set_context`, the checker's `reset`, the snapshot in the engine): it feeds
the run's first token, `GO`, to the decoder, to the checker if given, then to the engine, and
returns `step(GO)`, the first tick's `pred` (the runner's resync does the same). The model's
first prediction is then stream token 1, where both reject a `GO`: `GO` fed to the engine alone
would leave them fresh, and a model predicting `GO` after `GO` would pass as the run's start.
`GO` is no tick's token (`Tick.tokens` and the per-tick counts are the tick's own); it is the
checker's stream index 0, so its divergence indices count it.

`drive_tick(step, pred, dec, words, checker=None)` runs one sim tick through a model: `step(t)`
feeds one token to the engine and returns its greedy prediction, `pred` is the prediction
pending (after a resync, what `start_stream` returned). At every record boundary where the decoder
wants input, the next of the tick's `words` (the host's pokes, ending with 0) is injected: its
value record replaces the model's predictions token by token. Everywhere else the prediction is
the token. Each token goes to the decoder (its stores land in the decoder's Memory: in `model`
mode the host's RAM), to the checker if given (`check` mode), then to the engine. The tick ends
after `YIELD` (stepped, so `Tick.pred` is the next tick's first prediction), `HALT` or a fault
record. It returns a `Tick` (`tokens`, `host` mask, `event`, `pred`, `unread` words) and raises
`DecodeError` (an illegal token; input wanted with no word left) or `CheckError` (`.divergence`).

```python
dec = Decoder(Memory.from_arrays(host.ram, host.drom, host.rom))     # the mirror: host RAM
chk = Checker(Ref(Memory.from_arrays(host.ram, host.drom, host.rom, own_ram=True)))
# at a resync: host.resync(); load the snapshot into the engine; chk.ref.mem.sync_ram();
#   dec / chk.ref .set_context(pc=entry, regs=[0, 0, sp, gp, 0, ...]); chk.reset();
#   pred = start_stream(engine.step, dec, chk)       # GO to the decoder, the checker, the engine
pokes = host.take_pokes()
tick = drive_tick(engine.step, pred, dec, pokes, chk)
pred = tick.pred
host.complete_sim_tick()
```

`RefModel(ref)` is a stand-in engine for testing such a loop: a perfect model that predicts the
reference's tokens and, at an input record, the circuit's `F0(0)` (queue each tick's words on
its `ref`; `step(GO)` before its first token is the run's start; `corrupt={index: token}` plants
wrong predictions, index counting the tokens stepped after `GO`).

`qd_roundtrip(ref, ref_mem, dec, dec_mem, map, max_records, msg, len)` (Python
`trace.roundtrip(ref, dec)`) is test support: it runs the reference record by record into the
decoder and compares anchor, n and x0..x31 after every record, the stored word after every store,
the end state and all of RAM at the end.

## Validation (tests/test_trace_*.py)

`.venv/bin/python -m pytest tests/test_trace_*.py`: 399 tests, about 20-50 s.

* `test_trace_ref.py`: every upstream test ELF (65 non-qemu ELFs in
  `third_party/qwenputer/experimental/riscv/tests/data/rv`, upstream map) token for token against
  upstream's Python `ref.Machine`, with equal final registers, RAM, exit codes and instruction
  counts; `calc.elf` (digit file) equal up to its first custom op, where ours faults `illegal`
  (calc-bridge never reaches one). 200 upstream fuzz programs (`chipformer.rv.fuzz`); 40 at
  qwendoom's addresses (upstream's machine with the map and input port patched in); 10 against
  `qwendoom.ref.Machine` plus a poke loop; one program per fault cause; odd words (mret, wfi, CSR
  writes, custom ops, encoding holes); CSR stubs; refreshes; DIV edge operands; the encoders
  against `spec.value_record` / `ref.div_record` (12,000 DIV cases); snapshots at n > 0; 20,000
  random instruction words in random register files (every decode path) against upstream and
  through the decoder; the input port (queue, rd = x0, NEED_INPUT, host-owned flags).
* `test_trace_decoder.py`: round trips (C reference -> decoder, every record) on every upstream
  ELF in strict and lenient mode and on 60 fuzz programs; upstream's Python tokens -> decoder ->
  registers / stores / RAM equal the Python machine's (no C reference involved); snapshot starts;
  input injection with the hand-assembled poke loop and with a clang-built `sim_main`
  (ARCHITECTURE.md's) on the qwendoom map; `drive_tick` with `RefModel` (injection, the mirror
  and the checker in sync; a planted wrong value caught by the checker at its index, a planted
  grammar error by the decoder; after `start_stream` a GO predicted after GO rejected, where with
  GO in the model only it passed, and a normal tick's tokens the same either way); every error
  code on a corrupted stream, strict and lenient.
* `test_trace_check.py`: matching streams end with `END`; 300 random single-token corruptions are
  each reported at their own index with the right record; host-owned positions; a leading GO;
  `qd_tokens.h` is current; `qwendoom.spec` has the same vocabulary and the same address map,
  input CSR and YIELD as `qd_mem.h`.
* `test_trace_doom.py` (needs `build/doom.elf`, `build/libqdhost.dylib`, Freedoom; skipped
  otherwise): the real sim. (1) 1,050 ticks (30 s of game time) of Freedoom E1M1 with random
  keys (walking, turning, firing, using): every tick the host's pokes go to the C reference,
  which must YIELD with exactly the instruction and trace-token counts of the host's native sim
  mode (`native/host`, an independent interpreter) and leave identical RAM (all 256 KB); the
  decoder rebuilds that RAM and the registers from the tokens alone; the checker (input taken
  from the stream) accepts every token. 17,983,945 instructions, 23,758,174 tokens (22,627 per
  tick, 33,518 of them injected), the player at 949 places. (2) model mode end to end: one host
  runs natively, another runs its ticks through `drive_tick` with `RefModel` as the engine, the
  decoder writing that host's own RAM; RAM equal after each of 70 ticks, and the next frame
  identical (1,445,526 tokens).
* `test_trace_speed.py` prints the rates below (`pytest -s`).
* Sanitizers: `make -C native/trace sanitize` builds `build/libqdtrace-san.dylib` (ASan +
  UBSan, no recovery); the Makefile's header has the command line that runs the tests on it
  (`QD_TRACE_LIB`). All tests but the speed ones pass on it with no report.

Measured on the M5 Max (one thread; other tracks' jobs were running, and the spread is mostly
P- vs E-core scheduling): on c-sudoku / c-fib (upstream map) the reference machine runs
**130-300 M records/s** counting only and 97-212 M records/s keeping tokens; the decoder
**53-116 M tokens/s**; the checker (reference + compare) **44-113 M tokens/s**. On DOOM ticks
(qwendoom map, per-tick calls from Python) the reference runs 73-112 M records/s, the decoder
56-84 M tokens/s, the checker 48-72 M tokens/s: one tick (~17k instructions, ~22.6k tokens) is
0.15-0.25 ms. `drive_tick`'s Python loop (decoder + checker + `RefModel`) moves about 0.4 M
tokens/s. Upstream's Python `ref.Machine` runs 0.59 M records/s (c-fib, 2.12 M records in 3.6 s).
