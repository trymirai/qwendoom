# qwendoom architecture

DOOM's game logic (movement, monsters, weapons, doors, lifts: everything `P_Ticker` runs) executes
on an RV32IM + Zba CPU compiled into the weights of a stock `Qwen3_5ForCausalLM` checkpoint. A host
program plays the rest of the computer around it: it loads the WAD, draws the 3D view, status bar
and menus, reads the keyboard, and talks to the CPU only through the model's token stream. This
mirrors [nesformer](https://github.com/dbreunig/nesformer) (6502 in the weights, PPU in the host),
built on [Qwenputer](https://github.com/divagr18/qwenputer)'s q35 compiler and RV32 core.

Everything runs on Apple silicon (M-series, macOS).

## Repository layout

| Path | Contents | License |
|---|---|---|
| `qwendoom/` | Python package: circuit (`spec`, `ref`, `core`), checkpoint emitter (`weights`), host bindings, runner, server, CLI | MIT |
| `native/host/` | the host computer: a fast RV32IM+Zba interpreter that runs the non-sim part of DOOM, ecall services, dirty tracking, the native sim mode | MIT |
| `native/trace/` | the C reference traced machine (token-exact port of `qwendoom.ref.Machine`), the trace decoder (mirror), lockstep checker | MIT |
| `native/engine/` | the inference engine: Qwen3.5's forward pass over `model.safetensors`, exact, sparse, Apple-silicon optimized | MIT |
| `doom/` | our DOOM build: patches applied to fetched doomgeneric, the platform layer, the sim/host split, linker script, minimal libc | GPL-2.0 (DOOM-derived) |
| `web/` | the browser UI (static) | MIT |
| `scripts/` | `fetch_third_party.py` (uv script) and helpers | MIT |
| `third_party/` | **gitignored**, filled by `uv run scripts/fetch_third_party.py`: doomgeneric, qwenputer, nesformer, riscv-tests, freedoom | — |
| `build/` | **gitignored** build outputs (`doom.elf`, dylibs, caches) | — |
| `weights/` | the compiled checkpoint (`model.safetensors` gitignored; rebuilt by `python -m qwendoom build`) | — |

Rules: never vendor third-party code into the repo. Upstream Python comes in as uv dependencies
(`chipformer` is a pinned git source in `pyproject.toml`); C sources and data come from
`scripts/fetch_third_party.py`. Code we derive from Qwenputer/nesformer (MIT) lives in our files with
a header crediting the source; their notices are `LICENSE.chipformer` and `LICENSE.nesformer`. DOOM
changes are patches in `doom/patches/` plus new files in `doom/` (GPL-2.0 text: `doom/COPYING`).

Toolchain: Homebrew `llvm` clang (`/opt/homebrew/opt/llvm/bin/clang --target=riscv32-unknown-elf
-march=rv32im_zba -mabi=ilp32`), `ld.lld` (`/opt/homebrew/bin/ld.lld`), Apple clang for native
code, `uv` + Python 3.12 venv (`.venv`, `uv sync --extra build --extra dev --extra hf`).

## The address map (shared by the model, the host and every tool)

One flat 32-bit address space. The model holds RAM, DROM and ROM in its Gated DeltaNet state; the
host interpreter holds all four regions.

| Region | Range | Size | Model access | Contents |
|---|---|---|---|---|
| RAM | `0x0000_0000`–`0x0003_FFFF` | 256 KB | load/store (words) | sim stack, sim globals, the level zone (mutable level data, mobjs, thinkers) |
| DROM | `0x0008_0000`–`0x000F_FFFF` | 512 KB | load only | sim `.rodata` (tables, `states`, `mobjinfo`, strings), read-only level data (vertexes, segs, subsectors, nodes, blockmap, reject, sector line lists) |
| ROM | `0x0010_0000`–`0x0013_FFFF` | 256 KB | fetch only (pre-decoded) | sim `.text` |
| HOST | `0x0100_0000`–`0x04FF_FFFF` | 64 MB | none | host-only code, data, heap/zone, stacks, framebuffers |

RAM layout: `0x0000`–`0x00FF` null guard; sim stack from `0x3000` down to `0x0100` (sp stays below
16 KB, so its values are 1-token `F0` records); sim `.sdata/.sbss` (gp-relative), `.data`, `.bss`
from `0x3000`; the level zone from the end of `.bss` to `0x40000`.

C constants live in `native/qd_mem.h`; Python constants in `qwendoom/spec.py`. They must agree.

## The CPU in the weights

Fork of upstream `chipformer.rv` (Qwenputer `experimental/riscv`, commit `bec1a0c`) with:

* the address map above (RAM 256 KB, DROM 512 KB, ROM 256 KB at `0x100000`; J/R payloads stay
  16-bit ROM word indices: `PC = 0x100000 + 4 * w`);
* **the same token vocabulary as upstream `chipformer.rv.spec`** (same families, sizes, order and
  ids; PAD = `VOCAB`), so token ids are stable across every tool. Families that are no longer
  emitted (the digit file's `BD`, `SC`; the bridge's `GO`) keep their ids;
* the bignum digit-file coprocessor (custom-0 / custom-1) and the P3 text bridge removed if that is
  clean; custom-0/1 opcodes are then illegal instructions;
* DIV/REM (QS/QD records), MUL, Zba as upstream;
* faults as upstream with the new ranges: loads outside RAM ∪ DROM, stores outside RAM, jumps
  outside ROM.

### Records (unchanged from upstream; see `chipformer/rv/spec.py`)

`F0 | V F1 | V V F2` value records for rd writes; `QS QD…` + value record for DIV-class; `NT`;
`J(word)`; `ST0/ST1` (SW; data is rs2 in-circuit); `V V V SS0/SS1` (SB/SH merged word); `BYTE`
(putchar, `ecall a7=1`, payload `a0 & 511`); `HALT` (`ecall a7=93`); `FLT HEX×8 HALT` (fault);
`R(word)` anchor refresh before an instruction when n = 63.

### Host I/O through the token stream

* **Input (host → CPU): the `time` CSR.** `csrr rd, time` (CSR `0xC01`) is the input port. The ROM
  folds it to `rd = 0` like every CSR stub, so the circuit predicts `F0(0)`; the host **injects**
  the value record of the next input word in its place (`F0(v)`, or `V F1`, or `V V F2` by size),
  exactly as nesformer injects `IN` tokens. The register file is written from the token payloads, so
  rd receives the injected word. In the reference machines (`qwendoom.ref.Machine`, the C trace
  machine) `time` returns the next word of an input queue and the record is that word's value
  record; checkers treat those positions as host-owned (not compared). `timeh`, `cycle`, `instret`
  and the rest stay 0.
* **Output (CPU → host):** `ecall a7=1, a0=256` emits `BYTE(256)` = **YIELD** (the tick is done).
  `ecall a7=93` = exit/`I_Error` (`HALT`). Everything else the host needs (sound events, flags) is
  read from sim RAM through the mirror.

## The sim/host split of DOOM

One RV32 ELF (`build/doom.elf`) holds all of DOOM. Its linker script places every function the
playsim can reach from `P_Ticker` (and `sim_main`) in ROM, the data they touch in RAM/DROM, and
everything else in HOST. The same code runs in two places:

* the **host interpreter** (`native/host`, fast, region-unrestricted) runs `D_DoomMain`, the main
  loop, `G_Ticker` (minus `P_Ticker`), rendering, menus, status bar, automap, intermission, level
  loading (`P_SetupLevel`, which writes the level into RAM/DROM);
* the **sim** runs `sim_main`, compiled into ROM:

```c
void sim_main(void) {              /* entry after every resync; never returns */
    for (;;) {
        for (;;) {                 /* host pokes: (addr, word) pairs, addr 0 ends the list */
            uint32_t a = qd_in();  /* csrr a0, time */
            if (!a) break;
            *(volatile uint32_t *)a = qd_in();
        }
        P_Ticker();
        qd_yield();                /* ecall a7=1, a0=256 -> BYTE(256) */
    }
}
```

**Host → sim state:** the host interpreter tracks every RAM word it writes (dirty bitmap). Before
each tick it sends the dirty words as pokes (ticcmds copied into `players[].cmd`, `HU_Ticker`
clearing `message`, renderer marks like `ML_MAPPED`, `validcount`, menu flags…). DROM and ROM are
written only at level load. **Resync**: after a level load (or loading a save) the host snapshots
RAM, DROM, ROM and the sim's initial context (`pc = sim_main`, `sp = 0x3000`, `gp`, other regs 0)
into the model's DeltaNet state, as nesformer loads a cartridge. A DROM write by the host mid-level
forces a resync.

**Sim → host state:** every model store lands in the host's memory through the trace decoder (the
mirror). The host renderer reads it. `P_Ticker`'s pause checks (`paused`, `menuactive`) run on
the host side before a tick is requested (no tokens are spent while paused or in the menu).

**Sim modes** (selected by the runner):

| Mode | Who runs `sim_main` | Use |
|---|---|---|
| `native` | the host interpreter in a second context with region checks on | development, instruction counts, the fast reference |
| `ref` | the C trace machine, emitting the reference tokens | token-level reference |
| `model` | the transformer engine; the trace decoder mirrors its stores; the host injects input records | the real thing |
| `check` | `model` with `ref` in lockstep: every model token must equal the reference's | verification |

All modes consume the same per-tick input queue (the pokes), so instruction and token counts match
across modes.

## The engine

`native/engine` computes Qwen3.5's forward pass for one position (what `Qwen3_5ForCausalLM` computes
in bf16: RMSNorm with `1 + w`, the qkv/z/a/b projections, causal conv + silu, l2-normed q/k, the
gated delta rule in fp32, gated RMSNorm, `out_proj`, SwiGLU, final norm, tied head, greedy argmax),
rounding to bf16 where the bf16 model does, from `weights/model.safetensors`. It may only skip work
that is exactly zero (zero weight blocks, zero inputs, heads/rows a one-hot key/query does not
touch). Reference semantics: `third_party/nesformer/nesformer/engine.py`. API (C, loaded via ctypes):

```c
qd_engine *qd_engine_open(const char *weights_dir, int n_threads);
void qd_engine_close(qd_engine *);
void qd_engine_reset(qd_engine *);                       /* zero recurrent + conv state */
void qd_engine_set_row(qd_engine *, int layer, int value_head, int row, const float *vec128);
int  qd_engine_step(qd_engine *, int token);             /* feed one token, return greedy next */
```

A run is: reset, step(PAD), load the snapshot rows, then step the tokens (the first is the token
the snapshot's trace "ends" with, as `chipformer.rv.firmware` does).

## The checkpoint

`weights/`: `config.json` (`Qwen3_5ForCausalLM`, `model_type: qwen3_5_text`), `generation_config.json`,
`model.safetensors` (bf16), `qwendoom.json` (vocabulary layout, code dims, the table → (layer,
key head, value heads) map, the address map), `SHA256SUMS`. Qwen3.5 DeltaNet head shapes
(head dim 128, 16 key heads per layer, value heads a multiple of 16), the circuit layers, one
all-zero full-attention layer last (the HF cache needs one), tied embeddings. It loads with stock
`transformers` and needs no custom code.

As built: 16 circuit layers + 1 zero attention layer, hidden 2304, 16 key heads and 64 value heads
(4 per key head) per layer, intermediate 8192, vocabulary 230,336 (PAD = 230324, GO = 230322).
2.57B parameters (5.1 GB in bf16), of which 1.52M circuit weights (plus the embedding's 8.9M) are
nonzero. RAM is 8 sublayers of 32 KB, DROM 2 sublayers and ROM 1 sublayer of 256 KB byte tables.
`docs/core.md` has the layer plan.

The engine is the sparse CPU engine (`native/engine`, about 31k tokens/s on DOOM's ticks on one M5 Max P-core; more threads do not help).
A GPU path was measured and rejected: batch-1 decoding is a chain of ~20-30 tiny dependent kernels
per token, and Metal's dispatch and submit latency alone exceeds the CPU's whole step.

## The browser UI and server

`python -m qwendoom serve` (aiohttp) serves `web/` and a WebSocket `/ws`:

* server → client binary frames: `0x01` + 768-byte palette + 64,000 palette indices (320×200);
  JSON text messages `{"type": "stats", "tokens", "tps", "frames", "state"}` and
  `{"type": "status", ...}`.
* client → server JSON: `{"type": "start"}`, `{"type": "stop"}` (end the game, back to a dark
  idle screen; the Start button reads Stop while a game runs), `{"type": "pause"}`,
  `{"type": "resume"}` (the Pause button reads Resume while paused),
  `{"type": "key", "action": A, "down": bool}` with A one of `left right forward back strafeleft
  straferight fire use run menu enter weapon1 weapon2 weapon3` or `{"type": "key", "char": "y", ...}`
  with one printable ASCII character (`!`–`~`, lowercased; anything else is refused with a status
  error; the page sends a letter or digit typed on a non-Latin layout as its physical key).

The server refuses a `/ws` handshake whose `Origin` names another host:port (403), so other sites
open in the browser cannot drive the game; clients that send no `Origin` (CLI tools) are let in.
Against DNS rebinding (another site's name re-pointed at 127.0.0.1, which then sends a matching
`Origin` and `Host`), every request, the page and `/ws` alike, must carry a `Host` naming
`localhost`, a `.localhost` or `.local` name, an IP literal (IPv6 in brackets) or the `--host`
value (case-insensitive); anything else, or no `Host`, gets a 403. With `--host 0.0.0.0`, open the
page by LAN IP or the Mac's `.local` name, or pass the LAN name you use as `--host`.

A game source runs its game loop on a thread and publishes frames and stats; it checks its state
and publishes while holding the lock its control methods take, so nothing is published after a
pause or stop it has not seen (`qwendoom/sources.py`).

Controls: Q/← turn left, W/↑ forward, E/→ turn right, A strafe left, S/↓ backward, D strafe right,
Ctrl (or F) fire, Space use/open, Shift run, Esc game menu, Enter confirm, 1–3 weapons.
