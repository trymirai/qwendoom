# DOOM on the qwendoom machine: the sim/host split and the host computer

This track builds one RV32IM+Zba ELF, `build/doom.elf`, holding all of DOOM (doomgeneric, which
is Chocolate Doom based, GPL-2.0), and the **host computer** that runs it: an RV32 interpreter
in C (`native/host` → `build/libqdhost.dylib`, wrapped by `qwendoom/host.py`). The playsim
(`P_Ticker` and everything it reaches) is linked into ROM / RAM / DROM so that the CPU in the
model's weights can run it; the rest of DOOM (main loop, renderer, menus, status bar, level
loading) is linked into HOST and runs only on the interpreter. See ARCHITECTURE.md for the
address map and the sim modes.

| Path | What |
|---|---|
| `doom/Makefile` | `make -C doom`: patched sources → objects → `split.py` → linker script → `build/doom.elf` |
| `doom/patches/*.patch` | our changes to doomgeneric (applied to a fresh copy in `build/doom-src/`) |
| `doom/qd_platform.c` | the host side: `main`, DG_* / I_* glue, fake clock, video, sim-tick requests, the DROM level allocator |
| `doom/qd_sim.c` | the sim side: `sim_main`, the sound-event ring, `I_Error`, the AM_Stop flag |
| `doom/qd_doom.h` | the hooks shared by the patches and the two files above |
| `doom/libc/` | a minimal freestanding libc (strings, ctype, printf / scanf family, a host heap, stdio over ecalls, 64-bit division and the few soft-float libcalls the host side needs) |
| `doom/split.py` | sim reachability and the linker script; `check` verifies the linked ELF |
| `doom/measure.py` | the measurements below (a deterministic bot plays; every map sized) |
| `native/host/` | the interpreter (`qd_host.c`, `qd_interp.inc`), the guest ABI (`qd_host_abi.h`), a CLI (`build/qdhost`), riscv-tests env (`isa/`) |
| `qwendoom/host.py` | ctypes bindings (`Host`) |
| `tests/test_host.py`, `tests/test_host_isa.py` | tests (below) |

## Building

```sh
make -C doom          # build/doom.elf, build/doom.map, build/doom-split.json (~12 s from clean)
make -C native/host   # build/libqdhost.dylib, build/qdhost
make -C doom patches  # regenerate doom/patches/ from an edited copy in build/doom-edit/
```

The DOOM sources are copied from `third_party/doomgeneric/doomgeneric` into `build/doom-src/` and
patched there; nothing in `third_party/` is modified. Flags: `--target=riscv32-unknown-elf
-march=rv32im_zba -mabi=ilp32 -O2 -ffunction-sections -fdata-sections -mrelax
-msmall-data-limit=8 -mllvm -enable-global-merge=false` (no C extension, so no compressed
instructions; one section per function / variable so the split can place each; global merging
off so that host and sim variables never share a section). `ld.lld --relax --relax-gp
--gc-sections`.

## The split (`doom/split.py`)

`split.py analyze` reads the relocatable objects and walks the graph their relocations form
(section → sections it references) from these roots:

* `sim_main`;
* every function of a playsim object (`p_*.o` except `p_setup.o`, `p_saveg.o`) whose address is
  taken anywhere (a non-call relocation against its function symbol): the thinkers (`T_MoveFloor`,
  `T_LightFlash`, `P_MobjThinker`, ...) that host code stores into `thinker_t` at level load and
  the sim calls through `P_RunThinkers`, and the `PIT_` / `PTR_` iterators. Jump tables
  (`.rodata.F` → `.text.F` against local `.L` labels) do not count. The `A_*` actions come in
  through `states[]`, which is data the sim reads.

Everything reached is sim: code → ROM, read-only data → DROM, mutable data → RAM (small data
first, so it is gp-relative), except the tables that are mutable in C but never written after
startup (`states`, `mobjinfo`, `weaponinfo`, `clipammo`, the chase-direction tables:
`READONLY` in split.py), which go to DROM. Everything else is HOST. The result: 20 roots, **213 sim functions** (`build/doom-split.json` lists them with
their sizes and, for every sim section, the section that pulled it in: `python doom/split.py why
build/doom-split.json SYMBOL`).

`split.py check` (run by the Makefile after linking) verifies that the ROM holds exactly the
analysed functions, that **every ROM instruction is one the circuit runs** (RV32IM + Zba,
`csrr` of `time` only, `ecall`, FENCE; no FENCE.I, no atomics, no other CSR; static branch and
JAL targets inside ROM), and writes the linked sizes into the JSON (`linked`). Sim code reached
from host-named objects is reported unless whitelisted (`R_PointInSubsector`, `R_PointToAngle*`,
`G_ExitLevel`, `G_SecretExitLevel`, `I_Error`, `I_Tactile`); currently none.

The static analysis is backed by the **dynamic check** of the native sim mode: any fetch outside
ROM, load outside RAM ∪ DROM, store outside RAM, access below 0x100 (the null guard) or
misaligned access is an error naming the PC's symbol and the address's symbol and region, e.g.

    sim store (4 bytes) to 0x81068 (mobjinfo+0xf90, DROM) at pc 0x10e5f8 (P_PlayerThink+0x38, ROM)

No such error occurs in any run below.

## Memory

```
RAM   0x00000-0x000ff  null guard (the sim may not touch it)
      0x00100-0x02fff  sim stack (sp = 0x3000 at entry, grows down)
      0x03000-0x05040  sim .sdata/.sbss (gp = 0x3800), .data, .bss          8,256 B
      0x05040-0x3ffff  the level zone: PU_LEVEL / PU_LEVSPEC + the sim's    241,600 B
                       allocations + texture/flat translation tables
DROM  0x80000-...      sim .rodata (finesine, tantoangle, states, mobjinfo, strings)  93,448 B
      then             textureheight (startup, permanent)                     3,852 B
      then             level data, a bump region reset per level: vertexes, segs,
                       subsectors, nodes, blockmap, reject, sector line lists
ROM   0x100000-        sim code                                              61,576 B
HOST  0x1000000-       host code, data, heap (16 MB main zone inside), argv + stack at the top
```

The level zone is a second `memzone_t` (`z_zone.c` patch): `Z_Malloc` sends PU_LEVEL and
PU_LEVSPEC to it, everything else to the host's 16 MB zone; `Z_Free` picks the zone by address.
Read-only level data is allocated with `qd_drom_alloc` (patched `P_LoadVertexes`, `P_LoadSegs`,
`P_LoadSubsectors`, `P_LoadNodes`, `P_LoadBlockMap`, `P_LoadReject`, `P_GroupLines`) from a bump
region after the sim's `.rodata`, reset by `P_SetupLevel`. Lines, sides, sectors, mobjs and
thinkers stay in RAM: the sim writes them.

## DOOM changes (`doom/patches/`)

| Patch | Why |
|---|---|
| `g_game.c` | `G_Ticker` calls `qd_sim_tick()` (→ SIM_TICK) instead of `P_Ticker()`, after `P_Ticker`'s pause / menu checks, which moved here (no tokens are spent while paused or in a menu); `qd_after_sim_tick()` acts on the sim's flags. The map31 check of `G_SecretExitLevel` (WAD directory = host data) moved to `G_DoCompleted`. |
| `p_tick.c` | the pause / menu checks left `P_Ticker` |
| `p_inter.c` | `AM_Stop()` → `qd_sim_am_stop()` (a flag; the host closes the automap after the tick) |
| `p_setup.c` | read-only level data in DROM; `qd_level_begin` / `qd_level_loaded` around the load (→ resync) |
| `r_data.c` | `textureheight` in DROM (the sim reads it); `texturetranslation`, `flattranslation` in the level zone (the sim writes them) |
| `z_zone.c`, `z_zone.h` | the level zone; `-qdmeasure` (measurement only, below) |
| `p_map.c`, `p_spec.c`, `p_doors.c` | no command line or stdio in the sim: the overrun emulations use their defaults, warnings dropped |
| `f_finale.c` | the end-of-game pictures are PU_CACHE (PU_LEVEL would put them in sim RAM) |
| `d_main.c` | no attract demos: the title / credit pages cycle without playing DEMO1-4 |

## The platform layer

**Host side (`qd_platform.c`).** `main` sets `singletics` (exactly one tic per `TryRunTics`)
and runs `D_DoomMain` + `doomgeneric_Tick`. The clock is fake: `DG_GetTicksMs` is
`1000 + ceil(frames * 1000 / 35) + slept`, so one tic passes per frame and a run depends only on
its inputs. Video is DOOM's own 8-bit 320x200 screen and the current palette after the gamma
table (PLAYPAL 0..13: damage / pickup / radiation palettes as DOOM sets them): `I_FinishUpdate`
issues a FRAME ecall with both addresses. Keys come from the host's queue (`DG_GetKey`). No sound
or music (stubs). `I_ZoneBase` gives a 16 MB host zone.

**Sim side (`qd_sim.c`).** `sim_main` is exactly ARCHITECTURE.md's loop. Stubs the playsim
reaches: `S_StartSound` / `S_StopSound` append `{origin, sfx}` (sfx = -1 for a stop) to
`qd_sound_ring[32]` in sim RAM (`qd_sound_head` counts events: the host can read new ones after
each tick, for a later sound feature); `I_Error` stores the format and `va_list` in RAM
(`qd_error_fmt`, `qd_error_args`) and exits (`ecall a7=93`, a1 = fmt, a2 = args; the interpreter
formats the message); `I_Tactile` is empty; `DEH_String` is the identity macro (no DEHACKED).

**Game flow.** `qwendoom.host.DEFAULT_ARGS` = `-warp 1 1 -skill 3`: straight into Freedoom E1M1
at Hurt Me Plenty. Without `-warp`, the title pages cycle (no demos) and the menus start a game.
Esc opens the menu on the host (the sim is not ticked while it is up); F2 / F3 save / load when a
config directory is set (`.savegame/` inside it); a load is a level load, so a resync.

## The guest ABI (`native/host/qd_host_abi.h`)

| a7 | Service |
|---|---|
| 1 | putchar `a0` (in the sim, `a0 = 256` is **YIELD**) |
| 64 | write(fd 1/2, buf, len) |
| 93 | exit(a0); with a1 != 0: I_Error(fmt = a1, va_list = a2) |
| 0x400-0x407 | open / close / read / fwrite / seek / remove / rename / mkdir: paths are relative, no `.` / `..` / empty components, under the config directory (reads fall back to the WAD's directory) |
| 0x410 | FRAME(screen, palette) |
| 0x411 | SIM_TICK |
| 0x412 | GETKEY → `pressed << 8 \| doomkey`, or -1 |
| 0x413 | LEVEL(1: load starts, 0: load done → resync) |
| 0x414 | window title |

The sim may only use YIELD and exit, the two the circuit implements.

## The host computer (`native/host`)

A direct-threaded interpreter: each 4-byte instruction word of ROM (both contexts) and of the
host text is decoded once at load into `{handler, imm, rd, rs1, rs2}` (branch / JAL targets as
instruction deltas, AUIPC folded, x0 writes to a scratch slot); the PC lives as a pointer into the
decoded array. The code is never rewritten after load (DOOM has no self-modifying code; a host
store into ROM is an error). It is built with `-mllvm -enable-tail-merge=false` so that every
handler keeps its own indirect branch (worth ~35%).

Memory is one 80 MB anonymous mapping; `qd_mem`'s four pointers point into it.

**Host context:** unrestricted over the four regions (stores to ROM or unmapped addresses are
errors), with a **dirty bitmap of every RAM word it writes** and **DROM-write detection**
(any DROM write sets `need_resync`; writes outside a level load after the first level are
counted in `drom_writes_midlevel`, 0 in every run).

**Sim context:** its own registers and PC, entered at `sim_main` with `sp = 0x3000`,
`gp = __global_pointer$`, other registers 0; the region checks above; `csrr time` pops the input
queue (the pokes, then 0); it counts the circuit's trace tokens per tick (value records, `NT`,
`J`, `R` refreshes every 63 commits, DIV-class `QS/QD` digits, `SS` merged words, `BYTE`, `HALT`)
and per-uop instruction counts. The token and instruction counts equal the trace track's
reference machine's exactly (test below).

### C API (`native/host/qd_host.h`)

```c
qd_host *qd_host_new(const char *elf, const char *wad, int argc, const char *const *argv);
        /* guest argv = {"doom", "-iwad", basename(wad), argv...}; NULL on failure */
void qd_host_free(qd_host *);
void qd_host_set_config_dir(qd_host *, const char *dir);  /* enables writes (config, saves) */
void qd_host_set_echo(qd_host *, int);                    /* guest stdout/stderr to stderr */

int  qd_host_run(qd_host *, uint64_t max_instr);
        /* QD_EV_FRAME (1), QD_EV_SIM_TICK (2), QD_EV_EXIT (3), QD_EV_ERROR (-1), QD_EV_BUDGET (0).
           After SIM_TICK the host waits (run returns SIM_TICK again) until complete_sim_tick. */
qd_mem *qd_host_mem(qd_host *);
const char *qd_host_error(qd_host *);   int qd_host_exit_code(qd_host *);
int  qd_host_frame(qd_host *, uint8_t idx[64000], uint8_t pal[768]);   /* -1 before the first */
void qd_host_key(qd_host *, int doom_key, int down);
int  qd_host_action_key(const char *action);   /* ARCHITECTURE.md's UI actions -> DOOM keys */

int  qd_host_take_pokes(qd_host *, uint32_t *out, int max);
        /* the RAM words the host changed since the sim last saw RAM, as sorted (addr, value)
           pairs and a final 0: returns n = 2 * pokes + 1 (or -n if max < n, nothing taken).
           Only words whose value differs from the sim's view are sent. Clears the dirty set and
           queues the words as the native sim's input. */
int  qd_host_pending_pokes(qd_host *);
int  qd_host_need_resync(qd_host *);    /* a level was loaded (or DROM written) */
void qd_host_sim_entry(qd_host *, uint32_t *pc, uint32_t *sp, uint32_t *gp);
void qd_host_resync(qd_host *);         /* snapshot taken: reset the sim context, clear dirty */

int64_t qd_host_sim_tick_native(qd_host *, qd_sim_stats *);  /* instructions, or -1 (error) */
int  qd_host_complete_sim_tick(qd_host *);   /* RAM holds the sim's stores: resume the host */

uint64_t qd_host_instructions(qd_host *);  uint64_t qd_host_sim_instructions(qd_host *);
int  qd_host_counters(qd_host *, uint64_t *out, int n);  /* QD_CTR_*: frames, sim_ticks,
        resyncs, levels, drom_writes, drom_writes_midlevel, ram_writes, pokes,
        host_null_accesses, sim_tokens */
uint32_t qd_host_symbol(qd_host *, const char *);  int qd_host_symbolize(qd_host *, uint32_t, char *, int);
const char *qd_host_uop_name(int);  int qd_host_output(qd_host *, char *, int);
```

`qd_sim_stats`: instructions, tokens, input_words, loads, stores, drom_loads, min_sp, ops[64].

### A tick, in every sim mode

```python
from qwendoom.host import Host, FRAME, SIM_TICK
h = Host()                                   # E1M1, skill 3; config_dir=... enables saves
while True:
    ev = h.run()
    if ev == FRAME:
        idx, pal = h.frame()                 # (200, 320) palette indices, (256, 3) RGB
    elif ev == SIM_TICK:
        if h.need_resync():                  # after a level load / savegame load
            # model / ref: snapshot h.ram, h.drom, h.rom and h.sim_entry() (pc, sp, gp)
            h.resync()
        pokes = h.take_pokes()               # the input words csrr time returns, in order
        h.sim_tick_native()                  # model / ref: run to YIELD; the mirror writes the
        h.complete_sim_tick()                #   sim's stores into h.ram (Memory.wrap(h.mem_address))
```

`Host.native_tick()` and `Host.next_frame()` wrap the native case; `h.action("fire", True)`
presses a UI action. A model-mode runner that sees `HALT` reads the error from `qd_error_fmt`.

**What gets poked** (E1M1, per tic): `gametic` and `validcount` (1 each), `viewx` / `viewy`
(the renderer moves them; `R_PointToAngle2` in the sim overwrites them, as in vanilla), the
ticcmd in `players[].cmd` when the input changes, `sector_t.validcount` of the sectors the
renderer drew (~16; `R_AddSprites`), `line_t.flags` (`ML_MAPPED`) of newly seen walls, the
player's `message` when the HUD clears it. Median 24 pokes = 49 input words.

**Determinism:** the same ELF, WAD, arguments and key events give the same RAM, DROM, frames,
instruction and token counts (tested); nothing reads wall time.

## Measurements

Apple M5 Max, macOS; the other five tracks were running (load average 7-10), so speeds vary by up
to 2x between runs. `python doom/measure.py` (60 s of E1M1, `build/doom-measure.json`) and
`python doom/measure.py --maps all` (`build/doom-maps.json`).

**E1M1, 2,100 tics (60 s at 35 Hz), native mode, bot:** it walks the hall round the pit, opens
the door at (960, -152), fights the zombieman and shotgun guys in the room beyond (they and the
imps across the hall wake at its first shots: 18 monsters awake for most of the run), picks up
items, runs out of bullets and ends on the fist at 4% health. 2,138 frames.

| Per tic | min | median | p95 | max | mean |
|---|---|---|---|---|---|
| sim instructions | 9,692 | 22,245 | 54,720 | 141,596 | 26,855 |
| ... with monsters awake (2,046 tics) | 9,692 | 22,531 | 56,332 | 141,596 | 27,247 |
| ... none awake (54 tics) | 9,797 | 11,745 | 16,942 | 17,449 | 11,989 |
| trace tokens (1.346 per instruction) | 12,345 | 29,703 | 75,089 | 193,244 | 36,141 |
| pokes | 0 | 24 | 42 | 134 | 26.2 |
| loads / stores | 3,143 / 745 | 6,553 / 2,016 | 14,583 / 5,341 | 47,186 / 19,057 | |
| host instructions per frame | | 1.71 M | 2.02 M | 2.33 M | 1.66 M |

| Region | E1M1 peak | Size |
|---|---|---|
| ROM | 61,576 B (213 functions, 15,394 instructions; median function 244 B, largest `P_TouchSpecialThing` 2,072 B) | 262,144 B |
| DROM | 232,520 B (93,448 .rodata + 3,852 textureheight + 135,220 level data) | 524,288 B |
| RAM | sim data 8,256 B; stack 2,080 B (sp ≥ 0x27e0); level zone 175,148 B of 241,600 (at load 174,580: lines 75,224, mobjs 210 × 180, sides 36,604, sectors 16,040, blocklinks 3,480, translation tables 4,892) | 262,144 B |

Interpreter speed (`build/qdhost`, 2,100 frames): host 650-1,305 MIPS (best of 6 runs 1,305;
typical under load 650-970), sim context 270-590 MIPS (it also counts tokens and per-uop
statistics). Through Python (`measure.py`, ctypes per event): ~560 host MIPS.

**Every map of freedoom1.wad** (needs measured at load with `-qdmeasure`, which puts level data in
host memory so that any level loads; then 1,050 tics on the real build for the maps that fit):

| Fits (10/36) | E1M1, E1M5, E1M8, E2M2, E2M3, E2M8, E3M1, E3M2, E3M6, E3M8 |
|---|---|
| Does not fit (26/36): level zone needed at load vs 241,600 | E1M2 304,760; E1M3 290,204; E1M4 300,464; E1M6 373,112; E1M7 577,332; E1M9 272,248; E2M1 246,204; E2M4 454,576; E2M5 298,560; E2M6 305,032; E2M7 585,108; E2M9 741,372; E3M3 558,612; E3M4 333,632; E3M5 397,148; E3M7 339,376; E3M9 589,176; E4M1 272,452; E4M2 624,804; E4M3 319,368; E4M4 298,440; E4M5 308,324; E4M6 422,972; E4M7 781,236; E4M8 307,328; E4M9 576,468 |
| DROM level data over 426,988 available | E1M7 487,080; E2M7 553,136; E2M9 668,140; E3M3 498,708; E3M9 531,364; E4M2 504,896; E4M7 670,252; E4M9 433,108 |

Sim instructions per tic on the maps that fit (bot holding its ground): median 3.8k (E1M8) to
45k (E1M5), max 142k (E1M1).

**Plan for the rest (the address map is not ours to change):** (1) as is, every map fits with
RAM ≥ ~880 KB and DROM ≥ ~770 KB (E4M7), i.e. 1 MB each. (2) Without growing RAM: `line_t` is
64 B of which the sim writes only `special`, `validcount` (and reads `flags`, written by the
renderer); moving the immutable 56 B per line and the immutable 12 B per side (offsets, sector) to
DROM would make 23/36 maps fit the current 256 KB RAM / 512 KB DROM (with 10% zone headroom);
5 more (E1M6, E2M4, E3M5, E3M7, E4M6) would then need only a bigger DROM, and 8 need both.
That split touches every `line_t` / `side_t` access in DOOM (playsim, renderer, automap), so it is
a sizeable patch.

## Tests

* `tests/test_host.py` (15 tests, ~15 s): boot into E1M1 (skill, palette = gamma(PLAYPAL 0), one
  tic per frame), every frame's palette one of PLAYPAL's 14 with damage and pickup palettes seen, a scripted run with poke-format checks, determinism (two identical runs equal,
  a third with one more key differs), pokes completeness (a mirror fed only the snapshot, the
  pokes and the sim's stores equals the host's RAM before every tick), **the native sim against
  the trace track's reference machine** (same RAM after every tick, same instruction and token
  counts; skipped if `libqdtrace` is unavailable), the four region checks (host load, DROM
  store, null guard, fetch from HOST) with their messages, the menu pausing the sim, the title
  without demos then a new game from the menu, savegame save and load (resync, then pokes still
  complete), the action-key table, and the split's placement.
* `tests/test_host_isa.py` (104 tests): riscv-tests rv32ui (minus `fence_i`, `ma_data`), rv32um,
  rv32uzba, each as sim code in ROM (native sim mode) and as host code, plus a test that fails on
  purpose to show failures are caught.
* Longer checks run by hand: `python doom/measure.py --ref` compares every one of the 2,100 tics
  of the measurement run with the reference machine (0 mismatches: RAM, instructions, tokens), and
  `python doom/measure.py --maps all`.

## Limitations

* 26 of Freedoom's 36 maps do not fit the current RAM (above).
* No sound or music; sound events are only recorded in the ring.
* Savegames and the config file need `config_dir`; without one, writes fail quietly as in a
  read-only directory.
* Host instruction counts include the renderer's whole frame; the screen wipe at a level start
  draws ~38 frames without tics.
