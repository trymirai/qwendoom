# The runner, the model game source and the command line

The runner plays DOOM on the host computer (`native/host`) and answers every sim tick
(`P_Ticker`) in one of the sim modes of ARCHITECTURE.md. In the `model` mode the transformer
(`native/engine`) runs `sim_main` token by token: the host injects the input words, the trace
decoder (`native/trace`) mirrors the model's stores into the host's RAM, and the host draws the
frame from that RAM.

| Path | What |
|---|---|
| `native/runner/qd_run.h`, `qd_run.c` | the hot loop: one tick of the model / check / ref modes in C (`build/libqdrun.dylib`) |
| `qwendoom/runner.py` | `Runner` (host + engine + decoder / checker, resyncs, frames, keys, counters), button scripts, I_Error's message |
| `qwendoom/sources.py` | `ModelSource` (`--source model`, the server's default): the runner behind the browser UI |
| `qwendoom/cli.py`, `qwendoom/__main__.py` | `python -m qwendoom serve / run / check / bench / build / info` (also the `qwendoom` script) |
| `tests/test_runner.py`, `tests/test_cli.py` | the tests (below) |

## Modes

| Mode | The tokens come from | Use |
|---|---|---|
| `model` | the engine's greedy predictions, the input records injected by the host | the real thing |
| `check` | `model`, with the C reference machine in lockstep: every model token must equal the reference's | verification |
| `ref` | the C reference machine (input words pushed to it) | the same pipeline without the engine (fast) |
| `native` | none: the host interpreter's sim context runs `sim_main` and counts the tokens | development |

In `model`, `check` and `ref` every token goes through the decoder into the host's RAM, so the
host sees the sim's stores exactly as the model made them; `native` writes RAM directly. All
modes consume the same input words (the host's pokes), so the token and instruction counts
agree across modes (tested below).

## The hot loop (`native/runner`)

```c
qd_run *qd_run_new(qd_engine *e, qd_dec *dec, qd_chk *chk);   /* model (chk NULL) / check */
qd_run *qd_run_new_ref(qd_ref *ref, qd_dec *dec);             /* ref */
void qd_run_set_pred(qd_run *r, int32_t pred);   /* after a resync: what step(GO) returned */
int  qd_run_tick(qd_run *r, const uint32_t *words, int n, qd_tick_stats *st);
uint64_t qd_run_live_tokens(const qd_run *r);    /* any thread */
void qd_run_cancel(qd_run *r);                   /* any thread */
void qd_run_pause(qd_run *r, int on);            /* any thread */
const char *qd_run_error(const qd_run *r);
int  qd_run_set_qos(void);                       /* the calling thread: user-interactive */
```

Also `qd_run_set_forced`, `qd_run_set_max_tokens`, `qd_run_set_log` / `qd_run_log_len` (every
token fed and its host flag, into the caller's buffers), `qd_run_last`, `qd_run_token_name`
(`qd_run.h` documents each).

`build/libqdrun.dylib` links against `build/libqdengine.dylib` and `build/libqdtrace.dylib`
through their public headers (`@rpath` = `@loader_path`, so it finds them next to it).
`runner.lib()` loads the engine and the trace library first and dyld binds libqdrun's
`@rpath` dependencies to those images by install name, so there is one copy of each, the one the
Python objects come from (checked from the image list at load: with the default builds, and with
an engine copy loaded through `QD_ENGINE_LIB`, which the runner then uses). The runner borrows the `qd_engine`, `qd_dec`, `qd_chk` and `qd_ref` of the
Python objects (`Engine.h`, `Decoder.p`, `Checker.p`, `Ref.p`).

`qd_run_tick` has `trace.drive_tick`'s semantics:

* at every record boundary where the decoder wants input (`qd_dec_wants_input`), the next input
  word's value record (`qd_value_record`, 1 to 3 tokens) is fed instead of the model's
  predictions: the tokens before the record's last one as `qd_engine_step_forced` (no head; their
  predictions are never used), the last one as `qd_engine_step` (its prediction is the next
  token);
* everywhere else the pending prediction is the token;
* every token goes to the decoder (`qd_dec_feed`: stores land in the host's RAM), then to the
  checker in check mode (`qd_chk_feed`), then to the engine;
* the tick ends after `YIELD` (stepped: its prediction is the next tick's first token, kept in the
  runner), `HALT` or a fault record.

In ref mode the reference's records (`qd_ref_record`) are the tokens; the loop checks that the
reference's input records come exactly where the decoder wants input.

Results: `QD_TICK_YIELD`, `HALT`, `FAULT`, `CANCELLED`, or an error: `E_DECODE` (the decoder's
text, with the tick's token index), `E_INPUT` (input wanted with no word left), `E_DIVERGED`
(the checker's divergence: stream index, record, got / want, the record's PC, word, anchor, n
and the reference's whole record), `E_REF`, `E_LIMIT` (no end within 2^26 tokens), `E_STATE`.
`qd_tick_stats`: tokens, injected, forced, instructions and records (the decoder's), input words
read and unread, seconds (pauses excluded), paused, the ending event.

**Threads.** `qd_run_tick` runs on the caller's thread with the GIL released (ctypes); the live
token counter is an atomic the loop stores after every token, so another thread reads it while
the tick runs (the UI's TOKENS and TOKENS-PER-SECOND counters). `qd_run_cancel` sets a flag the
loop reads before every token: the tick returns `CANCELLED` between two tokens, and so does every
later tick (the sim is mid-tick; the game is over). `qd_run_pause(1)` makes the loop wait on a
condition variable before its next token until `qd_run_pause(0)` or a cancel. The runner's
callers put the game thread at user-interactive QoS (`qd_run_set_qos`, so macOS keeps it on a
performance core); the engine's workers already are.

**Forced steps are exact.** A forced step runs every layer up to the last one with a token mixer
and skips that layer's MLP and the layers after it, which only feed the head
(`native/engine/step.c`): the recurrent state, the conv windows and the position advance exactly
as in a full step. `test_model_ticks_check_mode_and_forced_steps` runs the first 5 ticks of E1M1
(74,738 tokens, 609 of them injected, 198 of those forced) with forced steps and again with every
token fully stepped: the same tokens, the same RAM, and afterwards every recurrent state row of
all 64 value heads of the 15 layers with a token mixer (0-14; the circuit's layer 15 has no
attention heads and layer 16 is the zero full-attention layer) bit-identical.

## The runner (`qwendoom/runner.py`)

```python
from qwendoom.runner import Runner, parse_buttons, E1M1_WALK
r = Runner("model")                   # mode, weights=, threads=, elf=, wad=, args=, config_dir=
idx, pal = r.step_frame()             # run the host to its next frame (and any sim tick)
r.action("forward", True)             # keys between frames: action / key(doom_key) / char / hold
r.run_frames(400, parse_buttons(E1M1_WALK), on_frame=lambda tic, idx, pal: ...)
r.tokens, r.ticks, r.frames, r.tps, r.counters()
```

* `step()` runs the host to its next event and answers it: `"tick"` (`last_tick`: a `Tick` with
  tokens, injected, forced, instructions, input words, seconds, resync), `"frame"` or `"exit"`;
  `step_frame()` runs to the next frame and sets `ticked` if a sim tick ran on the way.
* **Resync.** When the host asks (`need_resync`: a level start, a death and reborn, loading a
  save), the next tick first loads the host's RAM, DROM, ROM and the sim's entry context
  (`pc = sim_main`, `sp`, `gp`, other registers 0) into the model: `snapshot.state_rows` (with
  `weights/qwendoom.json`'s head map), `engine.start` (reset, PAD, the rows), the decoder's and
  the checker's contexts (the checker's reference gets a fresh copy of RAM), GO to the decoder
  and the checker (the model's first token is then stream token 1, where a GO is an error), then
  `step(GO)` gives the pending prediction. About 40 ms (53,473 state rows at E1M1's start).
* Every model-mode tick runs in C (above). `live_tokens` (any thread), `cancel()`, `pause(on)`.
* Errors: `SimHalt` (the sim called `I_Error`: the message is formatted from sim RAM,
  `qd_error_fmt` / `qd_error_args`, as the host interpreter formats it), `SimFault`, `Cancelled`,
  `trace.CheckError` (with the `Divergence`), `trace.DecodeError`; native mode raises
  `HostError` with the interpreter's message, and so does any mode for an I_Error in the host's
  DOOM (a bad WAD, say). A failed tick is not counted in `ticks` (the clean ticks; its error says
  `tick N` with N = `ticks`), but its tokens, injected tokens, instructions and time are.
* `threads`: the engine's threads, `DEFAULT_THREADS` = 1 (or `QD_THREADS`), the fastest on real
  ticks (below). `log=N` keeps the first N tokens and their host flags (`token_log()`);
  `engine=` reuses an open engine; `forced=False` steps every injected token fully.

**Button scripts** (`parse_buttons`): entries `action[+action]@a-b[+c-d]` joined by commas, over
tic numbers (a-b inclusive, or a single tic a; an open range such as `40-` is an error). A tic is a frame of DOOM's clock (the platform layer's clock advances one tic per
frame); a level start shows ~38 frames of screen wipe without a sim tick, and a menu stops the sim
ticks while frames go on. Actions are the UI's (`left right forward back strafeleft straferight
fire use run menu enter weapon1 weapon2 weapon3`) plus aliases (`turnleft turnright backward open
shoot esc`). `E1M1_WALK` walks Freedoom E1M1 from the start along the hall, turns right to the
door at (960, -152), opens it and walks into the room beyond, firing on the way: 400 tics, 362
sim ticks.

## The model game source

`ModelSource` (`qwendoom/sources.py`, registered as `model`, the server's default source) runs a
`Runner` in a game thread per Start, like `HostSource` (a generation counter; state checked and
published under the lock the control methods take). Options: `mode` (`model`, `check`, `ref`,
`native`), `threads`, `fps`, and the Runner's (`weights`, `elf`, `wad`, `args`, `config_dir`).

* **Pace.** Frames go out as the model produces them. Frames without a sim tick (menus, the
  screen wipe at a level start) come at host speed and are paced to 35 fps (DOOM's tic rate), as
  is everything in the fast modes.
* **Stats** come from their own thread at ~10 Hz, also while a tick runs: `tokens` = the model's
  tokens so far (the live counter), `tps` = tokens per second over the last ~2 s (`null` when no
  token came in that window: menus, paused), `frames` = frames shown.
* **Start** shows the counters moving at once: the state is `starting` (status "loading the level
  into the model, then its first tic") through the resync and the two ticks before DOOM's first
  frame (25,488 tokens: 1.6-2.3 s), then `running` with the first frame, then the wipe at 35 fps.
* **Pause** holds the model between two tokens (`qd_run_pause`): the token counter stops at once,
  mid-tick; a frame finished meanwhile waits for Resume. **Stop** cancels the running tick
  (`qd_run_cancel`) and goes back to a dark idle screen; a later Start begins a fresh game.
* A divergence in check mode, an `I_Error` or a fault ends the game in the `error` state with the
  message.

## The command line

```sh
python -m qwendoom serve [--port 8080] [--source model|native|mock] [--threads N] [--mode M] [--open]
python -m qwendoom run [--tics N] [--mode model|check|native|ref] [--buttons SCRIPT]
                       [--gif PATH] [--png DIR] [--every K] [--scale S] [--threads N] [--quiet]
python -m qwendoom check [--tics N] [--buttons SCRIPT] [--verbose]
python -m qwendoom bench [--tics N] [--threads N] [--buttons SCRIPT] [--no-engine]
python -m qwendoom build [--weights] [--native] [--doom]
python -m qwendoom info [WEIGHTS] [--json]
```

* `serve`: the web UI (`qwendoom/server.py`); `--threads` and `--mode` go to the model source,
  `--open` opens the page in the default browser.
* `run`: headless play for `--tics` frames (default 105) with a button script; prints a line per
  sim tick (tic, tick, tokens, injected, instructions, seconds, tok/s) and a summary; saves the
  frames as an animated GIF (default `qwendoom.gif`; each frame keeps its own palette) and / or
  PNGs (`--png` alone: no GIF), every K-th frame, scaled S times.
* `check`: check mode; reports the tokens compared (all but the host-injected ones), or the first
  divergence with its context (`MISMATCH after N clean ticks`, exit status 1). The sim's own
  I_Error or fault, which the reference makes too, is reported as not a model mismatch (exit
  status 1). A missing or bad `--wad` ends run, check and bench with a one-line message.
* `bench`: model mode on real tics (default `E1M1_WALK`'s first 105), then the engine alone on
  the same tokens from the same snapshot (`qd_engine_check`, teacher-forced in C: the same steps
  without the decoder, the checker or the loop), and the ratio.
* `build`: `make -C doom`, the four native libraries, the checkpoint emitter
  (`qwendoom.weights.write_checkpoint`); all three without a flag.
* `info`: the checkpoint from `weights/config.json` and `qwendoom.json`: shapes, nonzero weights,
  the tables per layer, the address map, the I/O conventions.

## Verification

Apple M5 Max, Freedoom E1M1 at skill 3, `weights/` as built. Other jobs were running (the engine
track's benchmarks among them; load average 4-8), so the rates vary by a few %. (a)-(e) ran on the
engine before its DOOM-tick tuning (docs/engine.md), at 13-18k tok/s; the tuned engine runs the same
ticks at about 31k tok/s on one thread ((d) and (f)).

**(a) Check mode, 362 ticks** (`python -m qwendoom check --tics 400 --buttons <E1M1_WALK>
--verbose`, one engine thread, log `build/runner-check-400.log`): the player walks the hall, turns
twice, opens the door at (960, -152) and walks into the room beyond while monsters wake and
shoot, firing 5 times on the way. 9,966,344 tokens, **9,946,385 model tokens compared with the
reference machine, all equal** (19,959 host-injected not compared), 7,495,731 instructions
(1.33 tokens each). Per tick: tokens min 12,345, median 23,129, p95 57,891, max 132,706; 0.87 to
9.6 s per tick, median 1.71 s; 11.8k-14.8k tok/s per tick, 13,547 overall. 736 s.
`test_check_mode_e1m1_walk` repeated it with 4 engine threads (the default then): the same 9,966,344 tokens, all
equal, 16,743 tok/s (596 s).

**(b) Model mode against native mode, 302 ticks** (`test_model_equals_native`, 340 tics of
`E1M1_WALK`, one engine thread): the two runners step event by event; after every tick all
256 KB of the model runner's RAM (written only by the host and by the decoder from the model's
tokens) equal the native runner's, and so do each tick's token, instruction, injected-token and
input-word counts; all 340 frames (indices and palette) are identical. 7,786,443 tokens,
13,348 tok/s (again with 4 threads: 17,630 tok/s, 442 s).

**(c) A resync mid-game** (`test_model_resync_after_load`): F2 saves after a walk, the player walks
on, F3 loads the save at tic 198: the level reloads, the host asks for a resync, and the model
gets a new snapshot before tick 128. Model against native as in (b) through all 300 tics: 190
ticks (62 after the load, with a walk, a turn and shots), 3,733,197 tokens, RAM equal after every
tick, every frame equal; 2 resyncs of 0.05 s each (one thread 13,182 tok/s, four 17,270).

**(d) Speed** (`python -m qwendoom bench --tics 80`: 42 ticks of `E1M1_WALK`'s start, 645,003
tokens, then `qd_engine_check` over the same tokens from the same snapshot; two rounds each,
`build/runner-bench.log`):

| Engine threads | hot loop (model mode) | engine alone, same tokens | ratio |
|---|---|---|---|
| 1 | 13,758 / 13,746 tok/s | 13,313 / 13,668 tok/s | 1.033 / 1.006 |
| 4 | 17,601 / 17,820 tok/s | 17,334 / 17,578 tok/s | 1.015 / 1.014 |

These rows were measured on the engine before its DOOM-tick tuning (docs/engine.md). On the
tuned engine the same command, hot loop only, gives 1 thread 31,897 tok/s, 2 threads 30,770,
4 threads 30,674: threads no longer help, hence `DEFAULT_THREADS` = 1.

The hot loop runs at the engine's own speed (the decoder and the checker cost ~20-40 ns per
token against the engine's ~31-73 us; the differences are run-to-run noise). Before the tuning,
on DOOM's ticks the engine was slower than on the C test programs of
docs/engine.md: E1M1's snapshot holds 53,473 state rows (every nonzero RAM word), and a tick is
mostly loads and stores. The Python per-token loop (`trace.drive_tick` with `Engine.step`, one
thread) on the same first 12 ticks: 12,277 tok/s, against 12,158 for the C loop in the same
minute (other jobs running), within 1% of each other: the per-token Python work was never what
limited the rate; the engine is (~80 us a token here). The C loop's gains are the live counter,
pause and cancel from other threads, and no Python per token.

**(e) The server with the model source** (`python -m qwendoom serve --port 8765` in the
background, one engine thread at the time, a Python WebSocket client): GET `/` 200; Start: `starting` stats at once, the token
counter moving every 100 ms through the resync and the first two ticks (542, 1,811, 3,102, ...
24,886 tokens; tps ~12k), the first frame 2.3 s after Start, then `running`; 44 frames in the next
6 s (the wipe at 35 fps, then the model's pace); `forward` held for 5 s (4 frames: ~30k tokens a
tick while walking); Pause: `paused` with tps null, then 3 s without a frame or a stats message
and the counter held; Resume: frames and tokens again; Stop: `idle`, tokens 0, no frame in the
next 2 s; a second Start: a fresh game (frame after 2.3 s); the server stopped with SIGTERM.
Again with the defaults of the time (4 engine threads, port 8766): the same sequence, every check of
the client passing, the first frame 1.8 s after Start (1.6 s for the second game).

**(f) Check mode, 2,026 ticks, on the tuned engine** (one engine thread, log
`build/check-2100.log`):

```bash
python -m qwendoom check --tics 2100 --verbose --buttons 'forward@40-128+141-225+254-400+430-700+740-1000+1040-1400+1460-1800+1840-2100,right@80-92+226-253+700-716+1400-1430+1800-1812,left@128-140+1000-1020+1600-1610,use@296-298+1100-1102+1500-1502+1900-1902,fire@200-203+215-218+330-333+345-348+380-383+500-503+560-563+620-623+800-803+900-903+1200-1203+1300-1303+1700-1703+2000-2003,strafeleft@1050-1080,straferight@1650-1680'
```

The walk of (a), then 1,700 more tics of walking, turning, strafing, firing and pressing use. The
player dies between tics 600 and 900, and the use press at tic 1,100 restarts the level, so the
model gets a second resync mid-run (die and respawn, which (c) did not cover). 48,026,481 tokens,
**47,939,803 model tokens compared with the reference machine, all equal** (86,678 host-injected
not compared), 36,393,214 instructions, 2 resyncs of 0.05 s. Per tick: tokens min 12,331, median
20,822, p95 44,031, max 132,706; median 0.68 s per tick, at most 4.4 s; 30,523 tok/s overall,
1,574 s.

## Tests

`.venv/bin/python -m pytest -q tests/test_runner.py tests/test_cli.py`: 14 tests, about 45 s
with the weights (the model tests skip without them, the DOOM tests without the host library,
doom.elf or Freedoom). `QD_SLOW=1` adds 3 (21.5 min in all on the last run: 17 passed):

* `test_runner.py`: button scripts; I_Error's message from sim RAM; ref mode against native mode
  in lockstep (RAM after every tick, token / instruction / injected counts, frames, the token
  log); check mode for 5 ticks and the forced-step proof above; the live counter, pause and
  cancel from another thread during a model tick; the model source over the WebSocket in ref
  and in model mode (start, frames, stats with tokens rising, keys, pause holding the model,
  resume, stop). `QD_SLOW=1` adds the long runs: check mode over `E1M1_WALK`, model against
  native for 340 tics, the save / load resync.
* `test_cli.py`: help, info, bad arguments, `run` in native mode (the GIF's frames equal a
  Runner's) and in ref mode with PNGs, `check` and `bench` for 42 tics, `build`'s commands.
