"""The command line (`python -m qwendoom ...`, qwendoom/cli.py): help, info, run (GIF / PNG
output, every mode's summary), check (its verdicts), bench, a missing or bad WAD, build (the
commands it runs), serve's options."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from qwendoom import cli
from qwendoom import host as H
from qwendoom import runner as R

ROOT = Path(__file__).resolve().parent.parent
needs_host = pytest.mark.skipif(
    not (H.LIB.exists() and H.DEFAULT_ELF.exists() and H.DEFAULT_WAD.exists()),
    reason="needs build/libqdhost.dylib, build/doom.elf and freedoom1.wad")
needs_weights = pytest.mark.skipif(
    not (R.DEFAULT_WEIGHTS / "model.safetensors").exists(), reason="needs weights/")


def qwendoom(*args, check=True, cwd=None) -> subprocess.CompletedProcess:
    r = subprocess.run([sys.executable, "-m", "qwendoom", *map(str, args)], cwd=cwd or ROOT,
                       capture_output=True, text=True, timeout=600)
    if check:
        assert r.returncode == 0, r.stdout + r.stderr
    return r


def test_help():
    out = qwendoom("--help").stdout
    for cmd in ("serve", "run", "check", "bench", "build", "info"):
        assert cmd in out
    serve = qwendoom("serve", "--help").stdout
    assert "--threads" in serve and "--open" in serve and "model (DOOM, sim on the model" in serve
    run = qwendoom("run", "--help").stdout
    for opt in ("--tics", "--mode", "--buttons", "--gif", "--png", "--every", "--scale",
                "--threads"):
        assert opt in run


def test_info():
    out = qwendoom("info").stdout
    assert "Qwen3_5ForCausalLM" in out and "1,521,437" in out
    assert "16 Gated DeltaNet + 1 full attention" in out
    assert "RAM 0x0+256 KB, DROM 0x80000+512 KB, ROM 0x100000+256 KB" in out
    assert "ram x16" in out and "rom x16" in out
    assert '"heads"' in qwendoom("info", "--json").stdout


def test_bad_arguments():
    r = qwendoom("run", "--buttons", "jump@1-5", "--mode", "native", check=False)
    assert r.returncode != 0 and "unknown action 'jump'" in r.stderr
    r = qwendoom("run", "--every", "0", check=False)
    assert r.returncode != 0
    r = qwendoom("serve", "--source", "mock", "--threads", "2", check=False)
    assert r.returncode != 0 and "--threads is an option of the model source" in r.stderr
    r = qwendoom("run", "--buttons", "forward@40-", "--mode", "native", check=False)
    assert r.returncode != 0 and "bad tic range '40-'" in r.stderr


def test_missing_wad(tmp_path):
    """A --wad that does not exist ends run, check and bench with a line saying so (before the
    model loads)."""
    wad = tmp_path / "missing.wad"
    for cmd in ("run", "check", "bench"):
        r = qwendoom(cmd, "--wad", wad, "--tics", 5, check=False, cwd=tmp_path)
        assert r.returncode != 0 and r.stderr == f"--wad: {wad}: no such file\n", (cmd, r.stderr)


@needs_host
def test_bad_wad(tmp_path):
    """A WAD DOOM refuses (the host's I_Error: a HostError) ends a run with its summary and a
    one-line message, not a traceback."""
    wad = tmp_path / "bad.wad"
    wad.write_bytes(b"not a WAD" * 100)
    r = qwendoom("run", "--mode", "native", "--wad", wad, "--tics", 5, check=False, cwd=tmp_path)
    assert r.returncode != 0 and "0 sim ticks in 0 frames (native mode)" in r.stdout
    assert r.stderr.startswith("HostError: host I_Error: Wad file bad.wad doesn't have IWAD")
    assert r.stderr.count("\n") == 1 and not (tmp_path / "qwendoom.gif").exists()


@needs_weights
@needs_host
def test_bad_wad_check_bench(tmp_path):
    """The same for check and bench (they caught no HostError before)."""
    wad = tmp_path / "bad.wad"
    wad.write_bytes(b"not a WAD" * 100)
    for cmd, mode in (("check", "check"), ("bench", "model")):
        r = qwendoom(cmd, "--wad", wad, "--tics", 5, check=False, cwd=tmp_path)
        assert r.returncode != 0 and r.stderr.startswith("HostError: host I_Error: Wad file"), \
            (cmd, r.stderr)
        assert r.stderr.count("\n") == 1 and f"0 sim ticks in 0 frames ({mode} mode)" in r.stdout


@needs_host
def test_check_verdicts(monkeypatch, capsys):
    """check's verdict when a tick fails, with a ref-mode runner standing in for the model (no
    weights needed): a token error in tick 0 is a MISMATCH after 0 clean ticks; the sim's
    I_Error (a HALT the reference makes too) is not a mismatch."""

    def check(setup) -> str:
        def runner(a, mode, log=0):
            assert mode == "check"
            r = R.Runner("ref")
            setup(r)
            return r

        monkeypatch.setattr(cli, "_runner", runner)
        with pytest.raises(SystemExit) as ex:
            cli.main(["check", "--tics", "45"])
        assert "0 sim ticks in " in capsys.readouterr().out
        return str(ex.value.code)

    msg = check(lambda r: R.lib().qd_run_set_max_tokens(r._run, 1000))
    assert msg.startswith("MISMATCH after 0 clean ticks (") and "DecodeError: tick 0: limit:" in msg

    def halt(r):  # tick 0 starts in I_Error("check halt\n") (the reference's and the decoder's)
        resync = r.resync

        def to_i_error():
            dt = resync()
            h = r.host
            pc, sp, gp = h.sim_entry()
            h.ram[sp - 64:sp - 52] = np.frombuffer(b"check halt\n\0", np.uint8)
            regs = [0] * 32
            regs[2], regs[3], regs[10] = sp - 128, gp, sp - 64
            r.ref.mem.sync_ram()
            for m in (r.ref, r.dec):
                m.set_context(pc=h.symbol("I_Error"), regs=regs)
            return dt

        r.resync = to_i_error

    msg = check(halt)
    assert msg.startswith("SimHalt: tick 0: the sim exited: check halt; not a model mismatch: "
                          "all ") and msg.endswith("tokens equal to the reference (0 clean ticks "
                                                   "before it)")


@needs_host
def test_run_native_gif(tmp_path):
    """A native-mode run saves every 2nd frame as a GIF, scaled, with each frame's own palette:
    the GIF's last frame is the frame a Runner shows at that tic."""
    from PIL import Image

    gif = tmp_path / "walk.gif"
    out = qwendoom("run", "--mode", "native", "--tics", 80, "--buttons", "forward@40-79",
                   "--gif", gif, "--every", 2, "--scale", 1).stdout
    assert "tick    0:" in out and "sim ticks in 80 frames (native mode)" in out
    assert "saved 40 frames" in out
    im = Image.open(gif)
    assert im.n_frames == 40 and im.size == (320, 200)
    im.seek(39)
    last = np.asarray(im.convert("RGB"))
    with R.Runner("native") as r:
        frames = {}
        r.run_frames(79, R.parse_buttons("forward@40-79"),
                     on_frame=lambda tic, idx, pal: frames.__setitem__(tic, pal[idx]))
    assert np.array_equal(last, frames[78])


@needs_host
def test_run_ref_png(tmp_path):
    out = qwendoom("run", "--mode", "ref", "--tics", 45, "--png", tmp_path / "png", "--every", 5,
                   "--quiet", cwd=tmp_path).stdout
    assert "(ref mode)" in out and "tick    0" not in out
    pngs = sorted((tmp_path / "png").glob("frame*.png"))
    assert [p.name for p in pngs] == [f"frame{i:05d}.png" for i in range(0, 45, 5)]
    assert not (tmp_path / "qwendoom.gif").exists()  # --png alone: no GIF


@needs_weights
@needs_host
def test_check_and_bench():
    out = qwendoom("check", "--tics", 42).stdout
    assert "OK: " in out and "model tokens compared with the reference machine, all equal" in out
    out = qwendoom("bench", "--tics", 42, "--threads", 1).stdout
    assert "hot loop (model mode, 1 engine threads)" in out
    assert "engine alone (qd_engine_check, same tokens)" in out and "MISMATCH" not in out


def test_build_commands(monkeypatch):
    calls = []
    monkeypatch.setattr(cli.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    cli.main(["build", "--native", "--doom"])
    assert calls == [["make", "-C", "doom"]] + [["make", "-C", f"native/{d}"] for d in
                                                ("host", "trace", "engine", "runner")]
