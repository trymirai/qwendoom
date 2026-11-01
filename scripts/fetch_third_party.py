# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Fetch the third-party sources and data qwendoom builds on into third_party/ (gitignored).

    uv run scripts/fetch_third_party.py            # everything
    uv run scripts/fetch_third_party.py doomgeneric freedoom

Nothing here is vendored into the repository: every source is a pinned git commit or a
checksummed release archive.
"""

import hashlib
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "third_party"

REPOS = {
    # DOOM, via doomgeneric (GPL-2.0): the game the RISC-V machine runs
    "doomgeneric": ("https://github.com/ozkl/doomgeneric",
                    "e257a09527062bfb986b6e5e0780839b6cbd5b71"),
    # Qwenputer (MIT): the q35 compiler and the RV32IM core qwendoom's circuit derives from
    "qwenputer": ("https://github.com/divagr18/qwenputer",
                  "bec1a0c9654ac8151b6f36ed0154a19c53141384"),
    # nesformer (MIT): the NES build on the same compiler (engine and host design reference)
    "nesformer": ("https://github.com/dbreunig/nesformer",
                  "26b1b053f826154498926892815574580dc45c24"),
    # riscv-tests (BSD-3-Clause): the ISA compliance suites (isa/), relinked for our address map
    "riscv-tests": ("https://github.com/riscv-software-src/riscv-tests",
                    "3df703cd52d5f0d8224e3c50bf03c6cb70efff98"),
}

# Freedoom (BSD-3-Clause): free game data; freedoom1.wad stands in for DOOM's IWAD
FREEDOOM = ("https://github.com/freedoom/freedoom/releases/download/v0.13.0/freedoom-0.13.0.zip",
            "3f9b264f3e3ce503b4fb7f6bdcb1f419d93c7b546f4df3e874dd878db9688f59")


def git_at(name: str, url: str, rev: str) -> None:
    d = DEST / name
    if (d / ".git").exists() and subprocess.run(
            ["git", "-C", d, "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip() == rev:
        print(f"{name}: up to date ({rev[:10]})")
        return
    d.mkdir(parents=True, exist_ok=True)
    if not (d / ".git").exists():
        subprocess.run(["git", "init", "-q", d], check=True)
        subprocess.run(["git", "-C", d, "remote", "add", "origin", url], check=True)
    subprocess.run(["git", "-C", d, "fetch", "-q", "--depth", "1", "origin", rev], check=True)
    subprocess.run(["git", "-C", d, "checkout", "-q", "--force", rev], check=True)
    print(f"{name}: {rev[:10]}")


def freedoom() -> None:
    d = DEST / "freedoom"
    wad = d / "freedoom1.wad"
    if wad.exists():
        print("freedoom: present")
        return
    d.mkdir(parents=True, exist_ok=True)
    url, sha = FREEDOOM
    zpath = d / Path(url).name
    with urllib.request.urlopen(url) as r:
        zpath.write_bytes(r.read())
    got = hashlib.sha256(zpath.read_bytes()).hexdigest()
    if got != sha:
        zpath.unlink()
        sys.exit(f"freedoom: checksum mismatch ({got})")
    with zipfile.ZipFile(zpath) as z:
        for member in ("freedoom1.wad", "COPYING.txt"):
            (d / member).write_bytes(z.read(f"freedoom-0.13.0/{member}"))
    zpath.unlink()
    print("freedoom: freedoom1.wad")


def main(argv: list[str]) -> None:
    names = argv or [*REPOS, "freedoom"]
    for name in names:
        if name == "freedoom":
            freedoom()
        elif name in REPOS:
            git_at(name, *REPOS[name])
        else:
            sys.exit(f"unknown source {name!r}; known: {', '.join([*REPOS, 'freedoom'])}")


if __name__ == "__main__":
    main(sys.argv[1:])
