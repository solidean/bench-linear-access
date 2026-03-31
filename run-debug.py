#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# ///
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def main():
    # build first
    import build as b
    b.build(config="debug")

    # run the benchmark
    config = "debug"
    if sys.platform == "win32":
        exe = ROOT / "build" / config / "bin" / config.title() / "bench-linear-access.exe"
    else:
        exe = ROOT / "build" / config / "bin" / "bench-linear-access"
    subprocess.run([str(exe)], check=True)


if __name__ == "__main__":
    main()
