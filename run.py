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
    b.build()

    # run the benchmark
    if sys.platform == "win32":
        exe = ROOT / "build" / "bin" / "Release" / "bench-linear-access.exe"
    else:
        exe = ROOT / "build" / "bin" / "bench-linear-access"
    subprocess.run([str(exe)], check=True)


if __name__ == "__main__":
    main()
