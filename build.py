#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# ///
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def build(config="release"):
    build_dir = ROOT / "build" / config
    subprocess.run(
        ["cmake", "-B", str(build_dir), f"-DCMAKE_BUILD_TYPE={config}"],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--config", config],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    build()
