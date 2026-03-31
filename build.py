#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# ///
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def build():
    subprocess.run(
        ["cmake", "-B", "build"],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        ["cmake", "--build", "build", "--config", "Release"],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    build()
