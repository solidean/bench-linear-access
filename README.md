# Benchmark: How Much Linear Memory Access Is Needed?

A small benchmark exploring how memory access patterns affect throughput. It processes a fixed amount of data split into blocks of varying sizes, measuring MB/s for each block size. When blocks are small, the CPU can't prefetch ahead effectively and cache line utilization suffers; as block size grows toward contiguous access, throughput should increase significantly.

The benchmark generates random float data, optionally shuffles block order to defeat prefetching, then repeatedly computes simple statistics (sum, sum of squares, min, max) over each block layout. The median timing across runs is reported.

TODO: describe setup in more detail once all variants are implemented

TODO: describe measurement policy (median of 9)

## Requirements

- [CMake](https://cmake.org/) >= 3.25
- A C++23 compiler: GCC 14+, Clang 17+, or MSVC 19.38+ (VS 2022 17.8+)
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) *(optional, for the shortcut scripts)*

## Quickstart

```bash
# Build and run with uv (recommended)
uv run run.py

# Build only
uv run build.py
```

## Manual steps

```bash
# Configure
cmake -B build

# Build
cmake --build build --config Release

# Run
./build/bin/bench-linear-access       # Linux / macOS
build\bin\bench-linear-access.exe     # Windows
```

## TODO: Results

<!-- Add benchmark results here once collected on target hardware. -->
