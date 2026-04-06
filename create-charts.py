#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "matplotlib"]
# ///

import argparse
import math
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ── Dark mode ────────────────────────────────────────────────────────────────
BG = "#121622"
FG = "#e0e0e0"
SPINE = "#444444"
LEGEND_BG = "#1e2436"

plt.rcParams.update({
    "figure.facecolor":   BG,
    "axes.facecolor":     BG,
    "text.color":         FG,
    "axes.labelcolor":    FG,
    "xtick.color":        FG,
    "ytick.color":        FG,
    "axes.edgecolor":     SPINE,
    "legend.facecolor":   LEGEND_BG,
    "legend.edgecolor":   SPINE,
    "legend.labelcolor":  FG,
})

parser = argparse.ArgumentParser()
parser.add_argument("csv", nargs="?", default="result.csv")
args = parser.parse_args()
csv_path = Path(args.csv)
out_dir = csv_path.parent

# ── Data loading ────────────────────────────────────────────────────────────
df = pd.read_csv(csv_path)
# pandas parses "True"/"False" strings as booleans automatically
df["throughput_mbs"] = df["working_set_bytes"] / df["duration_secs"] / 1e6

# ── Color helpers ────────────────────────────────────────────────────────────
# Each kernel gets a (start_color, end_color) pair; log2(working_set) drives t
KERNEL_COLORS = {
    "simd_sum":     ("#a5d6a7", "#43a047"),  # medium-light green → vivid green
    "scalar_stats": ("#90caf9", "#1e88e5"),  # medium-light blue → vivid blue
    "heavy_sin":        ("#ffcc80", "#ef6c00"),  # medium-light orange → vivid orange
}
WS_MIN = 20  # log2(1 MB)
WS_MAX = 26  # log2(64 MB)


def hex_to_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255 for i in (0, 2, 4))


def lerp_color(start_hex: str, end_hex: str, t: float) -> tuple[float, float, float]:
    t = max(0.0, min(1.0, t))
    s = hex_to_rgb(start_hex)
    e = hex_to_rgb(end_hex)
    return tuple(s[i] + (e[i] - s[i]) * t for i in range(3))


def kernel_color(kernel: str, working_set_bytes: int) -> tuple[float, float, float]:
    start, end = KERNEL_COLORS[kernel]
    t = (math.log2(working_set_bytes) - WS_MIN) / (WS_MAX - WS_MIN)
    return lerp_color(start, end, t)


def fmt_bytes(n: int) -> str:
    if n >= 1 << 20:
        return f"{n >> 20} MB"
    if n >= 1 << 10:
        return f"{n >> 10} KB"
    return f"{n} B"


def setup_xaxis(ax):
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: fmt_bytes(int(v))))
    ax.set_xlabel("Block size")
    ax.grid(True, which="both", color="#30333e", linewidth=0.6)
    ax.set_axisbelow(True)


# ── Plot 1: scalar_stats, randomized ────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(1018/96, 1018/96*5/8), dpi=96)

data1 = df[(df["kernel"] == "scalar_stats") & (df["randomized"] == True)]
for ws in sorted(data1["working_set_bytes"].unique()):
    sub = data1[data1["working_set_bytes"] == ws].sort_values("block_size_bytes")
    color = kernel_color("scalar_stats", ws)
    ax1.plot(sub["block_size_bytes"], sub["throughput_mbs"],
             color=color, marker="o", markersize=3, label=fmt_bytes(ws))

setup_xaxis(ax1)
ax1.set_ylabel("Throughput (MB/s)")
ax1.set_ylim(bottom=0)
ax1.set_title("scalar_stats — randomized")
ax1.legend(title="Working set", fontsize=8, title_fontsize=8)
fig1.tight_layout()
fig1.savefig(out_dir / "chart1_scalar_randomized.svg", format="svg", facecolor=fig1.get_facecolor())
print("Saved chart1_scalar_randomized.svg")

# ── Plot 2: scalar_stats, repeated ──────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(1018/96, 1018/96*5/8), dpi=96)

data2 = df[(df["kernel"] == "scalar_stats") & (df["randomized"] == False)]
for ws in sorted(data2["working_set_bytes"].unique()):
    sub = data2[data2["working_set_bytes"] == ws].sort_values("block_size_bytes")
    color = kernel_color("scalar_stats", ws)
    ax2.plot(sub["block_size_bytes"], sub["throughput_mbs"],
             color=color, marker="o", markersize=3, label=fmt_bytes(ws))

setup_xaxis(ax2)
ax2.set_ylabel("Throughput (MB/s)")
ax2.set_ylim(bottom=0)
ax2.set_title("scalar_stats — repeated")
ax2.legend(title="Working set", fontsize=8, title_fontsize=8)
fig2.tight_layout()
fig2.savefig(out_dir / "chart2_scalar_repeated.svg", format="svg", facecolor=fig2.get_facecolor())
print("Saved chart2_scalar_repeated.svg")

# ── Plot 3: all kernels, randomized, working_set=64MB ───────────────────────
fig3, ax3 = plt.subplots(figsize=(1018/96, 1018/96*5/8), dpi=96)

data3 = df[(df["randomized"] == True) & (df["working_set_bytes"] == 67108864)]
for kernel in ["simd_sum", "scalar_stats", "heavy_sin"]:
    sub = data3[data3["kernel"] == kernel].sort_values("block_size_bytes")
    _, end_color = KERNEL_COLORS[kernel]
    color = hex_to_rgb(end_color)
    ax3.plot(sub["block_size_bytes"], sub["throughput_mbs"],
             color=color, marker="o", markersize=3, label=kernel)

setup_xaxis(ax3)
ax3.set_yscale("log")
ax3.set_ylabel("Throughput (MB/s)")
ax3.set_title("All kernels — randomized, 64 MB working set")
ax3.legend(title="Kernel", fontsize=8, title_fontsize=8)
fig3.tight_layout()
fig3.savefig(out_dir / "chart3_kernels_64mb.svg", format="svg", facecolor=fig3.get_facecolor())
print("Saved chart3_kernels_64mb.svg")

# ── Plot 4: normalized throughput (all data) ─────────────────────────────────
fig4, ax4 = plt.subplots(figsize=(1018/96, 1018/96*5/8), dpi=96)

groups = df.groupby(["kernel", "randomized", "working_set_bytes"])
for (kernel, randomized, ws), grp in groups:
    grp = grp.sort_values("block_size_bytes")
    max_tp = grp["throughput_mbs"].max()
    normalized = grp["throughput_mbs"] / max_tp
    color = kernel_color(kernel, ws)
    linestyle = "-" if randomized else ":"
    ax4.plot(grp["block_size_bytes"], normalized,
             color=color, linestyle=linestyle, linewidth=1.2,
             alpha=0.85)

setup_xaxis(ax4)
ax4.set_ylabel("Normalized throughput")
ax4.set_title("Normalized throughput — all kernels & working sets")
ax4.set_ylim(0, 1.05)

# Legend: kernel color patches + line style for randomized/repeated
legend_elements = [
    Patch(facecolor=hex_to_rgb(KERNEL_COLORS["heavy_sin"][1]),        label="heavy_sin"),
    Patch(facecolor=hex_to_rgb(KERNEL_COLORS["scalar_stats"][1]), label="scalar_stats"),
    Patch(facecolor=hex_to_rgb(KERNEL_COLORS["simd_sum"][1]),     label="simd_sum"),
    Line2D([0], [0], color=FG, linestyle="-",  label="randomized"),
    Line2D([0], [0], color=FG, linestyle=":",  label="repeated"),
]
ax4.legend(handles=legend_elements, fontsize=8)
fig4.tight_layout()
fig4.savefig(out_dir / "chart4_normalized.svg", format="svg", facecolor=fig4.get_facecolor())
print("Saved chart4_normalized.svg")
