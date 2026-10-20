#!/usr/bin/env python3
"""画新旧版 × 单卡/4 卡的显存占用随时间的曲线。

数据：results/perf_probe/vram_*.csv（由 run/vram_sample.sh 每 5 s 采一次）
      + results/perf_probe/vram_phases.json（阶段边界）
采样 t=0 与 sbatch 脚本起始对齐，所以能和日志里算出的阶段边界直接叠图。

重新生成（本地需要 matplotlib）：
    python3 -m venv .venv-plot && .venv-plot/bin/pip install matplotlib
    MPLCONFIGDIR=/tmp/mplcfg .venv-plot/bin/python run/plot_vram_timeline.py
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DDIR = os.path.join(ROOT, "results", "perf_probe")
OUT = os.path.join(DDIR, "vram_timeline.png")

plt.rcParams["font.family"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

GPU_COLORS = ["#1f77b4", "#e8801a", "#2ca02c", "#8e44ad"]
C_COMPILE = "#c0392b"
C_OPT = "#2980b9"
YMAX = 7400


def load_series(path):
    """-> {gpu_index: [(t, used_mib), ...]}，只保留真正被占用的卡。"""
    raw = {}
    with open(path) as fh:
        for row in csv.reader(fh):
            if len(row) < 3:
                continue
            t, idx, used = float(row[0]), int(row[1]), float(row[2])
            raw.setdefault(idx, []).append((t, used))
    out = {}
    for g, pts in raw.items():
        pts.sort()
        if max(m for _, m in pts) > 0:      # 空闲卡不画
            out[g] = pts
    return out


with open(os.path.join(DDIR, "vram_phases.json")) as fh:
    meta = json.load(fh)

order = ["旧版 1 卡", "新版 1 卡", "旧版 4 卡", "新版 4 卡"]
cfgs = sorted(meta["configs"], key=lambda c: order.index(c["label"]))

fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.1), sharey=True)
axes = axes.ravel()

for ax, c in zip(axes, cfgs):
    series = load_series(os.path.join(DDIR, c["csv"]))
    gpus = sorted(series)

    (c0, c1), (o0, o1) = c["compile"], c["optimize"]
    ax.axvspan(c0, c1, color=C_COMPILE, alpha=0.09, lw=0, zorder=0)
    ax.axvspan(o0, o1, color=C_OPT, alpha=0.09, lw=0, zorder=0)
    ax.axvline(c1, color=C_COMPILE, ls="--", lw=1.0, alpha=0.5, zorder=1)

    for g in gpus:
        xs = [p[0] for p in series[g]]
        ys = [p[1] for p in series[g]]
        # 5 s 采样：两次采样之间只知道"保持"，故用阶梯线
        ax.step(xs, ys, where="post", color=GPU_COLORS[g % 4], lw=1.9, zorder=3)

    peak = max(max(m for _, m in series[g]) for g in gpus)
    ax.axhline(peak, color="#444", ls=":", lw=1.3, alpha=0.8, zorder=2)
    ax.annotate(
        "峰值 %d MiB" % peak,
        xy=(c["elapsed"] * 0.99, peak), xytext=(0, -6),
        textcoords="offset points", ha="right", va="top",
        fontsize=10.5, fontweight="bold", color="#1a1a1a",
    )

    # 阶段名放在图内顶部，避开峰值标注
    ax.text((c0 + c1) / 2, YMAX * 0.99, "编译", ha="center", va="top",
            fontsize=10.5, color=C_COMPILE, fontweight="bold")
    ax.text((o0 + o1) / 2, YMAX * 0.99, "优化", ha="center", va="top",
            fontsize=10.5, color=C_OPT, fontweight="bold")

    if len(gpus) > 1:
        spread = max(max(m for _, m in series[g]) for g in gpus) - min(
            max(m for _, m in series[g]) for g in gpus)
        note = "4 卡全部在用，峰值相差仅 %d MiB" % spread
    else:
        note = "只占用 GPU 0，其余 3 卡空闲"
    # 放在坐标区上方，避免与阶梯线重叠
    ax.text(0.5, 1.015, note, transform=ax.transAxes,
            fontsize=9.5, color="#555", ha="center", va="bottom")

    ax.set_title("%s  ·  %d 事例  ·  总 %d s"
                 % (c["label"], c["n_events"], c["elapsed"]),
                 fontsize=12.5, pad=24)
    ax.set_xlim(0, c["elapsed"])
    ax.set_ylim(0, YMAX)
    ax.grid(alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    ax.set_ylabel("显存占用 (MiB)", fontsize=10.5)

for ax in axes[2:]:
    ax.set_xlabel("作业开始后时间 (s)", fontsize=10.5)

handles = [Patch(facecolor=C_COMPILE, alpha=0.30, label="编译阶段"),
           Patch(facecolor=C_OPT, alpha=0.30, label="minimize 优化阶段"),
           Line2D([], [], color="#444", ls=":", lw=1.3, label="全程峰值")]
handles += [Line2D([], [], color=GPU_COLORS[g], lw=2.0, label="GPU %d" % g)
            for g in range(4)]
fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=10,
           frameon=False, bbox_to_anchor=(0.5, 0.004))

fig.suptitle(
    "显存占用随时间变化：旧版单卡最吃显存（6.7 GB），新版 4 卡每卡仅 2.8 GB\n"
    "V100-SXM2-16GB（总 16384 MiB）· 每 5 s 采样一次",
    fontsize=13, fontweight="bold", y=0.985,
)
fig.tight_layout(rect=[0, 0.042, 1, 0.925])
fig.savefig(OUT, dpi=170)
print("wrote", OUT)

print("\n%-10s %8s %8s %8s %8s" % ("配置", "峰值", "编译末", "优化末", "总时长"))
for c in cfgs:
    print("%-10s %7d %8d %8d %8d" % (
        c["label"], c["peak"], c["compile"][1], c["optimize"][1], c["elapsed"]))
