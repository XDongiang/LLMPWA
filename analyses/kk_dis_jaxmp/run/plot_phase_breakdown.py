#!/usr/bin/env python3
"""画新旧版 × 单卡/4 卡的四段耗时堆叠柱状图。

数据来自 logs/phase_breakdown.json（由 run/phase_breakdown.py 生成），
四段相加严格等于 sacct 的 Elapsed。

重新生成：

    # 1) 在集群上抽四段耗时（读作业日志，需要 ssh 到超算）
    python3 run/phase_breakdown.py

    # 2) 本地画图（需要 matplotlib）
    python3 -m venv .venv-plot && .venv-plot/bin/pip install matplotlib
    MPLCONFIGDIR=/tmp/mplcfg .venv-plot/bin/python run/plot_phase_breakdown.py

输出：results/perf_probe/phase_breakdown.png
中文字体用 Noto Sans CJK SC；若系统没有，回退 DejaVu Sans（中文会变方框）。
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "results", "perf_probe", "phase_breakdown.json")
OUT = os.path.join(ROOT, "results", "perf_probe", "phase_breakdown.png")

plt.rcParams["font.family"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SEGMENTS = [
    ("overhead", "启动 / 读数据 / 收尾", "#b8bfc9"),
    ("compile", "编译 (XLA)", "#c0392b"),
    ("optimize", "minimize 优化", "#2980b9"),
    ("hessian", "Hessian + 存盘", "#27ae60"),
]

with open(DATA) as fh:
    rows = json.load(fh)

# 把固定的启动/收尾开销合并成一个"非拟合"段
for r in rows:
    r["overhead"] = r["startup"] + r["tail"]

# 顺序：旧版 1卡, 新版 1卡, 旧版 4卡, 新版 4卡
order = ["旧版 1卡", "新版 1卡", "旧版 4卡", "新版 4卡"]
rows = sorted(rows, key=lambda r: order.index(r["label"]))
labels = [r["label"] for r in rows]
x = range(len(rows))

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))

# ---- 左图：绝对耗时（堆叠） ----
ax = axes[0]
bottom = [0.0] * len(rows)
for key, name, color in SEGMENTS:
    vals = [r[key] for r in rows]
    ax.bar(x, vals, 0.62, bottom=bottom, color=color, label=name,
           edgecolor="white", linewidth=0.8)
    bottom = [b + v for b, v in zip(bottom, vals)]

for i, r in enumerate(rows):
    ax.text(i, r["elapsed"] + 12, "%.0f s" % r["elapsed"], ha="center",
            va="bottom", fontsize=11, fontweight="bold")
    # 段内标注：够宽写在色块里，太窄就引到右侧写
    acc = 0.0
    for key, name, _ in SEGMENTS:
        v = r[key]
        if v >= 30:
            ax.text(i, acc + v / 2, "%.0f" % v, ha="center", va="center",
                    fontsize=9.5, color="white", fontweight="bold")
        elif v > 0:
            ax.annotate(
                "%.0f" % v,
                xy=(i + 0.31, acc + v / 2),
                xytext=(i + 0.36, acc + v / 2),
                fontsize=8.5, color="#555a60", va="center", ha="left",
                arrowprops=dict(arrowstyle="-", color="#9aa0a6", lw=0.7,
                                shrinkA=0, shrinkB=0),
            )
        acc += v

ax.set_xticks(list(x))
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("端到端墙钟 (s)", fontsize=11)
ax.set_title("四段耗时拆账（绝对）", fontsize=12.5, pad=12)
ax.set_xlim(-0.55, len(rows) - 0.15)
ax.set_ylim(0, max(r["elapsed"] for r in rows) * 1.16)
ax.grid(axis="y", alpha=0.28, linestyle="--")
ax.set_axisbelow(True)
ax.legend(fontsize=9.5, loc="upper right", framealpha=0.95)

# ---- 右图：归一化到 100%（突出占比变化） ----
ax = axes[1]
bottom = [0.0] * len(rows)
for key, name, color in SEGMENTS:
    vals = [100.0 * r[key] / r["elapsed"] for r in rows]
    ax.bar(x, vals, 0.62, bottom=bottom, color=color, edgecolor="white",
           linewidth=0.8)
    for i, v in enumerate(vals):
        if v >= 6:
            ax.text(i, bottom[i] + v / 2, "%.0f%%" % v, ha="center",
                    va="center", fontsize=9.5, color="white", fontweight="bold")
    bottom = [b + v for b, v in zip(bottom, vals)]

ax.set_xticks(list(x))
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("占端到端比例 (%)", fontsize=11)
ax.set_title("各段占比：旧版几乎全在编译，新版全在优化", fontsize=12.5, pad=12)
ax.set_ylim(0, 100)
ax.grid(axis="y", alpha=0.28, linestyle="--")
ax.set_axisbelow(True)

fig.suptitle("K K̄ 离散度拟合：新旧版 × 单卡/4 卡 耗时拆账",
             fontsize=14, fontweight="bold", y=0.985)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT, dpi=170)
print("wrote", OUT)

# 控制台复核
print("\n%-9s %8s %8s %8s %7s %8s" %
      ("配置", "编译", "优化", "Hess", "非拟合", "总计"))
for r in rows:
    print("%-9s %8.1f %8.1f %8.1f %7.1f %8d" %
          (r["label"], r["compile"], r["optimize"],
           r["hessian"], r["overhead"], r["elapsed"]))
