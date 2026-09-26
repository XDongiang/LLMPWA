# kk_dis 画图（draw / plot）运行环境说明

> 本文档给 `generate_draw_weight` / `generate_draw_plot` 生成 agent 提供：**在哪台机器、
> 用哪个 python、跑哪条命令** 来试跑/验证绘图代码。同时给出本地与远程（HEP1）的
> 环境差异与调整建议。
> 配套：`document/draw_plot_constraints.md`（绘图约束规范）。

---

## 1. 两个环境，两种用途

| 环境 | 用途 | pyROOT | JAX / SciPy |
|---|---|---|---|
| **本地** `analyses/kk_dis/.venv`（Python 3.13） | 跑 LLM 生成流水线 + `draw_weight`（JAX 冒烟） | 通常无 | 有（CPU） |
| **本地** `analyses/kk_dis/.pyroot_envs/rootplot`（conda, Python 3.11） | **画图冒烟**（`draw_plot`） | ✅ 6.40.04 | 无（不需要） |
| **远程** HEP1 `~/miniconda3/envs/rootenv`（conda） | **远程最终出图**（真实 PNG） | ✅ 6.40.04 | numpy 2.4.6 |

- `draw_weight_script.py` 需要 jax/scipy → 用**本地 `.venv`**。
- `draw_plot_script.py` 只需要 numpy + pyROOT → 用**本地 `.pyroot_envs/rootplot`**
  或**远程 `rootenv`**。

---

## 2. 本地画图环境（`.pyroot_envs/rootplot`）

本地用 conda 在 `analyses/kk_dis/.pyroot_envs/rootplot` 装了独立 pyROOT 环境
（pyROOT 6.40.04 + numpy 2.5.3），不污染 `.venv`。用法：

```bash
cd analyses/kk_dis
# 指定 conda 前缀包目录（首次需要；之后可省略）
export CONDA_PKGS_DIRS="${PWD}/.pyroot_pkgs"
export CONDA_ENVS_DIRS="${PWD}/.pyroot_envs"
export XDG_CACHE_HOME="${HOME}/.cache"

# 验证
.pyroot_envs/rootplot/bin/python -c "import ROOT; print(ROOT.__version__)"

# 跑画图脚本（batch 模式，无 GUI）
.pyroot_envs/rootplot/bin/python run/draw_plot_script.py
```

> 目录由生成 agent 自动创建。**若 conda 写入受限/网络慢**，可跳过本地 ROOT 冒烟，
> 只做 `py_compile` 语法检查，把真实出图放到远程 `rootenv`（§3）。

### 如何重建（可选）
```bash
cd analyses/kk_dis
mkdir -p .pyroot_pkgs .pyroot_envs
export CONDA_PKGS_DIRS="${PWD}/.pyroot_pkgs"
export CONDA_ENVS_DIRS="${PWD}/.pyroot_envs"
export XDG_CACHE_HOME="${HOME}/.cache"
unset CONDA_PLUGINS
/home/iso/miniconda3/bin/conda --no-plugins create -y -p .pyroot_envs/rootplot \
  -c conda-forge --override-channels --solver classic root
```

---

## 3. 远程 HEP1 画图环境（`rootenv`）

远程已存在 `~/miniconda3/envs/rootenv`（pyROOT 6.40.04 + numpy 2.4.6）。
生成 agent 通过 `remote_exec` / ssh 在该 python 下运行画图脚本产出真实 PNG：

```bash
# 远程：~/LLMPWA/kk_dis
~/miniconda3/envs/rootenv/bin/python run/draw_plot_script.py
```

前提：本地 `run/draw_plot_script.py`、`run/draw_weight_script.py`、
`output/draw/weight.npz` + `output/draw/weight_truth.npz` 已同步到远程
`~/LLMPWA/kk_dis/`（`run_pipeline.sh` 或 rsync）。远程 `output/draw/` 已有
`weight.npz`（38MB）与 `weight_truth.npz`（4.8MB），可直接复用。

> 远程画图只需 numpy + pyROOT，不需要 jax/scipy/kernel，`rootenv` 足够。

---

## 4. 运行顺序与命令

```bash
# 1) 权重（本地 .venv，jax CPU）
cd analyses/kk_dis
mkdir -p logs                    # 必须存在：setup_logging() 写 logs/fit.log / logs/draw.log
.venv/bin/python run/draw_weight_script.py      # 产出 output/draw/weight.npz + weight_truth.npz

# 2) 画图（本地 .pyroot_envs/rootplot，或远程 rootenv）
.png/rootplot python run/draw_plot_script.py    # 产出 output/pictures/*.png
```

> 若 `logs/` 不存在，`run/draw_weight_script.py` 的 `setup_logging()` 会抛
> `FileNotFoundError: logs/fit.log`。生成 agent 脚本开头应 `os.makedirs("logs", exist_ok=True)`
> （或运行前先 `mkdir -p logs`）。

---

## 5. 代码生成 agent 运行时如何选择 python

`generate_draw_plot` agent 在试跑时按以下顺序探测可用的 pyROOT python：

1. `~/miniconda3/envs/rootenv/bin/python`（远程，若通过 remote_exec 访问）
2. 本地 `analyses/kk_dis/.pyroot_envs/rootplot/bin/python`
3. 本地 `analyses/kk_dis/.venv/bin/python`（通常无 ROOT）

都没有 ROOT 时才降级为 `python -m py_compile ...` 语法检查，并在 test_result 中注明。

---

## 6. 环境调整建议（给生成 agent）

- **路径硬编码**：脚本内不要硬编码 `/home/iso/...` 或 `/home/hyx/...` 绝对路径；
  用 `os.path.dirname(os.path.dirname(os.path.abspath(__file__)))` 定位 analysis 根，
  再 `os.chdir` 到根目录（与现有 `run/base_functions.py` 一致）。
- **数据路径**：统一相对路径 `data/real_data/*.npy`、`data/mc_truth/*.npy`、
  `output/draw/*.npz`、`output/pictures/*.png`。
- **batch 模式**：脚本开头 `ROOT.gROOT.SetBatch(True)`，避免无显示环境报错。
- **ROOT 版本差异**：本地/远程均为 6.40.04；若换版本，`kOrange+7` 等 color + offset
  写法保持兼容。
