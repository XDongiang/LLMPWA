# kk_dis 工作目录操作规范（Agent 约束）

> 本文件是 **agent（含 LLMPWA 代码生成器与人工驱动的拟合 agent）如何操作 `analyses/kk_dis` 工作目录的约束**。
> 违反以下规范的操作都应避免；有疑问先读本文件与 `document/` 下的手册。
> （项目架构说明见 `document/README_architecture.md`，本文件专注操作约束；
> 跨节点分布式拟合与互连测试见 §5。）

---

## 1. 目录结构总览

```text
analyses/kk_dis/
├── README.md                 ← 本规范（约束 agent 操作 + 仓库索引）
├── llm_config_fit.toml       ← 代码生成流水线配置（8 个 stage）
├── resonances_config.toml    ← 共振态/振幅物理配置（**单一来源，见 §4**）
├── node_config.toml          ← 多节点 Docker 节点表（HEP1 header + HEP3 worker，见 §5）
├── node_config_3gpu.toml     ← 2 节点 3 GPU（HEP1×2 + HEP3×1，每卡 1 容器，见 §5.3a）
├── node_config_2gpu_self.toml ← 单机 2 GPU 自检配置（见 §5.3a）
├── Dockerfile.jax            ← jax-fit 容器镜像定义（JAX+CUDA+SciPy，见 §5.2）
├── run_fit_multinode.sh      ← 多节点运行脚本（原版在 LLMPWA/docker/，见 §5）
├── run_multinode_starts.sh   ← 多起点随机初值稳定性测试驱动（见 §5.4）
├── run_pipeline.sh           ← 生成器入口脚本（no_proxy 修正 + venv + agent.cli）
├── config/                   ← 运行配置（logconfig_fit.json）
├── data/                     ← 物理数据（real_data/mc_truth/draw/weight），**只读，勿改**
├── document/                 ← 参考文档与操作手册（见 §2）
├── tmp/                      ← 临时文件（同步分片、临时日志等；**可随时清理**）
├── task/                     ← 拟合任务记录（见 §3；跨节点互连测试已归档为 ⭐）
├── gen/                      ← 生成器中间产物（prompts/handlers/templates/fragments/llm_logs）
├── run/                      ← 生成/手写的可执行代码（见 §4、§5）
├── output/                   ← 拟合输出（fit/ 与 multistart/ 报告）
│
│   # 以下互连测试相关文件已归档，见
│   # task/20260916_跨节点互连测试_50起点稳定性/：
│   #   1_运行代码/ time_fit_iter.py bench_nccl_socket.py hep3_coord_fit.sh
│   #                run_dist_fit.sh run_dist_time.sh fit_script_dist.py
│   #                node_config.toml(.bak)
│   #   3_报告/    INTERCONNECT_IS_NOT_BOTTLENECK.md（结论）
│   #                timing_results.md（计时）、ms_ms50/（50 起点原始数据+统计）
└── .dsh/  .venv/  logs/      ← 工具链与运行时（勿手改）
```

---

## 2. document/ —— 参考文档与操作手册

存放**长期有效**的文档，按主题分类，命名清晰：

| 文件 | 内容 |
|---|---|
| `document/OPERATIONS.md` | **操作手册**：端到端流程（环境→生成→同步→拟合→监控） |
| `document/ENV_NOTES.md` | 环境依赖清单与坑（本地 venv / 远程 kk_fit） |
| `document/combined_likelihood_math.md` | 似然函数数学说明（**生成似然的唯一公式来源**） |
| `document/random_initial_perturbation.md` | 随机初值扰动算法（多起点拟合用） |
| `document/README_architecture.md` | 项目架构说明（原 README 存档） |

**规则**：
- 新增/修改文档放这里，不散落根目录。
- 公式类（似然、扰动算法）改动会直接影响生成器的 `[ref]` 引用，改完要同步 `llm_config_fit.toml` 的 ref 路径。

---

## 3. task/ —— 拟合任务记录

每个拟合任务一个文件夹，命名 `<日期>_<工作内容>`（如 `20260912_run100随机初值拟合_bench双卡对比`）。
内部分 5 个子目录：

> 归档实例 · `20260912_run100随机初值拟合_bench双卡对比`：
> run100 的收敛分析数据 `convergence.json` / `all_params.npy` / `all_errors.json` /
> `x0_all.npy` 已从根目录移入其 `3_报告/`，分析脚本在 `1_运行代码/convergence_analysis.py`
> （详见 `3_报告/convergence_path_analysis.md` 头注释）。

```text
task/<日期>_<工作内容>/
├── status.json        ← 任务状态机（机器可读，前端快速解析，见下）
├── 1_运行代码/    ← 本次运行用的完整代码快照（fit_script.py / likelihood_function.py / 多起点 runner 等）
├── 2_工作记录/    ← status.md：做什么、结果摘要、遇到的问题与修复、关键命令、产物位置
├── 3_报告/        ← 完成后输出的报告 md（结果分析、结论、对比表）
├── 4_图片/        ← 相关图片（若有：plot、nvidia-smi 截图等）
└── 5_生成器日志/  ← 代码生成器的详细日志（llm_logs/）、提示词（prompts/）、配置（llm_config_fit.toml、resonances_config.toml）
```

**规则**：
- **远程是同构结构**：`~/LLMPWA/kk_dis/task/<日期>_<内容>/`，拟合输出（summary/work 产物）**留在远程**，
  需要展示/归档的按需 `remote_pull` 到本地。
- 每次新拟合前：先查 `task/` 看是否已有类似任务（复用代码/避免重跑）。
- 每次任务**创建时**：写 `status.json`（`status = "running"`）；**结束时**更新为 `"completed"`（或 `"failed"`），
  填入结果指标与产物路径。`2_工作记录/status.md` 是人类可读的详细版，两者**同步维护**。
- 大产物（work 目录、summary）不复制，记录路径即可。
- `tmp/` 是唯一可随便清理的地方；task/ 与 document/ 要长期保留。

### 3.1 status.json 字段约定（供前端解析）

```json
{
  "task_id": "<日期>_<工作内容>",       // 必填，唯一标识
  "title": "简短标题",
  "status": "running | completed | failed | paused",  // 必填，任务状态机
  "date_start": "ISO8601",              // 开始时间（可空）
  "date_end": "ISO8601",                // 结束时间（未完成时为 null）
  "environment": { "host": "", "gpu": "", "python_env": "" },
  "task_type": "fit_multistart | fit_single | bench | ...",
  "summary": {                          // 结果指标（completed 后填写）
    "n_starts": 0, "n_success": 0, "success_rate": 0.0,
    "best_nll": 0.0, "best_start": 0, "near_global_frac": 0.0,
    "wall_clock_h": 0.0
  },
  "artifacts": {
    "code_dir": "1_运行代码/",
    "records_dir": "2_工作记录/",
    "reports_dir": "3_报告/",
    "images_dir": "4_图片/",
    "generator_logs_dir": "5_生成器日志/",
    "remote_work_dir": "~/LLMPWA/kk_dis/task/<id>/...",
    "summary_files": ["...", "..."]
  },
  "reports": ["3_报告/xxx.md"],
  "issues_fixed": ["...", "..."],
  "notes": "可选，引用相关文档"
}
```

- `status` 取值约束：`running` / `completed` / `failed` / `paused`（前端按此渲染颜色/进度）。
- `summary` 内的字段键名固定，前端可按 `best_nll` / `success_rate` 等直接取数。
- 未知/附加字段允许扩展，但**不要改变已有键的语义**（前端依赖）。
- 每次更新 status.json 时，HTML 特殊字符（`<>&"`）需保持 JSON 转义合法（UTF-8）。

---

## 4. 代码生成流程约束（关键）

### 4.1 能改 vs 不能改

**可以修改**（运行时适配，直接改代码即可，不用重跑生成器）：
- `run/fit_script.py`、`run/fit_script_dist.py`（分布式版）的 **main 函数**：分布式初始化、
  Mesh、数据加载/归一化、shard、JIT nll/grad/hvp、SciPy minimize、Hessian/误差、
  结果保存——为适配拟合流程可自由改（如改优化器、加约束、改收敛判据、改输出格式）。
- `run` 下的工具/runner：如 `multi_start_fit.py`（多起点调度）、`multi_start_single.py`
  （单节点多起点）、`bench_gpu_modes.py`（A/B）。

**不可直接修改**（由共振态配置/似然公式决定，必须改提示词→重跑生成器）：
- **共振态分组**：`resonances_config.toml` 里的共振列表、传播子类型（BW/Flatté）、
  耦合结构——改它 = 改物理模型 → 要重跑 `config_strip` → `classification` → **重新生成 likelihood**。
- **似然函数**：`run/likelihood_function.py` 的提取/组合/约束逻辑——它由
  `gen/prompts/generate_likelihood_*` 提示词 + `document/combined_likelihood_math.md` 公式生成。
  **若手改，下次重跑生成器会被覆盖丢改动**。

**规则一句话**：改共振态分组或似然公式 → **必须改提示词，重跑代码生成器**；
只适配拟合流程（主函数）→ 直接改 `run/fit_script.py`。

### 4.2 修改提示词的流程

1. 改公式：更新 `document/combined_likelihood_math.md`（或提示词里的公式描述）。
2. 改共振态：更新 `resonances_config.toml`（共振/传播子/耦合）。
3. 改提示词：编辑 `gen/prompts/generate_likelihood_{system,task}.txt`、
   `generate_fit_script_*` 等。
4. 重跑生成器：
   ```bash
   ./run_pipeline.sh                        # 全流程
   ./run_pipeline.sh --stage generate_likelihood   # 只重跑似然
   ```
5. 生成后检查 `run/likelihood_function.py`、`run/likelihood_test_result.txt`（冒烟 PASS）。
6. 同步变更文件到远程 `~/LLMPWA/kk_dis/`。

### 4.3 共振态配置只读语义

`resonances_config.toml` 是物理模型的**单一来源**。生成器 stage `config_strip` 会把它
拆成 `run/free_params.toml`（自由参数+range）与 `gen/fragments/*`。**不要绕过生成器手改
`run/free_params.toml` 的物理内容**（拟合初值例外——`multi_start_fit.py` 用临时副本做扰动）。

---

## 5. 多节点分布式拟合与跨卡互连测试

**背景**：传统超算常单机仅 2-4 卡、机器间走以太网。要回答"跨卡互连是不是这类
physics-fit（JAX 似然、~65 参数、事件并行）的瓶颈，1 Gbps 以太网（无 IB/RoCE）能否
支撑 100-200 参数 / 100-200 GB 的拟合"——本目录用 HEP1+HEP3 双机、1 Gbps socket
互连做了完整验证。**结论：互连不是瓶颈**（详见
`task/20260916_跨节点互连测试_50起点稳定性/3_报告/INTERCONNECT_IS_NOT_BOTTLENECK.md`）。

### 5.1 拓扑与架构

- 节点：**HEP1 (192.168.200.125) = header = JAX process 0**；**HEP3 (192.168.200.171) = worker = process 1**。
  （早期版本以 HEP3 作协调器，已废弃；`hep3_coord_fit.sh` 仅存档。）
- 互连：1 Gbps 以太网 socket → NCCL 用 `NCCL_SOCKET_IFNAME`（非 `NCCL_IB_HCA`）；
  实测 socket allreduce 带宽 ≈ 0.117 GB/s（≈ 线速），每迭代通信载荷仅 ~520 B → ~4 µs，
  相对每迭代 24-40 ms 计算，占比 ~0.02%。
- 架构：**chief-only SciPy** —— 只在 process 0 跑 `scipy.optimize.minimize(Newton-CG)`，
  worker 只做下采样数据的并行 objective/grad/HVP，结果经 `multihost_utils.broadcast_one_to_all`
  同步。这保证两进程 collectives 完全对称（规避 CG 路径分叉死锁）。
- 数据：`data_size=10000` 事件在 event 轴真实分片（`shard_data_distributed` + `NamedSharding`），
  每次 objective/grad/HVP 都触发跨设备 collective —— 真实通信路径。
- 每节点只挂 1 卡：env 里 `CUDA_VISIBLE_DEVICES=0`（不设则 2 卡都被 JAX 认到，
  mesh 3 设备 sharding 10000 报 `IndivisibleError`）。

### 5.2 容器镜像（`Dockerfile.jax`）

- 镜像名 `jax-fit:latest`，两机都有。JAX 环境 python 是容器内 **`/opt/env/bin/python`**
  （HEP1 宿主 `python3` 无 numpy，宿主也没有 `/opt/env/bin/python`）。
- 构建：`docker build -f Dockerfile.jax -t jax-fit:latest .`（基于
  `nvidia/cuda:12.2.0-devel-ubuntu22.04`，devel 镜像自带 CUDA 工具链，构建不需宿主 GPU；
  **运行**才需要宿主有 NVIDIA Container Toolkit + `--gpus all`）。
- 改完 Dockerfile 需在两机重新 build 或 `sync_image.sh`（见 `LLMPWA/docker/`）。

### 5.3 运行脚本（原版在 `LLMPWA/docker/`，本地副本在根目录）

| 脚本 | 作用 | 备注 |
|---|---|---|
| `run_fit_multinode.sh` | 单次多节点拟合：rsync workdir → 两机 docker → wait → 拉回 output | 需 `--workdir`（+ 可选 `--config`） |
| `run_multinode_starts.sh` | 50 起点随机初值稳定性测试驱动 | 循环调上面脚本，输出 `output/ms_<tag>/summary.toml` |
| `time_fit_iter.py` | 每算子计时（obj/grad/hvp） | 单机/分布式配置均可测 |
| `bench_nccl_socket.py` | socket allreduce 带宽实测 | 用于通信预算 |
| `run/fit_script_dist.py` | 分布式拟合主脚本（chief-only SciPy） | 容器内入口：`/opt/env/bin/python` |

**对原版 `run_fit_multinode.sh` 的关键适配**（注意别回退）：
1. `NCCL_IB_HCA=${ib_if}` → `NCCL_SOCKET_IFNAME=${ib_if}`（以太网）；
2. 容器入口 `${IMAGE} python3 ...` → `${IMAGE} /opt/env/bin/python ...`；
3. 三处 rsync 都要 **`--no-group`**（容器以 root 写文件，rsync `-a` 会尝试 chgrp
   → 用户权限不足 → rc=23 → `set -euo pipefail` 中断整个驱动）。

**node_config.toml**：header = HEP1(.125, port 12345, enp4s0)，worker = HEP3(.171, enp5s0)；
container.env = `NCCL_DEBUG=VERSION, NCCL_IGNORE_CPU_AFFINITY=1,
JAX_COORDINATOR_TIMEOUT=1000, JAX_ENABLE_X64=1, CUDA_VISIBLE_DEVICES=0`。
（旧 IB 风格备份在 `node_config.toml.bak`。）

### 5.3a 每卡 1 容器多 GPU（非对称卡数）

`run_fit_multinode.sh` 支持每个节点多卡：header / worker 段加 `Chips = [0, 1]`，
每 chip = 1 个容器 = `-e CUDA_VISIBLE_DEVICES=<chip>`，进程号按 header chips →
worker chips 顺序分配（容器名 `kkfit*_p<pid>`，同机不冲突）。无 Chips 字段 = 单卡
（向后兼容）。

**2 节点 3 GPU 实测**（`node_config_3gpu.toml`：HEP1 `Chips=[0,1]` + HEP3 `Chips=[0]`，
端口 12347，数据 `n_repeat=3` = 30000 事件，30000/3=10000/卡，整除）：

| 配置 | success | NLL | 迭代 | 墙钟 |
|---|---|---|---|---|
| 单机 2 GPU（HEP1 2 容器） | True | -12127.3404819 | 147 | 214.5 s |
| **2 节点 3 GPU**（HEP1×2 + HEP3×1，1 Gbps） | True | -12127.3401347 | 147 | **155.4 s** |

- 两者收敛到同一 NLL（差 ~3.5e-4 = allreduce 浮点噪声），迭代数相同。
- **3 卡（含跨机以太网）比 2 卡单机快 28%** —— 跨机 1 Gbps 通信零拖累，规模收益真实。
- 数据量须整除设备数（见 §4/第 4 坑，30000/3 满足）；`n_repeat = 3` 在
  `run/base_functions.py::load_data()`（运行期适配，可改）。
- **不要**用 JAX 原生"1 节点 1 进程多 GPU"跑非对称卡数：`multihost_utils` 要求各进程
  本地设备数相同（jax 0.10.2/0.11.1 实测 `ValueError: cannot reshape ...`，
  见归档 task 工作记录）。

### 5.4 一次 50 起点跨节点测试（示例命令，HEP1 上执行）

```bash
cd ~/LLMPWA/kk_dis
setsid nohup bash run_multinode_starts.sh --starts 50 --seed 2024 --tag ms50 \
    > /tmp/mn50.out 2>&1 < /dev/null &
```

每起点：生成扰动初值 `ms_x0/start_N.npy`（驱动脚本运行时生成在工作目录，见
`1_运行代码/run_multinode_starts.sh`；const/theta 重采样到半径 0.1 圆 +
全参数 ±1% 乘性抖动，`RandomState(seed+s*1000)`）→ 生成 `node_config_ms_N.toml`
（注入 `KKDIS_X0_NPY` / `KKDIS_OUTDIR=ms_<tag>/start_N` / `KKDIS_MAXITER=400`）→
两机后台 5 s 采样 GPU util → 调 `run_fit_multinode.sh` → 解析结果追加到
`output/ms_<tag>/summary.toml`。容器内 python 一律用相对路径（cwd=/workspace/work）。

**常见坑**：
- docker run 带 heredoc stdin 需 **`-i`**，否则 python 读不到输入。
- 两机如有残留 `kkfit` 容器，新 run 会因"different incarnation"起不来 → 先 `docker rm -f kkfit`。
- `KKDIS_MAXITER=400` 给收敛慢的起点设上限（否则单起点可跑 1000+ 迭代 / 15+ 分钟）。
- HEP1 自 ssh 免密与 HEP1→HEP3 ssh/rsync 免密都要先配好。

### 5.5 结果与交付物

**结论报告**：`task/20260916_跨节点互连测试_50起点稳定性/3_报告/INTERCONNECT_IS_NOT_BOTTLENECK.md`
（§2 每算子计时、§3 端到端匹配拟合、§4 通信预算、§5 50 起点稳定性、§6 结论）。

**50 起点数据**（seed 2024，tag ms50，2026-09-16 跑完，约 3.5 h）：

| 指标 | 值 |
|---|---|
| 起点数 | 50 |
| returncode=0 | 50/50（无崩溃/死锁/超时） |
| 收敛 | 24/50 = 48%（其余 26 个撞 400 迭代上限 = 局部极小） |
| 全局最优（\|NLL+3950.002\|<0.01） | 6/50 = 12% |
| NLL | mean -3756.1 ± 106.1, min -3950.003, max -3582.73 |
| 每起点耗时 | mean 251.6 s（收敛点 160.3 s），min 84 s / max 621 s |
| GPU 利用 | HEP1(chief) 82.1% / HEP3(worker) 83.9% |

- 原始明细：`task/20260916_跨节点互连测试_50起点稳定性/3_报告/ms_ms50/summary.toml`（50 行 results）；
- 统计脚本：同目录 `ms_ms50/analyze.py`（本地 `python3 analyze.py` 重跑）。
- 计时基准：`3_报告/timing_results.md`（单节点 vs 跨节点每算子一致；匹配拟合收敛到同一 NLL，
  跨节点 104.3 s / 188 it vs 单节点 116.6 s / 205 it）。

**关键洞察**（写入结论报告）：
- 每 GPU 的计算量决定耗时，通信（~520 B/迭代 = 4 µs）可忽略；
- 扩展到 100-200 参数时 allreduce 载荷仅升至 ~1.6 KB，1 Gbps 仍绰绰有余；
  真正的约束是单卡算力与显存能否装下分片后数据；
- 50 起点 48% 收敛 / 12% 到全局最优 = 模型景观多峰性（单节点 2 起点对照行为一致），
  与互连无关。

---

## 6. 运行代码的边界

- `run/base_functions.py`：模板拼接产物（工具/物理/数据/初值/存盘），由 `assemble_help_functions`
  stage 生成。少手改，除非是通用 bug 修复（`setup_logging` mkdir 这类防御）。
- `run/likelihood_function.py`：见 §4.1，**生成器产物，勿手改物理逻辑**。
- `run/fit_script.py`：主函数可改（§4.1）。若改动会被后续重跑生成器覆盖（`generate_fit_script`
  stage 会重写它），把适配性的改动挪到**独立模块**（如 `run/fit_adapt.py`、`run/fit_script_dist.py`）
  或记入 `task/<日期>/2_工作记录/status.md`，重跑后重新应用。

---

## 7. 数据与产物约束

- `data/` **只读**：真实数据/MC/权重，任何拟合不得改写。
- `output/` 可写：`output/fit/`（单次结果）、`output/multistart/`（多起点 summary+报告）、
  `output/ms_<tag>/`（跨节点多起点 summary，驱动脚本直接写，勿手改）。
- `task/` 内各任务的 `3_报告/` 是对外交付物（如互连测试的原始数据 + 统计脚本在
  `20260916_跨节点互连测试_50起点稳定性/3_报告/ms_ms50/`），保留勿清。
- 报告（md）写完后：副本归档到 `task/<日期>_<内容>/3_报告/`。
- 远程产物留在远程，本地只保留报告/小文件。

---

## 8. 环境

- 本地生成器：`analyses/kk_dis/.venv`（openai==2.54.0 / jax cpu）。跑生成器用 `run_pipeline.sh`。
- 远程拟合（单机）：`~/miniconda3/envs/kk_fit`（jax 0.10.2 cuda12）。跑拟合用
  `~/miniconda3/envs/kk_fit/bin/python run/fit_script.py`。
- 远程拟合（多机）：容器 `jax-fit:latest`，容器内 `/opt/env/bin/python`（见 §5）。
- 详细信息：`document/ENV_NOTES.md`、`document/OPERATIONS.md`。

---

## 9. Agent 操作速查（TL;DR）

1. **先查 document/ 手册**，再动手。
2. **查 task/ 历史**，避免重复劳动。
3. 拟合跑批 → 用 `run/multi_start_fit.py`（`--resume` 断点续跑）。
4. 改拟合主函数 → 直接改 `run/fit_script.py`（分布式版 `run/fit_script_dist.py`）。
5. 改共振态分组/似然 → **改提示词 + 重跑生成器**，不手改生成产物。
6. 多机拟合 → `run_fit_multinode.sh`（单次）/ `run_multinode_starts.sh`（多起点），
   注意 §5.3 的三个关键适配与坑。
7. 新任务 → 建 `task/<日期>_<内容>/`，写 `status.json`（status=running）+ `2_工作记录/status.md`。
8. 结果 → 报告归档 `3_报告/`；更新 `status.json`（status=completed + 结果指标）；
   远程产物留远程。跨节点结论见
   `task/20260916_跨节点互连测试_50起点稳定性/3_报告/INTERCONNECT_IS_NOT_BOTTLENECK.md`。
9. 临时文件 → `tmp/`（可清理）；task 归档目录保留勿动。
