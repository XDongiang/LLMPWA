# kk_pipi 操作流程总览（End-to-End）

> 整理时间：2026-09-14（同步最新状态：单机双 GPU 直跑，不需要 Docker）。
> 覆盖：环境准备 → 代码生成 → 远程同步 → 远程拟合 → 监控与结果。
> 相关文档：`document/ENV_NOTES.md`（依赖与坑）、`document/README_architecture.md`（架构与流水线设计）、
> 工作目录 `README.md`（agent 操作规范）。

---

## 0. 拓扑速览

```text
本地开发机 (iso)                         远程 HEP1 (hyx@127.0.0.1:20402，经跳板 root@www.insolitude.me:22)
┌────────────────────────────┐          ┌──────────────────────────────────────────────┐
│ LLMPWA 仓库                  │          │ ~/LLMPWA/kk_pipi      （与本地工作区同步）     │
│  analyses/kk_pipi/           │          │ ~/miniconda3               base（生成器环境）  │
│    .venv       生成器环境     │  ssh     │ ~/miniconda3/envs/kk_fit   拟合环境(GPU)     │
│    data/       数据           │ ───────► │ 2×RTX 3090 / CUDA 12.2 / 125GiB RAM        │
│    run/        生成代码       │          │                                              │
└────────────────────────────┘          └──────────────────────────────────────────────┘
```

两条命脉：

- **本地 `.venv`**：跑 LLM 代码生成流水线（`python -m agent.cli`）
- **远程 `kk_fit`（conda env）**：跑 JAX GPU 拟合（`run/fit_script.py`）

---

## 1. 环境准备

### 1.1 本地生成器环境（`analyses/kk_pipi/.venv`）

本地 base conda site-packages **只读**，必须用 venv：

```bash
cd /home/iso/openclaw/deepseek-harness/LLMPWA/analyses/kk_pipi
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install openai==2.54.0 python-dotenv toml scipy "jax[cpu]"
```

要点：
- `openai` **必须钉 2.54.0**（3.x 依赖 httpx2，与本环境 no_proxy bug 叠加会炸）
- 本地 jax 用 CPU 版即可（只做冒烟测试）
- `.env`（LLM API key）在仓库根 `LLMPWA/.env`，引擎从 cwd 找

### 1.2 远程拟合环境（HEP1 上 `kk_fit` conda env）

```bash
# HEP1 上执行
~/miniconda3/bin/conda create -y -n kk_fit -c conda-forge -c nvidia \
  python=3.11 "jax[cuda12]" "jaxlib=*=cuda12*" numpy scipy
~/miniconda3/envs/kk_fit/bin/pip install toml        # 生成器/结果 TOML 读写用
```

验证：

```bash
~/miniconda3/envs/kk_fit/bin/python -c "import jax; print(jax.__version__, jax.devices())"
# 期望：jax 0.10.2 [CudaDevice(id=0), CudaDevice(id=1)]
```

---

## 2. 运行代码生成流水线（本地）

### 2.1 入口

```bash
# 静态检查（不真正跑 LLM）
./analyses/kk_pipi/run_pipeline.sh --check-only

# 完整流水线（全自动，无需人工审阅）
./analyses/kk_pipi/run_pipeline.sh

# 只重跑某一 stage（例如改了似然提示词后）
./analyses/kk_pipi/run_pipeline.sh --stage generate_args
```

`run_pipeline.sh` 内部做了三件事（不要手动绕过）：

1. 修正 `no_proxy`/`NO_PROXY`（去掉 `[::1]` 方括号）→ 规避 httpx `Invalid port: ':1]'` bug
2. `cd` 到仓库根（让引擎找到 `.env`）
3. 用 `.venv/bin/python -m agent.cli --workdir analyses/kk_pipi --config llm_config_combine.toml` 执行

### 2.2 联合拟合 stage

`llm_config_combine.toml` 定义的 stage（对照 `combine_check` → `config_strip_kk/pipi` →
`generate_args` → `split_args` → `classification_kk/pipi` → `likelihood_*` → `assemble_final_code`）
见 `document/README_architecture.md` 或直接读 `llm_config_combine.toml`。

当前模式：**全非交互**（`require_approval = false`，tools 无 `ask_user`）。

---

## 3. 同步到远程 HEP1

### 3.1 目标布局

```text
/home/hyx/LLMPWA/kk_pipi/          ← 与本地 analyses/kk_pipi 对齐（代码+数据+config）
/home/hyx/LLMPWA/.env              ← API key（生成器可选）
```

### 3.2 同步方式

- 用 `remote_push` / `remote_exec` 按文件同步；`data/` 较大建议 rsync 或按需推送。
- 远程结构保持与本地同构（`config/`、`data/`、`run/`、`output/`、`task/`）。

---

## 4. 远程拟合

### 4.1 单机双 GPU（当前模式，无需 Docker）

```bash
cd ~/LLMPWA/kk_pipi
~/miniconda3/envs/kk_fit/bin/python run/fit_script.py
```

### 4.2 多节点 Docker

```bash
./docker/run_fit_multinode.sh --workdir analyses/kk_pipi
```

节点与容器配置见 `node_config.toml`。

---

## 5. 监控与结果

- 拟合日志：`logs/fit.log`（`config/logconfig_fit.json` 指定）。
- 结果：`output/fit/`、`output/multistart/`。
- 报告归档到 `task/<日期>_<内容>/3_报告/`；`status.json` 同步任务状态。

---

## 6. 关键坑

1. **httpx/no_proxy 括号 IPv6 bug**：跑前 `export no_proxy=...,::1` 和 `NO_PROXY=...,::1`。
2. **openai 必须钉 2.x**：3.x 依赖 httpx2 不稳。
3. **本地 base site-packages 只读**：一律用 `.venv`。
4. **`.env` 在仓库根**：必须从仓库根执行 `python -m agent.cli`。
