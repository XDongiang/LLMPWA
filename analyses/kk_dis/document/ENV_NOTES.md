# kk_dis 运行环境依赖清单（本地 + 远程 HEP1）

> 整理时间：2026-09-11。用途：LLMPWA code-generator（`agent/`）+ 生成的 JAX 拟合代码。
> 已知坑：httpx/no_proxy 括号 IPv6 bug —— 运行时需修正 `no_proxy`/`NO_PROXY`（见下文）。

## 一、两组用途

| 用途 | 环境 | 需要哪些包 |
|---|---|---|
| A. 代码生成流水线（跑 `python -m agent.cli`） | 本地 venv + 远程 base 均可 | `openai` `python-dotenv` `toml` |
| B. 拟合代码运行（`run/fit_script.py`） | 远程 **kk_fit**（GPU，2026-09-11 重建） | `numpy` `scipy` `jax[cuda12]` `jaxlib`（+`toml`） |

## 二、本地（开发/生成，CPU 冒烟）

### 现状
- 已建 venv：`analyses/kk_dis/.venv`（Python 3.13）
- 已装：`openai 2.54.0` `python-dotenv 1.2.3` `toml 0.10.2` `numpy 2.5.3` `scipy 1.18.1` `jax 0.11.1 (cpu)` `jaxlib 0.11.1`
- 注意：base conda 的 site-packages 是**只读**，不能 pip install；必须用 venv。

### 若重建 venv
```bash
cd analyses/kk_dis
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install openai==2.54.0 python-dotenv toml scipy "jax[cpu]"
```

## 三、远程 HEP1（拟合运行，GPU 2×RTX3090 / CUDA 12.2）

### base（跑生成器，可选但推荐）
```bash
~/miniconda3/bin/pip install openai==2.54.0 python-dotenv toml
```
> `openai` 必须钉 2.x：3.x 依赖 `httpx2`，与本环境的 no_proxy bug 组合后无法构造 client。
> 远程 base 的 `numpy/scipy/jax` 在生成器阶段不需要；拟合运行在 deep-torch。

### kk_fit（跑 fit_script，2026-09-11 重建）
> 旧 deep-torch 因 conda CPU jax + dnnl 符号冲突损坏，已删除。
```bash
~/miniconda3/bin/conda create -y -n kk_fit -c conda-forge -c nvidia \
  python=3.11 "jax[cuda12]" "jaxlib=*=cuda12*" numpy scipy
~/miniconda3/envs/kk_fit/bin/pip install toml
```
验证：`~/miniconda3/envs/kk_fit/bin/python -c "import jax; print(jax.version); print(jax.devices())"`
CUDA 与驱动 535.146.02 / CUDA 12.2 匹配；当前 jax 0.10.2 识别 2×RTX 3090。

### 磁盘提醒
HEP1 `/home` 只剩 ~50G（93%），jax cuda12 wheel 较大（数百 MB），装前确认空间。

## 四、运行代码生成器（本地）

```bash
cd LLMPWA
export no_proxy='localhost,127.0.0.1,.localdomain,::1'
export NO_PROXY='localhost,127.0.0.1,.localdomain,::1'
./analyses/kk_dis/run_pipeline.sh            # 等于 engine 跑全流程
./analyses/kk_dis/run_pipeline.sh --check-only
./analyses/kk_dis/run_pipeline.sh --stage classification
```

> 已封装 `analyses/kk_dis/run_pipeline.sh`（内部修正 no_proxy + 用 .venv + cd 到仓库根读 .env）。

## 五、远程同步与运行
```bash
# 推送分析目录到 HEP1 ~/LLMPWA/kk_dis
# （用 remote_push / remote_exec 按文件同步；data/ 较大建议 rsync 或按需推送）
# 远程跑拟合：
cd ~/LLMPWA/kk_dis
~/miniconda3/envs/kk_fit/bin/python run/fit_script.py   # 单机双 GPU：mesh 直接 shard，无需 docker/多进程
```

## 六、关键坑备忘
1. **`NO_PROXY` 含 `[::1]` + httpx** → `Invalid port: ':1]'`。
   解法：跑前 `export no_proxy=...,::1` 和 `NO_PROXY=...,::1`（去掉 `[::1]` 的方括号）。
   本地/远程跑任何 openai/httpx 相关都建议带。
2. **openai 3.x + httpx2** 不稳定 → 钉 `openai==2.54.0`。
3. **本地 base site-packages 只读** → 一律用 `analyses/kk_dis/.venv`。
4. `.env` 在仓库根（`LLMPWA/.env`），引擎 `load_dotenv()` 从 cwd 找，所以必须在仓库根执行。
