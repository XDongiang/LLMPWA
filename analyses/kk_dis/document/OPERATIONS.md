# kk_dis 操作流程总览（End-to-End）

> 整理时间：2026-09-11（同步最新状态：单机双 GPU 直跑，不需要 Docker）。
> 覆盖：环境准备 → 代码生成 → 远程同步 → 远程拟合 → 监控与结果。
> 相关文档：`document/ENV_NOTES.md`（依赖与坑）、`document/README_architecture.md`（架构与流水线设计）、
> 工作目录 `README.md`（agent 操作规范）。

---

## 0. 拓扑速览

```text
本地开发机 (iso)                         远程 HEP1 (hyx@127.0.0.1:20402，经跳板 root@www.insolitude.me:22)
┌────────────────────────────┐          ┌──────────────────────────────────────────────┐
│ LLMPWA 仓库                  │          │ ~/LLMPWA/kk_dis        （与本地工作区同步）   │
│  analyses/kk_dis/            │          │ ~/miniconda3               base（生成器环境）  │
│    .venv       生成器环境     │  ssh     │ ~/miniconda3/envs/kk_fit   拟合环境(GPU)     │
│    data/       数据(811MB)    │ ───────► │ 2×RTX 3090 / CUDA 12.2 / 125GiB RAM        │
│    run/        生成代码       │          │                                              │
└────────────────────────────┘          └──────────────────────────────────────────────┘
```

两条命脉：

- **本地 `.venv`**：跑 LLM 代码生成流水线（`python -m agent.cli`）
- **远程 `kk_fit`（conda env）**：跑 JAX GPU 拟合（`run/fit_script.py`）

---

## 1. 环境准备

### 1.1 本地生成器环境（`analyses/kk_dis/.venv`）

本地 base conda site-packages **只读**，必须用 venv：

```bash
cd /home/iso/openclaw/deepseek-harness/LLMPWA/analyses/kk_dis
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install openai==2.54.0 python-dotenv toml scipy "jax[cpu]"
```

要点：
- `openai` **必须钉 2.54.0**（3.x 依赖 httpx2，与本环境 no_proxy bug 叠加会炸）
- 本地 jax 用 CPU 版即可（只做冒烟测试）
- `.env`（LLM API key）在仓库根 `LLMPWA/.env`，引擎从 cwd 找

### 1.2 远程拟合环境（HEP1 上 `kk_fit` conda env）

> 2026-09-11 重建：旧 `deep-torch` env 因 conda CPU jax + dnnl 符号冲突损坏，已删除。

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

> 若只需要远程跑生成器（可选），`~/miniconda3/bin/pip install openai==2.54.0 python-dotenv toml`。
> 磁盘提醒：jax cuda12 wheel 较大，装前确认 `/home` 空间。

---

## 2. 运行代码生成流水线（本地）

### 2.1 入口

```bash
# 静态检查（不真正跑 LLM）
./analyses/kk_dis/run_pipeline.sh --check-only

# 完整流水线（8 个 stage 全自动，无需人工审阅）
./analyses/kk_dis/run_pipeline.sh

# 只重跑某一 stage（例如改了 likeliood 提示词后）
./analyses/kk_dis/run_pipeline.sh --stage generate_likelihood
```

`run_pipeline.sh` 内部做了三件事（不要手动绕过）：

1. 修正 `no_proxy`/`NO_PROXY`（去掉 `[::1]` 方括号）→ 规避 httpx `Invalid port: ':1]'` bug
2. `cd` 到仓库根（让引擎找到 `.env`）
3. 用 `.venv/bin/python -m agent.cli --workdir analyses/kk_dis --config llm_config_fit.toml` 执行

### 2.2 流水线 8 个 stage

| # | stage | kind | 产物 |
|---|-------|------|------|
| 1 | `config_strip` | python | `run/free_params.toml`、`gen/fragments/stripped_config.toml`、`free_params_range.toml` |
| 2 | `inspect_data_shapes` | agent | `gen/fragments/*_func.py`（load/normalize/shard）、`data_shapes_report.txt` |
| 3 | `make_initial_args` | python | `gen/fragments/make_initial_args.py` |
| 4 | `save_results` | python | `gen/fragments/save_results.py` |
| 5 | `assemble_help_functions` | python | **`run/base_functions.py`**（模板+片段拼接） |
| 6 | `classification` | llm | `gen/fragments/classification.json`（传播子分类） |
| 7 | `generate_likelihood` | agent | **`run/likelihood_function.py`** + `run/likelihood_test_result.txt`（冒烟 PASS≈401.25） |
| 8 | `generate_fit_script` | agent | **`run/fit_script.py`**（分布式主入口） |

当前模式：**全非交互**（`require_approval = false`，tools 无 `ask_user`）。公式来源 `document/combined_likelihood_math.md` 通过 `[ref].likelihood_math` 注入 prompt。

> 历史备注：之前版本 `generate_likelihood`/`generate_fit_script` 是 `require_approval = true` 的交互模式；2026-09-11 已改为非交互，并可随时改回（`llm_config_fit.toml` 中对应 stage 的 `require_approval` 字段）。

---

## 3. 同步到远程 HEP1

### 3.1 目标布局

```text
/home/hyx/LLMPWA/kk_dis/          ← 与本地 analyses/kk_dis 对齐（代码+数据+config）
/home/hyx/LLMPWA/.env             ← API key（生成器可选）
```

### 3.2 同步方式

- **小文件**（代码/config/提示词）：`remote_push` 逐个推；批量改动用 `tar` 打包后 push。
  只含代码的包很小（几十 KB），例如仓库根（agent 引擎）`repo_root_sync.tar.gz`（48K）。
- **大文件**（data/ 共 811MB，62 个 npy）：单个 tar.gz > 256MB 上传上限，**拆包**：

```bash
# 本地
tar czf kk_sync.tar.gz data config run gen llm_config_fit.toml resonances_config.toml node_config.toml ...
split -b 200M kk_sync.tar.gz kk_sync_part_    # → part_aa..part_ad（共 ~788MB）
# 依次 remote_push 每个 part 到远程（4 次）

# 远程：合并解包
cat kk_sync_part_a* > kk_sync.tar.gz
tar xzf kk_sync.tar.gz -C ~/LLMPWA/kk_dis
rm kk_sync.tar.gz kk_sync_part_*
```

> 换机器/换路径时可用 rsync + ProxyJump 一条命令替代，但 DSH remote_push 已配置好（见 `.dsh/config.yml`），本流程直接用 remote_* 工具即可。

### 3.3 `.dsh/config.yml`（remote_* 工具的跳板配置）

```yaml
remote:
  host: 127.0.0.1
  port: 20402
  user: hyx
  remoteRoot: /home/hyx/LLMPWA/kk_dis
  auth: {kind: key, keyPath: /home/iso/.ssh/id_ed25519}
  hostKeyFingerprint: <target 指纹>          # 注意 ssh2 格式 base64 带尾部 =
  proxyJump:
    host: www.insolitude.me
    port: 22
    user: root
    auth: {kind: key, keyPath: /home/iso/.ssh/id_ed25519}
    hostKeyFingerprint: <jump 指纹>
```

要点：
- 指纹是 **ssh2 raw-key SHA256 base64，末尾带 `=` 填充**，与 `ssh-keygen -lf` 输出（无 `=`）不同；用 node `ssh2` hostVerifier 探测获得。
- `remote_exec` 默认 workdir 是 remoteRoot；**后台起长任务务必 `cd` 到目标目录再 nohup**（曾因 cwd 不对导致找不到 `run/fit_script.py`）。

---

## 4. 远程拟合运行（单机双 GPU，无 Docker）

### 4.1 直接跑（前台）

```bash
cd ~/LLMPWA/kk_dis
~/miniconda3/envs/kk_fit/bin/python run/fit_script.py
```

### 4.2 后台跑 + 日志（推荐，方便监控）

```bash
cd ~/LLMPWA/kk_dis
nohup ~/miniconda3/envs/kk_fit/bin/python run/fit_script.py > /tmp/fit_run.log 2>&1 &
echo "pid $!"
```

### 4.3 为什么不需要 Docker / 多进程

`run/fit_script.py` 关键路径：

```python
def build_mesh():
    return Mesh(jax.devices(), axis_names=("event",))   # 单进程拿到全部本地 GPU
```

- 单机单进程：`jax.devices()` 直接返回 2 个 GPU，Mesh 把事件轴 shard 到两卡
- `init_distributed()` 仅在 `JAX_NUM_PROCESSES > 1` 时才 `jax.distributed.initialize`（多机才需要）
- 参数与 constellation 复制到所有设备（`NamedSharding(mesh, P())`），事件数据 `P("event")` 分片
- 无 NCCL 显式管理：XLA 自动插跨设备 all-reduce

实测（2026-09-11，HEP1）：

```text
日志：starting distributed fit: processes=1 devices=2
GPU0: 3118 MiB / 24576 MiB, ~90% util, ~310 W
GPU1: 3110 MiB / 24576 MiB, ~90% util, ~300 W
收敛：iteration 188: fit complete: success=True, nll=-3950.00223001
```

每卡仅 ~3.1GB（12.7%）显存 —— 余量 ~21GB/卡，若加大 `n_truth_mc`（当前 150000）或事件数也不会爆。

---

## 5. 监控与结果

### 5.1 监控

```bash
# GPU 占用（迭代过程中采样）
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader

# 拟合进度
tail -5 /tmp/fit_run.log
# 期望每 ~0.3s 一条：fit: iteration N: nll=...

# 进程状态
ps aux | grep fit_script | grep -v grep
```

### 5.2 结果产物（远程 `~/LLMPWA/kk_dis/output/fit/`）

| 文件 | 内容 |
|------|------|
| `fit_result_values.npy` | 65 自由参数拟合值（`result.x`） |
| `fit_result_errors.npy` | √diag(H⁻¹) 误差（Hessian 列式构造后取逆） |
| `free_params_fitted.toml` | 与 `free_params.toml` 同结构的结果 TOML |

收敛判据：日志出现 `fit complete: success=True`，nll 稳定在 **-3950.002**（三次运行一致）。

### 5.3 拉回本地

`remote_pull` 三个产物到 `analyses/kk_dis/output/fit/` 即可归档/后续绘图。

---

## 6. 关键坑速查（详见 document/ENV_NOTES.md）

1. **httpx no_proxy bug**：`NO_PROXY` 含 `[::1]` → `Invalid port: ':1]'`。跑任何 openai/httpx 前 export `no_proxy`/`NO_PROXY` = `localhost,127.0.0.1,.localdomain,::1`（`run_pipeline.sh` 已内置）。
2. **openai 3.x + httpx2** 冲突 → 钉 `openai==2.54.0`。
3. **本地 base 只读** → 一律用 `analyses/kk_dis/.venv`。
4. **后台远程任务**：`nohup ... &` + 日志文件轮询；remote_exec 有超时（SIGPIPE），长任务别前台跑。
5. **传大文件**：单文件 >256MB 上限 → `split -b 200M` 拆包。
6. **中文字符/路径**：远程 HEP1 locale 无中文包，命令与日志用英文粘贴复制避免乱码。
7. **conda 重建 env**：不要用 conda CPU jax 混进带 dnnl 的旧 env（undefined symbol: dnnl_*）；GPU 环境用 `"jax[cuda12]"` 从零建。

---

## 4.5 长时拟合与随机初值（多起点）方案

> 补充于 2026-09-11。解决了「真实拟合要跑很久 + 随机初值避免局部最优」两个问题。
> 已实测：2 起点 demo 跑通，找到两个不同局部极小（-3950.00 与 -3789.04），选优逻辑正常。

### 为什么之前总是 sleep？

`remote_exec` 没有 `run_in_background`，只有超时，所以长任务过去只能：

```text
remote_exec 起 nohup → 反复 sleep → remote_exec tail 日志
```

### 根治方法：本地 `bash` 后台任务 + ssh（关键发现）

DSH 的**本地 `bash` 工具支持 `run_in_background: true`**，任务结束时系统**自动通知**，无需 sleep 轮询。而且本地可直连 ssh 到 HEP1：

```bash
ssh -F /dev/null -o BatchMode=yes -o ConnectTimeout=20 \
    -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/tmp/dsh_probe_known_hosts \
    -J root@www.insolitude.me -p 20402 hyx@127.0.0.1 '远程命令'
```

长任务推荐模式（带输入重定向防止 ssh 挂起）：

```bash
# 本地后台任务（run_in_background: true）里执行：
ssh ... 'cd ~/LLMPWA/kk_dis && nohup ~/miniconda3/envs/kk_fit/bin/python run/multi_start_fit.py \
    --starts 8 --tag phys_run --seed 2024 > /tmp/multistart.log 2>&1 < /dev/null &'
```

远程进程用 nohup 脱离 ssh 通道，**ssh 断开/超时被杀都不影响拟合**。
本地这个后台任务本身只负责一次 ssh 启动，秒级完成；真正的等待由系统通知承担。

### 多起点随机初值 runner：`run/multi_start_fit.py`

```bash
# 远程执行（HEP1）
cd ~/LLMPWA/kk_dis
~/miniconda3/envs/kk_fit/bin/python run/multi_start_fit.py \
    --starts 8 --tag run_phys --seed 2024 --disturb 50 --timeout-min 600
```

参数：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--starts` | 8 | 起点数（含 start 0 基线） |
| `--seed` | 42 | 随机种子（每个起点 seed+N*1000） |
| `--disturb` | 50 | 乘性抖动幅度：50 → ±1%（改小则抖动更大，如 10 → ±5%） |
| `--tag` | run1 | 输出目录名 `output/multistart/<tag>/` |
| `--timeout-min` | 600 | 单个起点超时（防发散挂死队列） |

行为（算法详见 `document/random_initial_perturbation.md`）：

1. 读取 `run/free_params.toml` 的 value 作为**基线**初值
2. start 0 = 基线（不扰动）；start N>0：
   - `Amplitude.constN/thetaN` 耦合对（25 对）重新采样到**半径 0.1 的圆**上（幅值固定 0.1，相位随机）
   - 全部参数额外乘性抖动 `(U(-0.5,0.5)/disturb + 1.0)`
   - `range` 字段是高斯约束 `(center, sigma)`，不做 clamp
3. 每个起点**子进程独立跑** `fit_script.py`（环境隔离，一个失败不阻塞其他）
4. 起点产物 → `output/multistart/<tag>/start_<N>/`（values/errors/fitted.toml）
5. 总表 → `output/multistart/<tag>_summary.toml`，stdout 打印 `BEST start=N nll=...`

选优：`summary` 里的 `best` 字段（已修正：`nll` 用 float 比较，非字符串）即为**全局最优起点**，取它对应当前目录产物。

实测效果（HEP1，2 起点）：

```text
start 0 -> success=True nll=-3950.0025   (基线)
start 1 -> success=True nll=-3789.0429   (旧算法 ±5% 均匀扰动 -> 另一个局部极小!)
BEST start=1 nll=-3789.0429   # 注：这是旧的字符串比较 bug 选错；v2 已修，会正确选 -3950
```

> ⚠️ 注意：demo 的 `summary.toml` 是旧代码产物（success/nll 为字符串类型）；v2 runner 已修正为 float/bool，且初值扰动算法已更新（v3，见 `document/random_initial_perturbation.md`）。后续 `--starts ≥ 2` 跑正式实验时用最新版本。

### 组合使用：真实长时多起点流程

```text
1. 本地 bash 后台任务 ssh 启动 multistart（上节命令）
2. 系统自动通知任务结束（或手动 job_output wait）
3. remote_exec 查看 output/multistart/<tag>_summary.toml 的 best
4. remote_pull 最优起点目录产物回本地
```

---

## 7. 一页速查

```bash
# 本地：生成代码
./analyses/kk_dis/run_pipeline.sh                    # 8 stage 全自动

# 本地 → 远程：同步（小文件 remote_push；大文件 split -b 200M + push + cat）

# 远程：跑拟合（HEP1）
cd ~/LLMPWA/kk_dis
nohup ~/miniconda3/envs/kk_fit/bin/python run/fit_script.py > /tmp/fit_run.log 2>&1 &

# 远程：多起点随机初值拟合（推荐，避免局部最优）
~/miniconda3/envs/kk_fit/bin/python run/multi_start_fit.py --starts 8 --tag run_phys --seed 2024
# 结果：output/multistart/run_phys_summary.toml 的 best 即最优起点

# 远程：监控
nvidia-smi; tail -5 /tmp/fit_run.log

# 远程：确认收敛后拉回结果 → 本地 output/fit/
```

当前状态摘要（2026-09-11）：

- 生成器 v2（非交互，8 stage 全自动）✓
- 本地 `.venv`（openai 2.54.0 / jax cpu）✓
- 远程 `kk_fit`（python 3.11 / jax 0.10.2 cuda12 / 2×3090）✓
- 单机单进程双卡拟合 success=True，NLL=-3950.002，显存 2×3.1GB ✓
