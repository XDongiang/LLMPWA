# kk_dis_jaxmp — KK 离散度拟合（JAX / 多卡）

用原生 JAX 重写的 KK 离散度拟合程序，替换原来的 Python `multiprocessing` +
`threading` 版本。在 4×V100 上端到端 **679 s → 94 s（约 7×）**，
单卡显存峰值 **6722 → 4636 MiB**，并修掉了原版误差棒的 bug。

本目录是可直接交付运行的完整包：代码、Slurm 脚本、镜像定义、运行说明。

---

## 1. 目录内容

```
kk_dis_jaxmp_release/
├── README.md                    ← 本文件
├── run_fit_4gpu.sbatch          单节点 4 卡（生产用，事例轴真分片）
├── run_fit_1gpu.sbatch          单卡全量（调试 / 对照基准）
├── run/
│   └── fit_object_kk.py         主程序（唯一的代码文件，约 1900 行）
├── images/
│   ├── jax-fit.def              Apptainer 镜像定义（一步即可构建）
│   └── README.md                镜像构建与验证说明
├── config/
│   └── logconfig_fit.json       日志配置（程序启动时读，必须在工作目录下）
├── scripts/
│   ├── check_data.py            数据自检（跑拟合前建议先跑）
│   └── vram_sample.sh           显存采样器（后台每 5 s 记一次 nvidia-smi）
└── reference/
    └── original_fit_object_kk.py   原始 multiprocessing 版（仅供对照，不参与运行）
```

**只有 `run/fit_object_kk.py` 是必须的程序代码**，其余是配置、脚本和参考。

---

## 2. 快速开始

假设部署到 `/scratch/<user>/kk_dis_jaxmp`：

```bash
# 0) 放好代码
cd /scratch/<user>/kk_dis_jaxmp

# 1) 建镜像（见 images/README.md，约 3.5 GB / 5–15 分钟）
module load singularity
apptainer build --fakeroot images/jax-fit.sif images/jax-fit.def

# 2) 改 run_fit_4gpu.sbatch 顶部的路径（WORKDIR / DATA / SIF / BIND）

# 3) 建日志目录（Slurm 不会自动建；否则作业会因写不了 stdout 而失败）
mkdir -p logs

# 4) 数据自检
apptainer exec --nv --bind /project/<user>,/scratch/<user> \
    images/jax-fit.sif python3 scripts/check_data.py --scan

# 5) 提交
sbatch run_fit_4gpu.sbatch
```

结果写在 `output/<KKDIS_OUTDIR>/`，默认 `output/fit/`。
判断成功与否看两处：日志里的 `=== fit rc=0 ===`，和
`output/<KKDIS_OUTDIR>/result_info.json`。

---

## 3. 数据要求

程序从**三个目录**读数据，用环境变量指定（见 §4）：

| 环境变量 | 内容 | 事例数（本次分析） |
|---|---|---|
| `KK_REAL_DIR` | 实验数据 | 23109 |
| `KK_MC_DIR` | MC 积分样本（归一化与 MC 似然共用） | 925000 |
| `KK_TRUTH_DIR` | MC 真值样本 | 150000 |
| `KK_WEIGHT_FILE` | 实验事例权重 `weight_kk.npy` | 23109 |

三个目录都需要下列 `.npy` 文件（每个约 12 个张量 + 4 个标量）：

```
张量（.npy）: phif0_kk, phif2_kk, u_rho_r_kk, u_rho_l_kk,
              u_b_r_kk, u_b_l_kk
标量（.npy）: phi_kk, f_kk, b124_kk, b123_kk
```

> 目录里可能还有 `*_pipi.npy`、`*_p_kk.npy`、`kst*` 等文件 —— 那是别的分析用的，
> 本程序不读，可以不管。

**权重文件缺失时**程序会告警并退化为不加权（权重全 1）；
**权重长度与事例数不匹配**时会直接报错，可用 `KK_WEIGHT_FALLBACK=1` 强制退化。

### 分片与事例数

4 卡分片要求**事例轴能被进程数整除**。23109 不能被 4 整除，
此时该数组会自动退化为「复制到每卡」——**结果仍然正确，只是不省显存**。
注意程序**不会为此打印日志**（避免刷屏），需要自己留意事例数是否被整除。
若要真正分片，用 `KK_DATA_CAP` 把事例数截到 4 的倍数：

```bash
KK_DATA_CAP=23108 sbatch run_fit_4gpu.sbatch
```

（本次对比就是这样跑的：4 卡用 23108，1 卡用全量 23109。）

---

## 4. 环境变量

### 数据位置

| 变量 | 默认 | 说明 |
|---|---|---|
| `KK_REAL_DIR` | `<KK_DATA_ROOT>/real_data` | 实验数据目录 |
| `KK_MC_DIR` | `<KK_DATA_ROOT>/mc_int` | MC 积分样本；`mc_int/` 不存在时自动退回 `mc_truth/` |
| `KK_TRUTH_DIR` | `<KK_DATA_ROOT>/mc_truth` | MC 真值样本 |
| `KK_WEIGHT_FILE` | `<KK_DATA_ROOT>/weight/weight_kk.npy` | 事例权重 |
| `KK_DATA_ROOT` | `data` | 上面几个的公共根目录（都被显式指定时无意义） |

### 运行参数

| 变量 | 默认 | 说明 |
|---|---|---|
| `KKDIS_OUTDIR` | `fit` | 输出到 `output/<这个值>/` |
| `KKDIS_MAXITER` | `0` | 最大迭代数，0 = 不限 |
| `KKDIS_XTOL` | `1e-6` | 牛顿-CG 收敛阈值 |
| `KK_DATA_CAP` | `0` | 实验事例上限，0 = 不截断 |
| `KKDIS_X0_NPY` | 空 | 指定初值 `.npy`（219 维），留空用内置初值 |
| `KK_WEIGHT_FALLBACK` | `0` | 设 `1` 时权重不匹配就退化为不加权 |
| `KK_TRUTH_MC_CAP` | 内置 | MC 真值事例上限 |

### 随机初始值（只对两个参数，默认关闭）

对 3773 侧的两个 `inta` 参数（复振幅）做随机起点，其余 90 个参数仍用内置初值：

| 变量 | 默认 | 说明 |
|---|---|---|
| `KK_RANDOM_INIT` | `0` | 设 `1` 开启随机初始值 |
| `KK_TRIALS` | `1` | 一个作业内跑几次，每次换一个随机起点 |
| `KK_RANDOM_SEED` | 空 | 随机种子；留空则每次从系统熵取新种子（种子会打进日志和汇总） |
| `KK_RANDOM_LO` / `KK_RANDOM_HI` | `-0.5` / `0.5` | 随机区间 `[lo, hi)` |

被随机的两个参数：

| 自由参数下标 | `ARGS_LIST` 槽位 | 名称 | 内置初值 |
|---|---|---|---|
| 90 | 125 | `kk_3773_amp_inta` | `0.0` |
| 91 | 126 | `kk_3773_tha_inta` | `0.2` |

开启后结果**按 trial 分目录**存放，并写一份汇总：

```
output/<KKDIS_OUTDIR>/
├── trials_summary.json      # 每个 trial 的 nll / 迭代数 / 随机起点 + 最优 trial
├── trial_0/                 # 与 §5 相同的六个文件
├── trial_1/
└── trial_2/
```

`trials_summary.json` 里的 `best_trial` / `best_nll` / `best_outdir`
直接给出 nll 最小的那个起点，便于挑选。

```bash
# 例：4 卡跑 5 个随机起点，固定种子以便复现
KK_RANDOM_INIT=1 KK_TRIALS=5 KK_RANDOM_SEED=20261020 sbatch run_fit_4gpu.sbatch
```

**注意**：

- 编译只做一次，所有 trial 共用同一份 jit，所以加 trial 的开销 ≈ 每次多一轮优化
  （约 40 s/次），不会重复付编译成本。
- `KK_RANDOM_SEED` 固定时，**抽样序列可逐位复现**（同一作业内多 trial 也一致）。
- 不开启时行为与之前**完全相同**：仍写 `output/<KKDIS_OUTDIR>/` 根目录，
  文件名不变，单卡下逐位可复现。

### 并行 / 性能

| 变量 | 说明 |
|---|---|
| `JAX_NUM_PROCESSES` | 进程总数 = 卡数（脚本自动设为 `$SLURM_NTASKS`） |
| `JAX_PROCESS_ID` | 本进程编号（脚本自动设为 `$SLURM_PROCID`） |
| `JAX_COORDINATOR_ADDRESS` | `<可路由IP>:<端口>`，多进程集合通信入口 |
| `XLA_PYTHON_CLIENT_PREALLOCATE` | 建议 `false`（V100 只有 16 GB） |
| `NCCL_ALGO` | **多卡想逐位复现就设 `Ring`**，见 §7 |

单卡运行时不要设 `JAX_COORDINATOR_ADDRESS`，并设 `CUDA_VISIBLE_DEVICES=0`
（程序按 `jax.devices()` 建 mesh，只靠进程数限制不住卡）。

---

## 5. 输出文件

写在 `output/<KKDIS_OUTDIR>/`：

| 文件 | 内容 |
|---|---|
| `result_info.json` | `success` / `status` / `nll` / `nit` / `message` |
| `fit_result_values.npy` | 92 个自由参数的最佳拟合值 |
| `fit_result_errors.npy` | 92 个参数的 1σ 误差（由 Hessian 逆的对角线开方） |
| `fit_result_full_args.npy` | 写回 219 槽的完整参数向量 |
| `fit_result_kk.npz` | `fvalue`（219 槽）+ `ferror`（92 个），兼容原版读取方式 |
| `pwa_info_kk.json` | 参数 + 道分类元信息（`mod_info` 默认为 `{}`，见下） |

若开启了随机初始值或多 trial（见 §4），上述六个文件会写进
`trial_<n>/` 子目录，并在上级目录多一个 `trials_summary.json`。

日志同时输出到终端和 `logs/fit.log`；多进程时每个 rank 写 `logs/fit_p<id>.log`
（都写同一个文件会互相交错）。

> **关于 `mod_info`**：程序会尝试读 `gen/fragments/classification.json` 作为
> 道分类元信息写进 `pwa_info_kk.json`。**本包不含该文件**（它属于另一个分析），
> 因此 `mod_info` 为 `{}` —— 这是预期行为，**不影响任何数值结果**，
> 拟合参数仍在 `free_args` 里。若你有对应本分析的分类文件，
> 放到 `<工作目录>/gen/fragments/classification.json` 即可自动带上。

---

## 6. 性能与资源（实测，4×V100-SXM2-16GB）

| 配置 | 端到端 | 编译 | 优化 | 显存峰值/卡 |
|---|---|---|---|---|
| 4 卡，23108 事例 | **94 s** | 32 s | 42 s | **2818 MiB** |
| 1 卡，23109 事例 | 274 s | 29 s | 211 s | 4636 MiB |

- **编译占 4 卡总耗时的 34%**。代码只建 3 个 jit（nll / grad / hvp），
  且数据作为 **jit 的显式参数**传入 —— 千万不要改成闭包常量，
  否则 XLA 会做常量折叠，编译时间会涨到 400 s 以上（这正是原版慢的主因）。
- 4 卡时显存不会降到 1/4：每卡有约 **2.2 GB 固定开销**（编译产物、权重、
  真值数据），只有事例轴被真正切分。所以再加到 8 卡收益很小。
- 申请资源注意：`gpu` 分区 `DefMemPerGPU=29650`，即**主机内存配额 = 29650 MB × 卡数**。
  单卡配额（约 29 GB）装不下全量数据，故 `run_fit_1gpu.sbatch` 里
  虽然只用 GPU 0，仍写 `--gres=gpu:4` 来换取内存配额。

---

## 7. ⚠️ 多卡结果的复现性

**单卡逐位可复现**（跨不同节点验证过）。

**多卡默认不可逐位复现**：同样的配置、同样的节点，每次结果在小数点后第 4 位
开始不同。实测同节点 3 次 4 卡运行：

| 运行 | nll | 迭代 |
|---|---|---|
| 1 | -3089.12091571 | 62 |
| 2 | -3089.14911219 | 60 |
| 3 | -3089.10265367 | 57 |

nll 极差 **0.14**，前 9 次迭代逐位相同、从第 10 次开始分岔。

**原因**：多卡时对事例轴的求和变成 NCCL all-reduce，NCCL 默认按拓扑自动选算法
（Ring / Tree / NVLS），不同算法的浮点加法结合顺序不同，末位差异经牛顿-CG 放大。

**解决**：固定归约算法即可完全消除。加一行环境变量：

```bash
export NCCL_ALGO=Ring     # 或 Tree
```

验证过：固定后两次运行的 `nll`、迭代数、92 个参数、误差棒**全部逐位一致**。

**影响有多大**：这 0.14 的抖动**不改变物理结论** ——
参数离散度相对 1σ 误差最大 0.51σ、中位 0.21σ，92 个参数全部远小于 1σ。
但若要发表或做严格对比，建议固定 `NCCL_ALGO`。

---

## 8. 常见问题

**`FileNotFoundError: config/logconfig_fit.json`**
程序启动时会 `os.chdir` 到自己的上级目录，并从**当前工作目录**读
`config/logconfig_fit.json`。请从包根目录（即 `run/` 的上一级）运行。

**`module load singularity` 后仍 `singularity: command not found`**
不要把 `module load` 放进管道（如 `module load singularity | head`），
那会在子 shell 里执行、PATH 不生效。单独一行执行。

**`apptainer build` 卡在 `Fetching OCI image` / 超时**
集群里 `registry-1.docker.io` 不通。`images/jax-fit.def` 里已改用
`docker.m.daocloud.io` 镜像源；换环境时改成你能访问的源。

**`--fakeroot` 报 `GLIBC_2.33 not found`**
说明 base 镜像太老（老 glibc 与 fakeroot 不兼容）。用现代 base，
如 `python:3.12-slim`（本包用的就是它）。

**多卡 OOM 或 `Detected 1 oom_kill event`**
多半是**主机内存**而非显存超了。检查 `--gres=gpu:N` 是否给了足够配额
（主机内存 = 29650 MB × N）。

**结果和单卡不一致**
先确认事例数是否相同（`KK_DATA_CAP`），再看 §7 的 `NCCL_ALGO`。
如果 4 卡跑的是截断数据、1 卡跑全量，nll 本来就会差（本次实测差 3.74）。

**4 卡显存没降下来 / 和单卡差不多**
多半是事例数不能被卡数整除，数据被复制到每张卡而不是分片（见 §3「分片与事例数」）。
用 `KK_DATA_CAP` 截到卡数的整数倍。程序不会为此报警，需自己检查。

**结尾出现一堆 `WatchTasksAsync failed ... Connection refused` 的 GRPC 报错**
**这是正常的**，不影响结果。拟合跑完后各进程陆续退出，
JAX 的协调服务先关掉，其余进程去连就会报连接被拒。
判断成功与否只看两处：`=== fit rc=0 ===`，以及
`output/<KKDIS_OUTDIR>/result_info.json` 里的 `success` / `nll`。

**`success: false` + `Maximum number of iterations has been exceeded`**
迭代数用完了（`KKDIS_MAXITER` 设得太小，或确实没收敛）。把
`KKDIS_MAXITER` 设为 0（不限）重跑。

---

## 9. 监控显存（可选）

`scripts/vram_sample.sh` 在后台每 5 s 记一次每张卡的 `memory.used`：

```bash
# 用法: vram_sample.sh <输出csv> <间隔秒> <总时长秒>
bash scripts/vram_sample.sh logs/vram.csv 5 21600 &
SAMPLER_PID=$!

# ... 跑拟合 ...

kill $SAMPLER_PID
# 峰值汇总
awk -F, '{if($3+0>m[$2])m[$2]=$3+0} END{for(k in m) printf "gpu%s peak=%d MiB\n",k,m[k]}' logs/vram.csv
```

CSV 格式：`elapsed_s,index,mem_used_mib,mem_total_mib,util_pct`。
`elapsed_s` 从采样器启动算起，与提交脚本同时开始，因此可以直接和日志里的
阶段时间戳对齐（前提是采样器在脚本开头启动，见 `run_fit_*.sbatch` 的写法）。

实测参考值：4 卡每卡峰值 **2818 MiB**，单卡 **4636 MiB**。

---

## 10. 版本要点

- **JAX 0.11.2（CUDA 12 线）**：V100 是 sm_70，CUDA 13 已移除 Volta 支持，
  必须锁在 CUDA 12 这条线上。
- **X64 必须开启**：拟合用 float64，程序内已 `config.update("jax_enable_x64", True)`。
- **依赖**：`jax[cuda12]`、`numpy`、`scipy`；`dlib` 可选
  （镜像里没装，程序内置了公式等价的替代实现，不影响结果）。
- 程序**不含任何随机数**，单卡下结果完全确定。

---

## 附：与原始版本的对应关系

`reference/original_fit_object_kk.py` 是原始的 `multiprocessing` 实现，
仅用于对照。两者的拟合模型、参数向量（219 槽 / 92 自由）、
目标函数完全等价：同一初值下 x0 处似然都是 `2126.41833611`，
最终 92 个参数相关系数 0.99989、全部落在 0.16σ 内。

主要差异：

| | 原版 | 本版 |
|---|---|---|
| 并行 | `multiprocessing` + `threading` | `jax.distributed` + `jax.sharding` |
| 梯度/HVP | 手写公式 | `jax.grad` / `jax.jvp` |
| 数据传入 | 闭包常量（触发常量折叠） | jit 显式参数 |
| 误差棒 | 在海森矩阵初始点求值 → 76/92 为 NaN | 在最优解处求值 → 92/92 有效 |
