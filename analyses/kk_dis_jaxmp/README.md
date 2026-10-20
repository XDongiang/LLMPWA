# kk_dis_jaxmp —— 原生 JAX 多进程 KK 拟合

旧版 `fit_object_kk.py`（Python `multiprocessing`，每卡一个独立拟合）的**原生 JAX
多进程**改造版：**一个拟合用满所有卡**，事例轴真分片，chief-only SciPy。

物理模型**逐字保留**旧版：22 条振幅道（3686 / 3773）、219 槽参数向量、92 个自由
参数、`weight_kk` 事例权重、lasso / step-function 份额约束、全部传播子。

**本目录不含数据**，只含代码 + 配置 + 自检脚本。数据请用环境变量指到已有位置。

---

## 目录内容

```
run/fit_object_kk.py            主程序（唯一需要运行的脚本）
config/logconfig_fit.json       日志配置（脚本按此读取）
gen/fragments/classification.json  道/参数分类信息（写进 pwa_info_kk.json；缺失不致命）
tests/verify_fit_object_kk.py   数学正确性：autodiff ≡ 旧版手工 grad/HVP
tests/smoke_fit_object_kk.py    单进程端到端（自造合成数据）
tests/smoke_multiproc.py        2 进程真分布式端到端（自造合成数据）
tests/check_equiv.py            1 进程 vs 2 进程数值一致性 + 分片真实性
tools/check_data.py             数据集布局自检（不加载进内存）
run_fit_multinode.sh            多节点 docker 启动脚本（沿用仓库既有约定）
node_config_3gpu.toml           2 节点 3 卡配置模板（script 已指向本目录脚本）
```

依赖只有 `numpy / jax / scipy`（`dlib` 可选，缺失时自动用内置纯 JAX 等价实现）。

---

## 快速开始

```bash
# 0. 先看数据够不够（会指出缺哪些文件、事例数是否自洽）
python tools/check_data.py --scan

# 1. 自检（不需要真实数据，自己造合成数据）
python tests/verify_fit_object_kk.py     # autodiff == 旧版公式
python tests/smoke_fit_object_kk.py      # 单进程跑通
python tests/smoke_multiproc.py          # 2 进程跑通

# 2. 单进程拟合
KK_DATA_ROOT=<数据根目录> python run/fit_object_kk.py

# 3. 多进程（每卡一个进程，env 与 run_fit_multinode.sh 一致）
JAX_COORDINATOR_ADDRESS=host:12345 JAX_NUM_PROCESSES=2 JAX_PROCESS_ID=0 \
    python run/fit_object_kk.py
```

`run/fit_object_kk.py` 会自动 `chdir` 到自己所在目录的上一层（即本目录），
所以从哪调用都行；但 `config/logconfig_fit.json` 与 `output/`、`logs/` 都相对
本目录解析。

---

> **2026-10-19 更新**：以下「数据现状」一节描述的是**旧数据集**。集群上
> `/scratch/caihao1/sunyc/cgpwa_kk/data` 已经重新导出过，用本目录
> `tools/check_data.py` 实测为**可用**（三组数据齐全、组内事例数自洽、权重匹配），
> 并已在 4×V100 上跑通真实数据拟合。详见
> [REMOTE_ENV.md](REMOTE_ENV.md) 与 [JAX_IMAGE_AND_MULTIGPU.md](JAX_IMAGE_AND_MULTIGPU.md)。

## ⚠️ 数据现状：现有数据集无法直接跑（重要）

旧模型需要 4 个只有部分数据集才有的张量：`u_rho_r_kk`、`u_rho_l_kk`、
`u_b_r_kk`、`u_b_l_kk`。`tools/check_data.py --scan` 的结果是：

| 目录 | 主数组事例数 | u_* 事例数 | 结论 |
|---|---|---|---|
| `data/real_data` | 10000 | **缺失** | 缺 4 个文件 |
| `data/mc_truth` | 1200000 | **缺失** | 缺 4 个文件 |
| `data/draw_mc/real_data` | 60097 / 120194 | 5865 | 有文件，但**事例数不一致** |
| `data/draw_mc` | 500000 / 1000000 | 500000 | 有文件，但 b123/b124 是 1000000 |

**没有任何一个目录是完全自洽的。** 权重 `weight_kk.npy` 是 5865，也对不上
`data/real_data` 的 10000。

> 为什么旧版能跑？旧版对**每个文件单独** `onp.array_split(...)`，然后在似然里
> 把 22 条道逐事例相加。长度不一致时 JAX 的广播**要么悄悄算错、要么在很深处
> 才炸**。这不是可移植的行为，所以新版不做这种猜测：`load_data` 会在开跑前
> 校验每个张量组内部事例数一致，不一致就明确报错。

**需要你做的一件事**：导出一份事例数自洽的数据集（同一组内 10 个张量长度一致，
且 `weight_kk.npy` 长度 = 实验事例数）。导出后用 `tools/check_data.py` 确认
"可用"，再跑拟合。

数据摆放约定（与旧版一致）：

```
<root>/real_data/<name>.npy      实验数据
<root>/mc_int/<name>.npy         MC 积分样本（缺失时自动退回 mc_truth/）
<root>/mc_truth/<name>.npy       truth-MC（用于 lasso 约束）
<root>/weight/weight_kk.npy      事例权重
```

`<name>` ∈ {phif0_kk, phif2_kk, u_rho_r_kk, u_rho_l_kk, u_b_r_kk, u_b_l_kk,
phi_kk, f_kk, b124_kk, b123_kk}。数据也可以分散在不同目录，用下面四个变量
分别指定。

---

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `JAX_COORDINATOR_ADDRESS` | — | 分布式协调地址 `host:port` |
| `JAX_NUM_PROCESSES` | `1` | 总进程数；`1` 时退化为本地单进程 |
| `JAX_PROCESS_ID` | `0` | 本进程编号，`0` 是 chief |
| `KK_DATA_ROOT` | `data` | 数据根目录 |
| `KK_REAL_DIR` | `<root>/real_data` | 实验数据目录 |
| `KK_MC_DIR` | `<root>/mc_int` | MC 积分目录（缺失退回 `mc_truth`） |
| `KK_TRUTH_DIR` | `<root>/mc_truth` | truth-MC 目录 |
| `KK_WEIGHT_FILE` | `<root>/weight/weight_kk.npy` | 权重文件 |
| `KKDIS_X0_NPY` | — | 用 `.npy` 覆盖 92 维初值 |
| `KKDIS_OUTDIR` | `fit` | `output/` 下的输出子目录 |
| `KKDIS_MAXITER` | `0` | Newton-CG 最大迭代（0 = 不限） |
| `KKDIS_XTOL` | `1e-6` | Newton-CG xtol（与旧版一致） |
| `KK_TRUTH_MC_CAP` | `150000` | truth-MC 截断事例数（与旧版一致） |
| `KK_MC_NORM_CONST` | `21181.60837312616` | 组合似然 MC 归一化常数 |
| `KK_MC_NORM_DENOM` | `925000.0` | MC 归一化分母 |
| `KK_WEIGHT_FALLBACK` | `0` | `1` = 权重长度不匹配时改用 1.0（不加权） |
| `KK_DATA_CAP` | `0` | 实验事例数上限（`0` = 不截断）。用于把事例数凑成进程数的整数倍，让 mesh 分片真正生效 |

---

## 多卡运行

> **集群上的实际跑法**见 [JAX_IMAGE_AND_MULTIGPU.md](JAX_IMAGE_AND_MULTIGPU.md)：
> 自建 `jax-fit.sif`（jax 0.11.2 + CUDA 12，V100/sm_70 可用）+ `run/fit_job.sbatch`。
> 单节点 4×V100、2 节点 ×2 卡、真实数据均已验证通过。

把 `node_config_3gpu.toml` 里的 IP / 网卡改成你的机器，然后：

```bash
./run_fit_multinode.sh --workdir <本目录>
```

脚本按 `[header].Chips` + 各 `[[worker]].Chips` 算出总进程数，每卡一个容器，
注入 `JAX_COORDINATOR_ADDRESS / JAX_NUM_PROCESSES / JAX_PROCESS_ID`，跑完把
header 的 `output/` 拷回来。容器内解释器是 `/opt/env/bin/python`。

**注意**：分片要求事例轴能被进程数整除，否则该数组自动退化为复制到每张卡
（分片只是省显存的优化，复制不会重复计数，结果与单进程完全一致）。所以奇数
事例数（如 5865）不会报错，只是不省显存。

---

## 输出

`output/<KKDIS_OUTDIR>/` 下：

| 文件 | 内容 |
|---|---|
| `fit_result_values.npy` | 92 个自由参数的拟合值 |
| `fit_result_errors.npy` | 误差（Hessian 逐列 → 协方差 → 对角开方） |
| `fit_result_kk.npz` | `fvalue=` / `ferror=`（旧版 `save_in_npz` 格式） |
| `fit_result_full_args.npy` | 写回 219 槽的完整参数向量 |
| `pwa_info_kk.json` | 道/参数分类 + 自由参数值 |
| `result_info.json` | success / status / nll / nit / message |

多进程时每个 rank 写自己的 `logs/fit_p<id>.log`（共享挂载下不会互相交错）。

---

## 与旧版的差异（只有并行层）

> **集群上的实测对比（数值等价性 + 性能 + 旧版 bug）见 [OLD_VS_NEW.md](OLD_VS_NEW.md)。**

| 旧版 | 新版 |
|---|---|
| `multiprocessing.Process` + `Manager`/`Barrier`/`Thread` | `jax.distributed.initialize()` |
| 每卡一个独立拟合、各拿一片数据 | **一个拟合用满所有卡**，`P("event")` 真分片 |
| 各进程各自 `scipy.optimize.minimize` | 只在 chief 跑 `minimize`，worker 服务请求 |
| `thread_grad_likelihood` / `thread_hvp` 手工拼 data+MC | 单个标量 autodiff（数学恒等） |
| `jit(fn, device=...)`、`host_callback` | 已移除 |
| `Logger` 类 | 复用 `config/logconfig_fit.json` |

**唯一必要的源码改动**：数据访问从闭包属性 `self.data_*` 改成 jit 的显式参数
`data["data_*"]`。闭包里的 jax 数组会被编进 jaxpr，在 mesh 下退化成"每卡一份
副本"，对事例轴的 `np.sum` 就不会变成跨设备 all-reduce；只有作为 jit 参数，
分片才真正生效。**表达式本身一字未改**（`tests/verify_fit_object_kk.py` 用
逐字比对 + autodiff 对照双重确认）。

---

## 已验证

| 检查 | 结果 |
|---|---|
| 模型逐字保留（calculate_* / BW / 两个 likelihood） | PASS |
| autodiff grad ≡ 旧版手工 data+MC 公式 | `max|Δ| = 5.8e-11`（量级 1.3e5） |
| autodiff grad ≡ 有限差分 | 相对误差 `9.2e-10` |
| HVP ≡ 梯度有限差分 | 相对误差 `2.6e-9` |
| 单进程端到端（合成数据） | PASS，6 个输出文件齐全 |
| 2 进程真分布式端到端 | PASS，chief-only 优化 + worker 服务 |
| 分片真实性 | rank 0 只持有 32/64 事例，数组跨进程不可 `asarray` |
| 1 进程 vs 2 进程数值透明 | NLL 相对差 `0.0`，梯度 `1.4e-16` |

上述测试均在 CPU、`jax 0.11.1` 上完成；远程 `jax-fit:latest` 镜像是 `jax 0.10.2`
（`jax.set_mesh` 存在，走同一代码路径）。

### 与旧版在集群真实数据上的对比（4×V100）

| 检查 | 结果 |
|---|---|
| x0 处似然（全量 23109 事例） | 两版**完全相同**：`2126.41833611` |
| 最终似然 | 旧 `-3092.851433167758` / 新 `-3092.874330542225`，`Δnll=0.023` |
| 92 个自由参数 | 相关系数 `0.99989`，**92/92 落在 0.16σ 内** |
| 新版可复现性 | 两次运行 `fit_result_values.npy` **逐位相同** |
| 旧版可复现性 | ❌ 每次随机种子不同，两次 `fcn` 差 `0.025` |
| 旧版误差棒 | ⚠️ 只有 16/92 有限，其余 NaN（海森矩阵在**初始点**求值） |
| 旧版 4 卡 | ❌ 真数据上 `array_split` 崩溃；凑成 23108 后目标函数被算错（lasso 惩罚重复计 4 遍），`fcn=-2900.85` vs 单卡 `-3089.15`，且不比单卡快 |
| 显存峰值（单卡） | 旧 **6722 MiB** → 新 **4636 MiB**（省 31%）；旧版曲线全程爬升，新版编译后即到顶 |
| 显存峰值（4 卡） | 新 **2818 MiB/卡**（四卡峰值差 0 MiB）；旧 3432 MiB/卡（但目标函数错误） |
| 端到端墙钟 | 旧 679–711 s → 新 94 s（4 卡），**快 ~7×** |
| 提速来源 | **75% 来自编译**（旧版把数据编成常量触发 constant folding，467 s → 32 s）；运行段单卡持平，其余提速来自 4 卡分片。详见 [OLD_VS_NEW.md §2](OLD_VS_NEW.md) |
| 新版 4 卡可复现性 | ⚠️ 默认**不可逐位复现**（NCCL 归约顺序），5 次运行 nll 极差 `0.14`；设 `NCCL_ALGO=Ring` 后逐位一致。1 卡不受影响 |

详见 [OLD_VS_NEW.md](OLD_VS_NEW.md)。

---

## 运行日志

`logs/fit.log`（单进程）或 `logs/fit_p<id>.log`（多进程）。运行中每个迭代打一行：

```
iteration 3: nll=-131383.411478 elapsed=8.3s
```

`nll` 是组合似然
`sum(data_lh) + 21181.60837312616 * log(sum(mc_lh) / 925000.0)`，
与旧版 `Control.thread_likelihood` 同一个式子。
