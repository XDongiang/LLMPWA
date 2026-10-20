# 远程环境速查（caihao1@202.114.96.180，swarm02 超算集群）

> 本次会话实测记录，2026-10-19。配置见本目录 [`.dsh/config.yml`](.dsh/config.yml)。

## 1. 连接

| 项 | 值 |
|---|---|
| 登录节点 | `caihao1@202.114.96.180:22`（主机名 `swarm02`） |
| 认证 | 本地 `~/.ssh/id_ed25519`（已授权，无需密码） |
| host key | `ssh-ed25519` SHA256 `qAOPZ8qfdAEbCD27mYHVThGSvNN4Q4vJbsIzp+FD50I=`（已在本地 `known_hosts`） |
| `remoteRoot` | `/scratch/caihao1` |

本地 OpenSSH 有一个坑：`/etc/ssh/ssh_config.d/20-systemd-ssh-proxy.conf` 权限不对，
直接用 `ssh` 会报 `Bad owner or permissions`。绕过办法是 `ssh -F /dev/null ...`。
DSH 的 `remote_*` 工具走 ssh2 库，不读系统 ssh_config，**不受影响**。

## 2. 路径映射（重要）

你给的两个路径在本集群上**字面不存在**，实际对应关系是：

| 你给的路径 | 实际路径 | 说明 |
|---|---|---|
| `~/project/hyx/` | `/project/caihao1/hyx/`（`~` 是 caihao1） | `~/project` → `/project/caihao1` 符号链接，**目前是空目录** |
| `~/scratch/sunyc/cgpwa_kk/temp.sbatch` | `/scratch/caihao1/sunyc/cgpwa_kk/temp.sbatch` | `~/scratch` → `/scratch/caihao1`；`sunyc` 在 scratch 下，不是独立用户 |
| `/home/sunyc/cgpwa_kk/data` | `/scratch/caihao1/sunyc/cgpwa_kk/data` | **`/home/sunyc` 不存在** |

`/scratch/caihao1/sunyc/` 下的文件属主都是 `caihao1`，可读写。

配额：`/scratch` 3T（已用 1.86T）、`/project` 500G（已用 413G）。
所以**工作目录建议放 `/scratch/caihao1/hyx/`**（本次已建 `/scratch/caihao1/hyx/kk_dis_jaxmp/`）。

## 3. Slurm 作业

`gpu` 分区账号权限已确认（`caihao1|gpu|normal`）。

```bash
#SBATCH --account=caihao1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:1
#SBATCH --time=168:00:00
#SBATCH --output=<绝对路径>/logs/xxx_%j.log
#SBATCH --job-name=xxx
```

| 分区 | 节点数 | 单节点 GPU | 时限 | 备注 |
|---|---|---|---|---|
| `gpu` | 45 | `gpu:4`（少数 `gpu:3`） | infinite | **本任务用这个** |
| `a100x4` | 22 | `gpu:4` | 14 天 | 示例 sbatch 里用的分区 |
| `hpxg` / `hpib` / `pub` 等 | — | 无 GPU | — | — |

- `gpu` 分区实测硬件：**Tesla V100-SXM2-16GB**，驱动 575.57.08，CUDA 12.9。
- 实测排队很快（探针作业秒级调度），空闲节点 7 个（`g[0041,0102-0104,0108-0110]`）。
- 用 `sbatch` 提交，日志写绝对路径（`--output` 相对路径相对提交目录）。
- 登录节点**没有 GPU**，也**没有 nvidia-smi**，只能在作业里看。

实测提交记录：`3728415`（probe）、`3728422`（GPU 探针），均 COMPLETED。

## 4. ⚠️ 运行时环境：JAX 尚未安装

这是当前**最大的阻塞项**。实测结果：

| 位置 | Python | JAX |
|---|---|---|
| 登录节点默认 `python3` | 3.12.1（miniconda3） | ❌ 无 |
| conda env `jax`（名字骗人） | 3.9.19 | ❌ 无（只有 numpy 1.26.4 / pandas） |
| 其余 6 个 conda env | — | ❌ 均无 |
| `singularity/tensorflow.sif` | 3.8.10 | ⚠️ jax **0.4.1**（2022 年，太旧） |
| `module avail` | — | ❌ 无 jax 模块 |

本分析（`kk_dis_jaxmp`）需要 `numpy / jax / scipy`，且用了 `jax.distributed.initialize()`
与 `jax.set_mesh`，README 记录在 `jax 0.11.1` 上验证、远程 `jax-fit:latest` 镜像是 `jax 0.10.2`。
集群上**没有这个镜像**（`jax-fit` 全盘搜不到，只有 `tensorflow.sif`）。

好消息：
- 计算节点能正常识别 GPU，CUDA 12.9 驱动就绪。
- PyPI 与清华镜像都通（`pip config` 已指向 `pypi.tuna.tsinghua.edu.cn`）。
- 有 `module load singularity/4.5`（也有 `apptainer/1.4.5`），`/usr/bin/podman` 存在，无 docker。

**下一步二选一**（等你定）：
1. 在 conda 里建环境装 GPU 版 jax：`pip install "jax[cuda12]"`（V100 = sm_70，需 jaxlib 支持；新版 jax 对 V100 仍支持但需确认）。
2. 自建 `jax-fit` singularity 镜像（对齐 README 里的 `jax 0.10.2`）。

## 5. 数据：可用 ✅

数据根目录 `/scratch/caihao1/sunyc/cgpwa_kk/data`（注意**不是** `KK_DATA_ROOT` 默认的 `data/`，
要用环境变量指过去）。用本目录的 `tools/check_data.py` 实测：

```
实验数据    .../data/real_data   齐全        23109 事例
MC 积分     .../data/mc_int      齐全        925000 / 1485000
truth-MC    .../data/mc_truth    齐全        150000
权重        .../data/weight/weight_kk.npy  长度 23109  匹配
可用。
```

⚠️ **README 里「现有数据集无法直接跑」的结论已经过时**：那份 README 写的是
`real_data` 10000 事例且缺 4 个 `u_*` 文件。现在的数据已经重新导出过，四个
`u_rho_r/l_kk`、`u_b_r/l_kk` 都在，组内事例数一致，权重也匹配。**可以直接跑拟合。**

需要的环境变量：

```bash
KK_REAL_DIR=/scratch/caihao1/sunyc/cgpwa_kk/data/real_data
KK_MC_DIR=/scratch/caihao1/sunyc/cgpwa_kk/data/mc_int
KK_TRUTH_DIR=/scratch/caihao1/sunyc/cgpwa_kk/data/mc_truth
KK_WEIGHT_FILE=/scratch/caihao1/sunyc/cgpwa_kk/data/weight/weight_kk.npy
```

数据规模：`real_data` 48M、`mc_int` 1.7G、`mc_truth` 4.0G（另有 `draw_data`/`draw_mc` 各 1.6G）。
注意 V100 只有 16GB 显存，而 `mc_int` 单数组 925000×若干道 × float64 很大，
**显存够不够是下一个要实测的点**（本分析的 `P("event")` 分片正是为此）。

## 6. 示例 sbatch 原文

`/scratch/caihao1/sunyc/cgpwa_kk/temp.sbatch`：

```bash
#!/bin/bash
#SBATCH --account=caihao1
#SBATCH --partition=a100x4
#SBATCH --cpus-per-task=5
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=168:00:00
#SBATCH --output="output/fit/fit_result_1/fit_1.log"
#SBATCH --job-name=sean_1

module load singularity
cd $SLURM_SUBMIT_DIR
singularity exec --nv --bind /project/`whoami`,/scratch/`whoami` \
    /scratch/caihao1/singularity/tensorflow.sif \
    python3 /scratch/caihao1/sunyc/cgpwa_kk/run/temp_pull_run_1.py
```

注意：它用 `a100x4` 分区、`tensorflow.sif` 镜像（内含旧 jax 0.4.1），
跑的是 `temp_pull_run_*.py` 拉取脚本，不是本分析的 `fit_object_kk.py`。
本任务按你的要求改用 `--partition=gpu`。

## 7. 远程工作目录

已在 `/scratch/caihao1/hyx/kk_dis_jaxmp/` 建好 `run/ logs/ tools/`，
`tools/check_data.py` 已上传并跑通。探针日志在 `logs/probe_*.log`。

## 8. 已知 DSH 工具问题

`remote_exec` 的 `workdir` 参数**当前被 provider 忽略**：无论传什么，
命令都在 `$HOME` 下执行（`remote-ssh2` 的 `run`/`start` 只写了 `request.command`，
没用 `request.cwd`；`env` 同样被忽略）。绕过办法：命令里显式 `cd <dir> && ...`。
