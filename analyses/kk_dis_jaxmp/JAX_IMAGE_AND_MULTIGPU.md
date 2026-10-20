# 自建 JAX 镜像 + 多 GPU 通信验证（caihao1@202.114.96.180）

> 会话实测记录，2026-10-19。全部结论都来自集群上真实跑过的 sbatch 作业（作业号见文末）。

## 结论速览

| 问题 | 结论 |
|---|---|
| 能否自建 singularity 镜像 | ✅ 可以。无 sudo，用 `--fakeroot` + 本地 base image 构建成功 |
| JAX 能否在 V100 上跑 | ✅ 可以，但**必须走 CUDA 12 线**（CUDA 13 砍掉了 sm_70） |
| 单卡是否识别 V100 | ✅ `backend=gpu`，jax 0.11.2 |
| 单节点 4×V100 卡间通信 | ✅ 全部通过，NVLink 全互联，实测有效带宽 ~250 GB/s |
| 多节点通信 | ✅ 2 节点 × 2 卡通过，~350–394 GB/s（实测口径见下） |
| 真实数据能否跑 | ✅ 23109 事例，4 卡峰值显存仅 ~2.8 GB/卡（V100 有 16GB，余量充足） |
| 本分析自带测试 | ✅ `VERIFY PASS` + `SMOKE PASS` + `MULTIPROC PASS` |

**镜像位置**：`/scratch/caihao1/hyx/kk_dis_jaxmp/images/jax-fit.sif`（3.5 GB）

---

## 1. 镜像构建

### 1.1 构建环境的关键约束

集群上**没有 sudo**，`caihao1` 也不在 `/etc/subuid` 里。但：

- `module load singularity` 实际加载的是 **apptainer 1.5.2**（`/software/apptainer/1.5/bin`）。
- `allow setuid` / `allow user ns` 都是 `yes`，用户命名空间上限 10000。
- 所以 **`--fakeroot` 可用**（走 root-mapped namespace + fakeroot）。

⚠️ 两个坑：

1. **`module load` 不能放在管道里**。`module load singularity | head` 会在子 shell 里执行，
   PATH 不会生效，表现为 `singularity: command not found`。要单独一行执行。

2. **`apptainer build` 里 `Bootstrap: docker` 会走 `registry-1.docker.io`，而它不通**
   （DNS 解析到 103.56.16.112 后 i/o timeout）。必须先 `apptainer pull` 一个 base 镜像，
   再用 `Bootstrap: localimage` 构建。

可用镜像源（实测）：

| 源 | 结果 |
|---|---|
| `docker.m.daocloud.io` | ✅ 401（正常，说明可达，能 pull） |
| `quay.io` | ✅ 401 |
| `docker.nju.edu.cn` | 403 |
| `registry-1.docker.io` | ❌ 超时 |
| `mirror.ccs.tencentyun.com` | ❌ 000 |

另外 `--fakeroot` 配 **老 glibc 的镜像会失败**：拿 `tensorflow.sif` 测 `--fakeroot` 报
`GLIBC_2.33/2.34 not found (required by faked)`。换成现代 base（ubuntu 22.04 / python:3.12-slim）
就正常，所以 base 镜像要选新的。

### 1.2 构建步骤

```bash
module load singularity
cd /scratch/caihao1/hyx/kk_dis_jaxmp/images

# 1) 先拉 base（走镜像源，不能用 docker://python:3.12-slim）
apptainer pull --force base_py312.sif docker://docker.m.daocloud.io/library/python:3.12-slim

# 2) 用 localimage 构建
apptainer build --fakeroot jax-fit.sif jax-fit.def
```

`jax-fit.def` 要点：

```
Bootstrap: localimage
From: /scratch/caihao1/hyx/kk_dis_jaxmp/images/base_py312.sif

%post
    export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
    python3 -m pip install --no-cache-dir "jax[cuda12]==0.11.2" numpy scipy
```

### 1.3 为什么锁 CUDA 12 + jax 0.11.2

V100 是 **sm_70**。NVIDIA 在 **CUDA 13 里移除了 Volta**，对应 JAX 的 `jax[cuda13]` 要求
`sm_75+`；只有 `jax[cuda12]` 这条线还带 `sm_52+`（含 sm_70）。

我没有只信文档，而是**直接解包 wheel 检查了 kernel 里的架构字符串**：

```
jax-cuda12-pjrt 0.11.2  ->  sm_70 ✅ (还有 sm_52..sm_93)
jax-cuda12-pjrt 0.10.2  ->  sm_70 ✅
```

（真正装 kernel 的是 `jax-cuda12-pjrt`，不是 `jax-cuda12-plugin` —— 后者只有 8.6 MB，
里面根本没有 sm_ 内核。这点容易搞错。）

参考：[JAX 安装文档](https://docs.jax.dev/en/latest/installation.html)、
[NVIDIA JAX container versions](https://docs.nvidia.com/jax-toolbox/container-versions)。

镜像内实测：`jax 0.11.2 / numpy 2.5.4 / scipy 1.18.1`，XLA `cuda 12090`，
NCCL `2.32.3+cuda12.9`，`nvidia-*` 运行时库齐全（cudnn/cublas/nccl 等）。

---

## 2. 多 GPU 通信验证

### 2.1 硬件

`gpu` 分区 = **Tesla V100-SXM2-16GB**，每节点 4 卡，驱动 575.57.08 / CUDA 12.9。

拓扑（`nvidia-smi topo -m`）是**全互联 NVLink**：

```
        GPU0  GPU1  GPU2  GPU3
GPU0     X    NV2   NV2   NV2
GPU1    NV2    X    NV2   NV2
GPU2    NV2   NV2    X    NV2
GPU3    NV2   NV2   NV2    X
```

每卡 6 条 NVLink × 25.781 GB/s，P2P 全 `OK`（`nvidia-smi topo -p2p r`）。

### 2.2 两种跑法都通

**方式 A：单进程看 4 卡**（`--gres=gpu:4`，一个 python 进程 `jax.devices()` 拿到 4 个设备）

**方式 B：4 进程每进程一卡**（`jax.distributed.initialize`，用节点 IP 作 coordinator）

两种方式都实测通过：

```
单进程:  4 devices, psum PASS, 分片每卡 (8,16), 跨设备 reduce PASS
4 进程:  rank 0..3, allgather -> [0. 1. 2. 3.], cross-process reduce got=6.0 want=6.0 -> PASS
```

**多节点（2 节点 × 2 卡 = 4 进程）也通过**，用 `srun` 拉起、取首个节点 IP 作 coordinator：

```
nodes: g[0108-0109], coordinator=10.34.0.108:12366
distributed.initialize ok in 0.1s
4 个 rank 全部: cross-process reduce: got=6.0 want=6.0 -> PASS
=== srun rc: 0 ===
```

### 2.3 通信真的走了 NCCL + NVLink

`NCCL_DEBUG=INFO` 证据：

```
NCCL INFO NCCL version 2.32.3+cuda12.9
NCCL INFO Check P2P Type isAllDirectP2p 1 directMode 0 isAllCudaP2p 1
NCCL INFO Channel 00/0 : 0[0] -> 1[1] via P2P/CUMEM
NCCL INFO comm ... nRanks 4 nNodes 1 localRanks 4
```

`isAllDirectP2p 1` = 全直接 P2P，`P2P/CUMEM` = 走 NVLink 的 CUDA 内存直连。

> 附带两条 NCCL WARN 是**环境正常现象**，不影响结果：
> `lib wrapper not initialized`（IB 库没初始化）、
> `rma_socket requires GDRCopy ... not available`。单节点通信走 NVLink，用不到 IB；
> 跨节点会退回 `NET/Socket`（实测跨节点带宽因此偏低，见下）。

### 2.4 实测带宽

必须说清楚**方法论**，因为我前几版探针测错了，数字虚高得离谱：

- ❌ **对分片轴做 `sum`**：各卡先本地求和，只 all-reduce 一个 `(nproc,)` 的小向量，
  测到的是 kernel 启动开销，**不是带宽**。
- ❌ **不 jit 就计时**：测到的是 trace/compile 时间，表现为与数据量无关的常数。
- ❌ **闭包捕获分片数组**：会被编成 jaxpr 常量，分布式下直接报
  `spans non-addressable devices`。**这正是本分析 README 里那条"唯一必要的源码改动"的同一个坑**，
  必须把数组作为 `jit` 的显式参数传入。

修正后的结果（`run/gpu_bandwidth.py`，单位 GB/s，busbw = 链路实际流量口径）：

**单进程 4 卡（NVLink）**

| 数据量 | all-gather busbw | all-reduce busbw | p2p ring busbw |
|---|---|---|---|
| 8 MiB | 20.2 | 41.4 | 28.9 |
| 64 MiB | 67.8 | 187.5 | 98.9 |
| 256 MiB | 77.7 | **255.6** | 118.6 |

**4 进程（每进程一卡）**

| 数据量 | all-gather busbw | all-reduce busbw | p2p ring busbw |
|---|---|---|---|
| 8 MiB | 24.4 | 53.4 | 38.9 |
| 64 MiB | 64.9 | 191.8 | 99.9 |
| 256 MiB | 79.8 | **246.3** | 124.5 |

all-reduce 到 ~250 GB/s 与 4 卡全互联 NVLink 的理论量级相符。正确性同时校验通过
（all-gather 每卡得到全集、all-reduce 每卡保持分片）。

**2 节点 × 2 卡** 用早期探针测到 ~350–394 GB/s 有效值，但那是**旧口径**（对分片轴求和，
不可比），跨节点受 `NET/Socket` 限制，真实跨机带宽应该更低 —— 需要的话用
`gpu_bandwidth.py --mp 4` 跨节点重测。

---

## 3. 真实数据端到端

`run/test_realdata.sbatch`：4 卡 + `KKDIS_MAXITER=1` + 真实数据。

```
native jax fit: processes=4 devices=4
data sizes: n_data=23109 n_mc=925000 n_truth=150000
smoke: nll=2126.41833611 hvp_shape=(92,) n_events=23109
iteration 1: nll=279.208091137 elapsed=18.0s
fit complete: nit=1
results written to output/realprobe   (6 个输出文件齐全)
```

**显存峰值 ~2.8 GB/卡**，V100 的 16GB 绰绰有余。

⚠️ 一个细节：`n_data=23109` **不能被 4 整除**（23109 = 4×5777 + 1）。
按 README 的说明，这时该数组会退化为**复制到每张卡**（分片只是省显存的优化，
复制不会重复计数，结果与单进程一致）。所以这次峰值显存不是分片后的最优值；
若要真正省显存，需要事例数能被进程数整除的数据集。

---

## 4. 可复用模板

远程 `/scratch/caihao1/hyx/kk_dis_jaxmp/` 下：

| 文件 | 用途 |
|---|---|
| `images/jax-fit.def` | 镜像定义 |
| `run/fit_job.sbatch` | **正式拟合模板**（支持多节点，`NODES=2 sbatch ...`） |
| `run/gpu_comm_probe.py` | 多卡通信正确性探针 |
| `run/gpu_bandwidth.py` | 卡间带宽基准 |
| `run/test_*.sbatch` | 各项验证作业（单卡/4卡/多节点/NCCL/冒烟/真实数据） |
| `tools/check_data.py` | 数据布局自检 |

正式跑拟合：

```bash
cd /scratch/caihao1/hyx/kk_dis_jaxmp
sbatch run/fit_job.sbatch                 # 单节点 4 卡
NODES=2 sbatch run/fit_job.sbatch         # 2 节点 × 4 卡
KKDIS_MAXITER=100 sbatch run/fit_job.sbatch
```

要点：每卡一进程；coordinator 用**首个节点的可路由 IP**（`hostname` 不一定可路由，
用 `hostname -i`）；`XLA_PYTHON_CLIENT_PREALLOCATE=false` 便于排查 OOM。

---

## 5. 作业号索引

| 作业 | 内容 | 结果 |
|---|---|---|
| 3728415 / 3728422 | GPU 环境探针 | COMPLETED |
| 3728548 | 单卡 JAX | PASS（1 device, backend=gpu） |
| 3728612 | 4 卡单进程 + 4 进程分布式 | PASS |
| 3728622 | 2 节点 × 2 卡 | PASS |
| 3728627 | `verify` + `smoke` 测试 | VERIFY PASS / SMOKE PASS |
| 3728642 | `smoke_multiproc`（2 进程，CPU） | MULTIPROC PASS |
| 3728646 | NCCL 验证 | PASS（P2P/CUMEM, isAllDirectP2p 1） |
| 3728684 | 带宽基准 | PASS（all-reduce ~250 GB/s） |
| 3728690 | 真实数据 4 卡 | PASS（峰值 2.8 GB/卡） |

> 注：`smoke_multiproc.py` 自带 `JAX_PLATFORMS=cpu`（它设计上是 CPU 测试），
> 所以日志里出现 `[Gloo]` 是正常的，不代表 GPU 路径走 Gloo。
> GPU 路径的 NCCL 证据来自 3728646。
