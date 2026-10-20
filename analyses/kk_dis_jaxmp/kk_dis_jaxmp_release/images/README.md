# 运行镜像构建

拟合程序跑在 Apptainer（Singularity）容器里，只需要建**一次**镜像。

---

## 1. 一步构建

```bash
module load singularity          # 注意：单独一行，不要放进管道
cd <包根目录>                     # 即本 README 的上一级

apptainer build --fakeroot images/jax-fit.sif images/jax-fit.def
```

镜像约 **3.5 GB**，视网速 5–15 分钟。

`jax-fit.def` 用的是 `Bootstrap: docker` + 国内可访问的镜像源
（`docker.m.daocloud.io`），**不需要预先拉 base 镜像**。

### 如果 `apptainer build` 卡在 `Fetching OCI image` 或超时

集群里 `registry-1.docker.io` 不通（DNS 能解析，但连接 i/o timeout）。
`jax-fit.def` 已改用镜像源。若你换到别的环境，把 `From:` 一行改成
你能访问的源即可，实测可用的：

| 源 | 结果 |
|---|---|
| `docker.m.daocloud.io` | ✅ 可用 |
| `quay.io` | ✅ 可用 |
| `docker.nju.edu.cn` | ❌ 403 |
| `registry-1.docker.io` | ❌ 超时 |

### 如果 `--fakeroot` 报 `GLIBC_2.33 not found (required by faked)`

说明 base 镜像太老（老 glibc 与 fakeroot 不兼容）。
本包用的 `python:3.12-slim` 是现代 base，不会有这个问题。
若你换成 ubuntu 20.04 及更早，会遇到这个错。

---

## 2. 验证

```bash
# 版本
apptainer exec images/jax-fit.sif python3 -c \
  "import jax, numpy, scipy; print(jax.__version__, numpy.__version__, scipy.__version__)"
# 期望: 0.11.2 <numpy> <scipy>

# 能看到 GPU（必须在计算节点上跑；登录节点没有 nvidia-smi）
srun --partition=gpu --gres=gpu:1 --pty \
  apptainer exec --nv images/jax-fit.sif python3 -c \
  "import jax; print(jax.devices())"
# 期望: [CudaDevice(id=0)]
```

> ⚠️ **登录节点上没有 `nvidia-smi`**，只有计算节点才有。
> 所以「镜像能不能用 GPU」一定要在 `srun` 里验证。

---

## 3. 为什么锁 `jax[cuda12]==0.11.2`

目标卡是 **Tesla V100（sm_70 / Volta）**。

- NVIDIA 在 **CUDA 13 里移除了 Volta**，对应 JAX 的 `jax[cuda13]`
  要求 `sm_75+`；只有 **CUDA 12** 这条线还带 `sm_70`。
- 真正装 CUDA kernel 的是 **`jax-cuda12-pjrt`**（不是 `jax-cuda12-plugin`
  —— 后者只有几 MB，里面没有 sm_ 内核）。这点容易搞错。

已验证：`jax-cuda12-pjrt 0.11.2` 含 `sm_70`（还含 sm_52…sm_93）。

若你的卡是 A100(sm_80) / H100(sm_90) 等，可以放宽到更新的 CUDA 线，
但**V100 必须留在 CUDA 12**。

---

## 4. 镜像里装了什么

| 包 | 用途 |
|---|---|
| `jax[cuda12]==0.11.2` | 计算主体（含 CUDA kernel） |
| `numpy` | 数组 / IO |
| `scipy` | `optimize.minimize`（牛顿-CG） |
| `matplotlib` | 可选，画图脚本用 |

**没有装 `dlib`**。`fit_object_kk.py` 里对 `dlib.dplex` 做了 `try/except`，
没有时用内置的公式等价实现，**不影响数值结果**。若你的环境里有 `dlib`，
把它挂进容器即可自动优先使用。

`%environment` 里设了 `XLA_PYTHON_CLIENT_PREALLOCATE=false` ——
V100 只有 16 GB，不预占整卡显存，按需分配，OOM 时更好定位。

---

## 5. 镜像内的路径与挂载

程序通过 `--bind` 访问宿主机目录，所以提交脚本里写的是：

```bash
apptainer exec --nv --bind /project/caihao1,/scratch/caihao1 \
    images/jax-fit.sif python3 run/fit_object_kk.py
```

换集群时，把 `--bind` 的两个路径改成你自己的项目目录和 scratch 目录，
并同步修改 `run_fit_*.sbatch` 顶部的 `WORKDIR` / `DATA` / `SIF`。

**不需要**把数据拷进镜像 —— 镜像里只有运行环境。
