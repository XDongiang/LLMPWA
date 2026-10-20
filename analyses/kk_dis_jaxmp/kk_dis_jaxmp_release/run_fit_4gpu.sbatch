#!/bin/bash
# ============================================================================
# kk_dis_jaxmp 拟合 —— 单节点 4 卡（每卡一个 JAX 进程，事例轴真分片）
#
# 用法：
#   mkdir -p logs                     # Slurm 不会自动建日志目录，必须先建
#   sbatch run_fit_4gpu.sbatch
#   KKDIS_OUTDIR=myrun sbatch run_fit_4gpu.sbatch
#
# 前置：先按 images/README.md 建好 images/jax-fit.sif。
# ============================================================================
#SBATCH --account=caihao1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=5
#SBATCH --time=06:00:00
#SBATCH --job-name=kk_fit4
#SBATCH --output=logs/fit4_%j.log

# 注意：这里不启用 `set -e`。我们要在 srun 失败时仍走到收尾（打印 rc 和结果），
# 所以必须自己检查退出码，而不是让 shell 提前退出。
set -uo pipefail

# ---------------------------------------------------------------------------
# 按你的环境修改这三个路径（也可用环境变量覆盖）
# ---------------------------------------------------------------------------
WORKDIR="${WORKDIR:-/scratch/caihao1/hyx/kk_dis_jaxmp}"
DATA="${DATA:-/scratch/caihao1/sunyc/cgpwa_kk/data}"
SIF="${SIF:-$WORKDIR/images/jax-fit.sif}"
# 容器要能看到的宿主机目录。数据在 /scratch、项目在 /project 时才需要两个；
# 若你的数据就在 WORKDIR 下，只留一个即可。
BIND="${BIND:-/project/caihao1,/scratch/caihao1}"

cd "$WORKDIR" || exit 1
mkdir -p logs output

module load singularity

# ---------------------------------------------------------------------------
# 数据位置
# ---------------------------------------------------------------------------
export KK_REAL_DIR="$DATA/real_data"      # 实验数据
export KK_MC_DIR="$DATA/mc_int"           # MC 积分样本（归一化 + 似然共用）
export KK_TRUTH_DIR="$DATA/mc_truth"      # MC 真值样本
export KK_WEIGHT_FILE="$DATA/weight/weight_kk.npy"

# ---------------------------------------------------------------------------
# 运行参数
# ---------------------------------------------------------------------------
export KKDIS_OUTDIR="${KKDIS_OUTDIR:-fit}"   # 输出到 output/<这个名字>/
export KKDIS_MAXITER="${KKDIS_MAXITER:-0}"   # 0 = 不限制迭代
export KKDIS_XTOL="${KKDIS_XTOL:-1e-6}"
export KK_DATA_CAP="${KK_DATA_CAP:-0}"       # >0 时把实验事例截断到该数

# V100 只有 16 GB；关掉预占，按需分配，OOM 时更好定位。
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# 多卡想逐位复现就固定 NCCL 归约算法（见 README §7）。
# 默认不设 = 结果仍正确，但每次末位会不同。
# export NCCL_ALGO=Ring

echo "job=$SLURM_JOB_ID host=$(hostname) date=$(date)"
echo "nodes=$SLURM_JOB_NODELIST ntasks=$SLURM_NTASKS"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# ---------------------------------------------------------------------------
# JAX 多进程集合通信的协调地址。
# 必须用第一个节点的「可路由 IP」—— hostname 在部分集群不可路由。
# ---------------------------------------------------------------------------
HEAD=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
HEAD_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD" hostname -i 2>/dev/null | awk '{print $1}')
[ -z "$HEAD_IP" ] && HEAD_IP="$HEAD"
export JAX_COORDINATOR_ADDRESS="${HEAD_IP}:$((12000 + SLURM_JOB_ID % 20000))"
export JAX_NUM_PROCESSES="$SLURM_NTASKS"
echo "coordinator=$JAX_COORDINATOR_ADDRESS processes=$JAX_NUM_PROCESSES"

# 每个 SLURM task 一个 JAX 进程，对应一张卡。
srun --kill-on-bad-exit=1 bash -c '
  export JAX_PROCESS_ID=$SLURM_PROCID
  exec apptainer exec --nv \
      --bind '"$BIND"' \
      "'"$SIF"'" python3 run/fit_object_kk.py
'
rc=$?

echo "=== fit rc=$rc date=$(date) ==="
cat "output/$KKDIS_OUTDIR/result_info.json" 2>/dev/null || true
echo
exit $rc
