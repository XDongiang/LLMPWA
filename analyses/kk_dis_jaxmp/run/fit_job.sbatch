#!/bin/bash
# ============================================================================
# kk_dis_jaxmp 拟合作业模板（V100 / gpu 分区）
#
# 用法:
#   sbatch run/fit_job.sbatch              # 单节点 4 卡，每卡一进程
#   NODES=2 sbatch run/fit_job.sbatch      # 2 节点 x 4 卡
#
# 关键约定:
#   * --partition=gpu, --account=caihao1
#   * 每张卡一个进程；进程总数 = 节点数 x 每节点卡数
#   * 用第一个节点的可路由 IP 作 coordinator（hostname 不一定可路由）
#   * 分片要求事例轴能被进程数整除，否则该数组退化为复制（结果仍正确）
# ============================================================================
#SBATCH --account=caihao1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=5
#SBATCH --time=168:00:00
#SBATCH --job-name=kk_fit
#SBATCH --output=/scratch/caihao1/hyx/kk_dis_jaxmp/logs/fit_%j.log

module load singularity
cd /scratch/caihao1/hyx/kk_dis_jaxmp

# ---------------- 可调参数 ----------------
DATA=/scratch/caihao1/sunyc/cgpwa_kk/data
export KK_REAL_DIR=$DATA/real_data
export KK_MC_DIR=$DATA/mc_int
export KK_TRUTH_DIR=$DATA/mc_truth
export KK_WEIGHT_FILE=$DATA/weight/weight_kk.npy
export KKDIS_OUTDIR="${KKDIS_OUTDIR:-fit}"
export KKDIS_MAXITER="${KKDIS_MAXITER:-0}"
# V100 只有 16GB；先不要预占满显存，按需分配更好排查 OOM。
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# -----------------------------------------

echo "job=$SLURM_JOB_ID host=$(hostname) date=$(date)"
echo "nodes=$SLURM_JOB_NODELIST ntasks=$SLURM_NTASKS gres=$SLURM_GPUS_ON_NODE"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# coordinator：取第一个节点的 IP
HEAD=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
HEAD_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD" hostname -i 2>/dev/null | awk '{print $1}')
[ -z "$HEAD_IP" ] && HEAD_IP="$HEAD"
export JAX_COORDINATOR_ADDRESS="${HEAD_IP}:$((12000 + SLURM_JOB_ID % 20000))"
export JAX_NUM_PROCESSES="$SLURM_NTASKS"
echo "coordinator=$JAX_COORDINATOR_ADDRESS processes=$JAX_NUM_PROCESSES"

mkdir -p output logs

# 每个 SLURM task 一个 JAX 进程；--gres=gpu:4 + ntasks-per-node=4 时
# SLURM 会给每个 task 分配一张卡，用 CUDA_VISIBLE_DEVICES 限定。
srun --kill-on-bad-exit=1 bash -c '
  export JAX_PROCESS_ID=$SLURM_PROCID
  exec apptainer exec --nv \
      --bind /project/caihao1,/scratch/caihao1 \
      images/jax-fit.sif \
      python3 run/fit_object_kk.py
'
rc=$?
echo "=== fit rc=$rc date=$(date) ==="
ls -la "output/$KKDIS_OUTDIR" 2>/dev/null
exit $rc
