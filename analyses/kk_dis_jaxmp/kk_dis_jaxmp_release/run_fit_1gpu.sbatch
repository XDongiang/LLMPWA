#!/bin/bash
# ============================================================================
# kk_dis_jaxmp 拟合 —— 单卡全量（不分片）
#
# 用途：
#   * 小规模调试 / 快速验证
#   * 作为 4 卡分片结果的对照基准
#
# 注意：这里申请 --gres=gpu:4 只是为了让 Slurm 给足主机内存配额。
# gpu 分区 DefMemPerGPU=29650（约 29 GB/卡），单卡配额装不下全量数据
# （MC 样本 925000 事例，约 281 MB，编译后展开更大）。
# 实际只用到 GPU 0（用 CUDA_VISIBLE_DEVICES=0 限定）。
#
# 用法：
#   mkdir -p logs
#   sbatch run_fit_1gpu.sbatch
# ============================================================================
#SBATCH --account=caihao1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=5
#SBATCH --time=06:00:00
#SBATCH --job-name=kk_fit1
#SBATCH --output=logs/fit1_%j.log

set -uo pipefail

WORKDIR="${WORKDIR:-/scratch/caihao1/hyx/kk_dis_jaxmp}"
DATA="${DATA:-/scratch/caihao1/sunyc/cgpwa_kk/data}"
SIF="${SIF:-$WORKDIR/images/jax-fit.sif}"
BIND="${BIND:-/project/caihao1,/scratch/caihao1}"

cd "$WORKDIR" || exit 1
mkdir -p logs output

module load singularity

export KK_REAL_DIR="$DATA/real_data"
export KK_MC_DIR="$DATA/mc_int"
export KK_TRUTH_DIR="$DATA/mc_truth"
export KK_WEIGHT_FILE="$DATA/weight/weight_kk.npy"

export KKDIS_OUTDIR="${KKDIS_OUTDIR:-fit_1gpu}"
export KKDIS_MAXITER="${KKDIS_MAXITER:-0}"
export KKDIS_XTOL="${KKDIS_XTOL:-1e-6}"
export KK_DATA_CAP="${KK_DATA_CAP:-0}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false

# 单进程单卡：JAX 直接用默认后端，不做分布式初始化。
export JAX_NUM_PROCESSES=1
export JAX_PROCESS_ID=0
export CUDA_VISIBLE_DEVICES=0

echo "job=$SLURM_JOB_ID host=$(hostname) date=$(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

apptainer exec --nv --bind "$BIND" \
    "$SIF" python3 run/fit_object_kk.py
rc=$?

echo "=== fit rc=$rc date=$(date) ==="
cat "output/$KKDIS_OUTDIR/result_info.json" 2>/dev/null || true
echo
exit $rc
