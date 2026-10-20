#!/bin/bash
# 复现性对照：与 3729583 (fit_full_vram) 完全相同的配置，用于判定
# 新版 4 卡在不同节点上给出不同 nll（-3089.007 vs -3089.121）是否为真实现象。
#SBATCH --account=caihao1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=5
#SBATCH --time=06:00:00
#SBATCH --nodelist=g0040
#SBATCH --job-name=kkfitv6
#SBATCH --output=/scratch/caihao1/hyx/kk_dis_jaxmp/logs/kkfitv6_%j.log

module load singularity
cd /scratch/caihao1/hyx/kk_dis_jaxmp

DATA=/scratch/caihao1/sunyc/cgpwa_kk/data
export KK_REAL_DIR=$DATA/real_data
export KK_MC_DIR=$DATA/mc_int
export KK_TRUTH_DIR=$DATA/mc_truth
export KK_WEIGHT_FILE=$DATA/weight/weight_kk.npy
export KK_DATA_CAP=23108
export KKDIS_OUTDIR=fit_full_vram6
export KKDIS_MAXITER=0
export KKDIS_XTOL=1e-6
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export NCCL_ALGO=Ring

echo "job=$SLURM_JOB_ID host=$(hostname) date=$(date)"

HEAD=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
HEAD_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD" hostname -i 2>/dev/null | awk '{print $1}')
export JAX_COORDINATOR_ADDRESS="${HEAD_IP}:$((12000 + SLURM_JOB_ID % 20000))"
export JAX_NUM_PROCESSES="$SLURM_NTASKS"

mkdir -p output logs
srun --kill-on-bad-exit=1 bash -c '
  export JAX_PROCESS_ID=$SLURM_PROCID
  exec apptainer exec --nv --bind /project/caihao1,/scratch/caihao1 \
      images/jax-fit.sif python3 run/fit_object_kk.py
'
rc=$?
echo "=== fit rc=$rc date=$(date) ==="
cat "output/fit_full_vram6/result_info.json" 2>/dev/null
echo
exit $rc
