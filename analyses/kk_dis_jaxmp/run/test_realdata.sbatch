#!/bin/bash
# 真实数据 + 4 卡：验证能否加载并跑通一步（关注显存是否够）。
#SBATCH --account=caihao1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=5
#SBATCH --time=01:00:00
#SBATCH --job-name=kkreal
#SBATCH --output=/scratch/caihao1/hyx/kk_dis_jaxmp/logs/kkreal_%j.log

module load singularity
cd /scratch/caihao1/hyx/kk_dis_jaxmp

DATA=/scratch/caihao1/sunyc/cgpwa_kk/data
export KK_REAL_DIR=$DATA/real_data
export KK_MC_DIR=$DATA/mc_int
export KK_TRUTH_DIR=$DATA/mc_truth
export KK_WEIGHT_FILE=$DATA/weight/weight_kk.npy
export KKDIS_OUTDIR=realprobe
export KKDIS_MAXITER=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "job=$SLURM_JOB_ID host=$(hostname) date=$(date)"
HEAD=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
HEAD_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD" hostname -i 2>/dev/null | awk '{print $1}')
export JAX_COORDINATOR_ADDRESS="${HEAD_IP}:$((12000 + SLURM_JOB_ID % 20000))"
export JAX_NUM_PROCESSES="$SLURM_NTASKS"
echo "coordinator=$JAX_COORDINATOR_ADDRESS processes=$JAX_NUM_PROCESSES"

mkdir -p output logs

srun --kill-on-bad-exit=1 bash -c '
  export JAX_PROCESS_ID=$SLURM_PROCID
  exec apptainer exec --nv \
      --bind /project/caihao1,/scratch/caihao1 \
      images/jax-fit.sif \
      python3 run/fit_object_kk.py
' &
FITPID=$!

# 后台采样显存峰值
( for i in $(seq 1 120); do
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null | tr "\n" " "; echo
    sleep 10
  done ) > logs/kkreal_mem_${SLURM_JOB_ID}.log 2>&1 &
MONPID=$!

wait $FITPID; rc=$?
kill $MONPID 2>/dev/null
echo "=== fit rc=$rc date=$(date) ==="
echo "--- peak memory samples ---"
sort -t, -k2 -rn logs/kkreal_mem_${SLURM_JOB_ID}.log 2>/dev/null | head -5
echo "--- output ---"
ls -la output/realprobe 2>/dev/null
