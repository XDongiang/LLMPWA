#!/bin/bash
# srun 下的单进程 worker：每个 SLURM task 一个 JAX 进程。
# 依赖外部已导出的 JAX_COORDINATOR_ADDRESS / JAX_NUM_PROCESSES。
module load singularity
export JAX_PROCESS_ID="${SLURM_PROCID}"
exec apptainer exec --nv \
    --bind /project/caihao1,/scratch/caihao1 \
    /scratch/caihao1/hyx/kk_dis_jaxmp/images/jax-fit.sif \
    python3 /scratch/caihao1/hyx/kk_dis_jaxmp/run/gpu_comm_probe.py --mp "${JAX_NUM_PROCESSES}"
