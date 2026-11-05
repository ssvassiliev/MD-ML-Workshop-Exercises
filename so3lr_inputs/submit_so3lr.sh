#!/bin/bash
#SBATCH --mem-per-cpu=4000 -c2 --time=1:0:0 -A def-svassili --gpus=h100:1
# --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1

SIF=$SCRATCH/so3lr0.1.0_jaxmd0.2.27_cu12.9.86_ubuntu24.04.sif
SO3LR_INPUT=$SCRATCH/benchmark_polyAla/jaxmd_input_so3lr_hdf5_restart.py

module load apptainer
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw \
  --format=csv -l 5 > gpu_${SLURM_JOB_ID}.csv &
LOGGER_PID=$!
trap 'kill $LOGGER_PID 2>/dev/null' EXIT

echo "Running SO3LR"
apptainer exec --nv $SIF \
       python $SO3LR_INPUT 

