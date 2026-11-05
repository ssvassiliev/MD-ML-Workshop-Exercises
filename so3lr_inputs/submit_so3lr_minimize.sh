#!/bin/bash
#SBATCH --job-name=so3lr
#SBATCH --output=so3lr-%j.out
#SBATCH --mem-per-cpu=4000
#SBATCH --cpus-per-task=1
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --time=1:0:0

CONTAINER_DIR=$SCRATCH/containers

unset REQUESTS_CA_BUNDLE
unset CURL_CA_BUNDLE
module load apptainer

echo "Running SO3LR minimization"
apptainer exec --nv \
    ${CONTAINER_DIR}/so3lr0.1.0_jaxmd0.2.27_cu12.9.86_ubuntu24.04.sif \
    python jaxmd_input_so3lr_minimize_amber_restart.py
