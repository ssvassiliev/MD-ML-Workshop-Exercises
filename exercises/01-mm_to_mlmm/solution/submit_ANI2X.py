#!/bin/bash
#SBATCH --job-name=ani2x
#SBATCH --output=ani2x-%j.out
#SBATCH --mem-per-cpu=4000
#SBATCH --cpus-per-task=1
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH--time=1:0:0 

CONTAINER_DIR=$SCRATCH/containers

unset REQUESTS_CA_BUNDLE
unset CURL_CA_BUNDLE
module load apptainer

echo "Running ANI2X"
apptainer exec --nv \
        ${CONTAINER_DIR}/torchani2.8.2_omm8.5.0_pt2.8.0_cu12.6.2_ubuntu24.04.sif \
        python openmm_input_ANI2X.py
