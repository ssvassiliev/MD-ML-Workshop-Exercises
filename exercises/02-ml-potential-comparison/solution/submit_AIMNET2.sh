#!/bin/bash
#SBATCH --job-name=aimnet2
#SBATCH --output=aimnet2-%j.out
#SBATCH --mem-per-cpu=4000
#SBATCH --cpus-per-task=1
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH--time=1:0:0 

CONTAINER_DIR=$SCRATCH/containers

unset REQUESTS_CA_BUNDLE
unset CURL_CA_BUNDLE
module load apptainer

echo "Running AIMNET2"
apptainer exec --nv \
        ${CONTAINER_DIR}/aimnet-0.2.0_omm8.5.2_pt2.12.0_cu12.6_ubuntu24.04.sif \
        python openmm_input_AIMNet2.py
