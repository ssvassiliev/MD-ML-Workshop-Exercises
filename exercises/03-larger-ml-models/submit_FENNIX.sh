#!/bin/bash
#SBATCH --job-name=fennix
#SBATCH --output=fennix-%j.out
#SBATCH --mem-per-cpu=4000
#SBATCH --cpus-per-task=1
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH--time=1:0:0 

CONTAINER_DIR=$SCRATCH/containers

unset REQUESTS_CA_BUNDLE
unset CURL_CA_BUNDLE
module load apptainer

echo "Running FENNIX"
apptainer exec --nv \
        ${CONTAINER_DIR}/fennol-2026.6.29_omm8.5.2_pt2.12.0_cu12.6.2_ubuntu24.04.sif\
        python openmm_input_FENNIX.py
