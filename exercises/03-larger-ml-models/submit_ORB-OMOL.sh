#!/bin/bash
#SBATCH --job-name=orb-omol
#SBATCH --output=orb-omol-%j.out
#SBATCH --mem-per-cpu=4000
#SBATCH --cpus-per-task=1
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH--time=1:0:0 

CONTAINER_DIR=$SCRATCH/containers

unset REQUESTS_CA_BUNDLE
unset CURL_CA_BUNDLE
module load apptainer

echo "Running ORB-OMOL"
apptainer exec --nv \
        ${CONTAINER_DIR}/orb_models-0.7.0_omm8.6.1_pt2.12.0_cu12.6.2_ubuntu24.04.sif\
        python openmm_input_ORB-OMOL.py
