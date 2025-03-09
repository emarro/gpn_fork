#!/bin/bash

# Shell script to set environment variables when running code in this repository.
# Usage:
#     source setup_env.sh

# Activate conda env
# shellcheck source=${HOME}/.bashrc disable=SC1091
#caduceus_env='/share/kuleshov/yzs2/myconda/envs/caduceus_env'
export CONDA_SHELL=/share/apps/anaconda3/2022.10/etc/profile.d/conda.sh
#conda_env='/home/emm392/.conda/envs/AxialCad'
conda_env='/share/kuleshov/emm392/conda/conda_envs/SPINCad'
#conda_env='/home/emm392/.conda/envs/SPINCad'
source "${CONDA_SHELL}"
if [ -z "${CONDA_PREFIX}" ]; then
    conda activate $conda_env
 elif [[ "${CONDA_PREFIX}" != *"/$conda_env" ]]; then
  conda deactivate
  conda activate $conda_env
fi

# Add root directory to PYTHONPATH to enable module imports
export PYTHONPATH="${PWD}"

#export HF_HOME="/share/kuleshov/bioLM_HF_cache" 
export HF_HOME="/share/kuleshov/emm392/hf_cache"
