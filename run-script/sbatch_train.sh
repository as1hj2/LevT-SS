#!/bin/bash
#SBATCH --gres=gpu:2
###SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=10
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=all
#SBATCH --job-name=train-levt
#SBATCH --partition=all
#SBATCH --constraint=a6000
####SBATCH --nodelist=n4
###SBATCH --chdir=/mnt/beegfs/home/zhou/levt/LevTSS/fairseq-levt-orig

source ~/miniconda3/bin/activate mt

# module load /usr/local/cuda/bin

# ls -ld /usr/local/cuda/lib64/*
# ls -ld /usr/local/cuda/targets/x86_64-linux/lib/*

#nvcc --version

#pip install torch==1.12.1+cu102 torchvision==0.13.1+cu102 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu102
#cuda_version=10.2
# export CUDA_HOME=/usr/local/cuda-${cuda_version}
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
# export PATH=$PATH:/usr/local/cuda/bin

export CUDA_HOME=/mnt/beegfs/home/zhou/miniconda3/envs/mt

# #nvcc --version
# python setup.py build_ext --inplace


./finetune.sh