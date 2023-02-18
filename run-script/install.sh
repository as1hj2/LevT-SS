#!/bin/bash
#SBATCH --gres=gpu:1
###SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=10
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=all
#SBATCH --job-name=train-levt
#SBATCH --partition=all
#SBATCH --constraint=a6000
####SBATCH --nodelist=n4
#SBATCH --chdir=/mnt/beegfs/home/zhou/levt/LevTSS/fairseq-levt-my

source ~/miniconda3/bin/activate mt
export CUDA_HOME=/mnt/beegfs/home/zhou/miniconda3/envs/mt
python setup.py build_ext --inplace
