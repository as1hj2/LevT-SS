#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=0:10:00
#SBATCH --cpus-per-task=10
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=all
#SBATCH --job-name=validate-levt
#SBATCH --partition=all
#SBATCH --constraint=a6000
####SBATCH --nodelist=n4
###SBATCH --chdir=/mnt/beegfs/home/zhou/levt/LevTSS/fairseq-levt-orig

source ~/miniconda3/bin/activate mt

export CUDA_HOME=/mnt/beegfs/home/zhou/miniconda3/envs/mt

python pld-whole-tolist.py