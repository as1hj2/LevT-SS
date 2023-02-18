#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=10:00:00
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
#levt_dev.wmt14.post_del.prob0.en-de
#finetune-orig-new-aggr0-sample5 finetune-orig-new-aggr5-sample5 finetune-orig-new-aggr9-sample5
#levt_a100_ss.wmt14.aggr0.5.sample0.5.en-de levt_a100_ss.wmt14.aggr0.9.sample0.5.en-de levt_a100_ss.wmt14.aggr0.5.sample0.5.new_del.en-de levt_a100_ss.wmt14.aggr0.9.sample0.5.new_del.en-de # finetune-orig-new-aggr0-sample5 finetune-orig-new-aggr5-sample5 finetune-orig-new-aggr9-sample5 #levt_dev.wmt14.post_del.prob0.en-de #finetune-orig-new-aggr5-sample5 ##levt_dev.wmt14.post_del.prob0.en-de ##levt_a100_ss.wmt14.aggr0.5.sample0.5.new_del.en-de
# for model_choice in finetune-orig-new-aggr0-sample5 finetune-orig-new-aggr5-sample5 finetune-orig-new-aggr9-sample5
# do
#     for checkpoint in checkpoint.avglast3epoch.pt
#     do
#         ./validate.sh $model_choice $checkpoint
#     done
# done

./validate.sh