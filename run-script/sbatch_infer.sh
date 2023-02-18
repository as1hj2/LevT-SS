#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=15:00:00
#SBATCH --cpus-per-task=10
#SBATCH --output=logs/%j.infer.out
#SBATCH --error=logs/%j.infer.err
#SBATCH --partition=all
#SBATCH --job-name=infer-de

source ~/miniconda3/bin/activate mt
#levt_dev.wmt14.post_del.prob0.en-de
###levt_a100_ss.wmt14.aggr0.5.sample0.5.new_del.en-de levt_a100_ss.wmt14.aggr0.9.sample0.5.en-de levt_dev.wmt14-kd.post_del.prob0.en-de
###finetune-orig-new-aggr0-sample5/checkpoint132.pt finetune-orig-new-aggr0-sample5/checkpoint134.pt finetune-orig-new-aggr0-sample5/checkpoint140.pt
#checkpoint130.pt checkpoint132.pt checkpoint134.pt checkpoint140.pt checkpoint150.pt

for model_choice in levt_dev.wmt14.post_del.prob0.en-de levt_a100_ss.wmt14.aggr0.5.sample0.5.new_del.en-de levt_a100_ss.wmt14.aggr0.9.sample0.5.new_del.en-de 
do
    for checkpoint in checkpoint.avglast10.pt
    do
        for test_choice in test-wmt14-tgtlen #dev-wmt14 ##wmt14-ratiolen wmt14-reglen wmt14-tgtlen
        do
            ./infer.sh $model_choice/$checkpoint $test_choice
        done
    done
done