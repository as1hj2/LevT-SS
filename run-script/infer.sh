#!/bin/bash
###CUDA_LAUNCH_BLOCKING=1

l1=en
l2=de

model_dir=/mnt/beegfs/home/zhou/levt/LevTSS/models
model_choice=$1
model_name=$1

dict_dir=/mnt/beegfs/home/zhou/levt/LevTSS/data/wmt14.en-de.bin

test_dir=/mnt/beegfs/home/zhou/levt/LevTSS/data
test_choice=$2
test_name=bpe.$2.en-de.en

res_dir=/mnt/beegfs/home/zhou/levt/LevTSS/run/$model_choice
res_name=$test_choice.out
mkdir -p $res_dir

# average model weights
if [ ! -e "$model_dir/$model_name/checkpoint.avglast10.pt" ]; then
    python /mnt/beegfs/home/zhou/fairseq/scripts/average_checkpoints.py \
        --inputs $model_dir/$model_name \
        ##--num-epoch-checkpoints 3 \
        --num-update-checkpoints 10 \
        --output $model_dir/$model_name/checkpoint.avglast10.pt
fi 

## fairseq-interactive: raw text, checkpoint.avglast10.pt
fairseq-interactive \
    $dict_dir \
    --path $model_dir/$model_name \
	--task translation_lev \
    --buffer-size 1024 \
    --source-lang $l1 \
    --target-lang $l2 \
    --input $test_dir/$test_name \
	--iter-decode-max-iter 10 \
	--beam 1 \
    --fp16 \
    --print-step \
    --retain-iter-history \
    --max-tokens 4096 \
    --delete-threshold 0 \
    --iter-decode-eos-penalty 0 \
    --use-pld-dp \
    --give-len \
    ## --give-initial \
| tee $res_dir/$res_name

