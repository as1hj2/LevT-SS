###CUDA_LAUNCH_BLOCKING=1 \

model=levt_a100_ss.wmt14.aggr0.9.sample0.5.en-de
checkpoint=checkpoint.avglast10.pt
sample=3

fairseq-validate \
    /mnt/beegfs/home/zhou/levt/LevTSS/data/wmt14.en-de.bin \
    --path /mnt/beegfs/home/zhou/levt/LevTSS/models/$model/$checkpoint \
    --fp16 \
    --ddp-backend legacy_ddp \
    --task translation_lev \
    --criterion nat_loss \
    --noise random_delete \
    --optimizer adam \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 10000 \
    --warmup-init-lr 1e-07 \
    --label-smoothing 0.1 \
    --weight-decay 0.01 \
    --log-format simple \
    --log-interval 200 \
    --fixed-validation-seed 7 \
    --max-tokens 7000 \
    --tensorboard-logdir checkpoints \
    --use-aggravate-prob 0.5 \
    --sample1-prob 0.5 \
    --new-del-input \
#| tee /mnt/beegfs/home/zhou/levt/LevTSS/run-script/useful/generate-$model-$checkpoint-$sample /mnt/beegfs/home/zhou/levt/LevTSS/analyse/validate-infos/generate-$model-$checkpoint-$sample