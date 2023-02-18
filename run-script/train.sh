# CUDA_LAUNCH_BLOCKING=1 \
fairseq-train \
    /mnt/beegfs/home/zhou/levt/LevTSS/data/wmt14.en-de.bin \
    --fp16 \
    --tensorboard-logdir /mnt/beegfs/home/zhou/levt/LevTSS/models/finetune-orig-new-aggr0-del1*0.5 \
    --save-dir /mnt/beegfs/home/zhou/levt/LevTSS/models/finetune-orig-new-aggr0-del1*0.5 \
    --restore-file /mnt/beegfs/home/zhou/levt/LevTSS/models/levt_dev.wmt14.post_del.prob0.en-de/checkpoint_last.pt \
    --ddp-backend legacy_ddp \
    --task translation_lev \
    --criterion nat_loss \
    --arch levenshtein_transformer \
    --noise random_delete \
    --share-all-embeddings \
    --optimizer adam \
    --lr 0.0005 \
    --lr-scheduler inverse_sqrt \
    --stop-min-lr 1e-09 \
    --warmup-updates 10000 \
    --warmup-init-lr 1e-07 \
    --label-smoothing 0.1 \
    --dropout 0.3 \
    --weight-decay 0.01 \
    --decoder-learned-pos \
    --encoder-learned-pos \
    --apply-bert-init \
    --log-format simple \
    --log-interval 200 \
    --fixed-validation-seed 7 \
    --max-tokens 8000 \
    --update-freq 4 \
    --save-interval-updates 10000 \
    --max-update 350000 \
    --post-del \
    --use-aggravate-prob 0 \
    --sample1-prob 0.5 \
    --new-del-input
    ###--max-update 350000 \
    ###--restore-file /mnt/beegfs/home/zhou/levt/LevTSS/models/levt_dev.wmt14.post_del.prob0.en-de/checkpoint_last.pt \