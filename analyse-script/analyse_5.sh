## if generate all top5 length generations
for model in levt_dev.wmt14.post_del.prob0.en-de #finetune-orig-new-aggr0-sample5 finetune-orig-new-aggr5-sample5 finetune-orig-new-aggr9-sample5
do
        for checkpoint in checkpoint.avglast10.pt
        do
                for test_choice in train-wmt14-3k       
                do
                        infer_dir=/mnt/beegfs/home/zhou/levt/LevTSS/run/$model/$checkpoint/
                        res_dir=/mnt/beegfs/home/zhou/levt/LevTSS/analyse/
                        res_name=len-$model-$checkpoint-$test_choice
                        python /mnt/beegfs/home/zhou/levt/LevTSS/analyse-script/analyse_5len.py $infer_dir $res_dir $model $test_choice $res_name
                done
        done
done
