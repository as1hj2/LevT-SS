sample=1
# levt_a100_ss.wmt14.aggr0.5.sample0.5.en-de levt_a100_ss.wmt14.aggr0.9.sample0.5.en-de levt_a100_ss.wmt14.aggr0.5.sample0.5.new_del.en-de levt_a100_ss.wmt14.aggr0.9.sample0.5.new_del.en-de
# finetune-orig-new-aggr0-sample5 finetune-orig-new-aggr5-sample5 finetune-orig-new-aggr9-sample5
#levt_dev.wmt14.post_del.prob0.en-de
for model_choice in finetune-orig-new-aggr0-sample5 finetune-orig-new-aggr5-sample5 finetune-orig-new-aggr9-sample5
do
    for checkpoint in checkpoint130.pt checkpoint132.pt checkpoint134.pt checkpoint140.pt checkpoint150.pt
    do
        python average_infos.py validate-$model_choice-$checkpoint-$sample
    done
done
