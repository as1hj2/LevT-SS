for aggr in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 #0.5 0.8 0.9
do
    for sample in 0.0
    do
        for del in 0.2 0.4 0.6 0.8
        do
            for eos in 1 2 3
            do
                ./analyse_bleu_single.sh $aggr $sample $del $eos  
            done
        done
    done
done