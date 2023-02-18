for model in wmt14 wmt14-kd
do
    for test_choice in wmt14-srclen wmt14-ratiolen wmt14-reglen wmt14-tgtlen        
    do
        ./analyse_bleu.sh $model $test_choice
    done
done