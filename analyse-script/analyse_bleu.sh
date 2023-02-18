## if generate all top5 length generations

model=levt_dev.wmt14.post_del.prob0.en-de
checkpoint=checkpoint.avglast10.pt
test=train-wmt14-3k

tmp=/mnt/beegfs/home/zhou/levt/LevTSS/analyse/tmp  # save intermediate files
infer_dir=/mnt/beegfs/home/zhou/levt/LevTSS/run/$model/$checkpoint
res_dir=/mnt/beegfs/home/zhou/levt/LevTSS/analyse

rm -f $res_dir/bleu_first_${model}_$test
rm -f $res_dir/bleu_last_${model}_$test

# extract step
for idx in 0 #1 2 3 4
do
    grep  --perl-regex "^E-${idx}-[0-9]*_2\t" $infer_dir/$test.out | cut -f2 > $tmp/bpe_first_${idx}
    grep  --perl-regex "^H-${idx}-[0-9]*" $infer_dir/$test.out | cut -f3 > $tmp/bpe_last_${idx}
done

# de-bpe and de-tok
for idx in 0 #1 2 3 4
do
    perl -pe 's/(@@ )|(@@ ?$)//g' $tmp/bpe_first_${idx} > $tmp/tok_first_${idx}
    perl ~/mosesdecoder/scripts/tokenizer/detokenizer.perl -l de -q < $tmp/tok_first_${idx} > $tmp/detok_first_${idx}
    
    perl -pe 's/(@@ )|(@@ ?$)//g' $tmp/bpe_last_${idx} > $tmp/tok_last_${idx} 
    perl ~/mosesdecoder/scripts/tokenizer/detokenizer.perl -l de -q < $tmp/tok_last_${idx} > $tmp/detok_last_${idx}
done

# compute bleu
for idx in 0 #1 2 3 4
do
    echo ${idx} >> $res_dir/bleu_first_${model}_$test
    sacrebleu /mnt/beegfs/home/zhou/levt/LevTSS/data/detok.test-wmt14.en-de.de -i $tmp/detok_first_${idx} -m bleu -w 4 -f text >> $res_dir/bleu_first_${model}_$test

    echo ${idx} >> $res_dir/bleu_last_${model}_$test
    sacrebleu /mnt/beegfs/home/zhou/levt/LevTSS/data/detok.test-wmt14.en-de.de -i $tmp/detok_last_${idx} -m bleu -w 4 -f text >> $res_dir/bleu_last_${model}_$test
done