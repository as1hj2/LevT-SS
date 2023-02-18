##model=levt_dev.wmt14.post_del.prob0.en-de levt_a100_ss.wmt14.aggr0.5.sample0.5.new_del.en-de levt_a100_ss.wmt14.aggr0.9.sample0.5.new_del.en-de
##model=levt_a100_ss.wmt14.aggr$1.sample$2.new_del.en-de
##test=del-$3-eos-$4-test-wmt14
checkpoint=checkpoint.avglast10.pt
model=levt_a100_ss.wmt14.aggr0.5.sample0.5.new_del.en-de
test=pld-dp-del0.2-test-wmt14-tgtlen

tmp=/mnt/beegfs/home/zhou/levt/LevTSS/analyse/tmp  # save intermediate files
infer_dir=/mnt/beegfs/home/zhou/levt/LevTSS/run/$model/$checkpoint
res_dir=/mnt/beegfs/home/zhou/levt/LevTSS/analyse
 
# rm -f $res_dir/bleu_first_$model-$checkpoint_$test
# rm -f $res_dir/bleu_second_$model-$checkpoint_$test
# rm -f $res_dir/bleu_last_$model-$checkpoint_$test

# extract step
grep  --perl-regex "^E-[0-9]*_2\t" $infer_dir/$test.out | cut -f2 > $tmp/bpe_first
grep  --perl-regex "^E-[0-9]*_5\t" $infer_dir/$test.out | cut -f2 > $tmp/bpe_second
grep  --perl-regex "^H-[0-9]*" $infer_dir/$test.out | cut -f3 > $tmp/bpe_last


# de-bpe and de-tok
perl -pe 's/(@@ )|(@@ ?$)//g' $tmp/bpe_first > $tmp/tok_first
perl ~/mosesdecoder/scripts/tokenizer/detokenizer.perl -l de -q < $tmp/tok_first > $tmp/detok_first

perl -pe 's/(@@ )|(@@ ?$)//g' $tmp/bpe_second > $tmp/tok_second
perl ~/mosesdecoder/scripts/tokenizer/detokenizer.perl -l de -q < $tmp/tok_second > $tmp/detok_second

perl -pe 's/(@@ )|(@@ ?$)//g' $tmp/bpe_last > $tmp/tok_last
perl ~/mosesdecoder/scripts/tokenizer/detokenizer.perl -l de -q < $tmp/tok_last > $tmp/detok_last


# compute bleu
#echo del-$3-eos-$4 >> $res_dir/bleu_last_$model-$checkpoint-del-eos-test-wmt14
echo $model-$checkpoint >> $res_dir/bleu-$test
echo first >> $res_dir/bleu-$test
cat $tmp/detok_first | sacrebleu /mnt/beegfs/home/zhou/levt/LevTSS/data/newstest2014.en-de.de -w 4 -f text >> $res_dir/bleu-$test
echo second >> $res_dir/bleu-$test
cat $tmp/detok_second | sacrebleu /mnt/beegfs/home/zhou/levt/LevTSS/data/newstest2014.en-de.de -w 4 -f text >> $res_dir/bleu-$test
echo last >> $res_dir/bleu-$test
cat $tmp/detok_last | sacrebleu /mnt/beegfs/home/zhou/levt/LevTSS/data/newstest2014.en-de.de -w 4 -f text >> $res_dir/bleu-$test
# sacrebleu /mnt/beegfs/home/zhou/levt/LevTSS/analyse-script/wmt14/en-de.de -i $tmp/detok_second -m bleu -w 4 -f text >> $res_dir/bleu_second_${model}_$test
# sacrebleu /mnt/beegfs/home/zhou/levt/LevTSS/analyse-script/wmt14/en-de.de -i $tmp/detok_last -m bleu -w 4 -f text >> $res_dir/bleu_last_${model}_$test
