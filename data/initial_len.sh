data_dir=/mnt/beegfs/home/zhou/levt/LevTSS/data
test=wmt14

# # src len
# awk '{printf("%s\t%0.f\n", $0, NF)}' $data_dir/bpe.test-$test.en-de.en > $data_dir/bpe.test-$test-srclen.en-de.en

# # ratio len
# awk '{printf("%s\t%0.f\n", $0, NF*1.053)}' $data_dir/bpe.test-$test.en-de.en > $data_dir/bpe.test-$test-ratiolen.en-de.en

# # reg len
# paste $data_dir/bpe.test-$test.en-de.en /mnt/beegfs/home/zhou/levt/LevTSS/analyse/reglen-wmt14-wmt14 > $data_dir/bpe.test-$test-reglen.en-de.en

## tgt len
awk '{print NF}' $data_dir/bpe.test-$test.en-de.de > $data_dir/tmp/tgtlen
paste $data_dir/bpe.test-$test.en-de.en $data_dir/tmp/tgtlen > $data_dir/bpe.test-$test-tgtlen.en-de.en

