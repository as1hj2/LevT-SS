#!/bin/bash

l1=en
l2=de
lpair=$l1-$l2

SCRIPTS=$WORK/mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
LID=$WORK/SrcTgtTag/corpus_clean_lid.py
LID_MODEL=$WORK/SrcTgtTag/lid.176.bin
BPEROOT=$WORK/subword-nmt/subword_nmt
BPE_TOKENS=32000

# echo "Concatenating raw data..."
# for l in $l1 $l2; do
#     cat data/raw/{commoncrawl,training/europarl-v7,training/news-commentary-v9}.$l2-$l1.$l > data/raw.wmt14.$lpair.$l
# done
# 
# echo "Cleaning parallel data..."
# perl -p -i -e "s/\r//g" data/raw.wmt14.$lpair.$l1 &
# perl -p -i -e "s/\r//g" data/raw.wmt14.$lpair.$l2 &
# wait
# python corpus-clean-bitext.py -src data/raw.wmt14.$lpair.$l1 -tgt data/raw.wmt14.$lpair.$l2 -out data -tag parallel -max 100 -tok conservative
# perl -p -i -e "s/\r//g" data/parallel.raw.wmt14.$lpair.$l1 &
# perl -p -i -e "s/\r//g" data/parallel.raw.wmt14.$lpair.$l2 &
# wait

# echo "Filtering language ID..."
# python $LID -s data/parallel.raw.wmt14.$lpair.$l1 -t data/parallel.raw.wmt14.$lpair.$l2 -m $LID_MODEL --src_lang $l1 --tgt_lang $l2 -o data/clean.wmt14.$lpair

# rm data/$langPair/train/parallel.*

# echo "Running initial aggressive tokenization..."
# for l in $l1 $l2; do
#     cat data/clean.wmt14.$lpair.$l | \
#         perl $NORM_PUNC $l | \
#         perl $REM_NON_PRINT_CHAR | \
#         perl $TOKENIZER -threads 10 -a -l $l > data/toka.train-wmt14.$lpair.$l &
# done
# wait

# echo "Cleaning parallel by length ratio..."
# perl $CLEAN -ratio 2 data/toka.clean.wmt14.$lpair $l1 $l2 data/toka.train-wmt14.$lpair 1 250

BPE_CODE=data/joint_bpe.$lpair.32k

# echo "Learning BPE model..."
# cat data/toka.train-wmt14.$lpair.{$l1,$l2} > TRAIN
# python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < TRAIN > $BPE_CODE
# rm TRAIN

# for l in $l1 $l2; do
#     f=data/toka.train-wmt14.$lpair.$l
#     echo "apply_bpe.py to $f..."
#     python $BPEROOT/apply_bpe.py -c $BPE_CODE < $f > ${f/toka./bpe.} &
# done
# wait
# 
# echo "Extracting dev and test data..."
# for l in $l1 $l2; do
#     if [ "$l" == "$l1" ]; then
#         t="src"
#     else
#         t="ref"
#     fi
#     grep '<seg id' data/raw/test-full/newstest2014-$l2$l1-$t.$l.sgm | \
#         sed -e 's/<seg id="[0-9]*">\s*//g' | \
#         sed -e 's/\s*<\/seg>\s*//g' | \
#         sed -e "s/\’/\'/g" \
#         > data/newstest2014.$lpair.$l
#     cp data/newstest2014.$lpair.$l data/test-wmt14.$lpair.$l
#     grep '<seg id' data/raw/dev/newstest2013-$t.$l.sgm | \
#         sed -e 's/<seg id="[0-9]*">\s*//g' | \
#         sed -e 's/\s*<\/seg>\s*//g' | \
#         sed -e "s/\’/\'/g" \
#         > data/newstest2013.$lpair.$l
#     cp data/newstest2013.$lpair.$l data/dev-wmt14.$lpair.$l
#     echo ""
# done
# 
# echo "Preprocessing dev and test data..."
# for cls in dev-wmt14 test-wmt14 newstest2014 newstest2013; do
#     for l in $l1 $l2; do
#         cat data/$cls.$lpair.$l | \
#             perl $NORM_PUNC $l | \
#             perl $REM_NON_PRINT_CHAR | \
#             perl $TOKENIZER -threads 10 -a -l $l > data/toka.$cls.$lpair.$l
#         python $BPEROOT/apply_bpe.py -c $BPE_CODE < data/toka.$cls.$lpair.$l > data/bpe.$cls.$lpair.$l
#     done
# done


################################################## Split Training Data for Distillation ##########################################

# echo "Splitting training data for distillation..."
# mkdir -p data/shards
# 
# split --lines 500000 --numeric-suffixes data/bpe.train-wmt14.$lpair.$l1 data/shards/bpe.train-wmt14.$lpair.$l1.

# echo "Concatenating distilled data shards..."
# for i in `seq -w 0 8`; do
#     f=infer_res/transformer.wmt14.$lpair.share_all/bpe.train-wmt14.$lpair.$l1.0$i.out
#     grep ^H $f | cut -f3- > data/shards/bpe.train-wmt14.$lpair.$l2.0$i.kd
# done
# cat data/shards/bpe.train-wmt14.$lpair.$l2.0*.kd > data/bpe.train-wmt14-kd.$lpair.$l2
# cp data/bpe.train-wmt14.$lpair.$l1 data/bpe.train-wmt14-kd.$lpair.$l1
# for l in $l1 $l2; do
#     cp data/bpe.dev-wmt14.$lpair.$l data/bpe.dev-wmt14-kd.$lpair.$l
# done

################################################## Mix Distilled and Original Training Data ##########################################

seed=12345

for i in `seq 1 9`; do
    ratio=0.$i
    python random_select_from_two.py --file1 data/bpe.train-wmt14.$lpair.$l2 --file2 data/bpe.train-wmt14-kd.$lpair.$l2 --ratio $ratio -seed $seed > data/bpe.train-wmt14-kd-$ratio.$lpair.$l2
    cp data/bpe.train-wmt14.$lpair.$l1 data/bpe.train-wmt14-kd-$ratio.$lpair.$l1
    for l in $l1 $l2; do
        cp data/bpe.dev-wmt14.$lpair.$l data/bpe.dev-wmt14-kd-$ratio.$lpair.$l
    done
done
