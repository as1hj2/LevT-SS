## binarize the dataset
TEXT=/mnt/beegfs/home/zhou/levt/LevTSS/data/fake-en-fr
fairseq-preprocess --source-lang en --target-lang fr \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --joined-dictionary \
    --destdir /mnt/beegfs/home/zhou/levt/LevTSS/data/fake-en-fr.bin \
    --workers 20
