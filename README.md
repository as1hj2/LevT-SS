# LevT-SS

## Structure and Modifications

    .
    ├── analyse-saved-tensors                          # scripts for saved_tensors
    ├── analyse-script                                 # scripts for analysing
    ├── analyse                                        # analyse results
    ├── data                                           # preprocess script & a toy example
    ├── run-script                                     # scripts for submitting jobs to slurm: train/valid/test 
    ├── run                                            # inference results
    ├── fairseq-levt-ss                                # forked from fairseq and modified                                 
    │   ├── fairseq_cli
    │   │   ├── preprocess.py 
    │   │   ├── train.py 
    │   │   ├── validate.py
    │   │   ├── generate.py 
    │   │   └── ...
    │   ├── fairseq
    │   │   ├── criterions/nat_loss.py                 # new options for printing & save tensors
    │   │   ├── data/language_pair_dataset.py          # allow giving initial len & initial sentence
    │   │   ├── dataclass/configs.py                   # new options for train/valid/inference
    │   │   ├── tasks/translation_lev.py               # modify train/valid step
    │   │   ├── iterative_refinement_generator.py      # add inference options
    │   │   ├── models/nat
    │   │   │   ├── levenshtein_transformer.py         # modify training process
    │   │   │   ├── levenshtein_utils.py               # add new reference computation & length search
    │   │   │   └── ...
    │   │   └── ...
    │   └── ...                 
    └── ...

## Dataset
WMT14 en-de dataset, with the test set containing 3003 samples.

## New options
### Train/finetune
--new-del-input: use method II  
--post-del: keep the original method unchanged  
--use-aggravate-prob: alpha  
--sample1-prob: beta  

### Validate
--save-tensors: save preds and refs during iter2 on training set, remember to set the file name in validate.py

### Test
--delete-threshold: delta_d  
--iter-decode-eos-penalty: delta_e  
--use-pld-dp: use length search  
--give-len: set iter1 [pld] pred as the given length, remember to add the given length in xxx.en file  
--give-initial: give initial tokens as the initialization, so model will do del-pld-tok-... remember to add the given tokens in xxx.en file  
