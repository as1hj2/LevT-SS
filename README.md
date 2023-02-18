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
│   │   ├── criterions
│   │   │   ├── nat_loss.py                        # new options for printing & save tensors
│   │   │   └── ...
│   │   ├── data  
│   │   │   ├── language_pair_dataset.py           # allow giving initial len & initial sentence
│   │   │   └── ...
│   │   ├── dataclass  
│   │   │   ├── configs.py                         # new options for train/valid/inference
│   │   │   └── ...
│   │   ├── models/nat
│   │   │   ├── levenshtein_transformer.py         # modify training process
│   │   │   ├── levenshtein_utils.py               # add new reference computation & length search
│   │   │   └── ...
│   │   ├── tasks 
│   │   │   ├── translation_lev.py                 # modify train/valid step
│   │   │   └── ...
│   │   ├── iterative_refinement_generator.py      # add inference options
│   │   └── ...
│   └── ...                 
└── ...
