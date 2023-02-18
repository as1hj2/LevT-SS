import re
import numpy as np
import sys
from matplotlib import pyplot as plt

addr = '/mnt/beegfs/home/zhou/levt/LevTSS/data/bpe.train-wmt14-3k.en-de.de'

lens = []
with open(addr, 'r') as f:
    lines = f.readlines()
    
    for line in lines:
        lens.append(len(line.split()))
        
    print(sum(lens)/len(lens))

    counts, bins = np.histogram(lens, bins=max(lens), density=True)

    fig, ax = plt.subplots(1, 1)
    ax.stairs(counts, bins)
    plt.savefig('train3k-ref-length')