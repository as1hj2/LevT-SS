import numpy as np
import torch
import pickle

dist = [0 for k in range(256)]

data_dir = '/mnt/beegfs/home/zhou/levt/LevTSS/data/'
addr = data_dir + 'bpe.train-wmt14.en-de.de'
addr_kd = data_dir + 'bpe.train-wmt14-kd.en-de.de'

n = 0
with open(addr, 'r') as f, open(addr_kd, 'r') as f_kd:
    for line1, line2 in zip(f, f_kd):
        ref = line1.split()
        kd = line2.split()
        n += 1
        length = 0

        for idx in range(min(len(ref), len(kd))):
            if kd[idx] == ref[idx]:
                length += 1
            else:
                break

        dist[length] += 1

        if n % 10000 == 0:
            print('at', n)

print(dist)

with open('/mnt/beegfs/home/zhou/levt/LevTSS/analyse/kd_match_lens', 'wb') as f:
        pickle.dump(dist, f)

        
        