import numpy as np
import torch
import pickle

train_ins_len = [0 for idx in range(256)]

data_dir = '/mnt/beegfs/home/zhou/levt/LevTSS/data/'
addr = data_dir + 'bpe.train-wmt14.en-de.de'

n = 0
with open(addr, 'r') as f:
    for line in f:
        n += 1
        length = min(len(line.split()), 255)
        tokens = torch.arange(0, length)

        score = tokens.float().uniform_()
        score, rank = score.sort()

        cutoff = length * score.new_zeros(1).uniform_()

        keep = rank.numpy()[:int(cutoff)+1]
        keep = np.sort(keep)
        keep1 = np.append(keep, length)
        keep2 = np.insert(keep, 0, -1)
        ref = keep1 - keep2 - 1

        for num in ref:
            if num != 0:
                train_ins_len[num] += 1
        
        if n % 10000 == 0:
            print('at', n)

print(train_ins_len)

with open('/mnt/beegfs/home/zhou/levt/LevTSS/analyse/train-ins-len', 'wb') as f:
        pickle.dump(train_ins_len, f)




