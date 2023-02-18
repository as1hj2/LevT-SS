# compute initial len by regression

import numpy as np
from sklearn.linear_model import LinearRegression

srclen = []
tgtlen = []

data_dir = '/mnt/beegfs/home/zhou/levt/LevTSS/data'
res_dir = '/mnt/beegfs/home/zhou/levt/LevTSS/analyse'

# use train data
addr = data_dir + '/bpe.train-wmt14.en-de.en'
with open(addr, 'r') as f:
    for line in f:
        srclen.append(len(line.split()))

addr = data_dir + '/bpe.train-wmt14.en-de.de'
with open(addr, 'r') as f:
    for line in f:
        tgtlen.append(len(line.split()))

clf = LinearRegression().fit(np.array(srclen).reshape(-1,1), tgtlen)
print('reg train score = ', clf.score(np.array(srclen).reshape(-1,1), tgtlen), clf.coef_, clf.intercept_)

# predict
addr = data_dir + '/bpe.test-wmt14.en-de.en'
testlen = []
with open(addr, 'r') as f:
    for line in f:
        testlen.append(len(line.split()))
        
res = clf.predict(np.array(testlen).reshape(-1,1))
print('linReg avg prediction = ', np.average(res))

with open(res_dir + '/reglen-wmt14-wmt14', 'w') as f:
    for r in res:
        f.write(str(round(r)) + '\n')