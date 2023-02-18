import re
import numpy as np
import sys
# from matplotlib import pyplot as plt

model = str(sys.argv[3])
test = str(sys.argv[4])
res_name = str(sys.argv[5])

def sent(lines):
    lens = []
    seconds = []
    firsts = []
    deletes = []
    lasts = []
    E2 = re.compile('^E-[0-9]*_2\t')
    E3 = re.compile('^E-[0-9]*_3\t')
    E5 = re.compile('^E-[0-9]*_5\t')
    for idx in range(len(lines)):
        if lines[idx][0] == 'I':
            lens.append(int(lines[idx].split()[1]))
        elif lines[idx][0] == 'H':
            lasts.append(len(lines[idx].split()) - 1)
        elif E2.match(lines[idx]):
            firsts.append(len(lines[idx].split()) - 1)
        elif E3.match(lines[idx]):
            deletes.append(len(lines[idx].split()) - 1)
        elif E5.match(lines[idx]):
            seconds.append(len(lines[idx].split()) - 1)
 
    return lens, np.array([sum(lens)/len(lens),  
                    sum(firsts)/len(firsts), 
                    sum(deletes)/len(deletes), 
                    sum(seconds)/len(seconds), 
                    sum(lasts)/len(lasts)])


def sent5(lines, num=5):
    lens = [[] for _ in range(num)]
    seconds = [[] for _ in range(num)]
    firsts = [[] for _ in range(num)]
    deletes = [[] for _ in range(num)]
    lasts = [[] for _ in range(num)]
    E2 = re.compile('^E-[0-9]-[0-9]*_2\t')
    E3 = re.compile('^E-[0-9]-[0-9]*_3\t')
    E5 = re.compile('^E-[0-9]-[0-9]*_5\t')
    for idx in range(len(lines)):
        if lines[idx][0] == 'I':
            lens[int(lines[idx][2])].append(int(lines[idx].split()[1]))
        elif lines[idx][0] == 'H':
            lasts[int(lines[idx][2])].append(len(lines[idx].split()) - 1)
        elif E2.match(lines[idx]):
            firsts[int(lines[idx][2])].append(len(lines[idx].split()) - 1)
        elif E3.match(lines[idx]):
            deletes[int(lines[idx][2])].append(len(lines[idx].split()) - 1)
        elif E5.match(lines[idx]):
            seconds[int(lines[idx][2])].append(len(lines[idx].split()) - 1)
 
    return lens, np.vstack([np.average(np.array(lens), axis=-1),  
                    np.average(np.array(firsts), axis=-1), 
                    np.array([sum(deletes[i])/len(deletes[i]) for i in range(num)]), 
                    np.array([sum(seconds[i])/len(seconds[i]) for i in range(num)]), 
                    np.average(np.array(lasts), axis=-1)]) # num x features

addr = str(sys.argv[1]) + test + '.out'
with open(addr, 'r') as f:
    lines = f.readlines()
    # lens, result = sent5(lines, 1)
    lens, result = sent(lines)

# plt.figure()
# plt.hist(lens[0], density=True, rwidth=0.5)
# plt.savefig(str(sys.argv[2]) + res_name)

with open(str(sys.argv[2]) + res_name, 'w') as f:
    np.savetxt(f, result, fmt='%.2f', delimiter='\t')