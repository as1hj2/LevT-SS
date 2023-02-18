import re
import numpy as np
import sys
# from matplotlib import pyplot as plt

file_dir = "/mnt/beegfs/home/zhou/levt/LevTSS/analyse/validate-infos/"
file_name = str(sys.argv[1])

def avg_infos(lines):
    del_accs, del_recalls, del_precs, del_nums, pld_accs, pld_l1s, pld_l1s_positif, pld_l1s_negatif, pld_l1s_sent, pld_nums, tok_accs, tok_nums = [], [], [], [], [], [], [], [], [], [], [], []
    
    batches = []
    batch = re.compile('^batch')

    del_acc = re.compile('^del_acc')
    del_recall = re.compile('^del_recall')
    del_prec = re.compile('^del_prec')
    del_num = re.compile('^del_num')

    pld_acc = re.compile('^pld_acc')
    pld_l1  = re.compile('^pld_l1:')
    pld_l1_positif  = re.compile('^pld_l1_positif')
    pld_l1_negatif  = re.compile('^pld_l1_negatif')
    pld_l1_sent  = re.compile('^pld_l1_sent')
    pld_num = re.compile('^pld_num')

    tok_acc  = re.compile('^tok_acc')
    tok_num = re.compile('^tok_num')
    
    for idx in range(len(lines)):
        if batch.match(lines[idx]):
            batches.append(int(lines[idx].split()[1]))

            pld_accs.append(float(lines[idx+1].split()[1]) * batches[-1])
            pld_l1s.append(float(lines[idx+2].split()[1]) * batches[-1])
            pld_l1s_positif.append(float(lines[idx+3].split()[1]) * batches[-1])
            pld_l1s_negatif.append(float(lines[idx+4].split()[1]) * batches[-1])
            pld_l1s_sent.append(float(lines[idx+5].split()[1]) * batches[-1])
            pld_nums.append(float(lines[idx+6].split()[1]) * batches[-1])

            tok_accs.append(float(lines[idx+7].split()[1]) * batches[-1])
            tok_nums.append(float(lines[idx+8].split()[1]) * batches[-1])

            del_accs.append(float(lines[idx+9].split()[1]) * batches[-1])
            del_recalls.append(float(lines[idx+10].split()[1]) * batches[-1])
            del_precs.append(float(lines[idx+11].split()[1]) * batches[-1])
            del_nums.append(float(lines[idx+12].split()[1]) * batches[-1])

            
        # elif del_acc.match(lines[idx]):
        #     del_accs.append(float(lines[idx].split()[1]) * batches[-1])
        # elif del_recall.match(lines[idx]):
        #     del_recalls.append(float(lines[idx].split()[1]) * batches[-1])
        # elif del_prec.match(lines[idx]):
        #     del_precs.append(float(lines[idx].split()[1]) * batches[-1])
        # elif del_num.match(lines[idx]):
        #     del_nums.append(float(lines[idx].split()[1]) * batches[-1])

        # elif pld_acc.match(lines[idx]):
        #     pld_accs.append(float(lines[idx].split()[1]) * batches[-1])
        # elif pld_l1.match(lines[idx]):
        #     pld_l1s.append(float(lines[idx].split()[1]) * batches[-1])
        # elif pld_l1_positif.match(lines[idx]):
        #     pld_l1s_positif.append(float(lines[idx].split()[1]) * batches[-1])
        # elif pld_l1_negatif.match(lines[idx]):
        #     pld_l1s_negatif.append(float(lines[idx].split()[1]) * batches[-1])
        # elif pld_l1_sent.match(lines[idx]):
        #     pld_l1s_sent.append(float(lines[idx].split()[1]) * batches[-1])
        # elif pld_num.match(lines[idx]):
        #     pld_nums.append(float(lines[idx].split()[1]) * batches[-1])

        # elif tok_acc.match(lines[idx]):
        #     tok_accs.append(float(lines[idx].split()[1]) * batches[-1])
        # elif tok_num.match(lines[idx]):
        #     tok_nums.append(float(lines[idx].split()[1]) * batches[-1])

    return [
        sum(del_accs)/sum(batches),
        sum(del_recalls)/sum(batches),
        sum(del_precs)/sum(batches),
        sum(del_nums)/sum(batches),

        sum(pld_accs)/sum(batches),
        sum(pld_l1s)/sum(batches),
        sum(pld_l1s_positif)/sum(batches),
        sum(pld_l1s_negatif)/sum(batches),
        sum(pld_l1s_sent)/sum(batches),
        sum(pld_nums)/sum(batches),

        sum(tok_accs)/sum(batches),
        sum(tok_nums)/sum(batches)
    ]
 
addr = file_dir + file_name
with open(addr, 'r') as f:
    lines = f.readlines()
    result = avg_infos(lines)

with open(addr, 'a') as f:
    f.write('\n')
    f.write('averaged results:\n')
    np.savetxt(f, np.array(result), fmt='%.2f', delimiter='\t')