import torch
import numpy as np
import pickle

name = './list_pld_top5_orig.pt'
list_name = 'list_top5sum_top1_orig'

list_pld_tensors = torch.load(name)

sum_top5_proba = []
top1_proba = []
top2_proba = []
top5_len_E = []
top2_len = []
diff_len = []
for pld_tensors in list_pld_tensors:
    masks, top5_proba, top5_len = pld_tensors[0], pld_tensors[1], pld_tensors[2]

    sum_top5_proba += top5_proba.float().sum(dim=-1)[masks].cpu().tolist()
    top5_len_E += torch.multiply(top5_proba, top5_len.half()).sum(dim=-1)[masks].cpu().tolist()
    
    top1_proba += (top5_proba[masks])[:, 0].cpu().tolist()
    top2_proba += (top5_proba[masks])[:, 1].cpu().tolist()

    top2_len += (top5_len[masks])[:, 1].cpu().tolist()
    diff_len += torch.abs((top5_len[masks])[:, 1] - (top5_len[masks])[:, 0]).cpu().tolist()

with open(list_name, "wb") as f:
    pickle.dump([sum_top5_proba, top1_proba, top2_proba, top2_len], f)

print(name)

print('avg_sum_top5_proba: ', sum(sum_top5_proba) / len(sum_top5_proba))
print('avg_top1_proba: ', sum(top1_proba) / len(top1_proba))
print('avg_top2_proba: ', sum(top2_proba) / len(top2_proba))

print('top2_len: ', sum(top2_len) / len(top2_len))
print('diff_len: ', sum(diff_len) / len(diff_len))