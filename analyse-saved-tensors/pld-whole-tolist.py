import torch
import numpy as np
# from matplotlib import pyplot as plt
import pickle

# correspond to the name in valide.py
name = './list_pld_0.5.pt'
list_name = "lists_0.5"
list_pld_tensors = torch.load(name)

list_masks = []
list_labels = []
list_preds = []
list_esperance = []

sum_top5_proba = []
top1_proba = []
top2_proba = []
top5_len_E = []
top2_len = []
diff_len = []

list_labels_sent = []
list_preds_sent = []
list_esperance_sent = []

# torch.set_printoptions(profile="full")
# print(list_pld_tensors[0][0].size(), list_pld_tensors[0][0])
# print(list_pld_tensors[0][1].size(), list_pld_tensors[0][1])
# print(list_pld_tensors[0][2].size(), list_pld_tensors[0][2])
# print(list_pld_tensors[0][3].size(), list_pld_tensors[0][3])

for pld_tensors in list_pld_tensors:
    pld_masks, pld_labels, pld_preds, pld_esperance, top5_proba, top5_len = pld_tensors[0], pld_tensors[1], pld_tensors[2], pld_tensors[3], pld_tensors[4], pld_tensors[5]
    
    # torch.set_printoptions(profile="full")

    # len per position
    list_labels += pld_labels[pld_masks].cpu().tolist()
    list_preds += pld_preds[pld_masks].cpu().tolist()
    list_esperance += pld_esperance[pld_masks].cpu().tolist()

    # avg len per sentence
    pld_labels_sent = pld_labels.masked_fill(~pld_masks, 0).float().sum(dim=-1)
    list_labels_sent += pld_labels_sent.cpu().tolist()

    pld_preds_sent = pld_preds.masked_fill(~pld_masks, 0).float().sum(dim=-1)
    list_preds_sent += pld_preds_sent.cpu().tolist()

    pld_esperance_sent = pld_esperance.masked_fill(~pld_masks, 0).float().sum(dim=-1)
    list_esperance_sent += pld_esperance_sent.cpu().tolist()

    # proba
    sum_top5_proba += top5_proba.float().sum(dim=-1)[pld_masks].cpu().tolist()
    top5_len_E += torch.multiply(top5_proba, top5_len.half()).sum(dim=-1)[pld_masks].cpu().tolist()
    
    top1_proba += (top5_proba[pld_masks])[:, 0].cpu().tolist()
    top2_proba += (top5_proba[pld_masks])[:, 1].cpu().tolist()

    top2_len += (top5_len[pld_masks])[:, 1].cpu().tolist()
    diff_len += torch.abs((top5_len[pld_masks])[:, 1] - (top5_len[pld_masks])[:, 0]).cpu().tolist()

with open(list_name + "-proba", "wb") as f:
    pickle.dump([sum_top5_proba, top1_proba, top2_proba, top2_len], f)

with open(list_name + "-len", "wb") as f:
    pickle.dump([list_labels, list_preds, list_esperance, list_labels_sent, list_preds_sent, list_esperance_sent], f)

print(name)

# avg of position
print(sum(list_labels) / len(list_labels))
print(sum(list_preds) / len(list_preds))
print(sum(list_esperance) / len(list_esperance))

# avg of sent
print(sum(list_labels_sent) / len(list_labels_sent))
print(sum(list_preds_sent) / len(list_preds_sent))
print(sum(list_esperance_sent) / len(list_esperance_sent))

# proba
print('avg_sum_top5_proba: ', sum(sum_top5_proba) / len(sum_top5_proba))
print('avg_top1_proba: ', sum(top1_proba) / len(top1_proba))
print('avg_top2_proba: ', sum(top2_proba) / len(top2_proba))

print('top2_len: ', sum(top2_len) / len(top2_len))
print('diff_len: ', sum(diff_len) / len(diff_len))