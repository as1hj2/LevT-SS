import torch
import numpy as np
# from matplotlib import pyplot as plt
import pickle

name = './list_pld_tensors2.pt'
list_name = "lists2"
list_pld_tensors = torch.load(name)

list_labels = []
list_masks = []
list_preds = []
list_esperance = []

list_labels_sent = []
list_preds_sent = []
list_esperance_sent = []

# torch.set_printoptions(profile="full")
# print(list_pld_tensors[0][0].size(), list_pld_tensors[0][0])
# print(list_pld_tensors[0][1].size(), list_pld_tensors[0][1])
# print(list_pld_tensors[0][2].size(), list_pld_tensors[0][2])
# print(list_pld_tensors[0][3].size(), list_pld_tensors[0][3])

for pld_tensors in list_pld_tensors:
    pld_labels, pld_masks, pld_preds, pld_esperance = pld_tensors[0], pld_tensors[1], pld_tensors[2], pld_tensors[3]
    
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

with open(list_name, "wb") as f:
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