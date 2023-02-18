import re
import numpy as np
import sys
from matplotlib import pyplot as plt
import pickle

list_name = "./lists_0.5-len"
choice = '5'
folder='./0.5/'

with open(list_name, "rb") as f:
    lists = pickle.load(f)
    list_labels, list_preds, list_esperance, list_labels_avg, list_preds_avg, list_esperance_avg = lists[0], lists[1], lists[2], lists[3],  lists[4], lists[5]
    
    # histogram of position
    counts, bins = np.histogram(list_labels, bins=max(list_labels), density=True)
    fig, ax = plt.subplots(1, 1)
    ax.stairs(counts[:20], bins[:21])
    plt.savefig(folder + 'train-' + choice + '-iter2-ref-length')

    counts, bins = np.histogram(list_preds, bins=max(list_preds), density=True)
    fig, ax = plt.subplots(1, 1)
    ax.stairs(counts[:20], bins[:21])
    plt.savefig(folder + 'train-' + choice + '-iter2-pred-length')

    counts, bins = np.histogram(list_esperance, bins=int(max(list_esperance)), density=True)
    fig, ax = plt.subplots(1, 1)
    ax.stairs(counts[:20], bins[:21])
    plt.savefig(folder + 'train-' + choice + '-iter2-esperance-length')

    # # histogram of sentence
    # counts, bins = np.histogram(list_labels_avg, bins=int(max(list_labels_avg)), density=True)
    # fig, ax = plt.subplots(1, 1)
    # ax.stairs(counts[:150], bins[:151])
    # plt.savefig(folder + 'train-' + choice + '-iter2-ref-sentence-length')

    # counts, bins = np.histogram(list_preds_avg, bins=int(max(list_preds_avg)), density=True)
    # fig, ax = plt.subplots(1, 1)
    # ax.stairs(counts[:100], bins[:101])
    # plt.savefig(folder + 'train-' + choice + '-iter2-pred-sentence-length')

    # counts, bins = np.histogram(list_esperance_avg, bins=int(max(list_esperance_avg)), density=True)
    # fig, ax = plt.subplots(1, 1)
    # ax.stairs(counts[:200], bins[:201])
    # plt.savefig(folder + 'train-' + choice + '-iter2-esperance-sentence-length')