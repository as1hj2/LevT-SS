from matplotlib import pyplot as plt
import numpy as np
import pickle

choice = 'finetuned'
list_name = 'list_top5sum_top1_finetuned'

with open(list_name, "rb") as f:
    lists = pickle.load(f)

    sum_top5_proba, top1_proba, top2_proba, top2_len = lists[0], lists[1], lists[2], lists[3]

    counts, bins = np.histogram(sum_top5, bins=max(sum_top5), density=True)
    fig, ax = plt.subplots(1, 1)
    ax.stairs(counts, bins)
    plt.savefig('train-' + choice + '-iter2-sum-top5-proba')

    counts, bins = np.histogram(top1, bins=max(top1), density=True)
    fig, ax = plt.subplots(1, 1)
    ax.stairs(counts, bins)
    plt.savefig('train-' + choice + '-iter2-top1-proba')

    counts, bins = np.histogram(top2, bins=max(top2), density=True)
    fig, ax = plt.subplots(1, 1)
    ax.stairs(counts, bins)
    plt.savefig('train-' + choice + '-iter2-top2-proba')