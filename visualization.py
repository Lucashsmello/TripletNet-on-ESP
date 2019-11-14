import matplotlib.pyplot as plt
import numpy as np


def plot_embeddings(embeddings, targets, classes_names, xlim=None, ylim=None):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']
    plt.figure(figsize=(10, 10))
    class_set = set(targets)
    for i in class_set:
        i = int(i)
        inds = np.where(targets == i)[0]
        if(len(inds) > 0):
            plt.scatter(embeddings[inds, 0], embeddings[inds, 1],
                        alpha=0.5, color=colors[i], label=classes_names[i])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend()
    plt.show()
