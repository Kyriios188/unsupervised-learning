#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 08:45:56 2022

@author: fmurat
"""

import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
from sklearn import cluster, metrics
import numpy as np
from scipy.io import arff
import time
import sys

# Easy files
data_files = {
    # 'aggregation.arff': 5,  # eps = 3
    # '3-spiral.arff': 3,  # eps = 3
    'zelnik3.arff': 3,  # eps = 0.05
    'cuboids.arff': 3,  # eps = 0.1
    'cassini.arff': 3,  # esp = 0.2

}
# # Hard files
data_hard_files = {
    'target.arff': 6,  # samples are low density but on a line
    'compound.arff': 6,
    'rings.arff': 3,
    'chainlink.arff': 2
}


def arff_to_nparray(file_name: str) -> np.array:
    """Transforms an arff file into a np array of the coordinates"""
    path = './artificial/'
    databrut = arff.loadarff(open(path + file_name, 'r'))
    data = [[x[0], x[1]] for x in databrut[0]]
    return np.array(data)


def plot_data(datanp, labels=None):
    f0 = [f[0] for f in datanp]
    f1 = [f[1] for f in datanp]
    if labels is not None:
        plt.scatter(f0, f1, c=labels, s=8)
    else:
        plt.scatter(f0, f1, s=8)
    plt.show()


def plot_dendrogramme(datanp):
    linked_mat = shc.linkage(datanp, 'single')
    plt.figure(figsize=(12, 12))
    shc.dendrogram(
        linked_mat,
        orientation='top',
        distance_sort='descending',
        show_leaf_counts=False
    )
    plt.title(file)
    plt.show()


def apply_dbsdcan(datanp, eps: float, min_samples: int) -> tuple[float, list]:
    tps1 = time.time()
    model = cluster.DBSCAN(
        eps=eps,
        min_samples=min_samples
    )
    model = model.fit(datanp)
    tps2 = time.time()
    labels = model.labels_
    return round((tps2 - tps1) * 1000, 2), labels


def compute_silhouette_score(datanp, labels) -> float:
    try:
        return metrics.silhouette_score(X=datanp, labels=labels)
    except ValueError:
        return -1


def compute_db_score(datanp, labels) -> float:
    try:
        return metrics.davies_bouldin_score(X=datanp, labels=labels)
    except ValueError:
        return 999  # Davies-Bouldin prefers low values, setting -1 will be counter-productive


def compute_cah_score(datanp, labels) -> float:
    try:
        return metrics.calinski_harabasz_score(X=datanp, labels=labels)
    except ValueError:
        return -1


# # Donnees dans datanp
for test in data_files.items():
    file: str = test[0]
    datanp = arff_to_nparray(file_name=file)
    # plot_dendrogramme(datanp)
    # set distance_threshold ( 0 ensures we compute the full tree )

    scores = []
    label_dict = {}
    for i, epsilon in enumerate(np.linspace(0.01, 0.3, 3000)):
        t, labels = apply_dbsdcan(datanp, eps=epsilon, min_samples=5)

        score = compute_silhouette_score(datanp, labels)

        scores.append(score)
        label_dict[i] = [labels, t, epsilon, score]

    # Davies-Bouldin -> minimize
    i_max: int = scores.index(max(scores))

    plot_data(datanp, label_dict[i_max][0])

    print(
        f"for: {test[0]}, expected = {test[1]}, runtime = {label_dict[i_max][1]}ms, "
        f"epsilon = {label_dict[i_max][2]}, score = {label_dict[i_max][3]}"
    )
