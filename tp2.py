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
    'zelnik3.arff': 3,
    'target.arff': 6,
    'cuboids.arff': 3,
    # 'aggreg<ation.arff': 5,
    # '3-spiral>.arff': 3,

}
# # Hard files
data_hard_files = {
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
print("Dendrogramme ’single’ donnees initiales")
for test in data_files.items():
    file: str = test[0]
    datanp = arff_to_nparray(file_name=file)
    # plot_dendrogramme(datanp)
    # set di stance_threshold ( 0 ensures we compute the full tree )

    scores = []
    label_dict = {}
    for i, dist in enumerate(np.linspace(0, 5, 5000)):
        tps1 = time.time()
        model = cluster.AgglomerativeClustering(
            distance_threshold=dist,
            n_clusters=None,
            linkage='single'
        )
        model = model.fit(datanp)
        tps2 = time.time()
        labels = model.labels_

        score = compute_cah_score(datanp, labels)

        scores.append(score)
        label_dict[i] = [labels, model.n_clusters_, round((tps2 - tps1) * 1000, 2), dist]

    # Davies-Bouldin -> minimize
    i_max: int = scores.index(max(scores))

    plot_data(datanp, label_dict[i_max][0])

    print(
        f"for: {test[0]}, nb clusters = {label_dict[i_max][1]}, expected = {test[1]}, "
        f"runtime = {label_dict[i_max][2]}ms, distance = {label_dict[i_max][3]}, score = {scores[i_max]}"
    )
