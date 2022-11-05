#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 08:45:56 2022

@author: fmurat
"""

from utils import *
from sklearn import cluster
import numpy as np
import time


# Easy files
data_files = {
    'zelnik3.arff': 3,
    'cuboids.arff': 3,
    'aggregation.arff': 5,
    '3-spiral.arff': 3,
    'cassini.arff': 3,
    'target.arff': 6,  # samples are low density but on a line
}
# # Hard files
data_hard_files = {
    'compound.arff': 6,
    # 'chainlink.arff': 2
}


# WITH VARYING DISTANCES

# Donnees dans datanp
for test in data_files.items():
    file: str = test[0]
    datanp = arff_to_nparray(file_name=file)
    # plot_dendrogramme(datanp)
    # set distance_threshold ( 0 ensures we compute the full tree )

    scores = []
    label_dict = {}
    for i, dist in enumerate(np.linspace(0, 30, 5000)):
        tps1 = time.time()
        model = cluster.AgglomerativeClustering(
            distance_threshold=dist,
            n_clusters=None,
            linkage='average'
        )
        model = model.fit(datanp)
        tps2 = time.time()
        labels = model.labels_

        # # Used to find the labels_true
        # if model.n_clusters_ == test[1]:
        #     list_chunks = chunks(labels, 100)
        #     print_chunks(list_chunks)
        #     plot_data(datanp, labels)
        #     sys.exit(0)

        score = compute_fm_score(datanp, labels, test[0])

        scores.append(score)
        label_dict[i] = [labels, model.n_clusters_, round((tps2 - tps1) * 1000, 2), dist]

    # Davies-Bouldin -> minimize
    i_max: int = scores.index(max(scores))

    plot_data(datanp, label_dict[i_max][0])

    print(
        f"for: {test[0]}, nb clusters = {label_dict[i_max][1]}, expected = {test[1]}, "
        f"runtime = {label_dict[i_max][2]}ms, distance = {label_dict[i_max][3]}, score = {scores[i_max]}"
    )

# WITH VARYING N_CLUSTERS

# for test in data_files.items():
#     file: str = test[0]
#     datanp = arff_to_nparray(file_name=file)
#     # plot_dendrogramme(datanp)
#     # set distance_threshold ( 0 ensures we compute the full tree )
#
#     scores = []
#     label_dict = {}
#     for i, n_clust in enumerate(range(1, 11)):
#         tps1 = time.time()
#         model = cluster.AgglomerativeClustering(
#             n_clusters=n_clust,
#             linkage='ward'
#         )
#         model = model.fit(datanp)
#         tps2 = time.time()
#         labels = model.labels_
#
#         # # Used to find the labels_true
#         # if model.n_clusters_ == test[1]:
#         #     list_chunks = chunks(labels, 100)
#         #     print_chunks(list_chunks)
#         #     plot_data(datanp, labels)
#         #     sys.exit(0)
#
#         score = compute_fm_score(datanp, labels, test[0])
#
#         scores.append(score)
#         label_dict[i] = [labels, model.n_clusters_, round((tps2 - tps1) * 1000, 2)]
#
#     # Davies-Bouldin -> minimize
#     i_max: int = scores.index(max(scores))
#
#     plot_data(datanp, label_dict[i_max][0])
#
#     print(
#         f"for: {test[0]}, nb clusters = {label_dict[i_max][1]}, "
#         f"runtime = {label_dict[i_max][2]}ms, score = {scores[i_max]}"
#     )
