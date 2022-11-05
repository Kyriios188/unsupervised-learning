#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 08:45:56 2022

@author: fmurat
"""

from utils import *

# Easy files
data_files = {
    'zelnik3.arff': 3,
    'cuboids.arff': 3,
    'aggregation.arff': 5,
    '3-spiral.arff': 3,
    'cassini.arff': 3,
    'target.arff': 6,  # samples are low density but on a line
}
# Hard files
data_hard_files = {
    'diamond9.arff': 9,
    'xclara.arff': 3,
    'twodiamonds.arff': 2,
    's-set1.arff': 15,
}

# This was determined to be the best value for min_samples
# obtained by making min_samples vary from 1 to 200 with epsilon = 1.0
hdbscan_min_samples = {
    'diamond9.arff': 8,
    'xclara.arff': 30,
    'twodiamonds.arff': 8,
    's-set1.arff': 16,
}

# # Donnees dans datanp
for test in data_hard_files.items():
    file: str = test[0]
    datanp = arff_to_nparray(file_name=file)
    # plot_dendrogramme(datanp)
    # set distance_threshold ( 0 ensures we compute the full tree )

    scores = []
    label_dict = {}
    for i, cluster_size in enumerate(range(2, 1000)):
        t, labels = apply_hdbsdcan(datanp, min_cluster_size=cluster_size, min_samples=hdbscan_min_samples[test[0]])

        score = compute_fm_score(datanp, labels, test[0])

        scores.append(score)
        label_dict[i] = [labels, t, cluster_size, score]

    # Davies-Bouldin -> minimize
    i_max: int = scores.index(max(scores))

    plot_data(datanp, label_dict[i_max][0])

    print(
        f"for: {test[0]}, expected = {test[1]}, runtime = {label_dict[i_max][1]}ms, "
        f"min_cluster_size = {label_dict[i_max][2]}, score = {label_dict[i_max][3]}"
    )
