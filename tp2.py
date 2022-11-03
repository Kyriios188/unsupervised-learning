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
    databrut = arff.loadarff(open(path+file_name, 'r'))
    data = [[x[0], x[1]] for x in databrut[0]]
    return np.array(data)

def plot_data(datanp, labels=None):
    f0 = [f[0] for f in datanp]
    f1 = [f[1] for f in datanp]
    if labels is not None:
        plt.scatter(f0, f1, c=labels, s=8)
    else:
        plt.scatter(f0, f1, s=8)
    plt.title('Donnees initiales')
    plt.show()

def plot_dendrogramme(datanp):
    linked_mat = shc.linkage(datanp, 'single')
    plt.figure(figsize=(12,12))
    shc.dendrogram(
        linked_mat,
        orientation='top',
        distance_sort='descending',
        show_leaf_counts=False
        )
    plt.title(file)
    plt.show()

def compute_silhouette_score(datanp, labels):
    try:
        score: float = metrics.silhouette_score(X=datanp, labels=labels)
        return score
    except ValueError:
        return -1
    


# # Donnees dans datanp
print ("Dendrogramme ’single’ donnees initiales")
for test in data_files.items():
    file: str = test[0]
    datanp = arff_to_nparray(file_name=file)    
    # plot_dendrogramme(datanp)
    # set di stance_threshold ( 0 ensures we compute the full tree )
    
    scores = []
    label_dict = {}
    for i, dist in enumerate(np.linspace(0, 0.3, 100)):
        
        tps1 = time.time()
        model = cluster.AgglomerativeClustering(
            distance_threshold=dist,
            n_clusters=None,
            linkage='single'
        )
        model = model.fit(datanp)
        tps2 = time.time()
        labels = model.labels_

        score = compute_silhouette_score(datanp, labels)
        #print(f"score={score}, i={i}, dist={dist}")
        scores.append(score)
        label_dict[i] = [labels, model.n_clusters, round((tps2 - tps1 ) * 1000, 2)]
        
    i_max: int = scores.index(max(scores))
    
    # Le nombre de cluseters est None ???
    # il faut juste trouver l'élément avec la meilleure silhouette et le plot
    print(label_dict[i_max][1])
    plot_data(datanp, label_dict[i_max][0])
    
    print(
          f"for: {test[0]}\n, nb clusters = {label_dict[i_max][1]}, expected={test[1]}, "
          f"runtime = {label_dict[i_max][2]}, ms" 
    )
    