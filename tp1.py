"""
Created on Tue Oct  4 15:47:49 2022

@author: fmurat
"""

from utils import *

import kmedoids
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import cluster, metrics


# Easy files
data_files = {
    'diamond9.arff': 9,
    'xclara.arff': 3,
    'twodiamonds.arff': 2,
    's-set1.arff': 15,
    'zelnik3.arff': 3,
    'cuboids.arff': 3,
    'aggregation.arff': 5,
    '3-spiral.arff': 3,
    'cassini.arff': 3,
    'target.arff': 6,  # samples are low density but on a line
}
# # Hard files
data_hard_files = {
    'zelnik3.arff': 3,
    'target.arff': 6,
    'rings.arff': 3
}

# On cherche le nombre optimal de clusters avec k-means et k-medoids
t1 = time.time()
for test in data_files.items():
    file: str = test[0]
    datanp = arff_to_nparray(file_name=file)
    scores = []
    for k in range(2, 21):
        # # KMEANS
        # score: float = compute_kmeans_score(k, datanp)
        # KMEDOIDS
        score: float = compute_kmedoids_score(k, datanp)
        scores.append(score)
        # the score at index 0 corresponds to k=1
        # so the score at index 5 corresponds to k=6
    t2 = time.time()
    print(scores)
    print(f"File={file}, EXPECTED={test[1]}, RESULT={scores.index(max(scores))+2}")
    print(f"runtime={round((t2 - t1)*1000,2)}ms")

# Easy files results
# File=xclara.arff, EXPECTED=3, RESULT=3
# runtime=27381.38ms
# File=twodiamonds.arff, EXPECTED=2, RESULT=2
# runtime=40597.15ms
# File=s-set1.arff, EXPECTED=15, RESULT=15
# runtime=88107.49ms

# Difficult files results:
# File=zelnik3.arff, EXPECTED=3, RESULT=7
# runtime=10251.26ms
# File=target.arff, EXPECTED=6, RESULT=7
# runtime=22028.85ms
# File=rings.arff, EXPECTED=3, RESULT=16
# runtime=35578.67ms

# On compare k-means, k-medoids et k-medoids squared en utilisant rand_score
for test in data_files.items():
    file: str = test[0]
    datanp = arff_to_nparray(file_name=file)
    scores = []
    k = test[1]
    # kmeans
    model = cluster.KMeans(n_clusters=k, init='k-means++')
    model.fit(datanp)
    labels_kmeans = model.labels_

    # kmedoids
    distmatrix = euclidean_distances(datanp)
    fp = kmedoids.fasterpam(distmatrix, k)
    labels_kmedoids = fp.labels
    # kmedoids squared
    distmatrix = euclidean_distances(datanp, squared=True)
    fp = kmedoids.fasterpam(distmatrix, k)
    labels_kmedoids_squared = fp.labels
    # compute the score
    compare = metrics.rand_score(labels_kmeans, labels_kmedoids)
    compare2 = metrics.rand_score(labels_kmeans, labels_kmedoids_squared)
    compare3 = metrics.rand_score(labels_kmedoids_squared, labels_kmedoids)
    print(f"{compare} pour {test[0]} (kmeans vs kmedoid)")
    print(f"{compare2} pour {test[0]} (kmeans vs squared kmedoid)")
    print(f"{compare3} pour {test[0]} (kmedoid vs squared kmedoid)")
    # Results :

    # Easy files
    # 0.9995560742469712 pour diamond9.arff
    # 1.0 pour xclara.arff
    # 1.0 pour twodiamonds.arff
    # 0.9998394078815763 pour s-set1.ar

    # Hard files
    # 0.9742942261313662 pour zelnik3.arff
    # 0.8474659280901153 pour target.arff
    # 0.6723663663663664 pour rings.arff

    # Les r??sultats des fichiers o?? les jeu de donn??es sont bien s??par??s donnent
    # des r??sultats tr??s similaires et ont des scores tr??s proches de 1
    # Les jeux de donn??es pas ??vident donnent des scores plus variables, qui descendent
    # jusqu'?? 0.67.

    # Easy files
    # 0.9995560742469712 pour diamond9.arff (kmeans vs kmedoid)
    # 0.9997041235967544 pour diamond9.arff (kmeans vs squared kmedoid)
    # 0.9998519506502167 pour diamond9.arff (kmedoid vs squared kmedoid)
    # 1.0 pour xclara.arff (kmeans vs kmedoid)
    # 0.9995449594309214 pour xclara.arff (kmeans vs squared kmedoid)
    # 0.9995449594309214 pour xclara.arff (kmedoid vs squared kmedoid)
    # 1.0 pour twodiamonds.arff (kmeans vs kmedoid)
    # 1.0 pour twodiamonds.arff (kmeans vs squared kmedoid)
    # 1.0 pour twodiamonds.arff (kmedoid vs squared kmedoid)
    # 0.9998394078815763 pour s-set1.arff (kmeans vs kmedoid)
    # 1.0 pour s-set1.arff (kmeans vs squared kmedoid)
    # 0.9998394078815763 pour s-set1.arff (kmedoid vs squared kmedoid)

    # Tout est sup??rieur ?? 0.9995 donc on ne peut pas tirer de conclusion tr??s intressante.

    # Hard files
    # 0.973670024116896 pour zelnik3.arff (kmeans vs kmedoid)
    # 0.9954887218045113 pour zelnik3.arff (kmeans vs squared kmedoid)
    # 0.9692722371967655 pour zelnik3.arff (kmedoid vs squared kmedoid)
    # 0.9955043655953929 pour target.arff (kmeans vs kmedoid)
    # 0.9886714066167902 pour target.arff (kmeans vs squared kmedoid)
    # 0.9845945991589685 pour target.arff (kmedoid vs squared kmedoid)
    # 0.693957957957958 pour rings.arff (kmeans vs kmedoid)
    # 0.5643343343343343 pour rings.arff (kmeans vs squared kmedoid)
    # 0.7290990990990991 pour rings.arff (kmedoid vs squared kmedoid)
