# Ce script a pour but de trouver les paramètres optimaux des
# méthodes pour pouvoir faire les comparaisons avec leurs "meilleurs" paramètres.
import numpy as np

from utils import *

# y1 est impossible à utiliser, beaucoup trop volumineux.
mystery_data_files = {
    # 'x1.txt': 15,
    # 'x2.txt': 15,
    # 'x3.txt': 15,
    # 'x4.txt': 15,
    'y1.txt': 2,
    # 'zz1.txt': 8,
    # 'zz2.txt': 5,
}


def print_optimal_km_params():
    # Cette boucle doit identifier le nombre de clusters optimal pour obtenir le
    # meilleur coefficient de silhouette avec kmeans et kmedoids
    for test in mystery_data_files.items():
        file: str = test[0]
        datanp = txt_to_nparray(file_name=file)
        kmeans_scores = []
        kmedoids_scores = []
        for k in range(2, 21):
            # # KMEANS
            # score: float = compute_kmeans_score(k, datanp)
            # KMEDOIDS
            kmeans_score: float = compute_kmeans_score(k, datanp)
            kmeans_scores.append(kmeans_score)

            try:
                kmedoids_score: float = compute_kmedoids_score(k, datanp)
                kmedoids_scores.append(kmedoids_score)
                # the score at index 0 corresponds to k=1
                # so the score at index 5 corresponds to k=6
            except Exception:  # Memory Error, protected exception class
                kmedoids_scores.append(-1)


        kmeans_i_max: int = kmeans_scores.index(max(kmeans_scores))
        kmeans_best_n_cluster = kmeans_i_max + 2
        kmedoids_i_max: int = kmedoids_scores.index(max(kmedoids_scores))
        kmedoids_best_n_cluster = kmedoids_i_max + 2

        print(
            f"{test[0]} : kmeans = {kmeans_best_n_cluster}, "
            f"kmedoids = {kmedoids_best_n_cluster}"
        )


# Uncomment to run kmeans/kmedoids tests
# print_optimal_km_params()

# x1.txt : kmeans = 15, kmedoids = 15
# x2.txt : kmeans = 15, kmedoids = 15
# x3.txt : kmeans = 15, kmedoids = 15
# x4.txt : kmeans = 15, kmedoids = 15
# y1.txt : kmeans = 8, kmedoids = 2
# zz1.txt : kmeans = 8, kmedoids = 8
# zz2.txt : kmeans = 5, kmedoids = 5





def print_optimal_hdbscan_params():
    for test in mystery_data_files.items():
        file: str = test[0]
        datanp = txt_to_nparray(file_name=file)

        scores = []
        index = []
        label_dict = {}
        for j, min_samples in enumerate(range(1, 101, 10)):
            for i, cluster_size in enumerate(range(2, 100, 5)):
                t, labels = apply_hdbsdcan(datanp, min_cluster_size=cluster_size, min_samples=min_samples)

                score = compute_cah_score(datanp, labels)

                scores.append(score)
                index.append(str(i) + str(j))
                label_dict[str(i) + str(j)] = [labels, t, cluster_size, score, min_samples]
            print(f"min_samples : {j}")

        i_max: int = scores.index(max(scores))
        label_dict_index = index[i_max]

        plot_data(datanp, label_dict[label_dict_index][0])

        print(
            f"for: {test[0]}, expected = {test[1]}, runtime = {label_dict[label_dict_index][1]}ms, "
            f"min_cluster_size = {label_dict[label_dict_index][2]}, score = {label_dict[label_dict_index][3]}, "
            f"min_samples = {label_dict[label_dict_index][4]}"
        )


# print_optimal_hdbscan_params()
# for: x1.txt, expected = 15, runtime = 51.01ms, min_cluster_size = 37, score = 0.6959871132578086, min_samples = 1
# for: x2.txt, expected = 15, runtime = 62.0ms, min_cluster_size = 22, score = 0.5265747986645998, min_samples = 4
# for: x3.txt, expected = 15, runtime = 130.03ms, min_cluster_size = 2, score = 0.292951506941413, min_samples = 1
# for: x4.txt, expected = 15, runtime = 126.97ms, min_cluster_size = 2, score = 0.2787206162872168, min_samples = 1
# for: y1.txt, expected = 2, runtime = 1717.0ms, min_cluster_size = 7, score = 18687.767359899564, min_samples = 61
# for: zz1.txt, expected = 8, runtime = 84.99ms, min_cluster_size = 7, score = 0.8434769999617807, min_samples = 3
# for: zz2.txt, expected = 5, runtime = 13.0ms, min_cluster_size = 87, score = 0.7693001086477487, min_samples = 58

def print_optimal_dbsdcan_params():
    for test in mystery_data_files.items():
        file: str = test[0]
        datanp = txt_to_nparray(file_name=file)

        scores = []
        index = []
        label_dict = {}
        for j, min_samples in enumerate(range(1, 2)):
            for i, epsilon in enumerate(np.linspace(0.1, 2, 10)):
                t, labels = apply_dbsdcan(datanp, eps=epsilon, min_samples=min_samples)

                score = compute_cah_score(datanp, labels)

                scores.append(score)
                index.append(str(i) + str(j))
                label_dict[str(i) + str(j)] = [labels, t, epsilon, score, min_samples]

        i_max: int = scores.index(max(scores))
        label_dict_index = index[i_max]

        plot_data(datanp, label_dict[label_dict_index][0])

        print(
            f"for: {test[0]}, expected = {test[1]}, runtime = {label_dict[label_dict_index][1]}ms, "
            f"epsilon = {label_dict[label_dict_index][2]}, score = {label_dict[label_dict_index][3]}, "
            f"min_samples = {label_dict[label_dict_index][4]}"
        )


# print_optimal_dbsdcan_params()


# for: x1.txt, expected = 15, runtime = 23.0ms, epsilon = 23.767676767676768, score = 416633831.28228086, min_samples = 1
# for: x2.txt, expected = 15, runtime = 23.0ms, epsilon = 23.272727272727273, score = 388141857.9395273, min_samples = 1
# for: x3.txt, expected = 15, runtime = 23.0ms, epsilon = 38.121212121212125, score = 107715333.74142061, min_samples = 1
# for: x4.txt, expected = 15, runtime = 24.0ms, epsilon = 17.333333333333336, score = 391222072.2337754, min_samples = 1
# for: y1.txt, expected = 2, runtime = 612.03ms, epsilon = 0.1, score = 285585620.97581446, min_samples = 1
# for: zz1.txt, expected = 8, runtime = 31.0ms, epsilon = 4.95959595959596, score = 150723707.90382504, min_samples = 1
# for: zz2.txt, expected = 5, runtime = 4.01ms, epsilon = 1.0015015015015014, score = 738381.6820297515, min_samples = 1


def print_optimal_agglo_params()
    for test in mystery_data_files.items():
        file: str = test[0]
        datanp = txt_to_nparray(file_name=file)

        scores = []
        index = []
        label_dict = {}
        for i, n_clusters in enumerate(range(2, 20)):
            model = cluster.AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage='single'
            )
            model = model.fit(datanp)
            tps2 = time.time()
            labels = model.labels_

            score = compute_cah_score(datanp, labels)

            scores.append(score)
            label_dict[i] = [labels, n_clusters, score]

        i_max: int = scores.index(max(scores))

        plot_data(datanp, label_dict[i_max][0])

        print(
            f"for: {test[0]}, n_clusters = {label_dict[i_max][1]}, score = {label_dict[i_max][2]}, "
        )

