from utils import *

kmeans_params = {
    'x1.txt': 15,
    'x2.txt': 15,
    'x3.txt': 15,
    'x4.txt': 15,
    'y1.txt': 8,
    'zz1.txt': 8,
    'zz2.txt': 5,
}

kmedoids_params = {
    'x1.txt': 15,
    'x2.txt': 15,
    'x3.txt': 15,
    'x4.txt': 15,
    'y1.txt': 2,
    'zz1.txt': 8,
    'zz2.txt': 5,
}
hdbscan_params = {
    'x1.txt': (37, 1),
    'x2.txt': (22, 4),
    'x3.txt': (2, 1),
    'x4.txt': (2, 1),
    'y1.txt': (7, 61),
    'zz1.txt': (6, 3),
    'zz2.txt': (87, 58),
}

dbscan_params = {
    'x1.txt': (23.7, 1),
    'x2.txt': (23.2, 1),
    'x3.txt': (38.1, 1),
    'x4.txt': (17.3, 1),
    'y1.txt': (0.1, 1),
    'zz1.txt': (4.95, 1),
    'zz2.txt': (1.00, 1),
}
agglo_params = {
    'x1.txt': (15, 'ward'),
    'x2.txt': (15, 'average'),
    'x3.txt': (15, 'ward'),
    'x4.txt': (17, 'ward'),
    'y1.txt': (2, 'single'),  # single, car c'est le seul test qu'on a réussi à faire jusqu'au bout
    'zz1.txt': (8, 'average'),
    'zz2.txt': (5, 'average'),
}

mystery_data_files = [
    'x1.txt',
    'x2.txt',
    'x3.txt',
    'x4.txt',
    'y1.txt',
    'zz1.txt',
    'zz2.txt'
]

for test in mystery_data_files:
    datanp = txt_to_nparray(file_name=test)

    # t1 = time.time()
    # labels_kmeans = compute_kmeans(k=kmeans_params[test], datanp=datanp)
    # t2 = time.time()
    # kmeans_delta = round((t2 - t1) * 1000)
    # kmeans_title: str = f"{test} : kmeans"
    # plot_data(datanp, labels=labels_kmeans, title=kmeans_title)
    # kmeans_score_cah = compute_cah_score(datanp, labels_kmeans)
    # kmeans_score_db = compute_db_score(datanp, labels_kmeans)
    #
    # print(
    #     f"KMEANS {test}: runtime = {kmeans_delta}ms, "
    #     f"score cah = {kmeans_score_cah}, score db = {kmeans_score_db}"
    # )
    #
    # t1 = time.time()
    # try:
    #     labels_kmedoids = compute_kmedoids(k=kmedoids_params[test], datanp=datanp)
    #     t2 = time.time()
    #     kmedoids_delta = round((t2 - t1) * 1000)
    #     kmedoids_title: str = f"{test} : kmedoids"
    #     plot_data(datanp, labels=labels_kmedoids, title=kmedoids_title)
    #     kmedoids_score_cah = compute_cah_score(datanp, labels_kmedoids)
    #     kmedoids_score_db = compute_db_score(datanp, labels_kmedoids)
    #
    #     print(
    #         f"KMEDOIDS {test}: runtime = {kmedoids_delta}ms, "
    #         f"score cah = {kmedoids_score_cah}, score db = {kmedoids_score_db}"
    #     )
    # except Exception:
    #     print("KMEDOIDS FAILURE")
    #
    # dbscan_delta, labels_dbscan = apply_dbsdcan(datanp, eps=dbscan_params[test][0], min_samples=dbscan_params[test][1])
    # dbscan_title: str = f"{test} : dbscan"
    # plot_data(datanp, labels=labels_dbscan, title=dbscan_title)
    # dbscan_score_cah = compute_cah_score(datanp, labels_dbscan)
    # try:
    #     dbscan_score_db = compute_db_score(datanp, labels_dbscan)
    # except Exception:
    #     dbscan_score_db = -1
    #
    # print(
    #     f"DBSCAN {test}: runtime = {dbscan_delta}ms, "
    #     f"score cah = {dbscan_score_cah}, score db = {dbscan_score_db}"
    # )

    # hdbscan_delta, labels_hdbscan = apply_hdbsdcan(
    #     datanp,
    #     min_cluster_size=hdbscan_params[test][0],
    #     min_samples=hdbscan_params[test][1]
    # )
    # hdbscan_title: str = f"{test} : hdbscan"
    # plot_data(datanp, labels=labels_hdbscan, title=hdbscan_title)
    # hdbscan_score_cah = compute_cah_score(datanp, labels_hdbscan)
    # try:
    #     hdbscan_score_db = compute_db_score(datanp, labels_hdbscan)
    # except Exception:
    #     hdbscan_score_db = -1
    #
    # print(
    #     f"HDBSCAN {test}: runtime = {hdbscan_delta}ms, "
    #     f"score cah = {hdbscan_score_cah}, score db = {hdbscan_score_db}"
    # )


    tps1 = time.time()
    model = cluster.AgglomerativeClustering(
        n_clusters=agglo_params[test][0],
        linkage=agglo_params[test][1]
    )
    model = model.fit(datanp)
    tps2 = time.time()
    agglo_delta = round((tps2 - tps1) * 1000)
    labels_agglo = model.labels_

    agglo_title: str = f"{test} : clustering agglomératif"
    plot_data(datanp, labels=labels_agglo, title=agglo_title)

    score_agglo_cah = compute_cah_score(datanp, labels_agglo)
    try:
        agglo_score_db = compute_db_score(datanp, labels_agglo)
    except Exception:
        agglo_score_db = -1

    print(
        f"Agglo {test}: runtime = {agglo_delta}ms, "
        f"score cah = {score_agglo_cah}, score db = {agglo_score_db}"
    )
