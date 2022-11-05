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

mystery_data_files = [
    # 'x1.txt',
    # 'x2.txt',
    # 'x3.txt',
    # 'x4.txt',
    'y1.txt',
    'zz1.txt',
    'zz2.txt'
]

for test in mystery_data_files:
    datanp = txt_to_nparray(file_name=test)

    t1 = time.time()
    labels_kmeans = compute_kmeans(k=kmeans_params[test], datanp=datanp)
    t2 = time.time()
    kmeans_delta = round((t2 - t1) * 1000)
    kmeans_title: str = f"{test} : kmeans"
    plot_data(datanp, labels=labels_kmeans, title=kmeans_title)
    kmeans_score_cah = compute_cah_score(datanp, labels_kmeans)
    kmeans_score_db = compute_db_score(datanp, labels_kmeans)

    print(
        f"KMEANS {test}: runtime = {kmeans_delta}ms, "
        f"score cah = {kmeans_score_cah}, score db = {kmeans_score_db}"
    )

    t1 = time.time()
    try:
        labels_kmedoids = compute_kmedoids(k=kmedoids_params[test], datanp=datanp)
        t2 = time.time()
        kmedoids_delta = round((t2 - t1) * 1000)
        kmedoids_title: str = f"{test} : kmedoids"
        plot_data(datanp, labels=labels_kmedoids, title=kmedoids_title)
        kmedoids_score_cah = compute_cah_score(datanp, labels_kmedoids)
        kmedoids_score_db = compute_db_score(datanp, labels_kmedoids)

        print(
            f"KMEDOIDS {test}: runtime = {kmedoids_delta}ms, "
            f"score cah = {kmedoids_score_cah}, score db = {kmedoids_score_db}"
        )
    except Exception:
        print("KMEDOIDS FAILURE")


    dbscan_delta, labels_dbscan = apply_dbsdcan(datanp, eps=dbscan_params[test][0], min_samples=dbscan_params[test][1])
    dbscan_title: str = f"{test} : dbscan"
    plot_data(datanp, labels=labels_dbscan, title=dbscan_title)
    dbscan_score_cah = compute_cah_score(datanp, labels_dbscan)
    try:
        dbscan_score_db = compute_db_score(datanp, labels_dbscan)
    except Exception:
        dbscan_score_db = -1

    print(
        f"DBSCAN {test}: runtime = {dbscan_delta}ms, "
        f"score cah = {dbscan_score_cah}, score db = {dbscan_score_db}"
    )

    hdbscan_delta, labels_hdbscan = apply_hdbsdcan(
        datanp,
        min_cluster_size=hdbscan_params[test][0],
        min_samples=hdbscan_params[test][1]
)
    hdbscan_title: str = f"{test} : hdbscan"
    plot_data(datanp, labels=labels_hdbscan, title=hdbscan_title)
    hdbscan_score_cah = compute_cah_score(datanp, labels_hdbscan)
    try:
        hdbscan_score_db = compute_db_score(datanp, labels_hdbscan)
    except Exception:
        hdbscan_score_db = -1

    print(
        f"HDBSCAN {test}: runtime = {dbscan_delta}ms, "
        f"score cah = {hdbscan_score_cah}, score db = {hdbscan_score_db}"
    )


# KMEANS x1.txt: runtime = 122ms, score cah = 22675.25398265906, score db = 0.3665165709413832
# KMEDOIDS x1.txt: runtime = 377ms, score cah = 22674.606167199447, score db = 0.3662086466871276
# DBSCAN x1.txt: runtime = 24.0ms, score cah = 416633831.28228086, score db = 7.979067854618894e-05
# HDBSCAN x1.txt: runtime = 24.0ms, score cah = 10224.651225697351, score db = 1.444095161466401
# KMEANS x2.txt: runtime = 50ms, score cah = 13506.719363408463, score db = 0.4653852153516968
# KMEDOIDS x2.txt: runtime = 408ms, score cah = 13501.333151854633, score db = 0.46576256162136637
# DBSCAN x2.txt: runtime = 23.0ms, score cah = 388141857.9395273, score db = 0.000209101778029385
# HDBSCAN x2.txt: runtime = 23.0ms, score cah = 3069.2152372848077, score db = 1.1842632401557207
# KMEANS x3.txt: runtime = 51ms, score cah = 7889.530175761837, score db = 0.6451804695079131
# KMEDOIDS x3.txt: runtime = 393ms, score cah = 7873.777954132968, score db = 0.6399357717117689
# DBSCAN x3.txt: runtime = 23.98ms, score cah = -1, score db = 999
# HDBSCAN x3.txt: runtime = 23.98ms, score cah = 14.955428874648081, score db = 1.3726950867328835
# KMEANS x4.txt: runtime = 73ms, score cah = 6205.099546105215, score db = 0.6606340461611526
# KMEDOIDS x4.txt: runtime = 396ms, score cah = 6180.542256227132, score db = 0.6502067662762185
# DBSCAN x4.txt: runtime = 25.03ms, score cah = 391222072.2337754, score db = 7.889923383048841e-05
# HDBSCAN x4.txt: runtime = 25.03ms, score cah = 13.459971963526138, score db = 1.4193719031416077


# KMEANS y1.txt: runtime = 722ms, score cah = 98172.39782931602, score db = 0.7909818472928289
# KMEDOIDS FAILURE
# DBSCAN y1.txt: runtime = 597.03ms, score cah = 285585620.97581446, score db = -1
# HDBSCAN y1.txt: runtime = 597.03ms, score cah = 18687.767359899564, score db = 2.547732791395315
# KMEANS zz1.txt: runtime = 31ms, score cah = 41907.51910226982, score db = 0.39179573896910524
# KMEDOIDS zz1.txt: runtime = 584ms, score cah = 41906.09441054583, score db = 0.39285733518169175
# DBSCAN zz1.txt: runtime = 32.97ms, score cah = 150723707.90382504, score db = 0.0001663551098199061
# HDBSCAN zz1.txt: runtime = 32.97ms, score cah = 30034.825475140486, score db = 2.085912679995276
# KMEANS zz2.txt: runtime = 36ms, score cah = 9235.838711440301, score db = 0.48663024303473296
# KMEDOIDS zz2.txt: runtime = 21ms, score cah = 9235.838711440303, score db = 0.48663024303473296
# DBSCAN zz2.txt: runtime = 4.96ms, score cah = 738381.6820297515, score db = 0.01016663317879917
# HDBSCAN zz2.txt: runtime = 4.96ms, score cah = 6436.146784901418, score db = 0.36537732829404335
