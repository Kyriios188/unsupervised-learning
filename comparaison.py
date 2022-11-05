from utils import *

data_files = {
    'x1.txt': 1,
    'x2.txt': 1,
    'x3.txt': 1,
    'x4.txt': 1,
    'y1.txt': 1,
    'zz1.txt': 5,
    'zz2.txt': 2,
}

for test in data_files.items():
    file: str = test[0]
    datanp = txt_to_nparray(file_name=file)

    # plot_dendrogramme(datanp)
    # set distance_threshold (0 ensures we compute the full tree)

    labels = compute_kmeans(k=1, datanp=datanp)
    #
    plot_data(datanp, labels)

    # print(
    #     f"for: {test[0]}, expected = {test[1]}, runtime = {label_dict[i_max][1]}ms, "
    #     f"min_cluster_size = {label_dict[i_max][2]}, score = {label_dict[i_max][3]}"
    # )