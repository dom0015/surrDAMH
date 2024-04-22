import numpy as np
import os
import matplotlib.pyplot as plt

test_stages = range(1, 40, 2)
surrogate = ["poly", "poly2", "polyhalf", "polyhalf2", "d1rbf", "d1rbf2", "d2rbf", "nn2", "POLY0"]  # "nn", "rbf"]
legend = ["poly", "poly", "poly-half", "poly-half", "RBF+deg1", "RBF+deg1", "RBF+deg2", "nn", "POLY0"]  # "NN(20n,20n)", "RBF"]
problem = ["SMU"]
data_x = np.cumsum([25 * 4] * len(test_stages))
n_chains = 1


def process_data(output_dir):
    accepted = []
    rejected = []
    prerejected = []
    sum_all = []
    for i in test_stages:
        path = os.path.join(output_dir, "sampling_output", "notes", "alg" + str(i).zfill(4) + "DAMH")
        tmp = np.zeros((n_chains,))
        for j in range(n_chains):
            filename = "rank" + str(j).zfill(4) + ".csv"
            file_path = os.path.join(path, filename)
            data = np.loadtxt(file_path, dtype=int, delimiter=",", skiprows=1)
            tmp = tmp + data[:4]
            # accepted,rejected,pre-rejected,sum,seed
            # 95,5,190,290,3
        accepted.append(tmp[0])
        rejected.append(tmp[1])
        prerejected.append(tmp[2])
        sum_all.append(tmp[3])
    print("REJ/(ACC+REJ):", np.array(rejected) / (np.array(rejected) + np.array(accepted)))
    print("ACC/ALL:", np.array(rejected) / np.array(sum_all))
    path = os.path.join(output_dir, "ratio_rejected.csv")
    arr = np.array(rejected) / (np.array(accepted) + np.array(rejected))
    np.savetxt(path, arr, delimiter=",")
    path = os.path.join(output_dir, "ratio_accepted_all.csv")
    arr = np.array(accepted) / np.array(sum_all)
    np.savetxt(path, arr, delimiter=",")


for i in range(len(problem)):
    fig = plt.figure(figsize=(4, 3))
    for j in range(len(surrogate)):
        output_dir = "output_" + problem[i] + "_" + surrogate[j]
        process_data(output_dir)
        path = os.path.join(output_dir, "ratio_rejected.csv")
        data_y = np.genfromtxt(path, delimiter=",")
        plt.plot(data_x, 100 * data_y, '.-')
    plt.legend(legend)
    plt.xlabel("number of available snapshots")
    plt.ylabel("rejected samples [%]")
    plt.grid()
    fig.savefig("results_rejected_" + problem[i] + ".pdf", bbox_inches="tight")


for i in range(len(problem)):
    fig = plt.figure(figsize=(4, 3))
    for j in range(len(surrogate)):
        output_dir = "output_" + problem[i] + "_" + surrogate[j]
        process_data(output_dir)
        path = os.path.join(output_dir, "ratio_accepted_all.csv")
        data_y = np.genfromtxt(path, delimiter=",")
        plt.plot(data_x, 100 * data_y, '.-')
    plt.legend(legend)
    plt.xlabel("number of available snapshots")
    plt.ylabel("rejected samples [%]")
    plt.grid()
    fig.savefig("results_accepted_" + problem[i] + ".pdf", bbox_inches="tight")
