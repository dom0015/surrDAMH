import numpy as np
import os
import matplotlib.pyplot as plt

test_stages = [1, 3, 5, 7, 9]
surrogate = ["1"]  # "nn", "rbf"]
legend = ["surrogate 1"]  # "NN(20n,20n)", "RBF"]
problem = ["SMU"]
data_x = np.cumsum([25*4] * len(test_stages))


def process_data(output_dir):
    accepted = []
    rejected = []
    prerejected = []
    sum_all = []
    for i in test_stages:
        path = os.path.join(output_dir, "sampling_output", "notes", "alg" + str(i).zfill(4) + "DAMH")
        tmp = np.zeros((4,))
        for j in range(4):
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
    print("REJ/SUM:", np.array(rejected)/np.array(sum_all))

    path = os.path.join(output_dir, "ratio_rejected.csv")
    arr = np.array(rejected)/np.array(sum_all)
    np.savetxt(path, arr, delimiter=",")


for i in range(len(problem)):
    fig = plt.figure(figsize=(4, 3))
    for j in range(len(surrogate)):
        output_dir = "output_" + problem[i] + "_" + surrogate[j]
        process_data(output_dir)
        path = os.path.join(output_dir, "ratio_rejected.csv")
        data_y = np.genfromtxt(path, delimiter=",")
        plt.plot(data_x, 100*data_y, '.-')
    plt.legend(legend)
    plt.xlabel("number of available snapshots")
    plt.ylabel("rejected samples [%]")
    plt.grid()
    fig.savefig("collected_results_" + problem[i] + ".pdf", bbox_inches="tight")
