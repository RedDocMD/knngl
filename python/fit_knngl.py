import numpy as np
import pandas as pd

data = pd.read_csv("../results/result_adult_rpi.csv")

def choose_k(data, k):
    indices = []
    for i in range(data.shape[0]):
        if data.iloc[i, 0] == k:
            indices.append(i)
    return data.filter(indices, axis=0)

kdata = choose_k(data, 3)
qcnt = kdata['Query Count'][4:]
gl = kdata['KnnGL'][4:]

print(np.polyfit(qcnt, gl, 1, full=True))
