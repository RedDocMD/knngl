import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys


fname = sys.argv[1]
suff = sys.argv[2]
data = pd.read_csv(fname)

def choose_k(data, k):
    indices = []
    for i in range(data.shape[0]):
        if data.iloc[i, 0] == k:
            indices.append(i)
    return data.filter(indices, axis=0)



kdata = choose_k(data, 3)
qcnt = kdata['Query Count']
gl = kdata['KnnGL']
sk = kdata['Scikit']
speedup = np.divide(sk, gl)

fig, ax = plt.subplots()
ax.plot(qcnt, speedup)
ax.set_xlabel('Query Count')
ax.set_ylabel('Speedup')
fig.savefig(f'../results/speedup_{suff}.png')
