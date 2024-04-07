import pandas as pd
import matplotlib.pyplot as plt
import sys


fname = sys.argv[1]
suff = sys.argv[2]

data = pd.read_csv(fname)

kvals = set(data['k'])

def choose_k(data, k):
    indices = []
    for i in range(data.shape[0]):
        if data.iloc[i, 0] == k:
            indices.append(i)
    return data.filter(indices, axis=0)


fig, ax = plt.subplots()
ax.set_xlabel('Query Count')
ax.set_ylabel('Time (s)')
for k in kvals:
    kdata = choose_k(data, k)
    qcnt = kdata['Query Count']
    gl = kdata['KnnGL']
    ax.plot(qcnt, gl, label=f'k = {k}')
ax.legend()
fig.savefig(f'../results/kcomp_{suff}.png')
