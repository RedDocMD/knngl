import pandas as pd
import matplotlib.pyplot as plt
import sys


fname = sys.argv[1]
data = pd.read_csv(fname)

kvals = set(data['k'])

for k in kvals:
    indices = []
    for i in range(data.shape[0]):
        if data.iloc[i, 0] == k:
            indices.append(i)
    kdata = data.filter(indices, axis=0)
    qcnt = kdata['Query Count']
    skt = kdata['Scikit']
    gl = kdata['KnnGL']
    fig, ax = plt.subplots()
    ax.plot(qcnt, skt, qcnt, gl)
    ax.legend(['Scikit', 'KnnGL'])
    ax.set_title(f'k = {k}')
    ax.set_xlabel('Query Count')
    ax.set_ylabel('Time (s)')
    fig.savefig(f'../results/sk_vs_knngl_{k}.png')
