import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('../results/result_adult.csv')
rpi_data = pd.read_csv('../results/result_adult_rpi.csv')

kvals = set(data['k'])

def choose_k(data, k):
    indices = []
    for i in range(data.shape[0]):
        if data.iloc[i, 0] == k:
            indices.append(i)
    return data.filter(indices, axis=0)


for k in kvals:
    kdata = choose_k(data, k)
    rpi_kdata = choose_k(rpi_data, k)
    qcnt = kdata['Query Count'][:3]
    gl = kdata['KnnGL'][:3]
    rpi_qcnt = rpi_kdata['Query Count']
    rpi_gl = rpi_kdata['KnnGL']
    fig, ax = plt.subplots()
    ax.plot(qcnt, gl, rpi_qcnt, rpi_gl)
    ax.legend(['Nvidia GTX 1080Ti', 'Raspberry Pi 4B+'])
    ax.set_title(f'k = {k}')
    ax.set_xlabel('Query Count')
    ax.set_ylabel('Time (s)')
    fig.savefig(f'../results/nvidia_vs_rpi_{k}.png')
