import knngl
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
import datetime as dt
import timeit
import csv

data = pd.read_csv("../python/rt-iot2022/RT_IOT2022")
data.drop('proto', axis=1, inplace=True)
data.drop('service', axis=1, inplace=True)
data.drop('flow_duration', axis=1, inplace=True)
data.drop('Attack_type', axis=1, inplace=True)

np_data = data.values.astype(np.float64)

def measure_times(d, q, k):
    nn = NearestNeighbors()
    def sknn():
        nn.fit(d)
        nn.kneighbors(q, k, return_distance=False)
    sktime = timeit.timeit(lambda: sknn(), number=5)

    neigh = knngl.Knn(es=False)
    gltime = timeit.timeit(lambda: neigh.knn_with_ssbo(d, q, k), number=1)
    neigh.destroy()

    return sktime, gltime

with open('../results/result_rtiot.csv', 'w') as csvfile:
    reswriter = csv.writer(csvfile)
    reswriter.writerow(['k', 'Query Count', 'Scikit', 'KnnGL'])
    kvals = [1, 3, 5]
    qvals = [20, 100, 500]
    for k in kvals:
        for qlen in qvals:
            d = np_data[:qlen, :]
            q = np_data[qlen:, :]
            sk, gl = measure_times(d, q, k)
            print(f"k = {k}, qlen = {qlen}, scikit = {sk} s, knngl = {gl} s")
            reswriter.writerow([str(k), str(qlen), str(sk), str(gl)])
