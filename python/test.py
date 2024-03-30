import knngl
import numpy as np
from sklearn.neighbors import NearestNeighbors

q = np.load("../python/queries.npy")
d = np.load("../python/data.npy")
k = 2

print("Scikit")
nn = NearestNeighbors()
nn.fit(d)
sn = nn.kneighbors(q, k, return_distance=False)
print(sn)

print("KNNGL")
neigh = knngl.Knn(es=False)
n = neigh.knn(d, q, k)
print(n)

print("KNNGL with SSBO")
n = neigh.knn_with_ssbo(d, q, k)
print(n)
