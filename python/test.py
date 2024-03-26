import knngl
import numpy as np
from sklearn.neighbors import NearestNeighbors

q = np.load("/home/dknite/work/cpp/compush/queries.npy")
d = np.load("/home/dknite/work/cpp/compush/data.npy")
k = 2

nn = NearestNeighbors()
nn.fit(d)
sn = nn.kneighbors(q, k, return_distance=False)
print(sn)

neigh = knngl.Knn()
n = neigh.knn(d, q, k)
print(n)
