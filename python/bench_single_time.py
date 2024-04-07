import knngl
import numpy as np

dim = 10
q = np.random.rand(1, dim)

dmax = 4500
k = 3
neigh = knngl.Knn(es=False)

for dcnt in range(k, dmax):
    d = np.random.rand(dcnt, dim)
    print(f"#d = {dcnt}", flush=True)
    neigh.knn_with_ssbo(d, q, k)
