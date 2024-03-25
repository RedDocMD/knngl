import knngl
import numpy as np

q = np.load("/home/dknite/work/cpp/compush/queries.npy")
d = np.load("/home/dknite/work/cpp/compush/data.npy")
k = 2

n = knngl.knn(d, q, k)
