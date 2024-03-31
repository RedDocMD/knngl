import knngl
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
import datetime as dt
import timeit
import csv

data = pd.read_csv("../python/adult/adult.data", header=None)
data.drop(data.columns[14], axis=1, inplace=True)

queries = pd.read_csv("../python/adult/adult.test", header=None)
queries.drop(queries.columns[14], axis=1, inplace=True)

column_desc = """age: continuous.
workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
fnlwgt: continuous.
education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
education-num: continuous.
marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
sex: Female, Male.
capital-gain: continuous.
capital-loss: continuous.
hours-per-week: continuous.
native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands."""

columns = []
classes = []

for i, line in enumerate(column_desc.split("\n")):
    line = line[:-1]
    parts = line.split(": ")
    if len(parts) != 2:
        raise RuntimeError(f"Too many parts: {len(parts)}")
    if parts[1] == "continuous":
        continue
    columns.append(i)
    classes.append(parts[1].split(", "))


def label_to_numbers(data, columns, classes):
    for col, cls in zip(columns, classes):
        def find_idx(x):
            xx = x.strip()
            if xx in classes:
                return cls.index(xx)
            else:
                return len(cls) + 1

        data.iloc[:, col] = data.iloc[:, col].apply(find_idx)
    return data


data = label_to_numbers(data, columns, classes)
queries = label_to_numbers(queries, columns, classes)

np_data = data.values.astype(np.float64)
np_queries = queries.values.astype(np.float64)


def measure_times(d, q, k):
    nn = NearestNeighbors()
    def sknn():
        nn.fit(d)
        nn.kneighbors(q, k, return_distance=False)
    sktime = timeit.timeit(lambda: sknn(), number=5)

    neigh = knngl.Knn(es=False)
    gltime = timeit.timeit(lambda: neigh.knn_with_ssbo(d, q, k), number=1)

    return sktime, gltime

with open('../results/result_adult.csv', 'w') as csvfile:
    reswriter = csv.writer(csvfile)
    reswriter.writerow(['k', 'Query Count', 'Scikit', 'KnnGL'])
    kvals = [1, 3, 5]
    qvals = [20, 100, 500, 1000, 5000, 16000]
    for k in kvals:
        for qlen in qvals:
            sk, gl = measure_times(np_data, np_queries[:qlen, :], k)
            print(f"k = {k}, qlen = {qlen}, scikit = {sk} s, knngl = {gl} s")
            reswriter.writerow([str(k), str(qlen), str(sk), str(gl)])
