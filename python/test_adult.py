import knngl
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
import datetime as dt
import timeit
import csv

data = pd.read_csv("../python/adult/adult.data", header=None)
old_data = data.copy(deep=True)
data.drop(data.columns[14], axis=1, inplace=True)

queries = pd.read_csv("../python/adult/adult.test", header=None)
old_queries = queries.copy(deep=True)
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

qlim = queries.shape[0]
k = 5

nn = NearestNeighbors()
nn.fit(np_data)
sknn = nn.kneighbors(np_queries[:qlim], k, return_distance=False)

neigh = knngl.Knn(es=False)
glnn = neigh.knn_with_ssbo(np_data, np_queries[:qlim], k)


def find_labels(neigh, data):
    labels = [''] * neigh.shape[0]
    for i, row in enumerate(neigh):
        curr_labels = {}
        for idx in row:
            label = data.iloc[idx, 13]
            curr_labels[label] = curr_labels.get(label, 0) + 1
        mxm, mxm_label = 0, None
        for label, cnt in curr_labels.items():
            if cnt > mxm:
                mxm = cnt
                mxm_label = label
        labels[i] = mxm_label
    return labels


def match_labels(labels, queries):
    cnt = 0
    for i, label in enumerate(labels):
        if queries.iloc[i, 13] == label:
            cnt += 1
    return cnt / len(labels) * 100


sknn_labels = find_labels(sknn, old_data)
glnn_labels = find_labels(glnn, old_data)

sknn_match = match_labels(sknn_labels, old_queries)
glnn_match = match_labels(glnn_labels, old_queries)

print(f"Scikit = {sknn_match} %")
print(f"KnnGL = {glnn_match} %")
