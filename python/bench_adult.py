import knngl
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np

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

np_queries = np_queries[:850, :]

k = 3

nn = NearestNeighbors()
nn.fit(np_data)
sn = nn.kneighbors(np_queries, k, return_distance=False)
print(sn)

neigh = knngl.Knn(es=False)
n = neigh.knn_with_ssbo(np_data, np_queries, k)
print(n)
