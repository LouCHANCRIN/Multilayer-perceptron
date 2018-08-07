import pandas as pd
import numpy as np
import sys

ressource = sys.argv[1]
data = pd.read_csv(ressource, header=None)
line, col = np.shape(data)
col -= 1

Y = data[1]
Y = np.reshape(Y, (line, 1))

X = data.drop([1], axis=1).values
X = np.reshape(X, (line, col))

name = []
for key in data:
    name.append(key)

def moy(X, line):
    count = 0
    _sum = 0
    for l in range(0, line):
        _sum += X[l]
        count += 1
    return (_sum / count)

def change_nan(X, col, line, data, name):
    for c in range(0, col):
        if (c != 1)
            _moy = moy(data[name[c]], line)
            for l in range(0, line):
                if (X[l][c] != X[l][c]):
                    X[l][c] = _moy
    return (X)

X = change_nan(X, col, line, data, name)
