import pandas as pd
import random
import numpy as np
import sys

ressource = sys.argv[1]
data = pd.read_csv(ressource, header=None)
line, col = np.shape(data)
col -= 1

Y = data[1]
Y = np.reshape(Y, (line, 1))

X = data.drop([1], axis=1).values
print(X)
X = np.reshape(X, (line, col))

#theta = {}
#theta['0'] = np.reshape([[0.0] * col], (col, 1))
#theta['1'] = np.reshape([[0.0] * col], (col, 1))

rand = []
for i in range(0, col * 2):
    rand.append(random.randint(-50, 50) * 0.000000001)

rand = np.reshape(rand, (col * 2, 1))
theta = np.reshape([[rand]], (2, col))

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
        if (c != 1):
            _moy = moy(data[name[c]], line)
            for l in range(0, line):
                if (X[l][c] != X[l][c]):
                    X[l][c] = _moy
    return (X)

def neural_network(X, Y, col, line, theta):
    print("X : ", np.shape(X))
    print(X)
    print("Theta : ", np.shape(theta))
    print(theta)
    Z = theta.dot(np.transpose(X))
    print("Z : ", np.shape(Z))
    print(Z)
    A = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
    print("A : ", np.shape(A))
    print(A)

X = change_nan(X, col, line, data, name)
neural_network(X, Y, col, line, theta)
