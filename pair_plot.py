import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

ressource = sys.argv[1]
data = pd.read_csv(ressource, header=None)
line, col = np.shape(data)
col -= 1

Y = data[1]
Y = np.reshape(Y, (line))

X = data.drop([1], axis=1).values
X = np.reshape(X, (line, col))

name = []
for key in data:
    name.append(key)


def moy(X, line):
    count = 0
    _sum = 0
    for l in range(0, line):
        if (X[l] == X[l]):
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

def pair_plot(X, name, col, line, Y):
    a = 1
    for c in range(0, col):
        for c2 in range (0, col):
            if (name[c] != '?'):
                V1 = {}
                V2 = {}
                V1['M'] = []
                V1['B'] = []
                V2['M'] = []
                V2['B'] = []
                for l in range(0, line):
                    if (Y[l] == 'M'):
                        if (X[l][c] == X[l][c]):
                            V1['M'].append(X[l][c])
                        if (X[l][c2] == X[l][c2]):
                            V2['M'].append(X[l][c2])
                    if (Y[l] == 'B'):
                        if (X[l][c] == X[l][c]):
                            V1['B'].append(X[l][c])
                        if (X[l][c2] == X[l][c2]):
                            V2['B'].append(X[l][c2])
                if (c == c2):
                    plt.subplot(5, 5, a)
                    plt.xlabel(name[c])
                    plt.ylabel(name[c2])
                    plt.hist([V1['M'], V1['B']], bins='auto', density='true',
                        color=['blue', 'red'], edgecolor='black')
                    plt.legend(['M', 'B'])
                    a += 1
                else:
                    plt.subplot(5, 5, a)
                    plt.xlabel(name[c])
                    plt.ylabel(name[c2])
                    plt.scatter(V1['M'], V2['M'], color='blue', edgecolor='black')
                    plt.scatter(V1['B'], V2['B'], color='red', edgecolor='black')
                    plt.legend(['M', 'B'])
                    a += 1
                if (a == 26):
                    a = 1
                    plt.show()
    plt.show()

X = change_nan(X, col, line, data, name)
pair_plot(X, name, col, line, Y)
