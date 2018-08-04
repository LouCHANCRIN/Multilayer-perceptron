import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

ressource = sys.argv[1]
data = pd.read_csv(ressource) # traiter les data pour avoir de vrai key
line, col = np.shape(data)

df = []
for key in data:
    df.append(key)
print(df)
data.append(df, ignore_index=True) #fail parce que la liste est une chaine de char et pas des int
print(data)

Y = data["M"]
print(Y)
Y = np.reshape(Y, (line, 1))

X = [np.insert(row, 0, 1) for row in data.drop(["M"], axis=1).values]
df = pd.DataFrame([[1.0] * 32])
X.append(df)
line += 1
X = np.reshape(X, (line, col))
print(X)

name = []
a = 1
for key in data:
    name.append(key)
    if (key != 'M'):
        X[line][a] = key
        a += 1
print(name)

def moy(X, line):
    count = 0
    _sum = 0
    for l in range(0, line):
        if (X[l] == X[l]):
            _sum += X[l]
            count += 1
    return (_sum / count)

def change_nan(X, col, line, data, name):
    a = 0
    for c in range(0, col):
        if (name[c] != "First Name"):
                _moy = moy(data[name[c]], line)
                for l in range(0, line):
                    if (X[l][c] != X[l][c]):
                        X[l][c] = _moy
                        a = a + 1
    return (X)

def pair_plot(X, name, col, line, Y):
    a = 1
    for c in range(1, col):
        for c2 in range (0, col):
            if (name[c] != 'First Name' and name[c] != 'Last Name'):
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
                    plt.hist([V1['M'], V1['B'], V1['M'], V1['B']], bins='auto', density='true',
                        color=['yellow', 'red', 'green', 'blue'], edgecolor='black')
                    plt.legend(['M', 'B'])
                    a += 1
                else:
                    plt.subplot(5, 5, a)
                    plt.xlabel(name[c])
                    plt.ylabel(name[c2])
                    plt.scatter(V1['M'], V2['M'], color='black', edgecolor='black')
                    plt.scatter(V1['B'], V2['B'], color='red', edgecolor='black')
                    plt.legend(['M', 'B'])
                    a += 1
                if (a == 26):
                    a = 1
                    plt.show()
    plt.show()

X = change_nan(X, col, line, data, name)
pair_plot(X, name, col, line, Y)
