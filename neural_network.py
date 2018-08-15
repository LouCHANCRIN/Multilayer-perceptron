import pandas as pd
import random
import numpy as np
import sys

ressource = sys.argv[1]
data = pd.read_csv(ressource, header=None)
line, col = np.shape(data)
col -= 1

res = data[1]
res = np.reshape(res, (line))

Y = np.reshape([[0] * line], (line))
for i in range(0, line):
    if (res[i] == 'M'):
        Y[i] = 1

X = data.drop([1], axis=1).values
X = np.reshape(X, (line, col))

def random_number(col, line):
    rand = []
    for i in range(0, col * line):
        rand.append(random.randint(-50, 50) * 0.01)
    return (rand)

rand = random_number(4, col)
theta1 = np.reshape([[rand]], (4, col))
rand = random_number(4, 4)
theta2 = np.reshape([[rand]], (4, 4))
rand = random_number(4, 4)
theta3 = np.reshape([[rand]], (4, 4))
rand = random_number(4, 1)
theta4 = np.reshape([[rand]], (1, 4))

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

_min = np.reshape([[0.0] * col], (col, 1))
_max = np.reshape([[0.0] * col], (col, 1))
_mean = np.reshape([[0.0] * col], (col, 1))

def scale(X, line, col):
    for c in range(0, col):
        _min[c] = X[0][c]
        _max[c] = X[0][c]
    for c in range(0, col):
        for l in range(0, line):
            if (X[l][c] < _min[c]):
                _min[c] = X[l][c]
            if (X[l][c] > _max[c]):
                _max[c] = X[l][c]
            _mean[c] += X[l][c]
    for c in range(0, col):
        _mean[c] /= line
    for c in range(0, col):
        for l in range(0, line):
            X[l][c] = (X[l][c] - _mean[c]) / (_max[c] - _min[c])
    return (X)

def tanh(Z):
    A = np.exp(Z)
    B = np.exp(-Z)
    return ((A - B) / (A + B))

def d_tanh(Z):
    tan = tanh(Z)
    tan = tan ** 2
    return (1 - tan)

def neural_network(X, Y, col, line, W1, W2, W3, W4):
    A0 = np.transpose(X)
    print("A0  : ", np.shape(A0), "\n")

#FORWARD PROPAGATION
    print("FORWARD PROPAGATION")

    #FIRST LAYER
    print("  FIRST LAYER")
    print("    W1  : ", np.shape(W1))
    Z1 = W1.dot(A0)
    print("    Z1  : ", np.shape(Z1))
    A1 =  tanh(Z1)
    print("    A1  : ", np.shape(A1), "\n")

    #SECOND LAYER
    print("  SECOND LAYER")
    print("    W2  : ", np.shape(W2))
    Z2 = W2.dot(A1)
    print("    Z2  : ", np.shape(Z2))
    A2 = tanh(Z2)
    print("    A2  : ", np.shape(A2), "\n")

    #THIRD LAYER
    print("  THIRD LAYER")
    print("    W3  : ", np.shape(W3))
    Z3 = W3.dot(A2)
    print("    Z3  : ", np.shape(Z3))
    A3 = tanh(Z3)
    print("    A3  : ", np.shape(A3), "\n")

    #FOURTH LAYER
    print("  FOURTH LAYER")
    print("    W4  : ", np.shape(W4))
    Z4 = W4.dot(A3)
    print("    Z4  : ", np.shape(Z4))
    A4 = tanh(Z4)
    YH = A4
    print("    A4  : ", np.shape(A4))
    print("    Y   : ", np.shape(Y), "\n")

#BACKWARD PROPAGATION
    print("BACKWARD PROPAGATION")

    #FOURTH LAYER
    print("  FOURTH LAYER")
    DA4 = (-(Y / A4) + ((1 - Y) / (1 - A4)))
    print("    DA4 : ", np.shape(DA4))
    DZ4 = DA4 * d_tanh(Z4)
    print("    DZ4 : ", np.shape(DZ4))
    DW4 = (1 / line) * DZ4.dot(np.transpose(A3)) #pas sur que transpose
    print("    DW4 : ", np.shape(DW4), "\n")

    #THIRD LAYER
    print("  THIRD LAYER")
    DA3 = np.transpose(W4).dot(DZ4)
    print("    DA3 : ", np.shape(DA3))
    DZ3 = DA3 * d_tanh(Z3)
    print("    DZ3 : ", np.shape(DZ3))
    DW3 = DZ3.dot(np.transpose(A2))
    print("    DW3 : ", np.shape(DW3), "\n")

    #SECOND LAYER
    print("  SECOND LAYER")
    DA2 = np.transpose(W3).dot(DZ3)
    print("    DA2 : ", np.shape(DA2))
    DZ2 = DA2 * d_tanh(Z2)
    print("    DZ2 : ", np.shape(DZ2))
    DW2 = DZ2.dot(np.transpose(A1))
    print("    DW2 : ", np.shape(DW2), "\n")

    #FIRST LAYER
    print("  FIRST LAYER")
    DA1 = np.transpose(W2).dot(DZ2)
    print("    DA1 : ", np.shape(DA1))
    DZ1 = DA1 * d_tanh(Z1)
    print("    DZ1 : ", np.shape(DZ1))
    DW1 = DZ1.dot(np.transpose(A0))
    print("    DW1 : ", np.shape(DW1))


X = change_nan(X, col, line, data, name)
X = scale(X, line, col)
neural_network(X, Y, col, line, theta1, theta2, theta3, theta4)
