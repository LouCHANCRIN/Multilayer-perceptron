import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import sys

ressource = sys.argv[1]
data = pd.read_csv(ressource, header=None)
line, col = np.shape(data)
col -= 1

################### RESULTAT ####################

res = data[1]
res = np.reshape(res, (line))

Y = np.reshape([[0] * line], (line))
for i in range(0, line):
    if (res[i] == 'M'):
        Y[i] = 1

################## DATA #############################

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

X = change_nan(X, col, line, data, name)
X = scale(X, line, col)
A0 = np.transpose(X)

################### THETA ##########################

def random_number(col, line):
    rand = []
    for i in range(0, col * line):
        rand.append(random.randint(-50, 50) * 0.01)
    return (rand)
W = []
for i in range(0, 5):
    W.append(0)

rand = random_number(30, col)
W[1] = np.reshape([[rand]], (30, col))

rand = random_number(30, 4)
W[2] = np.reshape([[rand]], (4, 30))

rand = random_number(30, 4)
W[3] = np.reshape([[rand]], (30, 4))

rand = random_number(30, 1)
W[4] = np.reshape([[rand]], (1, 30))

################### NEURAL NETWORK ####################

def cost_function(Y, YH, line):
    ret = ((YH ** Y) * ((1 - YH) ** (1 - Y)))
    return (-np.sum(np.log(ret)) / line)

def sig(Z):
    return (1 / (1 + np.exp(-Z)))

def tanh(Z):
    A = np.exp(Z)
    B = np.exp(-Z)
    return ((A - B) / (A + B))

def d_tanh(Z):
    tan = tanh(Z)
    tan = tan ** 2
    return (1 - tan)

def forward(A, Z, W):
    Z[1] = W[1].dot(A[0])
    A[1] =  tanh(Z[1])

    Z[2] = W[2].dot(A[1])
    A[2] = tanh(Z[2])

    Z[3] = W[3].dot(A[2])
    A[3] = tanh(Z[3])

    Z[4] = W[4].dot(A[3])
    A[4] = sig(Z[4])
    return (A, Z)

def backward(A, Z, Y, W, alpha):
    DZ4 = A[4] - Y
    DW4 = (1 / line) * DZ4.dot(np.transpose(A[3])) #pas sur que transpose
    DA3 = np.transpose(W[4]).dot(DZ4)
    DZ3 = DA3 * d_tanh(Z[3])
    DW3 = (1 / line) * DZ3.dot(np.transpose(A[2]))
    DA2 = np.transpose(W[3]).dot(DZ3)
    DZ2 = DA2 * d_tanh(Z[2])
    DW2 = (1 / line) * DZ2.dot(np.transpose(A[1]))
    DA1 = np.transpose(W[2]).dot(DZ2)
    DZ1 = DA1 * d_tanh(Z[1])
    DW1 = (1 / line) * DZ1.dot(np.transpose(A[0]))

    W[1] = W[1] - alpha * DW1
    W[2] = W[2] - alpha * DW2
    W[3] = W[3] - alpha * DW3
    W[4] = W[4] - alpha * DW4
    return (W)

def neural_network(A0, Y, line, W, num_iters, alpha, cost):
    A = []
    Z = []
    for i in range(0, 5):
        A.append(0)
        Z.append(0)
    A[0] = A0
    for i in range(0, num_iters):
        print(i)
        A, Z = forward(A, Z, W)
        W = backward(A, Z, Y, W, alpha)
        cost.append(cost_function(Y, A[4], line))
    return (A[4], cost)

num_iters = 5000
alpha = 0.3
cost = []
YH, cost = neural_network(A0, Y, line, W, num_iters, alpha, cost)

################# COST VISU #################

def print_res(Y, YH):
    for i in range(0, line):
        print(Y[i], "  ", YH[i])

def print_cost(cost):
    plt.plot(cost)
    plt.show()

YH = np.reshape(YH, (line))
#print_res(Y, YH)
print_cost(cost)
print("Last value of cost : ", cost[num_iters - 1])
print("* 100              : ", cost[num_iters - 1] * 100)
