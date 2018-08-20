import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import sys

ressource = sys.argv[1]
data = pd.read_csv(ressource, header=None)
line, col = np.shape(data)
data = data.iloc[np.random.permutation(line)]
data = data.reset_index(drop=True)

################### RESULTAT ####################

line_train = 483
if (line_train > line):
    line_train = line
line_test = line - line_train

res = data[1]
res = np.reshape(res, (line))

Y = np.reshape([[0] * line_train], (line_train))
for i in range(0, line_train):
    if (res[i] == 'M'):
        Y[i] = 1

Y_test = np.reshape([[0] * (line - line_train)], (line - line_train))
for i in range(line_train, line):
    if (res[i] == 'M'):
        Y_test[i - line_train] = 1

################## DATA #############################

col -= 1
A0 = data.drop([1], axis=1).values[:line_train,:]
A0 = np.reshape(A0, (line_train, col))
X_test = data.drop([1], axis=1).values[line_train:,:]
X_test = np.reshape(X_test, (line_test, col))

name = []
for key in data:
    name.append(key)

def moy(A0, line):
    count = 0
    _sum = 0
    for l in range(0, line):
        _sum += A0[l]
        count += 1
    return (_sum / count)

def change_nan(A0, col, line, data, name):
    for c in range(0, col):
        if (c != 1):
            _moy = moy(data[name[c]], line)
            for l in range(0, line):
                if (A0[l][c] != A0[l][c]):
                    A0[l][c] = _moy
    return (A0)

_min = np.reshape([[0.0] * col], (col, 1))
_max = np.reshape([[0.0] * col], (col, 1))
_mean = np.reshape([[0.0] * col], (col, 1))

def scale(A0, line, col):
    for c in range(0, col):
        _min[c] = A0[0][c]
        _max[c] = A0[0][c]
    for c in range(0, col):
        for l in range(0, line):
            if (A0[l][c] < _min[c]):
                _min[c] = A0[l][c]
            if (A0[l][c] > _max[c]):
                _max[c] = A0[l][c]
            _mean[c] += A0[l][c]
    for c in range(0, col):
        _mean[c] /= line
    for c in range(0, col):
        for l in range(0, line):
            A0[l][c] = (A0[l][c] - _mean[c]) / (_max[c] - _min[c])
    return (A0)

A0 = change_nan(A0, col, line_train, data, name)
A0 = scale(A0, line_train, col)
A0 = np.transpose(A0)
X_test = change_nan(X_test, col, line_test, data, name)
X_test = scale(X_test, line_test, col)
X_test = np.transpose(X_test)

################### THETA BIAS ##########################

def random_number(col, line):
    rand = []
    for i in range(0, col * line):
        rand.append(random.randint(-50, 50) * 0.01)
    return (rand)

W = []
B = []
W.append(0)
B.append(0)

n = []
n.append(col)

# n[x] = number of neuron for layer x (n[0] = input)
n.append(30) #n[1]
n.append(30) #n[2]
n.append(30) #n[3]
n.append(1)  #n[4]

for i in range(1, 5):
    rand = random_number(n[i], n[i - 1])
    W.append(np.reshape([[rand]], (n[i], n[i - 1])))
    rand = random_number(n[i], 1)
    B.append(np.reshape([[0.0] * n[i]], (n[i], 1)))

################### NEURAL NETWORK ####################

def cost_function(Y, W, line_test, B, A_test, Z_test):
    A_test, Z, B = forward(A_test, Z_test, W, B)
    ret = 0
    YH = np.transpose(A_test[4])
    for i in range(0, line_test):
        if (Y[i] == 1):
            ret += np.log(YH[i])
        else:
            ret += np.log(1 - YH[i])
    return (-ret / line_test)

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

def forward(A, Z, W, B):
    Z[1] = W[1].dot(A[0]) + B[1]
    A[1] =  tanh(Z[1])

    Z[2] = W[2].dot(A[1]) + B[2]
    A[2] = tanh(Z[2])

    Z[3] = W[3].dot(A[2]) + B[3]
    A[3] = tanh(Z[3])

    Z[4] = W[4].dot(A[3]) + B[4]
    A[4] = sig(Z[4])
    return (A, Z, B)

def backward(A, Z, Y, W, alpha, B, line_train):
    DZ4 = A[4] - Y
    DW4 = (1 / line_train) * DZ4.dot(np.transpose(A[3]))
    DB4 = (1 / line_train) * np.sum(DZ4)

    DA3 = np.transpose(W[4]).dot(DZ4)
    DZ3 = DA3 * d_tanh(Z[3])
    DB3 = (1 / line_train) * np.sum(DZ3)
    DW3 = (1 / line_train) * DZ3.dot(np.transpose(A[2]))

    DA2 = np.transpose(W[3]).dot(DZ3)
    DZ2 = DA2 * d_tanh(Z[2])
    DB2 = (1 / line_train) * np.sum(DZ2)
    DW2 = (1 / line_train) * DZ2.dot(np.transpose(A[1]))

    DA1 = np.transpose(W[2]).dot(DZ2)
    DZ1 = DA1 * d_tanh(Z[1])
    DB1 = (1 / line_train) * np.sum(DZ1)
    DW1 = (1 / line_train) * DZ1.dot(np.transpose(A[0]))

    W[1] = W[1] - alpha * DW1
    W[2] = W[2] - alpha * DW2
    W[3] = W[3] - alpha * DW3
    W[4] = W[4] - alpha * DW4

    B[1] = B[1] - alpha * DB1
    B[2] = B[2] - alpha * DB2
    B[3] = B[3] - alpha * DB3
    B[4] = B[4] - alpha * DB4

    return (W, B)

def neural_network(A0, X_test, Y, Y_test, line_train, line_test, W, num_iters, alpha, cost, B):
    A = []
    A_test = []
    Z = []
    Z_test = []
    for i in range(0, 5):
        A.append(0)
        A_test.append(0)
        Z.append(0)
        Z_test.append(0)
    A[0] = A0
    A_test[0] = X_test
    for i in range(0, num_iters):
        if (i % 100 == 0):
            print(i)
        A, Z, B = forward(A, Z, W, B)
        W, B = backward(A, Z, Y, W, alpha, B, line_train)
        cost.append(cost_function(Y_test, W, line_test, B, A_test, Z_test))
    A[0] = X_test
    A, Z, B = forward(A, Z, W, B)
    return (A[4], cost, B)

################ HYPER PARAM ######################

num_iters = 8000
alpha = 0.002
cost = []
YH, cost, B = neural_network(A0, X_test, Y, Y_test, line_train, line - line_train, W, num_iters, alpha, cost, B)
YH = np.reshape(YH, (line_test))

################# COST VISU #################

def print_cost(cost):
    plt.plot(cost)
    plt.show()

def accuracy(line_test, confu):
    print("Accuracy : ", (confu['vn'] + confu['vp']) / line_test * 100)

def precision(confu):
    print("Precision : ", (confu['vp'] / (confu['vp'] + confu['fp']) * 100))

def recall(confu):
    print("Recall : ", (confu['vp'] / (confu['vp'] + confu['fn'])))

confusion = {}
confusion['vp'] = 0
confusion['vn'] = 0
confusion['fp'] = 0
confusion['fn'] = 0

for i in range(0, line_test):
    if (Y_test[i] == 1):
        if (YH[i] >= 0.5):
            confusion['vp'] += 1
        else:
            confusion['fn'] += 1
    else:
        if (YH[i] >= 0.5):
            confusion['fp'] += 1
        else:
            confusion['vn'] += 1


accuracy(line_test, confusion)
precision(confusion)
recall(confusion)
print_cost(cost)
print("Last value of cost : ", cost[num_iters - 1])
