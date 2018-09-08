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

################### HYPER PARAM ##########################

num_iters = 5000
alpha = 0.05
gradient = 0

momentum = 0 # set a 1 to activate momentum gradient
beta = 0.9

lambd = 0 # set as 1 to activate l2 regularization

drop = [1] # features to drop in for A0 
col -= 1
#drop = [0,1,2,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,25,26,27,28,2,9,30,31]
#col -= 26

line_train = int(line * 0.60)
if (line_train > line):
    line_train = line
line_test = line - line_train

################# ACTIVATION FUNCTION / LAYER ###################

activation = []
activation.append(0)
activation.append("relu")       # ATTENTION il peut etre nescessaire de modifier
activation.append("leaky_relu") # Y et/ou le nombre de neurone
activation.append("tanh")       # en fonction de la fonction d'activation
activation.append("soft_max")   # utiliser sur le denier layer 
                                # (1 neurone pour la sigmoid, le nombre de classe
                                # si softmax (toujours avoir au moins 2 classe pour soft))
n = []
n.append(col)

nb_layer = 4
# n[x] = number of neuron for layer x (n[0] = input)
n.append(30) #n[1]
n.append(30) #n[2]
n.append(30) #n[3] # ATTENTION il peut etre nescessaire de modifier
n.append(2) #n[4] # Y en fonction du nombre de neurone sur le dernier layer  
                              
##################### THETA BIAS ##########################

def random_number(col, line, size):
    rand = []
    for i in range(0, col * line):
        rand.append(random.randint(-50, 50) * size)
    return (rand)

W = []
B = []
W.append(0)
B.append(0)

for i in range(1, nb_layer + 1):
    rand = random_number(n[i], n[i - 1], 0.01)
    #rand = random_number(n[i], n[i - 1], 2 / line_train)
    W.append(np.reshape([[rand]], (n[i], n[i - 1])))
    rand = random_number(n[i], 1, 0.00)
    B.append(np.reshape([[0.0] * n[i]], (n[i], 1)))

##################### Y (RESULTAT) ####################

res = data[1]
res = np.reshape(res, (line))

Y = np.reshape([[0] * line_train], (1, line_train))
for i in range(0, line_train):
    if (res[i] == 'M'):
        Y[0][i] = 1

if (line_test > 0):
    Y_test = np.reshape([[0] * (line - line_train)], (1, line - line_train))
    for i in range(line_train, line):
        if (res[i] == 'M'):
            Y_test[0][i - line_train] = 1
else:
    Y_test = Y

################ Y SOFT FONCTION 2 CLASS ###################

Y_soft = np.reshape([[0] * line_train * 2], (2, line_train))
for i in range(0, line_train):
    if (res[i] == 'M'):
        Y_soft[0][i] = 1
        Y_soft[1][i] = 0
    else:
        Y_soft[1][i] = 1

if (line_test > 0):
    Y_soft_test = np.reshape([[0] * (line - line_train) * 2], (2, line - line_train))
    for i in range(line_train, line):
        if (res[i] == 'M'):
            Y_soft_test[0][i - line_train] = 1
        else:
            Y_soft_test[1][i - line_train] = 1
else:
    Y_soft_test = Y_soft

################## DATA SCALING #############################

A0 = data.drop(drop, axis=1).values[:line_train,:]
A0 = np.reshape(A0, (line_train, col))
X_test = data.drop(drop, axis=1).values[line_train:,:]
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
    nb_nan = 0
    for c in range(0, col):
        if (c != 1):
            _moy = moy(data[name[c]], line)
            for l in range(0, line):
                if (A0[l][c] != A0[l][c]):
                    A0[l][c] = _moy
                    nb_nan += 1
    #print("nb nan :", nb_nan)
    return (A0)

def scale(matrix):
    x, y = np.shape(matrix)
    _min = np.reshape([[0.0] * y], (y, 1))
    _max = np.reshape([[0.0] * y], (y, 1))
    _mean = np.reshape([[0.0] * y], (y, 1))
    for c in range(0, y):
        _min[c] = matrix[0][c]
        _max[c] = matrix[0][c]
    for c in range(0, y):
        for l in range(0, x):
            if (matrix[l][c] < _min[c]):
                _min[c] = matrix[l][c]
            if (matrix[l][c] > _max[c]):
                _max[c] = matrix[l][c]
            _mean[c] += matrix[l][c]
    for c in range(0, y):
        _mean[c] /= x
    for c in range(0, y):
        for l in range(0, x):
            matrix[l][c] = (matrix[l][c] - _mean[c]) / (_max[c] - _min[c])
    return (matrix)

A0 = change_nan(A0, col, line_train, data, name)
A0 = scale(A0)
A0 = np.transpose(A0)
if (line_test > 0):
    X_test = change_nan(X_test, col, line_test, data, name)
    X_test = scale(X_test)
    X_test = np.transpose(X_test)

################### CHECK FUNCTION ####################

def print_cost(cost):
    plt.plot(cost)
    plt.show()

def accuracy(line_test, confu):
    if (line_test != 0):
        print("Accuracy : ", (confu['vn'] + confu['vp']) / line_test * 100)

def precision(confu):
    if (confu['vp'] + confu['fp'] != 0):
        print("Precision : ", (confu['vp'] / (confu['vp'] + confu['fp']) * 100))

def recall(confu):
    if (confu['vp'] + confu['fn'] != 0):
        print("Recall : ", (confu['vp'] / (confu['vp'] + confu['fn'])) * 100)

def cost_function(Y, W, B, A_test, Z_test, activation):
    forward(A_test, Z_test, W, B, activation)
    ret = 0
    YH = A_test[nb_layer]
    y, x = np.shape(YH)
    #print(np.shape(Y), np.shape(YH))
    if (activation[nb_layer] == "soft_max"):
        for i in range(0, y):
            for j in range(0, x):
                if (Y[i][j] == 1):
                    ret += Y[i][j] * np.log(YH[i][j])
    else:
        for i in range(0, y):
            for j in range(0, x):
                if (Y[i][j] == 1):
                    ret += np.log(YH[i][j])
                else:
                    ret += np.log(1 - YH[i][j])
    if (lambd != 1):
        return ((-ret / (x)))
    else:
        for i in range(1, nb_layer + 1):
            regu = np.sum(np.transpose(W[i]).dot(W[i]))
            return ((-ret / (x * y)) + ((lambd / (2 * x)) * regu))

def gradient_checking(W, B, DW, DB, Y_test, line_test, A_test, Z_test, nb_layer):
    size = 0
    for i in range(1, nb_layer + 1):
        x, y = np.shape(W[i])
        size += x * y + x
    DT = np.reshape([[0.0] * size], (size, 1))
    Dapprox = np.reshape([[0.0] * size], (size, 1))
    a = 0
    for i in range(1, nb_layer + 1):
        x, y = np.shape(W[i])
        for j in range(0, x):
            for k in range(0, y):
                DT[a] = DW[i][j][k]
                a += 1
        for j in range(0, x):
            DT[a] = DB[i][j]
            a += 1
    eps = 0.0000001
    print("epsilon : ", eps)
    l = 0
    for i in range(1, nb_layer + 1):
        x, y = np.shape(W[i])
        for j in range(0, x):
            for k in range(0, y):
                W[i][j][k] += eps
                cost1 = cost_function(Y_test, W, B, A_test, Z_test, activation)
                W[i][j][k] -= (2 * eps)
                cost2 = cost_function(Y_test, W, B, A_test, Z_test, activation)
                W[i][j][k] += eps
                Dapprox[l] = ((cost1 - cost2) / (2 * eps))
                l += 1
        for j in range(0, x):
            B[i][j] += eps
            cost1 = cost_function(Y_test, W, B, A_test, Z_test, activation)
            B[i][j] -= (2 * eps)
            cost2 = cost_function(Y_test, W, B, A_test, Z_test, activation)
            B[i][j] += eps
            Dapprox[l] = ((cost1 - cost2) / (2 * eps))
            l += 1
    a = np.sqrt(np.sum((Dapprox - DT) ** 2))
    b = np.sqrt(np.sum((Dapprox ** 2)))
    c = np.sqrt(np.sum((DT ** 2)))
    check = a / (b + c)
    print("Gradient checking : ", check)
    #for i in range(0, size):
     #   print(Dapprox[i], DT[i])
    return (0)

################# ATCIVATION FUNCTION #################

def soft_max(Z):
    return (np.exp(Z) / np.sum(np.exp(Z), axis = 0))

def relu(Z):
    if (Z.any() < 0):
        Z = 0
    return (Z)

def d_relu(Z):
    if (Z.any() < 0):
        Z = 0
    else:
        Z = 1
    return (Z)

def leaky_relu(Z):
    if (Z.any() < 0):
        Z *= 0.01
    return (Z)

def d_leaky_relu(Z):
    if (Z.any() < 0):
        Z = 0.01
    else:
        Z = 1
    return (Z)

def sigmoid(Z):
    return (1 / (1 + np.exp(-Z)))

def d_sigmoid(Z):
    s = sigmoid(Z)
    return (s * (1 - s))

def tanh(Z):
    A = np.exp(Z)
    B = np.exp(-Z)
    return ((A - B) / (A + B))

def d_tanh(Z):
    tan = tanh(Z)
    tan = tan ** 2
    return (1 - tan)

##################### NEURAL NETWORK ##################

def forward(A, Z, W, B, activation):
    for l in range(1, nb_layer + 1):
        Z[l] = W[l].dot(A[l - 1]) + B[l]
        #Z[l] = scale(Z[l])
        if (activation[l] == "relu"):
            A[l] =  relu(Z[l])
        elif (activation[l] == "leaky_relu"):
            A[l] =  leaky_relu(Z[l])
        elif (activation[l] == "tanh"):
            A[l] =  tanh(Z[l])
        elif (activation[l] == "soft_max"):
            A[l] =  soft_max(Z[l])
        elif (activation[l] == "sigmoid"):
            A[l] =  sigmoid(Z[l])

def backward(A, DA, W, DW, B, DB, Z, DZ, Y, activation, alpha, gradient, nb_layer):
    l = nb_layer
    DZ[l] = A[l] - Y
    DW[l] = (1 / line_train) * (DZ[l].dot(np.transpose(A[l - 1])))
    DB[l] = (1 / line_train) * (np.sum(DZ[l], axis=1, keepdims=True))
    for x in range(1, l):
        DA[l - x] = np.transpose(W[l - x + 1]).dot(DZ[l - x + 1])
        if (activation[l - x] == "relu"):
            DZ[l - x] = DA[l - x] * d_relu(Z[l - x])
        elif (activation[l - x] == "leaky_relu"):
            DZ[l - x] = DA[l - x] * d_leaky_relu(Z[l - x])
        elif (activation[l - x] == "tanh"):
            DZ[l - x] = DA[l - x] * d_tanh(Z[l - x])
        DW[l - x] = (1 / line_train) * (DZ[l - x].dot(np.transpose(A[l - 1 - x])))
        DB[l - x] = (1 / line_train) * (np.sum(DZ[l - x], axis=1, keepdims=True))
    if (gradient == 1):
        gradient_checking(W, B, DW, DB, Y, line_train, A, Z, nb_layer)
    if (momentum != 1):
        for i in range(1, nb_layer + 1):
            W[i] = W[i] - alpha * (DW[i] + ((lambd / (2 * line_train)) * W[i]))
            B[i] = B[i] - alpha * DB[i]
    else:
        for i in range(1, nb_layer + 1):
            x, y = np.shape(W[i])
            vdw = np.reshape([[0.0] * x * y], (x, y))
            x, y = np.shape(B[i])
            vdb = np.reshape([[0.0] * x * y], (x, y))
            vdw = (beta * vdw) + ((1 - beta) * DW[i])
            vdb = (beta * vdb) + ((1 - beta) * DB[i])
            W[i] = W[i] - (alpha * vdw)
            B[i] = B[i] - (alpha * vdb)

def neural_network(Y, Y_test, W, cost, B, activation, alpha, gradient):
    for i in range(0, num_iters):
        if (i % 100 == 0):
            #alpha *= 0.98
            print(i)
        forward(A, Z, W, B, activation)
        backward(A, DA, W, DW, B, DB, Z, DZ, Y, activation, alpha, gradient, nb_layer)
        if (gradient == 1):
            gradient = 0
        cost.append(cost_function(Y_test, W, B, A_test, Z_test, activation))
        if (cost[i] > cost[i - 1]):
            print(i + 1)
            A[0] = X_test
            forward(A, Z, W, B, activation)
            return (A[nb_layer], i)
    A[0] = X_test
    forward(A, Z, W, B, activation)
    return (A[nb_layer], num_iters)

############## INITIALISATION #################

A = []
A_test = []
Z = []
Z_test = []
DA = []
DW = []
DZ = []
DB = []
for i in range(0, nb_layer + 1):
    A.append(0)
    DA.append(0)
    DW.append(0)
    DZ.append(0)
    DB.append(0)
    A_test.append(0)
    Z.append(0)
    Z_test.append(0)
A[0] = A0
if (line_train == line):
    A_test[0] = A0
else:
    A_test[0] = X_test
cost = []

if (activation[nb_layer] == "soft_max"):
    YH, num_iters = neural_network(Y_soft, Y_soft_test, W, cost, B, activation, alpha, gradient)
else:
    YH, num_iters = neural_network(Y, Y_test, W, cost, B, activation, alpha, gradient)
YH = np.reshape(YH, (n[nb_layer], line_test))

################# COST VISU CHECK #################

confusion = {}
confusion['vp'] = 0
confusion['vn'] = 0
confusion['fp'] = 0
confusion['fn'] = 0

def print_YH(Y, YH):
    for i in range(0, line_test):
        print(YH[0][i], Y[i])

for i in range(0, line_test):
    if (Y_test[0][i] == 1):
        if (YH[0][i] >= 0.5):
            confusion['vp'] += 1
        else:
            confusion['fn'] += 1
    else:
        if (YH[0][i] >= 0.5):
            confusion['fp'] += 1
        else:
            confusion['vn'] += 1


#print_YH(Y, Y_test)
accuracy(line_test, confusion)
precision(confusion)
recall(confusion)
print("Last value of cost : ", cost[num_iters - 1])
print_cost(cost)
