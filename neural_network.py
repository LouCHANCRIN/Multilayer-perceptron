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
X = np.reshape(X, (line, col))

def random_number(col, line):
    rand = []
    for i in range(0, col * line):
        rand.append(random.randint(-50, 50) * 0.000000001)
    return (rand)

rand = random_number(4, line)
theta1 = np.reshape([[rand]], (4, line))
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


def tanh(Z):
    return ((np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z)))

def d_tanh(Z):
    tan = tanh(Z)
    tan = tan * tan
    return (1 - tan)

def neural_network(X, Y, col, line, theta1, theta2, theta3, theta4):
    Z1 = np.reshape([[0.0] * 4 * line], (4, line))
    Z2 = np.reshape([[0.0] * 4 * line], (4, line))
    Z3 = np.reshape([[0.0] * 4 * line], (4, line))
    Z4 = np.reshape([[0.0] * 4 * line], (4, line))

#FORWARD PROPAGATION
    A0 = X
    print("A0 : ", np.shape(A0), "\n")

    #FIRST LAYER
    print("Theta1 : ", np.shape(theta1))
    Z1 = theta1.dot(A0)
    print("Z1 : ", np.shape(Z1))
    A1 =  tanh(Z1)
    print("A1 : ", np.shape(A1), "\n")

    #SECOND LAYER
    print("Theta2 : ", np.shape(theta2))
    Z2 = theta2.dot(A1)
    print("Z2 : ", np.shape(Z2))
    A2 = tanh(Z2)
    print("A2 : ", np.shape(A2), "\n")

    #THIRD LAYER
    print("Theta3 : ", np.shape(theta3))
    Z3 = theta3.dot(A2)
    print("Z3 : ", np.shape(Z3))
    A3 = tanh(Z3)
    print("A3 : ", np.shape(A3), "\n")

    #FOURTH LAYER
    print("Theta4 : ", np.shape(theta4))
    Z4 = theta4.dot(A3)
    print("Z4 : ", np.shape(Z4))
    A4 = tanh(Z4)
    print("A4 : ", np.shape(A4))

#BACKWARD PROPAGATION


X = change_nan(X, col, line, data, name)
neural_network(X, Y, col, line, theta1, theta2, theta3, theta4)
