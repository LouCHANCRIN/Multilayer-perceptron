import pandas as pd
import random
import numpy as np

def random_number(col, line, size):
    rand = []
    for i in range(0, col * line):
        rand.append(random.randint(-50, 50) * size)
    return (rand)

def set_theta(W, B, nb_layer, n):
    for i in range(1, nb_layer + 1):
        rand = random_number(n[i], n[i - 1], 0.01)
        W[i] = (np.reshape([[rand]], (n[i], n[i - 1])))
        rand = random_number(n[i], 1, 0.00)
        B[i] = (np.reshape([[0.0] * n[i]], (n[i], 1)))
    return (W, B)

class neural_network:

    def __init__(self, nb_layer, nb_neurone, data):
        self.A = data.A
        self.A_test = data.A_test
        self.W = [0] * (nb_layer + 1)
        self.B = [0] * (nb_layer + 1)
        self.Z = [0] * (nb_layer + 1)
        self.Z_test = [0] * (nb_layer + 1)
        self.DA = [0] * (nb_layer + 1)
        self.DW = [0] * (nb_layer + 1 )
        self.DZ = [0] * (nb_layer + 1)
        self.DB = [0] * (nb_layer + 1)
        self.W, self.B = set_theta(self.W, self.B, nb_layer, nb_neurone)
    
################ ACTIVATION ###################

    def relu(self, l):
        if (self.Z[l].any() < 0):
            self.Z[l] = 0
        return (self.Z[l])

    def d_relu(self, l):
        if (self.Z[l].any() < 0):
            self.Z[l] = 0
        else:
            self.Z[l] = 1
        return (self.Z[l])

    def leaky_relu(self, l):
        if (self.Z[l].any() < 0):
            self.Z[l] *= 0.01
        return (self.Z[l])

    def d_leaky_relu(self, l):
        if (self.Z[l].any() < 0):
            self.Z[l] = 0.01
        else:
            self.Z[l] = 1
        return (self.Z[l])

    def tanh(self, l):
        a = np.exp(self.Z[l])
        b = np.exp(-self.Z[l])
        return ((a - b) / (a + b))

    def d_tanh(self, l):
        tan = self.tanh(l)
        tan = tan ** 2
        return (1 - tan)

    def sigmoid(self, l):
        return (1 / (1 + np.exp(-Z)))

    def d_sigmoid(self, l):
        s = self.sigmoid(l)
        return (s * (1 - s))

    def soft_max(self, l):
        return (np.exp(self.Z[l]) / np.sum(np.exp(self.Z[l]), axis=0))

######### FORWARD BACKWARD ############

    def forward(self, nb_layer, activation):
        for l in range(1, nb_layer + 1):
            self.Z[l] = self.W[l].dot(self.A[l - 1]) + self.B[l]
            if (activation[l] == "relu"):
                self.A[l] = self.relu(l)
            elif (activation[l] == "leaky_relu"):
                self.A[l] = self.leaky_relu(l)
            elif (activation[l] == "tanh"):
                self.A[l] = self.tanh(l)
            elif (activation[l] == "soft_max"):
                self.A[l] = self.soft_max(l)
            elif (activation[l] == "sigmoid"):
                self.A[l] = self.sigmoid(l)
        self.YH = self.A[nb_layer]

    def backward(self, nb_layer, dt, activation):
        l = nb_layer
        i = 1 / dt.line_train
        self.DZ[l] = self.A[l] - dt.Y
        self.DW[l] = i * (self.DZ[l].dot(np.transpose(self.A[l - 1])))
        self.DB[l] = i * (np.sum(self.DZ[l], axis=1, keepdims=True))
        for x in range(1, l):
            y = l - x
            self.DA[y] = np.transpose(self.W[y + 1]).dot(self.DZ[y + 1])
            if (activation[y] == "relu"):
                self.DZ[y] = self.DA[y] * self.d_relu(y)
            elif (activation[y] == "leaky_relu"):
                self.DZ[y] = self.DA[y] * self.d_leaky_relu(y)
            elif (activation[y] == "tanh"):
                self.DZ[y] = self.DA[y] * self.d_tanh(y)
            self.DW[y] = i * self.DZ[y].dot(np.transpose(self.A[y - 1]))
            self.DB[y] = i * np.sum(self.DZ[y], axis=1, keepdims=True)

    def update(self, nb_layer, alpha):
        for l in range(1, nb_layer + 1):
            self.W[l] = self.W[l] - (alpha * self.DW[l])
            self.B[l] = self.B[l] - (alpha * self.DB[l])
