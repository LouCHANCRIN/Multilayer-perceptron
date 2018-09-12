import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt

class metric:

    def __init__(self):
        self.cost = []

    def create_confu(self, dt, YH):
        self.confu = {}
        self.confu['vp'] = 0
        self.confu['vn'] = 0
        self.confu['fp'] = 0
        self.confu['fn'] = 0
        for i in range(0, dt.line_test):
            if (dt.Y_test[0][i] == 1):
                if (YH[0][i] >= 0.5):
                    self.confu['vp'] += 1
                else:
                    self.confu['fn'] += 1
            else:
                if (YH[0][i] >= 0.5):
                    self.confu['fp'] += 1
                else:
                    self.confu['vn'] += 1

    def print_cost(self):
        plt.plot(self.cost)
        plt.show()

    def accuracy(self, line_test):
        if (line_test != 0):
            x = (self.confu['vn'] + self.confu['vp']) / line_test * 100
            print("Accuracy : ", x)

    def precision(self):
        if (self.confu['vp'] + self.confu['fp'] != 0):
            x = (self.confu['vp'] / (self.confu['vp'] + self.confu['fp']) * 100)
            print("Precision : ", x)

    def recall(self):
        if (self.confu['vp'] + self.confu['fn'] != 0):
            x = (self.confu['vp'] / (self.confu['vp'] + self.confu['fn'])) * 100
            print("Recall : ", x)

#    def forward_cost(self, nn, dt, activation, nb_layer):
#        for l in range(1, nb_layer + 1):
#            print("ok\n")
#            nn.Z_test[l] = nn.W[l].dot(nn.A[l - 1]) + nn.B[l]
#            if (activation[l] == "relu"):
#                nn.A_test[l] = nn.relu(l)
#            elif (activation[l] == "leaky_relu"):
#                nn.A_test[l] = nn.leaky_relu(l)
#            elif (activation[l] == "tanh"):
#                nn.A_test[l] = nn.tanh(l)
#            elif (activation[l] == "soft_max"):
#                nn.A_test[l] = nn.soft_max(l)
#            elif (activation[l] == "sigmoid"):
#                nn.A_test[l] = nn.sigmoid(l)
#        return (nn.A_test[nb_layer])

    def add_cost(self, nn, dt, activation, nb_layer):
        self.cost.append(self.cost_function(nn, dt, activation, nb_layer))

    def cost_function(self, nn, dt, activation, nb_layer):
        nn.forward(nb_layer, activation)
#       need to do forward with test set
        ret = 0
        y, x = np.shape(nn.YH)
        if (activation[nb_layer] == "soft_max"):
            for i in range(0, y):
                for j in range(0, x):
                    if (dt.Y_test[i][j] == 1):
                        ret += dt.Y_test[i][j] * np.log(nn.YH[i][j])
        else:
            for i in range(0, y):
                for j in range(0, x):
                    if (dt.Y_test[i][j] == 1):
                        ret += np.log(nn.YH[i][j])
                    else:
                        ret += np.log(1 - nn.YH[i][j])
        return (-ret / x)

    def gradient_checking(self, nn, dt, nb_layer, activation):
        size = 0
        for i in range(1, nb_layer + 1):
            x, y = np.shape(nn.W[i])
            size += x * y + x
        DT = np.reshape([[0.0] * size], (size, 1))
        Dapprox = np.reshape([[0.0] * size], (size, 1))
        a = 0
        for i in range(1, nb_layer + 1):
            x, y = np.shape(nn.W[i])
            for j in range(0, x):
                for k in range(0, y):
                    DT[a] = nn.DW[i][j][k]
                    a += 1
            for j in range(0, x):
                DT[a] = nn.DB[i][j]
                a += 1
        eps = 0.0000001
        print("epsilon : ", eps)
        l = 0
        for i in range(1, nb_layer + 1):
            x, y = np.shape(nn.W[i])
            for j in range(0, x):
                for k in range(0, y):
                    nn.W[i][j][k] += eps
                    cost1 = self.cost_function(nn, dt, activation, nb_layer)
                    nn.W[i][j][k] -= (2 * eps)
                    cost2 = self.cost_function(nn, dt, activation, nb_layer)
                    nn.W[i][j][k] += eps
                    Dapprox[l] = ((cost1 - cost2) / (2 * eps))
                    l += 1
            for j in range(0, x):
                nn.B[i][j] += eps
                cost1 = self.cost_function(nn, dt, activation, nb_layer)
                nn.B[i][j] -= (2 * eps)
                cost2 = self.cost_function(nn, dt, activation, nb_layer)
                nn.B[i][j] += eps
                Dapprox[l] = ((cost1 - cost2) / (2 * eps))
                l += 1
        a = np.sqrt(np.sum((Dapprox - DT) ** 2))
        b = np.sqrt(np.sum((Dapprox ** 2)))
        c = np.sqrt(np.sum((DT ** 2)))
        check = a / (b + c)
        print("Gradient checking : ", check)
