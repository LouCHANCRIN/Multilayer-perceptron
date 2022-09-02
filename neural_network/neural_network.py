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
        # Data train
        self.A = data.A
        # Data test
        self.A_test = data.A_test
        # Weight
        self.W = [0] * (nb_layer + 1)
        # Bias
        self.B = [0] * (nb_layer + 1)
        # 
        self.Z = [0] * (nb_layer + 1)
        self.Z_test = [0] * (nb_layer + 1)
        self.DA = [0] * (nb_layer + 1)
        self.DW = [0] * (nb_layer + 1 )
        self.DZ = [0] * (nb_layer + 1)
        self.DB = [0] * (nb_layer + 1)
        self.W, self.B = set_theta(self.W, self.B, nb_layer, nb_neurone)
    
################ ACTIVATION ###################

    def return_random_activation_function(self):
        available_activations = ['relu', 'leaky_relu', 'tanh']

        return available_activations[random.randrange(len(available_activations))]

    def relu(self, Z):
        # Linear function where y = x if x > 0 else y = 0.
        # The negative inputs are turned to 0 which decrease the precision of the model
        if (Z.any() < 0):
            Z = 0
        return (Z)

    def d_relu(self, Z):
        if (Z.any() < 0):
            Z = 0
        else:
            Z = 1
        return (Z)

    def leaky_relu(self, Z):
        # Linear function where y = x if x > 0 else y = x * 0.01
        # Similar to the relu function where negative input are reduce but still negative
        if (Z.any() < 0):
            Z *= 0.01
        return (Z)

    def d_leaky_relu(self, Z):
        if (Z.any() < 0):
            Z = 0.01
        else:
            Z = 1
        return (Z)

    def tanh(self, Z):
        # Sigmoid function that goes from -1 to 1. The difference with the classic sigmoid is that
        # it goes to negative values which gives more weights to negative inputs
        a = np.exp(Z)
        b = np.exp(-Z)
        return ((a - b) / (a + b))

    def d_tanh(self, Z):
        tan = self.tanh(Z)
        tan = tan ** 2
        return (1 - tan)

    def sigmoid(self, Z):
        # Sigmoid function that goes between 0 and 1.
        return (1 / (1 + np.exp(-Z)))

    def d_sigmoid(self, Z):
        sig = self.sigmoid(Z)
        return (sig * (1 - sig))

    def soft_max(self, Z):
        # Used in the final layer of neural network classifiers. Used or multi class classification.
        # The sum of the output will be equal to 1 as it gives a probability our data correspond to each class.
        return (np.exp(Z) / np.sum(np.exp(Z), axis=0))

######### FORWARD BACKWARD ############

    def forward_test(self, nb_layer, activation):
        for l in range(1, nb_layer + 1):
            self.Z_test[l] = self.W[l].dot(self.A_test[l - 1]) + self.B[l]
            if (activation[l] == "relu"):
                self.A_test[l] = self.relu(self.Z_test[l])
            elif (activation[l] == "leaky_relu"):
                self.A_test[l] = self.leaky_relu(self.Z_test[l])
            elif (activation[l] == "tanh"):
                self.A_test[l] = self.tanh(self.Z_test[l])
            elif (activation[l] == "soft_max"):
                self.A_test[l] = self.soft_max(self.Z_test[l])
            elif (activation[l] == "sigmoid"):
                self.A_test[l] = self.sigmoid(self.Z_test[l])
        return (self.A_test[nb_layer])

    def forward(self, nb_layer, activation):
        for l in range(1, nb_layer + 1):
            self.Z[l] = self.W[l].dot(self.A[l - 1]) + self.B[l]
            if (activation[l] == "relu"):
                self.A[l] = self.relu(self.Z[l])
            elif (activation[l] == "leaky_relu"):
                self.A[l] = self.leaky_relu(self.Z[l])
            elif (activation[l] == "tanh"):
                self.A[l] = self.tanh(self.Z[l])
            elif (activation[l] == "soft_max"):
                self.A[l] = self.soft_max(self.Z[l])
            elif (activation[l] == "sigmoid"):
                self.A[l] = self.sigmoid(self.Z[l])
        self.YH = self.A[nb_layer]
        return (self.YH)

    def backward(self, nb_layer, data, activation):
        l = nb_layer
        i = 1 / data.line_train
        
        # Difference between our prediction and the reality
        self.DZ[l] = self.A[l] - data.Y
        
        # Use the difference to calculate the derivative of our weights and bias
        self.DW[l] = (1 / data.line_train) * (self.DZ[l].dot(np.transpose(self.A[l - 1])))
        self.DB[l] = (1 / data.line_train) * (np.sum(self.DZ[l], axis=1, keepdims=True))

        # Backpropagate to each layer to calculate all weight and bias derivatives
        for x in range(1, l):
            y = l - x
            self.DA[y] = np.transpose(self.W[y + 1]).dot(self.DZ[y + 1])
            if (activation[y] == "relu"):
                self.DZ[y] = self.DA[y] * self.d_relu(self.Z[y])
            elif (activation[y] == "leaky_relu"):
                self.DZ[y] = self.DA[y] * self.d_leaky_relu(self.Z[y])
            elif (activation[y] == "tanh"):
                self.DZ[y] = self.DA[y] * self.d_tanh(self.Z[y])
            self.DW[y] = (1 / data.line_train) * self.DZ[y].dot(np.transpose(self.A[y - 1]))
            self.DB[y] = (1 / data.line_train) * np.sum(self.DZ[y], axis=1, keepdims=True)

    def update(self, nb_layer, alpha):
        for l in range(1, nb_layer + 1):
            self.W[l] = self.W[l] - (alpha * self.DW[l])
            self.B[l] = self.B[l] - (alpha * self.DB[l])
