import pandas as pd
import random
import numpy as np


def random_number(col, line, size):
    rand = []
    for i in range(0, col * line):
        rand.append(random.randint(-50, 50) * size)
    return (rand)

def set_theta(W, B, hyp, n):
    for i in range(1, hyp.nb_layer + 1):
        rand = random_number(n[i], n[i - 1], 0.01)
        #rand = random_number(n[i], n[i - 1], 2 / line_train)
        W[i] = (np.reshape([[rand]], (n[i], n[i - 1])))
        rand = random_number(n[i], 1, 0.00)
        B[i] = (np.reshape([[0.0] * n[i]], (n[i], 1)))
    return (W, B)

class parameters:

    def __init__(self, hyp, nb_neurone, data):
        self.A = data.A
        self.A_test = data.A_test
        self.W = [0] * (hyp.nb_layer + 1)
        self.B = [0] * (hyp.nb_layer + 1)
        self.Z = [0] * (hyp.nb_layer + 1)
        self.Z_test = [0] * (hyp.nb_layer + 1)
        self.DA = [0] * (hyp.nb_layer + 1)
        self.DW = [0] *(hyp.nb_layer + 1 )
        self.DZ = [0] * (hyp.nb_layer + 1)
        self.DB = [0] * (hyp.nb_layer + 1)
        self.cost = []
        self.W, self.B = set_theta(self.W, self.B, hyp, nb_neurone)
