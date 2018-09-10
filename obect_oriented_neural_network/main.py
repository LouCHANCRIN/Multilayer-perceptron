import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import sys
import data_transformation as dat
import parameters as param

ressource = sys.argv[3]

class hyper_parameters:

    def __init__(self):
        self.num_iters = 5000
        self.alpha = 0.05    # learning rate
        self.gradient = 0 # set as 1 to activate gradient checking
        self.momentum = 0 # set a 1 to activate momentum gradient
        self.beta = 0.9
        self.lambd = 0    # set as 1 to activate l2 regularization
        self.nb_layer = 4

hyp = hyper_parameters()

dt = dat.data_set(ressource, 0.6)
dt.Y_soft()

drop = [1] # features to drop in for A0
nb_drop = 1
dt.create_A(hyp, drop, nb_drop)
nb_neurone = [dt.col, 30, 30, 30, 2]
par = param.parameters(hyp, nb_neurone)
print(dt.A[0])
