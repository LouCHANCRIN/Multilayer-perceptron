import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import sys
import data_transformation as dat
import neural_network as nn
import metric

ressource = sys.argv[1]

num_iters = 2000
alpha = 0.05
nb_layer = 4
gradient_checking = 0
dt = dat.data_set(ressource, 0.6)
dt.Y_soft()

drop = [1] # features to drop in for A0
nb_drop = 1
dt.create_A(nb_layer, drop, nb_drop)

nb_neurone_by_layer = [dt.col, 30, 30, 30, 2] 
nn = nn.neural_network(nb_layer, nb_neurone_by_layer, dt)
met = metric.metric()

def main(nn, dt, metric, num_iters, alpha, nb_layer, gradient_checking):
    for i in range(0, num_iters):
        if (i % 100 == 0):
            print(i)
        activation = [0, "relu", "leaky_relu", "tanh", "soft_max"]
        nn.forward(nb_layer, activation)
        nn.backward(nb_layer, dt, activation)
        if (gradient_checking == 1):
            met.gradient_checking(nn, dt, nb_layer, activation)
            gradient_checking = 0
        nn.update(nb_layer, alpha)
        met.add_cost(nn, dt, activation, nb_layer)
        if (met.cost[i - 1] < met.cost[i]):
            break
    YH = nn.forward_cost(nb_layer, activation)
    met.create_confu(dt, YH)
    met.accuracy(dt.line_test)
    met.precision()
    met.recall()
    met.print_cost()
        

main(nn, dt, metric, num_iters, alpha, nb_layer, gradient_checking)
