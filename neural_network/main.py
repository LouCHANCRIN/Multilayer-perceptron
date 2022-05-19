import sys
import argparse

import metric
import neural_network
import data_transformation as dat

def early_stopping(met, i):
    if met.cost[i - 1] < met.cost[i]:
        return True
    else:
        return False

def main(args):
    num_iters = 2000
    alpha = 0.05
    nb_layer = args.layers
    gradient_checking = args.gradient_checking

    dt = dat.data_set(args.path, 0.6)
    dt.Y_soft()
    drop = [1] # features to drop in for A0
    nb_drop = 1
    dt.create_A(nb_layer, drop, nb_drop)
    nb_neurone_by_layer = [dt.col, 30, 30, 30, 2]
    nn = neural_network.neural_network(nb_layer, nb_neurone_by_layer, dt)
    met = metric.metric()
    for i in range(0, num_iters):
        if (i % 100 == 0):
            print(i)
        activation = [0, "relu", "leaky_relu", "tanh", "soft_max"]
        nn.forward(nb_layer, activation)
        nn.backward(nb_layer, dt, activation)
        if (gradient_checking):
            print("Gradient checking")
            met.gradient_checking(nn, dt, nb_layer, activation)
            gradient_checking = False
        nn.update(nb_layer, alpha)
        met.add_cost(nn, dt, activation, nb_layer)
        if early_stopping(met, i):
            break
    YH = nn.forward_cost(nb_layer, activation)
    met.create_confu(dt, YH)
    met.accuracy(dt.line_test)
    met.precision()
    met.recall()
    met.print_cost()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse for bonus')
    parser.add_argument('--path', dest='path', help='Path to csv data', required=True)
    parser.add_argument('--layers', dest='layers', default=4, choices=range(1,11), type=int, help='Number of layers to use in the neural network')
    parser.add_argument('--gradient_checking', dest='gradient_checking', default=False, action='store_true', help='Split the training data set in training and testsing set so that we can measure how our model perform on data that it hasn\'t used to train and stop when the loss on the testing dataset to prevent overfitting')
    # parser.add_argument('--early_stopping', dest='early_stopping', default=False, action='store_true', help='Split the training data set in training and testsing set so that we can measure how our model perform on data that it hasn\'t used to train and stop when the loss on the testing dataset to prevent overfitting')

    args = parser.parse_args()

    if not args.path:
        sys.exit("No file given")
    
    main(args)
