import sys
import argparse
import random
import json
import pickle

import metric
import neural_network
import data_transformation

def write_model(nn, activation, nb_neurone_by_layer):
    model = {'activation': activation, 'neurone_per_layer': nb_neurone_by_layer, 'weights': nn.W, 'bias': nn.B}
    with open('model.pickle', 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return

def early_stopping(met, i):
    if met.cost_test[i - 1] < met.cost_test[i]:
        return True
    else:
        return False

def main(args):
    num_iters = 2000
    alpha = 0.05
    gradient_checking = args.gradient_checking
    nb_layer = args.layers

    data = data_transformation.data_set(args.path, 0.6)
    data.Y_soft()
    drop = [1] # features to drop in for A0
    nb_drop = 1
    data.create_A(nb_layer, drop, nb_drop)

    # Last layer contain 2 possible output for the soft_max function and 1 for the sigmoid function
    nb_neurone_by_layer = [data.col, *[random.randrange(15, 30) for i in range(0, nb_layer - 1)], 2]

    nn = neural_network.neural_network(nb_layer, nb_neurone_by_layer, data)

    # activation = [0, *[nn.return_random_activation_function() for i in range(0, nb_layer - 1)], "sigmoid"]
    activation = [0, *[nn.return_random_activation_function() for i in range(0, nb_layer - 1)], "soft_max"]

    met = metric.metric()

    for i in range(0, num_iters):
        
        nn.forward(nb_layer, activation)
        nn.backward(nb_layer, data, activation)
        if args.gradient_checking:
            print("Gradient checking in progress. This can take a while")
            met.gradient_checking(nn, data, nb_layer, activation)
            args.gradient_checking = False
        nn.update(nb_layer, alpha)

        met.add_cost(nn, activation, nb_layer, data.Y, data.Y_test)
        if args.early_stopping and early_stopping(met, i):
            break
        print(f"Epoch {i}/{num_iters} - loss : {met.cost[-1]} - validation loss : {met.cost_test[-1]}")
    if args.accuracy or args.precision or args.recall or args.confusion_matrix:
        YH = nn.forward_test(nb_layer, activation)
        met.create_confu(data, YH)
    if args.accuracy:
        met.accuracy(data.line_test)
    if args.precision:
        met.precision()
    if args.recall:
        met.recall()
    if args.confusion_matrix:
        print(met.confu)
    write_model(nn, activation, nb_neurone_by_layer)
    met.print_cost()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse for bonus')
    parser.add_argument('--path', dest='path', help='Path to csv data', required=True)
    parser.add_argument('--layers', dest='layers', default=4, choices=range(1,11), type=int, help='Number of layers to use in the neural network')
    parser.add_argument('--gradient_checking', dest='gradient_checking', default=False, action='store_true', help='Split the training data set in training and testsing set so that we can measure how our model perform on data that it hasn\'t used to train and stop when the loss on the testing dataset to prevent overfitting')
    parser.add_argument('--early_stopping', dest='early_stopping', default=False, action='store_true', help='Stop the training session when the cost function start increasing')
    parser.add_argument('--accuracy', dest='accuracy', default=False, action='store_true', help='Calculate the accuracy')
    parser.add_argument('--precision', dest='precision', default=False, action='store_true', help='Calculate the precision')
    parser.add_argument('--recall', dest='recall', default=False, action='store_true', help='Calculate the recall')
    parser.add_argument('--confusion_matrix', dest='confusion_matrix', default=False, action='store_true', help='Create and show the confusion matrix')
    parser.add_argument('--cost', dest='cost', default=False, action='store_true', help='Calculate the cost function')

    args = parser.parse_args()

    if not args.path:
        sys.exit("No file given")
    
    main(args)
