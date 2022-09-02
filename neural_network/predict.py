import json
import numpy as np
import argparse
import pickle

import neural_network
import data_transformation
import metric

def main(args):
    with open('model.pickle', 'rb') as handle:
        model = pickle.load(handle)

    for i in range(1, len(model['weights'])):
        model['weights'][i] = np.array(model['weights'][i])
        model['bias'][i] = np.array(model['bias'][i])

    nb_layer = len(model['activation']) - 1

    data = data_transformation.data_set(args.path, 1)
    # data = data_transformation.data_set(args.path, 0.6)
    data.Y_soft()
    drop = [1] # features to drop in for A0
    nb_drop = 1
    data.create_A(nb_layer, drop, nb_drop)



    nn = neural_network.neural_network(nb_layer, model['neurone_per_layer'], data)
    nn.W = model['weights']
    nn.B = model['bias']

    YH = nn.forward(nb_layer, model['activation'])

    met = metric.metric()

    met.create_confu(data, YH, test=False)
    met.accuracy(data.line_test)
    met.precision()
    met.recall()
    print(met.confu)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse for bonus')
    parser.add_argument('--path', dest='path', help='Path to csv data', required=True)

    args = parser.parse_args()

    if not args.path:
        sys.exit("No file given")
    main(args)