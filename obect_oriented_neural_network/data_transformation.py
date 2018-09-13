import pandas as pd
import numpy as np

def moy(A0, line):
    count = 0
    _sum = 0
    for l in range(0, line):
        _sum += A0[l]
        count += 1
    return (_sum / count)

def change_nan(A, data):
    nb_nan = 0
    line_data, col_data = np.shape(data)
    line, col = np.shape(A)
    for c in range(0, col):
        _moy = moy(data[:,c], line_data)
        for l in range(0, line):
            if (A[l][c] != A[l][c]):
                A[l][c] = _moy
                nb_nan += 1
    #print("nb nan :", nb_nan)
    return (A)

def scale(matrix, X):
    matrix = change_nan(matrix, X)
    x, y = np.shape(matrix)
    _min = np.reshape([[0.0] * y], (y, 1))
    _max = np.reshape([[0.0] * y], (y, 1))
    _mean = np.reshape([[0.0] * y], (y, 1))
    for c in range(0, y):
        _min[c] = matrix[0][c]
        _max[c] = matrix[0][c]
    for c in range(0, y):
        for l in range(0, x):
            if (matrix[l][c] < _min[c]):
                _min[c] = matrix[l][c]
            if (matrix[l][c] > _max[c]):
                _max[c] = matrix[l][c]
        _mean[c] = moy(X[:,c], x)
    for c in range(0, y):
        for l in range(0, x):
            matrix[l][c] = (matrix[l][c] - _mean[c]) / (_max[c] - _min[c])
    return (matrix)


class data_set:

    def __init__(self, ressource, percent_train):
        print("Ressource :", ressource)
        self.data = pd.read_csv(ressource, header=None)
        self.line, self.col = np.shape(self.data)
        self.data = self.data.iloc[np.random.permutation(self.line)]
        self.data = self.data.reset_index(drop=True)
        self.line_train = int(self.line * percent_train)
        self.line_test = self.line - self.line_train

    def Y_sig(self):
        self.Y = np.reshape([[0] * self.line_train], (1, self.line_train))
        for i in range(0, self.line_train):
            if (res[i] == 'M'):
                self.Y[0][i] = 1
        if (self.line_test > 0):
            self.Y_test = np.reshape([[0] * (self.line_test)], (1, self.line_test))
            for i in range(0, self.line_test):
                if (res[i + self.line_train] == 'M'):
                    self.Y_test[0][i] = 1
        else:
            self.Y_test = self.Y

    def Y_soft(self):
        res = self.data[1]
        res = np.reshape(res, (self.line))
        self.Y = np.reshape([[0] * self.line_train * 2], (2, self.line_train))
        for i in range(0, self.line_train):
            if (res[i] == 'M'):
                self.Y[0][i] = 1
            else:
                self.Y[1][i] = 1
        if (self.line_test > 0):
            self.Y_test = np.reshape([[0] * self.line_test * 2], (2, self.line_test))
            for i in range(0, self.line_test):
                if (res[i + self.line_train] == 'M'):
                    self.Y_test[0][i] = 1
                else:
                    self.Y_test[1][i] = 1
        else:
            self.Y_test = self.Y

    def create_A(self, nb_layer, drop, nb_drop):
        if (nb_drop < self.col):
            self.col -= nb_drop
        self.A = [0] * (nb_layer + 1)
        self.A_test = [0] * (nb_layer + 1)
        X = self.data.drop(drop, axis=1).values[:self.line,:]
        self.A[0] = self.data.drop(drop, axis=1).values[:self.line_train,:]
        self.A[0] = np.reshape(self.A[0], (self.line_train, self.col))
        self.A_test[0] = self.data.drop(drop, axis=1).values[self.line_train:,:]
        self.A_test[0] = np.reshape(self.A_test[0], (self.line_test, self.col))
        self.A[0] = scale(self.A[0], X)
        self.A[0] = np.transpose(self.A[0])
        if (self.line_test > 0):
            self.A_test[0] = scale(self.A_test[0], X)
            self.A_test[0] = np.transpose(self.A_test[0])
