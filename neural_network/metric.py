import numpy as np
import matplotlib.pyplot as plt

class metric:
    '''
    Metrics object containing to evaluate the precision of the model
    '''

    def __init__(self):
        self.cost = []
        self.cost_test = []

    '''
    Sumarize the results
    vp (vrai positif): positive prediction / positive real value
    vn (vrai négatif): negative prediction / negative real value
    fp (faux positif): positive prediction / negative real value
    fn (faux négatif): negative prediction / positive real value
    The confusion show the precision of the model and the type of errors
    '''
    def create_confu(self, data, YH, test=True):
        if test:
            data_Y = data.Y_test
            data_length = data.line_test
        else:
            data_Y = data.Y
            data_length = data.line
        self.confu = {'vp': 0, 'vn': 0, 'fp': 0, 'fn': 0}
        for i in range(0, data_length):
            if (data_Y[0][i] == 1):
                if (YH[0][i] >= 0.5):
                    self.confu['vp'] += 1
                else:
                    self.confu['fp'] += 1
            else:
                if (YH[0][i] >= 0.5):
                    self.confu['fn'] += 1
                else:
                    self.confu['vn'] += 1

    def print_cost(self):
        a, = plt.plot(self.cost, label='loss', color='blue')
        b, = plt.plot(self.cost_test, label='validation loss', color='red')
        plt.legend(handles=[a, b])
        plt.show()

    '''
    Percentage of correct predictions
    '''
    def accuracy(self, line_test):
        if (line_test != 0):
            x = (self.confu['vn'] + self.confu['vp']) / line_test * 100
            print("Accuracy : ", x)

    '''
    How good the model is at predicting a specific category (malignent or benign)
    Here we are calculating the precision on malignent tumors by looking at how many times
    the model predicted wrongly on a positive value.
    '''
    def precision(self):
        if (self.confu['vp'] + self.confu['fp'] != 0):
            x = (self.confu['vp'] / (self.confu['vp'] + self.confu['fp']) * 100)
            print("Precision : ", x)

    '''
    Tells how many times the model was able to detect a specific category (malignent or benign)
    Calculate how many time the model predicted correctly on positive value and how many time it were
    wrong when predicting value as positive
    '''
    def recall(self):
        if (self.confu['vp'] + self.confu['fn'] != 0):
            x = (self.confu['vp'] / (self.confu['vp'] + self.confu['fn'])) * 100
            print("Recall : ", x)

    def add_cost(self, nn, activation, nb_layer, Y, Y_test):
        self.cost.append(self.cost_function(nn, activation, nb_layer, False, Y, "train"))
        self.cost_test.append(self.cost_function(nn, activation, nb_layer, False, Y_test))

    def cost_function(self, nn, activation, nb_layer, gradient, Y, dataset=None):
        if gradient == True or dataset == "train":
            YH = nn.forward(nb_layer, activation)
        else:
            YH = nn.forward_test(nb_layer, activation)
        ret = 0
        y, x = np.shape(YH)
        for i in range(0, y):
            for j in range(0, x):
                if (activation[nb_layer] == "soft_max"):
                    ret += Y[i][j] * np.log(YH[i][j])
                else:
                    if (Y[i][j] == 1):
                        ret += np.log(YH[i][j])
                    else:
                        ret += np.log(1 - YH[i][j])
        return (-ret / x)


    def gradient_checking(self, nn, data, nb_layer, activation):
        size = 0
        eps = 0.0000001
        approx_index = 0
        dt_index = 0
        for i in range(1, nb_layer + 1):
            x, y = np.shape(nn.W[i])
            size += x * y + x
        DT = np.reshape([[0.0] * size], (size, 1))
        Dapprox = np.reshape([[0.0] * size], (size, 1))
        for i in range(1, nb_layer + 1):
            x, y = np.shape(nn.W[i])

            for j in range(0, x):
                for k in range(0, y):
                    DT[dt_index] = nn.DW[i][j][k]
                    dt_index += 1
            for j in range(0, x):
                DT[dt_index] = nn.DB[i][j]
                dt_index += 1

        for i in range(1, nb_layer + 1):
            x, y = np.shape(nn.W[i])
            for j in range(0, x):
                for k in range(0, y):
                    nn.W[i][j][k] += eps
                    cost1 = self.cost_function(nn, activation, nb_layer, True, data.Y)
                    nn.W[i][j][k] -= (2 * eps)
                    cost2 = self.cost_function(nn, activation, nb_layer, True, data.Y)
                    nn.W[i][j][k] += eps
                    Dapprox[approx_index] = ((cost1 - cost2) / (2 * eps))
                    approx_index += 1
            for j in range(0, x):
                nn.B[i][j] += eps
                cost1 = self.cost_function(nn, activation, nb_layer, True, data.Y)
                nn.B[i][j] -= (2 * eps)
                cost2 = self.cost_function(nn, activation, nb_layer, True, data.Y)
                nn.B[i][j] += eps
                Dapprox[approx_index] = ((cost1 - cost2) / (2 * eps))
                approx_index += 1
        a = np.sqrt(np.sum((Dapprox - DT) ** 2))
        b = np.sqrt(np.sum((Dapprox ** 2)))
        c = np.sqrt(np.sum((DT ** 2)))
        check = a / (b + c)
        print(f"Gradient checking : {check} (epsilon : {eps})")



    '''
    Check the derivatives computed by the neural network to increase confidence in the
    implementation of the code.
    Backporpagation is prone to mistakes when implementing neural network from scratch.
    With gradient checking we estimate the gradient (derivative) and if it is close to
    the calculated gradient we can consider it was implement correctly.
    '''
    def gradient_checking2(self, nn, data, nb_layer, activation):
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
        approx_index = 0
        for i in range(1, nb_layer + 1):
            x, y = np.shape(nn.W[i])
            for j in range(0, x):
                for k in range(0, y):
                    nn.W[i][j][k] += eps
                    cost1 = self.cost_function(nn, data, activation, nb_layer, True, data.Y)
                    nn.W[i][j][k] -= (2 * eps)
                    cost2 = self.cost_function(nn, data, activation, nb_layer, True, data.Y)
                    nn.W[i][j][k] += eps
                    Dapprox[approx_index] = ((cost1 - cost2) / (2 * eps))
                    approx_index += 1
            for j in range(0, x):
                nn.B[i][j] += eps
                cost1 = self.cost_function(nn, data, activation, nb_layer, True, data.Y)
                nn.B[i][j] -= (2 * eps)
                cost2 = self.cost_function(nn, data, activation, nb_layer, True, data.Y)
                nn.B[i][j] += eps
                Dapprox[approx_index] = ((cost1 - cost2) / (2 * eps))
                approx_index += 1
        a = np.sqrt(np.sum((Dapprox - DT) ** 2))
        b = np.sqrt(np.sum((Dapprox ** 2)))
        c = np.sqrt(np.sum((DT ** 2)))
        check = a / (b + c)
        print(f"Gradient checking : {check} (epsilon : {eps})")
