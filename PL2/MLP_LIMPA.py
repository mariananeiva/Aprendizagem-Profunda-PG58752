# -*- coding: utf-8 -*-

import numpy as np


class Dataset:
    def __init__(self, filename=None, X=None, Y=None):
        if filename is not None:
            self.readDataset(filename)
        elif X is not None and Y is not None:
            self.X = X
            self.Y = Y
    
    def readDataset(self, filename, sep=","):
        data = np.genfromtxt(filename, delimiter=sep)
        self.X = data[:, 0:-1]
        self.Y = data[:, -1]

    def getXy(self):
        return self.X, self.Y


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class MLP:
    
    def __init__(self, ds, hidden_nodes=2):
        self.X, self.y = ds.getXy()
        self.X = np.hstack((np.ones([self.X.shape[0], 1]), self.X))
        self.h = hidden_nodes
        self.w1 = np.zeros([hidden_nodes, self.X.shape[1]])
        self.w2 = np.zeros([1, hidden_nodes + 1])

    def predict(self, instance):
        x = np.empty([self.X.shape[1]])
        x[0] = 1
        x[1:] = np.array(instance)

        a1 = np.dot(self.w1, x)
        hidden_output = sigmoid(a1)

        b = np.empty(hidden_output.shape[0] + 1)
        b[0] = 1
        b[1:] = hidden_output

        c = np.dot(self.w2, b)
        output = sigmoid(c)

        return output

    def costFunction(self):
        m = self.X.shape[0]

        z1 = np.dot(self.X, self.w1.T)
        hidden = sigmoid(z1)
        hidden = np.hstack((np.ones([hidden.shape[0], 1]), hidden))

        z2 = np.dot(hidden, self.w2.T)
        predict = sigmoid(z2)

        sqe = (predict.flatten() - self.y) ** 2
        cost = np.sum(sqe) / (2 * m)

        return cost

    def setWeights(self, w1, w2):
        self.w1 = w1
        self.w2 = w2


def test():
    ds = Dataset("xnor.data")
    nn = MLP(ds, 2)

    w1 = np.array([[-30, 20, 20],
                   [10, -20, -20]])

    w2 = np.array([[-10, 20, 20]])

    nn.setWeights(w1, w2)

    print(nn.predict(np.array([0, 0])))
    print(nn.predict(np.array([0, 1])))
    print(nn.predict(np.array([1, 0])))
    print(nn.predict(np.array([1, 1])))

    print("Cost:", nn.costFunction())


test()
