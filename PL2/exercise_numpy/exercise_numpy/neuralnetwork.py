#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from layers import DenseLayer



class NeuralNetwork:
 
    def __init__(self):
        # attributes
        self.layers = []

    def add(self, layer, biases = None, weights = None):
        if self.layers:
            layer.set_input_shape(input_shape=self.layers[-1].output_shape())
        if biases is not None: layer.set_biases(biases)
        if weights is not None: layer.set_weigths(weights)
        self.layers.append(layer)
        return self
    
    # Completei aqui
    def forward_propagation(self, X, training):
        output = X
        for layer in self.layers:
            output = layer.forward_propagation(output, training)
        return output

    def predict(self, dataset):
        return self.forward_propagation(dataset.X, training=False)

    def score(self, dataset, predictions):
        if self.metric is not None:
            return self.metric(dataset.y, predictions)
        else:
            raise ValueError("No metric specified for the neural network.")


if __name__ == '__main__':
    from activation import SigmoidActivation
    from data import read_csv

    # training data
    dataset = read_csv('xnor.data', sep=',', features=False, label=True)

    # network - complete code
    net = NeuralNetwork()
    n_features = dataset.X.shape[1]

    # complete here to create the network - 1 hidden layer with 2 neurons + one output layer
    # weights should be the ones from the hands-on exercise
    # activation layer should be Sigmoid
    net.add(DenseLayer(2, (n_features,)), biases=np.array([-30, 10]), weights=np.array([[20, 20], [-20, -20]]))
    net.add(SigmoidActivation())
    net.add(DenseLayer(1), biases=np.array([-10]), weights=np.array([[20], [20]]))
    net.add(SigmoidActivation())
  
    
    print("Predictions for the training dataset:")
    print(net.predict(dataset))
    
    

