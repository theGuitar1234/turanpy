from turanpy.deep_learning import activation
from turanpy.deep_learning import neural_network
from turanpy.deep_learning import initLayers

import random

import time

def main(X, Y, hidden_layers, number_of_classes, number_of_features, act):
    
    WB = initLayers(X, hidden_layers, number_of_classes, number_of_features)

    Y_HAT = neural_network(X, WB, act)

    LOSS = [loss(Y_HAT[i][0], Y[i][0]) for i in range(len(Y))]
    
    print(LOSS)

def loss(y_hat, y):
    return 1/2 * (y_hat - y)**2

if __name__ == "__main__":

    X = [
        [0.2, 1.1],
        [0.3, 1.0],
        [0.4, 0.9],
        [0.5, 1.2],
        [0.6, 1.0],

        [1.2, 0.1],
        [1.0, 0.2],
        [1.3, 0.3],
        [1.5, 0.4],
        [1.4, 0.2],

        [2.0, 1.5],
        [2.2, 1.7],
        [2.1, 1.6],
        [1.9, 1.4],
        [2.3, 1.8],
    ]

    Y = [
        [0],
        [0],
        [0],
        [0],
        [0],

        [1],
        [1],
        [1],
        [1],
        [1],

        [1],
        [1],
        [1],
        [1],
        [1],
    ]

    hidden_layers = 4
    number_of_classes = 3
    number_of_features = 2

    main(X, Y, hidden_layers, number_of_classes, number_of_features, activation.relu)