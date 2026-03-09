from turanpy.deep_learning import activation
from turanpy.deep_learning import neural_network
from turanpy.deep_learning import initLayers

import random

import time

def main(X, layers, act):
    start = time.time()
    print(neural_network(X, layers, act))
    print(time.time() - start) #0.0005333423614501953

if __name__ == "__main__":

    # X = [
    #     [0.2, 1.1],
    #     [0.4, 0.9],
    #     [1.2, 0.1],
    #     [1.0, 0.2],
    #     [2.0, 1.5],
    #     [2.2, 1.7],
    # ] 
     
    random.seed(9)
    X = [[random.randint(1, 3) for _ in range(100)] for _ in range(100)]

    hidden_layers = 4
    number_of_classes = 2
    number_of_features = 3

    main(X, initLayers(X, hidden_layers, number_of_classes, number_of_features), activation.relu)