from turanpy.calculus import softmax
from turanpy.deep_learning import activation
from turanpy.algebra.linear_model import multi_class_lm
from turanpy.algebra.matrix_multiplication import mul
from turanpy.algebra.transpoze import transpoze
from turanpy.util import pretty_print
from turanpy.algebra.add_matrix import add

import random

import time

def main(X, layers, act):
    start = time.time()
    print(neural_network(X, layers, act))
    print(time.time() - start) #0.00026679039001464844

def neural_network(X, WB, act):
    H = X
    for layer_idx in range(len(WB)):
        W, b = WB[layer_idx]

        if layer_idx == len(WB) - 1:
            Z = forward_pass(H, W, b, None)
            probs = []
            for z in Z:
                probs.append(softmax(z))
            return probs

        H = forward_pass(H, W, b, act)

def forward_pass(X, W, B, act):
    WT = transpoze(W)
    Z = mul(X, WT)

    Z_biased = []
    for row in Z:
        Z_biased.append([row[j] + B[j] for j in range(len(B))])
    
    if act is not None:
        Z_biased = [[act(v) for v in row] for row in Z_biased]
    
    return Z_biased

def initLayers(X, hidden_layers, output_dim=None, start_width=None):
    n_in = len(X[0])

    if start_width is None:
        start_width = n_in

    WB = []
    width = start_width

    for _ in range(hidden_layers):
        n_out = width
        WB.append(initWeightsAndBias(n_out, n_in))
        n_in = n_out
        width *= 2

    if output_dim is not None:
        WB.append(initWeightsAndBias(output_dim, n_in))

    return WB

def initWeightsAndBias(rows, cols):
    W = [[random.uniform(-1, 1) for _ in range(cols)] for _ in range(rows)]
    b = [random.uniform(-1, 1) for _ in range(rows)]
    return W, b

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

    hidden_layers = 3
    number_of_classes = 2
    number_of_features = 2

    main(X, initLayers(X, hidden_layers, number_of_classes, number_of_features), activation.relu)