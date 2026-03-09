from turanpy.calculus import softmax
from turanpy.algebra import multi_class_lm

import random

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

def forward_pass(X, W, b, act):
    result = []
    for x in X:
        logits = []
        for j in range(len(W)):
            z = multi_class_lm(W[j], x, b[j])
            logits.append(z)
        if act is not None:
            logits = [act(v) for v in logits]
        result.append(logits)
    return result

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
    pass