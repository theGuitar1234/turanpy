from turanpy.algebra import multi_class_lm
from turanpy.calculus import softmax
from turanpy.deep_learning import activation

import random

def main(X, W, B, W_out, B_out):
    probs = []
    for s in range(len(X)):

        x = X[s]

        #Hidden Layers
        for i in range(len(W)):
            w = W[i]
            b = B[i]
            logits_hidden = []
            for j in range(len(w)):
                z = multi_class_lm(w[j], x, b[j])
                logits_hidden.append(z)
            a = []
            for k in logits_hidden:
                a.append(activation.relu(k))
            x = a
        
        #Output Layer
        logits_out = []
        for t in range(len(W_out)):
            z = multi_class_lm(W_out[t], x, B_out[t])
            logits_out.append(z)
        
        probs.append(softmax(logits_out))

    print(f"Probabilities : {probs}")
    #[[0.15922072962866934, 0.5287763114487045, 0.3120029589226262]]

def initWeightsAndBias(rows, cols):
    W = [[random.uniform(-1, 1) for _ in range(cols)] for _ in range(rows)]
    b = [random.uniform(-1, 1) for _ in range(rows)]
    return W, b

if __name__ == "__main__":

    number_of_classes = 3
    number_of_features = 2

    W_1, B_1 = initWeightsAndBias(2, number_of_features)
    W_2, B_2 = initWeightsAndBias(4, 2)
    W_3, B_3 = initWeightsAndBias(8, 4)
    W_out, B_out = initWeightsAndBias(number_of_classes, 8)

    W = [W_1, W_2, W_3]
    B = [B_1, B_2, B_3]

    X = [[0.2, 1.1]]

    main(X, W, B, W_out, B_out)