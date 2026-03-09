from turanpy.algebra import multi_class_lm
from turanpy.calculus import softmax
from turanpy.deep_learning import activation

import random

def main(X, W_hidden, W_out, B_hidden, B_out):
    probs = []
    for i in range(len(X)):

        x = X[i]

        #Hidden Layer
        logits_hidden = []
        for j in range(len(W_hidden)):
            z = multi_class_lm(W_hidden[j], x, B_hidden[j])
            logits_hidden.append(z)
        
        a1 = []
        for k in logits_hidden:
            a1.append(activation.relu(k))
        
        #Output Layer
        logits_out = []
        for t in range(len(W_out)):
            z = multi_class_lm(W_out[t], a1, B_out[t])
            logits_out.append(z)
        
        probs.append(softmax(logits_out))

    print(f"Probabilities : {probs}")
    #Probabilities : [[0.5295055796403376, 0.3190525927366926, 0.1514418276229697]]

def initWeightsAndBias(rows, cols):
    W = [[random.uniform(-1, 1) for _ in range(cols)] for _ in range(rows)]
    b = [random.uniform(-1, 1) for _ in range(rows)]
    return W, b

if __name__ == "__main__":

    number_of_classes = 3
    number_of_features = 2

    hidden_units = 5

    W_hidden, B_hidden = initWeightsAndBias(hidden_units, number_of_features)
    W_out, B_out = initWeightsAndBias(number_of_classes, hidden_units)

    X = [[0.2, 1.1]]

    main(X, W_hidden, W_out, B_hidden, B_out)