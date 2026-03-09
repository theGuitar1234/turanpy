from turanpy.algebra import multi_class_lm
from turanpy.calculus import sigmoid
from turanpy.calculus import softmax

import random

def main(X, W, B):
    probs = []
    for i in range(len(X)):
        logits = []
        for j in range(len(W)):
            z = multi_class_lm(W[j], X[i], B[j])
            logits.append(z)
        probs.append(softmax(logits))
    print(f"Probabilities : {probs}")

if __name__ == "__main__":

    number_of_classes = 3
    number_of_features = 2

    W = [
        [random.uniform(0, 1) for _ in range(number_of_features)] for _ in range(number_of_classes)
    ]

    B = [random.uniform(0, 1) for _ in range(number_of_classes)]

    X = [
        [0.2, 1.1],
        [0.4, 0.9],
        [1.2, 0.1],
    ]

    main(X, W, B)
    # Probabilities : [[0.1956916380507029, 0.5321408129159054, 0.27216754903339174], [0.2160983723790893, 0.5208426258294703, 0.2630590017914405], [0.3123125439670525, 0.4645659192646663, 0.2231215367682811]]