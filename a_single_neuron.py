from turanpy.classification import single_class_logistic_gradient_descent as gd
from turanpy.calculus import sigmoid
from turanpy.algebra import lm
from turanpy.classification import MSE_gradient_descent as msgd

import random 

def main(X, y):
    res = msgd(y, X, 0.01, 50000)
    for _ in range(5):
        h = random.randint(0, 10)
        print(f"Hours studied: {h}", sigmoid(lm(h, res[0], res[1])))

if __name__ == "__main__":
    X = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 10]
    y = [0,   0,  0,   0, 0,   0, 1,   0, 1,   0,  1,   1,  1,   1,  1,   1,  1,   1,  1,  1]

    main(X, y)
    # 1.4562492866712637 -5.4489052181944615
    # Hours studied: 6 0.9640353177716885
    # Hours studied: 10 0.9998898533457552
    # Hours studied: 5 0.862040397416545
    # Hours studied: 8 0.9979768797567548
    # Hours studied: 8 0.9979768797567548