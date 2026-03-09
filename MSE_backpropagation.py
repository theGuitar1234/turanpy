import random
import math


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def dsigmoid_from_output(a):
    return a * (1.0 - a)


def loss_one(y_hat, y):
    total = 0.0
    for k in range(len(y)):
        total += 0.5 * (y_hat[k] - y[k]) ** 2
    return total

def init_layers(n_inputs, n_hidden, n_outputs):
    W1 = [
        [random.uniform(-1.0, 1.0) for _ in range(n_inputs)]
        for _ in range(n_hidden)
    ]

    b1 = [random.uniform(-1.0, 1.0) for _ in range(n_hidden)]

    W2 = [
        [random.uniform(-1.0, 1.0) for _ in range(n_hidden)]
        for _ in range(n_outputs)
    ]

    b2 = [random.uniform(-1.0, 1.0) for _ in range(n_outputs)]

    return [W1, b1, W2, b2]


def forward_one(x, WB):
    W1, b1, W2, b2 = WB

    z1 = []
    a1 = []

    #Hidden Layer
    for j in range(len(W1)):
        total = b1[j]

        for i in range(len(x)):
            total += W1[j][i] * x[i]

        z1.append(total)
        a1.append(sigmoid(total))

    z2 = []
    a2 = []

    #Output Layer
    for k in range(len(W2)):
        total = b2[k]

        for j in range(len(a1)):
            total += W2[k][j] * a1[j]

        z2.append(total)
        a2.append(sigmoid(total))

    return [z1, a1, z2, a2]


def backward_one(x, y, WB, cache):
    W1, b1, W2, b2 = WB
    z1, a1, z2, a2 = cache

    delta2 = []

    for k in range(len(a2)):
        y_hat = a2[k]
        delta = (y_hat - y[k]) * dsigmoid_from_output(y_hat)
        delta2.append(delta)

    dW2 = [
        [0.0 for _ in range(len(a1))]
        for _ in range(len(W2))
    ]

    db2 = [0.0 for _ in range(len(W2))]

    for k in range(len(W2)):
        for j in range(len(a1)):
            dW2[k][j] = delta2[k] * a1[j]

        db2[k] = delta2[k]

    delta1 = []

    for j in range(len(a1)):
        back_signal = 0.0

        for k in range(len(W2)):
            back_signal += delta2[k] * W2[k][j]

        delta = back_signal * dsigmoid_from_output(a1[j])
        delta1.append(delta)

    dW1 = [
        [0.0 for _ in range(len(x))]
        for _ in range(len(W1))
    ]

    db1 = [0.0 for _ in range(len(W1))]

    for j in range(len(W1)):
        for i in range(len(x)):
            dW1[j][i] = delta1[j] * x[i]

        db1[j] = delta1[j]

    return [dW1, db1, dW2, db2]


def update_layers(WB, grads, lr):
    W1, b1, W2, b2 = WB
    dW1, db1, dW2, db2 = grads

    for j in range(len(W1)):
        for i in range(len(W1[j])):
            W1[j][i] -= lr * dW1[j][i]

        b1[j] -= lr * db1[j]

    for k in range(len(W2)):
        for j in range(len(W2[k])):
            W2[k][j] -= lr * dW2[k][j]

        b2[k] -= lr * db2[k]

def predict_one(x, WB):
    cache = forward_one(x, WB)
    return cache[3]


def train(X, Y, n_inputs, n_hidden, n_outputs, lr, epochs):
    WB = init_layers(n_inputs, n_hidden, n_outputs)

    for epoch in range(epochs):
        total_loss = 0.0

        for sample_index in range(len(X)):
            x = X[sample_index]
            y = Y[sample_index]

            cache = forward_one(x, WB)
            y_hat = cache[3]

            total_loss += loss_one(y_hat, y)

            grads = backward_one(x, y, WB, cache)
            update_layers(WB, grads, lr)

        # if epoch % 500 == 0:
        #     avg_loss = total_loss / len(X)
        #     print("epoch =", epoch, "avg_loss =", avg_loss)

    return WB


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

    X_test = [
        [0.25, 1.05],
        [0.35, 0.95],
        [0.45, 1.10],
        [0.55, 0.85],
        [0.65, 1.15],

        [1.10, 0.15],
        [1.25, 0.25],
        [1.35, 0.35],
        [1.45, 0.30],
        [1.55, 0.45],

        [1.95, 1.45],
        [2.05, 1.55],
        [2.15, 1.65],
        [2.25, 1.75],
        [2.35, 1.85],
    ]

    Y_test = [
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
    
    n_inputs = 2
    n_hidden = 4
    n_outputs = 1
    learning_rate = 0.5
    epochs = 3000

    WB = train(X, Y, n_inputs, n_hidden, n_outputs, learning_rate, epochs)

    print("\nPredicted Weights and Biases\n")
    print("Input Layer Weights : ", WB[0])
    print("Input Layer Biases: ", WB[1])
    print()
    print("Output Layer Weights : ", WB[2])
    print("Output Layer Biases : ", WB[3])

    print("\nPredictions:\n")

    for i in range(len(X)):
        y_hat = predict_one(X[i], WB)[0]
        predicted_class = 1 if y_hat >= 0.5 else 0

        print(
            "x =", X[i],
            "y =", Y[i][0],
            "y_hat =", round(y_hat, 4),
            "class =", predicted_class
        )
    
    print("\nTest Predictions:\n")

    for i in range(len(X_test)):
        y_hat = predict_one(X_test[i], WB)[0]
        predicted_class = 1 if y_hat >= 0.5 else 0

        print(
            "x =", X_test[i],
            "y =", Y_test[i][0],
            "y_hat =", round(y_hat, 4),
            "class =", predicted_class
        )
    
    correct = 0

    for i in range(len(X_test)):
        y_hat = predict_one(X_test[i], WB)[0]
        predicted_class = 1 if y_hat >= 0.5 else 0

        if predicted_class == Y_test[i][0]:
            correct += 1

    accuracy = correct / len(X_test)
    print("\nTest accuracy =", accuracy)
