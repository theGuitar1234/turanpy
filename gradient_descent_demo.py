X = [1, 2, 4, 8]
Y = [2, 4, 8, 16]

def gradient_descent(y, X, learning_rate, epochs):
    w = 0.0
    b = 0.0

    for _ in range(epochs):
        y_hat = [w*X[i] + b for i in range(len(X))]

        error = [y_hat[i] - y[i] for i in range(len(y_hat))]

        grad_w = 2*mean([error[i]*X[i] for i in range(len(error))])
        grad_b = 2*mean(error)

        w -= learning_rate*grad_w
        b -= learning_rate*grad_b
    
    return f"""
        Predicted weight: {w}
        Predicted bias: {int(b)}
    """

def mean(v):
    sum = 0.0
    for i in range(len(v)):
        sum += v[i]
    return sum/len(v)

print(gradient_descent(Y, X, 0.01, 50000))
# Predicted weight: 1.9999999999999973
# Predicted bias: 0
