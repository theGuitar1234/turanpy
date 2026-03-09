from ..classification.gradient_descent import multi_class_logistic_gradient_descent as gd

def OvR(X, y_multiclass, K, learning_rate, epochs):
    """Trains multi-class input data via 'One Vs the Rest'."""
    params = []
    for c in range(K):
        y_binary = [1 if yi == c else 0 for yi in y_multiclass]
        w, b = gd(y_binary, X, learning_rate, epochs)
        params.append((w, b))
    print(params)
    return params 

if __name__ == "__main__":
    pass