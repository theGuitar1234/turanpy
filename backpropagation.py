import random
import math
import copy

from enum import Enum

class DataAugmentation(Enum):
    JITTER_NOISE = 1
    SAME_CLASS_INTERPOLATION = 2
    MEASUREMENT_NOISE = 3

def sigmoid(z):
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    else:
        return math.exp(z) / (1 + math.exp(z))

def dsigmoid_from_output(a):
    return a * (1 - a)

def relu(z):
    return z if z > 0 else 0.0

def drelu_from_output(a):
    return 1.0 if a > 0 else 0.0

def dot_product(a, b):
    total = 0.0
    for i in range(len(a)):
        total += a[i] * b[i]
    return total

def linear_model(w, x, b):
    return dot_product(w, x)+b

def loss(y_hat, y):
    eps = 1e-15
    p = y_hat[0]
    y = y[0]

    if p < eps:
        p = eps
    elif p > 1.0 - eps:
        p = 1.0 - eps

    return -(y * math.log(p) + (1 - y) * math.log(1 - p))

def output_delta_binary_bce(y_hat, y):
    return y_hat - y

def initWeightsAndBias(rows, cols):
    W = [[random.uniform(-1, 1) for _ in range(cols)] for _ in range(rows)]
    b = [random.uniform(-1, 1) for _ in range(rows)]
    return W, b

def sumWeightSquares(WB):
    total = 0.0
    for l in range(len(WB)):
        W, b = WB[l]
        for row in range(len(W)):
            for col in range(len(W[row])):
                total += W[row][col] * W[row][col]
    return total

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

def bernoulli(size, p):
    q = 1 - p

    rand = [random.uniform(0, 1) for _ in range(size)]
    mask = [int(r < q) for r in rand]

    return mask

def forward_pass(x, WB, hidden_act, output_act, training_mode=False, drop_out_rate=0.0):
    Z = []
    A = [x]
    M = []

    current_input = x
    keep_prob = 1.0 - drop_out_rate

    for l in range(len(WB)):
        W, b = WB[l]

        current_z = []
        for j in range(len(W)):
            current_z.append(linear_model(W[j], current_input, b[j]))

        if l == len(WB) - 1:
            current_a = [output_act(z) for z in current_z]
        else:
            current_a = [hidden_act(z) for z in current_z]
            if training_mode and drop_out_rate > 0.0:
                mask = bernoulli(len(current_a), drop_out_rate)
                current_a = [mask[j] * current_a[j] / keep_prob for j in range(len(current_a))]
                M.append(mask)
            else:
                M.append(None)

        Z.append(current_z)
        A.append(current_a)

        current_input = current_a

    return Z, A, M

def backward_pass(D, A, M, WB, dhidden_act_from_output, training_mode=False, drop_out_rate=0.0):
    for l in range(len(WB) - 2, -1, -1):
        D[l] = []

        for j in range(len(WB[l][0])):
            back_signal = 0.0

            for k in range(len(D[l + 1])):
                back_signal += D[l + 1][k] * WB[l + 1][0][k][j]

            delta = back_signal * dhidden_act_from_output(A[l + 1][j])
            D[l].append(delta)
    
    if training_mode and drop_out_rate > 0.0:
        keep_prob = 1.0 - drop_out_rate
        for i in range(len(D) - 1):
            if M[i] is not None:
                d = D[i]
                m = M[i]
                D[i] = [m[j] * d[j] / keep_prob for j in range(len(d))]

    grad_W = [None] * len(WB)
    grad_b = [None] * len(WB)

    for l in range(len(WB)):
        grad_W[l] = []
        grad_b[l] = []

        for row in range(len(WB[l][0])):
            current_row = []

            for col in range(len(A[l])):
                current_row.append(D[l][row] * A[l][col])

            grad_W[l].append(current_row)
            grad_b[l].append(D[l][row])
    
    return grad_W, grad_b

def update_parameters(WB, grad_W, grad_b, learning_rate):
    for l in range(len(WB)):
        for row in range(len(WB[l][0])):
            for col in range(len(WB[l][0][row])):
                WB[l][0][row][col] -= learning_rate * grad_W[l][row][col]

        for row in range(len(WB[l][1])):
            WB[l][1][row] -= learning_rate * grad_b[l][row]

def train_validation_split(X, Y, val_ratio=0.2, seed=42):
    X_shuffled, Y_shuffled = shuffle_dataset(X, Y, seed)

    val_size = max(1, int(len(X_shuffled) * val_ratio))

    X_val = X_shuffled[:val_size]
    Y_val = Y_shuffled[:val_size]

    X_train = X_shuffled[val_size:]
    Y_train = Y_shuffled[val_size:]

    return X_train, Y_train, X_val, Y_val

def shuffle_dataset(X, Y, seed=42):
    pairs = list(zip(X, Y))
    random.Random(seed).shuffle(pairs)
    X_shuffled = [x for x, _ in pairs]
    Y_shuffled = [y for _, y in pairs]
    return X_shuffled, Y_shuffled

def evaluate_dataset(X, Y, WB, hidden_act, output_act):
    total_loss = 0.0
    correct = 0

    for i in range(len(X)):
        y_hat = predict_one(X[i], WB, hidden_act, output_act)
        total_loss += loss(y_hat, Y[i])

        predicted_class = 1 if y_hat[0] >= 0.5 else 0
        if predicted_class == Y[i][0]:
            correct += 1

    avg_loss = total_loss / len(X)
    accuracy = correct / len(X)
    return avg_loss, accuracy

def data_augmentation(train_pairs, augmentation_type):
    augmented_pairs = [([v for v in x], [t for t in y]) for x, y in train_pairs]

    match augmentation_type:
        case DataAugmentation.JITTER_NOISE:
            jitter_strength = 0.03
            for x, y in augmented_pairs:
                for j in range(len(x)):
                    x[j] += random.uniform(-jitter_strength, jitter_strength)
        case DataAugmentation.MEASUREMENT_NOISE:
            measurement_strength = 0.02
            for x, y in augmented_pairs:
                for j in range(len(x)):
                    x[j] *= (1.0 + random.uniform(-measurement_strength, measurement_strength))
                    x[j] = max(0.0, x[j])
        case DataAugmentation.SAME_CLASS_INTERPOLATION:
            same_class_groups = {}

            for x, y in train_pairs:
                label = y[0]
                if label not in same_class_groups:
                    same_class_groups[label] = []
                same_class_groups[label].append((x, y))

            synthetic_pairs = []

            for label, samples in same_class_groups.items():
                for _ in range(len(samples) // 2):
                    if len(samples) >= 2:
                        (x1, y1), (x2, y2) = random.sample(samples, 2)
                    else:
                        x1, y1 = samples[0]
                        x2, y2 = samples[0]

                    alpha = random.uniform(0.2, 0.8)

                    x_new = [
                        alpha * x1[j] + (1.0 - alpha) * x2[j]
                        for j in range(len(x1))
                    ]

                    synthetic_pairs.append((x_new, y1[:]))

            augmented_pairs.extend(synthetic_pairs)
        case _:
            raise ValueError("Unknown Data Augmentation Type")
    return augmented_pairs


def train(X, Y, X_val, Y_val, hidden_layers, output_dim, start_width, learning_rate, hidden_act, output_act, dact_from_output, dloss_output_delta, epochs, batch_size, l2_lambda=0, training_mode=False, drop_out_rate=0.0, data_augmentation_type=None):
    if not (0.0 <= drop_out_rate < 1.0):
        raise ValueError("drop_out_rate must be in [0.0, 1.0)")

    WB = initLayers(X, hidden_layers, output_dim, start_width)

    best_val_loss = float("inf")
    best_WB = None
    patience = 400
    patience_counter = 0

    current_lr = learning_rate
    initial_lr = learning_rate
    decay_factor = 0.5
    step_size = 500

    for epoch in range(epochs):
        train_data_loss = 0.0
        train_reg_loss = 0.0
        train_total_loss = train_data_loss + train_reg_loss

        train_pairs = list(zip(X, Y))
        random.shuffle(train_pairs)

        if data_augmentation_type is not None:
            train_pairs = data_augmentation(train_pairs, data_augmentation_type)

        for batch_start in range(0, len(train_pairs), batch_size):
            batch = train_pairs[batch_start:batch_start + batch_size]

            sum_gradW = [[[0 for _ in range(len(j))] for j in i[0]] for i in WB]
            sum_gradb = [[0 for _ in range(len(i[1]))] for i in WB]

            for x, y in batch:
                Z, A, M = forward_pass(x, WB, hidden_act, output_act, training_mode, drop_out_rate)

                y_hat = A[-1]

                bce_loss = loss(y_hat, y)
                train_data_loss += bce_loss

                D = [None] * len(WB)
                D[-1] = [dloss_output_delta(y_hat[i], y[i]) for i in range(len(y_hat))]

                grad_W, grad_b = backward_pass(D, A, M, WB, dact_from_output, training_mode, drop_out_rate)

                for i in range(len(sum_gradW)):
                    for j in range(len(sum_gradW[i])):
                        for k in range(len(sum_gradW[i][j])):
                            sum_gradW[i][j][k] += grad_W[i][j][k] / len(batch)
                for i in range(len(sum_gradb)):
                    for j in range(len(sum_gradb[i])):
                        sum_gradb[i][j] += grad_b[i][j] / len(batch)
            
            for l in range(len(WB)):
                W, b = WB[l]
                for row in range(len(W)):
                    for col in range(len(W[row])):
                        sum_gradW[l][row][col] += l2_lambda * W[row][col]
            update_parameters(WB, sum_gradW, sum_gradb, current_lr)
      
        train_data_loss = train_data_loss / len(train_pairs)
        train_reg_loss = (l2_lambda / 2) * sumWeightSquares(WB)
        train_total_loss = train_data_loss + train_reg_loss

        val_loss, val_acc = evaluate_dataset(X_val, Y_val, WB, hidden_act, output_act)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_WB = copy.deepcopy(WB)
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch > 0 and epoch % 500 == 0:
            current_lr = initial_lr * (decay_factor ** (epoch // step_size))

        if epoch % 500 == 0:
            print(
                "epoch =", epoch,
                "train_data_loss =", round(train_data_loss, 6),
                "train_reg_loss =", round(train_reg_loss, 6),
                "train_total_loss =", round(train_total_loss, 6),
                "val_loss =", round(val_loss, 6),
                "val_acc =", round(val_acc, 4)
            )
            
        if patience_counter >= patience:
            print("early stopping at epoch", epoch)
            break
    
    final_WB = best_WB if best_WB is not None else WB
        
    x1 = X[0]
    x2 = X[-1]
    print(f"\nFinal Predictions: {forward_pass(x1, final_WB, hidden_act, output_act)[1][-1]}")
    print(f"Final Predictions: {forward_pass(x2, final_WB, hidden_act, output_act)[1][-1]}")

    return final_WB

def predict_one(x, WB, hidden_act, output_act):
    cache = forward_pass(x, WB, hidden_act, output_act)
    return cache[1][-1]

if __name__ == "__main__":
    X_full = [
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

    Y_full = [
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

    X_train, Y_train, X_val, Y_val = train_validation_split(X_full, Y_full, val_ratio=0.2, seed=42)

    print("train size =", len(X_train))
    print("validation size =", len(X_val))
    print("test size =", len(X_test))
    print()

    WB = train(
        X_train, Y_train,
        X_val, Y_val,
        hidden_layers=3,
        output_dim=1,
        start_width=2,
        learning_rate=0.05,
        hidden_act=relu,
        output_act=sigmoid,
        dact_from_output=drelu_from_output,
        dloss_output_delta=output_delta_binary_bce,
        epochs=3000,
        batch_size=4,
        l2_lambda=0.03,
        training_mode=True,
        drop_out_rate=0.2,
        data_augmentation_type=None
    )

    val_loss, val_acc = evaluate_dataset(X_val, Y_val, WB, relu, sigmoid)
    test_loss, test_acc = evaluate_dataset(X_test, Y_test, WB, relu, sigmoid)

    print("\nFinal validation loss =", round(val_loss, 6), "validation accuracy =", round(val_acc, 4))
    print("Final test loss =", round(test_loss, 6), "test accuracy =", round(test_acc, 4))

    print("\nPredicted Weights and Biases\n")
    print("Input Layer Weights : ", WB[0][0])
    print("Input Layer Biases: ", WB[0][1])
    print()
    print("Output Layer Weights : ", WB[-1][0])
    print("Output Layer Biases : ", WB[-1][1])

    print("\nPredictions:\n")

    for i in range(len(X_full)):
        y_hat = predict_one(X_full[i], WB, relu, sigmoid)[0]
        predicted_class = 1 if y_hat >= 0.5 else 0

        print(
            "x =", X_full[i],
            "y =", Y_full[i][0],
            "y_hat =", round(y_hat, 4),
            "class =", predicted_class
        )
    
    print("\nTest Predictions:\n")

    for i in range(len(X_test)):
        y_hat = predict_one(X_test[i], WB, relu, sigmoid)[0]
        predicted_class = 1 if y_hat >= 0.5 else 0

        print(
            "x =", X_test[i],
            "y =", Y_test[i][0],
            "y_hat =", round(y_hat, 4),
            "class =", predicted_class
        )
    
    correct = 0

    for i in range(len(X_test)):
        y_hat = predict_one(X_test[i], WB, relu, sigmoid)[0]
        predicted_class = 1 if y_hat >= 0.5 else 0

        if predicted_class == Y_test[i][0]:
            correct += 1

    accuracy = correct / len(X_test)
    print("\nTest accuracy =", accuracy)