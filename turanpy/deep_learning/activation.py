from turanpy.config import DEFAULTS

def sigmoid(z):
    return 1 / (1 + DEFAULTS.e**(-z))

def relu(z):
    return max(0, z)

def tanh(z):
    e = DEFAULTS.e
    return (e**z - e**(-z)) / (e**z + e**(-z))

def leaky_relu(z):
    return max(0.01 * z, z)

if __name__ == "__main__":
    pass