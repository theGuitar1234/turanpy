from turanpy import classification
from turanpy.config import DEFAULTS

def main(train_data, input, w, b):
    res = classification.MSEloss(train_data, input, w, b)
    print(f"""
        For the weight {w} and bias {b}, 
        the accuracy of the linear model is: {res}
    """)
    print(classification.single_feature_gradient_descent(6, 3, 0.1, DEFAULTS.epochs))

if __name__ == "__main__":

    train_data = [25, 50, 100, 200, 400]
    input = [
        [50],
        [100],
        [200],
        [400],
        [800]
    ]

    w = 2
    b = 0

    main(train_data, input, w, b)
    # For the weight 6.00877193 and bias 44.166666666666664,
    # the accuracy of the linear model is: 5.629039701908306