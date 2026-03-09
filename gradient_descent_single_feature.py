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

    train_data = [50, 55, 70, 75, 85]
    input = [
        [1],
        [2],
        [4],
        [5],
        [6]
    ]

    w = 6.00877193
    b = 45.166666666666664

    main(train_data, input, w, b)
    # For the weight 6.00877193 and bias 44.166666666666664,
    # the accuracy of the linear model is: 5.629039701908306