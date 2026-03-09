from turanpy import algebra
from turanpy import classification
from turanpy.util import console

def main(X, params, probs):
    
    for x in X:
        p = predict_proba_softmax(x, params)   # [P(Dog), P(Cat), P(Rabbit)]
        pred = predict_class(x, params)        # 0, 1, or 2
        print(f"x={x}  probs={p}  pred={pred}")

    print("max prob anywhere:", max(max(row) for row in probs))
    print("first sample probs:", probs[0])
    print("first sample predicted class:", predict_class(X_all[0], params))

def predict_proba_softmax(x, params):
    logits = [algebra.multi_class_lm(w, x, b) for (w, b) in params]
    return classification.softmax(logits)

def predict_class(x, params):
    p = predict_proba_softmax(x, params)
    return max(range(len(p)), key=lambda k: p[k])

if __name__ == "__main__":

    Dog = [
        [1.0, 1.0],
        [3.0, 10.0],
        [4.0, 10.0],
        [5.0, 25.0]
    ]

    Cat = [
        [1.0, 5.0],
        [2.0, 10.0],
        [4.0, 20.0],
        [5.0, 25.0]
    ]

    Rabbit = [
        [1.0, 5.0],
        [2.0, 10.0],
        [4.0, 24.0],
        [6.0, 25.0]
    ]

    X_test = [
        [2.5,  8.0],
        [4.5, 12.0],
        [5.5, 22.0],     
        [1.2,  2.0],
        [3.8, 24.5],     
        [6.5, 26.0],      
    ]

    X_all = Dog + Cat + Rabbit

    print(X_all)

    y_all = [0]*len(Dog) + [1]*len(Cat) + [2]*len(Rabbit)
    #[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]

    print(y_all)

    K = 3 # 3 classes


    params = classification.OvR(X_all, y_all, K=3, learning_rate=0.0001, epochs=50000)

    probs = [predict_proba_softmax(x, params) for x in X_all]

    main(X_test, params, probs)

    # x=[2.5, 8.0]  probs=[0.5318159315756557, 0.22896944085487256, 0.2392146275694717]  pred=0
    # x=[4.5, 12.0]  probs=[0.7865239960771635, 0.09886077207660601, 0.11461523184623035]  pred=0
    # x=[5.5, 22.0]  probs=[0.42057726067722717, 0.26355134222299104, 0.31587139709978174]  pred=0
    # x=[1.2, 2.0]  probs=[0.5748332352492785, 0.2142720205288955, 0.21089474422182594]  pred=0
    # x=[3.8, 24.5]  probs=[0.04967770871313056, 0.456468612559722, 0.4938536787271475]  pred=2
    # x=[6.5, 26.0]  probs=[0.4288802468698026, 0.253086094834156, 0.31803365829604135]  pred=0
    # max prob anywhere: 0.7835887277005654
    # first sample probs: [0.5865183988284476, 0.20931299335740508, 0.20416860781414736]
    # first sample predicted class: 0

