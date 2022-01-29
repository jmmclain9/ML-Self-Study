import numpy as np

def SVM(X, Y):

    w = np.zeros((X.Shape[0], 1))
    b = 0
    lr = 0.001
    num_epochs = 1000
    lambd = 1/num_epochs
    y = np.where(Y <= 0, -1, 1)
    pred = []

    for epoch in range(num_epochs):
        for j, prod in enumerate(X):
            prod =  (np.dot(X[j], w) - b)
            if (y[j] * prod >= 1):
               w = w + (2 * lr * lambd * w)
            else: 
                w = w + lr * (2 * lambd * w) - np.dot(X[j], y[j])
                b = lr * y[j] 
            
            pred.append(np.dot(X[j], w) - b)

    return w, b, pred
