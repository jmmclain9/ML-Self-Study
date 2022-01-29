import numpy as np

def forward_backward_pass(w, b, X, Y):

    #Initialize params
    w = np.zeros((X.shape[1], 1)) 
    b = 0
    m = X.shape[1] #get total number of examples from columns of X

    #Forward pass to get the cost
    fx = 1 / (1 + np.exp((np.dot(w.T, X) + b)))
    cost = np.sum((Y * np.log(fx) + (1-Y) * np.log(1-fx)))

    #Backpropagation for updating parameters
    dw = (np.dot(X, (fx-Y).T)) / m
    db = (np.sum(fx-Y)) / m

    return dw, db, cost

def optimize_cost(w, b, X, Y, num_epochs, lr):

    costs = [] # initialize list to collect cost

    for i in range(num_epochs):

        #Calculate the derivatives and current cost 
        dw, db, cost = forward_backward_pass(w, b, X, Y)

        #Update parameters based on learning rates, derivatives
        w = w - (lr * dw)
        b = b - (lr * db)

        # Record cost every 50 iterations
        if i % 50 == 0:
            costs.append(cost)

        return w, dw, b, db, costs

def predictions(w, b, X):

    m = X.shape[1] #get total number of examples from columns of X
    Y_pred = np.zeros((1, m)) #one prediction for each example

    # Compute probabilities
    fx = 1 / (1 + np.exp((np.dot(w.T, X) + b)))

    # Evaluate binary prediction for each probability
    for i in range(fx.shape[1]):
        if fx[0,i] >= 0.5:
            Y_pred[0,i] = 1
        else:
            Y_pred[0,i] = 0

    return Y_pred


def logit_model(X_train, Y_train, X_test, num_epochs = 1000, lr=0.2):
    # Initialize empty params for model
    w = np.zeros((X_train.shape[0], 1))
    b = 0

    # Compute costs, updated params, derivatives
    w, dw, b, db, costs = optimize_cost(w, b, X_train, Y_train, num_epochs, lr)

    # Get predictions 
    Y_pred_train = predictions(w, b, X_train)
    Y_pred_test = predictions(w, b, X_test)
    

    return w, dw, b, db, costs, Y_pred_train, Y_pred_test
