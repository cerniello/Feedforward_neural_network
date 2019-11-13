import numpy as np

#### MACROS:
n = 2        # dimensions

# Hyperparameters:
# chosen after GridSearch cross-validation (using K-Fold)

N = 30       # number of neurons
sigma = 1.8  # hyperparameter for the activation function
rho = 1e-05  # regularization parameter
#### END_MACROS

def g_fun(T, sigma):
    """
    Activation function of the hidden layer

    Parameters:
    ------
    T:  NxP array
        resultant matrix after the linear combination of the input
        components with their weights (plus the bias) for all the 
        observations
    sigma: hyperparameter
    Returns:
    ------
    NxP matrix
        the same matrix after the transformation 
    """
    
    num = np.exp(2*sigma*T) - 1
    den = np.exp(2*sigma*T) + 1
    return num / den

def fun_EL_MLP(X, v, W, b):
    """
    Extreme Learning procedure
    MLP function

    Parameters:
    ------
    X: Pxn numpy array
        P observations of n-dimensional points (n=2): (X1, X2)
    v: 1 dim numpy array
        parameters from the hidden to the output layer
    W:  N,n numpy array
        matrix weights of the first hidden layer
    b: Nx1 numpy array
        bias vector for the N units of the hiddel layer

    Returns
    ------
    y_pred: 1xP numpy array
        containing P predictions of the EL_MLP function

    """


    # merge W with b
    Wb = np.append(W, b, axis=1)

    # creating matrix X1 (X matrix plus array of ones)
    X1 = np.append(X, -1 * np.ones((X.shape[0], 1)), axis=1)

    y_true = np.dot(v.T, g_fun(np.dot(Wb, np.transpose(X1)), sigma))
    return y_true.reshape(1, -1)


def fun_grad_EL_MLP(v, W, b, X_train, y_true):
    """
    Function which implement the gradient computation for
    the regularized loss function with respect to only v (W and b are fixed)

    Parameters:
    -------
    v: Nx1 numpy array:
        parameters from the hidden to the output layer
    W: Nxn numpy array
        Matrix weights of the first hidden layer
    b: Nx1 numpy array
        bias vector for the N units of the hidden layer
    X: Pxn numpy array
        P observations of n-dimensional points (n=2): (X1, X2)
    y_true: 1dim array
        true values of the function for the p's observations

    Returns:
    -------
    dE_dv as 1dim numpy array
    """

    Wb = np.append(W, b, axis=1)

    # creating matrix X1 (X matrix plus array of ones)
    X1 = np.append(X_train, -1 * np.ones((X_train.shape[0], 1)), axis=1)

    # pick parameters from omega one
    v = v.reshape(N, 1)

    ### compute partial derivatives in a "pythonic" way:

    # for the sake of clearness, let's define some variables
    # (MATRIX1) dot product between Wb and X1
    T = np.dot(Wb, np.transpose(X1))  # NxX_train.shape[0] matrix

    # derivative of g()
    # g_der = 4 * sigma * np.exp(2 * sigma * T) / (np.exp(2 * sigma * T) + 1)**2


    dE_dv = 1 / X_train.shape[0]
    dE_dv *= np.dot(g_fun(T, sigma), np.transpose(fun_EL_MLP(X_train, v, W, b) - y_true))
    dE_dv += 2 * rho * v

    return dE_dv.reshape(-1)

def loss_EL_MLP(v, W, b, X, y_true):
    """
    Implement the quadratic convex training error for the MLP
    (W and b are fixed)

    Parameters:
    -------
    v: Nx1 numpy array:
        parameters from the hidden to the output layer
    W: Nxn numpy array
        Matrix weights of the first hidden layer
    b: Nx1 numpy array
        bias vector for the N units of the hidden layer
    X: Pxn numpy array
        P observations of n-dimensional points (n=2): (X1, X2)
    y_true: 1dim array
        true values of the function for the p's observations

    Returns:
    -------
    the quadratic convex training error for the P observations
    """
    y_pred = fun_EL_MLP(X, v, W, b)
    l = np.sum((y_pred - y_true) ** 2) / (2 * X.shape[0]) + rho * np.linalg.norm(v) ** 2
    return l



def MSE(y_true, y_pred):
    """
    Compute the Mean Squared Error from y_true and y_predicted
    """
    # reshape y's in order to do not have errors
    y_true = y_true.reshape(-1,)
    y_pred = y_pred.reshape(-1,)
    return np.mean(np.square(y_true - y_pred)) / 2

    