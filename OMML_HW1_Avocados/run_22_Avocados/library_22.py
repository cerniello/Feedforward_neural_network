import numpy as np

#### MACROS:
n = 2        # dimensions

# Hyperparameters:
# chosen after GridSearch cross-validation (using K-Fold, K = 5)

N = 30       # number of neurons
sigma = 0.8  # hyperparameter for the gaussian function
rho = 1e-05  # regularization parameter
#### END_MACROS


def fun_EL_RBF(X, v, c):
    """
    It implements the Radial Basis Function network with one hidden layer:
    Given the observations X and the vector parameters v and c
    it returns the approximations of y. Centers c are fixed

    Parameters:
    ------
    X: Pxn numpy array
        P observations of n-dimensional points (n=2): (X1, X2)

    v: Nx1-array
        parameters from the hidden to the output layer
    c: Nxn Matrix
        N centroids in the first hidden layer
    Returns
    ------
    y_pred: 1xP array
        containing P predictions of the RBF function
    """

    # reshaping v (for the final dot product)
    v = v.reshape(-1, 1)


    ### compute the RBF function in a 'pythonic' way

    # replicate c and X values as array
    c_array = np.tile(c.reshape(-1), X.shape[0])
    X_array = np.tile(X, N).reshape(-1)

    # create a tensor representing ||X-c||**2 matrix
    mat = ((c_array - X_array).reshape(X.shape[0], N, 2)) ** 2

    # sum (X[0] - c) with (X[1] - c) for each observation
    col = mat[:, :, 0] + mat[:, :, 1]
    col = np.exp(-col / (sigma ** 2))

    # now that we have the output of the hidden layer
    # make the dot product with v vector
    return np.dot(col, v).reshape(1, -1)


def fun_grad_EL_RBF(v, c, X, y_true):
    """
    Function which implements the gradient computation for the quadratic
    convex error function with respect to v (centroids are fixed)

    Parameters:
    -------
    v: Nx1-array:
        parameters from the hidden to the output layer
    c: Nxn Matrix:
        N centroids in the first hidden layer
    X: Pxn numpy array
        P observations of n-dimensional points (n=2): (X1, X2)
    y_true: 1dim array
        true values of the function

    Returns:
    -------
    dE_dv: 1 dim numpy array
    """

    v = v.reshape(-1, 1)
    ### dE_dv
    c_array = np.tile(c.reshape(-1), X.shape[0])
    X_array = np.tile(X, N).reshape(-1)

    # ||X-c||**2 matrix
    mat = ((c_array - X_array).reshape(X.shape[0], N, 2)) ** 2
    col = mat[:, :, 0] + mat[:, :, 1]

    # activation function
    col = np.exp(-col / (sigma ** 2))

    # dE_dv
    dE_dv = np.dot((fun_EL_RBF(X, v, c) - y_true), col) / X.shape[0] + 2 * rho * v.T
    dE_dv = dE_dv.reshape(-1, 1)

    return dE_dv.reshape(-1)


def loss_EL_RBF(v, c, X, y_true):
    """
    Implement the quadratic convex training error for the RBF
    (with centers c fixed)

    Parameters:
    ------
    v: Nx1-array:
        parameters from the hidden to the output layer
    c: Nxn Matrix:
        N centroids in the first hidden layer (should be Fixed)
    X: Pxn numpy array
        P observations of n-dimensional points (n=2): (X1, X2)
    y_true: true values of the function

    Returns:
    l: the quadratic convex training error for the P observations
    ------
    """

    y_pred = fun_EL_RBF(X, v, c)
    l = np.sum((y_pred - y_true)**2)/(2 * X.shape[0]) + rho * np.linalg.norm(v)**2
    return l


def MSE(y_true, y_pred):
    """
    Compute the Mean Squared Error from y_true and y_predicted
    """
    # reshape y's in order to do not have errors
    y_true = y_true.reshape(-1,)
    y_pred = y_pred.reshape(-1,)
    return np.mean(np.square(y_true - y_pred))