import numpy as np

#### MACROS:
n = 2        # dimensions

# Hyperparameters:
# chosen after GridSearch cross-validation (using K-Fold)

N = 30       # number of neurons
sigma = 0.8  # spread of the Gaussian function
rho = 1e-05  # regularization parameter
#### END_MACROS


def fun_RBF(X, omega):
    """
    It implements the Radial Basis Function network with one hidden layer:
    Given the observations X and the vector parameter omega
    it returns the approximations of y

    Parameters:
    ------
    X: Pxn numpy array
        P observations of n-dimensional points (n=2): (X1, X2)
    omega: 1D numpy array
        It contains all the parameters in the order:
            v: Nx1-array: parameters from the hidden to the output layer
            c: Nxn Matrix: N centroids in the first hidden layer
    Returns
    ------
    y_pred: 1xP array
        containing P predictions of the RBF function
    """

    # unpack the parameters from omega
    v = omega[0:N].reshape(N, 1)
    c = omega[N:].reshape(N, n)

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



def fun_grad_RBF(omega, X, y_true):
    """
    Function which implements the gradient computation for
    the regularized loss function with respect to v and c vector parameters

    Parameters:
    -------
    omega: 1D numpy array
        It contains all the parameters in the order:
            v: Nx1-array: parameters from the hidden to the output layer
            c: Nxn Matrix: N centroids in the first hidden layer
    X: Pxn numpy array
        P observations of n-dimensional points (n=2): (X1, X2)
    y_true: 1dim array
        true values of the function

    Returns:
    -------
    dE_dv, dE_dc as 1dim array
    """
    v = omega[0:N].reshape(N, 1)
    c = omega[N:].reshape(N, n)

    #### dE_dv
    # create a tensor representing X - c
    c_array = np.tile(c.reshape(-1), X.shape[0])
    X_array = np.tile(X, N).reshape(-1)

    # create a tensor representing ||X-c||**2 matrix
    mat = ((c_array - X_array).reshape(X.shape[0], N, 2)) ** 2
    col = mat[:, :, 0] + mat[:, :, 1]

    # activation function
    col = np.exp(-col / (sigma ** 2))

    # dE_dv
    dE_dv = np.dot((fun_RBF(X, omega) - y_true), col) / X.shape[0] + 2 * rho * v.T
    dE_dv = dE_dv.reshape(-1, 1)

    #### dE_dc
    # mat1 and mat2 are matrices that correspond to calcuations 
    # performed on the first and second components of X respectively
    
    mat1 = (-(c_array - X_array)).reshape(X.shape[0], N, 2)
    mat1 = mat1[:, :, 0]
    mat1 = 2 * (col * v.T * mat1) / (sigma ** 2)
    mat1 = np.dot((fun_RBF(X, omega) - y_true), mat1) / X.shape[0]

    mat2 = (-(c_array - X_array)).reshape(X.shape[0], N, 2)
    mat2 = mat2[:, :, 1]
    mat2 = 2 * (col * v.T * mat2) / (sigma ** 2)
    mat2 = np.dot((fun_RBF(X, omega) - y_true), mat2) / X.shape[0]

    # now merge the results
    fusion = np.append(mat1.T, mat2.T, axis=1)

    # dE_dc
    dE_dc = fusion + 2 * rho * c

    return np.concatenate((dE_dv.reshape(1, -1), dE_dc.reshape(1, -1)), axis=1).reshape(-1)



def loss_RBF(omega, X, y_true):
    """
    Implement the regularized training error function of the RBF network

    Parameters:
    ------
    omega: 1D numpy array
        It contains all the parameters in the order:
            v: Nx1-array: parameters from the hidden to the output layer
            c: Nxn Matrix: N centroids in the first hidden layer
    X: Pxn numpy array
        P observations of n-dimensional points (n=2): (X1, X2)
    y_true: true values of the function

    Returns:
    l: The regularized training function E(omega, sigma, rho)
    ------
    """
    y_pred = fun_RBF(X, omega)

    # objective function
    l = np.sum((y_pred - y_true)**2)/(2 * X.shape[0]) + rho * np.linalg.norm(omega)**2
    return l

def MSE(y_true, y_pred):
    """
    Compute the Mean Squared Error from y_true and y_predicted
    """
    # reshape y's in order to do not have errors
    y_true = y_true.reshape(-1,)
    y_pred = y_pred.reshape(-1,)
    return np.mean(np.square(y_true - y_pred)) / 2


