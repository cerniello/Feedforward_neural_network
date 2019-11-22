import numpy as np

#### MACROS:
n = 2        # dimensions

# Hyperparameters:
# chosen after GridSearch cross-validation (using K-Fold)

N = 30       # number of neurons
sigma = 1.8  # spread of the Gaussian function
rho = 1e-05  # regularization parameter
#### END_MACROS

def fun_MLP(X, omega):
    """
    It implements the Multi Layer Perceptron network with one hidden layer:
    Given the observations X and the vector parameter omega
    it returns the approximations of y

    Parameters:
    ------
    X: Pxn numpy array
        P observations of n-dimensional points (n=2): (X1, X2)
    omega: 1D numpy array
        It contains all the parameters in the order:
            v: Nx1-array: parameters from the hidden to the output layer
            W: Nxn Matrix: weights from input layer to hidden layer
            b: Nx1-array: bias from input layer to hidden layer
    Returns
    ------
    y_pred: 1xP array
        containing P predictions of the MLP function
    """
    
    # creating matrix X1 (X matrix plus array of ones)
    X1 = np.append(X, -1* np.ones((X.shape[0], 1)), axis=1)

    # extract param vectors from omega
    v = omega[0:N].reshape(1, N)
    W = omega[N:3*N].reshape(N, n)
    b = omega[3*N:].reshape(N, 1)
    
    # merge W with b
    Wb = np.append(W, b, axis=1)
    
    return np.dot(v, g_fun(np.dot(Wb, np.transpose(X1)), sigma))

def g_fun(T, sigma):
    num = np.exp(2*sigma*T) - 1
    den = np.exp(2*sigma*T) + 1
    return num / den

def fun_grad_MLP(omega, X_train, y_true):
    """
    Function which implements the gradient computation for
    the regularized loss function with respect to v and W and b parameters

    Parameters:
    -------
    omega: 1D numpy array
        It contains all the parameters in the order:
            v: Nx1-array: parameters from the hidden to the output layer
            W: Nxn Matrix: weights from input layer to hidden layer
            b: Nx1-array: bias from input layer to hidden layer
    X: Pxn numpy array
        P observations of n-dimensional points (n=2): (X1, X2)
    y_true: 1dim array
        true values of the function

    Returns:
    -------
    dE_dv, dE_dW, dE_db as 1dim array
    """
    
    # creating matrix X1 (X matrix plus array of ones)
    X1 = np.append(X_train, -1* np.ones((X_train.shape[0],1)), axis=1)


    # pick parameters from omega one
    v = omega[0:N].reshape(N,1)
    W = omega[N:3*N].reshape(N, n)
    b = omega[3*N:].reshape(N, 1)
    
    
    Wb = np.append(W,b, axis=1)
    
    ### compute partial derivatives in a "pythonic" way:
    
    # for the sake of clearness, let's define some variables
    # (MATRIX1) dot product between Wb and X1
    T = np.dot(Wb, np.transpose(X1)) # NxX_train.shape[0] matrix
    
    # derivative of g()
    g_der = 4 * sigma * np.exp(2 * sigma * T) / (np.exp(2 * sigma * T) + 1)**2

    dE_dv = 1 / X_train.shape[0] * np.dot(g_fun(T, sigma), np.transpose(fun_MLP(X_train, omega) - y_true)) + 2 * rho * v
    dE_db = 1 / X_train.shape[0] * np.dot(-1 * v * g_der, np.transpose(fun_MLP(X_train, omega) - y_true)) + 2 * rho * b
    
    # dealing with dE_dW
    mat1 = v * g_der * X1[:,0]
    mat1 = np.dot(mat1, np.transpose(fun_MLP(X_train, omega) - y_true))
    mat2 = v * g_der * X1[:,1]
    mat2 = np.dot(mat2, np.transpose(fun_MLP(X_train, omega) - y_true))

    fusion = np.append(mat1, mat2, axis=1)
    dE_dW = 1 / X_train.shape[0] * fusion + 2 * rho * W

    omega_gradient = np.concatenate((dE_dv, dE_dW.reshape(N*n,1), dE_db))
    return omega_gradient.reshape(-1)

def loss_MLP(omega, X, y_true):
    """
    Implement the regularized training error function of the MLP network

    Parameters:
    ------
    omega: 1D numpy array
        It contains all the parameters in the order:
            v: Nx1-array: parameters from the hidden to the output layer
            W: Nxn Matrix: weights from input layer to hidden layer
            b: Nx1-array: bias from input layer to hidden layer
    X: Pxn numpy array
        P observations of n-dimensional points (n=2): (X1, X2)
    y_true: true values of the function

    Returns:
    l: The regularized training function E(omega, sigma, rho)
    ------
    """
    y_pred = fun_MLP(X, omega)
    l = np.sum((y_pred - y_true)**2)/(X.shape[1] * X.shape[0]) + rho * np.linalg.norm(omega)**2
    return l

def MSE(y_true, y_pred):
    """
    Compute the Mean Squared Error from y_true and y_predicted
    """
    # reshape y's in order to do not have errors
    y_true = y_true.reshape(-1,)
    y_pred = y_pred.reshape(-1,)
    return np.mean(np.square(y_true - y_pred)) / 2