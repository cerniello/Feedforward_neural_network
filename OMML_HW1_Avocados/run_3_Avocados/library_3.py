import numpy as np

# Hyperparameters:
# chosen after GridSearch cross-validation (using K-Fold)

N = 30       # number of neurons
sigma = 1.8  # spread of the Gaussian function
rho = 1e-05  # regularization parameter
n = 2
#### END_MACROS

def convex_fun(X, v, W, b):
    """
    It implements the Multi Layer Perceptron network with one hidden layer
    which will be used in the convex optimization step:
    Given the observations X and the vector parameter omega
    it returns the approximations of y

    Parameters:
    ------
    X: Pxn numpy array
        P observations of n-dimensional points (n=2): (X1, X2)
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
    Wb = np.append(W, b, axis=1)

    return np.dot(v.T, g_fun(np.dot(Wb, np.transpose(X1)), sigma))

def g_fun(T, sigma):
    num = np.exp(2*sigma*T) - 1
    den = np.exp(2*sigma*T) + 1
    return num / den

def convex_Grad(v, X_train, y_true, W, b):
    """
    Function which implements the gradient computation for
    the regularized loss function with respect to v
    in the convex optimization step

    Parameters:
    -------
    v: Nx1-array: parameters from the hidden to the output layer
    X_train: Pxn numpy array
        P observations of n-dimensional points (n=2): (X1, X2)
    y_true: 1dim array
        true values of the function
    W: Nxn Matrix: weights from input layer to hidden layer
    b: Nx1-array: bias from input layer to hidden layer

    Returns:
    -------
    dE_dv as 1dim array
    """
    
    # merge W with b
    Wb = np.append(W, b, axis=1)
    
    # creating matrix X1 (X matrix plus array of ones)
    X1 = np.append(X_train, -1* np.ones((X_train.shape[0],1)), axis=1)

    # pick parameters from o
    v = v.reshape(N,1)
    omega = v.reshape(1,N)
        
    ### compute partial derivatives in a "pythonic" way:
    
    # for the sake of clearness, let's define some variables
    # (MATRIX1) dot product between Wb and X1
    T = np.dot(Wb, np.transpose(X1)) # NxX_train.shape[0] matrix
    
    # derivative of g()
    #g_der = 4 * sigma * np.exp(2 * sigma * T) / (np.exp(2 * sigma * T) + 1)**2

    dE_dv = 1 / X_train.shape[0] * np.dot(g_fun(T, sigma), \
                                          np.transpose(convex_fun(X_train, v, W, b) - y_true)) + 2 * rho * v

    return dE_dv.reshape(-1)

def convex_loss(v, X, y_true, W, b):
    """
    Implement the regularized training error function of the MLP network
    for the convex optimization step

    Parameters:
    ------
    v: Nx1-array: parameters from the hidden to the output layer
    X: Pxn numpy array
       P observations of n-dimensional points (n=2): (X1, X2)
    y_true: true values of the function
    W: Nxn Matrix: weights from input layer to hidden layer
    b: Nx1-array: bias from input layer to hidden layer

    Returns:
    l: The regularized training function E(omega, sigma, rho)
    ------
    """
    y_pred = convex_fun(X, v, W, b)
    l = np.sum((y_pred - y_true)**2)/(2 * X.shape[0]) + rho * np.linalg.norm(v)**2
    return l


def non_convex_loss(omega, X, y_true, v):
    """
    Implement the regularized training error function of the MLP network
    for the convex optimization step

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
    v: Nx1-array: parameters from the hidden to the output layer

    Returns:
    l: The regularized training function E(omega, sigma, rho)
    ------
    """
    y_pred = non_convex_fun(X, omega, v)
    l = np.sum((y_pred - y_true)**2)/(2 * X.shape[0]) + rho * np.linalg.norm(omega)**2
    return l


def non_convex_fun(X, omega, v):  
    """
    It implements the Multi Layer Perceptron network with one hidden layer
    which will be used in the non-convex optimization step:
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
    v: Nx1-array: parameters from the hidden to the output layer
    Returns
    ------
    y_pred: 1xP array
        containing P predictions of the MLP function
    """
    # creating matrix X1 (X matrix plus array of ones)
    X1 = np.append(X, -1* np.ones((X.shape[0], 1)), axis=1)

    # extract param vectors from omega
    W = omega[0:n*N].reshape(N, n)
    b = omega[n*N:].reshape(N, 1)
    
    # merge W with b
    Wb = np.append(W, b, axis=1)
    return np.dot(v.T, g_fun(np.dot(Wb, np.transpose(X1)), sigma))

def non_convex_Grad(omega, X_train, y_true, v):
    """
    Function which implements the gradient computation for
    the regularized loss function with respect to W and b parameters
    in the non-convex optimization step

    Parameters:
    -------
    omega: 1D numpy array
        It contains all the parameters in the order:
            v: Nx1-array: parameters from the hidden to the output layer
            W: Nxn Matrix: weights from input layer to hidden layer
            b: Nx1-array: bias from input layer to hidden layer
    X_train: Pxn numpy array
        P observations of n-dimensional points (n=2): (X1, X2)
    y_true: 1dim array
        true values of the function
    v: Nx1-array: parameters from the hidden to the output layer

    Returns:
    -------
    dE_dW, dE_db as 1dim array
    """
    
    # creating matrix X1 (X matrix plus array of ones)
    X1 = np.append(X_train, -1* np.ones((X_train.shape[0],1)), axis=1)

    # pick parameters from omega one
    v = v.reshape(N,1)
    W = omega[0:n*N].reshape(N, n)
    b = omega[n*N:].reshape(N, 1)
    
    
    Wb = np.append(W,b, axis=1)
    
    ### compute partial derivatives in a "pythonic" way:
    
    # for the sake of clearness, let's define some variables
    # (MATRIX1) dot product between Wb and X1
    T = np.dot(Wb, np.transpose(X1)) # NxX_train.shape[0] matrix
    
    # derivative of g()
    g_der = 4 * sigma * np.exp(2 * sigma * T) / (np.exp(2 * sigma * T) + 1)**2

    dE_db = 1 / X_train.shape[0] * np.dot(-1 * v * g_der, \
                                          np.transpose(non_convex_fun(X_train, omega, v) - y_true)) + 2 * rho * b
    
    # dealing with dE_dW
    mat1 = v * g_der * X1[:,0]
    mat1 = np.dot(mat1, np.transpose(non_convex_fun(X_train, omega, v) - y_true))
    mat2 = v * g_der * X1[:,1]
    mat2 = np.dot(mat2, np.transpose(non_convex_fun(X_train, omega, v) - y_true))

    fusion = np.append(mat1, mat2, axis=1)
    dE_dW = 1 / X_train.shape[0] * fusion + 2 * rho * W

    omega_gradient = np.concatenate((dE_dW.reshape(N*n,1), dE_db))
    return omega_gradient.reshape(-1)

def MLP_fun(X, omega):
    """
    It implements the Multi Layer Perceptron network with one hidden layer
    Given the observations X and the vector parameter omega
    it returns the approximations of y

    Parameters:
    ------
    X: Pxn numpy array
        P observations of n-dimensional points (n=2): (X1, X2)
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
    v = omega[0:N].reshape(N, 1)
    W = omega[N:3*N].reshape(N, n)
    b = omega[3*N:].reshape(N,1)
    
    # merge W with b
    Wb = np.append(W, b, axis=1)
    
    return np.dot(v.T, g_fun(np.dot(Wb, np.transpose(X1)), sigma))

def MLP_loss(omega, X, y_true):
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
    y_pred = MLP_fun(X, omega)
    l = np.sum((y_pred - y_true)**2)/(2 * X.shape[0]) + rho * np.linalg.norm(omega)**2
    return l

def stopping_criteria(X_train, y_train, v, W, b, threshold=[1e-8, 1e-3]):  
    """
    Implement the stopping criteria for the 2-block decomposition loop

    Parameters:
    ------
    X: Pxn numpy array
       P observations of n-dimensional points (n=2): (X1, X2)
    y_train: true values of the function
    v: Nx1-array: parameters from the hidden to the output layer
    W: Nxn Matrix: weights from input layer to hidden layer
    b: Nx1-array: bias from input layer to hidden layer
    threshold: thresholds for the norm of the gradient in the 
               convex and non-convex optimization steps respectively

    Returns:
    True if the stopping criteria are satisfied
    False if the stopping criteria are not satisfied
    ------
    """
    
    eps1 = threshold[0]
    eps2 = threshold[1]
    
    conv_gradient = convex_Grad(v, X_train, y_train, W, b)
    
    W_and_b = np.concatenate((W.reshape(N*n,1), b)).reshape(-1)
    non_conv_gradient = non_convex_Grad(W_and_b, X_train, y_train, v)
    

    conv_module = np.linalg.norm(conv_gradient)
    non_conv_module = np.linalg.norm(non_conv_gradient)
    
    if (conv_module < eps1) and (non_conv_module < eps2):
        return True
    else:
        return False