# import modules
import pandas as pd
import numpy as np

from time import sleep
from time import time

from warnings import filterwarnings

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

from scipy.optimize import minimize as minimize
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

file = pd.ExcelFile('dataPoints.xlsx')
df = file.parse('Foglio1')

def fun_MLP(X, omega, N, sigma):
    
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

def fun_grad_MLP(omega, X_train, y_true, N, sigma, rho):
    
    # creating matrix X1 (X matrix plus array of ones)
    X1 = np.append(X_train, -1*np.ones((X_train.shape[0],1)), axis=1)


    # pick parameters from omega one
    v = omega[0:N].reshape(N,1)
    W = omega[N:3*N].reshape(N, n)
    b = omega[3*N:].reshape(N, 1)
    
    
    Wb = np.append(W,b, axis=1)
    
    ### compute partial derivatives in a "pythonic" way:
    
    y_pred = fun_MLP(X_train, omega, N, sigma)
    
    # for the sake of clearness, let's define some variables
    # (MATRIX1) dot product between Wb and X1
    T = np.dot(Wb, np.transpose(X1)) # NxX_train.shape[0] matrix
    
    # derivative of g()
    g_der = 4 * sigma * np.exp(2 * sigma * T) / (np.exp(2 * sigma * T) + 1)**2

    dE_dv = 1 / X_train.shape[0] * np.dot(g_fun(T, sigma), np.transpose(y_pred - y_true)) + 2 * rho * v
    dE_db = 1 / X_train.shape[0] * np.dot(-1 * v * g_der, np.transpose(y_pred - y_true)) + 2 * rho * b
    
    # dealing with dE_dW
    mat1 = v * g_der * X1[:,0]
    mat1 = np.dot(mat1, np.transpose(y_pred - y_true))
    mat2 = v * g_der * X1[:,1]
    mat2 = np.dot(mat2, np.transpose(y_pred - y_true))

    fusion = np.append(mat1, mat2, axis=1)
    dE_dW = 1 / X_train.shape[0] * fusion + 2 * rho * W

    omega_gradient = np.concatenate((dE_dv, dE_dW.reshape(N*n,1), dE_db))
    return omega_gradient.reshape(-1)

def loss(omega, X, y_true, N, sigma, rho):
    y_pred = fun_MLP(X, omega, N, sigma)
    l = np.sum((y_pred - y_true)**2)/(X.shape[1] * X.shape[0]) + rho * np.linalg.norm(omega)**2
    return l


X = df[['x1', 'x2']].to_numpy()

X_train = np.copy(X)
n = X.shape[1]

y_true = df[['y']].to_numpy().reshape(1,-1)


def grid_search(X, y, N, K=5):
    
    # reshaping y array (for Kfold split)
    y = y.reshape(-1, 1)

    n = X.shape[1]
    res_list = []
    sigma_list = [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.5, 1.8]
    rho_list = [0.00001, 0.00005, 0.0001, 0.0005, 0.001]

    print('----------- Started routine for N = {}\n'.format(N), end='')

    for sigma in sigma_list:
        for rho in rho_list:
            # ---- DEBUG
            # print('N', N, 'rho:', rho, 'sigma', sigma)
            print('N: {}, sigma: {}, rho: {}' .format(N,sigma,rho))
       
            ### parameters initialization
            W = np.random.random((N,n))
            b = np.random.random((N,1))
            v = np.random.random((N,1))

            # all the parameters
            omega = np.concatenate((v, W.reshape(N*n,1), b))

            # create result list for k_train and k_validation
            train_error = loss(omega, X, y, N, sigma, rho)
            k_train_err = []
            k_val_err = []
            func_eval = []
            time_exec = []
            nfev = []
            nit = []
            njev = []

            # ---- DEBUG
            # print('N: {}, simga: {}'.format(N, sigma))

            k_fold = KFold(K, shuffle=True)
            for train_indices, val_indices in k_fold.split(X):
                X_train, y_train = X[train_indices], y[train_indices]
                X_val, y_val = X[val_indices], y[val_indices]

                # now that we split, we need to readjust y vectors

                y_train, y_val = y_train.reshape(1, -1), y_val.reshape(1, -1)

                # train the model with gradient
                t1 = time()
                res = minimize(loss, omega, jac=fun_grad_MLP, args=(X_train, y_train, N, sigma, rho))
                time_exec.append(time() - t1)

                # error on the train and validation
                k_train_err.append(loss(res.x, X_train, y_train, N, sigma, rho))
                k_val_err.append(loss(res.x, X_val, y_val, N, sigma, rho))

                # store results
                nfev.append(res.nfev)
                nit.append(res.nit)
                njev.append(res.njev)

            # ---- DEBUG ----
            # print('N', N)
            # print('rho', rho)
            # print('res.success', res.success)
            # print('k_train', k_train_err)
            # print('k_val_err', k_val_err)
            # print('time_exec', time_exec)
            # print('nfev', nfev)
            # print('nit', nit)
            # print('njev', njev)

            # create a list and append it to res_list
            res_list.append(['Grad', K, N, sigma, rho, res.success, train_error, np.mean(k_train_err),
                             np.mean(k_val_err), np.mean(time_exec),
                             int(np.mean(nfev)), int(np.mean(nit)), int(np.mean(njev))])
            sleep(0.4)
    print('----------- N: {} ===> end' .format(N))
    return res_list

# split train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_true.reshape(-1,1), test_size=0.15, random_state=1869097)


filterwarnings('ignore')
N_list = [5, 10, 15, 20, 25, 30]
K = 5
ncpus = cpu_count()
results = []
with ProcessPoolExecutor(max_workers=ncpus) as executor:
    futures = list((executor.submit(grid_search, X_train, y_train, N, K) for N in N_list))

for future in as_completed(futures):
    results += future.result()


final_res = pd.DataFrame(results, columns=['gradient', 'K', 'N', 'sigma', 'rho', 'success', 'train_error', 'train_error_fit',
                                           'validation_error', 'time_exec(s)', 'nfev', 'nit', 'njev'])


final_res.to_csv('KFOLD_MLP.csv', index=False)
