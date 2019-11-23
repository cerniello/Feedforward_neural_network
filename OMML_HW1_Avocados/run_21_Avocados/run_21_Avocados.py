import pandas as pd
import numpy as np
from scipy.optimize import minimize  # minimizer
from time import time
from sklearn.model_selection import train_test_split  # split in train and test

import sys # imported only for dynamic displaying (print on terminal)

# libraries used for eventual plots 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

import library_21 as l21   # our library

plot_the_function = 0    # decide wether to show or not the function
save_the_plot = 0        # decide wether to save or not the function
img_path = 'plot_21.png'

# loading file into pandas dataframe
file = pd.ExcelFile('dataPoints.xlsx')
df = file.parse('Foglio1')

# taking X and y values (300 observations)
X = df[['x1', 'x2']].to_numpy()
y_true = df[['y']].to_numpy().reshape(1,-1)

# splitting in train and test
RANDOM_SEED = 1869097  # seed (N° matricola)
X_train, X_test, y_train, y_test = train_test_split(X, y_true.reshape(-1,1), test_size=0.15,
                                                    random_state=RANDOM_SEED)

# seed: for reproducebility
np.random.seed(RANDOM_SEED)

# reshaping y's in a proper way for the library_12 functions
y_train = y_train.reshape(1,-1)
y_test = y_test.reshape(1,-1)

# hyperparameters are defined in library_21
# Notice that the hyperparameters used are the same
# of run_11 and run_12!
N = l21.N
n = X.shape[1]


H = 5000
min_list = [] # tracking the H results of minimizer

print('-----------------')
print('-------- Team Avocados run.')
print('-------- Exercise 2.1: Extreme Learning process on MLP --------')
print('-------- Random Choice of W and b minimizing {} times' .format(H))


# H iterations
"""
for h in range(H):
    # print dinamically the progress
    sys.stdout.write("\r{}Iteration {}/{}" .format("-------- ", h+1, H))
    sys.stdout.flush()

    # initialize the parameters
    W = np.random.randn(N,n)
    b = np.random.randn(N,1)
    v = np.random.randn(N)

    res = minimize(l21.loss_EL_MLP, v, jac = l21.fun_grad_EL_MLP, args=(W, b, X_train, y_train), method = "BFGS")
    
    mse_res = l21.MSE(y_train, l21.fun_EL_MLP(X_train, res.x, W, b))
    min_list.append([mse_res, W, b, h+1])

print(' Finished.')
# sorting the results and picking the best one
min_list.sort(key = lambda x: x[0])

# picking the best result
W = min_list[0][1] #np.random.randn(N,n)
b = min_list[0][2] #np.random.randn(N,1)
v = np.random.randn(N)

best_iteration = min_list[0][3]

print('-------- Picking W and b of the iteration n. {}' .format(best_iteration))
"""


# it turned out that our best iteration with our seed was 
# 2790/5000 (2789 since we count from 0)
best_iteration = 2790
print('-------- Picking W and b of the iteration n. {}' .format(best_iteration))


for h in range(best_iteration):

    # initialize the parameters
    W = np.random.randn(N,n)
    b = np.random.randn(N,1)
    v = np.random.randn(N)


t1 = time()
res = minimize(l21.loss_EL_MLP, v, jac = l21.fun_grad_EL_MLP, args=(W, b, X_train, y_train), method = "BFGS")
t1 = time()-t1

# predicting y with the new parameters omega (res.x) after the optimization process
y_train_pred = l21.fun_EL_MLP(X_train, res.x, W, b)
y_test_pred = l21.fun_EL_MLP(X_test, res.x, W, b)

print('Number of neurons N:', l21.N)
print('sigma value:', l21.sigma)
print('rho value:', l21.rho)
print('Optimization solver chosen: BFGS')
print('nfev:', res.nfev)
print('nit:', res.nit)
print('njev:', res.njev)
print('exec time:', round(t1, 5))
print('training error (MSE):', l21.MSE(y_train, y_train_pred))
print('test error (MSE):', l21.MSE(y_test, y_test_pred))
print('Norm of gradient:', np.linalg.norm(res.jac))



# PLOTTING THE RESULTING FUNCTION 

if plot_the_function == 1 or save_the_plot == 1:
    print('-----------------')
    print('Preparing now the plot of the function...')
    X_1 = np.linspace(-2,2,300)
    X_2 = np.linspace(-1,1,300)
    X_1, X_2 = np.meshgrid(X_1, X_2)
    zs = np.array([l21.fun_EL_MLP(np.array([x,y]).reshape(1,2), res.x, W, b) for x,y in zip(np.ravel(X_1), np.ravel(X_2))])
    Z = zs.reshape(X_1.shape)
    fig = plt.figure(figsize=(12,8))
    ax = fig.gca(projection='3d')
    ax.plot_surface(X_1, X_2, Z ,linewidth=0,cmap=cm.viridis, antialiased=False)
    ax.set_xticks((np.linspace(-2,2,10)))
    ax.view_init(elev=15, azim=240)
    ax.set_xlabel('X_1')
    ax.set_ylabel('X_2')
    ax.set_zlabel('Z Label')
    print('Done')


    if save_the_plot == 1:
        plt.savefig(img_path, dpi=600)
    
    if plot_the_function == 1:
        plt.show()

    plt.close()
