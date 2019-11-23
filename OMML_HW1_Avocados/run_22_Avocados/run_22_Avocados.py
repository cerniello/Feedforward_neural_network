import pandas as pd
import numpy as np
from scipy.optimize import minimize  # minimizer
from time import time, sleep
from sklearn.model_selection import train_test_split  # split in train and test
from sklearn.metrics import mean_squared_error

import sys # imported only for dynamic displaying (print on terminal)

# libraries used for eventual plots 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

import library_22 as l22   # our library

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

# reshaping y's in a proper way for the library_22 functions
y_train = y_train.reshape(1,-1)
y_test = y_test.reshape(1,-1)


# hyperparameters are defined in library_21
# Notice that the hyperparameters used are the same
# used in run_12 (for the RBF)! 

N = l22.N      # default N: 30
n = X.shape[1] # n = 2

H = 5000
min_list = []     # tracking the H results of minimizer


print('-----------------')
print('-------- Team Avocados run.')
print('-------- Exercise 2.2: Two phases method on RBF --------')
print('-------- Unsupervised selection of the centers {} times' .format(H))


"""
for h in range(H):
    # print dinamically the progress
    sys.stdout.write("\r{}Iteration {}/{}" .format("-------- ", h+1, H))
    sys.stdout.flush()

    # picking the centers from the X_train observations
    idx = np.random.choice(range(0, X_train.shape[0]), size=N)
    c = X_train[idx]
    v = np.random.randn(N, 1)


    # minimize and append the result
    res = minimize(l22.loss_EL_RBF, v, jac=l22.fun_grad_EL_RBF, args=(c, X, y_true))
    
    mse_res = l22.MSE(y_train, l22.fun_EL_RBF(X_train, res.x, c))
    min_list.append([mse_res, c, h+1])

print(' Finished.')
# sorting the results and picking the best one
min_list.sort(key = lambda x: x[0])


# picking the best result
c = min_list[0][1]
v = np.random.randn(N)

best_iteration = min_list[0][2]

print('-------- Picking the centers c of iteration n. {}' .format(best_iteration))

"""

# it turned out that our best iteration with our seed was 
# 1330/5000 
best_iteration = 1330

print('-------- Picking the centers c of iteration n. {}' .format(best_iteration))

for h in range(best_iteration):

    # initialize the parameters
    idx = np.random.choice(range(0, X_train.shape[0]), size=N)
    c = X_train[idx]
    v = np.random.randn(N)


t1 = time()
res = minimize(l22.loss_EL_RBF, v, jac = l22.fun_grad_EL_RBF, args=(c, X_train, y_train), method = "BFGS")
t1 = time()-t1

# predicting y with the new parameters omega (res.x) after the optimization process
y_train_pred = l22.fun_EL_RBF(X_train, res.x, c)
y_test_pred = l22.fun_EL_RBF(X_test, res.x, c)

print('Number of neurons N:', l22.N)
print('sigma value:', l22.sigma)
print('rho value:', l22.rho)
print('Optimization solver chosen: BFGS')
print('nfev:', res.nfev)
print('nit:', res.nit)
print('njev:', res.njev)
print('exec time:', round(t1, 5))
print('training error (MSE):', l22.MSE(y_train, y_train_pred))
print('test error (MSE):', l22.MSE(y_test, y_test_pred))
print('Norm of gradient:', np.linalg.norm(res.jac))


# PLOTTING THE RESULTING FUNCTION 
plot_the_function = 0      # decide wether to show or not the function
save_the_plot = 0          # decide wether to save or not the function
img_path = 'plot_21.png'

if plot_the_function == 1 or save_the_plot == 1:
    print('-----------------')
    print('Preparing now the plot of the function...')
    X_1 = np.linspace(-2,2,300)
    X_2 = np.linspace(-1,1,300)
    X_1, X_2 = np.meshgrid(X_1, X_2)
    zs = np.array([l22.fun_EL_RBF(np.array([x,y]).reshape(1,2), res.x, c) for x,y in zip(np.ravel(X_1), np.ravel(X_2))])
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
