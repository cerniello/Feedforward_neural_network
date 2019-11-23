#import modules
import pandas as pd
import numpy as np
from scipy.optimize import minimize  # minimizer
from time import time
from sklearn.model_selection import train_test_split  # split in train and test
from sklearn.metrics import mean_squared_error

# libraries used for eventual plots 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

import library_3 as l3   # our library

plot_the_function = 0      # decide wether to show or not the function
save_the_plot = 0          # decide wether to save or not the function
img_path = 'plot_3.png'


# loading file into pandas dataframe
file = pd.ExcelFile('dataPoints.xlsx')
df = file.parse('Foglio1')

# taking X and y values (300 observations)
X = df[['x1', 'x2']].to_numpy()
y_true = df[['y']].to_numpy().reshape(1,-1)

# splitting in train and test
RANDOM_SEED = 1869097  # seed (N° matricola) - for reproducebility
X_train, X_test, y_train, y_test = train_test_split(X, y_true.reshape(-1,1), test_size=0.15,
                                                    random_state=RANDOM_SEED)

np.random.seed(RANDOM_SEED)
# reshaping y's in a proper way for the library_11 functions
y_train = y_train.reshape(1,-1)
y_test = y_test.reshape(1,-1)

# other hyperparameters (rho and sigma) are defined in library_11
N = l3.N
sigma = l3.sigma
rho = l3.rho
n = X.shape[1]
max_iter = 100

cnt = 1

W = np.random.randn(N,n)
b = np.random.randn(N,1)
v = np.random.randn(N,1)

print('\nInitializing 2 block-decomposition routine, max number of iterations: {}' .format(max_iter))

t = time()
while not l3.stopping_criteria(X_train, y_train, v, W, b) and cnt < max_iter:
    
    res = minimize(l3.convex_loss, v, jac = l3.convex_Grad, args=(X_train, y_train, W, b), method = "BFGS",
               options = {'gtol': 1e-4/(cnt*100)})
    
    v_star = res.x.copy()
    
    omega = res.x.copy()
    
    W_and_b = np.concatenate((W.reshape(N*n,1), b))
    
    
    res = minimize(l3.non_convex_loss, W_and_b, jac=l3.non_convex_Grad, args=(X_train, y_train, v_star), 
                   method = "BFGS", options={'gtol':1e-3/cnt})
    
    W_star = res.x[0:n*N].reshape(N,n)
    b_star = res.x[n*N:].reshape(N,1)
    
    v = v_star.copy()
    W = W_star.copy()
    b = b_star.copy()
    cnt = cnt + 1

t = time() - t
omega_star = np.concatenate((v.reshape(-1,1), W.reshape(N*n,1), b)).reshape(-1)
y_pred_train = l3.fun_MLP(X_train, omega_star)
y_pred_test = l3.fun_MLP(X_test, omega_star)

print('-----------------')
print('-------- Team Avocados run.')
print('-------- Exercise 3: MLP 2-Block decomposition implementation --------')
print('-----------------')
print('Number of neurons N:', l3.N)
print('sigma value:', l3.sigma)
print('rho value:', l3.rho)
print('Optimization solver chosen: BFGS')
print("Number of 2-block iterations: ", max_iter)
print('exec time for 2-block loop:', round(t, 5))
print('training error (MSE):', l3.MSE(y_train, y_pred_train))
print('test error (MSE):', l3.MSE(y_test, y_pred_test))
print('Norm of gradient:', np.linalg.norm(res.jac))


# PLOTTING THE RESULTING FUNCTION 

if plot_the_function == 1 or save_the_plot == 1:
    print('-----------------')
    print('Preparing now the plot of the function...')
    X_1 = np.linspace(-2,2,300)
    X_2 = np.linspace(-1,1,300)
    X_1, X_2 = np.meshgrid(X_1, X_2)
    zs = np.array([l3.fun_MLP(np.array([x,y]).reshape(1,2), omega_star) for x,y in zip(np.ravel(X_1), np.ravel(X_2))])
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
        print("Saving the function in '{}' ... " .format(img_path), end='')
        plt.savefig(img_path, dpi=600)
        print('Done')

    if plot_the_function == 1:
        print('Plotting the function..')
        plt.show()


    plt.close()