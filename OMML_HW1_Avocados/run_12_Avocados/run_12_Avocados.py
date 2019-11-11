import pandas as pd
import numpy as np
from scipy.optimize import minimize  # minimizer
from time import time
from sklearn.model_selection import train_test_split  # split in train and test
from sklearn.metrics import mean_squared_error

import library_12 as l12   # our library

# loading file into pandas dataframe
file = pd.ExcelFile('dataPoints.xlsx')
df = file.parse('Foglio1')

# taking X and y values (300 observations)
X = df[['x1', 'x2']].to_numpy()
y_true = df[['y']].to_numpy().reshape(1,-1)

# splitting in train and test
RANDOM_SEED = 1869097 # seed (N° matricola)
X_train, X_test, y_train, y_test = train_test_split(X, y_true.reshape(-1,1), test_size=0.15,
                                                    random_state=RANDOM_SEED)

np.random.seed(RANDOM_SEED)
# reshaping y's in a proper way for the library_12 functions
y_train = y_train.reshape(1,-1)
y_test = y_test.reshape(1,-1)

N = 30
n = X.shape[1]

# initializing omega vector
#   it "packs" both v and c vector parameters (minimize takes a unique vector)
omega = np.random.randn(N + N*n)

# minimizing the function using training set
t1 = time()
res = minimize(l12.loss_RBF, omega, jac=l12.fun_grad_RBF, args=(X_train, y_train))
t1 = time()-t1

# predicting y with the new parameters omega (res.x) after the optimization process
y_train_pred = l12.fun_RBF(X_train, res.x)
y_test_pred = l12.fun_RBF(X_test, res.x)

print('-----------------')
print('-------- Team Avocados run.')
print('-------- Exercise 1.2: RBF implementation --------')
print('-----------------')
print('Number of neurons N:', l12.N)
print('sigma value:', l12.sigma)
print('rho value:', l12.rho)
print('Optimization solver chosen: BFGS')
print('nfev:', res.nfev)
print('nit:', res.nit)
print('njev:', res.njev)
print('exec time:', round(t1, 5))
print('training error (MSE):', l12.MSE(y_train, y_train_pred))
print('test error (MSE):', l12.MSE(y_test, y_test_pred))