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


import library_11 as l11   # our library

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
N = l11.N
n = X.shape[1]

# initializing omega vector
#   it "packs" both v vector, W matrix and b vector (minimize takes a unique vector)
omega = np.random.randn(N + N*n + N)

# minimizing the function using training set
t1 = time()
res = minimize(l11.loss_MLP, omega, jac=l11.fun_grad_MLP, args=(X_train, y_train))
t1 = time()-t1

# predicting y with the new parameters omega (res.x) after the optimization process
y_train_pred = l11.fun_MLP(X_train, res.x)
y_test_pred = l11.fun_MLP(X_test, res.x)

print('-----------------')
print('-------- Team Avocados run.')
print('-------- Exercise 1.1: MLP implementation --------')
print('-----------------')
print('Number of neurons N:', l11.N)
print('sigma value:', l11.sigma)
print('rho value:', l11.rho)
print('Optimization solver chosen: BFGS')
print('nfev:', res.nfev)
print('nit:', res.nit)
print('njev:', res.njev)
print('exec time:', round(t1, 5))
print('training error (MSE):', l11.MSE(y_train, y_train_pred))
print('test error (MSE):', l11.MSE(y_test, y_test_pred))

# PLOTTING THE RESULTING FUNCTION 
plot_the_function = 0      # decide wether to show or not the function
save_the_plot = 0          # decide wether to save or not the function
img_path = 'plot_11.png'

if plot_the_function == 1 or save_the_plot == 1:
	print('-----------------')
	print('Preparing now the plot of the function...')
	X_1 = np.linspace(-2,2,300)
	X_2 = np.linspace(-1,1,300)
	X_1, X_2 = np.meshgrid(X_1, X_2)
	zs = np.array([l11.fun_MLP(np.array([x,y]).reshape(1,2), res.x) for x,y in zip(np.ravel(X_1), np.ravel(X_2))])
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