'''https://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html#sphx-glr-auto-examples-model-selection-plot-underfitting-overfitting-py'''

import math
import statistics
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = "Times New Roman"
np.random.seed(0)

def cubic(x):
    return x ** 3 + 2 * x ** 2 - 3 * x + 5
    
def seven(x):
    return -36 * x + 49 * x ** 3 - 14 * x ** 5 + x ** 7
    
def ten(x):
    return -36 * x + 49 * x ** 5 - 14 * x ** 7 + x ** 10
    
def two(x):
    return (x)**2
    
def sine_exp(x):
    return 5*np.sin(x) + np.exp(x)/5
    
def cos(x):
    return np.cos(1.5 * np.pi * x)
    
def get_data(n):
    return np.sort(np.random.rand(n))

def run_poly(x_train, y_train, x_test, y_test, degree, regularize=True, alpha=0.01):
    
    polynomial_features = PolynomialFeatures(degree=degree, include_bias=True)
    linear_regression = LinearRegression()
    
    extracted_train = polynomial_features.fit_transform(x_train[:, np.newaxis])
    extracted_test = polynomial_features.fit_transform(x_test[:, np.newaxis])
    
    if regularize:
        regularization_x = np.diag([alpha*idx**2 for idx in range(degree+1)])
        regularization_y = np.zeros(degree+1)
        extracted_train = np.concatenate([extracted_train, regularization_x])
        y_train_new = np.concatenate([y_train, regularization_y])
    else:
        y_train_new = y_train

    linear_regression.fit(extracted_train, y_train_new)
    y_pred = linear_regression.predict(extracted_test)
    mse = mean_squared_error(y_pred, y_test)
    return mse



test_samples = 100
train_samples_range = range(5,71)
runs_per_size = 100
func = two
func_name = 'Polynomial Degree 2'
poly_small_size = 2
poly_big_size = 10
poly_reg_size = 10
alpha = 0.01
set_title = False

'''
test_samples = 100
train_samples_range = range(5,71)
runs_per_size = 100
func = cos
func_name = 'Cosine'
poly_small_size = 2
poly_big_size = 10
poly_reg_size = 10
alpha = 0.01
set_title = False
'''

'''
test_samples = 100
train_samples_range = range(5,101)
runs_per_size = 100
func = ten
func_name = 'Polynomial Degree 10'
poly_small_size = 2
poly_big_size = 10
poly_reg_size = 10
alpha = 0.001
set_title = False
'''

poly_small_accs = {}
poly_big_accs = {}
poly_reg_accs = {}
poly_small_errs = {}
poly_big_errs = {}
poly_reg_errs = {}

for train_samples in train_samples_range:
    poly_small_acc = []
    poly_big_acc = []
    poly_reg_acc = []
    poly_small_err = []
    poly_big_err = []
    poly_reg_err = []
    for run in range(runs_per_size):
        x_train = get_data(train_samples)
        y_train = func(x_train) + np.random.randn(train_samples) * 0.1
        x_test = np.linspace(0, 1, num=test_samples, endpoint=True)
        y_test = func(x_test) #+ np.random.randn(test_samples) * 0.1
        poly_small_acc.append(run_poly(x_train,y_train,x_test,y_test,poly_small_size, regularize = False, alpha=alpha))
        y_test = func(x_test) #+ np.random.randn(test_samples) * 0.1
        poly_big_acc.append(run_poly(x_train,y_train,x_test,y_test,poly_big_size, regularize = False, alpha=alpha))
        y_test = func(x_test) #+ np.random.randn(test_samples) * 0.1
        poly_reg_acc.append(run_poly(x_train,y_train,x_test,y_test,poly_reg_size, regularize = True, alpha=alpha))
    poly_small_accs[train_samples] = statistics.mean(poly_small_acc)
    poly_big_accs[train_samples] = statistics.mean(poly_big_acc)
    poly_reg_accs[train_samples] = statistics.mean(poly_reg_acc)
    poly_small_errs[train_samples] = statistics.stdev(poly_small_acc)/math.sqrt(runs_per_size)
    poly_big_errs[train_samples] = statistics.stdev(poly_big_acc)/math.sqrt(runs_per_size)
    poly_reg_errs[train_samples] = statistics.stdev(poly_reg_acc)/math.sqrt(runs_per_size)

  
### Plotting

x = [idx for idx in train_samples_range]
y_small = [poly_small_accs[idx] for idx in train_samples_range]
y_big = [poly_big_accs[idx] for idx in train_samples_range]
y_reg = [poly_reg_accs[idx] for idx in train_samples_range]

y_small_err = [poly_small_errs[idx] for idx in train_samples_range]
y_big_err = [poly_big_errs[idx] for idx in train_samples_range]
y_reg_err = [poly_reg_errs[idx] for idx in train_samples_range]
    
    
import matplotlib.pyplot as plt
# Plot accs_by_mdl
fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
ax.plot(x, y_small, label='Degree '+ str(poly_small_size) +' Polynomial')
ax.fill_between(x, np.array(y_small) - np.array(y_small_err), np.array(y_small) + np.array(y_small_err), alpha=0.2,linewidth=0)
ax.plot(x, y_big, label='Degree '+ str(poly_big_size) +' Polynomial')
ax.fill_between(x, np.array(y_big) - np.array(y_big_err), np.array(y_big) + np.array(y_big_err), alpha=0.2,linewidth=0)
ax.plot(x, y_reg, label='Degree '+ str(poly_reg_size) +' Regularized')
ax.fill_between(x, np.array(y_reg) - np.array(y_reg_err), np.array(y_reg) + np.array(y_reg_err), alpha=0.2,linewidth=0)
ax.set_xlabel('# training samples')  # Add an x-label to the axes.
ax.set_ylabel('Mean Squared Error')  # Add a y-label to the axes.
if set_title:
    ax.set_title("MSE by Training Samples - " + func_name)  # Add a title to the axes.
ax.legend();  # Add a legend.
plt.yscale('log')
plot_path = os.path.join('./', func_name.replace(" ", "_") + '.pdf')
plt.savefig(plot_path) 
plt.show()






