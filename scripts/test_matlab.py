# import libraries
import numpy as np, pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import statsmodels.api as sm
import scipy.stats as stats
import sys
import matlab.engine
import scipy.io as sio

# generate data
N = 3
x = np.linspace(0,20,N)
e = np.random.normal(loc = 0.0, scale = 5.0, size = N)
y = 3*x + e
df = pd.DataFrame({'y':y, 'x':x})
df['constant'] = 1

sio.savemat('df.mat', {'x':df['x'], 'y':df['y'], 'constant':df['constant']})
eng = matlab.engine.start_matlab()
# eng.load('df.mat')
# eng.load('MLERegression.m')
x, fval = eng.bads('@ll', 
					'[0 1 1 1 1 1 1 1 0 1]',  # x0
					'[-2 -20 -20 -20 -20 -20 -20 -20 -15 0]', # lb
					'[2 10 10 10 10 10 10 10 15 20]', # ub
					'[-1 -15 -15 -15 -15 -10 -10 -10 -10 0]', # plb
					'[1 5 5 5 5 5 2 2 10 15]') # pub
# x, fval = eng.bads('MLERegression', [5.0,5.0,2.0], [-10.0,-10.0,-10.0],[10.0,10.0,10.0],[-10.0,-10.0,-10.0],[10.0,10.0,10.0])
print(x)
print(fval)
eng.quit()